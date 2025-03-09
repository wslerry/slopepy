"""GPU-accelerated contour generation module with CPU fallback using Marching Squares."""

import numpy as np
import rasterio
from rasterio.crs import CRS
from typing import List, Tuple
import os
import geopandas as gpd
from shapely.geometry import LineString
from .utils import read_terrain, reproject_dem, get_utm_zone

try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

class MarchingSquares:
    def __init__(self, contour_levels: List[float], use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.contour_levels = self.xp.array(contour_levels, dtype=self.xp.float32)
        self.lookup_table = self.xp.array([
            [-1, -1], [3, 0], [0, 1], [3, 1], [1, 2], [3, 2], [0, 2], [3, 0],
            [2, 3], [0, 2], [0, 3], [1, 2], [1, 3], [0, 1], [3, 2], [-1, -1]
        ], dtype=self.xp.int8)

    def _interpolate_edge(self, dem: 'xp.ndarray', level: 'xp.ndarray', i: int, j: int, edge: int) -> Tuple[float, float]:
        rows, cols = dem.shape
        if edge == 0:
            v0, v1 = dem[i, j], dem[i, min(j + 1, cols - 1)]
            x0, y0 = j, i
            x1, y1 = j + 1, i
        elif edge == 1:
            v0, v1 = dem[i, min(j + 1, cols - 1)], dem[min(i + 1, rows - 1), min(j + 1, cols - 1)]
            x0, y0 = j + 1, i
            x1, y1 = j + 1, i + 1
        elif edge == 2:
            v0, v1 = dem[min(i + 1, rows - 1), j], dem[min(i + 1, rows - 1), min(j + 1, cols - 1)]
            x0, y0 = j, i + 1
            x1, y1 = j + 1, i + 1
        else:
            v0, v1 = dem[i, j], dem[min(i + 1, rows - 1), j]
            x0, y0 = j, i
            x1, y1 = j, i + 1
        t = 0.5 if v0 == v1 else float((level - v0) / (v1 - v0))
        t = min(max(t, 0.0), 1.0)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        return x, y

    def generate_contours(self, dem: 'xp.ndarray', transform: rasterio.transform.Affine) -> List[dict]:
        rows, cols = dem.shape
        contours = []
        dem_size = rows * cols * 4
        total_size = dem_size * 2
        if self.use_gpu and total_size > 2e9:
            raise MemoryError("DEM too large for GPU memory. Falling back to CPU.")

        for level in self.contour_levels:
            states = (dem >= level).astype(self.xp.uint8)
            cell_indices = self.xp.zeros((rows - 1, cols - 1), dtype=self.xp.uint8)
            cell_indices += states[:-1, :-1] << 0
            cell_indices += states[:-1, 1:] << 1
            cell_indices += states[1:, 1:] << 2
            cell_indices += states[1:, :-1] << 3
            mask = (cell_indices > 0) & (cell_indices < 15)
            i_coords, j_coords = self.xp.where(mask)
            del states, mask

            if i_coords.size == 0:
                continue

            edges = self.lookup_table[cell_indices[i_coords, j_coords]]
            if self.use_gpu:
                i_coords, j_coords, edges = cp.asnumpy(i_coords), cp.asnumpy(j_coords), cp.asnumpy(edges)
            del cell_indices

            segments = []
            for idx in range(len(i_coords)):
                i, j = i_coords[idx], j_coords[idx]
                start_edge, end_edge = edges[idx]
                if start_edge == -1:
                    continue
                x0, y0 = self._interpolate_edge(dem, level, i, j, start_edge)
                x1, y1 = self._interpolate_edge(dem, level, i, j, end_edge)
                lon0, lat0 = transform * (x0, y0)
                lon1, lat1 = transform * (x1, y1)
                segments.append(LineString([(lon0, lat0), (lon1, lat1)]))

            if segments:
                contours.extend([{'geometry': seg, 'elevation': float(level)} for seg in segments])
            
            del i_coords, j_coords, edges
            if self.use_gpu:
                cp.cuda.Stream.null.synchronize()

        return contours

class GPUContourGenerator:
    def __init__(self, contour_levels: List[float] = None, dst_crs: str = None, force_gpu: bool = False) -> None:
        self.contour_levels = contour_levels if contour_levels is not None else [100, 500, 1000, 1500]
        self.dst_crs = dst_crs
        self.force_gpu = force_gpu
        self.use_gpu = False
        if CUDA_AVAILABLE:
            print("GPU available. Will attempt GPU acceleration unless memory constraints apply.")
        else:
            print("GPU not available. Using CPU backend.")
        self.marching_squares = None

    def process_dem(self, input_path: str, output_prefix: str, save_vector: bool = False) -> None:
        dem_full, profile, src_crs = read_terrain(input_path)
        nodata = profile['nodata']
        rows, cols = dem_full.shape
        
        effective_dst_crs = self.dst_crs
        # Handle src_crs as string or CRS object
        if isinstance(src_crs, str):
            src_crs_obj = CRS.from_string(src_crs)
        elif isinstance(src_crs, CRS):
            src_crs_obj = src_crs
        else:
            raise ValueError(f"Invalid src_crs type: {type(src_crs)}")
        
        if effective_dst_crs is None and src_crs_obj.is_geographic:
            bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            utm_crs = get_utm_zone(center_lon, center_lat)
            effective_dst_crs = utm_crs.to_string()
            dem_full, profile = reproject_dem(dem_full, profile, src_crs_obj.to_string(), effective_dst_crs)
            print(f"Reprojected DEM to UTM: {effective_dst_crs}")
        elif effective_dst_crs is not None:
            dem_full, profile = reproject_dem(dem_full, profile, src_crs_obj.to_string(), effective_dst_crs)
            print(f"Reprojected DEM to {effective_dst_crs}")

        dem_size = rows * cols * 4
        total_size = dem_size * 2
        vram_limit = 2e9
        
        if self.force_gpu and not CUDA_AVAILABLE:
            raise RuntimeError("GPU forced but not available.")
        
        self.use_gpu = self.force_gpu or (CUDA_AVAILABLE and total_size <= vram_limit)
        if CUDA_AVAILABLE and not self.use_gpu and not self.force_gpu:
            print(f"DEM size ({total_size / 1e6:.1f} MB) exceeds GPU VRAM limit ({vram_limit / 1e6:.1f} MB). Falling back to CPU.")
        
        xp = cp if self.use_gpu else np
        self.marching_squares = MarchingSquares(self.contour_levels, use_gpu=self.use_gpu)
        dem = xp.asarray(dem_full, dtype=xp.float32)
        del dem_full

        try:
            contours = self.marching_squares.generate_contours(dem, profile['transform'])
        except MemoryError as e:
            if self.use_gpu:
                print(f"GPU memory error: {str(e)}. Falling back to CPU.")
                self.use_gpu = False
                self.marching_squares = MarchingSquares(self.contour_levels, use_gpu=False)
                dem = np.asarray(dem, dtype=np.float32)
                contours = self.marching_squares.generate_contours(dem, profile['transform'])
            else:
                raise
        
        del dem
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()

        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if save_vector:
            if gpd is None:
                print("Warning: geopandas or shapely not installed. Skipping vector output.")
            else:
                try:
                    gpkg_path = f"{output_prefix}_contours.gpkg"
                    for level in self.contour_levels:
                        level_contours = [c for c in contours if c['elevation'] == level]
                        if level_contours:
                            gdf = gpd.GeoDataFrame(level_contours, geometry='geometry', crs=profile['crs'])
                            gdf.to_file(gpkg_path, layer=f"Contour_{int(level)}", driver="GPKG")
                except Exception as e:
                    print(f"Error saving GeoPackage: {str(e)}")

        print(f"Contour generation completed using {'GPU' if self.use_gpu else 'CPU'} backend. Generated files:")
        if save_vector and gpd is not None and os.path.exists(gpkg_path):
            print(f"- {output_prefix}_contours.gpkg (layers: {', '.join([f'Contour_{int(lvl)}' for lvl in self.contour_levels if any(c['elevation'] == lvl for c in contours)])})")

__all__ = ['GPUContourGenerator']