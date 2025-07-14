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

    def _interpolate_edge(self, dem, level, i, j, edge) -> Tuple[float, float]:
        rows, cols = dem.shape
        v0, v1 = (dem[i, j], dem[i, j + 1]) if edge == 0 else (
            dem[i, j + 1], dem[i + 1, j + 1]) if edge == 1 else (
            dem[i + 1, j], dem[i + 1, j + 1]) if edge == 2 else (dem[i, j], dem[i + 1, j])
        
        t = 0.5 if v0 == v1 else float((level - v0) / (v1 - v0))
        t = min(max(t, 0.0), 1.0)
        
        x0, y0, x1, y1 = (j, i, j + 1, i) if edge == 0 else (
            j + 1, i, j + 1, i + 1) if edge == 1 else (
            j, i + 1, j + 1, i + 1) if edge == 2 else (j, i, j, i + 1)
        
        return x0 + t * (x1 - x0), y0 + t * (y1 - y0)

    def generate_contours(self, dem, transform) -> List[dict]:
        rows, cols = dem.shape
        contours = []
        total_size = rows * cols * 4 * 2  # Memory estimate
        
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
            for i, j, (start_edge, end_edge) in zip(i_coords, j_coords, edges):
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
        self.contour_levels = contour_levels or [20]
        self.dst_crs = dst_crs
        self.force_gpu = force_gpu
        self.use_gpu = CUDA_AVAILABLE if force_gpu else False
        print("GPU" if CUDA_AVAILABLE else "CPU", "backend selected.")

    def process_dem(self, input_path: str, output_prefix: str, save_vector: bool = False) -> None:
        dem_full, profile, src_crs = read_terrain(input_path)

        effective_dst_crs = self.dst_crs or get_utm_zone(profile['transform'][2], profile['transform'][5]).to_string()
        dem_full, profile = reproject_dem(dem_full, profile, src_crs, effective_dst_crs)

        total_size = dem_full.size * 4 * 2  # Memory estimate
        if self.force_gpu and not CUDA_AVAILABLE:
            raise RuntimeError("GPU forced but not available.")

        self.use_gpu = self.force_gpu or (CUDA_AVAILABLE and total_size <= 2e9)
        if not self.use_gpu and CUDA_AVAILABLE:
            print("Falling back to CPU due to DEM size.")

        xp = cp if self.use_gpu else np
        self.marching_squares = MarchingSquares(self.contour_levels, use_gpu=self.use_gpu)
        dem = xp.asarray(dem_full, dtype=xp.float32)

        try:
            contours = self.marching_squares.generate_contours(dem, profile['transform'])
        except MemoryError:
            if self.use_gpu:
                print("GPU memory error. Falling back to CPU.")
                self.use_gpu = False
                self.marching_squares = MarchingSquares(self.contour_levels, use_gpu=False)
                dem = dem.get() if isinstance(dem, cp.ndarray) else np.asarray(dem, dtype=np.float32)
                contours = self.marching_squares.generate_contours(dem, profile['transform'])
            else:
                raise
        
        del dem
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()

        if save_vector and gpd:
            gpkg_path = f"{output_prefix}_contours.gpkg"
            gdf = gpd.GeoDataFrame(contours, geometry='geometry', crs=profile['crs'])
            gdf.to_file(gpkg_path, layer="Contours", driver="GPKG")
            print(f"Contours saved to {gpkg_path}")

__all__ = ['GPUContourGenerator']
