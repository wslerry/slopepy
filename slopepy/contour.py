"""GPU-accelerated contour generation module."""

import cupy as cp
import numpy as np
import rasterio
from rasterio.features import shapes
from typing import List, Tuple
import os
import geopandas as gpd
from shapely.geometry import shape
from .utils import read_terrain, reproject_dem, get_utm_zone

class GPUContourGenerator:
    def __init__(self, contour_levels: List[float] = None, dst_crs: str = None) -> None:
        """
        Initialize the GPU-based contour generator.

        Args:
            contour_levels (List[float]): Elevation levels for contours (meters). Default is [100, 500, 1000, 1500].
            dst_crs (str): Optional destination CRS (e.g., 'EPSG:32633'). If None, defaults to UTM.
        """
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA is not available. This class requires a GPU with CuPy support.")
        self.contour_levels = contour_levels if contour_levels is not None else [100, 500, 1000, 1500]
        self.dst_crs = dst_crs

    def generate_contours(self, dem: cp.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Generate contour masks from DEM tile on GPU and transfer to CPU.

        Args:
            dem (cp.ndarray): DEM tile on GPU

        Returns:
            List[Tuple[np.ndarray, float]]: List of (contour mask, level) pairs
        """
        dem_np = cp.asnumpy(dem)  # Transfer to CPU for contour processing
        contours = []
        
        for level in self.contour_levels:
            tolerance = 1.0  # Adjust tolerance for smoother/thinner contours
            mask = np.abs(dem_np - level) < tolerance
            mask = mask.astype(np.uint8)
            contours.append((mask, level))
        
        del dem_np
        cp.cuda.Stream.null.synchronize()
        return contours

    def process_dem(self, input_path: str, output_prefix: str, save_vector: bool = False) -> None:
        """
        Process DEM to generate contours using tiling.

        Args:
            input_path (str): Path to input DEM file (e.g., .hgt or .tif)
            output_prefix (str): Prefix for output files
            save_vector (bool): If True, save contours as separate layers in a GeoPackage
        """
        # Load DEM
        dem, profile, src_crs = read_terrain(input_path)
        nodata = profile['nodata']

        # Determine effective CRS: user-provided > UTM if geographic > input CRS
        effective_dst_crs = self.dst_crs
        if effective_dst_crs is None and profile['crs'].is_geographic:
            bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            effective_dst_crs = get_utm_zone(center_lon, center_lat).to_string()
            dem, profile = reproject_dem(dem, profile, src_crs, effective_dst_crs)
            print(f"Reprojected DEM to UTM: {profile['crs']}")
        elif effective_dst_crs is not None:
            dem, profile = reproject_dem(dem, profile, src_crs, effective_dst_crs)
            print(f"Reprojected DEM to {effective_dst_crs}")

        # Tile processing
        tile_size = 1024
        rows, cols = dem.shape
        contour_masks = {level: np.zeros((rows, cols), dtype=np.uint8) for level in self.contour_levels}

        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                i_end = min(i + tile_size, rows)
                j_end = min(j + tile_size, cols)
                dem_tile = dem[i:i_end, j:j_end].copy()
                contours = self.generate_contours(dem_tile)
                
                for mask, level in contours:
                    contour_masks[level][i:i_end, j:j_end] |= mask  # Combine overlapping contours with OR
                
                del dem_tile, contours
                cp.cuda.Stream.null.synchronize()

        # Prepare output directory
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save raster contours
        profile.update(dtype=rasterio.uint8, nodata=0)
        for level, mask in contour_masks.items():
            with rasterio.open(f"{output_prefix}_contour_{int(level)}.tif", 'w', **profile) as dst:
                dst.write(mask, 1)

        # Save as vector layers in GeoPackage
        if save_vector:
            try:
                gpkg_path = f"{output_prefix}_contours.gpkg"
                for level, mask in contour_masks.items():
                    shape_gen = shapes(mask, transform=profile['transform'], connectivity=4)
                    shapes_list = []
                    for geom, _ in shape_gen:
                        geom_shapely = shape(geom)
                        shapes_list.append({'geometry': geom_shapely, 'elevation': level})
                    
                    if shapes_list:
                        gdf = gpd.GeoDataFrame(shapes_list, geometry='geometry', crs=profile['crs'])
                        gdf.to_file(gpkg_path, layer=f"Contour_{int(level)}", driver="GPKG")
            except ImportError:
                print("Warning: geopandas or shapely not installed. Skipping vector output.")
            except Exception as e:
                print(f"Error saving GeoPackage: {str(e)}")

        print("GPU-accelerated contour generation completed. Generated files:")
        for level in self.contour_levels:
            print(f"- {output_prefix}_contour_{int(level)}.tif")
        if save_vector and 'gpd' in globals() and os.path.exists(gpkg_path):
            print(f"- {output_prefix}_contours.gpkg (layers: {', '.join([f'Contour_{int(lvl)}' for lvl in self.contour_levels])})")

__all__ = ['GPUContourGenerator']