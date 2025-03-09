"""GPU-accelerated contour generation module."""

import cupy as cp
import numpy as np
import rasterio
from rasterio.features import shapes
from typing import List, Tuple
import os
import geopandas as gpd
from shapely.geometry import shape
from .utils import read_hgt, reproject_dem

class GPUContourGenerator:
    def __init__(self, contour_levels: List[float] = None, dst_crs: str = None) -> None:
        """
        Initialize the GPU-based contour generator.

        Args:
            contour_levels (List[float]): Elevation levels for contours (meters). Default is [100, 500, 1000, 1500].
            dst_crs (str): Optional destination CRS (e.g., 'EPSG:32633'). If None, auto-detects UTM zone.
        """
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA is not available. Slopy requires a GPU with CuPy support.")
        
        self.contour_levels = contour_levels if contour_levels is not None else [100, 500, 1000, 1500]
        self.dst_crs = dst_crs

    def generate_contours(self, dem: cp.ndarray, profile: dict) -> List[Tuple[cp.ndarray, float]]:
        """
        Generate contour masks from DEM on GPU.

        Args:
            dem (cp.ndarray): Reprojected DEM on GPU
            profile (dict): Rasterio profile with updated CRS and transform

        Returns:
            List[Tuple[cp.ndarray, float]]: List of (contour mask on GPU, level) pairs
        """
        contours = []
        tolerance = cp.float32(1.0)  # Adjustable tolerance for contour thickness
        for level in self.contour_levels:
            mask = cp.abs(dem - level) < tolerance
            contours.append((mask, level))
        return contours

    def save_vector_contours(self, contours: List[Tuple[cp.ndarray, float]], profile: dict, gpkg_path: str) -> None:
        """
        Save contours as vector layers in a GeoPackage.

        Args:
            contours (List[Tuple[cp.ndarray, float]]): List of (contour mask, level) pairs
            profile (dict): Rasterio profile
            gpkg_path (str): Path to output GeoPackage
        """
        try:
            for mask, level in contours:
                mask_np = cp.asnumpy(mask).astype(np.uint8)
                shape_gen = shapes(mask_np, transform=profile['transform'], connectivity=4)
                shapes_list = [{'geometry': shape(geom), 'elevation': level} for geom, _ in shape_gen if geom]
                
                if shapes_list:
                    gdf = gpd.GeoDataFrame(shapes_list, geometry='geometry', crs=profile['crs'])
                    gdf.to_file(gpkg_path, layer=f"Contour_{int(level)}", driver="GPKG")
        except ImportError:
            print("Warning: geopandas or shapely not installed. Skipping vector output.")
        except Exception as e:
            print(f"Error saving GeoPackage: {str(e)}")

    def process_dem(self, input_path: str, output_prefix: str, save_vector: bool = False) -> None:
        """
        Process SRTM .hgt DEM to generate contours.

        Args:
            input_path (str): Path to input .hgt file
            output_prefix (str): Prefix for output files
            save_vector (bool): If True, save contours as separate layers in a GeoPackage
        """
        dem, profile, src_crs = read_hgt(input_path)
        dem, profile = reproject_dem(dem, profile, src_crs, self.dst_crs)

        contours = self.generate_contours(dem, profile)
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        profile.update(dtype=rasterio.uint8, nodata=0)
        for mask, level in contours:
            mask_np = cp.asnumpy(mask).astype(np.uint8)
            with rasterio.open(f"{output_prefix}_contour_{int(level)}.tif", 'w', **profile) as dst:
                dst.write(mask_np, 1)

        if save_vector:
            gpkg_path = f"{output_prefix}_contours.gpkg"
            self.save_vector_contours(contours, profile, gpkg_path)

        print("GPU-accelerated contour generation completed. Generated files:")
        for level in self.contour_levels:
            print(f"- {output_prefix}_contour_{int(level)}.tif")
        if save_vector and os.path.exists(gpkg_path):
            print(f"- {output_prefix}_contours.gpkg (layers: {', '.join([f'Contour_{int(lvl)}' for lvl in self.contour_levels])})")

__all__ = ['GPUContourGenerator']