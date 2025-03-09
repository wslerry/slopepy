"""GPU-accelerated hillshade calculation module."""

import os
import math
import numpy as np
import rasterio
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
from typing import List, Tuple
from .utils import read_hgt, reproject_dem

class HillshadeCalculatorGPU:
    def __init__(self, azimuth: float = 317.0, altitude: float = 45.0, 
                 blur_window_sizes: List[float] = None, dst_crs: str = None) -> None:
        """
        Initialize the GPU-based hillshade calculator.

        Args:
            azimuth (float): Light source azimuth in degrees (0-360). Default is 317.0.
            altitude (float): Light source altitude in degrees (0-90). Default is 45.0.
            blur_window_sizes (List[float]): List of Gaussian blur window sizes. Default is [10, 20].
            dst_crs (str): Optional destination CRS (e.g., 'EPSG:32633'). If None, auto-detects UTM zone.
        """
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA is not available. Slopy requires a GPU with CuPy support.")
        if azimuth > 360.0:
            raise ValueError("Azimuth must be <= 360 degrees")
        if altitude > 90.0:
            raise ValueError("Altitude must be <= 90 degrees")
        
        self.AZIMUTH = azimuth
        self.ALTITUDE = altitude
        self.blur_window_sizes = blur_window_sizes if blur_window_sizes is not None else [10, 20]
        self.dst_crs = dst_crs
        self.azimuthrad = cp.float32((360.0 - self.AZIMUTH) * math.pi / 180.0)
        self.altituderad = cp.float32(self.ALTITUDE * math.pi / 180.0)

    def calculate_slope_aspect(self, dem: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Calculate slope and aspect on GPU."""
        y, x = cp.gradient(dem)
        slope = cp.pi / 2.0 - cp.arctan(cp.sqrt(x * x + y * y))
        aspect = cp.arctan2(-x, y)
        del x, y  # Free memory
        cp.cuda.Stream.null.synchronize()
        return slope, aspect

    def calculate_hillshade(self, slope: cp.ndarray, aspect: cp.ndarray) -> cp.ndarray:
        """Calculate hillshade on GPU."""
        shaded = (cp.sin(self.altituderad) * cp.sin(slope) +
                  cp.cos(self.altituderad) * cp.cos(slope) * 
                  cp.cos((self.azimuthrad - cp.pi / 2.0) - aspect))
        hillshade = (255 * (shaded + 1) / 2).astype(cp.uint8)
        del shaded
        cp.cuda.Stream.null.synchronize()
        return hillshade

    def _process_tile(self, dem_tile: cp.ndarray, resolution: float) -> cp.ndarray:
        """Process a single tile for hillshade calculation."""
        dem_scaled = dem_tile / resolution
        slope, aspect = self.calculate_slope_aspect(dem_scaled)
        hillshade = self.calculate_hillshade(slope, aspect)
        del slope, aspect, dem_scaled
        cp.cuda.Stream.null.synchronize()
        return hillshade

    def process_dem(self, input_path: str, output_prefix: str) -> None:
        """
        Process SRTM .hgt DEM with GPU-accelerated hillshade calculation using tiling.

        Args:
            input_path (str): Path to input .hgt file
            output_prefix (str): Prefix for output files
        """
        dem, profile, src_crs = read_hgt(input_path)
        dem, profile = reproject_dem(dem, profile, src_crs, self.dst_crs)

        resolution = abs(profile['transform'][0])
        tile_size = 1024
        rows, cols = dem.shape
        hillshade_output = cp.zeros((rows, cols), dtype=cp.uint8)

        # Tile processing for original hillshade
        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                i_end = min(i + tile_size, rows)
                j_end = min(j + tile_size, cols)
                dem_tile = dem[i:i_end, j:j_end].copy()
                hillshade_tile = self._process_tile(dem_tile, resolution)
                hillshade_output[i:i_end, j:j_end] = hillshade_tile
                del dem_tile, hillshade_tile
                cp.cuda.Stream.null.synchronize()

        # Save original hillshade
        profile.update(dtype=rasterio.uint8, count=1, nodata=255)
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        hillshade_np = cp.asnumpy(hillshade_output)
        with rasterio.open(f"{output_prefix}_original.tif", 'w', **profile) as dst:
            dst.write(hillshade_np, 1)

        # Process blurred versions
        for window_size in self.blur_window_sizes:
            blurred_output = cp.zeros((rows, cols), dtype=cp.uint8)
            for i in range(0, rows, tile_size):
                for j in range(0, cols, tile_size):
                    i_end = min(i + tile_size, rows)
                    j_end = min(j + tile_size, cols)
                    dem_tile = dem[i:i_end, j:j_end].copy()
                    dem_scaled = dem_tile / resolution
                    blurred_dem = cp_ndimage.gaussian_filter(dem_scaled, sigma=window_size / 6)
                    blur_slope, blur_aspect = self.calculate_slope_aspect(blurred_dem)
                    blur_hillshade = self.calculate_hillshade(blur_slope, blur_aspect)
                    blurred_output[i:i_end, j:j_end] = blur_hillshade
                    del dem_tile, dem_scaled, blurred_dem, blur_slope, blur_aspect, blur_hillshade
                    cp.cuda.Stream.null.synchronize()

            blur_hillshade_np = cp.asnumpy(blurred_output)
            with rasterio.open(f"{output_prefix}_blur_{window_size}.tif", 'w', **profile) as dst:
                dst.write(blur_hillshade_np, 1)

        print(f"Hillshade processing completed for {input_path}")

__all__ = ['HillshadeCalculatorGPU']