"""GPU-accelerated DEM suitability classification module."""

import cupy as cp
import numpy as np
import rasterio
from typing import Tuple
import os
from .utils import read_terrain, reproject_dem, get_utm_zone


class GPUSuitabilityClassifier:
    def __init__(self, dst_crs: str = None, elev_bins: cp.ndarray = None, elev_scores: cp.ndarray = None,
                 slope_bins: cp.ndarray = None, slope_scores: cp.ndarray = None) -> None:
        """
        Initialize the GPU-based suitability classifier for elevation and slope.

        Args:
            dst_crs (str): Optional destination CRS (e.g., 'EPSG:32633'). If None, auto-detects UTM zone.
            elev_bins (cp.ndarray): Custom elevation thresholds (masl). Default: [0, 1500, 2000, 3000, inf].
            elev_scores (cp.ndarray): Custom elevation scores. Default: [5, 3, 1, 0].
            slope_bins (cp.ndarray): Custom slope thresholds (percent). Default: [0, 15, 30, 45, inf].
            slope_scores (cp.ndarray): Custom slope scores. Default: [5, 3, 1, 0].
        """
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA is not available. Slopy requires a GPU with CuPy support.")
        
        self.dst_crs = dst_crs
        # Elevation thresholds (masl) and scores
        self.elev_bins = elev_bins if elev_bins is not None else cp.array([0, 1500, 2000, 3000, float('inf')], dtype=cp.float32)
        self.elev_scores = elev_scores if elev_scores is not None else cp.array([5, 3, 1, 0], dtype=cp.uint8)
        # Slope thresholds (percent) and scores
        self.slope_bins = slope_bins if slope_bins is not None else cp.array([0, 15, 30, 45, float('inf')], dtype=cp.float32)
        self.slope_scores = slope_scores if slope_scores is not None else cp.array([5, 3, 1, 0], dtype=cp.uint8)

    def calculate_slope(self, dem: cp.ndarray, resolution: float) -> cp.ndarray:
        """
        Calculate slope in percent on GPU.

        Args:
            dem (cp.ndarray): DEM tile on GPU
            resolution (float): Pixel resolution in meters

        Returns:
            cp.ndarray: Slope in percent
        """
        y, x = cp.gradient(dem, resolution)
        slope_rad = cp.arctan(cp.sqrt(x * x + y * y))
        slope_deg = slope_rad * 180.0 / cp.pi
        slope_percent = slope_deg * 100.0 / 45.0
        del x, y, slope_rad, slope_deg
        cp.cuda.Stream.null.synchronize()
        return slope_percent

    def classify_array(self, array: cp.ndarray, bins: cp.ndarray, scores: cp.ndarray) -> cp.ndarray:
        """
        Classify an array into scores based on bins on GPU.

        Args:
            array (cp.ndarray): Input array (elevation or slope)
            bins (cp.ndarray): Thresholds (e.g., [0, 1500, 2000, 3000, inf])
            scores (cp.ndarray): Scores for each bin (e.g., [5, 3, 1, 0])

        Returns:
            cp.ndarray: Classified array with scores
        """
        indices = cp.digitize(array, bins, right=False) - 1  # Classify into bins
        indices = cp.clip(indices, 0, len(scores) - 1)  # Ensure within bounds

        # Manual fix for binning accuracy
        for i in range(len(bins) - 1):
            mask = (array >= bins[i]) & (array < bins[i + 1])
            indices[mask] = i

        classified = scores[indices]
        del indices
        cp.cuda.Stream.null.synchronize()
        return classified

    def _process_tile(self, dem_tile: cp.ndarray, resolution: float) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Process a single DEM tile for elevation and slope classification.

        Args:
            dem_tile (cp.ndarray): DEM tile on GPU
            resolution (float): Pixel resolution in meters

        Returns:
            Tuple[cp.ndarray, cp.ndarray]: (elevation classification, slope classification)
        """
        elev_class = self.classify_array(dem_tile, self.elev_bins, self.elev_scores)
        slope = self.calculate_slope(dem_tile, resolution)
        slope_class = self.classify_array(slope, self.slope_bins, self.slope_scores)
        del slope
        cp.cuda.Stream.null.synchronize()
        return elev_class, slope_class

    def process_dem(self, input_path: str, output_prefix: str) -> None:
        """
        Process terrain data to classify elevation and slope suitability using tiling.

        Args:
            input_path (str): Path to input terrain file (.hgt, GeoTIFF, etc.)
            output_prefix (str): Prefix for output files
        """
        # Load and reproject DEM
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

        resolution = abs(profile['transform'][0])
        tile_size = 1024  # Adjust based on GPU memory
        rows, cols = dem.shape
        elev_class_output = cp.zeros((rows, cols), dtype=cp.uint8)
        slope_class_output = cp.zeros((rows, cols), dtype=cp.uint8)

        # Tile processing with overlap for slope calculation
        overlap = 1  # Minimal overlap for gradient calculation
        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                i_start = max(i - overlap, 0)
                j_start = max(j - overlap, 0)
                i_end = min(i + tile_size + overlap, rows)
                j_end = min(j + tile_size + overlap, cols)
                dem_tile = dem[i_start:i_end, j_start:j_end].copy()
                
                elev_class_tile, slope_class_tile = self._process_tile(dem_tile, resolution)
                
                # Trim overlap for output
                i_out_start = i - i_start
                j_out_start = j - j_start
                i_out_end = i_out_start + min(tile_size, rows - i)
                j_out_end = j_out_start + min(tile_size, cols - j)
                elev_class_output[i:i + tile_size, j:j + tile_size] = elev_class_tile[i_out_start:i_out_end, j_out_start:j_out_end]
                slope_class_output[i:i + tile_size, j:j + tile_size] = slope_class_tile[i_out_start:i_out_end, j_out_start:j_out_end]
                
                del dem_tile, elev_class_tile, slope_class_tile
                cp.cuda.Stream.null.synchronize()

        # Transfer to CPU and save
        elev_class_np = cp.asnumpy(elev_class_output)
        slope_class_np = cp.asnumpy(slope_class_output)

        profile.update(dtype=rasterio.uint8, nodata=255)
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with rasterio.open(f"{output_prefix}_elev_class.tif", 'w', **profile) as dst:
            dst.write(elev_class_np, 1)
            dst.write_colormap(1, {5: (0, 255, 0), 3: (255, 255, 0), 1: (255, 165, 0), 0: (255, 0, 0)})

        with rasterio.open(f"{output_prefix}_slope_class.tif", 'w', **profile) as dst:
            dst.write(slope_class_np, 1)
            dst.write_colormap(1, {5: (0, 255, 0), 3: (255, 255, 0), 1: (255, 165, 0), 0: (255, 0, 0)})

        print("GPU-accelerated suitability classification completed. Generated files:")
        print("Elevation:\n5=not suitable,\n3=marginally,\n1=moderately,\n0=highly")
        print(f"- {output_prefix}_elev_class.tif (Elevation: 5=not suitable, 3=marginally, 1=moderately, 0=highly)")
        print(f"- {output_prefix}_slope_class.tif (Slope: 5=not suitable, 3=marginally, 1=moderately, 0=highly)")

__all__ = ['GPUSuitabilityClassifier']