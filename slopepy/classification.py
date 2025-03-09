"""GPU-accelerated DEM suitability classification module."""

import cupy as cp
import numpy as np
import rasterio
from typing import Tuple
import os
from .utils import read_terrain, reproject_dem

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
            dem (cp.ndarray): DEM array on GPU
            resolution (float): Pixel resolution in meters

        Returns:
            cp.ndarray: Slope in percent
        """
        y, x = cp.gradient(dem, resolution)
        slope_rad = cp.arctan(cp.sqrt(x * x + y * y))
        slope_deg = slope_rad * 180.0 / cp.pi
        return slope_deg * 100.0 / 45.0

    # def classify_array(self, array: cp.ndarray, bins: cp.ndarray, scores: cp.ndarray) -> cp.ndarray:
    #     """
    #     Classify an array into scores based on bins on GPU.

    #     Args:
    #         array (cp.ndarray): Input array (elevation or slope)
    #         bins (cp.ndarray): Thresholds
    #         scores (cp.ndarray): Scores for each bin

    #     Returns:
    #         cp.ndarray: Classified array with scores
    #     """
    #     # indices = cp.digitize(array, bins) - 1
    #     indices = cp.digitize(array, bins, right=True) - 1
    #     indices = cp.clip(indices, 0, len(scores) - 1)
    #     return scores[indices]
    
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

        # ✅ **Manual fix for values that incorrectly map to a lower bin**
        for i in range(len(bins) - 1):
            mask = (array >= bins[i]) & (array < bins[i + 1])  # Ensure values within correct range
            indices[mask] = i  # Correct the bin index

        return scores[indices]

    def process_dem(self, input_path: str, output_prefix: str) -> None:
        """
        Process terrain data to classify elevation and slope suitability.

        Args:
            input_path (str): Path to input terrain file (.hgt, GeoTIFF, etc.)
            output_prefix (str): Prefix for output files
        """
        dem, profile, src_crs = read_terrain(input_path)
        dem, profile = reproject_dem(dem, profile, src_crs, self.dst_crs)

        resolution = abs(profile['transform'][0])
        elev_class = self.classify_array(dem, self.elev_bins, self.elev_scores)
        slope = self.calculate_slope(dem, resolution)
        slope_class = self.classify_array(slope, self.slope_bins, self.slope_scores)

        elev_class_np = cp.asnumpy(elev_class)
        slope_class_np = cp.asnumpy(slope_class)

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
        print(f"- {output_prefix}_elev_class.tif (Elevation: 5=highly, 3=moderately, 1=marginally, 0=not suitable)")
        print(f"- {output_prefix}_slope_class.tif (Slope: 5=highly, 3=moderately, 1=marginally, 0=not suitable)")

__all__ = ['GPUSuitabilityClassifier']