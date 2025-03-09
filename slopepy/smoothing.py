"""GPU-accelerated feature-preserving DEM smoothing module based on Sun et al. (2007) and Lindsay et al. (2019)."""

import cupy as cp
import numpy as np
import rasterio
import os
from typing import Tuple, Optional
from .utils import read_terrain

class FeaturePreservingSmoothing:
    """A GPU-accelerated class to perform feature-preserving smoothing on DEMs."""

    def __init__(self, filter_size: int = 11, norm_diff: float = 15.0, num_iter: int = 3,
                 max_diff: Optional[float] = 0.5, z_factor: Optional[float] = None) -> None:
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA is not available. This class requires a GPU with CuPy support.")
        if filter_size < 3 or filter_size % 2 == 0:
            raise ValueError("Filter size must be an odd integer >= 3")
        if num_iter < 1:
            raise ValueError("Number of iterations must be >= 1")
        if norm_diff > 180.0:
            raise ValueError("Normal difference threshold must be <= 180 degrees")

        self.filter_size = filter_size
        self.norm_diff = norm_diff
        self.num_iter = num_iter
        self.max_diff = max_diff if max_diff is not None else float('inf')
        self.z_factor = z_factor
        self.threshold = cp.float32(np.cos(np.radians(norm_diff)))

        midpoint = filter_size // 2
        dx = cp.arange(filter_size) - midpoint
        dy = cp.arange(filter_size) - midpoint
        self.dx, self.dy = cp.meshgrid(dx, dy)
        self.dx, self.dy = self.dx.ravel(), self.dy.ravel()

    def _calculate_normals(self, dem: cp.ndarray, res_x: float, res_y: float, nodata: float) -> cp.ndarray:
        rows, cols = dem.shape
        normals = cp.zeros((rows, cols, 2), dtype=cp.float32)
        dx = cp.array([1, 1, 1, 0, -1, -1, -1, 0], dtype=cp.int32)
        dy = cp.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=cp.int32)
        eight_res_x = cp.float32(res_x * 8.0)
        eight_res_y = cp.float32(res_y * 8.0)

        dem_padded = cp.pad(dem, 1, mode='edge')
        for i in range(8):
            shifted = dem_padded[1 + dy[i]:rows + 1 + dy[i], 1 + dx[i]:cols + 1 + dx[i]]
            mask = (shifted != nodata)
            values = cp.where(mask, shifted * self.z_factor, dem * self.z_factor)

            if i == 0:
                v = cp.zeros_like(dem, dtype=cp.float32)  # Ensure float32 here
            v += values * (1 if i in [0, 2, 6] else 2 if i in [1, 5] else -1 if i in [4, 6] else -2 if i in [3, 7] else 0)

        a = -v / eight_res_x
        b = -v / eight_res_y
        normals[..., 0] = a
        normals[..., 1] = b
        normals[dem == nodata] = 0
        return normals

    def _smooth_normals(self, dem: cp.ndarray, normals: cp.ndarray, nodata: float) -> cp.ndarray:
        rows, cols = dem.shape
        smoothed = cp.zeros_like(normals)
        dem_padded = cp.pad(dem, self.filter_size // 2, mode='edge')
        normals_padded = cp.pad(normals, ((self.filter_size // 2, self.filter_size // 2), (self.filter_size // 2, self.filter_size // 2), (0, 0)), mode='constant')

        for i in range(self.filter_size * self.filter_size):
            r_shift, c_shift = self.dy[i], self.dx[i]
            n_shifted = normals_padded[self.filter_size // 2 + r_shift:rows + self.filter_size // 2 + r_shift,
                                       self.filter_size // 2 + c_shift:cols + self.filter_size // 2 + c_shift]
            diff = self._angle_between(normals, n_shifted)
            mask = (diff > self.threshold) & (dem_padded[self.filter_size // 2 + r_shift:rows + self.filter_size // 2 + r_shift,
                                                         self.filter_size // 2 + c_shift:cols + self.filter_size // 2 + c_shift] != nodata)
            w = cp.where(mask, (diff - self.threshold) ** 2, 0)
            smoothed[..., 0] += cp.where(mask, n_shifted[..., 0] * w, 0)
            smoothed[..., 1] += cp.where(mask, n_shifted[..., 1] * w, 0)
            sum_w = cp.sum(w, axis=None) if i == 0 else sum_w + cp.sum(w, axis=None)

        smoothed /= cp.where(sum_w > 0, sum_w, 1)
        smoothed[dem == nodata] = 0
        return smoothed

    def _angle_between(self, n1: cp.ndarray, n2: cp.ndarray) -> cp.ndarray:
        denom = cp.sqrt((n1[..., 0]**2 + n1[..., 1]**2 + 1) * (n2[..., 0]**2 + n2[..., 1]**2 + 1))
        return (n1[..., 0] * n2[..., 0] + n1[..., 1] * n2[..., 1] + 1) / denom

    def _update_elevations(self, dem: cp.ndarray, normals: cp.ndarray, nodata: float,
                          res_x: float, res_y: float) -> cp.ndarray:
        output = dem.astype(cp.float32)  # Convert to float32 upfront
        dx = cp.array([1, 1, 1, 0, -1, -1, -1, 0], dtype=cp.int32)
        dy = cp.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=cp.int32)
        x = cp.array([-res_x, -res_x, -res_x, 0, res_x, res_x, res_x, 0], dtype=cp.float32)
        y = cp.array([-res_y, 0, res_y, res_y, res_y, 0, -res_y, -res_y], dtype=cp.float32)

        for _ in range(self.num_iter):
            dem_padded = cp.pad(output, 1, mode='edge')
            normals_padded = cp.pad(normals, ((1, 1), (1, 1), (0, 0)), mode='constant')
            z = cp.zeros_like(output, dtype=cp.float32)  # Explicitly float32
            sum_w = cp.zeros_like(output, dtype=cp.float32)

            for n in range(8):
                shifted_dem = dem_padded[1 + dy[n]:dem.shape[0] + 1 + dy[n], 1 + dx[n]:dem.shape[1] + 1 + dx[n]]
                shifted_normals = normals_padded[1 + dy[n]:dem.shape[0] + 1 + dy[n], 1 + dx[n]:dem.shape[1] + 1 + dx[n]]
                diff = self._angle_between(normals, shifted_normals)
                mask = (diff > self.threshold) & (shifted_dem != nodata)
                w = cp.where(mask, (diff - self.threshold) ** 2, 0)
                z += cp.where(mask, -(shifted_normals[..., 0] * x[n] + shifted_normals[..., 1] * y[n] - shifted_dem) * w, 0)
                sum_w += w

            zn = cp.where(sum_w > 0, z / sum_w, output)
            diff = cp.abs(zn - output)
            output = cp.where((diff <= self.max_diff) & (output != nodata), zn, output)

        return output

    def process_dem(self, input_path: str, output_path: str) -> None:
        """Process a DEM file to apply GPU-accelerated feature-preserving smoothing."""
        dem, profile, _ = read_terrain(input_path)
        nodata = profile['nodata']
        res_x, res_y = abs(profile['transform'][0]), abs(profile['transform'][4])

        if profile['crs'].is_geographic and self.z_factor is None:
            bounds = profile['transform'] * (0, 0) + profile['transform'] * (profile['width'], profile['height'])
            mid_lat = cp.deg2rad((bounds[1] + bounds[3]) / 2)
            self.z_factor = cp.float32(1.0) / (cp.float32(111320.0) * cp.cos(mid_lat))
            print(f"DEM in geographic coordinates. Z-factor set to {float(self.z_factor):.6f}")
        elif self.z_factor is None:
            self.z_factor = cp.float32(1.0)

        normals = self._calculate_normals(dem, res_x, res_y, nodata)
        smoothed_normals = self._smooth_normals(dem, normals, nodata)
        smoothed_dem = self._update_elevations(dem, smoothed_normals, nodata, res_x, res_y)

        profile.update(dtype=rasterio.float32, nodata=nodata)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(cp.asnumpy(smoothed_dem).astype(np.float32), 1)
            # dst.add_tags(
            #     TOOL="FeaturePreservingSmoothing",
            #     INPUT_FILE=input_path,
            #     FILTER_SIZE=str(self.filter_size),
            #     NORM_DIFF=str(self.norm_diff),
            #     NUM_ITER=str(self.num_iter),
            #     MAX_DIFF=str(self.max_diff),
            #     Z_FACTOR=str(float(self.z_factor))
            # )

        print(f"GPU-accelerated feature-preserving smoothing completed. Output saved to {output_path}")

__all__ = ['FeaturePreservingSmoothing']