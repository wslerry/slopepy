"""GPU-accelerated feature-preserving DEM smoothing module based on Sun et al. (2007) and Lindsay et al. (2019)."""

import cupy as cp
import numpy as np
import rasterio
import os
from typing import Tuple, Optional
from .utils import read_terrain, reproject_dem, get_utm_zone

class FeaturePreservingSmoothing:
    """A GPU-accelerated class to perform feature-preserving smoothing on DEMs."""

    def __init__(self, filter_size: int = 11, norm_diff: float = 15.0, num_iter: int = 3,
                 max_diff: Optional[float] = 0.5, z_factor: Optional[float] = None, 
                 dst_crs: Optional[str] = None) -> None:
        """
        Initialize the FeaturePreservingSmoothing class.

        Args:
            filter_size (int): Size of the smoothing filter (odd integer >= 3). Default is 11.
            norm_diff (float): Normal difference threshold in degrees (<= 180). Default is 15.0.
            num_iter (int): Number of smoothing iterations (>= 1). Default is 3.
            max_diff (Optional[float]): Maximum elevation difference per iteration. Default is 0.5.
            z_factor (Optional[float]): Scaling factor for elevation. Auto-computed if None for geographic CRS.
            dst_crs (Optional[str]): Destination CRS (e.g., 'EPSG:32633'). If None, defaults to UTM.
        """
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
        self.dst_crs = dst_crs  # Store user-provided CRS
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
                v = cp.zeros_like(dem, dtype=cp.float32)
            v += values * (1 if i in [0, 2, 6] else 2 if i in [1, 5] else -1 if i in [4, 6] else -2 if i in [3, 7] else 0)
        normals[..., 0] = -v / eight_res_x
        normals[..., 1] = -v / eight_res_y
        normals[dem == nodata] = 0
        del v, dem_padded
        cp.cuda.Stream.null.synchronize()
        return normals

    def _smooth_normals(self, dem: cp.ndarray, normals: cp.ndarray, nodata: float) -> cp.ndarray:
        rows, cols = dem.shape
        smoothed = cp.zeros_like(normals)
        dem_padded = cp.pad(dem, self.filter_size // 2, mode='edge')
        normals_padded = cp.pad(normals, ((self.filter_size // 2, self.filter_size // 2), 
                                         (self.filter_size // 2, self.filter_size // 2), (0, 0)), mode='constant')

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
            sum_w = cp.sum(w) if i == 0 else sum_w + cp.sum(w)
        smoothed /= cp.where(sum_w > 0, sum_w, 1)
        smoothed[dem == nodata] = 0
        del dem_padded, normals_padded, w
        cp.cuda.Stream.null.synchronize()
        return smoothed

    def _angle_between(self, n1: cp.ndarray, n2: cp.ndarray) -> cp.ndarray:
        denom = cp.sqrt((n1[..., 0]**2 + n1[..., 1]**2 + 1) * (n2[..., 0]**2 + n2[..., 1]**2 + 1))
        result = (n1[..., 0] * n2[..., 0] + n1[..., 1] * n2[..., 1] + 1) / denom
        del denom
        cp.cuda.Stream.null.synchronize()
        return result

    def _update_elevations(self, dem: cp.ndarray, normals: cp.ndarray, nodata: float,
                          res_x: float, res_y: float) -> cp.ndarray:
        output = dem.astype(cp.float32)
        dx = cp.array([1, 1, 1, 0, -1, -1, -1, 0], dtype=cp.int32)
        dy = cp.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=cp.int32)
        x = cp.array([-res_x, -res_x, -res_x, 0, res_x, res_x, res_x, 0], dtype=cp.float32)
        y = cp.array([-res_y, 0, res_y, res_y, res_y, 0, -res_y, -res_y], dtype=cp.float32)

        for _ in range(self.num_iter):
            dem_padded = cp.pad(output, 1, mode='edge')
            normals_padded = cp.pad(normals, ((1, 1), (1, 1), (0, 0)), mode='constant')
            z = cp.zeros_like(output, dtype=cp.float32)
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
            del dem_padded, normals_padded, z, sum_w, zn
            cp.cuda.Stream.null.synchronize()
        return output

    def _process_tile(self, dem_tile: cp.ndarray, res_x: float, res_y: float, nodata: float) -> cp.ndarray:
        """Process a single tile for smoothing."""
        normals = self._calculate_normals(dem_tile, res_x, res_y, nodata)
        smoothed_normals = self._smooth_normals(dem_tile, normals, nodata)
        smoothed_dem = self._update_elevations(dem_tile, smoothed_normals, nodata, res_x, res_y)
        del normals, smoothed_normals
        cp.cuda.Stream.null.synchronize()
        return smoothed_dem

    def process_dem(self, input_path: str, output_path: str, dst_crs: Optional[str] = None) -> None:
        """Process a DEM file to apply GPU-accelerated feature-preserving smoothing with tiling.

        Args:
            input_path (str): Path to input DEM file.
            output_path (str): Path to output smoothed DEM file.
            dst_crs (Optional[str]): Destination CRS. If None, defaults to UTM; overrides class-level dst_crs if provided.
        """
        dem, profile, src_crs = read_terrain(input_path)
        nodata = profile['nodata']
        res_x, res_y = abs(profile['transform'][0]), abs(profile['transform'][4])

        # Determine effective CRS: user-provided > class-level > UTM if geographic > input CRS
        effective_dst_crs = dst_crs if dst_crs is not None else self.dst_crs
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

        if profile['crs'].is_geographic and self.z_factor is None:
            bounds = profile['transform'] * (0, 0) + profile['transform'] * (profile['width'], profile['height'])
            mid_lat = cp.deg2rad((bounds[1] + bounds[3]) / 2)
            self.z_factor = cp.float32(1.0) / (cp.float32(111320.0) * cp.cos(mid_lat))
            print(f"DEM in geographic coordinates. Z-factor set to {float(self.z_factor):.6f}")
        elif self.z_factor is None:
            self.z_factor = cp.float32(1.0)

        tile_size = 1024
        rows, cols = dem.shape
        output = cp.zeros((rows, cols), dtype=cp.float32)
        overlap = self.filter_size // 2 + 1  # Enough overlap for padding

        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                i_start = max(i - overlap, 0)
                j_start = max(j - overlap, 0)
                i_end = min(i + tile_size + overlap, rows)
                j_end = min(j + tile_size + overlap, cols)
                dem_tile = dem[i_start:i_end, j_start:j_end].copy()
                smoothed_tile = self._process_tile(dem_tile, res_x, res_y, nodata)
                # Adjust indices to exclude overlap in output
                i_out_start = i - i_start
                j_out_start = j - j_start
                i_out_end = i_out_start + min(tile_size, rows - i)
                j_out_end = j_out_start + min(tile_size, cols - j)
                output[i:i + tile_size, j:j + tile_size] = smoothed_tile[i_out_start:i_out_end, j_out_start:j_out_end]
                del dem_tile, smoothed_tile
                cp.cuda.Stream.null.synchronize()

        profile.update(dtype=rasterio.float32, nodata=nodata)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(cp.asnumpy(output), 1)
        print(f"GPU-accelerated feature-preserving smoothing completed. Output saved to {output_path}")

__all__ = ['FeaturePreservingSmoothing']