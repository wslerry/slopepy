"""Utility functions for DEM processing in Slopy."""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cupy as cp
from typing import Tuple, Optional

def read_hgt(input_path: str) -> Tuple[cp.ndarray, dict, rasterio.crs.CRS]:
    if not input_path.endswith('.hgt'):
        raise ValueError("Input file must be in .hgt format")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"HGT file not found: {input_path}")

    base = os.path.basename(input_path).replace('.hgt', '')  # Fixed typo: 'baseline' -> 'base'
    lat = int(base[1:3]) * (1 if base[0] == 'N' else -1)
    lon = int(base[4:7]) * (1 if base[3] == 'E' else -1)

    filesize = os.path.getsize(input_path)
    samples = 3601 if filesize == 25934402 else 1201 if filesize == 2884802 else None
    if samples is None:
        raise ValueError(f"Unsupported .hgt file size: {filesize} bytes")

    with open(input_path, 'rb') as f:
        dem_np = np.fromfile(f, dtype='>i2').reshape((samples, samples))
    dem = cp.asarray(dem_np)

    src_crs = rasterio.crs.CRS.from_epsg(4326)
    transform = rasterio.transform.from_bounds(lon, lat, lon + 1, lat + 1, samples, samples)
    profile = {
        'driver': 'GTiff', 'height': samples, 'width': samples,
        'count': 1, 'dtype': rasterio.int16, 'crs': src_crs,
        'transform': transform, 'nodata': -32768
    }
    return dem, profile, src_crs

def read_terrain(input_path: str) -> Tuple[cp.ndarray, dict, rasterio.crs.CRS]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Terrain file not found: {input_path}")

    if input_path.endswith('.hgt'):
        return read_hgt(input_path)
    else:
        with rasterio.open(input_path) as src:
            if src.count != 1:
                raise ValueError(f"Expected single-band terrain data, got {src.count} bands")
            dem_np = src.read(1)
            profile = src.profile.copy()
            src_crs = src.crs if src.crs else rasterio.crs.CRS.from_epsg(4326)
        dem = cp.asarray(dem_np)
        return dem, profile, src_crs

def get_utm_zone(lon: float, lat: float) -> rasterio.crs.CRS:
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    epsg = 32600 + zone if hemisphere == 'N' else 32700 + zone
    return rasterio.crs.CRS.from_epsg(epsg)

def reproject_dem(dem: cp.ndarray, profile: dict, src_crs: rasterio.crs.CRS, dst_crs: Optional[str] = None) -> Tuple[cp.ndarray, dict]:
    if dst_crs is None:
        bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        dst_crs_obj = get_utm_zone(center_lon, center_lat)
    else:
        dst_crs_obj = rasterio.crs.CRS.from_string(dst_crs)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs_obj, profile['width'], profile['height'],
        left=profile['transform'][2], bottom=profile['transform'][5],
        right=profile['transform'][2] + profile['width'] * profile['transform'][0],
        top=profile['transform'][5] + profile['height'] * profile['transform'][4]
    )

    dem_np = cp.asnumpy(dem)
    dst_dem_np = np.zeros((dst_height, dst_width), dtype=dem_np.dtype)
    reproject(
        source=dem_np, destination=dst_dem_np,
        src_transform=profile['transform'], src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs_obj,
        resampling=Resampling.bilinear, src_nodata=profile.get('nodata', -32768), dst_nodata=-32768
    )

    dst_dem = cp.asarray(dst_dem_np)
    profile.update({'crs': dst_crs_obj, 'transform': dst_transform, 'width': dst_width, 'height': dst_height})
    return dst_dem, profile

__all__ = ['read_hgt', 'read_terrain', 'get_utm_zone', 'reproject_dem']