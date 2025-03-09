"""Utility functions for DEM processing in Slopy."""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cupy as cp
from typing import Tuple


def read_hgt(input_path: str) -> Tuple[cp.ndarray, dict, rasterio.crs.CRS]:
    """
    Read SRTM .hgt file and transfer to GPU.

    Args:
        input_path (str): Path to .hgt file (e.g., 'N01E110.hgt')

    Returns:
        tuple: (DEM array on GPU, rasterio profile, source CRS object)
    """
    if not input_path.endswith('.hgt'):
        raise ValueError("Input file must be in .hgt format")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"HGT file not found: {input_path}")

    base = os.path.basename(input_path).replace('.hgt', '')
    lat = int(base[1:3]) * (1 if base[0] == 'N' else -1)
    lon = int(base[4:7]) * (1 if base[3] == 'E' else -1)

    filesize = os.path.getsize(input_path)
    samples = 3601 if filesize == 25934402 else 1201 if filesize == 2884802 else None
    if samples is None:
        raise ValueError(f"Unsupported .hgt file size: {filesize} bytes")

    with open(input_path, 'rb') as f:
        dem_np = np.fromfile(f, dtype='>i2').reshape((samples, samples))
    dem = cp.asarray(dem_np)

    src_crs = rasterio.crs.CRS.from_epsg(4326)  # WGS84
    transform = rasterio.transform.from_bounds(lon, lat, lon + 1, lat + 1, samples, samples)
    profile = {
        'driver': 'GTiff', 'height': samples, 'width': samples,
        'count': 1, 'dtype': rasterio.int16, 'crs': src_crs,
        'transform': transform, 'nodata': -32768
    }
    return dem, profile, src_crs

def read_terrain(input_path: str) -> Tuple[cp.ndarray, dict, rasterio.crs.CRS]:
    """
    Read terrain data from .hgt or GDAL-supported formats and transfer to GPU.

    Args:
        input_path (str): Path to terrain file

    Returns:
        tuple: (DEM array on GPU, rasterio profile, source CRS object)
    """
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
            src_crs = src.crs if src.crs else rasterio.crs.CRS.from_epsg(4326)  # Assume WGS84 if unspecified
        dem = cp.asarray(dem_np)
        return dem, profile, src_crs
    
def get_utm_zone(lon: float, lat: float) -> str:
    """
    Determine UTM zone from longitude and latitude.

    Args:
        lon (float): Longitude in degrees
        lat (float): Latitude in degrees

    Returns:
        str: UTM CRS (e.g., 'EPSG:32633')
    """
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    return f"EPSG:{32600 + zone}" if hemisphere == 'N' else f"EPSG:{32700 + zone}"

def reproject_dem(dem: cp.ndarray, profile: dict, src_crs: str, dst_crs: str = None) -> Tuple[cp.ndarray, dict]:
    """
    Reproject DEM to UTM or user-specified CRS.

    Args:
        dem (cp.ndarray): Source DEM array on GPU
        profile (dict): Source rasterio profile
        src_crs (str): Source CRS (e.g., 'EPSG:4326')
        dst_crs (str, optional): Destination CRS. If None, auto-detects UTM zone.

    Returns:
        tuple: (reprojected DEM on GPU, updated profile)
    """
    if dst_crs is None:
        bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        dst_crs = get_utm_zone(center_lon, center_lat)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, profile['width'], profile['height'],
        left=profile['transform'][2], bottom=profile['transform'][5] + profile['height'] * profile['transform'][4],
        right=profile['transform'][2] + profile['width'] * profile['transform'][0], top=profile['transform'][5]
    )

    dem_np = cp.asnumpy(dem)
    dst_dem_np = np.zeros((dst_height, dst_width), dtype=np.int16)
    reproject(
        source=dem_np, destination=dst_dem_np,
        src_transform=profile['transform'], src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear, src_nodata=profile.get('nodata', -32768), dst_nodata=-32768
    )

    dst_dem = cp.asarray(dst_dem_np)
    profile.update({'crs': dst_crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height})
    return dst_dem, profile

__all__ = ['read_hgt', 'read_terrain', 'get_utm_zone', 'reproject_dem']