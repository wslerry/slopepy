"""Tests for slopepy.contour module."""

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from unittest.mock import patch
import os
import geopandas as gpd
from shapely.geometry import LineString
from slopepy.contour import GPUContourGenerator, CUDA_AVAILABLE, MarchingSquares

@pytest.fixture
def small_dem():
    dem = np.array([
        [100, 120, 130, 140],
        [110, 150, 160, 170],
        [200, 210, 220, 230],
        [240, 250, 260, 270]
    ], dtype=np.float32)
    transform = from_bounds(0, 0, 4, 4, 4, 4)
    profile = {
        'driver': 'GTiff', 'height': 4, 'width': 4, 'count': 1,
        'dtype': rasterio.float32, 'crs': 'EPSG:4326', 'transform': transform,
        'nodata': -9999
    }
    return dem, profile, 'EPSG:4326'

@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "output"

@pytest.fixture
def hgt_file():
    hgt_path = os.path.join(os.path.dirname(__file__), "data", "N01E110.hgt")
    if not os.path.exists(hgt_path):
        pytest.skip("N01E110.hgt not found in tests/data. Skipping real-world test.")
    return hgt_path

def test_gpu_availability():
    assert isinstance(CUDA_AVAILABLE, bool)
    if CUDA_AVAILABLE:
        import cupy
        assert cupy.cuda.is_available()

def test_init_default():
    generator = GPUContourGenerator()
    assert generator.contour_levels == [10, 20, 50, 100]
    assert generator.dst_crs is None
    assert generator.force_gpu is False
    assert generator.use_gpu is False

def test_init_custom():
    generator = GPUContourGenerator(contour_levels=[10, 20], dst_crs="EPSG:3857", force_gpu=True)
    assert generator.contour_levels == [10, 20]
    assert generator.dst_crs == "EPSG:3857"
    assert generator.force_gpu is True

@patch('slopepy.contour.read_terrain')
@patch('slopepy.contour.reproject_dem')
def test_process_dem_small_gpu(mock_reproject, mock_read, small_dem, tmp_output):
    dem, profile, src_crs = small_dem
    mock_read.return_value = (dem, profile, src_crs)
    mock_reproject.return_value = (dem, profile)
    
    output_prefix = str(tmp_output / "contour")
    generator = GPUContourGenerator(contour_levels=[10], force_gpu=CUDA_AVAILABLE)
    
    generator.process_dem("fake_path.hgt", output_prefix, save_vector=True)
    
    assert generator.use_gpu == CUDA_AVAILABLE
    gpkg_path = f"{output_prefix}_contours.gpkg"
    if gpd:
        assert os.path.exists(gpkg_path)
        gdf = gpd.read_file(gpkg_path, layer="Contour_150")
        assert len(gdf) > 0
        assert all(gdf['elevation'] == 10)
        assert all(isinstance(geom, LineString) for geom in gdf['geometry'])

@patch('slopepy.contour.read_terrain')
@patch('slopepy.contour.reproject_dem')
def test_process_dem_force_gpu_unavailable(mock_reproject, mock_read, small_dem, tmp_output):
    dem, profile, src_crs = small_dem
    mock_read.return_value = (dem, profile, src_crs)
    mock_reproject.return_value = (dem, profile)
    
    with patch('slopepy.contour.CUDA_AVAILABLE', False):
        generator = GPUContourGenerator(contour_levels=[150], force_gpu=True)
        output_prefix = str(tmp_output / "contour")
        with pytest.raises(RuntimeError, match="GPU forced but not available"):
            generator.process_dem("fake_path.hgt", output_prefix)

def test_process_dem_hgt(hgt_file, tmp_output):
    output_prefix = str(tmp_output / "contour_hgt")
    generator = GPUContourGenerator(contour_levels=[10, 20], dst_crs="EPSG:32649")
    
    generator.process_dem(hgt_file, output_prefix, save_vector=True)
    
    gpkg_path = f"{output_prefix}_contours.gpkg"
    if gpd:
        assert os.path.exists(gpkg_path)
        for level in [10, 20]:
            try:
                gdf = gpd.read_file(gpkg_path, layer=f"Contour_{level}")
                if len(gdf) > 0:
                    assert all(gdf['elevation'] == level)
                    assert all(isinstance(geom, LineString) for geom in gdf['geometry'])
                    assert gdf.crs.to_string() == "EPSG:32649"
            except Exception as e:
                print(f"No contours found for level {level}: {e}")

@patch('slopepy.contour.read_terrain')
@patch('slopepy.contour.reproject_dem')
def test_process_dem_large_fallback(mock_reproject, mock_read, tmp_output):
    large_dem = np.random.rand(10000, 10000).astype(np.float32) * 2000
    transform = from_bounds(0, 0, 10000, 10000, 10000, 10000)
    profile = {
        'driver': 'GTiff', 'height': 10000, 'width': 10000, 'count': 1,
        'dtype': rasterio.float32, 'crs': 'EPSG:4326', 'transform': transform,
        'nodata': -9999
    }
    mock_read.return_value = (large_dem, profile, 'EPSG:4326')
    mock_reproject.return_value = (large_dem, profile)
    
    output_prefix = str(tmp_output / "contour_large")
    generator = GPUContourGenerator(contour_levels=[1000])
    
    with patch('slopepy.contour.MarchingSquares.generate_contours', side_effect=MemoryError("Mock GPU OOM")):
        generator.process_dem("fake_path.hgt", output_prefix, save_vector=True)
        assert not generator.use_gpu
        gpkg_path = f"{output_prefix}_contours.gpkg"
        if gpd:
            assert os.path.exists(gpkg_path)
            gdf = gpd.read_file(gpkg_path, layer="Contour_1000")
            assert len(gdf) > 0

@patch('slopepy.contour.read_terrain')
def test_process_dem_invalid_file(mock_read, tmp_output):
    mock_read.side_effect = FileNotFoundError("File not found")
    generator = GPUContourGenerator(contour_levels=[100])
    output_prefix = str(tmp_output / "contour")
    with pytest.raises(FileNotFoundError):
        generator.process_dem("nonexistent.hgt", output_prefix)

@patch('slopepy.contour.read_terrain')
@patch('slopepy.contour.reproject_dem')
@patch('slopepy.contour.gpd', None)
def test_process_dem_no_geopandas(mock_reproject, mock_read, small_dem, tmp_output, capsys):
    dem, profile, src_crs = small_dem
    mock_read.return_value = (dem, profile, src_crs)
    mock_reproject.return_value = (dem, profile)
    
    output_prefix = str(tmp_output / "contour")
    generator = GPUContourGenerator(contour_levels=[20])
    generator.process_dem("fake_path.hgt", output_prefix, save_vector=True)
    
    captured = capsys.readouterr()
    assert "Warning: geopandas or shapely not installed" in captured.out
    assert not os.path.exists(f"{output_prefix}_contours.gpkg")

def test_marching_squares_contours(small_dem):
    dem, profile, _ = small_dem
    ms = MarchingSquares(contour_levels=[20], use_gpu=CUDA_AVAILABLE)
    xp = ms.xp
    dem_xp = xp.asarray(dem)
    
    contours = ms.generate_contours(dem_xp, profile['transform'])
    assert len(contours) > 0
    for contour in contours:
        assert contour['elevation'] == 20
        assert isinstance(contour['geometry'], LineString)
        assert len(contour['geometry'].coords) == 2

if __name__ == "__main__":
    pytest.main(["-v"])