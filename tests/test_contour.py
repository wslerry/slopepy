import os
import pytest
import tempfile
import geopandas as gpd
import numpy as np
import rasterio
from unittest.mock import patch
from rasterio.transform import from_bounds
from unittest.mock import patch
from shapely.geometry import LineString
from slopepy.contour import GPUContourGenerator, CUDA_AVAILABLE, MarchingSquares


# Path to test HGT file
SMOL_DEM = os.path.join(os.path.dirname(__file__), "data", "small_dem.tif")
SAMPLE_HGT = os.path.join(os.path.dirname(__file__), "data", "N01E110.hgt")

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

# @pytest.fixture
# def tmp_output():
#     # Create a temporary directory and file prefix
#     with tempfile.TemporaryDirectory() as tmpdir:
#         output_prefix = os.path.join(tmpdir, "output")
#         yield output_prefix


@pytest.mark.gpu
def test_gpu_availability():
    assert isinstance(CUDA_AVAILABLE, bool)
    if CUDA_AVAILABLE:
        import cupy
        assert cupy.cuda.is_available()

@pytest.mark.gpu
def test_init_custom():
    generator = GPUContourGenerator(contour_levels=[50, 100], dst_crs="EPSG:32649", force_gpu=True)
    assert generator.contour_levels == [50, 100]
    assert generator.dst_crs == "EPSG:32649"
    assert generator.force_gpu is True

@patch('slopepy.contour.read_terrain')
@patch('slopepy.contour.reproject_dem')
def test_process_dem_small_gpu(mock_reproject, mock_read, small_dem, tmp_output):
    dem, profile, src_crs = small_dem
    mock_read.return_value = (dem, profile, src_crs)
    mock_reproject.return_value = (dem, profile)

    output_prefix = tmp_output  # already a file-like path string

    generator = GPUContourGenerator(
        contour_levels=[150],
        dst_crs="EPSG:32649",
        force_gpu=CUDA_AVAILABLE
    )

    # ✅ Generate contours
    generator.process_dem(SMOL_DEM, output_prefix, save_vector=True)

    # ✅ Check generated file
    gpkg_path = os.path.join(tmp_output, f"{output_prefix}_contours.gpkg")
    assert os.path.exists(gpkg_path), "Contour GeoPackage file was not created!"

    # ✅ Verify file contents
    gdf = gpd.read_file(gpkg_path)
    print(gdf.head())
    assert len(gdf) > 0, "No contours found in the generated file!"
    assert all(gdf['elevation'] == 150), "Unexpected contour elevation values!"
    assert all(isinstance(geom, LineString) for geom in gdf['geometry']), "Invalid geometry type!"

@patch('slopepy.contour.read_terrain')
def test_process_dem_invalid_file(mock_read, tmp_output):
    mock_read.side_effect = FileNotFoundError("File not found")
    generator = GPUContourGenerator(contour_levels=[100])
    output_prefix = str(tmp_output / "contour")
    with pytest.raises(FileNotFoundError):
        generator.process_dem("nonexistent.hgt", output_prefix)

# @pytest.mark.skipif(not os.path.exists(SAMPLE_HGT), reason="Sample HGT file not found in tests/data/")
# @pytest.mark.gpu
# def test_process_dem_real_data(tmp_output):
#     output_prefix = str(tmp_output.with_name("contour_real"))  # creates "contour_real" in the temp dir

#     generator = GPUContourGenerator(contour_levels=[50], dst_crs="EPSG:32649")
#     generator.process_dem(SAMPLE_HGT, output_prefix, save_vector=True)

#     gpkg_path = os.path.join(tmp_output, f"{output_prefix}_contours.gpkg")
#     assert os.path.exists(gpkg_path), "Contour file was not created!"

#     if gpd:
#         gdf = gpd.read_file(gpkg_path, layer="Contour_50")
#         assert len(gdf) > 0, "No contours found!"
#         assert all(gdf['elevation'] == 50), "Unexpected contour elevation values!"
#         assert all(isinstance(geom, LineString) for geom in gdf['geometry']), "Invalid geometry type!"
#         assert gdf.crs.to_string() == "EPSG:32649", "CRS does not match expected!"

@pytest.mark.gpu
def test_marching_squares_contours(small_dem):
    dem, profile, _ = small_dem
    ms = MarchingSquares(contour_levels=[150], use_gpu=CUDA_AVAILABLE)
    xp = ms.xp
    dem_xp = xp.asarray(dem)

    contours = ms.generate_contours(dem_xp, profile['transform'])
    print("Generated contours:", contours)
    assert len(contours) > 0, "No contours generated! Check contour levels and DEM range."
    for contour in contours:
        assert contour['elevation'] == 150
        assert isinstance(contour['geometry'], LineString)
        assert len(contour['geometry'].coords) >= 2


if __name__ == "__main__":
    pytest.main(["-v"])
