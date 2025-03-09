import pytest
import cupy as cp
import numpy as np
import rasterio
import os
from slopepy.hillshade import HillshadeCalculatorGPU
from slopepy.utils import read_hgt

# 📂 Path to the test HGT file
SAMPLE_HGT = os.path.join(os.path.dirname(__file__), "data", "N01E110.hgt")

@pytest.mark.gpu
def test_hillshade_init():
    """Test initialization of HillshadeCalculatorGPU."""
    calculator = HillshadeCalculatorGPU(azimuth=315.0, altitude=45.0, blur_window_sizes=[5, 10], dst_crs="EPSG:32633")
    assert calculator.AZIMUTH == 315.0
    assert calculator.ALTITUDE == 45.0
    assert calculator.blur_window_sizes == [5, 10]
    assert calculator.dst_crs == "EPSG:32633"

@pytest.mark.gpu
def test_hillshade_invalid_params():
    """Test invalid azimuth and altitude values."""
    with pytest.raises(ValueError):
        HillshadeCalculatorGPU(azimuth=361.0)
    with pytest.raises(ValueError):
        HillshadeCalculatorGPU(altitude=91.0)

@pytest.mark.gpu
def test_cuda_unavailable(monkeypatch):
    """Test behavior when CUDA is not available."""
    monkeypatch.setattr(cp.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        HillshadeCalculatorGPU()

@pytest.mark.gpu
def test_calculate_slope_aspect():
    """Test slope and aspect calculation using a simple DEM."""
    calculator = HillshadeCalculatorGPU()
    dem = cp.array([[10, 20], [30, 40]], dtype=cp.float32)  # Simulated DEM
    slope, aspect = calculator.calculate_slope_aspect(dem)

    assert slope.shape == dem.shape
    assert aspect.shape == dem.shape
    assert cp.all(cp.isfinite(slope))
    assert cp.all(cp.isfinite(aspect))

@pytest.mark.gpu
def test_calculate_hillshade():
    """Test hillshade calculation on a simple slope."""
    calculator = HillshadeCalculatorGPU()
    dem = cp.array([[10, 20], [30, 40]], dtype=cp.float32)
    slope, aspect = calculator.calculate_slope_aspect(dem)
    hillshade = calculator.calculate_hillshade(slope, aspect)

    assert hillshade.shape == dem.shape
    assert hillshade.dtype == cp.uint8
    assert cp.all(hillshade >= 0) and cp.all(hillshade <= 255)

@pytest.mark.gpu
def test_process_dem_invalid_path():
    """Test processing with a non-existent file."""
    calculator = HillshadeCalculatorGPU()
    with pytest.raises(FileNotFoundError):
        calculator.process_dem("nonexistent_file.hgt", "output/test")

@pytest.mark.skipif(not os.path.exists(SAMPLE_HGT), reason="Sample HGT file not found in tests/data/")
@pytest.mark.gpu
def test_process_dem(tmp_path):
    """Test hillshade processing with the sample HGT file."""
    calculator = HillshadeCalculatorGPU(blur_window_sizes=[10])
    output_prefix = str(tmp_path / "hillshade_output")

    calculator.process_dem(SAMPLE_HGT, output_prefix)

    # ✅ Check if output files exist
    assert os.path.exists(f"{output_prefix}_original.tif")
    assert os.path.exists(f"{output_prefix}_blur_10.tif")

    # ✅ Validate output TIF files
    with rasterio.open(f"{output_prefix}_original.tif") as src:
        assert src.count == 1
        assert src.dtypes[0] == 'uint8'
        assert src.nodata == 255

    with rasterio.open(f"{output_prefix}_blur_10.tif") as src:
        assert src.count == 1
        assert src.dtypes[0] == 'uint8'

__all__ = [
    'test_hillshade_init', 'test_hillshade_invalid_params', 'test_cuda_unavailable',
    'test_calculate_slope_aspect', 'test_calculate_hillshade', 'test_process_dem_invalid_path', 'test_process_dem'
]
