import pytest
import cupy as cp
import numpy as np
import rasterio
import os
from slopepy.classification import GPUSuitabilityClassifier
from slopepy.utils import read_terrain

# Path to the sample HGT file
SAMPLE_HGT = os.path.join(os.path.dirname(__file__), "data", "N01E110.hgt")

def test_classifier_init():
    """Test initialization of GPUSuitabilityClassifier."""
    classifier = GPUSuitabilityClassifier(dst_crs="EPSG:32633")
    assert classifier.dst_crs == "EPSG:32633"
    assert cp.all(classifier.elev_bins == cp.array([0, 1500, 2000, 3000, float('inf')]))
    assert cp.all(classifier.elev_scores == cp.array([5, 3, 1, 0]))
    assert cp.all(classifier.slope_bins == cp.array([0, 15, 30, 45, float('inf')]))
    assert cp.all(classifier.slope_scores == cp.array([5, 3, 1, 0]))

def test_classifier_custom_bins():
    """Test initialization with custom bins and scores."""
    elev_bins = cp.array([0, 1000, 2000, float('inf')], dtype=cp.float32)
    elev_scores = cp.array([4, 2, 0], dtype=cp.uint8)
    slope_bins = cp.array([0, 20, 40, float('inf')], dtype=cp.float32)
    slope_scores = cp.array([4, 2, 0], dtype=cp.uint8)
    classifier = GPUSuitabilityClassifier(
        elev_bins=elev_bins, elev_scores=elev_scores,
        slope_bins=slope_bins, slope_scores=slope_scores
    )
    assert cp.all(classifier.elev_bins == elev_bins)
    assert cp.all(classifier.elev_scores == elev_scores)
    assert cp.all(classifier.slope_bins == slope_bins)
    assert cp.all(classifier.slope_scores == slope_scores)

def test_cuda_unavailable(monkeypatch):
    """Test behavior when CUDA is not available."""
    monkeypatch.setattr(cp.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        GPUSuitabilityClassifier()

def test_calculate_slope():
    """Test slope calculation on a simple DEM."""
    classifier = GPUSuitabilityClassifier()
    dem = cp.array([[0, 1], [0, 1]], dtype=cp.float32)  # 45-degree slope
    resolution = 1.0
    slope = classifier.calculate_slope(dem, resolution)
    expected = cp.full((2, 2), 100.0, dtype=cp.float32)  # 45° = 100% slope
    assert cp.allclose(slope, expected, atol=1e-5)

def test_classify_array():
    """Test array classification on GPU."""
    classifier = GPUSuitabilityClassifier()
    array = cp.array([0, 1000, 2000, 4000], dtype=cp.float32)
    bins = cp.array([0, 1500, 2000, 3000, float('inf')], dtype=cp.float32)
    scores = cp.array([5, 3, 1, 0], dtype=cp.uint8)
    result = classifier.classify_array(array, bins, scores)
    expected = cp.array([5, 3, 1, 0], dtype=cp.uint8)
    assert cp.all(result == expected)

def test_process_dem_invalid_path():
    """Test processing with an invalid file path."""
    classifier = GPUSuitabilityClassifier()
    with pytest.raises(FileNotFoundError):
        classifier.process_dem("nonexistent_file.hgt", "output/test")

@pytest.mark.skipif(not os.path.exists(SAMPLE_HGT), reason="Sample HGT file N01E110.hgt not found in tests/data/")
def test_process_dem(tmp_path):
    """Test full DEM processing with the sample HGT file."""
    classifier = GPUSuitabilityClassifier()
    output_prefix = str(tmp_path / "output/test")
    classifier.process_dem(SAMPLE_HGT, output_prefix)
    
    assert os.path.exists(f"{output_prefix}_elev_class.tif")
    assert os.path.exists(f"{output_prefix}_slope_class.tif")
    
    with rasterio.open(f"{output_prefix}_elev_class.tif") as src:
        assert src.count == 1
        assert src.dtypes[0] == 'uint8'
        assert src.nodata == 255

__all__ = [
    'test_classifier_init', 'test_classifier_custom_bins', 'test_cuda_unavailable',
    'test_calculate_slope', 'test_classify_array', 'test_process_dem_invalid_path',
    'test_process_dem'
]