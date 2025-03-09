import os
import pytest
import cupy as cp
from slopepy.smoothing import FeaturePreservingSmoothing

SAMPLE_HGT = "tests/data/N01E110.hgt"

@pytest.mark.skipif(not os.path.exists(SAMPLE_HGT), reason="Sample HGT file not found")
def test_process_dem(tmp_path):
    smoother = FeaturePreservingSmoothing()
    output = str(tmp_path / "smoothed.tif")
    smoother.process_dem(SAMPLE_HGT, output)
    assert os.path.exists(output)