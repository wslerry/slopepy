import pytest
import os
from slopepy.contour import GPUContourGenerator
import geopandas as gpd

SAMPLE_HGT = os.path.join(os.path.dirname(__file__), "data", "N01E110.hgt")

@pytest.mark.skipif(not os.path.exists(SAMPLE_HGT), reason="Sample HGT file not found")
def test_contour_generation(tmp_path):
    # Test with single interval
    contour_gen = GPUContourGenerator(contour_levels=[20], dst_crs="EPSG:32649")  # Zone 49N
    contour_gen.process_dem(SAMPLE_HGT, str(tmp_path))

    gpkg_path = str(tmp_path / "N01E110_contour.gpkg")
    assert os.path.exists(gpkg_path)
    gdf = gpd.read_file(gpkg_path, layer="elevation_50")
    assert len(gdf) > 0
    assert "elevation" in gdf.columns
    assert gdf.crs.to_string() == "EPSG:32649"
    assert all(gdf["elevation"] % 50 == 0)

    gpq_path = str(tmp_path / "N01E110_contour_50.gpq")
    assert os.path.exists(gpq_path)
    gdf_gpq = gpd.read_parquet(gpq_path)
    assert len(gdf_gpq) > 0

    # Test with multiple intervals, let UTM auto-detect
    contour_gen = GPUContourGenerator(contour_levels=[10, 20], output_dir=str(tmp_path))
    contour_gen.process_dem(SAMPLE_HGT, str(tmp_path))

    for interval in [10, 20]:
        layer_name = f"elevation_{interval}"
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        assert len(gdf) > 0
        assert "elevation" in gdf.columns
        assert all(gdf["elevation"] % interval == 0)

        gpq_path = str(tmp_path / f"N01E110_contour_{interval}.gpq")
        assert os.path.exists(gpq_path)
        gdf_gpq = gpd.read_parquet(gpq_path)
        assert len(gdf_gpq) > 0