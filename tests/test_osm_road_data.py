from pathlib import Path

import rasterio

from sample.osm_road_data import OSMRoadData


def test_road_image_from_road_image_from_bounding_rasterio_dataset():
    test_city_id = 13  # Hartford, CT - smallest city in the dataset
    cities_osm_folder = Path("data/osm_networks")
    test_city_graphml_file = cities_osm_folder / f"{test_city_id}.graphml.gz"
    assert test_city_graphml_file.exists()
    test_city_root_img_path = Path(f"data/sentinel2_images/{test_city_id}")
    test_city_tif_path = next(test_city_root_img_path.rglob("*.tif"))

    test_city_visual_ds = rasterio.open(test_city_tif_path)
    osm_road_data = OSMRoadData(test_city_graphml_file)

    roads_img = osm_road_data.road_image_from_bounding_rasterio_dataset(
        test_city_visual_ds
    )
    visual_ds_shape = (test_city_visual_ds.count,) + test_city_visual_ds.shape
    assert roads_img.shape == visual_ds_shape
    assert roads_img.max() == 255
    assert roads_img.min() == 0
