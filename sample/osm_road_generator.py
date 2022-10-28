from multiprocessing import Pool
from pathlib import Path
import os

import rasterio
from tqdm import tqdm
import geopandas as gpd

from sample.osm_road_data import OSMRoadData
from sample.utils import get_simple_logger


class OSMRoadGenerator:
    """
    Class to generate a set of OpenStreetMap raster images, given a set of cities and
    previously-generated Sentinel-2 images. Generation can be run in either serial or
    parallel.
    """

    def __init__(
        self,
        cities_geojson_path: Path = Path("data/city_ids_and_bounds.geojson"),
        root_s2_img_path: Path = Path("data/sentinel2_images"),
        root_osm_output_path: Path = Path("data/osm_images"),
    ):
        self.logger = get_simple_logger(self.__class__.__name__)

        self.root_s2_img_path = root_s2_img_path
        self.root_osm_output_path = root_osm_output_path

        if not self.root_osm_output_path.exists():
            self.root_osm_output_path.mkdir(exist_ok=True)

        # initialize cities GeoDataFrame
        self.cities_gdf = gpd.read_file(cities_geojson_path)

        # initialize OSM graphml artifact
        self.cities_osm_folder = Path("data/osm_networks")

    def _get_city_osm_graphml(self, city_id: int) -> Path:
        city_graphml_file = self.cities_osm_folder / f"{city_id}.graphml.gz"
        if not city_graphml_file.exists():
            self.logger.error(f"Failed to find OSM graphml file: {city_graphml_file}")
            raise FileNotFoundError(city_graphml_file)
        return city_graphml_file

    def _get_city_visual_tif_path(self, city_id: int) -> Path:
        city_tif_filename = None
        city_root_img_path = self.root_s2_img_path / str(city_id)
        city_img_paths = sorted(city_root_img_path.rglob("*.tif"))

        if len(city_img_paths) == 0:
            self.logger.error(f"Failed to find any RGB images for city {city_id}")
        else:
            # all RGB tifs should have the same bounds, regardless of year/season, so just use the first one
            city_tif_filename = city_img_paths[0]

        return city_tif_filename

    def _generate_roads_helper(self, city_id: int):
        city_graphml_file = self._get_city_osm_graphml(city_id)
        osm_road_data = OSMRoadData(city_graphml_file)

        tif_filename = self._get_city_visual_tif_path(city_id)

        if tif_filename is None:
            return city_id

        try:
            visual_ds = rasterio.open(tif_filename)
        except rasterio.errors.RasterioIOError:
            return city_id

        roads_img = osm_road_data.road_image_from_bounding_rasterio_dataset(visual_ds)

        # output
        kwargs = visual_ds.meta.copy()
        kwargs["dtype"] = rasterio.uint8
        kwargs["compress"] = "lzw"
        kwargs["count"] = roads_img.shape[0]  # (c, h, w)

        roads_output_filename = self.root_osm_output_path / f"{city_id}.tif"

        with rasterio.open(roads_output_filename, "w", **kwargs) as dst:
            dst.write(roads_img)

    def generate_roads_parallel(self):
        with Pool(os.cpu_count() // 2) as p:
            r = list(
                tqdm(
                    p.imap_unordered(
                        self._generate_roads_helper,
                        self.cities_gdf["asset_identifier"],
                    ),
                    total=len(self.cities_gdf),
                    desc="Processing images",
                )
            )
        self.logger.info(f"Images with errors: {[p for p in r if p is not None]}")

    def generate_roads(self):
        r = []
        for city_path in tqdm(
            self.cities_gdf["asset_identifier"], desc="Processing images"
        ):
            r.append(self._generate_roads_helper(city_path))
        self.logger.info(f"Images with errors: {[p for p in r if p is not None]}")
