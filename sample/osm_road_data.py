from pathlib import Path
from typing import Dict, Iterable

import rasterio.features
import rasterio.transform
import numpy as np
import osmnx as ox
import rasterio
import rasterio.vrt
import numpy as np
from shapely.geometry import box

from sample.utils import get_simple_logger


class OSMRoadData:
    """
    Class to rasterize OSM road data that has already been saved in graphml format.

    Note that this class assumes alignment between the data in the provided OSM graphml
    and the imagery for which OSM roads will be rasterized.
    """

    def __init__(self, osm_graphml_path: Path):
        self.simple_type_to_osm_highway_type_mapping = {
            "primary": ["motorway", "motorway_link", "trunk", "trunk_link"],
            "secondary": ["primary", "primary_link", "secondary", "secondary_link"],
            "local": [
                "tertiary",
                "tertiary_link",
                "unclassified",
                "residential",
                "living_street",
            ],
        }

        self.logger = get_simple_logger(self.__class__.__name__)
        self.osm_graphml_path = osm_graphml_path
        assert self.osm_graphml_path.exists()
        self.logger.info(f"Loading OSM graph from {self.osm_graphml_path}...")
        self.osm_nx_graph = ox.load_graphml(self.osm_graphml_path)
        self.logger.info("Done.")

        self.osm_graph_nodes_gdf, self.osm_graph_edges_gdf = ox.graph_to_gdfs(
            self.osm_nx_graph
        )

    def get_roads_in_bbox(self, bbox: box) -> Dict:
        gdf_edges_within_bbox = self.osm_graph_edges_gdf.intersection(bbox, align=False)

        primary_geoseries = gdf_edges_within_bbox[
            self.osm_graph_edges_gdf["highway"].isin(
                self.simple_type_to_osm_highway_type_mapping["primary"]
            )
        ]
        primary_geoseries = primary_geoseries[~primary_geoseries.is_empty]
        primary_gdf = self.osm_graph_edges_gdf.loc[primary_geoseries.index][
            ["geometry"]
        ]

        secondary_geoseries = gdf_edges_within_bbox[
            self.osm_graph_edges_gdf["highway"].isin(
                self.simple_type_to_osm_highway_type_mapping["secondary"]
            )
        ]
        secondary_geoseries = secondary_geoseries[~secondary_geoseries.is_empty]
        secondary_gdf = self.osm_graph_edges_gdf.loc[secondary_geoseries.index][
            ["geometry"]
        ]

        local_geoseries = gdf_edges_within_bbox[
            self.osm_graph_edges_gdf["highway"].isin(
                self.simple_type_to_osm_highway_type_mapping["local"]
            )
        ]
        local_geoseries = local_geoseries[~local_geoseries.is_empty]
        local_gdf = self.osm_graph_edges_gdf.loc[local_geoseries.index][["geometry"]]

        return {
            "primary": primary_gdf,
            "secondary": secondary_gdf,
            "local": local_gdf,
        }

    @staticmethod
    def _rasterize_shapes(
        shapes: Iterable,
        raster_ds: rasterio.DatasetReader,
        all_touched=False,
        dtype=rasterio.uint8,
    ) -> np.ndarray:
        """
        NOTE: shapes (iterable of (geometry, value) pairs or geometries)
        See https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.rasterize
        """
        out_shape = (raster_ds.height, raster_ds.width)
        out_img = np.zeros(out_shape, dtype=dtype)
        if len(shapes) > 0:
            out_img = rasterio.features.rasterize(
                shapes,
                out_shape=out_shape,
                fill=0,
                all_touched=all_touched,
                dtype=dtype,
                transform=raster_ds.transform,
            )
        return out_img

    def road_image_from_bounding_rasterio_dataset(
        self, input_ds: rasterio.DatasetReader
    ) -> np.ndarray:
        warped_input_ds = rasterio.vrt.WarpedVRT(
            input_ds, crs=self.osm_graph_edges_gdf.crs
        )
        warped_input_ds_bbox = box(*warped_input_ds.bounds)

        # Find intersecting OSM roads within the bounds of the provided dataset
        intersecting_road_dict = self.get_roads_in_bbox(warped_input_ds_bbox)

        primary_geoseries = intersecting_road_dict["primary"]["geometry"]
        secondary_geoseries = intersecting_road_dict["secondary"]["geometry"]
        local_geoseries = intersecting_road_dict["local"]["geometry"]

        # Rasterize the roads
        primary_road_img = self._rasterize_shapes(primary_geoseries, warped_input_ds)
        secondary_road_img = self._rasterize_shapes(
            secondary_geoseries, warped_input_ds
        )
        # NOTE: all_touched=True for local roads, as they tend to be more disconnected.
        # Burning in all touching pixels helps make them more continuous in the rasterized image.
        local_road_img = self._rasterize_shapes(
            local_geoseries, warped_input_ds, all_touched=True
        )

        warped_roads_img = np.stack(
            (primary_road_img, secondary_road_img, local_road_img)
        )

        warped_roads_img *= 255
        warped_roads_img = warped_roads_img.astype("uint8")
        assert warped_roads_img.shape == (
            3,
            warped_input_ds.height,
            warped_input_ds.width,
        )

        # reproject to the src datasets crs
        output_roads_img = np.zeros(
            (3, input_ds.height, input_ds.width), dtype=warped_roads_img.dtype
        )
        output_roads_img, _dst_transform = rasterio.warp.reproject(
            warped_roads_img,
            output_roads_img,
            src_transform=warped_input_ds.transform,
            src_crs=warped_input_ds.crs,
            dst_crs=input_ds.crs,
            dst_transform=input_ds.transform,
        )
        assert _dst_transform == input_ds.transform
        assert output_roads_img.dtype == np.uint8

        return output_roads_img


def main():
    test_cities = {13: "hartford", 3: "houston"}
    test_city_id = 3
    cities_osm_folder = Path("data/osm_networks")
    test_city_graphml_file = cities_osm_folder / f"{test_city_id}.graphml.gz"
    assert test_city_graphml_file.exists()
    test_city_root_img_path = Path(f"data/sentinel2_images/{test_city_id}")
    test_city_tif_path = (
        test_city_root_img_path / f"{test_cities[test_city_id]}_visual.tif"
    )

    test_city_visual_ds = rasterio.open(test_city_tif_path)

    osm_road_data = OSMRoadData(test_city_graphml_file)

    roads_img = osm_road_data.road_image_from_bounding_rasterio_dataset(
        test_city_visual_ds
    )
    road_id_img = osm_road_data.road_id_map_from_bounding_rasterio_dataset(
        test_city_visual_ds
    )
    assert roads_img.shape == road_id_img.shape


if __name__ == "__main__":
    main()
