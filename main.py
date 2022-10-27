from pathlib import Path

import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from sample.sentinel_downloader import Season
from sample.sentinel_cities_downloader import SentinelCitiesDownloader
from sample.osm_road_generator import OSMRoadGenerator


def main():
    cities_geojson = Path("data/small_city_ids_and_bounds.geojson")
    if not cities_geojson.exists():
        raise FileNotFoundError(cities_geojson)

    cities_gdf = gpd.read_file(cities_geojson)

    # Download S2 mosaic for each city
    s2_cities_downloader = SentinelCitiesDownloader(
        cities_geojson_path=cities_geojson,
        years=[2021],
        seasons=[Season.Spring, Season.Summer, Season.Fall],
        use_cache=False,
        verbose=True,
        force_new_download=True,
    )
    s2_cities_downloader.download_all(parallel=False, debug=False)

    # Generate OpenStreetMap rasters for each city, save each as separate GeoTIFF
    osm_road_generator = OSMRoadGenerator(cities_geojson_path=cities_geojson)
    osm_road_generator.generate_roads_parallel()

    # visualize
    plot_city_id = 10
    city_name = cities_gdf[cities_gdf["asset_identifier"] == plot_city_id][
        "asset_name"
    ].iloc[0]

    with rasterio.open(f"data/sentinel2_images/{plot_city_id}/2021/summer.tif") as ds:
        visual_img = np.moveaxis(ds.read(), 0, -1)
    with rasterio.open(f"data/osm_images/{plot_city_id}.tif") as ds:
        road_img = np.moveaxis(ds.read(), 0, -1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(visual_img)
    axes[0].get_yaxis().set_visible(False)
    axes[0].get_xaxis().set_visible(False)
    axes[1].imshow(road_img)
    axes[1].get_yaxis().set_visible(False)
    axes[1].get_xaxis().set_visible(False)
    plt.suptitle(city_name)
    plt.tight_layout()
    fig.savefig(f"{city_name}.png")


if __name__ == "__main__":
    main()
