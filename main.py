from pathlib import Path

import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np

from sample.sentinel_downloader import Season
from sample.sentinel_cities_downloader import SentinelCitiesDownloader
from sample.osm_road_generator import OSMRoadGenerator
from sample.utils import get_simple_logger


def main():
    logger = get_simple_logger("main")

    # Load city IDs, names, and geographic bounds
    cities_geojson = Path("data/city_ids_and_bounds.geojson")
    if not cities_geojson.exists():
        raise FileNotFoundError(cities_geojson)

    logger.info(f"Loading city data from {cities_geojson}...")
    cities_gdf = gpd.read_file(cities_geojson)

    logger.info("Downloading Sentinel-2 mosaic for each city...")
    # Download Sentinel-2 mosaic for each city for a few seasons in 2021
    s2_cities_downloader = SentinelCitiesDownloader(
        cities_geojson_path=cities_geojson,
        years=[2021],
        seasons=[Season.Spring, Season.Summer, Season.Fall],
    )
    s2_cities_downloader.download_all(parallel=True, debug=False)

    # Generate OpenStreetMap road network rasters for each city, save each as separate GeoTIFF
    logger.info("Generating OpenStreetMap road network rasters for each city...")
    osm_road_generator = OSMRoadGenerator(cities_geojson_path=cities_geojson)
    osm_road_generator.generate_roads_parallel()

    # Visualize RGB and roads side-by-side, save as .png
    logger.info("Visualizing RGB and roads side-by-side for each city...")
    plot_output_path = Path("plots")
    plot_output_path.mkdir(exist_ok=True)

    for city_tuple in cities_gdf.itertuples():
        logger.info(f"Creating plot for {city_tuple.asset_name}...")

        city_tif_paths = sorted(Path(f"data/sentinel2_images/{city_tuple.asset_identifier}").rglob("*.tif"))
        if len(city_tif_paths) == 0:
            logger.warning(f"Failed to find visual images for {city_tuple.asset_name}")
        else:
            with rasterio.open(city_tif_paths[0]) as ds:
                visual_img = np.moveaxis(ds.read(), 0, -1)
            with rasterio.open(f"data/osm_images/{city_tuple.asset_identifier}.tif") as ds:
                road_img = np.moveaxis(ds.read(), 0, -1)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            axes[0].imshow(visual_img)
            axes[0].get_yaxis().set_visible(False)
            axes[0].get_xaxis().set_visible(False)
            axes[1].imshow(road_img)
            axes[1].get_yaxis().set_visible(False)
            axes[1].get_xaxis().set_visible(False)
            plt.suptitle(city_tuple.asset_name)
            plt.tight_layout()

            city_plot_output_path = plot_output_path / f"{city_tuple.asset_name}.png"
            fig.savefig(city_plot_output_path)
            logger.info(f"Saved plot to {city_plot_output_path}")


if __name__ == "__main__":
    main()
