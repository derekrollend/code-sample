import click
from pathlib import Path

import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np

from sample.sentinel_downloader import Season
from sample.sentinel_cities_downloader import SentinelCitiesDownloader
from sample.osm_road_generator import OSMRoadGenerator
from sample.utils import get_simple_logger


def load_city_data(cities_geojson_path):
    # ... existing code ...
    return gpd.read_file(cities_geojson_path)


def download_sentinel_data(cities_geojson_path, years, seasons):
    # ... existing code ...
    s2_cities_downloader = SentinelCitiesDownloader(
        cities_geojson_path=cities_geojson_path,
        years=years,
        seasons=seasons,
    )
    s2_cities_downloader.download_all(parallel=True, debug=False)


def generate_osm_road_data(cities_geojson_path):
    # ... existing code ...
    osm_road_generator = OSMRoadGenerator(cities_geojson_path=cities_geojson_path)
    osm_road_generator.generate_roads_parallel()


def load_image(file_path):
    with rasterio.open(file_path) as ds:
        return np.moveaxis(ds.read(), 0, -1)


def create_plot(visual_img, road_img, city_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(visual_img)
    axes[0].get_yaxis().set_visible(False)
    axes[0].get_xaxis().set_visible(False)
    axes[1].imshow(road_img)
    axes[1].get_yaxis().set_visible(False)
    axes[1].get_xaxis().set_visible(False)
    plt.suptitle(city_name)
    plt.tight_layout()
    return fig


def save_plot(fig, output_path):
    fig.savefig(output_path)


def visualize_cities(cities_gdf, plot_output_path):
    plot_output_path.mkdir(exist_ok=True)
    logger = get_simple_logger("visualize_cities")

    for city_tuple in cities_gdf.itertuples():
        logger.info(f"Creating plot for {city_tuple.asset_name}...")

        city_tif_paths = sorted(Path(f"data/sentinel2_images/{city_tuple.asset_identifier}").rglob("*.tif"))
        if len(city_tif_paths) == 0:
            logger.warning(f"Failed to find visual images for {city_tuple.asset_name}")
            continue

        visual_img = load_image(city_tif_paths[0])
        road_img = load_image(f"data/osm_images/{city_tuple.asset_identifier}.tif")

        fig = create_plot(visual_img, road_img, city_tuple.asset_name)
        city_plot_output_path = plot_output_path / f"{city_tuple.asset_name}.png"
        save_plot(fig, city_plot_output_path)
        logger.info(f"Saved plot to {city_plot_output_path}")


@click.command()
@click.option(
    "--cities-geojson",
    type=click.Path(exists=True),
    default="data/city_ids_and_bounds.geojson",
    help="Path to the cities GeoJSON file",
)
@click.option("--plot-output", type=click.Path(), default="plots", help="Path to save the output plots")
@click.option("--years", type=int, multiple=True, default=[2021], help="Years to download Sentinel-2 data for")
@click.option(
    "--seasons",
    type=click.Choice(["Spring", "Summer", "Fall", "Winter"]),
    multiple=True,
    default=["Spring", "Summer", "Fall"],
    help="Seasons to download Sentinel-2 data for",
)
def main(cities_geojson, plot_output, years, seasons):
    logger = get_simple_logger("main")

    cities_geojson_path = Path(cities_geojson)
    plot_output_path = Path(plot_output)

    logger.info(f"Loading city data from {cities_geojson_path}...")
    cities_gdf = load_city_data(cities_geojson_path)

    logger.info("Downloading Sentinel-2 mosaic for each city...")
    seasons_enum = [Season[s] for s in seasons]
    download_sentinel_data(cities_geojson_path, years=years, seasons=seasons_enum)

    logger.info("Generating OpenStreetMap road network rasters for each city...")
    generate_osm_road_data(cities_geojson_path)

    logger.info("Visualizing RGB and roads side-by-side for each city...")
    visualize_cities(cities_gdf, plot_output_path)


if __name__ == "__main__":
    main()
