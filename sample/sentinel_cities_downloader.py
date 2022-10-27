from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
import geopandas as gpd
import rioxarray

from sample.sentinel_downloader import SentinelDownloader, Season


class SentinelCitiesDownloader(SentinelDownloader):
    """
    Sentinel-2 RGB mosaic downloader for cities contained in specified GeoJSON file.
    """

    def __init__(
        self,
        cities_geojson_path: Path = Path("data/city_ids_and_bounds.geojson"),
        s2_mosaics_output_path: Path = Path("data/sentinel2_images"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Load GeoDataFrame of cities
        if not cities_geojson_path.exists():
            raise FileNotFoundError(cities_geojson_path)

        self.cities_gdf = gpd.read_file(cities_geojson_path)

        # Setup output folder for saving images
        s2_mosaics_output_path.mkdir(parents=True, exist_ok=True)
        self.output_folder = s2_mosaics_output_path

    def _download_helper(self, info):
        asset_identifier, asset_geom = info
        download_seasons = {
            k: v for k, v in self.season_dates.items() if k in self.seasons
        }

        for year in tqdm(self.years, desc=f"{asset_identifier} - years"):
            # Store images by year
            curr_output_folder = self.output_folder / str(asset_identifier) / str(year)
            curr_output_folder.mkdir(parents=True, exist_ok=True)

            for season, (start_mm_dd, end_mm_dd) in tqdm(
                download_seasons.items(), desc=f"{asset_identifier} - seasons"
            ):
                output_mosaic_path = curr_output_folder / f"{season}.tif"
                if (
                    output_mosaic_path.exists()
                    and output_mosaic_path.stat().st_size > 0
                ):
                    print(f"Skipping existing mosaic {output_mosaic_path}")
                    continue

                end_year = year if season != Season.Winter else year + 1
                daterange = f"{year}-{start_mm_dd}/{end_year}-{end_mm_dd}"

                try:
                    rgb_mosaic = self._get_rgb_mosaic_for_bounds(
                        asset_geom.bounds, daterange
                    )

                    if rgb_mosaic is not None:
                        rgb_mosaic.rio.to_raster(output_mosaic_path)
                    else:
                        print(f"Failed to retrieve RGB mosaic for {season} - {year}")
                except Exception as e:
                    print(f"Caught exception: {e}")

    def download_all(self, parallel=False, debug=False):
        """
        Download Sentinel-2 Level-2A products from AWS STAC Catalog, using StackStacDownloader
        """

        location_list = list(
            zip(self.cities_gdf["asset_identifier"], self.cities_gdf["geometry"])
        )

        if parallel:
            # TODO: sometimes this freezes when downloading images...
            with Pool(self.pool_size) as p:
                r = list(
                    tqdm(
                        p.imap_unordered(self._download_helper, location_list),
                        total=len(location_list),
                        desc="Processing cities",
                    )
                )
        else:
            for location in tqdm(location_list, desc="Processing cities"):
                self._download_helper(location)


def main():
    # download mosaics using StackStacDownloader
    downloader = SentinelCitiesDownloader(
        years=[2021],
        seasons=[Season.Summer],
        use_cache=False,
        verbose=True,
        force_new_download=False,
    )
    downloader.download_all(parallel=True, debug=False)


if __name__ == "__main__":
    main()
