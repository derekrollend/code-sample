from enum import Enum
from pathlib import Path
from typing import Iterable
from enum import Enum, auto
import os

import xrspatial.multispectral as ms
import xarray as xr

from stackstac_downloader import StackStacDownloader


class Season(Enum):
    Fall = auto()
    Winter = auto()
    Spring = auto()
    Summer = auto()

    def __str__(self):
        return self.name.lower()


class SentinelDownloader:
    """
    Class to handle downloading Sentinel-2 mosaics for any/all seasons,
    utilizing the functionality in StackStacDownloader.
    """

    def __init__(
        self,
        max_cloudcover: int = 15,
        years: Iterable[int] = range(2017, 2022),
        seasons: Iterable[Season] = list(Season),
        pool_size: int = os.cpu_count() // 2,
        use_cache: bool = True,
        archive_path: Path = Path("data/s2_archive"),
        stac_catalog_url: str = "https://earth-search.aws.element84.com/v0",
        force_new_download: bool = False,
        verbose: bool = False,
    ):
        self.max_cloudcover = max_cloudcover
        self.pool_size = pool_size
        self.force_new_download = force_new_download
        self.verbose = verbose
        self.archive_path = archive_path
        if not self.archive_path.exists():
            self.archive_path.mkdir(exist_ok=True)
        self.downloader = StackStacDownloader(
            self.archive_path, use_cache=use_cache, stac_catalog_url=stac_catalog_url
        )
        self.years = years
        self.seasons = seasons
        self.season_dates = {
            Season.Spring: ["03-01", "05-31"],
            Season.Summer: ["06-01", "08-31"],
            Season.Fall: ["09-01", "11-30"],
            Season.Winter: ["12-01", "02-28"],
        }

    def _get_rgb_mosaic_for_bounds(
        self,
        bounds: Iterable[float],
        daterange: str,
    ) -> xr.DataArray:
        rgb_mosaic = None
        try:
            mosaic = self.downloader.stack_and_mosaic(
                daterange=daterange,
                bounds=bounds,
                verbose=self.verbose,
                max_cloudcover=self.max_cloudcover,
                force_new_download=self.force_new_download,
            )
            rgb_mosaic = (
                ms.true_color(*mosaic, nodata=0)
                .isel(band=[0, 1, 2])
                .transpose("band", ...)
            )
            rgb_mosaic.attrs = mosaic.attrs.copy()
        except Exception as e:
            print(f"Failed to create mosaic for {bounds} ({daterange}): {e}")

        return rgb_mosaic
