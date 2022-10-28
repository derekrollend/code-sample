from typing import Iterable
from enum import Enum, auto
import os

import xrspatial.multispectral as ms
import xarray as xr

from sample.stackstac_downloader import StackStacDownloader
from sample.utils import get_simple_logger


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
        max_cloudcover: int = 10,
        years: Iterable[int] = range(2017, 2022),
        seasons: Iterable[Season] = list(Season),
        pool_size: int = os.cpu_count() // 2,
    ):
        self.logger = get_simple_logger(self.__class__.__name__)
        self.max_cloudcover = max_cloudcover
        self.pool_size = pool_size
        self.downloader = StackStacDownloader()
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
        """
        Retrieves an Sentinel-2 RGB mosaic from STAC. The xarray-spatial `true_color` function
        is used to scale the native uint16 reflectance values to the range 0-255 as uint8.
        """
        rgb_mosaic = None
        try:
            mosaic = self.downloader.stack_and_mosaic(
                daterange=daterange,
                bounds=bounds,
                max_cloudcover=self.max_cloudcover,
            )
            rgb_mosaic = (
                ms.true_color(*mosaic, nodata=0)
                .isel(band=[0, 1, 2])
                .transpose("band", ...)
            )
            rgb_mosaic.attrs = mosaic.attrs.copy()
        except Exception as e:
            self.logger.error(
                f"Failed to create mosaic for {bounds} ({daterange}): {e}"
            )

        return rgb_mosaic
