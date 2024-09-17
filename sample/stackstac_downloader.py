import pystac_client
import stackstac
import numpy as np


class StackStacDownloader:
    """
    Class to handle querying a STAC catalog and retrieving mosaic images from
    potential matched STAC items. Mosaics are returned as a xarray.DataArray.

    This has not been tested extensively with STAC catalogs other than the
    Sentinel-2 catalog on AWS, and is heavily geared towards creating low-cloud
    visual RGB mosaics.
    """

    def __init__(
        self,
        stac_catalog_url="https://earth-search.aws.element84.com/v1",
    ):
        self.stac_catalog_url = stac_catalog_url
        self.api = None

    def standardize_bounds(self, bounds):
        # Bounds are [lon1, lat1, lon2, lat2]
        latlon_offsets = np.array([180, 90, 180, 90])
        bounds = np.array(bounds) + latlon_offsets
        bounds = bounds % (2 * latlon_offsets)
        bounds = bounds - latlon_offsets
        return bounds.tolist()

    def stack_and_mosaic(
        self,
        daterange,
        bounds,
        max_cloudcover=10,
        assets=["red", "green", "blue"],
        epsg=4326,
    ):
        # Bounds should be standardized to (-180,180), (-90, 90)
        bounds = self.standardize_bounds(bounds)

        if self.api is None:
            self.api = pystac_client.Client.open(self.stac_catalog_url)

        s2_search = self.api.search(
            datetime=daterange,
            bbox=bounds,
            limit=100,
            collections=["sentinel-2-l2a"],
            query=[f"eo:cloud_cover<={max_cloudcover}"],
            sortby=["+properties.eo:cloud_cover"],
        )

        if s2_search.matched() == 0:
            raise RuntimeError("Failed to find any images for specified date range and geographic bounds.")

        items = s2_search.item_collection()

        rgb_stack = stackstac.stack(items, assets=assets, epsg=epsg, bounds_latlon=bounds, sortby_date=False).where(
            lambda x: x > 0, other=np.nan
        )  # sentinel-2 uses 0 as nodata

        if len(rgb_stack) > 0:
            mosaic = stackstac.mosaic(
                rgb_stack, reverse=False
            )  # mosaic starting at the front of the stack (lowest cloud cover values)
            return mosaic
        else:
            raise RuntimeError("Failed to find any matching STAC items.")
