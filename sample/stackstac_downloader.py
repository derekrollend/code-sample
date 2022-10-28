import pystac_client
import stackstac
import numpy as np
import geopandas as gpd
import shapely.geometry


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
        stac_catalog_url="https://earth-search.aws.element84.com/v0",
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
        assets=["B04", "B03", "B02"],
        epsg=4326,
    ):
        # Bounds should be standardized (-180,180), (-90, 90)
        bounds = self.standardize_bounds(bounds)

        if self.api == None:
            self.api = pystac_client.Client.open(self.stac_catalog_url)

        s2_search = self.api.search(
            datetime=daterange,
            bbox=bounds,
            limit=500,
            collections="sentinel-s2-l2a-cogs",
        )

        if s2_search.matched() > 0:
            items = [
                item.to_dict()
                for item in s2_search.items()
                if item.properties["eo:cloud_cover"] < max_cloudcover
            ]

            if len(items) > 0:
                # Sort items by cloud cover
                items_df = gpd.GeoDataFrame(items)
                items_df["geometry"] = items_df["geometry"].apply(
                    shapely.geometry.shape
                )
                items_df["cloud_cover"] = items_df["properties"].apply(
                    lambda x: x["eo:cloud_cover"]
                )
                items_df = items_df.sort_values("cloud_cover")

                # Filter out items that cover the same geographic region but with higher cloud cover values
                bounds_shp = shapely.geometry.box(*bounds)
                mosaic_shp = shapely.geometry.shape(
                    {"type": "Polygon", "coordinates": []}
                )
                optimal_ids = []
                for _, item in items_df.iterrows():
                    overlapping_item_geom = item["geometry"].intersection(bounds_shp)
                    if not mosaic_shp.contains(overlapping_item_geom):
                        optimal_ids.append(item["id"])
                        mosaic_shp = mosaic_shp.union(item["geometry"])

                    if mosaic_shp.contains(bounds_shp):
                        break

                # Filter items according to the optimal set with minimal cloud cover
                items = [item for item in items if item["id"] in optimal_ids]
            else:
                raise RuntimeError(
                    f"Failed to find any images with eo:cloud_cover <= {max_cloudcover}."
                )
        else:
            raise RuntimeError(
                f"Failed to find any images for specified date range and geographic bounds."
            )

        rgb_stack = stackstac.stack(
            items, assets=assets, epsg=epsg, bounds_latlon=bounds, sortby_date=False
        ).where(
            lambda x: x > 0, other=np.nan
        )  # sentinel-2 uses 0 as nodata

        if len(rgb_stack) > 0:
            mosaic = stackstac.mosaic(
                rgb_stack, reverse=False
            )  # mosaic starting at the front of the stack (lowest cloud cover values)
            return mosaic
        else:
            raise RuntimeError("Failed to find any matching STAC items.")
