from pathlib import Path
import pystac_client
import stackstac
import numpy as np
import dask
import fastdl
from tqdm.auto import tqdm
import geopandas as gpd
import shelve
import shapely.geometry


class StackStacDownloader:
    """
    Set use_cache=False to skip local caching on the shared drive. This can speed up RGB
    mosaicing operations significantly and should be used when repeated queries with the same
    spatiotemporal bounds are not used, and when saving a local copy is not important.
    """
    def __init__(
        self,
        archive_path,
        use_cache=True,
        stac_catalog_url="https://earth-search.aws.element84.com/v0",
    ):
        self.archive_path = Path(archive_path)
        self.query_shelve = shelve.open(str(self.archive_path / "stac_queries.shelve"))
        self.stac_catalog_url = stac_catalog_url
        self.use_cache = use_cache
        self.api = None

    def download_if_not_exists(self, href, verbose=False):
        item_relative_path = Path(*href.split("/")[3:])
        abs_local_path = self.archive_path / item_relative_path
        if not abs_local_path.exists():
            # Download the image
            fastdl.download(
                href,
                dir_prefix=abs_local_path.parent,
                progressbar=verbose,
                force_download=True,
            )
        return str(item_relative_path)

    def standardize_bounds(self, bounds):
        # Bounds are [lon, lat, lon, lat]
        latlon_offsets = np.array([180, 90, 180, 90])
        bounds = np.array(bounds) + latlon_offsets
        bounds = bounds % (2 * latlon_offsets)
        bounds = bounds - latlon_offsets
        return bounds.tolist()

    def update_item_hrefs_to_abs(self, items, assets):
        for item in items:
            for asset in assets:
                item["assets"][asset]["href"] = str(
                    self.archive_path / item["assets"][asset]["href"]
                )
        return items

    def stack_and_mosaic(
        self,
        daterange,
        bounds,
        max_cloudcover=15,
        assets=["B04", "B03", "B02"],
        epsg=4326,
        optimize=True,
        verbose=False,
        force_new_download=False,
    ):
        # Bounds should be standardized (-180,180), (-90, 90)
        bounds = self.standardize_bounds(bounds)

        query_string = f"daterange:{daterange}\nbounds:{bounds}\nmax_cloudcover:{max_cloudcover}\nassets:{assets}\nepsg:{epsg}"

        if (
            self.use_cache
            and query_string in self.query_shelve
            and not force_new_download
        ):
            items = self.query_shelve[query_string]
            items = self.update_item_hrefs_to_abs(items, assets)
        else:
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
                    for item in s2_search.get_items()
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
                        overlapping_item_geom = item["geometry"].intersection(
                            bounds_shp
                        )
                        if not mosaic_shp.contains(overlapping_item_geom):
                            optimal_ids.append(item["id"])
                            mosaic_shp = mosaic_shp.union(item["geometry"])

                        if mosaic_shp.contains(bounds_shp):
                            break

                    # Filter items according to the optimal set with minimal cloud cover
                    items = [item for item in items if item["id"] in optimal_ids]

                    if self.use_cache:
                        for item in tqdm(
                            items, desc="Downloading images", disable=not verbose
                        ):
                            for asset in assets:
                                item["assets"][asset][
                                    "href"
                                ] = self.download_if_not_exists(
                                    item["assets"][asset]["href"], verbose=verbose
                                )

                        # Sort items by cloud_cover
                        items = sorted(
                            items, key=lambda x: x["properties"]["eo:cloud_cover"]
                        )
                        self.query_shelve[query_string] = items
                        items = self.update_item_hrefs_to_abs(items, assets)
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

            if optimize:
                # Optimize the computational graph so that loading subsets is fast
                if verbose:
                    print(
                        "Optimizing mosaic computational graph, this make take some time...",
                        end=" ",
                        flush=True,
                    )
                mosaic = dask.optimize(mosaic)[0]
                if verbose:
                    print("Done")

            return mosaic
        else:
            raise RuntimeError("Failed to find any matching STAC items.")


def main():
    archive_path = Path("data/s2_archive")
    downloader = StackStacDownloader(archive_path)
    daterange = "2017-07-01/2017-10-30"
    bounds = [
        -148.56536865234375,
        60.80072385643073,
        -147.44338989257812,
        61.18363894915102,
    ]
    mosaic = downloader.stack_and_mosaic(daterange, bounds, force_new_download=False)
    print(mosaic)
    # Call mosaic.compute() to load the data


if __name__ == "__main__":
    main()
