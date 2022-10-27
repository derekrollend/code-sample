# code-sample

## Overview
This is a small subset of code from a previous project, adapted to be a standalone Python package. Given a GeoJSON containing a set of cities and their geometries, it provides the ability to:
  1. Retrieve a Sentinel-2 RGB mosaic using STAC and [`stackstac`](https://stackstac.readthedocs.io/en/latest/)
  2. Create a corresponding road network data raster image, utilizing data from [OpenStreetMap](https://www.openstreetmap.org/) (OSM).
  3. Visualize the city mosaic and road network image side-by-side.

## Running
The entry point is `main.py`, which will load city data from `data/city_ids_and_bounds.geojson`. Saved Sentinel-2 RGB mosaics and OSM road rasters will be saved in `data/sentinel2_images` and `data/osm_images`, respectively. 

## Caveats
The code relies on previously-extracted OSM road network data, saved in `data/osm_networks` as `.graphml.gz` files. These files have been previously extracted from a larger OSM binary file, but could also be retrieved using the [OSMnx](https://osmnx.readthedocs.io/en/stable/) Python package.
