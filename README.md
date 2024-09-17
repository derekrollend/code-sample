# code-sample
Derek Rollend

## Overview
This is a small subset of code from a previous project, adapted to be a standalone Python package. Given a GeoJSON containing a set of cities and their geometries, it provides the ability to:
  1. Retrieve a Sentinel-2 RGB mosaic using STAC and [`stackstac`](https://stackstac.readthedocs.io/en/latest/)
  2. Create a corresponding road network data raster image, utilizing data from [OpenStreetMap](https://www.openstreetmap.org/) (OSM).
  3. Visualize the city mosaic and road network image side-by-side.

A cities GeoJSON is provided in `data/city_ids_and_bounds.geojson`, and currently contains three U.S. cities: Hartford CT, Pittsburgh PA, and Orlando FL.

OpenStreetMap roads are categorized into three types of roads: primary (e.g., highways), secondary (e.g., arterial or collector roads), and local (e.g., residential roads). Each road type is rasterized in a separate image, and the resulting images are channel concatenated into a single, three-channel image. Thus, in the resulting plots, primary roads are represented as red, secondary as green, and local as blue.

## Environment Setup
1. Install Poetry if you haven't already: https://python-poetry.org/docs/#installation
2. Run `poetry install` to create a virtual environment and install dependencies
3. Use `poetry run` to invoke commands for the virtual environment

## Running
The entry point is `main.py`, which will load city data from `data/city_ids_and_bounds.geojson`. Sentinel-2 RGB mosaics and OSM road rasters will be saved in `data/sentinel2_images` and `data/osm_images`, respectively. Plots of RGB mosaics and corresponding roads will be saved in the `plots` directory as .png files.

## Notes
* Sentinel-2 mosaic creation can take a couple minutes, depending on internet speeds and host machine compute power and available memory.
* The code relies on previously-extracted OSM road network data, saved in `data/osm_networks` as `.graphml.gz` files. These files have been previously extracted from a larger OSM binary file using [Osmium](https://osmcode.org/osmium-tool/), but could also be retrieved using the [OSMnx](https://osmnx.readthedocs.io/en/stable/) Python package.
* This combination of Sentinel-2 RGB imagery and OSM road network data have previously been used to train machine learning models to predict road transportation-related variables (e.g., emissions, vehicle traffic).