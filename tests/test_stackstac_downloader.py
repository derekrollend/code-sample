from pathlib import Path

import xarray

from sample.stackstac_downloader import StackStacDownloader


def test_stack_and_mosaic():
    archive_path = Path("data/s2_archive")
    downloader = StackStacDownloader()
    daterange = "2017-07-01/2017-10-30"
    bounds = [
        -148.56536865234375,
        60.80072385643073,
        -147.44338989257812,
        61.18363894915102,
    ]
    mosaic = downloader.stack_and_mosaic(daterange, bounds)
    assert isinstance(mosaic, xarray.core.dataarray.DataArray)
    assert mosaic.shape[0] == 3
    assert mosaic.dims == ("band", "y", "x")
