# utilities for reading/creating lidar files (no actual calculations)

import os
from datetime import datetime


def createFilename(date: datetime, destdir: str, filetype: str,
                   prefix: str = None) -> str:
    fname = filetype + "_"
    if prefix:
        fname += prefix + "_"
    fname += date.strftime('%Y%m%d') + '.nc'
    return os.path.join(destdir, fname)


def test_createFilename():
    # VAD
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/vad"
    fname = destdir + "/VAD_339_20220329.nc"
    dt = datetime(2022, 3, 29)
    assert createFilename(dt, destdir, "VAD", "339") == fname
    # without prefix
    fname = destdir + "/VAD_20220329.nc"
    assert createFilename(dt, destdir, "VAD", None) == fname
    # destdir with trailing /
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/vad/"
    assert createFilename(dt, destdir, "VAD", None) == fname
    # consensus
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/consensus"
    fname = destdir + "/30min_winds_339_20220329.nc"
    assert createFilename(dt, destdir, "30min_winds", "339") == fname
    # without prefix
    fname = destdir + "/30min_winds_20220329.nc"
    assert createFilename(dt, destdir, "30min_winds", None) == fname
