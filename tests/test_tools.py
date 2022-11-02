from datetime import datetime
from tools import create_filename


def test_create_filename():
    # VAD
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/vad"
    fname = destdir + "/VAD_339_20220329.nc"
    dt = datetime(2022, 3, 29)
    assert create_filename(dt, destdir, "VAD", "339") == fname
    # without prefix
    fname = destdir + "/VAD_20220329.nc"
    assert create_filename(dt, destdir, "VAD", None) == fname
    # destdir with trailing /
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/vad/"
    assert create_filename(dt, destdir, "VAD", None) == fname
    # consensus
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/consensus"
    fname = destdir + "/30min_winds_339_20220329.nc"
    assert create_filename(dt, destdir, "30min_winds", "339") == fname
    # without prefix
    fname = destdir + "/30min_winds_20220329.nc"
    assert create_filename(dt, destdir, "30min_winds", None) == fname