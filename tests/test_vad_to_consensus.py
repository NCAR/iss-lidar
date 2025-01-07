from datetime import datetime
from iss_lidar.vad_to_consensus import create_cns_filename


def test_create_cns_filename():
    # with prefix
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/consensus_30"
    fname = destdir + "/30min_winds_339_20220329.nc"
    dt = datetime(2022, 3, 29)
    assert create_cns_filename(30, dt, destdir, "339") == fname
    # without prefix
    fname = destdir + "/30min_winds_20220329.nc"
    assert create_cns_filename(30, dt, destdir) == fname
    # different timespan
    fname = destdir + "/60min_winds_20220329.nc"
    assert create_cns_filename(60, dt, destdir) == fname

