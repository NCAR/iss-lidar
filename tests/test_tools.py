from datetime import datetime
import numpy as np
from pathlib import Path

from iss_lidar.tools import create_filename, wspd_wdir_from_uv, read_cfradial

datadir = Path(__file__).parent.parent.joinpath("testdata")

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


def test_wind_from_uv():
    u = np.array([-1, -0.0, 0, 1, 0, -2, np.nan], dtype=float)
    v = np.array([-np.inf, -0.0, 0, 0, -1, -2, 2], dtype=float)
    xspd = np.array([np.inf, 0, 0, 1, 1, np.sqrt(8), np.nan], dtype=float)
    xdir = np.array([0, 90, 270, 270, 0, 45, np.nan], dtype=float)
    wspd, wdir = wspd_wdir_from_uv(u, v)
    np.testing.assert_allclose(wspd, xspd, equal_nan=True)
    np.testing.assert_allclose(wdir, xdir, equal_nan=True)


def test_cfradial1_vs_cfradial2():
    # test that cfradial1 and cfradial2 files return the same data for the same scan
    file_1 = datadir.joinpath("cfrad.20260915_014052_WLS200s-181_185_PPI_50m.nc")
    file_2 = datadir.joinpath("cfrad2.20260915_014052_WLS200s-181_ppi_PPI.nc")
    data1 = read_cfradial(str(file_1))
    data2 = read_cfradial(str(file_2))
    for d1, d2 in zip(data1, data2):
        if isinstance(d1, np.ndarray):
            np.testing.assert_allclose(d1, d2, equal_nan=True)
        else:
            assert d1 == d2