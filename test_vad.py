import pytest
import pickle
import numpy.ma as ma
import numpy as np
import numpy.testing
from pathlib import Path

from ppi import PPI
import vad
from vad import xyz, non_nan_idxs, calc_A, calc_b
from vad import VAD, VADSet

datadir = Path(__file__).parent.joinpath("testdata")


def assert_allclose(actual, desired, rtol=1e-06, atol=1e-05,
                    equal_nan=True, err_msg='', verbose=True) -> None:
    "Wrap numpy.testing.assert_allclose() with different default tolerances."
    numpy.testing.assert_allclose(actual, desired, rtol, atol,
                                  equal_nan, err_msg, verbose)


@pytest.fixture
def ppi():
    """ Read in PPI file """
    ppi = PPI.fromFile(f"{datadir}/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    ppi.threshold_cnr(-22) # use default threshold val in ppi_scans_to_vad of -22
    return ppi

@pytest.fixture
def ppis():
    """ Read in 3 PPI scans """
    a = PPI.fromFile(f"{datadir}/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    a.threshold_cnr(-22)
    b = PPI.fromFile(f"{datadir}/cfrad.20210630_171644_WLS200s-181_133_PPI_50m.nc")
    b.threshold_cnr(-22)
    c = PPI.fromFile(f"{datadir}/cfrad.20210630_174238_WLS200s-181_133_PPI_50m.nc")
    c.threshold_cnr(-22)
    return(a,b,c)


@pytest.fixture
def locations():
    x = pickle.load(open(f"{datadir}/vad/x.p", "rb"))
    y = pickle.load(open(f"{datadir}/vad/y.p", "rb"))
    z = pickle.load(open(f"{datadir}/vad/z.p", "rb"))
    return(x,y,z)

@pytest.fixture
def final_vad_winds():
    u = pickle.load(open(f"{datadir}/vad/u.p", "rb"))
    v = pickle.load(open(f"{datadir}/vad/v.p", "rb"))
    w = pickle.load(open(f"{datadir}/vad/w.p", "rb"))
    return (u, v, w)

@pytest.fixture
def final_vad_errs():
    du = pickle.load(open(f"{datadir}/vad/du.p", "rb"))
    dv = pickle.load(open(f"{datadir}/vad/dv.p", "rb"))
    dw = pickle.load(open(f"{datadir}/vad/dw.p", "rb"))
    return (du, dv, dw)
    
@pytest.fixture
def derived_products():
    speed = pickle.load(open(f"{datadir}/vad/speed.p", "rb"))
    wdir = pickle.load(open(f"{datadir}/vad/wdir.p", "rb"))
    res = pickle.load(open(f"{datadir}/vad/residual.p", "rb"))
    cor = pickle.load(open(f"{datadir}/vad/correlation.p", "rb"))
    return (speed, wdir, res, cor)

def test_xyz(ppi, locations):
    saved_x, saved_y, saved_z = locations
    x, y, z = xyz(ppi.ranges, ppi.elevation, ppi.azimuth)
    assert_allclose(x, saved_x)
    assert_allclose(y, saved_y)
    assert_allclose(z, saved_z)

def test_non_nan_idxs(ppi):
    idxs = non_nan_idxs(ppi.vr, 0)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo0.p", "rb"))
    assert_allclose(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 1)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo1.p", "rb"))
    assert_allclose(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 2)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo2.p", "rb"))
    assert_allclose(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 3)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo3.p", "rb"))
    assert_allclose(idxs, saved_idxs)

def test_calc_A(ppi):
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 0))
    saved_A = pickle.load(open(f"{datadir}/vad/A_0.p", "rb"))
    assert_allclose(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 1))
    saved_A = pickle.load(open(f"{datadir}/vad/A_1.p", "rb"))
    assert_allclose(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 2))
    saved_A = pickle.load(open(f"{datadir}/vad/A_2.p", "rb"))
    assert_allclose(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 3))
    saved_A = pickle.load(open(f"{datadir}/vad/A_3.p", "rb"))
    assert_allclose(A, saved_A)

def test_calc_b(ppi):
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 0), 0)
    saved_b = pickle.load(open(f"{datadir}/vad/b_0.p", "rb"))
    assert_allclose(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 1), 1)
    saved_b = pickle.load(open(f"{datadir}/vad/b_1.p", "rb"))
    assert_allclose(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 2), 2)
    saved_b = pickle.load(open(f"{datadir}/vad/b_2.p", "rb"))
    assert_allclose(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 3), 3)
    saved_b = pickle.load(open(f"{datadir}/vad/b_3.p", "rb"))
    assert_allclose(b, saved_b)

def test_arm_vad(ppi, final_vad_winds, final_vad_errs, derived_products):
    vad = VAD.calculate_ARM_VAD(ppi)
    saved_u, saved_v, saved_w = final_vad_winds
    assert_allclose(vad.u, saved_u, equal_nan=True)
    assert_allclose(vad.v, saved_v, equal_nan=True)
    assert_allclose(vad.w, saved_w, equal_nan=True)
    saved_du, saved_dv, saved_dw = final_vad_errs
    assert_allclose(vad.du, saved_du, equal_nan=True)
    assert_allclose(vad.dv, saved_dv, equal_nan=True)
    assert_allclose(vad.dw, saved_dw, equal_nan=True)
    saved_wspd, saved_wdir, saved_res, saved_cor = derived_products
    assert_allclose(vad.speed, saved_wspd, equal_nan=True)
    assert_allclose(vad.wdir, saved_wdir, equal_nan=True)
    assert_allclose(vad.residual, saved_res, equal_nan=True)
    assert_allclose(vad.correlation, saved_cor, equal_nan=True)

def test_vadset_netcdf(ppis):
    """ Compare data and data types between VADSet from VADs and VADSet from netcdf """
    vads = []
    for p in ppis:
        vads.append(VAD.calculate_ARM_VAD(p))
    vs = VADSet.from_VADs(vads, -22)
    vs.to_ARM_netcdf(f"{datadir}/test_vadset.nc")
    f = VADSet.from_file(f"{datadir}/test_vadset.nc")
    # vadset from file should match original vadset
    assert vs.min_cnr == f.min_cnr
    assert type(vs.min_cnr) == type(f.min_cnr)
    assert_allclose(vs.alt, f.alt, equal_nan=True)
    assert type(vs.alt) == type(f.alt)
    # differences of like 6e-7 in these vals
    assert vs.lat == pytest.approx(f.lat)
    assert type(vs.lat) == type(f.lat)
    assert vs.lon == pytest.approx(f.lon)
    assert type(vs.lon) == type(f.lon)
    assert_allclose(vs.height, f.height)
    assert type(vs.height) == type(f.height)
    assert_allclose(vs.mean_cnr, f.mean_cnr)
    assert type(vs.mean_cnr) == type(f.mean_cnr)
    assert vs.stime == f.stime
    assert type(vs.stime) == type(f.stime)
    assert vs.etime ==  f.etime
    assert type(vs.etime) == type(f.etime)
    assert vs.el == f.el
    assert type(vs.el) == type(f.el)
    assert vs.nbeams == f.nbeams
    assert type(vs.nbeams) == type(f.nbeams)
    assert_allclose(vs.u, f.u, equal_nan=True)
    assert type(vs.u) == type(f.u)
    assert_allclose(vs.du, f.du, equal_nan=True)
    assert type(vs.du) == type(f.du)
    assert_allclose(vs.w, f.w, equal_nan=True)
    assert type(vs.w) == type(f.w)
    assert_allclose(vs.dw, f.dw, equal_nan=True)
    assert type(vs.dw) == type(f.dw)
    assert_allclose(vs.v, f.v, equal_nan=True)
    assert type(vs.v) == type(f.v)
    assert_allclose(vs.dv, f.dv, equal_nan=True)
    assert type(vs.dv) == type(f.dv)
    assert_allclose(vs.speed, f.speed, equal_nan=True)
    assert type(vs.speed) == type(f.speed)
    assert_allclose(vs.wdir, f.wdir, equal_nan=True)
    assert type(vs.wdir) == type(f.wdir)
    assert_allclose(vs.residual, f.residual, equal_nan=True)
    assert type(vs.residual) == type(f.residual)
    assert_allclose(vs.correlation, f.correlation, equal_nan=True)
    assert type(vs.correlation) == type(f.correlation)

def test_wind_from_uv():
    u = np.array([-1, np.NZERO, 0, 1, 0, -2, np.nan], dtype=float)
    v = np.array([np.NINF, np.NZERO, 0, 0, -1, -2, 2], dtype=float)
    xspd = np.array([np.inf, 0, 0, 1, 1, np.sqrt(8), np.nan], dtype=float)
    xdir = np.array([0, 90, 270, 270, 0, 45, np.nan], dtype=float)
    wspd, wdir = vad.wspd_wdir_from_uv(u, v)
    assert_allclose(wspd, xspd, equal_nan=True)
    assert_allclose(wdir, xdir, equal_nan=True)
