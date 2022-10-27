import pytest
import pickle
import numpy as np
import numpy.testing
from pathlib import Path

from ppi import PPI
from vad import VAD, VADSet

datadir = Path(__file__).parent.joinpath("testdata")


def assert_allclose(actual, desired, rtol=1e-06, atol=1e-05,
                    equal_nan=True, err_msg='', verbose=True) -> None:
    "Wrap numpy.testing.assert_allclose() with different default tolerances."
    numpy.testing.assert_allclose(actual, desired, rtol, atol,
                                  equal_nan, err_msg, verbose)


def compare_vadsets(a: VADSet, b: VADSet):
    """ Compare data from two VADSet objects, allowing for small differences
    from saving/reading files, etc"""
    assert a.min_cnr == b.min_cnr
    assert isinstance(b.min_cnr, int)
    assert_allclose(a.alt, b.alt, equal_nan=True)
    assert isinstance(b.alt, numpy.ma.MaskedArray)
    # differences of like 6e-7 in these vals
    assert a.lat == pytest.approx(b.lat)
    assert isinstance(b.lat, numpy.ma.MaskedArray)
    assert a.lon == pytest.approx(b.lon)
    assert isinstance(b.lon, numpy.ma.MaskedArray)
    assert_allclose(a.height, b.height)
    assert isinstance(b.height, numpy.ma.MaskedArray)
    assert_allclose(a.mean_cnr, b.mean_cnr)
    assert isinstance(b.mean_cnr, numpy.ndarray)
    assert a.stime == b.stime
    assert isinstance(b.stime, list)
    assert a.etime == b.etime
    assert isinstance(b.etime, list)
    assert a.el == b.el
    assert isinstance(b.el, list)
    assert a.nbeams == b.nbeams
    assert isinstance(b.nbeams, list)
    assert_allclose(a.u, b.u, equal_nan=True)
    assert isinstance(b.u, numpy.ndarray)
    assert_allclose(a.du, b.du, equal_nan=True)
    assert isinstance(b.du, numpy.ndarray)
    assert_allclose(a.w, b.w, equal_nan=True)
    assert isinstance(b.w, numpy.ndarray)
    assert_allclose(a.dw, b.dw, equal_nan=True)
    assert isinstance(b.dw, numpy.ndarray)
    assert_allclose(a.v, b.v, equal_nan=True)
    assert isinstance(b.v, numpy.ndarray)
    assert_allclose(a.dv, b.dv, equal_nan=True)
    assert isinstance(b.dv, numpy.ndarray)
    assert_allclose(a.speed, b.speed, equal_nan=True)
    assert isinstance(b.speed, numpy.ndarray)
    assert_allclose(a.wdir, b.wdir, equal_nan=True)
    assert isinstance(b.wdir, numpy.ndarray)
    assert_allclose(a.residual, b.residual, equal_nan=True)
    assert isinstance(b.residual, numpy.ndarray)
    assert_allclose(a.correlation, b.correlation, equal_nan=True)
    assert isinstance(b.correlation, numpy.ndarray)



@pytest.fixture
def ppi():
    """ Read in PPI file """
    ppi = PPI.from_file(
        f"{datadir}/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    # use default threshold val in ppi_scans_to_vad of -22
    ppi.threshold_cnr(-22)
    return ppi


@pytest.fixture
def ppis():
    """ Read in 3 PPI scans """
    a = PPI.from_file(
        f"{datadir}/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    a.threshold_cnr(-22)
    b = PPI.from_file(
        f"{datadir}/cfrad.20210630_171644_WLS200s-181_133_PPI_50m.nc")
    b.threshold_cnr(-22)
    c = PPI.from_file(
        f"{datadir}/cfrad.20210630_174238_WLS200s-181_133_PPI_50m.nc")
    c.threshold_cnr(-22)
    return(a, b, c)


@pytest.fixture
def locations():
    x = pickle.load(open(f"{datadir}/vad/x.p", "rb"))
    y = pickle.load(open(f"{datadir}/vad/y.p", "rb"))
    z = pickle.load(open(f"{datadir}/vad/z.p", "rb"))
    return(x, y, z)


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
    x, y, z = VAD.xyz(ppi.ranges, ppi.elevation, ppi.azimuth)
    assert_allclose(x, saved_x)
    assert_allclose(y, saved_y)
    assert_allclose(z, saved_z)


def test_non_nan_idxs(ppi):
    idxs = VAD.non_nan_idxs(ppi.vr, 0)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo0.p", "rb"))
    assert_allclose(idxs, saved_idxs)
    idxs = VAD.non_nan_idxs(ppi.vr, 1)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo1.p", "rb"))
    assert_allclose(idxs, saved_idxs)
    idxs = VAD.non_nan_idxs(ppi.vr, 2)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo2.p", "rb"))
    assert_allclose(idxs, saved_idxs)
    idxs = VAD.non_nan_idxs(ppi.vr, 3)
    saved_idxs = pickle.load(open(f"{datadir}/vad/foo3.p", "rb"))
    assert_allclose(idxs, saved_idxs)


def test_calc_A(ppi):
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 0))
    saved_A = pickle.load(open(f"{datadir}/vad/A_0.p", "rb"))
    assert_allclose(A, saved_A)
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 1))
    saved_A = pickle.load(open(f"{datadir}/vad/A_1.p", "rb"))
    assert_allclose(A, saved_A)
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 2))
    saved_A = pickle.load(open(f"{datadir}/vad/A_2.p", "rb"))
    assert_allclose(A, saved_A)
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 3))
    saved_A = pickle.load(open(f"{datadir}/vad/A_3.p", "rb"))
    assert_allclose(A, saved_A)


def test_calc_b(ppi):
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 0), 0)
    saved_b = pickle.load(open(f"{datadir}/vad/b_0.p", "rb"))
    assert_allclose(b, saved_b)
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 1), 1)
    saved_b = pickle.load(open(f"{datadir}/vad/b_1.p", "rb"))
    assert_allclose(b, saved_b)
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 2), 2)
    saved_b = pickle.load(open(f"{datadir}/vad/b_2.p", "rb"))
    assert_allclose(b, saved_b)
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 3), 3)
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
    """ Compare data and data types between VADSet from VADs and VADSet from
    netcdf """
    vads = []
    for p in ppis:
        vads.append(VAD.calculate_ARM_VAD(p))
    vs = VADSet.from_VADs(vads, -22)
    vs.to_ARM_netcdf(f"{datadir}/test_vadset.nc")
    f = VADSet.from_file(f"{datadir}/test_vadset.nc")
    # vadset from file should match original vadset
    compare_vadsets(vs, f)


def test_wind_from_uv():
    u = np.array([-1, np.NZERO, 0, 1, 0, -2, np.nan], dtype=float)
    v = np.array([np.NINF, np.NZERO, 0, 0, -1, -2, 2], dtype=float)
    xspd = np.array([np.inf, 0, 0, 1, 1, np.sqrt(8), np.nan], dtype=float)
    xdir = np.array([0, 90, 270, 270, 0, 45, np.nan], dtype=float)
    wspd, wdir = VAD.wspd_wdir_from_uv(u, v)
    assert_allclose(wspd, xspd, equal_nan=True)
    assert_allclose(wdir, xdir, equal_nan=True)


def test_vadset_from_PPIs(ppis):
    """ Test that vadset from_PPIs function produces the same result as
    calculating VADs individually and then creating VADSet from them """
    vads = []
    for p in ppis:
        vads.append(VAD.calculate_ARM_VAD(p))
    fromVADs = VADSet.from_VADs(vads, -22)
    files = [f"{datadir}/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc",
             f"{datadir}/cfrad.20210630_171644_WLS200s-181_133_PPI_50m.nc",
             f"{datadir}/cfrad.20210630_174238_WLS200s-181_133_PPI_50m.nc"]
    fromPPIs = VADSet.from_PPIs(files, -22)
    assert fromVADs == fromPPIs


#     def create_ARM_nc(self, mean_cnr: np.ndarray, max_cnr: float,
#                      altitude: float, latitude: float, longitude: float,
#                      stime: list, etime: list, file_path: str):

def test_vad_vs_vadset(ppi):
    # try to see if i get the same netcdf from a single VAD as I get from a
    # VADSet with one VAD in it
    single = VAD.calculate_ARM_VAD(ppi)
    single.to_ARM_nc(f"{datadir}/test_single_vad.nc")
    set = VADSet.from_PPIs([f"{datadir}/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc"], -22)
    set.to_ARM_netcdf(f"{datadir}/test_single_vadset.nc")
    a = VADSet.from_file(f"{datadir}/test_single_vad.nc")
    b = VADSet.from_file(f"{datadir}/test_single_vadset.nc")
    compare_vadsets(a, b)

