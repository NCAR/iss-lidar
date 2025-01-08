import pytest
import json
import netCDF4
import numpy as np
import numpy.ma as ma
import numpy.testing
from pathlib import Path

from iss_lidar.ppi import PPI
from iss_lidar.vad import VAD, VADSet

datadir = Path(__file__).parent.parent.joinpath("testdata")


def assert_allclose(actual, desired, rtol=1e-06, atol=1e-05,
                    equal_nan=True, err_msg='', verbose=True) -> None:
    "Wrap numpy.testing.assert_allclose() with different default tolerances."
    numpy.testing.assert_allclose(actual, desired, rtol, atol,
                                  equal_nan, err_msg, verbose)


def compare_vadsets(a: VADSet, b: VADSet, compare_errors=True):
    """ Compare data from two VADSet objects, allowing for small differences
    from saving/reading files, etc"""
    assert a.min_cnr == b.min_cnr
    assert isinstance(b.min_cnr, int)
    assert_allclose(a.alt, b.alt)
    assert isinstance(b.alt, numpy.ma.MaskedArray)
    # differences of like 6e-7 in these vals
    assert_allclose(a.lat, b.lat)
    assert isinstance(b.lat, numpy.ma.MaskedArray)
    assert_allclose(a.lon, b.lon)
    assert isinstance(b.lon, numpy.ma.MaskedArray)
    assert_allclose(a.height, b.height)
    assert isinstance(b.height, numpy.ma.MaskedArray)
    assert_allclose(a.mean_cnr, b.mean_cnr)
    assert isinstance(b.mean_cnr, numpy.ndarray)
    assert a.stime == b.stime
    assert isinstance(b.stime, list)
    assert a.etime == b.etime
    assert isinstance(b.etime, list)
    assert_allclose(a.el, b.el)
    assert isinstance(b.el, numpy.ma.MaskedArray)
    assert_allclose(a.nbeams, b.nbeams)
    assert isinstance(b.nbeams, numpy.ma.MaskedArray)
    assert_allclose(a.u, b.u, equal_nan=True)
    assert isinstance(b.u, numpy.ndarray)
    if compare_errors:
        assert_allclose(a.du, b.du, equal_nan=True)
        assert isinstance(b.du, numpy.ndarray)
    assert_allclose(a.w, b.w, equal_nan=True)
    assert isinstance(b.w, numpy.ndarray)
    if compare_errors:
        assert_allclose(a.dw, b.dw, equal_nan=True)
        assert isinstance(b.dw, numpy.ndarray)
    assert_allclose(a.v, b.v, equal_nan=True)
    assert isinstance(b.v, numpy.ndarray)
    if compare_errors:
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
    locs = json.load(open(f"{datadir}/vad/locations.json", "r"))
    x = locs["x"]
    y = locs["y"]
    z = locs["z"]
    return(x, y, z)


@pytest.fixture
def final_vad_winds():
    winds = json.load(open(f"{datadir}/vad/winds.json", "r"))
    u = winds["u"]
    v = winds["v"]
    w = winds["w"]
    return (u, v, w)


@pytest.fixture
def final_vad_errs():
    errs = json.load(open(f"{datadir}/vad/wind_errors.json", "r"))
    du = errs["du"]
    dv = errs["dv"]
    dw = errs["dw"]
    return (du, dv, dw)


@pytest.fixture
def derived_products():
    prods = json.load(open(f"{datadir}/vad/derived_products.json", "r"))
    speed = prods["spd"]
    wdir = prods["dir"]
    res = prods["res"]
    cor = prods["cor"]
    return (speed, wdir, res, cor)


def test_xyz(ppi, locations):
    saved_x, saved_y, saved_z = locations
    x, y, z = VAD.xyz(ppi.ranges, ppi.elevation, ppi.azimuth)
    assert_allclose(x, saved_x)
    assert_allclose(y, saved_y)
    assert_allclose(z, saved_z)


def test_non_nan_idxs(ppi):
    saved_idxs = json.load(open(f"{datadir}/vad/non_nan_idxs.json", "r"))
    idxs = VAD.non_nan_idxs(ppi.vr, 0)
    assert_allclose(idxs, saved_idxs["0"])
    idxs = VAD.non_nan_idxs(ppi.vr, 1)
    assert_allclose(idxs, saved_idxs["1"])
    idxs = VAD.non_nan_idxs(ppi.vr, 2)
    assert_allclose(idxs, saved_idxs["2"])
    idxs = VAD.non_nan_idxs(ppi.vr, 3)
    assert_allclose(idxs, saved_idxs["3"])


def test_calc_A(ppi):
    saved_A = json.load(open(f"{datadir}/vad/a.json", "r"))
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 0))
    assert_allclose(A, saved_A["a0"])
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 1))
    assert_allclose(A, saved_A["a1"])
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 2))
    assert_allclose(A, saved_A["a2"])
    A = VAD.calc_A(ppi.elevation, ppi.azimuth, VAD.non_nan_idxs(ppi.vr, 3))
    assert_allclose(A, saved_A["a3"])


def test_calc_b(ppi):
    saved_b = json.load(open(f"{datadir}/vad/b.json", "r"))
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 0), 0)
    assert_allclose(b, saved_b["b0"])
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 1), 1)
    assert_allclose(b, saved_b["b1"])
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 2), 2)
    assert_allclose(b, saved_b["b2"])
    b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr,
                   VAD.non_nan_idxs(ppi.vr, 3), 3)
    assert_allclose(b, saved_b["b3"])


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
    # skip comparing du,dv,dw which are not saved to netcdf currently
    compare_vadsets(vs, f, compare_errors=False)


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


def test_vad_vs_vadset(ppi):
    # try to see if i get the same netcdf from a single VAD as I get from a
    # VADSet with one VAD in it
    single = VAD.calculate_ARM_VAD(ppi)
    single.to_ARM_nc(f"{datadir}/test_single_vad.nc")
    set = VADSet.from_PPIs(
        [f"{datadir}/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc"], -22)
    set.to_ARM_netcdf(f"{datadir}/test_single_vadset.nc")
    a = VADSet.from_file(f"{datadir}/test_single_vad.nc")
    b = VADSet.from_file(f"{datadir}/test_single_vadset.nc")
    compare_vadsets(a, b)


def test_missing_val_if_nan():
    assert VADSet.missing_val_if_nan(25.3) == 25.3
    assert VADSet.missing_val_if_nan(None) == -9999.0
    assert VADSet.missing_val_if_nan(np.nan) == -9999.0
    assert VADSet.missing_val_if_nan(np.nan, -999.0) == -999.0


def test_get_mask():
    # masks made up from thresholds {"correlation_min": 0.9, "residual_max":
    # 0.7, "mean_snr_max": -29.0} cor/res are much stricter than they will be
    # in real life so i can guarantee they will remove values
    corr_mask = np.array([[False, False, False, False, False, False, False,
                           False, False, False, False, False, False, False,
                           False, False, False, False, False, False, False,
                           False, False, False, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True],
                          [False, False, True, True, False, False, False, True,
                           False, False, False, True, True, True, False, False,
                           False, True, True, False, False, False, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True],
                          [False, False, False, False, False, False, False,
                           True, True, True, True, True, True, True, True,
                           True, False, False, False, False, False, False,
                           False, True, False, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True,
                           True, True]])
    res_mask = np.array([[False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True],
                         [False, False, False, False, False, False, False,
                          False, False, False, False, True, True, True, True,
                          False, False, True, False, False, False, False,
                          False, False, False, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True],
                         [False, False, False, False, False, False, False,
                          False, True, False, True, True, True, True, True,
                          False, False, False, False, False, False, False,
                          False, False, False, False, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True]])
    snr_mask = np.array([[False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True],
                         [False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True],
                         [False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False,
                          False, False, False, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True]])
    vs = VADSet.from_file(f"{datadir}/test_vadset.nc")
    # correlation mask
    vs.thresholds = {"correlation_min": 0.9}
    assert ma.allequal(corr_mask, vs.get_mask())
    # residual mask
    vs.thresholds = {"residual_max": 0.7}
    assert ma.allequal(res_mask, vs.get_mask())
    # snr mask
    vs.thresholds = {"mean_snr_min": -29.0}
    assert ma.allequal(snr_mask, vs.get_mask())
    # all three combined
    vs.thresholds = {"correlation_min": 0.9, "residual_max": 0.7,
                     "mean_snr_min": -29.0}
    combined = ma.mask_or(ma.mask_or(corr_mask, res_mask), snr_mask)
    assert ma.allequal(combined, vs.get_mask())
    # make sure other vals in dict don't break processing
    vs.thresholds = {"correlation_min": 0.9, "residual_max": 0.7,
                     "mean_snr_min": -29.0, "test_value": 0.1}
    assert ma.allequal(combined, vs.get_mask())
    # check that blank mask (all false) returned when no thresholds present
    vs.thresholds = {}
    assert np.all(~vs.get_mask())


def test_apply_mask():
    vs_original = VADSet.from_file(f"{datadir}/test_vadset.nc")
    mask = np.full(vs_original.correlation.shape, False)
    mask[0, 0] = mask[0, 1] = mask[0, 2] = mask[0, 3] = mask[0, 4] = True
    mask[1, 5] = mask[1, 6] = mask[1, 7] = mask[1, 8] = mask[1, 9] = True
    mask[2, 0] = mask[2, 1] = mask[2, 2] = mask[2, 3] = mask[2, 4] = True
    vs_masked = VADSet.from_file(f"{datadir}/test_vadset.nc")
    vs_masked.thresholds = {"correlation_min": 0.9, "residual_max": 0.7,
                            "mean_snr_min": -29.0}
    vs_masked.apply_thresholds(mask=mask)
    arr = np.array([True, True, True, True, True])
    assert ma.allequal((ma.getmaskarray(vs_masked.u))[0, 0:5], arr)
    assert ma.allequal((ma.getmaskarray(vs_masked.u))[1, 5:10], arr)
    assert ma.allequal((ma.getmaskarray(vs_masked.u))[2, 0:5], arr)
    assert ma.allequal((ma.getmaskarray(vs_masked.v))[0, 0:5], arr)
    assert ma.allequal((ma.getmaskarray(vs_masked.w))[0, 0:5], arr)
    assert ma.allequal((ma.getmaskarray(vs_masked.du))[0, 0:5], arr)
    assert ma.allequal((ma.getmaskarray(vs_masked.residual))[0, 0:5], arr)
    assert ma.allequal((ma.getmaskarray(vs_masked.speed))[0, 0:5], arr)
    # check that previous mask still present
    # i know that none of the values i masked out were previously masked
    assert ma.allequal(ma.getmaskarray(vs_masked.u),
                       ma.mask_or(ma.getmaskarray(vs_original.u), mask))


def test_nbeams_fraction_threshold():
    nbeams = [100, 50, 200]
    nbeams_used = np.array([[100, 90, 80, 60, 40],[50, 40,30, 20, 10],
                            [200, 150, 125, 100, 75]])
    expected_mask = np.array([[False, False, False, False, True],
                              [False, False, False, True, True],
                              [False, False, False, False, True]])
    mask = VADSet.get_nbeams_used_mask(nbeams, nbeams_used, 0.5)
    assert ma.allequal(mask, expected_mask)
    vs = VADSet.from_file(f"{datadir}/test_vadset.nc")
    mask = VADSet.get_nbeams_used_mask(vs.nbeams, vs.nbeams_used, 0.5)
    # nbeams in this vadset is always 360. check that no nbeams_used are left
    # if nbeams_used is less than 180
    vs.nbeams_used.mask = ma.mask_or(vs.nbeams_used.mask, mask)
    assert ma.min(vs.nbeams_used) >= 180


def test_thresholds_in_netcdf():
    """ Test including threshold vals as attributes in netcdf. """
    vs = VADSet.from_file(f"{datadir}/test_vadset.nc")
    vs.thresholds = {"correlation_min": 0.9, "residual_max": 0.7,
                     "mean_snr_min": -29.0}
    vs.apply_thresholds()
    vs.to_ARM_netcdf(f"{datadir}/test_vadset_thresholds.nc")
    vs_thresholds = netCDF4.Dataset(f"{datadir}/test_vadset_thresholds.nc",
                                    'r')
    attrs = vs_thresholds.__dict__
    assert "threshold_correlation_min" in attrs
    assert "threshold_residual_max" in attrs
    assert "threshold_mean_snr_min" in attrs
    assert attrs["threshold_correlation_min"] == 0.9
    assert attrs["threshold_residual_max"] == 0.7
    assert attrs["threshold_mean_snr_min"] == -29.0
# updating tests example:
# d = {"du": vad.du.tolist(), "dv": vad.dv.tolist(), "dw": vad.dw.tolist()}
# f = open("testdata/vad/wind_errors.json", "w")
# json.dump(d, f)
# f.close()