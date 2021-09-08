import pytest
import pickle
import numpy.ma as ma
import numpy as np

from ppi import PPI
from vad import xyz, non_nan_idxs, calc_A, calc_b
from vad import VAD, VADSet

@pytest.fixture
def ppi():
    """ Read in PPI file """
    ppi = PPI.fromFile("/data/iss/lotos2021/iss1/lidar/cfradial/20210630/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    ppi.threshold_cnr(-22) # use default threshold val in ppi_scans_to_vad of -22
    return ppi

@pytest.fixture
def ppis():
    """ Read in 3 PPI scans """
    a = PPI.fromFile("/data/iss/lotos2021/iss1/lidar/cfradial/20210630/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    a.threshold_cnr(-22)
    b = PPI.fromFile("/data/iss/lotos2021/iss1/lidar/cfradial/20210630/cfrad.20210630_171644_WLS200s-181_133_PPI_50m.nc")
    b.threshold_cnr(-22)
    c = PPI.fromFile("/data/iss/lotos2021/iss1/lidar/cfradial/20210630/cfrad.20210630_174238_WLS200s-181_133_PPI_50m.nc")
    c.threshold_cnr(-22)
    return(a,b,c)


@pytest.fixture
def locations():
    x = pickle.load(open("testdata/vad/x.p", "rb"))
    y = pickle.load(open("testdata/vad/y.p", "rb"))
    z = pickle.load(open("testdata/vad/z.p", "rb"))
    return(x,y,z)

@pytest.fixture
def final_vad_winds():
    u = pickle.load(open("testdata/vad/u.p", "rb"))
    v = pickle.load(open("testdata/vad/v.p", "rb"))
    w = pickle.load(open("testdata/vad/w.p", "rb"))
    return (u, v, w)

@pytest.fixture
def final_vad_errs():
    du = pickle.load(open("testdata/vad/du.p", "rb"))
    dv = pickle.load(open("testdata/vad/dv.p", "rb"))
    dw = pickle.load(open("testdata/vad/dw.p", "rb"))
    return (du, dv, dw)
    
@pytest.fixture
def derived_products():
    speed = pickle.load(open("testdata/vad/speed.p", "rb"))
    wdir = pickle.load(open("testdata/vad/wdir.p", "rb"))
    res = pickle.load(open("testdata/vad/residual.p", "rb"))
    cor = pickle.load(open("testdata/vad/correlation.p", "rb"))
    return (speed, wdir, res, cor)

def test_xyz(ppi, locations):
    saved_x, saved_y, saved_z = locations
    x, y, z = xyz(ppi.ranges, ppi.elevation, ppi.azimuth)
    assert ma.allequal(x, saved_x)
    assert ma.allequal(y, saved_y)
    assert ma.allequal(z, saved_z)

def test_non_nan_idxs(ppi):
    idxs = non_nan_idxs(ppi.vr, 0)
    saved_idxs = pickle.load(open("testdata/vad/foo0.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 1)
    saved_idxs = pickle.load(open("testdata/vad/foo1.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 2)
    saved_idxs = pickle.load(open("testdata/vad/foo2.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 3)
    saved_idxs = pickle.load(open("testdata/vad/foo3.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)

def test_calc_A(ppi):
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 0))
    saved_A = pickle.load(open("testdata/vad/A_0.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 1))
    saved_A = pickle.load(open("testdata/vad/A_1.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 2))
    saved_A = pickle.load(open("testdata/vad/A_2.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 3))
    saved_A = pickle.load(open("testdata/vad/A_3.p", "rb"))
    assert ma.allequal(A, saved_A)

def test_calc_b(ppi):
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 0), 0)
    saved_b = pickle.load(open("testdata/vad/b_0.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 1), 1)
    saved_b = pickle.load(open("testdata/vad/b_1.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 2), 2)
    saved_b = pickle.load(open("testdata/vad/b_2.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 3), 3)
    saved_b = pickle.load(open("testdata/vad/b_3.p", "rb"))
    assert ma.allequal(b, saved_b)

def test_arm_vad(ppi, final_vad_winds, final_vad_errs, derived_products):
    vad = VAD.calculate_ARM_VAD(ppi)
    saved_u, saved_v, saved_w = final_vad_winds
    assert np.array_equal(vad.u, saved_u, equal_nan=True)
    assert np.array_equal(vad.v, saved_v, equal_nan=True)
    assert np.array_equal(vad.w, saved_w, equal_nan=True)
    saved_du, saved_dv, saved_dw = final_vad_errs
    assert np.array_equal(vad.du, saved_du, equal_nan=True)
    assert np.array_equal(vad.dv, saved_dv, equal_nan=True)
    assert np.array_equal(vad.dw, saved_dw, equal_nan=True)
    saved_wspd, saved_wdir, saved_res, saved_cor = derived_products
    assert np.array_equal(vad.speed, saved_wspd, equal_nan=True)
    assert np.array_equal(vad.wdir, saved_wdir, equal_nan=True)
    assert np.array_equal(vad.residual, saved_res, equal_nan=True)
    assert np.array_equal(vad.correlation, saved_cor, equal_nan=True)

def test_vadset_netcdf(ppis):
    """ Compare data and data types between VADSet from VADs and VADSet from netcdf """
    vads = []
    for p in ppis:
        vads.append(VAD.calculate_ARM_VAD(p))
    vs = VADSet.from_VADs(vads, -22)
    vs.to_ARM_netcdf("testdata/test_vadset.nc")
    f= VADSet.from_file("testdata/test_vadset.nc")
    # vadset from file should match original vadset
    assert vs.min_cnr == f.min_cnr
    assert type(vs.min_cnr) == type(f.min_cnr)
    assert np.array_equal(vs.alt, f.alt, equal_nan=True)
    assert type(vs.alt) == type(f.alt)
    # differences of like 6e-7 in these vals
    assert vs.lat == pytest.approx(f.lat)
    assert type(vs.lat) == type(f.lat)
    assert vs.lon == pytest.approx(f.lon)
    assert type(vs.lon) == type(f.lon)
    assert np.array_equal(vs.height, f.height)
    assert type(vs.height) == type(f.height)
    assert np.allclose(vs.mean_cnr, f.mean_cnr)
    assert type(vs.mean_cnr) == type(f.mean_cnr)
    assert vs.stime == f.stime
    assert type(vs.stime) == type(f.stime)
    assert vs.etime ==  f.etime
    assert type(vs.etime) == type(f.etime)
    assert vs.el == f.el
    assert type(vs.el) == type(f.el)
    assert vs.nbeams == f.nbeams
    assert type(vs.nbeams) == type(f.nbeams)
    assert np.allclose(vs.u, f.u, equal_nan=True)
    assert type(vs.u) == type(f.u)
    assert np.allclose(vs.du, f.du, equal_nan=True)
    assert type(vs.du) == type(f.du)
    assert np.allclose(vs.w, f.w, equal_nan=True)
    assert type(vs.w) == type(f.w)
    assert np.allclose(vs.dw, f.dw, equal_nan=True)
    assert type(vs.dw) == type(f.dw)
    assert np.allclose(vs.v, f.v, equal_nan=True)
    assert type(vs.v) == type(f.v)
    assert np.allclose(vs.dv, f.dv, equal_nan=True)
    assert type(vs.dv) == type(f.dv)
    assert np.allclose(vs.speed, f.speed, equal_nan=True)
    assert type(vs.speed) == type(f.speed)
    assert np.allclose(vs.wdir, f.wdir, equal_nan=True)
    assert type(vs.wdir) == type(f.wdir)
    assert np.allclose(vs.residual, f.residual, equal_nan=True)
    assert type(vs.residual) == type(f.residual)
    assert np.allclose(vs.correlation, f.correlation, equal_nan=True)
    assert type(vs.correlation) == type(f.correlation)

    
