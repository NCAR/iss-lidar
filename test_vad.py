import pytest
import pickle
import numpy.ma as ma

from ppi import PPI
from vad import xyz, non_nan_idxs, calc_A, calc_b
from vad import VAD

@pytest.fixture
def ppi():
    """ Read in PPI file """
    ppi = PPI.fromFile("/data/iss/lotos2021/iss1/lidar/cfradial/20210630/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    return ppi

@pytest.fixture
def locations():
    x = pickle.load(open("pickles/x.p", "rb"))
    y = pickle.load(open("pickles/y.p", "rb"))
    z = pickle.load(open("pickles/z.p", "rb"))
    return(x,y,z)

@pytest.fixture
def final_vad_winds():
    u = pickle.load(open("pickles/u.p", "rb"))
    v = pickle.load(open("pickles/v.p", "rb"))
    w = pickle.load(open("pickles/w.p", "rb"))
    return (u, v, w)

@pytest.fixture
def final_vad_errs():
    du = pickle.load(open("pickles/du.p", "rb"))
    dv = pickle.load(open("pickles/dv.p", "rb"))
    dw = pickle.load(open("pickles/dw.p", "rb"))
    return (du, dv, dw)
    
@pytest.fixture
def derived_products():
    speed = pickle.load(open("pickles/speed.p", "rb"))
    wdir = pickle.load(open("pickles/wdir.p", "rb"))
    res = pickle.load(open("pickles/residual.p", "rb"))
    cor = pickle.load(open("pickles/correlation.p", "rb"))
    return (speed, wdir, res, cor)

def test_xyz(ppi, locations):
    saved_x, saved_y, saved_z = locations
    x, y, z = xyz(ppi.ranges, ppi.elevation, ppi.azimuth)
    assert ma.allequal(x, saved_x)
    assert ma.allequal(y, saved_y)
    assert ma.allequal(z, saved_z)

def test_non_nan_idxs(ppi):
    idxs = non_nan_idxs(ppi.vr, 0)
    saved_idxs = pickle.load(open("pickles/foo0.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 1)
    saved_idxs = pickle.load(open("pickles/foo1.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 2)
    saved_idxs = pickle.load(open("pickles/foo2.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 3)
    saved_idxs = pickle.load(open("pickles/foo3.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)

def test_calc_A(ppi):
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 0))
    saved_A = pickle.load(open("pickles/A_0.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 1))
    saved_A = pickle.load(open("pickles/A_1.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 2))
    saved_A = pickle.load(open("pickles/A_2.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 3))
    saved_A = pickle.load(open("pickles/A_3.p", "rb"))
    assert ma.allequal(A, saved_A)

def test_calc_b(ppi):
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 0), 0)
    saved_b = pickle.load(open("pickles/b_0.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 1), 1)
    saved_b = pickle.load(open("pickles/b_1.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 2), 2)
    saved_b = pickle.load(open("pickles/b_2.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 3), 3)
    saved_b = pickle.load(open("pickles/b_3.p", "rb"))
    assert ma.allequal(b, saved_b)

def test_arm_vad(ppi, final_vad_winds, final_vad_errs, derived_products):
    vad = VAD.calculate_ARM_VAD(ppi.vr, ppi.ranges, ppi.elevation, ppi.azimuth)
    saved_u, saved_v, saved_w = final_vad_winds
    assert ma.allequal(vad.u, saved_u)
    assert ma.allequal(vad.v, saved_v)
    assert ma.allequal(vad.w, saved_w)
    saved_du, saved_dv, saved_dw = final_vad_errs
    assert ma.allequal(vad.du, saved_du)
    assert ma.allequal(vad.dv, saved_dv)
    assert ma.allequal(vad.dw, saved_dw)
    saved_wspd, saved_wdir, saved_res, saved_cor = derived_products
    assert ma.allequal(vad.speed, saved_wspd)
    assert ma.allequal(vad.wdir, saved_wdir)
    assert ma.allequal(vad.residual, saved_res)
    assert ma.allequal(vad.correlation, saved_cor)

