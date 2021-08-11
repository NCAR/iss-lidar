import pytest
import pickle
import numpy.ma as ma
import numpy as np

from ppi import PPI
from vad import xyz, non_nan_idxs, calc_A, calc_b
from vad import VAD

@pytest.fixture
def ppi():
    """ Read in PPI file """
    ppi = PPI.fromFile("/data/iss/lotos2021/iss1/lidar/cfradial/20210630/cfrad.20210630_152022_WLS200s-181_133_PPI_50m.nc")
    ppi.threshold_cnr(-22) # use default threshold val in ppi_scans_to_vad of -22
    return ppi

@pytest.fixture
def locations():
    x = pickle.load(open("pickles/vad/x.p", "rb"))
    y = pickle.load(open("pickles/vad/y.p", "rb"))
    z = pickle.load(open("pickles/vad/z.p", "rb"))
    return(x,y,z)

@pytest.fixture
def final_vad_winds():
    u = pickle.load(open("pickles/vad/u.p", "rb"))
    v = pickle.load(open("pickles/vad/v.p", "rb"))
    w = pickle.load(open("pickles/vad/w.p", "rb"))
    return (u, v, w)

@pytest.fixture
def final_vad_errs():
    du = pickle.load(open("pickles/vad/du.p", "rb"))
    dv = pickle.load(open("pickles/vad/dv.p", "rb"))
    dw = pickle.load(open("pickles/vad/dw.p", "rb"))
    return (du, dv, dw)
    
@pytest.fixture
def derived_products():
    speed = pickle.load(open("pickles/vad/speed.p", "rb"))
    wdir = pickle.load(open("pickles/vad/wdir.p", "rb"))
    res = pickle.load(open("pickles/vad/residual.p", "rb"))
    cor = pickle.load(open("pickles/vad/correlation.p", "rb"))
    return (speed, wdir, res, cor)

def test_xyz(ppi, locations):
    saved_x, saved_y, saved_z = locations
    x, y, z = xyz(ppi.ranges, ppi.elevation, ppi.azimuth)
    assert ma.allequal(x, saved_x)
    assert ma.allequal(y, saved_y)
    assert ma.allequal(z, saved_z)

def test_non_nan_idxs(ppi):
    idxs = non_nan_idxs(ppi.vr, 0)
    saved_idxs = pickle.load(open("pickles/vad/foo0.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 1)
    saved_idxs = pickle.load(open("pickles/vad/foo1.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 2)
    saved_idxs = pickle.load(open("pickles/vad/foo2.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)
    idxs = non_nan_idxs(ppi.vr, 3)
    saved_idxs = pickle.load(open("pickles/vad/foo3.p", "rb"))
    assert ma.allequal(idxs, saved_idxs)

def test_calc_A(ppi):
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 0))
    saved_A = pickle.load(open("pickles/vad/A_0.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 1))
    saved_A = pickle.load(open("pickles/vad/A_1.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 2))
    saved_A = pickle.load(open("pickles/vad/A_2.p", "rb"))
    assert ma.allequal(A, saved_A)
    A = calc_A(ppi.elevation, ppi.azimuth, non_nan_idxs(ppi.vr, 3))
    saved_A = pickle.load(open("pickles/vad/A_3.p", "rb"))
    assert ma.allequal(A, saved_A)

def test_calc_b(ppi):
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 0), 0)
    saved_b = pickle.load(open("pickles/vad/b_0.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 1), 1)
    saved_b = pickle.load(open("pickles/vad/b_1.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 2), 2)
    saved_b = pickle.load(open("pickles/vad/b_2.p", "rb"))
    assert ma.allequal(b, saved_b)
    b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, non_nan_idxs(ppi.vr, 3), 3)
    saved_b = pickle.load(open("pickles/vad/b_3.p", "rb"))
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

def test_vadset():
    pass
