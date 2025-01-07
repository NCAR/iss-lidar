import datetime as dt
import pytz
import numpy as np
import numpy.ma as ma
from pathlib import Path
from iss_lidar.consensus_set import ConsensusSet
from iss_lidar.vad import VADSet

datadir = Path(__file__).parent.parent.joinpath("testdata")


def compare_consensus(a: ConsensusSet, b: ConsensusSet):
    """ Compare data from two consensus objects, allowing for small differences
    from saving/reading files, etc"""
    np.testing.assert_allclose(a.alt, b.alt)
    assert isinstance(b.alt, ma.MaskedArray)
    # differences of like 6e-7 in these vals
    np.testing.assert_allclose(a.lat, b.lat)
    assert isinstance(b.lat, ma.MaskedArray)
    np.testing.assert_allclose(a.lon, b.lon)
    assert isinstance(b.lon, ma.MaskedArray)
    np.testing.assert_allclose(a.height, b.height)
    assert isinstance(b.height, ma.MaskedArray)
    np.testing.assert_allclose(a.mean_cnr, b.mean_cnr)
    assert isinstance(b.mean_cnr, ma.MaskedArray)
    assert a.stime == b.stime
    assert isinstance(b.stime, list)
    assert a.etime == b.etime
    assert isinstance(b.etime, list)
    np.testing.assert_allclose(a.u, b.u, equal_nan=True)
    assert isinstance(b.u, ma.MaskedArray)
    np.testing.assert_allclose(a.n_u, b.n_u, equal_nan=True)
    assert isinstance(b.n_u, ma.MaskedArray)
    np.testing.assert_allclose(a.w, b.w, equal_nan=True)
    assert isinstance(b.w, ma.MaskedArray)
    np.testing.assert_allclose(a.n_w, b.n_w, equal_nan=True)
    assert isinstance(b.n_w, ma.MaskedArray)
    np.testing.assert_allclose(a.v, b.v, equal_nan=True)
    assert isinstance(b.v, ma.MaskedArray)
    np.testing.assert_allclose(a.n_v, b.n_v, equal_nan=True)
    assert isinstance(b.n_v, ma.MaskedArray)
    np.testing.assert_allclose(a.speed, b.speed, equal_nan=True)
    assert isinstance(b.speed, ma.MaskedArray)
    np.testing.assert_allclose(a.wdir, b.wdir, equal_nan=True)
    assert isinstance(b.wdir, ma.MaskedArray)
    np.testing.assert_allclose(a.residual, b.residual, equal_nan=True)
    assert isinstance(b.residual, ma.MaskedArray)
    np.testing.assert_allclose(a.correlation, b.correlation, equal_nan=True)
    assert isinstance(b.correlation, ma.MaskedArray)
    assert a.span == b.span
    assert a.window == b.window


def test_create_time_ranges():
    day = dt.date(2021, 6, 30)
    res = ConsensusSet.create_time_ranges(day, dt.timedelta(minutes=30))
    assert len(res) == 48
    assert res[0] == dt.datetime(2021, 6, 30, 00, 00, 00, 00, pytz.UTC)
    assert res[-1] == dt.datetime(2021, 6, 30, 23, 30, 00, 00, pytz.UTC)
    # try hour bins
    res = ConsensusSet.create_time_ranges(day, dt.timedelta(minutes=60))
    assert len(res) == 24
    assert res[0] == dt.datetime(2021, 6, 30, 00, 00, 00, 00, pytz.UTC)
    assert res[1] == dt.datetime(2021, 6, 30, 1, 00, 00, 00, pytz.UTC)
    assert res[-1] == dt.datetime(2021, 6, 30, 23, 00, 00, 00, pytz.UTC)


def test_median_from_consensus_idxs():
    vals = ma.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    idxs = [0, 1, 2, 7, 8]
    median = ConsensusSet.median_from_consensus_idxs(vals, idxs)
    assert median == ma.median([1, 2, 3, 2, 1])
    # if idxs is empty, return nan without producing numpy warning
    idxs = []
    median = ConsensusSet.median_from_consensus_idxs(vals, idxs)
    assert np.isnan(median)


def test_write_read_netcdf():
    vs = VADSet.from_file(f"{datadir}/VAD_20220501.nc")
    cs = ConsensusSet.from_VADSet(vs, 5, dt.timedelta(minutes=30))
    cs.to_ARM_netcdf(f"{datadir}/test_consensus.nc")
    cs_file = ConsensusSet.from_file(f"{datadir}/test_consensus.nc")
    compare_consensus(cs, cs_file)
