import datetime as dt
import pytz
import numpy.ma as ma
from pathlib import Path
from consensus_set import ConsensusSet
from vad import VADSet

datadir = Path(__file__).parent.parent.joinpath("testdata")


def test_create_time_ranges():
    day = dt.date(2021, 6, 30)
    res = ConsensusSet.create_time_ranges(day, dt.timedelta(minutes=30))
    assert len(res) == 48
    assert res[0] == dt.datetime(2021, 6, 30, 00, 00, 00, 00, pytz.UTC)
    assert res[-1] == dt.datetime(2021, 6, 30, 23, 30, 00, 00, pytz.UTC)


def test_median_from_consensus_idxs():
    vals = ma.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    idxs = [0, 1, 2, 7, 8]
    median = ConsensusSet.median_from_consensus_idxs(vals, idxs)
    assert median == ma.median([1, 2, 3, 2, 1])


def test_write_netcdf():
    vs = VADSet.from_file(f"{datadir}/VAD_20220501.nc")
    cs = ConsensusSet.from_VADSet(vs, 5, dt.timedelta(minutes=30))
    cs.to_ARM_netcdf(f"{datadir}/test_consensus.nc")

