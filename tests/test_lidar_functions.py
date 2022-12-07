import numpy as np
import numpy.ma as ma
from Lidar_functions import consensus_avg
from vad import VADSet
from vad_to_30min_winds import create_time_ranges
from pathlib import Path
datadir = Path(__file__).parent.parent.joinpath("testdata")


# def test_consensus_avg_old():
#     vs = VADSet.from_file(f"{datadir}/VAD_42_20220501.nc")
#     ranges = create_time_ranges(vs.stime[0].date())
#     u_mean, v_mean, w_mean = vs.consensus_average(ranges)
#     assert False


def test_consensus_avg():
    # examples taken from one day's worth of SWEX VADs. In this file there are never more than three VADs in any half hour bin, have taken some selections at random
    # should return nan for all masked data
    window = 5
    vals = ma.array([np.nan, np.nan, np.nan])
    assert np.isnan(consensus_avg(vals, window))
    vals = ma.masked_all((3,))
    assert np.isnan(consensus_avg(vals, window))
    vals = ma.array([-6.539889, -2.504943, -0.9367407]) # only two within window
    assert consensus_avg(vals, window) == -1.72084185
    vals = ma.array([0.9532883, 0.78370535, 0.12152061]) # all within same window
    assert consensus_avg(vals, window) == 0.6195047533333333
