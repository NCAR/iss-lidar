import numpy as np
import numpy.ma as ma
from iss_lidar.Lidar_functions import consensus_avg


def test_consensus_avg():
    # examples taken from one day's worth of SWEX VADs. In this file there are
    # never more than three VADs in any half hour bin, have taken some
    # selections at random
    # should return nan for all masked data
    window = 5
    vals = ma.array([np.nan, np.nan, np.nan])
    avg, idxs = consensus_avg(vals, window)
    assert np.isnan(avg)
    assert not idxs
    vals = ma.masked_all((3,))
    avg, idxs = consensus_avg(vals, window)
    assert np.isnan(avg)
    assert not idxs
    # only two within window
    vals = ma.array([-6.539889, -2.504943, -0.9367407])
    avg, idxs = consensus_avg(vals, window)
    assert avg == -1.72084185
    assert idxs == [1, 2]
    # change the order
    vals = ma.array([-2.504943, -6.539889, -0.9367407])
    avg, idxs = consensus_avg(vals, window)
    assert avg == -1.72084185
    assert idxs == [0, 2]
    # all within same window
    vals = ma.array([0.9532883, 0.78370535, 0.12152061])
    avg, idxs = consensus_avg(vals, window)
    assert avg == 0.6195047533333333
    assert sorted(idxs) == [0, 1, 2]
    # make sure that masked vals are getting correctly ignored
    vals = ma.array([0.9532883, np.nan, 0.78370535, np.nan, 0.12152061])
    avg, idxs = consensus_avg(vals, window)
    assert avg == 0.6195047533333333
    assert sorted(idxs) == [0, 2, 4]
    # make sure that masked vals aren't being used in calculations
    vals = ma.array([0.9532883, 2.0, 0.78370535, 2.0, 0.12152061])
    vals = ma.masked_greater(vals, 1.5)
    avg, idxs = consensus_avg(vals, window)
    assert avg == 0.6195047533333333
    assert sorted(idxs) == [0, 2, 4]
    # should return nan if there is no window with more than one point in it
    vals = ma.array([0.0, 10.0, 20.0])
    avg, idxs = consensus_avg(vals, window)
    assert np.isnan(avg)
    assert not idxs
