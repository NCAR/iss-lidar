import numpy.ma as ma
from iss_lidar.ppi import PPI


def test_threshold_cnr():
    vel = ma.arange(1, 20)
    ppi = PPI(cnr=ma.arange(-29, -10), vr=vel)
    ppi.threshold_cnr(-22)
    vel = ma.masked_array(vel, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0])
    assert ma.allequal(vel, ppi.vr)
