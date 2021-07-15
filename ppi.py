# Object to hold data from a PPI scan

import numpy as np
import numpy.ma as ma
import Lidar_functions

class PPI:
    # can create by reading cfradial file
    def __init__(self, filename):
        [self.cnr, self.ranges, self.vr, self.elevation, self.azimuth, self.str_start, self.str_end, self.lat, self.lon, self.alt] = Lidar_functions.read_cfradial(filename)

    def __init__(self, cnr=None, ranges=None, vr=None, elevation=None, azimuth=None, str_start=None, str_end=None, lat=None, lon=None, alt=None):
        """ Create PPI object manually """
        self.cnr = cnr
        self.ranges = ranges
        self.vr = vr
        self.elevation = elevation
        self.azimuth = azimuth
        self.str_start = str_start
        self.str_end = str_end
        self.lat = lat
        self.lon = lon
        self.alt = alt



    def threshold_cnr(self, max_cnr):
        """ Set vr to nan if cnr is below threshold """
        self.vr = ma.masked_where(self.cnr < max_cnr, self.vr)


def test_threshold_cnr():
    vel = ma.arange(1,20)
    ppi = PPI(cnr = ma.arange(-29,-10), vr=vel)
    ppi.threshold_cnr(-22)
    vel = ma.masked_array(vel, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert ma.allequal(vel, ppi.vr)
