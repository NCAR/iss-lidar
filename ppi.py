# Object to hold data from a PPI scan

import numpy as np
import numpy.ma as ma
import Lidar_functions


class PPI:

    def __init__(self, cnr=None, ranges=None, vr=None, elevation=None,
                 azimuth=None, starttime=None, endtime=None, lat=None,
                 lon=None, alt=None):
        """ Create PPI object from numpy arrays of data"""
        self.cnr = cnr
        self.ranges = ranges
        self.vr = vr
        self.elevation = elevation
        self.azimuth = azimuth
        self.starttime = starttime
        self.endtime = endtime
        self.lat = lat
        self.lon = lon
        self.alt = alt

    @classmethod
    def fromFile(cls, filename):
        """ Create PPI object by reading cfradial scan file """
        [cnr, ranges, vr, elevation, azimuth, starttime, endtime, lat, lon,
         alt] = Lidar_functions.read_cfradial(filename)
        return cls(cnr, ranges, vr, elevation, azimuth, starttime, endtime,
                   lat, lon, alt)

    def threshold_cnr(self, max_cnr):
        """ Set vr to nan if cnr is below threshold """
        self.vr = ma.masked_where(self.cnr < max_cnr, self.vr)

    def mean_cnr(self):
        """ Return mean value of CNR """
        return np.nanmean(self.cnr, axis=0)


def test_threshold_cnr():
    vel = ma.arange(1, 20)
    ppi = PPI(cnr=ma.arange(-29, -10), vr=vel)
    ppi.threshold_cnr(-22)
    vel = ma.masked_array(vel, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0])
    assert ma.allequal(vel, ppi.vr)
