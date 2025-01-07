# Object to hold data from a PPI scan

import numpy as np
import numpy.ma as ma
from datetime import datetime
from .tools import read_cfradial


class PPI:

    def __init__(self, cnr: np.ma.MaskedArray = None,
                 ranges: np.ma.MaskedArray = None,
                 vr: np.ma.MaskedArray = None,
                 elevation: np.ma.MaskedArray = None,
                 azimuth: np.ma.MaskedArray = None, starttime: datetime = None,
                 endtime: datetime = None,
                 lat: np.ma.core.MaskedConstant = None,
                 lon: np.ma.core.MaskedConstant = None,
                 alt: np.ma.core.MaskedConstant = None):
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
    def from_file(cls, filename: str):
        """ Create PPI object by reading cfradial scan file """
        [cnr, ranges, vr, elevation, azimuth, starttime, endtime, lat, lon,
         alt] = read_cfradial(filename)
        return cls(cnr, ranges, vr, elevation, azimuth, starttime, endtime,
                   lat, lon, alt)

    def threshold_cnr(self, max_cnr: int):
        """ Set vr to nan if cnr is below threshold """
        self.vr = ma.masked_where(self.cnr < max_cnr, self.vr)

    def mean_cnr(self) -> np.ndarray:
        """ Return mean value of CNR """
        return np.nanmean(self.cnr, axis=0)
