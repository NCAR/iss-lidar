# Object to hold data from a PPI scan

import numpy as np
import Lidar_functions

class PPI:
    # can create by reading cfradial file
    def __init__(self, filename):
        [self.cnr_ppi, self.ranges_ppi, self.vr, self.elevation, self.azimuth, self.str_start_ppi, self.str_end_ppi, self.lat, self.lon, self.alt] = Lidar_functions.read_cfradial(filename)


    def threshold_cnr(self, max_cnr):
        """ Set vr to nan if cnr is below threshold """
        # replace w/ np.masked_where?
        for x in range(len(self.azimuth)):
            for y in range(len(self.ranges_ppi)):
                if self.cnr_ppi[x, y] < max_cnr:
                    self.vr[x, y] = np.nan
