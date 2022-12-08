import numpy as np
import numpy.ma as ma
import datetime as dt
import pytz
from typing import List
import netCDF4

import tools
from vad import VADSet
from Lidar_functions import consensus_avg


class ConsensusSet(VADSet):

    @staticmethod
    def create_time_ranges(day: dt.date, size: dt.timedelta
                           ) -> List[dt.datetime]:
        """ Create a list of datetimes every 30 minutes for given day """
        start = dt.datetime(day.year, day.month, day.day, tzinfo=pytz.UTC)
        end = start + dt.timedelta(days=1)
        ranges = []
        while (start < end):
            ranges.append(start)
            start = start + size
        return ranges

    @staticmethod
    def median_from_consensus_idxs(vals: ma.array, idxs: List) -> float:
        """
        Calculate the median of an array from the subset of elements at the
        same indices as were used for consensus averaging for w.
        """
        return ma.median(vals[idxs])

    def __init__(self, alt: float, lat: float, lon: float, height: np.array,
                 time: list, u: ma.array, v: ma.array, w: ma.array,
                 n_u: ma.array, n_v: ma.array, n_w: ma.array,
                 residual: ma.array, correlation: ma.array, mean_cnr: ma.array,
                 wspd: ma.array, wdir: ma.array, window: float,
                 span: dt.timedelta):
        # separating out calculating from vadset into a classmethod in case we
        # want another classmethod to read from file in the future
        self.alt = alt
        self.lat = lat
        self.lon = lon
        self.height = height
        self.stime = time
        # generate etime from stime and span, no need to supply it
        self.etime = [s + span for s in self.stime]
        self.u = u
        self.v = v
        self.w = w
        self.n_u = n_u  # number of indices averaged in consensus
        self.n_v = n_v
        self.n_w = n_w
        self.residual = residual
        self.correlation = correlation
        self.mean_cnr = mean_cnr
        self.speed = wspd
        self.wdir = wdir
        self.window = window  # size of consensus buckets
        self.span = span  # timespan to average over

    @classmethod
    def from_VADSet(cls, vs: VADSet, window: float, span: dt.timedelta):
        # create half-hour time bins
        ranges = ConsensusSet.create_time_ranges(vs.stime[0].date(), span)
        u_mean = n_u = ma.zeros((len(ranges), len(vs.height)))
        v_mean = n_v = ma.zeros((len(ranges), len(vs.height)))
        w_mean = n_w = ma.zeros((len(ranges), len(vs.height)))
        residual = ma.zeros((len(ranges), len(vs.height)))
        correlation = ma.zeros((len(ranges), len(vs.height)))
        mean_cnr = ma.zeros((len(ranges), len(vs.height)))

        for idx, r in enumerate(ranges):
            start = r
            end = start + span
            bin_idxs = [i for i in range(len(vs.stime)) if vs.stime[i] >= start
                        and vs.stime[i] < end]

            # no vads in this time window
            if len(bin_idxs) == 0:
                u_mean[idx, :] = np.nan
                v_mean[idx, :] = np.nan
                w_mean[idx, :] = np.nan
                residual[idx, :] = np.nan
                correlation[idx, :] = np.nan
                mean_cnr[idx, :] = np.nan
                n_u[idx, :] = np.nan
                n_v[idx, :] = np.nan
                n_w[idx, :] = np.nan
                continue

            u_bin = ma.array([vs.u[i] for i in bin_idxs])
            v_bin = ma.array([vs.v[i] for i in bin_idxs])
            w_bin = ma.array([vs.w[i] for i in bin_idxs])
            res_bin = ma.array([vs.residual[i] for i in bin_idxs])
            cor_bin = ma.array([vs.correlation[i] for i in bin_idxs])
            mc_bin = ma.array([vs.mean_cnr[i] for i in bin_idxs])

            for hgt in range(len(vs.height)):
                avg, idxs = consensus_avg(u_bin[:, hgt], window)
                u_mean[idx, hgt] = avg
                n_u[idx, hgt] = len(idxs)
                avg, idxs = consensus_avg(v_bin[:, hgt], window)
                v_mean[idx, hgt] = avg
                n_v[idx, hgt] = len(idxs)
                avg, idxs = consensus_avg(w_bin[:, hgt], window)
                w_mean[idx, hgt] = avg
                n_w[idx, hgt] = len(idxs)
                # take median of idxs used for w to avg res, corr, mean_cnr
                residual[idx, hgt] = ConsensusSet.\
                    median_from_consensus_idxs(res_bin, idxs)
                correlation[idx, hgt] = ConsensusSet.\
                    median_from_consensus_idxs(cor_bin, idxs)
                mean_cnr[idx, hgt] = ConsensusSet.\
                    median_from_consensus_idxs(mc_bin, idxs)

        # calculate wspd/wdir from averaged u and v
        wspd, wdir = tools.wspd_wdir_from_uv(u_mean, v_mean)

        return cls(vs.alt, vs.lat, vs.lon, vs.height, ranges, u_mean, v_mean,
                   w_mean, n_u, n_v, n_w, residual, correlation,
                   mean_cnr, wspd, wdir, window, span)

    def add_aux_variables(self, nc_file: netCDF4.Dataset):
        # number of points used in consensus for uvw
        n_u = nc_file.createVariable('u_npoints', 'f', ('time', 'height'))
        n_u.missing_value = -9999.0
        n_u[:, :] = self.n_u
        n_u.long_name = ('Number of points used in consensus averaging window '
                         'for eastward component of winds')
        n_u.units = 'unitless'
        n_v = nc_file.createVariable('v_npoints', 'f', ('time', 'height'))
        n_v.missing_value = -9999.0
        n_v[:, :] = self.n_v
        n_v.long_name = ('Number of points used in consensus averaging window '
                         'for northward component of winds')
        n_v.units = 'unitless'
        n_w = nc_file.createVariable('w_npoints', 'f', ('time', 'height'))
        n_w.missing_value = -9999.0
        n_w[:, :] = self.n_w
        n_w.long_name = ('Number of points used in consensus averaging window '
                         'for vertical component of winds')
        n_w.units = 'unitless'

    def add_attributes(self, nc_file: netCDF4.Dataset):
        nc_file.consensus_avg_window = self.window