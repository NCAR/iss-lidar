import numpy as np
import numpy.ma as ma
import datetime as dt
import pytz
from typing import List
import netCDF4

import iss_lidar.tools as tools
from .vad import VADSet
from .Lidar_functions import consensus_avg


class ConsensusSet(VADSet):

    @staticmethod
    def create_time_ranges(day: dt.date, size: dt.timedelta
                           ) -> List[dt.datetime]:
        """ Create a list of datetimes every given interval for this day """
        # currently if the interval size is not divisible into 24 hours, the
        # last time range will extend into the following day
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
        # avoid "mean of an empty slice" numpy runtime warning
        if not idxs:
            return np.nan
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
        u_mean = ma.zeros((len(ranges), len(vs.height)))
        v_mean = ma.zeros((len(ranges), len(vs.height)))
        w_mean = ma.zeros((len(ranges), len(vs.height)))
        n_u = ma.zeros((len(ranges), len(vs.height)))
        n_v = ma.zeros((len(ranges), len(vs.height)))
        n_w = ma.zeros((len(ranges), len(vs.height)))
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
                    median_from_consensus_idxs(res_bin[:, hgt], idxs)
                correlation[idx, hgt] = ConsensusSet.\
                    median_from_consensus_idxs(cor_bin[:, hgt], idxs)
                mean_cnr[idx, hgt] = ConsensusSet.\
                    median_from_consensus_idxs(mc_bin[:, hgt], idxs)

        # calculate wspd/wdir from averaged u and v
        wspd, wdir = tools.wspd_wdir_from_uv(u_mean, v_mean)

        return cls(vs.alt,
                   vs.lat,
                   vs.lon,
                   vs.height,
                   ranges,
                   ma.masked_invalid(u_mean),
                   ma.masked_invalid(v_mean),
                   ma.masked_invalid(w_mean),
                   ma.masked_invalid(n_u),
                   ma.masked_invalid(n_v),
                   ma.masked_invalid(n_w),
                   ma.masked_invalid(residual),
                   ma.masked_invalid(correlation),
                   ma.masked_invalid(mean_cnr),
                   ma.masked_invalid(wspd),
                   ma.masked_invalid(wdir),
                   window,
                   span)

    @classmethod
    def from_file(cls, filename: str):
        """ Create a ConsensusSet object from a daily consensus netcdf file """
        f = netCDF4.Dataset(filename, 'r')
        stime = list(netCDF4.num2date(f.variables['time'][:],
                                      f.variables['time'].units,
                                      only_use_python_datetimes=True,
                                      only_use_cftime_datetimes=False))
        # reconstitute endtime from scan duration
        # make dates tz aware
        stime = [s.replace(tzinfo=pytz.utc) for s in stime]
        # scalar values with a missing_value get read in as np.ma.MaskedArray
        # if the single value is not masked, but as np.ma.core.MaskedConstant
        # if the single value is masked. Instances of MaskedConstant don't
        # compare well to each other or to other data types, so if the scalar
        # lat/lon/alt is masked, replace it with a null-dimensioned masked
        # array with a value of nan.
        alt = (ma.array(np.nan) if f.variables['alt'][:] is np.ma.masked
               else f.variables['alt'][:])
        lat = (ma.array(np.nan) if f.variables['lat'][:] is np.ma.masked
               else f.variables['lat'][:])
        lon = (ma.array(np.nan) if f.variables['lon'][:] is np.ma.masked
               else f.variables['lon'][:])
        # reconstitute span by finding the difference between subsequent times
        span = stime[1] - stime[0]
        return cls(alt,
                   lat,
                   lon,
                   f.variables['height'][:],
                   stime,
                   ma.array(f.variables['u'][:]),
                   ma.array(f.variables['v'][:]),
                   ma.array(f.variables['w'][:]),
                   ma.array(f.variables['u_npoints'][:]),
                   ma.array(f.variables['v_npoints'][:]),
                   ma.array(f.variables['w_npoints'][:]),
                   ma.array(f.variables['residual'][:]),
                   ma.array(f.variables['correlation'][:]),
                   ma.array(f.variables['mean_snr'][:]),
                   ma.array(f.variables['wind_speed'][:]),
                   ma.array(f.variables['wind_direction'][:]),
                   int(f.consensus_avg_window),
                   span)

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