# pylint: disable=C0103,E1101

import os
import sys
import datetime as dt
import pytz
import numpy as np
import numpy.ma as ma
import netCDF4
import matplotlib.pyplot as plt
from typing import Tuple, List
import json

import iss_lidar.Lidar_functions as Lidar_functions
import iss_lidar.tools as tools
from .ppi import PPI


class VAD:
    """
    This is a class created for VAD data
    """

    @staticmethod
    def xyz(ranges: np.ma.MaskedArray, el: np.ma.MaskedArray,
            az: np.ma.MaskedArray) -> Tuple[np.ma.MaskedArray,
                                            np.ma.MaskedArray,
                                            np.ma.MaskedArray]:
        """ Calculate x, y, and z coordinates from range, elevation, and azimuth
        """
        # [None,:] / [:, None] syntax creates 2d array from 1d range and
        # azimuth
        x = (ranges[None, :] * np.cos(np.radians(el))
             * np.sin(np.radians(az[:, None])))
        y = (ranges[None, :] * np.cos(np.radians(el))
             * np.cos(np.radians(az[:, None])))
        z = ranges * np.sin(np.radians(el))
        return (x, y, z)

    @staticmethod
    def non_nan_idxs(vr: np.ma.MaskedArray, i: int) -> np.ma.MaskedArray:
        """ This variable is used to index azimuths, but I'm not really sure why
        """
        return np.argwhere(~np.isnan(vr[:, i])).flatten()

    @staticmethod
    def calc_A(el: np.ma.MaskedArray, az: np.ma.MaskedArray,
               idxs: np.ndarray) -> np.ndarray:
        """ Calculate contents of A matrix """
        A11 = ((np.cos(np.deg2rad(el))**2)
               * np.sum(np.sin(np.deg2rad(az[idxs]))**2))
        A12 = ((np.cos(np.deg2rad(el))**2)
               * np.sum(np.sin(np.deg2rad(az[idxs]))
                        * np.cos(np.deg2rad(az[idxs]))))
        A13 = ((np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el)))
               * np.sum(np.sin(np.deg2rad(az[idxs]))))
        A22 = ((np.cos(np.deg2rad(el))**2)
               * np.sum(np.cos(np.deg2rad(az[idxs]))**2))
        A23 = ((np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el)))
               * np.sum(np.cos(np.deg2rad(az[idxs]))))
        A33 = len(az[idxs]) * (np.sin(np.deg2rad(el))**2)

        A = ma.array([[A11, A12, A13], [A12, A22, A23], [A13, A23, A33]])
        return A

    @staticmethod
    def nan_if_masked(barray: list) -> list:
        """ Not currently used, but leaving for reference """
        return [b if not np.ma.is_masked(b) else np.nan for b in barray]

    @staticmethod
    def calc_b(el: np.ma.MaskedArray, az: np.ma.MaskedArray,
               vr: np.ma.MaskedArray, idxs: np.ndarray, i: int) -> np.ndarray:
        """ Calculate contents of b matrix """
        # If all of the vr[idxs, i] array is masked, then b1, b2, and b3 will
        # be masked, and the np.array() creation will report a warning about
        # converting a masked element to nan.  The best way I could figure out
        # to avoid that warning was to explicitly convert a masked result.
        b1 = np.cos(np.deg2rad(el)) * np.sum(vr[idxs, i] *
                                             np.sin(np.deg2rad(az[idxs])))
        b2 = np.cos(np.deg2rad(el)) * np.sum(vr[idxs, i] *
                                             np.cos(np.deg2rad(az[idxs])))
        b3 = np.sin(np.deg2rad(el)) * np.sum(vr[idxs, i])

        # now using ma arrays in VAD, so no need to convert to np arrays w/ nan
        # b = np.array(VAD.nan_if_masked([b1, b2, b3]))
        b = ma.array([b1, b2, b3])
        return b

    def __init__(self, ppi: PPI, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                 speed: np.ndarray, wdir: np.ndarray, du: np.ndarray,
                 dv: np.ndarray, dw: np.ndarray, nbeams_used: np.ndarray,
                 z: np.ndarray, residual: np.ndarray, correlation: np.ndarray):
        self.u = ma.masked_invalid(u)
        self.v = ma.masked_invalid(v)
        self.w = ma.masked_invalid(w)
        self.speed = ma.masked_invalid(speed)
        self.wdir = ma.masked_invalid(wdir)
        self.du = ma.masked_invalid(du)
        self.dv = ma.masked_invalid(dv)
        self.dw = ma.masked_invalid(dw)
        self.nbeams_used = ma.masked_invalid(nbeams_used)
        self.z = ma.masked_invalid(z)
        self.residual = ma.masked_invalid(residual)
        self.correlation = ma.masked_invalid(correlation)
        self.stime = ppi.starttime
        self.etime = ppi.endtime
        self.el = ppi.elevation
        self.nbeams = len(ppi.azimuth)
        self.alt = ppi.alt
        self.lat = ppi.lat
        self.lon = ppi.lon
        self.mean_cnr = ppi.mean_cnr()

    @classmethod
    def calculate_ARM_VAD(cls, ppi: PPI, missing=None):
        """
        This function calculates VAD wind profiles using the technique shown in
        Newsom et al. (2019). This function calculates VAD output for a single
        PPI scan.
        vr: 2d array (azimuth x range)
        ranges: 1d array
        el: scalar
        az: 1d array
        """
        # remove missing radial vel data
        if missing is not None:
            ppi.vr[ppi.vr == missing] = np.nan

        # calculate XYZ coordinates of data
        x, y, z = VAD.xyz(ppi.ranges, ppi.elevation, ppi.azimuth)

        u = np.ones(len(ppi.ranges))*np.nan
        v = np.ones(len(ppi.ranges))*np.nan
        w = np.ones(len(ppi.ranges))*np.nan
        du = np.ones(len(ppi.ranges))*np.nan
        dv = np.ones(len(ppi.ranges))*np.nan
        dw = np.ones(len(ppi.ranges))*np.nan
        nbeams_used = np.ones(len(ppi.ranges))*np.nan

        for i in range(len(ppi.ranges)):
            idxs = VAD.non_nan_idxs(ppi.vr, i)
            nbeams_used[i] = len(idxs)

            # need at least 25% of the azimuth radial velocities available
            if nbeams_used[i] <= len(ppi.azimuth)/4 or np.isnan(nbeams_used[i]):
                continue

            A = VAD.calc_A(ppi.elevation, ppi.azimuth, idxs)
            invA = np.linalg.inv(A)

            du[i] = np.sqrt(invA[0, 0])
            dv[i] = np.sqrt(invA[1, 1])
            dw[i] = np.sqrt(invA[2, 2])

            b = VAD.calc_b(ppi.elevation, ppi.azimuth, ppi.vr, idxs, i)
            temp = invA.dot(b)

            u[i] = temp[0]
            v[i] = temp[1]
            w[i] = temp[2]

        # calculate derived products
        speed, wdir = tools.wspd_wdir_from_uv(u, v)

        residual = np.sqrt(np.nanmean(((((u*x)+(v*y)+((w*z)
                           [None, :]))/np.sqrt(x**2+y**2+z**2))-ppi.vr)**2,
                                      axis=0))
        u_dot_r = ((u*x)+(v*y)+((w*z)[None, :]))/np.sqrt(x**2+y**2+z**2)
        mean_u_dot_r = np.nanmean(((u*x)+(v*y)+((w*z)[None, :])) /
                                  np.sqrt(x**2+y**2+z**2), axis=0)
        mean_vr = np.nanmean(ppi.vr, axis=0)
        correlation = (np.nanmean((u_dot_r-mean_u_dot_r)*(ppi.vr-mean_vr),
                                  axis=0) /
                       (np.sqrt(np.nanmean((u_dot_r-mean_u_dot_r)**2, axis=0))
                        * np.sqrt(np.nanmean((ppi.vr-mean_vr)**2, axis=0))))

        return cls(ppi, u, v, w, speed, wdir, du, dv, dw, nbeams_used,
                   z, residual, correlation)

    def plot_(self, filename: str, plot_time: int = None, title: str = None):
        """
        This function will plot wind profiles from the VAD object
        """

        if plot_time is None:
            plot_time = len(self.time) - 1
            start = 0
        else:
            start = plot_time

        for i in range(start, plot_time+1):
            plt.figure(figsize=(12, 8))
            ax = plt.subplot(141)
            plt.plot(self.speed[i], self.z)
            ax.set_xlim(0, 30)
            ax.set_ylim([0, np.max(self.z)])
            ax.set_xlabel('Wind Speed [m/s]')
            ax.set_ylabel('Height [m]')

            ax = plt.subplot(142)
            plt.plot(self.wdir[i], self.z)
            ax.set_xlim([0, 360])
            ax.set_ylim([0, np.max(self.z)])
            ax.set_xlabel('Wind Direction')

            ax = plt.subplot(143)
            plt.plot(self.w[i], self.z)
            ax.set_xlim(-5, 5)
            ax.set_ylim([0, np.max(self.z)])
            ax.set_xlabel('Vertical Velocity [m/s]')

            ax = plt.subplot(144)
            plt.plot(self.residual[i], self.z)
            ax.set_xlim(0, 10)
            ax.set_ylim([0, np.max(self.z)])
            ax.set_xlabel('Residual')

            if title is not None:
                plt.title(title)

            plt.tight_layout()

            if (start == 0) and (len(self.time) > 1):
                if os.path.isdir(filename):
                    plt.savefig(filename + str(self.time[i]))
                else:
                    os.mkdir(filename)
                    plt.savefig(filename + str(self.time[i]))
            else:
                plt.savefig(filename)

            plt.close()

    def plot_comp(self, altitude: float, ws: np.ndarray, wd: np.ndarray,
                  w: np.ndarray, z: np.ndarray, wprof_alt: float,
                  filename: str, plot_time: int = None, title: str = None):
        """
        This function will plot wind profiles from the VAD object and
        Comparison to other ws,wd,w,z
        """
        if plot_time is None:
            plot_time = len(self.time) - 1
            start = 0
        else:
            start = plot_time

        for i in range(start, plot_time+1):
            plt.figure(figsize=(12, 8))
            ax = plt.subplot(141)
            plt.plot(self.speed[i], self.z+altitude, label='Lidar VAD')
            if len(ws) > 0:
                plt.plot(ws, z+wprof_alt, color='r', label='Wind Profiler')
            plt.legend(loc='upper right')
            ax.set_xlim(0, 30)
            ax.set_ylim([0, 3000])
            ax.set_xlabel('Wind Speed [m/s]')
            ax.set_ylabel('Height [m]')

            ax = plt.subplot(142)
            plt.plot(self.wdir[i], self.z+altitude, label='Lidar VAD')
            if len(wd) > 0:
                plt.plot(wd, z+wprof_alt, color='r', label='Wind Profiler')
            plt.legend(loc='upper right')
            ax.set_xlim([0, 360])
            ax.set_ylim([0, 3000])
            ax.set_xlabel('Wind Direction')

            ax = plt.subplot(143)
            plt.plot(self.w[i], self.z+altitude, label='Lidar VAD')
            if len(w) > 0:
                plt.plot(w, z+wprof_alt, color='r', label='Wind Profiler')
            plt.legend(loc='upper right')
            ax.set_xlim(-5, 5)
            ax.set_ylim([0, 3000])
            ax.set_xlabel('Vertical Velocity [m/s]')

            ax = plt.subplot(144)
            plt.plot(self.residual[i], self.z+altitude)
            ax.set_xlim(0, 10)
            ax.set_ylim([0, 3000])
            ax.set_xlabel('Residual')

            if title is not None:
                plt.title(title)

            plt.tight_layout()

            if (start == 0) and (len(self.time) > 1):
                if os.path.isdir(filename):
                    plt.savefig(filename + str(self.time[i]))
                else:
                    os.mkdir(filename)
                    plt.savefig(filename + str(self.time[i]))
            else:
                plt.savefig(filename)

            plt.close()

    def to_ARM_nc(self, file_path: str, min_cnr: float = -22):
        """
        Create VAD output in ARM netCDF format
        """
        # really a single VAD should have the same netCDF representation as a
        # VADSet with one vad in it, so use that code to generate this netCDF
        vs = VADSet.from_VADs([self], min_cnr)
        vs.to_ARM_netcdf(file_path)


class VADSet:
    """ Class to hold data from a series of VAD calculations """

    @staticmethod
    def missing_val_if_nan(val: float, missing_val: float = -9999.0) -> float:
        """ netCDF4 library automatically sets nans to missing in writing to
        array variables, but not scalars. """
        if val is None or np.isnan(val):
            return missing_val
        return val

    def __init__(self, mean_cnr: np.ndarray, min_cnr: int,
                 alt: np.ma.MaskedArray, lat: np.ma.MaskedArray,
                 lon: np.ma.MaskedArray, height: np.ma.MaskedArray,
                 stime: list, etime: list, el: list, nbeams: list,
                 u: np.ndarray, du: np.ndarray, v: np.ndarray,
                 dv: np.ndarray, w: np.ndarray, dw: np.ndarray,
                 nbeams_used: np.ndarray, speed: np.ndarray,
                 wdir: np.ndarray, residual: np.ndarray,
                 correlation: np.ndarray):
        self.mean_cnr = mean_cnr
        self.min_cnr = min_cnr
        self.alt = alt
        self.lat = lat
        self.lon = lon
        self.height = height
        self.stime = stime
        self.etime = etime
        self.el = el
        self.nbeams = nbeams
        self.u = u
        self.v = v
        self.w = w
        self.du = du
        self.dv = dv
        self.dw = dw
        self.nbeams_used = nbeams_used
        self.speed = speed
        self.wdir = wdir
        self.residual = residual
        self.correlation = correlation
        # use load_thresholds to add thresholds dict
        self.thresholds = None

    @classmethod
    def from_VADs(cls, vads: List[VAD], min_cnr: int):
        # avoid using the heights from a VAD with bad heights
        i = 0
        while (vads[i].z[0] == 0 and i < len(vads)):
            i += 1
        z = vads[i].z
        return cls(ma.array([i.mean_cnr for i in vads]),
                   min_cnr,
                   # use any vad for location, presumably it doesn't change
                   vads[0].alt,
                   vads[0].lat,
                   vads[0].lon,
                   # currently we assume that heights are the same for all VAD
                   # in a set
                   z,
                   # lists of datetime objects, tz-aware
                   [i.stime for i in vads],
                   [i.etime for i in vads],
                   # data from VAD objects
                   [i.el for i in vads],
                   ma.array([i.nbeams for i in vads]),
                   ma.array([i.u for i in vads]),
                   ma.array([i.du for i in vads]),
                   ma.array([i.v for i in vads]),
                   ma.array([i.dv for i in vads]),
                   ma.array([i.w for i in vads]),
                   ma.array([i.dw for i in vads]),
                   ma.array([i.nbeams_used for i in vads]),
                   ma.array([i.speed for i in vads]),
                   ma.array([i.wdir for i in vads]),
                   ma.array([i.residual for i in vads]),
                   ma.array([i.correlation for i in vads]))

    @classmethod
    def from_file(cls, filename: str):
        """ Create a VADSet object from a daily VAD netcdf file """
        f = netCDF4.Dataset(filename, 'r')
        stime = list(netCDF4.num2date(f.variables['time'][:],
                                      f.variables['time'].units,
                                      only_use_python_datetimes=True,
                                      only_use_cftime_datetimes=False))
        # reconstitute endtime from scan duration
        etime = list(netCDF4.num2date(f.variables['time'][:] +
                                      f.variables['scan_duration'],
                                      f.variables['time'].units,
                                      only_use_python_datetimes=True,
                                      only_use_cftime_datetimes=False))
        # make dates tz aware
        stime = [s.replace(tzinfo=pytz.utc) for s in stime]
        etime = [e.replace(tzinfo=pytz.utc) for e in etime]
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
        min_cnr = None
        try:  # accept min_cnr as either an attribute or a variable
            min_cnr = int(f.input_ppi_min_cnr)
        except AttributeError:
            min_cnr = f.variables['snr_threshold'][:]
        # create blank du, dv, dw if not present in netcdf
        du = np.full(f.variables['u'][:].shape, np.nan)
        dv = np.full(f.variables['v'][:].shape, np.nan)
        dw = np.full(f.variables['w'][:].shape, np.nan)
        try:
            du = f.variables['u_error'][:]
            dv = f.variables['v_error'][:]
            dw = f.variables['w_error'][:]
        except KeyError:
            pass
        return cls(ma.array(f.variables['mean_snr'][:]),
                   min_cnr,
                   alt,
                   lat,
                   lon,
                   f.variables['height'][:],
                   stime,
                   etime,
                   ma.array(f.variables['elevation_angle'][:]),
                   ma.array(f.variables['nbeams'][:]),
                   ma.array(f.variables['u'][:]),
                   ma.array(du),
                   ma.array(f.variables['v'][:]),
                   ma.array(dv),
                   ma.array(f.variables['w'][:]),
                   ma.array(dw),
                   ma.array(f.variables['nbeams_used'][:]),
                   ma.array(f.variables['wind_speed'][:]),
                   ma.array(f.variables['wind_direction'][:]),
                   ma.array(f.variables['residual'][:]),
                   ma.array(f.variables['correlation'][:]))

    @classmethod
    def from_PPIs(cls, ppi_files: List[str], min_cnr: int):
        """ Create a VADSet object by performing VAD calculation on each of a
        list of PPI files """
        vads = []
        for f in ppi_files:
            ppi = PPI.from_file(f)

            # for low elevation angles, VAD output isn't very helpful
            if ppi.elevation < 6:
                continue
            # need at least 3 different azimuths to calculate VAD winds
            if len(ppi.azimuth) < 3:
                continue
            ppi.threshold_cnr(min_cnr)

            # generate VAD for this timestep
            vad = VAD.calculate_ARM_VAD(ppi)
            vads.append(vad)
        if not vads:
            # didn't successfully create any vads. can't continue processing.
            return

        return cls.from_VADs(vads, min_cnr)

    def __eq__(self, other):
        # for testing.
        if (np.array_equal(self.mean_cnr, other.mean_cnr)
            and np.array_equal(self.min_cnr, other.min_cnr)
            and np.array_equal(self.alt, other.alt, equal_nan=True)
            and np.array_equal(self.lat, other.lat, equal_nan=True)
            and np.array_equal(self.lon, other.lon, equal_nan=True)
            and np.array_equal(self.height, other.height, equal_nan=True)
            and np.array_equal(self.stime, other.stime)
            and np.array_equal(self.etime, other.etime)
            and np.array_equal(self.el, other.el, equal_nan=True)
            and np.array_equal(self.nbeams, other.nbeams, equal_nan=True)
            and np.array_equal(self.u, other.u, equal_nan=True)
            and np.array_equal(self.v, other.v, equal_nan=True)
            and np.array_equal(self.w, other.w, equal_nan=True)
            and np.array_equal(self.du, other.du, equal_nan=True)
            and np.array_equal(self.dv, other.dv, equal_nan=True)
            and np.array_equal(self.dw, other.dw, equal_nan=True)
            and np.array_equal(self.nbeams_used, other.nbeams_used,
                               equal_nan=True)
            and np.array_equal(self.speed, other.speed, equal_nan=True)
            and np.array_equal(self.wdir, other.wdir, equal_nan=True)
            and np.array_equal(self.residual, other.residual, equal_nan=True)
            and np.array_equal(self.correlation, other.correlation,
                               equal_nan=True)):
            return True
        return False

    def load_thresholds(self, config: str):
        """ Read dict of thresholding values from the given json file path. """
        # sample threshold config contents:
        # {
        #     "correlation_min": 0.3,
        #     "residual_max": 10.0,
        #     "mean_snr_min": -29.0,
        #     "nbeams_used_min_fraction": 0.5
        # }
        # note: mean_snr name used here instead of min_cnr so it will match
        # what is in netcdf input/output. really the quantity is cnr, and in
        # the VAD objects it's correctly named mean_cnr
        self.thresholds = json.load(open(config, "r"))

    @staticmethod
    def get_nbeams_used_mask(nbeams: List, nbeams_used: np.array,
                             threshold: float) -> np.array:
        """ Since total nbeams can vary between scans, we give the threshold as
        a fraction of possible nbeams that should be present. """
        # separate out this function for testing
        # nbeams_used = time x height array
        # nbeams is just list of length times. broadcast it to an array of
        # time x height
        nbeams = ma.repeat(ma.array([nbeams]), nbeams_used.shape[-1],
                           axis=0).swapaxes(1, 0)
        fraction = nbeams_used / nbeams
        fraction = ma.masked_where(fraction < threshold,
                                   fraction)
        return ma.getmaskarray(fraction)

    def get_mask(self) -> np.array:
        """ Go through all variables to threshold on, combine into a single
        mask that can be applied to all variables. Return the mask, in the form
        of a boolean numpy array where true indicates the value is masked"""
        masks = []
        # create base mask of all false (so, will not mask any values) assuming
        # all time-height arrays will have same dimensions, so just use
        # correlation to get their shape
        final_mask = np.full(self.correlation.shape, False)
        if "correlation_min" in self.thresholds:
            tmp = ma.masked_where(self.correlation
                                  < self.thresholds["correlation_min"],
                                  self.correlation)
            masks.append(ma.getmaskarray(tmp))
        if "residual_max" in self.thresholds:
            tmp = ma.masked_where(self.residual
                                  > self.thresholds["residual_max"],
                                  self.residual)
            masks.append(ma.getmaskarray(tmp))
        if "mean_snr_min" in self.thresholds:
            tmp = ma.masked_where(self.mean_cnr
                                  < self.thresholds["mean_snr_min"],
                                  self.mean_cnr)
            masks.append(ma.getmaskarray(tmp))
        if "nbeams_used_min_fraction" in self.thresholds:
            mask = VADSet.get_nbeams_used_mask(self.nbeams, self.nbeams_used,
                                               self.thresholds["nbeams_used_mi"
                                                               "n_fraction"])
            masks.append(mask)

        # combine all masks
        for m in masks:
            final_mask = ma.mask_or(final_mask, m)
        return final_mask

    def apply_thresholds(self, mask: np.array = None):
        """ Apply thresholding for max/min values of different variables, if
        present. Option to supply a mask is for testing, shoud otherwise use
        get_mask() to mask by thresholds"""
        if self.thresholds is None:
            # can't apply thresholds if they're not present in the object.
            return
        if mask is None:
            mask = self.get_mask()
        # apply mask to values. use mask_or to preserve existing masked vals.
        # wind components
        self.u.mask = ma.mask_or(ma.getmaskarray(self.u), mask)
        self.v.mask = ma.mask_or(ma.getmaskarray(self.v), mask)
        self.w.mask = ma.mask_or(ma.getmaskarray(self.w), mask)
        self.du.mask = ma.mask_or(ma.getmaskarray(self.du), mask)
        self.dv.mask = ma.mask_or(ma.getmaskarray(self.dv), mask)
        self.dw.mask = ma.mask_or(ma.getmaskarray(self.dw), mask)
        self.speed.mask = ma.mask_or(ma.getmaskarray(self.speed), mask)
        self.wdir.mask = ma.mask_or(ma.getmaskarray(self.wdir), mask)
        # ancillary variables
        self.nbeams_used.mask = ma.mask_or(ma.getmaskarray(self.nbeams_used),
                                           mask)
        self.residual.mask = ma.mask_or(ma.getmaskarray(self.residual), mask)
        self.correlation.mask = ma.mask_or(ma.getmaskarray(self.correlation),
                                           mask)
        self.mean_cnr.mask = ma.mask_or(ma.getmaskarray(self.mean_cnr), mask)

    def consensus_average(self, ranges: list) -> Tuple[ma.array, ma.array,
                                                       ma.array]:
        """ Return consensus averaged u,v,w for 30-min increments starting at
        list of time ranges """
        u_mean = np.zeros((len(ranges), len(self.height)))
        v_mean = np.zeros((len(ranges), len(self.height)))
        w_mean = np.zeros((len(ranges), len(self.height)))

        # for ind_start in range(len(secs)-1):
        for idx, r in enumerate(ranges):
            start = r
            end = start + dt.timedelta(minutes=30)
            thirty_min_ind = [i for i in range(len(self.stime))
                              if self.stime[i] >= start and
                              self.stime[i] < end]

            if len(thirty_min_ind) == 0:
                u_mean[idx, :] = np.nan
                v_mean[idx, :] = np.nan
                w_mean[idx, :] = np.nan
                continue

            u_all = np.array([self.u[i] for i in thirty_min_ind])
            v_all = np.array([self.v[i] for i in thirty_min_ind])
            w_all = np.array([self.w[i] for i in thirty_min_ind])

            for hgt in range(len(self.height)):
                # run consensus averaging with a window of 5 m/s
                u_mean[idx, hgt] = Lidar_functions.consensus_avg(u_all[:, hgt],
                                                                 5)
                v_mean[idx, hgt] = Lidar_functions.consensus_avg(v_all[:, hgt],
                                                                 5)
                w_mean[idx, hgt] = Lidar_functions.consensus_avg(w_all[:, hgt],
                                                                 5)
        return (ma.masked_invalid(u_mean), ma.masked_invalid(v_mean),
                ma.masked_invalid(w_mean))

    def to_ARM_netcdf(self, filepath: str):
        str_start_time = self.stime[0].strftime('%Y-%m-%d %H:%M:%S %Z')
        str_day_start_time = self.stime[0].strftime('%Y-%m-%d')
        # create netcdf file
        nc_file = netCDF4.Dataset(filepath, 'w', format='NETCDF4')
        # create dimensions: time, height, bound
        nc_file.createDimension('time', None)
        # still currently assuming that all files in a VADSet have the same
        # heights
        nc_file.createDimension('height', len(self.height))
        nc_file.createDimension('bound', 2)
        # create time variables
        base_time = nc_file.createVariable('base_time', 'i')
        base_time.string = str_start_time
        base_time.long_name = 'Base time in Epoch'
        base_time.units = 'seconds since 1970-01-01 00:00:00 UTC'
        base_time.ancillary_variables = 'time_offset'
        base_time[:] = netCDF4.date2num(self.stime[0], base_time.units)
        time_offset = nc_file.createVariable('time_offset', 'd', 'time')
        time_offset.long_name = 'Time offset from base_time'
        time_offset.units = 'seconds since ' + str_start_time
        time_offset.ancillary_variables = "base_time"
        time_offset[:] = netCDF4.date2num(self.stime, time_offset.units)
        stimes = nc_file.createVariable('time', 'd', 'time')
        stimes.long_name = 'Time offset from midnight'
        stimes.units = 'seconds since ' + str_day_start_time + ' 00:00:00 UTC'
        stimes.bounds = 'time_bounds'
        stimes[:] = netCDF4.date2num(self.stime, stimes.units)
        time_bounds = nc_file.createVariable('time_bounds', 'd',
                                             ('time', 'bound'))
        time_bounds[:, :] = list(zip(netCDF4.date2num(self.stime,
                                                      base_time.units),
                                     netCDF4.date2num(self.etime,
                                                      base_time.units)))
        # create height variable
        height = nc_file.createVariable('height', 'f', 'height')
        height[:] = self.height
        height.long_name = 'Height above ground level'
        height.units = 'm'
        height.standard_name = 'height'

        # add base vars
        self.add_base_variables(nc_file)
        self.add_aux_variables(nc_file)

        # add basic attributes for all file types
        nc_file.Conventions = 'ARM-1.1'
        nc_file.history = 'created on ' + dt.datetime.utcnow().strftime(
            '%Y/%m/%d %H:%M:%S UTC')
        nc_file.command_line = " ".join(sys.argv)
        # add file type specific attributes
        self.add_attributes(nc_file)

        nc_file.close()

    def add_base_variables(self, nc_file: netCDF4.Dataset):
        """
        Add base wind variables to netCDF
        """
        # location
        lat = nc_file.createVariable('lat', 'f')
        lat.missing_value = -9999.0
        lat[:] = VADSet.missing_val_if_nan(self.lat)
        lat.long_name = 'North latitude'
        lat.units = 'degree_N'
        lat.valid_min = -90
        lat.valid_max = 90
        lat.standard_name = 'latitude'
        lon = nc_file.createVariable('lon', 'f')
        lon.missing_value = -9999.0
        lon[:] = VADSet.missing_val_if_nan(self.lon)
        lon.long_name = 'East longitude'
        lon.units = 'degree_E'
        lon.valid_min = -180
        lon.valid_max = 180
        lon.standard_name = 'longitude'
        alt = nc_file.createVariable('alt', 'f')
        alt.missing_value = -9999.0
        alt[:] = VADSet.missing_val_if_nan(self.alt)
        alt.long_name = 'Altitude above mean sea level'
        alt.units = 'm'
        alt.standard_name = 'altitude'
        # uvw
        u = nc_file.createVariable('u', 'f', ('time', 'height'))
        u.missing_value = -9999.0
        u[:, :] = self.u
        u.long_name = 'Eastward component of wind vector'
        u.units = 'm/s'
        v = nc_file.createVariable('v', 'f', ('time', 'height'))
        v.missing_value = -9999.0
        v[:, :] = self.v
        v.long_name = 'Northward component of wind vector'
        v.units = 'm/s'
        w = nc_file.createVariable('w', 'f', ('time', 'height'))
        w.missing_value = -9999.0
        w[:, :] = self.w
        w.long_name = 'Vertical component of wind vector'
        w.units = 'm/s'
        # wspd/wdir
        wind_speed = nc_file.createVariable('wind_speed', 'f',
                                            ('time', 'height'))
        wind_speed.missing_value = -9999.0
        wind_speed[:, :] = self.speed
        wind_speed.long_name = 'Wind speed'
        wind_speed.units = 'm/s'
        wind_direction = nc_file.createVariable('wind_direction', 'f',
                                                ('time', 'height'))
        wind_direction.missing_value = -9999.0
        wind_direction[:, :] = self.wdir
        wind_direction.long_name = 'Wind direction'
        wind_direction.units = 'degree'
        # residual, correlation, mean_snr (present in both vad + consensus)
        residual = nc_file.createVariable('residual', 'f', ('time', 'height'))
        residual.missing_value = -9999.0
        residual[:, :] = self.residual
        residual.long_name = 'Fit residual'
        residual.units = 'm/s'
        correlation = nc_file.createVariable('correlation', 'f',
                                             ('time', 'height'))
        correlation.missing_value = -9999.0
        correlation[:, :] = self.correlation
        correlation.long_name = 'Fit correlation coefficient'
        correlation.units = 'unitless'
        # put mean cnr in file as mean snr to match ARM format (it is still cnr
        # though)
        mean_snr = nc_file.createVariable('mean_snr', 'f',
                                          ('time', 'height'))
        mean_snr.missing_value = -9999.0
        mean_snr[:, :] = self.mean_cnr
        mean_snr.long_name = ('Signal to noise ratio averaged over nbeams '
                              '(derived from CNR)')
        mean_snr.units = 'unitless'

    def add_aux_variables(self, nc_file: netCDF4.Dataset):
        """
        Add auxiliary variables (generally metadata, that are different between
        VADSet and ConsensusSet)
        """
        # Currently not used because values are incorrect
        # Need to relook at the equations in calcA() 
        # uvw errors
        #u_error = nc_file.createVariable('u_error', 'f', ('time', 'height'))
        #u_error.missing_value = -9999.0
        #u_error[:, :] = self.du
        #u_error.long_name = ('Sampling uncertainty in eastward component of'
        #                     ' wind due to azimuths used assuming 1 m/s'
        #                     ' error in radial velocities')
        #u_error.units = 'm/s'
        #v_error = nc_file.createVariable('v_error', 'f', ('time', 'height'))
        #v_error.missing_value = -9999.0
        #v_error[:, :] = self.dv
        #v_error.long_name = ('Sampling uncertainty in northward component of'
        #                     ' wind due to azimuths used assuming 1 m/s'
        #                     ' error in radial velocities')
        #v_error.units = 'm/s'
        #w_error = nc_file.createVariable('w_error', 'f', ('time', 'height'))
        #w_error.missing_value = -9999.0
        #w_error[:, :] = self.dw
        #w_error.long_name = ('Sampling uncertainty in vertical component of'
        #                     ' wind due to azimuths used assuming 1 m/s'
        #                     ' error in radial velocities')
        #w_error.units = 'm/s'
        # wspd/wdir errors (not currently used)
        # wind_speed_error = nc_file.createVariable('wind_speed_error', 'f',
        #                                           ('time', 'height'))
        # wind_speed_error.missing_value = -9999.0
        # # not currently calculating wind speed error?
        # wind_speed_error[:, :] = wind_speed[:, :] * np.nan
        # wind_speed_error.long_name = 'Wind speed error'
        # wind_speed_error.units = 'm/s'
        # wind_direction_error = nc_file.createVariable('wind_direction_error',
        #                                               'f', ('time', 'height'))
        # wind_direction_error.missing_value = -9999.0
        # wind_direction_error[:, :] = wind_direction[:, :] * np.nan
        # wind_direction_error.long_name = 'Wind direction error'
        # wind_direction_error.units = 'm/s'
        # scan metadata
        scan_duration = nc_file.createVariable('scan_duration', 'f', 'time')
        scan_duration.missing_value = -9999.0
        scan_duration[:] = [(i[0] - i[1]).total_seconds()
                            for i in zip(self.etime, self.stime)]
        scan_duration.long_name = 'PPI scan duration'
        scan_duration.units = 'second'
        elevation_angle = nc_file.createVariable('elevation_angle', 'f',
                                                 'time')
        elevation_angle.missing_value = -9999.0
        elevation_angle[:] = self.el
        elevation_angle.long_name = 'Beam elevation angle'
        elevation_angle.units = 'degree'
        nbeams = nc_file.createVariable('nbeams', 'i', 'time')
        nbeams[:] = self.nbeams
        nbeams.long_name = ('Number of beams (azimuth angles) in each PPI')
        nbeams.units = 'unitless'
        nbeams_used = nc_file.createVariable('nbeams_used', 'i', ('time',
                                                                  'height'))
        nbeams_used.missing_value = -9999.0
        nbeams_used[:, :] = self.nbeams_used
        nbeams_used.long_name = ('Number of beams (azimuth angles) used in '
                                 'wind vector estimations')
        nbeams_used.units = 'unitless'

    def add_attributes(self, nc_file: netCDF4.Dataset):
        """
        Add metadata attributes to netCDF
        """
        nc_file.input_ppi_min_cnr = self.min_cnr
        # add thresholding vals as attributes, if present
        if self.thresholds:
            for t in self.thresholds:
                name = "threshold_" + t
                nc_file.setncattr(name, self.thresholds[t])
