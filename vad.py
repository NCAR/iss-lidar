# pylint: disable=C0103,E1101

import os
import datetime as dt
import pytz
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

import Lidar_functions

def xyz(ranges, el, az):
    """ Calculate x, y, and z coordinates from range, elevation, and azimuth """
    # [None,:] / [:, None] syntax creates 2d array from 1d range and azimuth
    x = ranges[None, :]*np.cos(np.radians(el))*np.sin(np.radians(az[:, None]))
    y = ranges[None, :]*np.cos(np.radians(el))*np.cos(np.radians(az[:, None]))
    z = ranges*np.sin(np.radians(el))
    return (x, y, z)

def non_nan_idxs(vr, i):
    """ This variable is used to index azimuths, but I'm not really sure why """
    return np.where(~np.isnan(vr[:, i]))[0]

def calc_A(el, az, idxs):
    """ Calculate contents of A matrix """
    A11 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[idxs]))**2)
    A12 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[idxs])) *\
            np.cos(np.deg2rad(az[idxs])))
    A13 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) *\
            np.sum(np.sin(np.deg2rad(az[idxs])))
    A22 = (np.cos(np.deg2rad(el))**2) * np.sum(np.cos(np.deg2rad(az[idxs]))**2)
    A23 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) *\
            np.sum(np.cos(np.deg2rad(az[idxs])))
    A33 = len(az[idxs]) * (np.sin(np.deg2rad(el))**2)

    A = np.array([[A11, A12, A13], [A12, A22, A23], [A13, A23, A33]])
    return A

def calc_b(el, az, vr, idxs, i):
    """ Calculate contents of b matrix """
    b1 = np.cos(np.deg2rad(el)) * np.sum(vr[idxs, i] *\
         np.sin(np.deg2rad(az[idxs])))
    b2 = np.cos(np.deg2rad(el)) * np.sum(vr[idxs, i] *\
         np.cos(np.deg2rad(az[idxs])))
    b3 = np.sin(np.deg2rad(el)) * np.sum(vr[idxs, i])

    b = np.array([b1, b2, b3])
    return b

class VAD:
    """
    This is a class created for VAD data
    """

    def __init__(self, ppi, u, v, w, speed, wdir, du, dv, dw, z, residual, correlation):
        self.u = np.array(u)
        self.v = np.array(v)
        self.w = np.array(w)
        self.speed = np.array(speed)
        self.wdir = np.array(wdir)
        self.du = np.array(du)
        self.dv = np.array(dv)
        self.dw = np.array(dw)
        self.z = z
        self.residual = np.array(residual)
        self.correlation = np.array(correlation)
        self.stime = ppi.starttime
        self.etime = ppi.endtime
        self.el = ppi.elevation
        self.nbeams = len(ppi.azimuth)
        self.alt = ppi.alt
        self.lat = ppi.lat
        self.lon = ppi.lon
        self.mean_cnr = ppi.mean_cnr()

    @classmethod
    def calculate_ARM_VAD(cls, ppi, missing=None):
        """
        This function calculates VAD wind profiles using the technique shown in
        Newsom et al. (2019). This function calculates VAD output for a single PPI scan.
        vr: 2d array (azimuth x range)
        ranges: 1d array
        el: scalar
        az: 1d array
        """
        # remove missing radial vel data
        if missing is not None:
            ppi.vr[ppi.vr == missing] = np.nan
            
        # calculate XYZ coordinates of data
        x,y,z = xyz(ppi.ranges, ppi.elevation, ppi.azimuth)

        u = np.ones(len(ppi.ranges))*np.nan
        v = np.ones(len(ppi.ranges))*np.nan
        w = np.ones(len(ppi.ranges))*np.nan
        du = np.ones(len(ppi.ranges))*np.nan
        dv = np.ones(len(ppi.ranges))*np.nan
        dw = np.ones(len(ppi.ranges))*np.nan

        for i in range(len(ppi.ranges)):
            idxs = non_nan_idxs(ppi.vr, i)

            # need at least 25% of the azimuth radial velocities available
            if len(idxs) <= len(ppi.azimuth)/4:
                u[i] = np.nan
                v[i] = np.nan
                w[i] = np.nan
                continue
                
            A = calc_A(ppi.elevation, ppi.azimuth, idxs)
            invA = np.linalg.inv(A)

            du[i] = invA[0, 0]
            dv[i] = invA[1, 1]
            dw[i] = invA[2, 2]
            
            b = calc_b(ppi.elevation, ppi.azimuth, ppi.vr, idxs, i)
            temp = invA.dot(b)

            u[i] = temp[0]
            v[i] = temp[1]
            w[i] = temp[2]

        # calculate derived products
        speed = np.sqrt(u**2 + v**2)
        wdir = 270 - np.rad2deg(np.arctan2(v, u))
        # mod by 360 to get deg < 360
        wdir = wdir % 360

        residual = np.sqrt(np.nanmean(((((u*x)+(v*y)+((w*z)\
                        [None, :]))/np.sqrt(x**2+y**2+z**2))-ppi.vr)**2, axis=0))
        u_dot_r = ((u*x)+(v*y)+((w*z)[None, :]))/np.sqrt(x**2+y**2+z**2)
        mean_u_dot_r = np.nanmean(((u*x)+(v*y)+((w*z)[None, :]))/\
                       np.sqrt(x**2+y**2+z**2), axis=0)
        mean_vr = np.nanmean(ppi.vr, axis=0)
        correlation = np.nanmean((u_dot_r-mean_u_dot_r)*(ppi.vr-mean_vr), axis=0)/\
                           (np.sqrt(np.nanmean((u_dot_r-mean_u_dot_r)**2, axis=0))*\
                            np.sqrt(np.nanmean((ppi.vr-mean_vr)**2, axis=0)))

        return cls(ppi, u, v, w, speed, wdir, du, dv, dw,
                   z, residual, correlation)


    def plot_(self, filename, plot_time=None, title=None):
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

    def plot_comp(self, altitude, ws, wd, w, z, wprof_alt, filename, plot_time=None, title=None):
        """
        This function will plot wind profiles from the VAD object and Comparison to
        other ws,wd,w,z
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

    def create_ARM_nc(self, mean_cnr, max_cnr, altitude, latitude, longitude, stime, etime,
                      file_path):
        """
        Create VAD output in ARM netCDF format
        """
        str_start_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y-%m-%d %H:%M:%S')
        str_day_start_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y-%m-%d')
        secs_midnight_time = dt.datetime.strptime(str_day_start_time + ' 00:00:00',\
                                                '%Y-%m-%d %H:%M:%S').timestamp()
        nc_file = netCDF4.Dataset(file_path, 'w', format='NETCDF4')
        nc_file.createDimension('time', None)
        nc_file.createDimension('height', len(self.z))
        nc_file.createDimension('bound', 2)
        base_time = nc_file.createVariable('base_time', 'i')
        base_time[:] = stime[0]
        base_time.string = str_start_time
        base_time.long_name = 'Base time in Epoch'
        base_time.units = 'seconds since 1970-01-01 00:00:00'
        base_time.ancillary_variables = 'time_offset'
        time_offset = nc_file.createVariable('time_offset', 'd', 'time')
        time_offset[:] = np.array(stime) - stime[0]
        time_offset.long_name = 'Time offset from base_time'
        time_offset.units = 'seconds since '+ str_start_time
        time_offset.ancillary_variables = "base_time"
        stimes = nc_file.createVariable('time', 'd', 'time')
        stimes[:] = np.array(stime) - secs_midnight_time
        stimes.long_name = 'Time offset from midnight'
        stimes.units = 'seconds since ' + str_day_start_time + ' 00:00:00'
        stimes.bounds = 'time_bounds'
        time_bounds = nc_file.createVariable('time_bounds', 'd', ('time', 'bound'))
        time_bounds[:, :] = list(zip(stime, etime))
        height = nc_file.createVariable('height', 'f', 'height')
        height[:] = self.z
        height.long_name = 'Height above ground level'
        height.units = 'm'
        height.standard_name = 'height'
        scan_duration = nc_file.createVariable('scan_duration', 'f', 'time')
        scan_duration[:] = np.subtract(etime, stime)
        scan_duration.long_name = 'PPI scan duration'
        scan_duration.units = 'second'
        elevation_angle = nc_file.createVariable('elevation_angle', 'f', 'time')
        elevation_angle[:] = self.el
        elevation_angle.long_name = 'Beam elevation angle'
        elevation_angle.units = 'degree'
        nbeams = nc_file.createVariable('nbeams', 'i', 'time')
        nbeams[:] = self.nbeams
        nbeams.long_name = 'Number of beams (azimuth angles) used in wind vector estimations'
        nbeams.units = 'unitless'
        u = nc_file.createVariable('u', 'f', ('time', 'height'))
        u[:, :] = self.u
        u.long_name = 'Eastward component of wind vector'
        u.units = 'm/s'
        u_error = nc_file.createVariable('u_error', 'f', ('time', 'height'))
        u_error[:, :] = self.du
        u_error.long_name = 'Estimated error in eastward component of wind vector'
        u_error.units = 'm/s'
        v = nc_file.createVariable('v', 'f', ('time', 'height'))
        v[:, :] = self.v
        v.long_name = 'Northward component of wind vector'
        v.units = 'm/s'
        v_error = nc_file.createVariable('v_error', 'f', ('time', 'height'))
        v_error[:, :] = self.dv
        v_error.long_name = 'Estimated error in northward component of wind vector'
        v_error.units = 'm/s'
        w = nc_file.createVariable('w', 'f', ('time', 'height'))
        w[:, :] = self.w
        w.long_name = 'Vertical component of wind vector'
        w.units = 'm/s'
        w_error = nc_file.createVariable('w_error', 'f', ('time', 'height'))
        w_error[:, :] = self.dw
        w_error.long_name = 'Estimated error in vertical component of wind vector'
        w_error.units = 'm/s'
        wind_speed = nc_file.createVariable('wind_speed', 'f', ('time', 'height'))
        wind_speed[:, :] = self.speed
        wind_speed.long_name = 'Wind speed'
        wind_speed.units = 'm/s'
        wind_speed_error = nc_file.createVariable('wind_speed_error', 'f', ('time', 'height'))
        wind_speed_error[:, :] = [np.zeros(len(self.z))*np.nan]
        wind_speed_error.long_name = 'Wind speed error'
        wind_speed_error.units = 'm/s'
        wind_direction = nc_file.createVariable('wind_direction', 'f', ('time', 'height'))
        wind_direction[:, :] = self.wdir
        wind_direction.long_name = 'Wind direction'
        wind_direction.units = 'degree'
        wind_direction_error = nc_file.createVariable('wind_direction_error', 'f', ('time',
                                                                                    'height'))
        wind_direction_error[:, :] = [np.zeros(len(self.z))*np.nan]
        wind_direction_error.long_name = 'Wind direction error'
        wind_direction_error.units = 'm/s'
        residual = nc_file.createVariable('residual', 'f', ('time', 'height'))
        residual[:, :] = self.residual
        residual.long_name = 'Fit residual'
        residual.units = 'm/s'
        correlation = nc_file.createVariable('correlation', 'f', ('time', 'height'))
        correlation[:, :] = self.correlation
        correlation.long_name = 'Fit correlation coefficient'
        correlation.units = 'unitless'
        mean_snr = nc_file.createVariable('mean_snr', 'f', ('time', 'height'))
        mean_snr[:, :] = mean_cnr
        mean_snr.long_name = 'Signal to noise ratio averaged over nbeams'
        mean_snr.units = 'unitless'
        snr_threshold = nc_file.createVariable('snr_threshold', 'f')
        snr_threshold[:] = max_cnr
        snr_threshold.long_name = 'SNR threshold'
        snr_threshold.units = 'unitless'
        lat = nc_file.createVariable('lat', 'f')
        lat[:] = latitude
        lat.long_name = 'North latitude'
        lat.units = 'degree_N'
        lat.valid_min = -90
        lat.valid_max = 90
        lat.standard_name = 'latitude'
        lon = nc_file.createVariable('lon', 'f')
        lon[:] = longitude
        lon.long_name = 'East longitude'
        lon.units = 'degree_E'
        lon.valid_min = -180
        lon.valid_max = 180
        lon.standard_name = 'longitude'
        alt = nc_file.createVariable('alt', 'f')
        alt[:] = altitude
        alt.long_name = 'Altitude above mean sea level'
        alt.units = 'm'
        alt.standard_name = 'altitude'

        nc_file.Conventions = 'ARM-1.1'
        nc_file.history = 'created on ' + dt.datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S UTC')

        nc_file.close()

class VADSet:
    """ Class to hold data from a series of VAD calculations """

    def __init__(self, mean_cnr, min_cnr, alt, lat, lon, height,
                 stime,etime, el, nbeams, u, du, v, dv, w, dw, speed, wdir,
                 residual, correlation):
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
        self.speed = speed
        self.wdir = wdir
        self.residual = residual
        self.correlation = correlation
    
    @classmethod
    def from_VADs(cls, vads, min_cnr):
        return cls(np.array([i.mean_cnr for i in vads]),
                   min_cnr,
                   # use any vad for location, presumably it doesn't change
                   vads[0].alt,
                   vads[0].lat,
                   vads[0].lon,
                   # currently we assume that heights are the same for all VAD in a set
                   vads[0].z,
                   # lists of datetime objects, tz-aware
                   [i.stime for i in vads],
                   [i.etime for i in vads],
                   # data from VAD objects
                   [i.el for i in vads],
                   [i.nbeams for i in vads],
                   np.array([i.u for i in vads]),
                   np.array([i.du for i in vads]),
                   np.array([i.v for i in vads]),
                   np.array([i.dv for i in vads]),
                   np.array([i.w for i in vads]),
                   np.array([i.dw for i in vads]),
                   np.array([i.speed for i in vads]),
                   np.array([i.wdir for i in vads]),
                   np.array([i.residual for i in vads]),
                   np.array([i.correlation for i in vads]))

    @classmethod
    def from_file(cls, filename):
        """ Create a VADSet object from a daily VAD netcdf file """
        f = netCDF4.Dataset(filename, 'r')
        stime = list(netCDF4.num2date(f.variables['time'][:], f.variables['time'].units, only_use_python_datetimes=True, only_use_cftime_datetimes=False))
        # reconstitute endtime from scan duration
        etime = list(netCDF4.num2date(f.variables['time'][:] + f.variables['scan_duration'], f.variables['time'].units, only_use_python_datetimes=True, only_use_cftime_datetimes=False))
        # make dates tz aware
        stime = [s.replace(tzinfo=pytz.utc) for s in stime]
        etime = [e.replace(tzinfo=pytz.utc) for e in etime]
        # all vars are MaskedArray, cast back to what they originally were
        return cls(np.array(f.variables['mean_snr'][:]),
                   int(f.variables['snr_threshold'][:]),
                   f.variables['alt'][:],
                   f.variables['lat'][:],
                   f.variables['lon'][:],
                   f.variables['height'][:],
                   stime,
                   etime,
                   list(f.variables['elevation_angle'][:]),
                   list(f.variables['nbeams'][:]),
                   np.array(f.variables['u'][:]),
                   np.array(f.variables['u_error'][:]),
                   np.array(f.variables['v'][:]),
                   np.array(f.variables['v_error'][:]),
                   np.array(f.variables['w'][:]),
                   np.array(f.variables['w_error'][:]),
                   np.array(f.variables['wind_speed'][:]),
                   np.array(f.variables['wind_direction'][:]),
                   np.array(f.variables['residual'][:]),
                   np.array(f.variables['correlation'][:]))

    def consensus_average(self, ranges):
        """ Return consensus averaged u,v,w for 30-min increments starting at list of time ranges """
        u_mean = np.zeros((len(ranges),len(self.height)))
        v_mean = np.zeros((len(ranges),len(self.height)))
        w_mean = np.zeros((len(ranges),len(self.height)))

        #for ind_start in range(len(secs)-1):
        for idx, r in enumerate(ranges):
            start = r
            end = start + dt.timedelta(minutes=30) 
            thirty_min_ind = [i for i in range(len(self.stime))\
                              if self.stime[i] >= start and\
                              self.stime[i] < end]

            if len(thirty_min_ind) == 0:
                u_mean[idx,:] = np.nan
                v_mean[idx,:] = np.nan
                w_mean[idx,:] = np.nan
                continue

            u_all = np.array([self.u[i] for i in thirty_min_ind]) 
            v_all = np.array([self.v[i] for i in thirty_min_ind]) 
            w_all = np.array([self.w[i] for i in thirty_min_ind]) 

            for hgt in range(len(self.height)):
            # run consensus averaging with a window of 5 m/s
                u_mean[idx,hgt] = Lidar_functions.consensus_avg(u_all[:,hgt],5)  
                v_mean[idx,hgt] = Lidar_functions.consensus_avg(v_all[:,hgt],5)  
                w_mean[idx,hgt] = Lidar_functions.consensus_avg(w_all[:,hgt],5)  
        return (u_mean, v_mean, w_mean)

    def to_ARM_netcdf(self, filepath):
        str_start_time = self.stime[0].strftime('%Y-%m-%d %H:%M:%S %Z')
        str_day_start_time = self.stime[0].strftime('%Y-%m-%d')
        secs_midnight_time = dt.datetime.strptime(str_day_start_time + ' 00:00:00',\
                                                '%Y-%m-%d %H:%M:%S').timestamp()
        nc_file = netCDF4.Dataset(filepath, 'w', format='NETCDF4')
        nc_file.createDimension('time', None)
        # still currently assuming that all files in a VADSet have the same heights
        nc_file.createDimension('height', len(self.height))
        nc_file.createDimension('bound', 2)
        base_time = nc_file.createVariable('base_time', 'i')
        base_time.string = str_start_time
        base_time.long_name = 'Base time in Epoch'
        base_time.units = 'seconds since 1970-01-01 00:00:00 UTC'
        base_time.ancillary_variables = 'time_offset'
        base_time[:] = netCDF4.date2num(self.stime[0], base_time.units)
        time_offset = nc_file.createVariable('time_offset', 'd', 'time')
        time_offset.long_name = 'Time offset from base_time'
        time_offset.units = 'seconds since '+ str_start_time
        time_offset.ancillary_variables = "base_time"
        time_offset[:] = netCDF4.date2num(self.stime, time_offset.units)
        stimes = nc_file.createVariable('time', 'd', 'time')
        stimes.long_name = 'Time offset from midnight'
        stimes.units = 'seconds since ' + str_day_start_time + ' 00:00:00 UTC'
        stimes.bounds = 'time_bounds'
        stimes[:] = netCDF4.date2num(self.stime, stimes.units)
        time_bounds = nc_file.createVariable('time_bounds', 'd', ('time', 'bound'))
        time_bounds[:, :] = list(zip(netCDF4.date2num(self.stime, base_time.units), netCDF4.date2num(self.etime, base_time.units)))
        height = nc_file.createVariable('height', 'f', 'height')
        height[:] = self.height
        height.long_name = 'Height above ground level'
        height.units = 'm'
        height.standard_name = 'height'
        scan_duration = nc_file.createVariable('scan_duration', 'f', 'time')
        scan_duration[:] = [(i[0] - i[1]).total_seconds() for i in zip(self.etime, self.stime)]
        scan_duration.long_name = 'PPI scan duration'
        scan_duration.units = 'second'
        elevation_angle = nc_file.createVariable('elevation_angle', 'f', 'time')
        elevation_angle[:] = self.el
        elevation_angle.long_name = 'Beam elevation angle'
        elevation_angle.units = 'degree'
        nbeams = nc_file.createVariable('nbeams', 'i', 'time')
        nbeams[:] = self.nbeams
        nbeams.long_name = 'Number of beams (azimuth angles) used in wind vector estimations'
        nbeams.units = 'unitless'
        u = nc_file.createVariable('u', 'f', ('time', 'height'))
        u[:, :] = self.u
        u.long_name = 'Eastward component of wind vector'
        u.units = 'm/s'
        u_error = nc_file.createVariable('u_error', 'f', ('time', 'height'))
        u_error[:, :] = self.du
        u_error.long_name = 'Estimated error in eastward component of wind vector'
        u_error.units = 'm/s'
        v = nc_file.createVariable('v', 'f', ('time', 'height'))
        v[:, :] = self.v
        v.long_name = 'Northward component of wind vector'
        v.units = 'm/s'
        v_error = nc_file.createVariable('v_error', 'f', ('time', 'height'))
        v_error[:, :] = self.dv
        v_error.long_name = 'Estimated error in northward component of wind vector'
        v_error.units = 'm/s'
        w = nc_file.createVariable('w', 'f', ('time', 'height'))
        w[:, :] = self.w
        w.long_name = 'Vertical component of wind vector'
        w.units = 'm/s'
        w_error = nc_file.createVariable('w_error', 'f', ('time', 'height'))
        w_error[:, :] = self.dw
        w_error.long_name = 'Estimated error in vertical component of wind vector'
        w_error.units = 'm/s'
        wind_speed = nc_file.createVariable('wind_speed', 'f', ('time', 'height'))
        wind_speed[:, :] = self.speed
        wind_speed.long_name = 'Wind speed'
        wind_speed.units = 'm/s'
        wind_speed_error = nc_file.createVariable('wind_speed_error', 'f', ('time', 'height'))
        # not currently calculating wind speed error?
        wind_speed_error[:, :] = wind_speed[:,:] * np.nan
        wind_speed_error.long_name = 'Wind speed error'
        wind_speed_error.units = 'm/s'
        wind_direction = nc_file.createVariable('wind_direction', 'f', ('time', 'height'))
        wind_direction[:, :] = self.wdir
        wind_direction.long_name = 'Wind direction'
        wind_direction.units = 'degree'
        wind_direction_error = nc_file.createVariable('wind_direction_error', 'f', ('time',
                                                                                    'height'))
        wind_direction_error[:, :] = wind_direction[:,:] * np.nan
        wind_direction_error.long_name = 'Wind direction error'
        wind_direction_error.units = 'm/s'
        residual = nc_file.createVariable('residual', 'f', ('time', 'height'))
        residual[:, :] = self.residual
        residual.long_name = 'Fit residual'
        residual.units = 'm/s'
        correlation = nc_file.createVariable('correlation', 'f', ('time', 'height'))
        correlation[:, :] = self.correlation
        correlation.long_name = 'Fit correlation coefficient'
        correlation.units = 'unitless'
        mean_snr = nc_file.createVariable('mean_snr', 'f', ('time', 'height'))
        mean_snr[:, :] = self.mean_cnr
        mean_snr.long_name = 'Signal to noise ratio averaged over nbeams'
        mean_snr.units = 'unitless'
        snr_threshold = nc_file.createVariable('snr_threshold', 'f')
        snr_threshold[:] = self.min_cnr
        snr_threshold.long_name = 'SNR threshold'
        snr_threshold.units = 'unitless'
        lat = nc_file.createVariable('lat', 'f')
        lat[:] = self.lat
        lat.long_name = 'North latitude'
        lat.units = 'degree_N'
        lat.valid_min = -90
        lat.valid_max = 90
        lat.standard_name = 'latitude'
        lon = nc_file.createVariable('lon', 'f')
        lon[:] = self.lon
        lon.long_name = 'East longitude'
        lon.units = 'degree_E'
        lon.valid_min = -180
        lon.valid_max = 180
        lon.standard_name = 'longitude'
        alt = nc_file.createVariable('alt', 'f')
        alt[:] = self.alt
        alt.long_name = 'Altitude above mean sea level'
        alt.units = 'm'
        alt.standard_name = 'altitude'

        nc_file.Conventions = 'ARM-1.1'
        nc_file.history = 'created on ' + dt.datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S UTC')

        nc_file.close()

