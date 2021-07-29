# pylint: disable=C0103,E1101

import os
import datetime as dt
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

def xyz(ranges, el, az):
    """ Calculate x, y, and z coordinates from range, elevation, and azimuth """
    # [None,:] / [:, None] syntax creates 2d array from 1d range and azimuth
    x = ranges[None, :]*np.cos(np.radians(el))*np.sin(np.radians(az[:, None]))
    y = ranges[None, :]*np.cos(np.radians(el))*np.cos(np.radians(az[:, None]))
    z = ranges*np.sin(np.radians(el))
    return (x, y, z)

def non_nan_idxs(vr, i):
    """ This variable is used to index azimuths, but I'm not really sure why """
    return np.where(~np.isnan(vr[:, i]))
    
    
class VAD:
    """
    This is a class created for VAD data
    """

    def __init__(self, u, v, w, speed, wdir, du, dv, dw, z, residual, correlation, time, el,
                 nbeams):
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
        self.time = time
        self.el = el
        self.nbeams = nbeams

    @classmethod
    def calculate_ARM_VAD(cls, vr, ranges, el, az, time=None, missing=None):
        """
        This function calculates VAD wind profiles using the technique shown in
        Newsom et al. (2019). This function calculates VAD output for a single PPI scan.
        vr: 2d array (azimuth x range)
        ranges: 1d array
        el: scalar
        az: 1d array
        """
        print("starting calculate_arm_vad")
        # remove missing radial vel data
        if missing is not None:
            vr[vr == missing] = np.nan
            
        # calculate XYZ coordinates of data
        x,y,z = xyz(ranges, el, az)

        temp_u = np.ones(len(ranges))*np.nan
        temp_v = np.ones(len(ranges))*np.nan
        temp_w = np.ones(len(ranges))*np.nan
        temp_du = np.ones(len(ranges))*np.nan
        temp_dv = np.ones(len(ranges))*np.nan
        temp_dw = np.ones(len(ranges))*np.nan

        for i in range(len(ranges)):
            idxs = non_nan_idxs(vr, i)

            # need at least 25% of the azimuth radial velocities available
            if len(idxs) <= len(az)/4:
                temp_u[i] = np.nan
                temp_v[i] = np.nan
                temp_w[i] = np.nan
                continue

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
            invA = np.linalg.inv(A)

            temp_du[i] = invA[0, 0]
            temp_dv[i] = invA[1, 1]
            temp_dw[i] = invA[2, 2]

            b1 = np.cos(np.deg2rad(el)) * np.sum(vr[idxs, i] *\
                 np.sin(np.deg2rad(az[idxs])))
            b2 = np.cos(np.deg2rad(el)) * np.sum(vr[idxs, i] *\
                 np.cos(np.deg2rad(az[idxs])))
            b3 = np.sin(np.deg2rad(el)) * np.sum(vr[idxs, i])

            b = np.array([b1, b2, b3])

            temp = invA.dot(b)
            temp_u[i] = temp[0]
            temp_v[i] = temp[1]
            temp_w[i] = temp[2]

        # calculate derived products
        speed = np.sqrt(temp_u**2 + temp_v**2)
        wdir = 270 - np.rad2deg(np.arctan2(temp_v, temp_u))
        # mod by 360 to get deg < 360
        wdir = wdir % 360

        residual = np.sqrt(np.nanmean(((((temp_u*x)+(temp_v*y)+((temp_w*z)\
                        [None, :]))/np.sqrt(x**2+y**2+z**2))-vr)**2, axis=0))
        u_dot_r = ((temp_u*x)+(temp_v*y)+((temp_w*z)[None, :]))/np.sqrt(x**2+y**2+z**2)
        mean_u_dot_r = np.nanmean(((temp_u*x)+(temp_v*y)+((temp_w*z)[None, :]))/\
                       np.sqrt(x**2+y**2+z**2), axis=0)
        mean_vr = np.nanmean(vr, axis=0)
        correlation = np.nanmean((u_dot_r-mean_u_dot_r)*(vr-mean_vr), axis=0)/\
                           (np.sqrt(np.nanmean((u_dot_r-mean_u_dot_r)**2, axis=0))*\
                            np.sqrt(np.nanmean((vr-mean_vr)**2, axis=0)))

        return cls(temp_u, temp_v, temp_w, speed, wdir, temp_du, temp_dv, temp_dw, z, residual, correlation, time, el, len(az))


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
    
    def __init__(self, vads,  mean_cnr, max_cnr, altitude, latitude, longitude, stime, etime):
        self.vads = vads
        self.mean_cnr = mean_cnr
        self.max_cnr = max_cnr
        self.alt = altitude
        self.lat = latitude
        self.lon = longitude
        self.stime = stime
        self.etime = etime

    def to_ARM_netcdf(self, filepath):
        pass # in progress
