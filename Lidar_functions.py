#!/opt/local/anaconda3/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,E1101
"""
Created on Thu Feb  4 11:16:18 2021

@author: jgebauer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import netCDF4
import datetime as dt

##############################################################################
# This is a class created for VAD data
##############################################################################
class VAD:
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

##############################################################################
# This is a class created for gridded RHI data
##############################################################################

class gridded_RHI:
    def __init__(self, field, x, z, dx, offset, grid_el, grid_range, time):
        self.field = np.array(field)
        self.x = np.array(x)
        self.z = np.array(z)
        self.dx = dx
        self.x_offset = offset[0]
        self.z_offset = offset[1]
        self.time = np.array(time)
        self.grid_el = grid_el
        self.grid_range = grid_range

##############################################################################
# This function calculates VAD wind profiles using the technique shown in
# Newsom et al. (2019). This function can calculate one VAD profile or a series
# of VAD profiles depending on the radial velocity input
##############################################################################

def ARM_VAD(radial_vel, ranges, el, az, time=None, missing=None):

    if (time is None) & (len(np.array(radial_vel).shape) == 2):
        times = 1
        time = [0]
        vr = np.array([radial_vel])
    elif (time is None) & (len(np.array(radial_vel).shape) == 3):
        time = np.arange(np.array(radial_vel).shape[0])
        vr = np.copy(radial_vel)
        times = len(time)
    else:
        times = len(time)
        vr = np.copy(radial_vel)

    if missing is not None:
        vr[vr == missing] = np.nan

    x = ranges[None, :]*np.cos(np.radians(el))*np.sin(np.radians(az[:, None]))
    y = ranges[None, :]*np.cos(np.radians(el))*np.cos(np.radians(az[:, None]))
    z = ranges*np.sin(np.radians(el))

    u = []
    v = []
    w = []
    du = []
    dv = []
    dw = []
    residual = []
    speed = []
    wdir = []
    correlation = []

    for j in range(times):
        temp_u = np.ones(len(ranges))*np.nan
        temp_v = np.ones(len(ranges))*np.nan
        temp_w = np.ones(len(ranges))*np.nan
        temp_du = np.ones(len(ranges))*np.nan
        temp_dv = np.ones(len(ranges))*np.nan
        temp_dw = np.ones(len(ranges))*np.nan

        for i in range(len(ranges)):
            foo = np.where(~np.isnan(vr[j, :, i]))[0]

            # need at least 25% of the azimuth radial velocities available
            if len(foo) <= len(az)/4:
                temp_u[i] = np.nan
                temp_v[i] = np.nan
                temp_w[i] = np.nan
                continue

            A11 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[foo]))**2)
            A12 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[foo])) *\
                    np.cos(np.deg2rad(az[foo])))
            A13 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) *\
                    np.sum(np.sin(np.deg2rad(az[foo])))
            A22 = (np.cos(np.deg2rad(el))**2) * np.sum(np.cos(np.deg2rad(az[foo]))**2)
            A23 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) *\
                    np.sum(np.cos(np.deg2rad(az[foo])))
            A33 = len(az[foo]) * (np.sin(np.deg2rad(el))**2)

            A = np.array([[A11, A12, A13], [A12, A22, A23], [A13, A23, A33]])
            invA = np.linalg.inv(A)

            temp_du[i] = invA[0, 0]
            temp_dv[i] = invA[1, 1]
            temp_dw[i] = invA[2, 2]

            b1 = np.cos(np.deg2rad(el)) * np.sum(vr[j, foo, i] *\
                 np.sin(np.deg2rad(az[foo])))
            b2 = np.cos(np.deg2rad(el)) * np.sum(vr[j, foo, i] *\
                 np.cos(np.deg2rad(az[foo])))
            b3 = np.sin(np.deg2rad(el)) * np.sum(vr[j, foo, i])

            b = np.array([b1, b2, b3])

            temp = invA.dot(b)
            temp_u[i] = temp[0]
            temp_v[i] = temp[1]
            temp_w[i] = temp[2]

        u.append(np.copy(temp_u))
        v.append(np.copy(temp_v))
        w.append(np.copy(temp_w))
        du.append(np.copy(temp_du))
        dv.append(np.copy(temp_dv))
        dw.append(np.copy(temp_dw))

        speed.append(np.sqrt(temp_u**2 + temp_v**2))
        temp_wdir = 270 - np.rad2deg(np.arctan2(temp_v, temp_u))

        for k in range(len(temp_wdir)):
            if temp_wdir[k] >= 360:
                temp_wdir[k] = temp_wdir[k] - 360

        wdir.append(temp_wdir)
        residual.append(np.sqrt(np.nanmean(((((temp_u*x)+(temp_v*y)+((temp_w*z)\
                        [None, :]))/np.sqrt(x**2+y**2+z**2))-vr[j])**2, axis=0)))
        u_dot_r = ((temp_u*x)+(temp_v*y)+((temp_w*z)[None, :]))/np.sqrt(x**2+y**2+z**2)
        mean_u_dot_r = np.nanmean(((temp_u*x)+(temp_v*y)+((temp_w*z)[None, :]))/\
                       np.sqrt(x**2+y**2+z**2), axis=0)
        mean_vr = np.nanmean(vr[j], axis=0)
        correlation.append(np.nanmean((u_dot_r-mean_u_dot_r)*(vr[j]-mean_vr), axis=0)/\
                           (np.sqrt(np.nanmean((u_dot_r-mean_u_dot_r)**2, axis=0))*\
                            np.sqrt(np.nanmean((vr[j]-mean_vr)**2, axis=0))))

    return VAD(u, v, w, speed, wdir, du, dv, dw, z, residual, correlation, time, el, len(az))

##############################################################################
# This function will plot wind profiles from VAD objects
##############################################################################

def plot_VAD(vad, filename, plot_time=None, title=None):

    if plot_time is None:
        plot_time = len(vad.time) - 1
        start = 0
    else:
        start = plot_time

    for i in range(start, plot_time+1):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(141)
        plt.plot(vad.speed[i], vad.z)
        ax.set_xlim(0, 30)
        ax.set_ylim([0, np.max(vad.z)])
        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_ylabel('Height [m]')

        ax = plt.subplot(142)
        plt.plot(vad.wdir[i], vad.z)
        ax.set_xlim([0, 360])
        ax.set_ylim([0, np.max(vad.z)])
        ax.set_xlabel('Wind Direction')

        ax = plt.subplot(143)
        plt.plot(vad.w[i], vad.z)
        ax.set_xlim(-5, 5)
        ax.set_ylim([0, np.max(vad.z)])
        ax.set_xlabel('Vertical Velocity [m/s]')

        ax = plt.subplot(144)
        plt.plot(vad.residual[i], vad.z)
        ax.set_xlim(0, 10)
        ax.set_ylim([0, np.max(vad.z)])
        ax.set_xlabel('Residual')

        if title is not None:
            plt.title(title)

        plt.tight_layout()

        if (start == 0) & (len(vad.time) > 1):
            if os.path.isdir(filename):
                plt.savefig(filename + str(vad.time[i]))
            else:
                os.mkdir(filename)
                plt.savefig(filename + str(vad.time[i]))
        else:
            plt.savefig(filename)

        plt.close()

##############################################################################
# This function will plot wind profiles from VAD objects and Comparison to
# other ws,wd,w,z
##############################################################################

def plot_VAD_comp(vad, altitude, ws, wd, w, z, wprof_alt, filename, plot_time=None, title=None):

    if plot_time is None:
        plot_time = len(vad.time) - 1
        start = 0
    else:
        start = plot_time

    for i in range(start, plot_time+1):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(141)
        plt.plot(vad.speed[i], vad.z+altitude, label='Lidar VAD')
        if len(ws) > 0:
            plt.plot(ws, z+wprof_alt, color='r', label='Wind Profiler')
        plt.legend(loc='upper right')
        ax.set_xlim(0, 30)
        ax.set_ylim([0, 3000])
        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_ylabel('Height [m]')

        ax = plt.subplot(142)
        plt.plot(vad.wdir[i], vad.z+altitude, label='Lidar VAD')
        if len(wd) > 0:
            plt.plot(wd, z+wprof_alt, color='r', label='Wind Profiler')
        plt.legend(loc='upper right')
        ax.set_xlim([0, 360])
        ax.set_ylim([0, 3000])
        ax.set_xlabel('Wind Direction')

        ax = plt.subplot(143)
        plt.plot(vad.w[i], vad.z+altitude, label='Lidar VAD')
        if len(w) > 0:
            plt.plot(w, z+wprof_alt, color='r', label='Wind Profiler')
        plt.legend(loc='upper right')
        ax.set_xlim(-5, 5)
        ax.set_ylim([0, 3000])
        ax.set_xlabel('Vertical Velocity [m/s]')

        ax = plt.subplot(144)
        plt.plot(vad.residual[i], vad.z+altitude)
        ax.set_xlim(0, 10)
        ax.set_ylim([0, 3000])
        ax.set_xlabel('Residual')

        if title is not None:
            plt.title(title)

        plt.tight_layout()

        if (start == 0) & (len(vad.time) > 1):
            if os.path.isdir(filename):
                plt.savefig(filename + str(vad.time[i]))
            else:
                os.mkdir(filename)
                plt.savefig(filename + str(vad.time[i]))
        else:
            plt.savefig(filename)

        plt.close()

##############################################################################
# This function will take RHI scans and will put them onto a 2-D cartesian grid
# using linear interpolation
##############################################################################

def grid_rhi(field, elevation, ranges, dims, dx, offset=None,
             time=None, missing=None):

    if len(dims) != 2:
        raise IOError('Dims must be a 2 length tuple')

    if offset is not None:
        if len(offset) != 2:
            raise IOError('If offset is specified it must be 2 length tuple')
    else:
        offset = (0, 0)

    if (time is None) & (len(field.shape) == 2):
        times = 1
        time = [0]
        raw = np.array([field])
        el = np.array([elevation])
    elif (time is None) & (len(field.shape) == 3):
        time = np.arange(field.shape[0])
        el = np.copy(elevation)
        raw = np.copy(field)
        times = len(time)
    else:
        times = len(time)
        raw = np.copy(field)
        el = np.copy(elevation)

    if missing is not None:
        raw[raw == missing] = np.nan

    x = ranges[None, :] * np.cos(np.deg2rad(180-el))[:, :, None] + offset[0]
    z = ranges[None, :] * np.sin(np.deg2rad(el))[:, :, None] + offset[1]

    grid_x, grid_z = np.meshgrid(np.arange(dims[0][0], dims[0][1]+1, dx),
                                 np.arange(dims[1][0], dims[1][1]+1, dx))

    grid_range = np.sqrt((grid_x-offset[0])**2 + (grid_z-offset[1])**2)
    grid_el = 180 - np.rad2deg(np.arctan2(grid_z-offset[1], grid_x-offset[0]))

    grid_field = []
    for i in range(times):
        foo = np.where(~np.isnan(x[i]))

        grid_field.append(scipy.interpolate.griddata((x[i, foo[0], foo[1]], z[i, foo[0], foo[1]]),
                                                     raw[i, foo[0], foo[1]], (grid_x, grid_z)))

    return gridded_RHI(grid_field, grid_x, grid_z, dx, offset, grid_el, grid_range, time)

##############################################################################
# This function calculates a coplanar wind field from two gridded RHIs
##############################################################################

def coplanar_analysis(vr1, vr2, el1, el2, az):

    u = np.ones(vr1.shape)*np.nan
    w = np.ones(vr1.shape)*np.nan

    for i in range(vr1.shape[0]):
        for j in range(vr1.shape[1]):

            if (~np.isnan(vr1[i, j])) & (~np.isnan(vr2[i, j])):
                M = np.array([[np.sin(np.deg2rad(az))*np.cos(np.deg2rad(el1[i, j])),
                               np.sin(np.deg2rad(el1[i, j]))],
                              [np.sin(np.deg2rad(az))*np.cos(np.deg2rad(el2[i, j])),
                               np.sin(np.deg2rad(el2[i, j]))]])

                U = np.linalg.solve(M, np.array([vr1[i, j], vr2[i, j]]))
                u[i, j] = U[0]
                w[i, j] = U[1]

    return u, w

##############################################################################
# The function calculates the vr-variance from a timeseries of scans
##############################################################################

def vr_variance(field, time, t_avg, axis=0):

    t_avg = t_avg*60
    start = 0
    yo = np.where(time < (time[0]+t_avg))[0]
    end = yo[-1]+1
    var = []
    time_avg = []
    while end < len(time):
        var.append(np.nanvar(field[start:end, :, :], axis=0))
        time_avg.append(np.nanmean(time[start:end]))
        start = end
        yo = np.where(time < (time[start]+t_avg))[0]
        end = yo[-1]+1

    return np.array(var), np.array(time_avg)

##############################################################################
# This function will work with LidarSim data
##############################################################################

def process_LidarSim_scan(scan, scantype, elevation, azimuth, ranges, time):

    if scantype == 'vad':
        el = np.nanmean(elevation)
        vad = ARM_VAD(scan, ranges, el, azimuth, time)

        return vad

    else:
        print('Not a valid scan type')
        return np.nan

##############################################################################
# This function puts vr observations from a RHI onto a vertical grid at one
# location to be used for virtual tower calculations
##############################################################################

def rhi_vertical_profile(field, elevation, azimuth, ranges, heights, dz, loc, offset=None,
                         time=None, missing=None):

    if len(loc) != 2:
        raise IOError('Dims must be a 2 length tuple')

    if offset is not None:
        if len(offset) != 3:
            raise IOError('If offset is specified it must be 3 length tuple')
    else:
        offset = (0, 0, 0)

    if (time is None) & (len(field.shape) == 2):
        times = 1
        time = [0]
        raw = np.array([field])
        el = np.array([elevation])
    elif (time is None) & (len(field.shape) == 3):
        time = np.arange(field.shape[0])
        el = np.copy(elevation)
        raw = np.copy(field)
        times = len(time)
    else:
        times = len(time)
        raw = np.copy(field)
        el = np.copy(elevation)

    if missing is not None:
        raw[raw == missing] = np.nan

    x = ranges[None, :] * np.cos(np.deg2rad(el))[:, :, None]*np.sin(np.deg2rad(azimuth)) + offset[0]
    y = ranges[None, :] * np.cos(np.deg2rad(el))[:, :, None]*np.cos(np.deg2rad(azimuth)) + offset[1]
    z = ranges[None, :] * np.sin(np.deg2rad(el))[:, :, None] + offset[2]

    r = np.sqrt(x**2 + y**2)

    z_interp = np.arange(heights[0], heights[1], dz)

    z_ranges = np.sqrt((loc[0]-offset[0])**2 + (loc[1]-offset[1])**2+(z_interp-offset[2])**2)
    z_el = np.rad2deg(np.arctan2(z_interp-offset[2], np.sqrt((loc[0]-offset[0])**2+
                                                             (loc[1]-offset[1])**2)))

    grid_field = []
    grid_x, grid_z = np.meshgrid(np.array(np.sqrt(loc[0]**2 + loc[1]**2)), z_interp)
    for i in range(times):
        grid_field.append(scipy.interpolate.griddata((r[i].ravel(), z[i].ravel()),
                                                     raw[i].ravel(), (grid_x, grid_z))[:, 0])

    return vertical_vr(grid_field, loc[0], loc[1], z_interp, dz, offset, z_el, z_ranges,
                       azimuth, time)

##############################################################################
# This function calculates the wind components for a virtual towers
##############################################################################

def virtual_tower(vr, elevation, azimuth, height, uncertainty=0.45):

    if len(vr) == 2:
        u = []
        v = []
        u_uncertainty = []
        v_uncertainty = []
        el1 = np.deg2rad(elevation[0])
        el2 = np.deg2rad(elevation[1])
        az1 = np.deg2rad(azimuth[0])
        az2 = np.deg2rad(azimuth[1])
        for i in range(len(height)):
            if np.isnan(vr[0][i]) | np.isnan(vr[1][i]):
                u.append(np.nan)
                v.append(np.nan)

                M = np.array([[np.sin(az1)*np.cos(el1[i]), np.cos(az1)*np.cos(el1[i])],
                              [np.sin(az2)*np.cos(el2[i]), np.cos(az2)*np.cos(el2[i])]])
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0, 0]**2)*(uncertainty**2) +
                                 (invM[0, 1]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1, 0]**2)*(uncertainty**2) +
                                 (invM[1, 1]**2)*(uncertainty**2))
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)

            else:
                M = np.array([[np.sin(az1)*np.cos(el1[i]), np.cos(az1)*np.cos(el1[i])],
                              [np.sin(az2)*np.cos(el2[i]), np.cos(az2)*np.cos(el2[i])]])
                temp = np.linalg.solve(M, np.array([vr[0][i], vr[1][i]]))
                u.append(np.copy(temp[0]))
                v.append(np.copy(temp[1]))

                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0, 0]**2)*(uncertainty**2) +
                                 (invM[0, 1]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1, 0]**2)*(uncertainty**2) +
                                 (invM[1, 1]**2)*(uncertainty**2))
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)

        return np.array([u, v]), np.array([u_uncertainty, v_uncertainty])

    elif len(vr) == 3:
        u = []
        v = []
        w = []
        u_uncertainty = []
        v_uncertainty = []
        w_uncertainty = []
        el1 = np.deg2rad(elevation[0])
        el2 = np.deg2rad(elevation[1])
        el3 = np.deg2rad(elevation[2])
        az1 = np.deg2rad(azimuth[0])
        az2 = np.deg2rad(azimuth[1])
        az3 = np.deg2rad(azimuth[2])
        for i in range(len(height)):
            if np.isnan(vr[0][i]) | np.isnan(vr[1][i]):
                u.append(np.nan)
                v.append(np.nan)

                M = np.array([[np.sin(az1)*np.cos(el1[i]), np.cos(az1)*np.cos(el1[i]), np.sin(el1)],
                              [np.sin(az2)*np.cos(el2[i]), np.cos(az2)*np.cos(el2[i]), np.sin(el2)],
                              [np.sin(az3)*np.cos(el3[i]), np.cos(az3)*np.cos(el3[i]),
                               np.sin(el3)]])
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0, 0]**2)*(uncertainty**2) +
                                 (invM[0, 1]**2)*(uncertainty**2) +
                                 (invM[0, 2]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1, 0]**2)*(uncertainty**2) +
                                 (invM[1, 1]**2)*(uncertainty**2) +
                                 (invM[1, 2]**2)*(uncertainty**2))
                temp_w = np.sqrt((invM[2, 0]**2)*(uncertainty**2) +
                                 (invM[2, 1]**2)*(uncertainty**2) +
                                 (invM[2, 2]**2)*(uncertainty**2))

                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                w_uncertainty.append(temp_w)
            else:
                M = np.array([[np.sin(az1)*np.cos(el1[i]), np.cos(az1)*np.cos(el1[i]), np.sin(el1)],
                              [np.sin(az2)*np.cos(el2[i]), np.cos(az2)*np.cos(el2[i]), np.sin(el2)],
                              [np.sin(az3)*np.cos(el3[i]), np.cos(az3)*np.cos(el3[i]),
                               np.sin(el3)]])
                temp = np.linalg.solve(M, np.array([vr[0][i], vr[1][i], vr[2][i]]))
                u.append(np.copy(temp[0]))
                v.append(np.copy(temp[1]))
                w.append(np.copy(temp[2]))

                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0, 0]**2)*(uncertainty**2) +
                                 (invM[0, 1]**2)*(uncertainty**2) +
                                 (invM[0, 2]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1, 0]**2)*(uncertainty**2) +
                                 (invM[1, 1]**2)*(uncertainty**2) +
                                 (invM[1, 2]**2)*(uncertainty**2))
                temp_w = np.sqrt((invM[2, 0]**2)*(uncertainty**2) +
                                 (invM[2, 1]**2)*(uncertainty**2) +
                                 (invM[2, 2]**2)*(uncertainty**2))

                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                w_uncertainty.append(temp_w)

        return np.array([u, v, w]), np.array([u_uncertainty, v_uncertainty, w_uncertainty])
    else:
        print('Input needs to be a length 2 or 3 tuple')
        return np.nan

##############################################################################
# This function performs a Lenshow correction to lidar data and calculates
# vertical velocity variance. This function was originally written by Tyler
# Bell at CIMMS in Norman, OK.
##############################################################################

def lenshow(x, freq=1, tau_min=3, tau_max=12, plot=False):
    """
    Reads in a timeseries. Freq is in Hz. Default taus are from avg values from
    Bonin Dissertation (2015)
    Returns avg w'**2 and avg error'**2
    """
    # Find the perturbation of x
    mean = np.mean(x)
    prime = x - mean
    # Get the autocovariance
    acorr, lags = xcorr(prime, prime)
    var = np.var(prime)
    acov = acorr# * var
    # Extract lags > 0
    lags = lags[int(len(lags)/2):] * freq
    acov = acov[int(len(acov)/2):]
    # Define the start and end lags
    lag_start = int(tau_min / freq)
    lag_end = int(tau_max / freq)
    # Fit the structure function
    fit_funct = lambda p, t: p[0] - p[1]*t**(2./3.)
    err_funct = lambda p, t, y: fit_funct(p, t) - y
    p1, success = leastsq(err_funct, [1, .001], args=(lags[lag_start:lag_end],
                                                      acov[lag_start:lag_end]))
    if plot:
        new_lags = np.arange(tau_min, tau_max)
        plt.plot(lags, acov)
        plt.plot(new_lags, fit_funct(p1, new_lags), 'gX')
        plt.plot(0, fit_funct(p1, 0), 'gX')
        plt.xlim(0, tau_max+20)
        plt.xlabel("Lag [s]")
        plt.ylabel("$M_{11} [m^2s^{-2}$]")
    return p1[0], acov[0] - p1[0]

##############################################################################
# This is a modified version of the Lenshow correction that adaptively selects
# taus based on the data. This function was originally written by Tyler Bell at
# CIMMS in Norman, OK.
###############################################################################

def lenshow_bonin(x, tau_min=1, tint_first_guess=3, freq=1, max_iter=100, plot=False):
    # Find the perturbation of x
    mean = np.mean(x)
    prime = x - mean
    # Get the autocovariance
    acorr, lags = xcorr(prime, prime)
    var = np.var(prime)
    acov = acorr #* var
    # Extract lags > 0
    lags = lags[int(len(lags)/2):] * freq
    acov = acov[int(len(acov)/2):]
    # Define the start and end lags
    lag_start = int(tau_min / freq)
    lag_end = int((tau_min+3) / freq)
    # Fit the structure function
    fit_funct = lambda p, t: p[0] - p[1]*t**(2./3.)
    err_funct = lambda p, t, y: fit_funct(p, t) - y
    # Iterate to find t_int
    last_tint = (tau_min+3)
    i = 0
    p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end],
                                                        acov[lag_start:lag_end]))
    tint = calc_tint(p1[0], freq, acov, lags)
    while np.abs(last_tint - tint) > 1.:
        if i >= max_iter:
            return None
        else:
            i += 1
            last_tint = tint
        p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end],
                                                            acov[lag_start:lag_end]))
        tint = calc_tint(p1[0], freq, acov, lags)
    # Find the time where M11(t) = M11(0)/2
    ind = np.min(np.where(acov <= acov[0]/2))
    # Determine what tau to use
    tau_max = np.min([tint/2., lags[ind]])
    # Do the process
    lag_end = int(tau_max / freq)
    if lag_start+1 >= lag_end:
        lag_end = lag_start + 2
    p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end],
                                                        acov[lag_start:lag_end]))
    if plot:
        new_lags = np.arange(tau_min, tau_max)
        plt.plot(lags, acov, 'k')
        plt.plot(new_lags, fit_funct(p1, new_lags), 'rX', label='Adaptive')
        plt.plot(0, fit_funct(p1, 0), 'rX')
        plt.xlim(0, tau_max+20)
        plt.xlabel("Lag [s]")
        plt.ylabel("$M_{11} [m^2s^{-2}$]")
    return p1[0], np.abs(acov[0] - p1[0]), tau_max

##############################################################################
# This function is used in the lenshow_bonin fuction
##############################################################################

def calc_tint(var, freq, acov, lags):
    ind = np.min(np.where(acov < 0))
    return freq**-1. + 1./var * sum(acov[1:ind] / freq)

##############################################################################
# This function calculates the lag correlation of a time series.
##############################################################################

def xcorr(y1, y2):

    if len(y1) != len(y2):
        raise ValueError('The lenghts of the inputs should be the same')

    corr = np.correlate(y1, y2, mode='full')
    unbiased_size = np.correlate(np.ones(len(y1)), np.ones(len(y1)), mode='full')
    corr = corr/unbiased_size

    maxlags = len(y1)-1
    lags = np.arange(-maxlags, maxlags + 1)

    return corr, lags

##############################################################################
#  Create VAD output in ARM netCDF format
##############################################################################

def VAD_ARM_nc_format(VAD, mean_cnr, max_cnr, altitude, latitude, longitude, stime, etime,
                      file_path):
    str_start_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y-%m-%d %H:%M:%S')
    str_day_start_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y-%m-%d')
    secs_midnight_time = dt.datetime.strptime(str_day_start_time + ' 00:00:00',\
                                            '%Y-%m-%d %H:%M:%S').timestamp()
    nc_file = netCDF4.Dataset(file_path, 'w', format='NETCDF4')
    nc_file.createDimension('time', None)
    nc_file.createDimension('height', len(VAD.z))
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
    height[:] = VAD.z
    height.long_name = 'Height above ground level'
    height.units = 'm'
    height.standard_name = 'height'
    scan_duration = nc_file.createVariable('scan_duration', 'f', 'time')
    scan_duration[:] = np.subtract(etime, stime)
    scan_duration.long_name = 'PPI scan duration'
    scan_duration.units = 'second'
    elevation_angle = nc_file.createVariable('elevation_angle', 'f', 'time')
    elevation_angle[:] = VAD.el
    elevation_angle.long_name = 'Beam elevation angle'
    elevation_angle.units = 'degree'
    nbeams = nc_file.createVariable('nbeams', 'i', 'time')
    nbeams[:] = VAD.nbeams
    nbeams.long_name = 'Number of beams (azimuth angles) used in wind vector estimations'
    nbeams.units = 'unitless'
    u = nc_file.createVariable('u', 'f', ('time', 'height'))
    u[:, :] = VAD.u
    u.long_name = 'Eastward component of wind vector'
    u.units = 'm/s'
    u_error = nc_file.createVariable('u_error', 'f', ('time', 'height'))
    u_error[:, :] = VAD.du
    u_error.long_name = 'Estimated error in eastward component of wind vector'
    u_error.units = 'm/s'
    v = nc_file.createVariable('v', 'f', ('time', 'height'))
    v[:, :] = VAD.v
    v.long_name = 'Northward component of wind vector'
    v.units = 'm/s'
    v_error = nc_file.createVariable('v_error', 'f', ('time', 'height'))
    v_error[:, :] = VAD.dv
    v_error.long_name = 'Estimated error in northward component of wind vector'
    v_error.units = 'm/s'
    w = nc_file.createVariable('w', 'f', ('time', 'height'))
    w[:, :] = VAD.w
    w.long_name = 'Vertical component of wind vector'
    w.units = 'm/s'
    w_error = nc_file.createVariable('w_error', 'f', ('time', 'height'))
    w_error[:, :] = VAD.dw
    w_error.long_name = 'Estimated error in vertical component of wind vector'
    w_error.units = 'm/s'
    wind_speed = nc_file.createVariable('wind_speed', 'f', ('time', 'height'))
    wind_speed[:, :] = VAD.speed
    wind_speed.long_name = 'Wind speed'
    wind_speed.units = 'm/s'
    wind_speed_error = nc_file.createVariable('wind_speed_error', 'f', ('time', 'height'))
    wind_speed_error[:, :] = [np.zeros(len(VAD.z))*np.nan]
    wind_speed_error.long_name = 'Wind speed error'
    wind_speed_error.units = 'm/s'
    wind_direction = nc_file.createVariable('wind_direction', 'f', ('time', 'height'))
    wind_direction[:, :] = VAD.wdir
    wind_direction.long_name = 'Wind direction'
    wind_direction.units = 'degree'
    wind_direction_error = nc_file.createVariable('wind_direction_error', 'f', ('time', 'height'))
    wind_direction_error[:, :] = [np.zeros(len(VAD.z))*np.nan]
    wind_direction_error.long_name = 'Wind direction error'
    wind_direction_error.units = 'm/s'
    residual = nc_file.createVariable('residual', 'f', ('time', 'height'))
    residual[:, :] = VAD.residual
    residual.long_name = 'Fit residual'
    residual.units = 'm/s'
    correlation = nc_file.createVariable('correlation', 'f', ('time', 'height'))
    correlation[:, :] = VAD.correlation
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
##############################################################################
# This function reads in cfradial files to variables needed for VAD analysis
##############################################################################

def read_cfradial(file_path):

    lidar_file = netCDF4.Dataset(file_path, 'r')
    cnr = lidar_file.variables['cnr'][:]
    ranges = lidar_file.variables['range'][:]
    vr = lidar_file.variables['radial_wind_speed'][:]
    azimuth = lidar_file.variables['azimuth'][:]
    elevation = lidar_file.variables['elevation'][0]
    latitude = lidar_file.variables['latitude'][:]
    longitude = lidar_file.variables['longitude'][:]
    altitude = lidar_file.variables['altitude'][:]
    str_start = lidar_file.start_time
    str_end = lidar_file.end_time

    return [cnr, ranges, vr, elevation, azimuth, str_start, str_end, latitude, longitude, altitude]
