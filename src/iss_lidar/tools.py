# utilities for reading/creating lidar files (no actual calculations)

import os
import netCDF4
import pytz
from datetime import datetime
import numpy as np
from typing import Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def create_filename(date: datetime, destdir: str, filetype: str,
                    prefix: str = None) -> str:
    fname = filetype + "_"
    if prefix:
        fname += prefix + "_"
    fname += date.strftime('%Y%m%d') + '.nc'
    return os.path.join(destdir, fname)


def read_cfradial(file_path):
    """
    This function reads in cfradial files to variables needed for VAD analysis
    """
    lidar_file = netCDF4.Dataset(file_path, 'r')
    cnr = lidar_file.variables['cnr'][:]
    ranges = lidar_file.variables['range'][:]
    vr = lidar_file.variables['radial_wind_speed'][:]
    azimuth = lidar_file.variables['azimuth'][:]
    elevation = lidar_file.variables['elevation'][0]
    latitude = lidar_file.variables['latitude'][:]
    longitude = lidar_file.variables['longitude'][:]
    altitude = lidar_file.variables['altitude'][:]
    # convert start/end to datetimes so they're easier to use.
    # times in cfradial files are in UTC. timezone is not specified in
    # start_time attribute, but is specified as 'Z' instead of 'UTC' in
    # start_datetime attribute, which datetime won't parse.
    start = datetime.strptime(lidar_file.start_time,
                              "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=pytz.utc)
    end = datetime.strptime(lidar_file.end_time,
                            "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=pytz.utc)

    return [cnr, ranges, vr, elevation, azimuth, start, end, latitude,
            longitude, altitude]


def wspd_wdir_from_uv(u: np.ndarray,
                      v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # calculate derived products
    speed = np.sqrt(u**2 + v**2)
    wdir = 270 - np.rad2deg(np.arctan2(v, u))
    notnan = ~np.isnan(wdir)
    wdir[notnan] %= 360
    return speed, wdir


def time_height_plot(filepath: str, u_mean: np.ndarray, v_mean: np.ndarray,
                     ranges: np.ndarray, heights: np.ndarray):
    ticklabels = mpl.dates.DateFormatter("%H:%M")
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))
    fig.suptitle('Winds starting at %s 00:00:00 for 24 hours'
                 % (ranges[0].strftime("%Y%m%d")))
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('HH:MM UTC')
    ax.set_ylim(0, 1500)
    ax.xaxis.set_major_formatter(ticklabels)
    # make times and heights 2d arrays
    times = np.repeat([np.array(ranges)],
                      u_mean.shape[-1], axis=0).swapaxes(1, 0)
    heights = np.repeat([heights], u_mean.shape[0], axis=0)
    # colorbar setup
    cmap = plt.cm.gist_rainbow_r
    norm = BoundaryNorm(list(range(0, 24, 2)), cmap.N)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 orientation='vertical', label='wind speed (m/s)')
    # calculate spd from u/v for barb color
    spd, _ = wspd_wdir_from_uv(u_mean, v_mean)
    ax.barbs(times, heights, u_mean, v_mean, spd,
             barb_increments=dict(half=2.5, full=5, flag=10), cmap=cmap,
             norm=norm)
    plt.savefig(filepath)
    plt.close()