# utilities for reading/creating lidar files (no actual calculations)

import os
import netCDF4
import pytz
from datetime import datetime


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


def test_create_filename():
    # VAD
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/vad"
    fname = destdir + "/VAD_339_20220329.nc"
    dt = datetime(2022, 3, 29)
    assert create_filename(dt, destdir, "VAD", "339") == fname
    # without prefix
    fname = destdir + "/VAD_20220329.nc"
    assert create_filename(dt, destdir, "VAD", None) == fname
    # destdir with trailing /
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/vad/"
    assert create_filename(dt, destdir, "VAD", None) == fname
    # consensus
    destdir = "/scr/rain-isf/apg/projects/swex_2022/iss1/ds/lidar/consensus"
    fname = destdir + "/30min_winds_339_20220329.nc"
    assert create_filename(dt, destdir, "30min_winds", "339") == fname
    # without prefix
    fname = destdir + "/30min_winds_20220329.nc"
    assert create_filename(dt, destdir, "30min_winds", None) == fname
