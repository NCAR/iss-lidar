#!/opt/local/anaconda3/bin/python
#
# Created June 2021
# Carol Costanza
#
# Output 30 minute consensus averaged VAD winds into ARM netCDF format
# from cfradial format
# Creates the 12 hour plot and netCDF with 30 minute winds
#
# EXAMPLE RUN FROM COMMAND LINE
# ./vad_to_30min_winds.py 'path_to_VAD_nc_file' 'path_30min_nc_file_dest' 'date'

import os
import sys
import argparse
import netCDF4
import pytz
import datetime as dt
import numpy as np
import warnings
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)

import Lidar_functions
from vad import VADSet

def parseArgs():
    parser = argparse.ArgumentParser(description="Generate consensus averaged netcdfs from VAD files")
    parser.add_argument("vadfile", help="daily VAD file")
    parser.add_argument("destdir", help="directory to save averaged file to")
    parser.add_argument("--plot", help="create PNG plot w/ same filename as netcdf", dest="plot", default=False, action='store_true')
    return parser.parse_args()

def create_time_ranges(day):
    """ Create a list of datetimes every 30 minutes for given day """
    start = dt.datetime(day.year, day.month, day.day, tzinfo=pytz.UTC)
    end = start + dt.timedelta(days=1)
    ranges = []
    while ( start < end):
        ranges.append(start)
        start = start + dt.timedelta(minutes=30)
    return ranges

def process(heights, ranges, stimes, u, v, w):
    # loop through the 30 minute time steps to get all the winds
    u_mean = np.zeros((len(ranges),len(heights)))
    v_mean = np.zeros((len(ranges),len(heights)))
    w_mean = np.zeros((len(ranges),len(heights)))

    #for ind_start in range(len(secs)-1):
    for idx, r in enumerate(ranges):
        start = r
        end = start + dt.timedelta(minutes=30) 
        thirty_min_ind = [i for i in range(len(stimes))\
                          if stimes[i] >= start and\
                          stimes[i] < end]

        if len(thirty_min_ind) == 0:
            u_mean[idx,:] = np.nan
            v_mean[idx,:] = np.nan
            w_mean[idx,:] = np.nan
            continue

        u_all = np.array([u[i] for i in thirty_min_ind]) 
        v_all = np.array([v[i] for i in thirty_min_ind]) 
        w_all = np.array([w[i] for i in thirty_min_ind]) 

        for hgt in range(len(heights)):
        # run consensus averaging with a window of 5 m/s
            u_mean[idx,hgt] = Lidar_functions.consensus_avg(u_all[:,hgt],5)  
            v_mean[idx,hgt] = Lidar_functions.consensus_avg(v_all[:,hgt],5)  
            w_mean[idx,hgt] = Lidar_functions.consensus_avg(w_all[:,hgt],5)  
    return (u_mean, v_mean, w_mean)


def plot(final_path, u_mean, v_mean, ranges, heights):
    ticklabels = matplotlib.dates.DateFormatter("%H:%M")
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    fig.suptitle('SWEX 30 minute winds starting at %s 00:00:00 for 24 hours' % (ranges[0].strftime("%Y%m%d")))
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('HH:MM UTC')
    ax.set_ylim(0,1500)
    ax.xaxis.set_major_formatter(ticklabels)
    # make times and heights 2d arrays
    times = np.repeat([np.array(ranges)], u_mean.shape[-1], axis=0).swapaxes(1,0)
    heights = np.repeat([heights], u_mean.shape[0], axis=0)
    ax.barbs(times,heights,u_mean,v_mean,barb_increments=dict(half=2.5,full=5,flag=10))
    plt.savefig('%s/30min_winds_%s.png' % (final_path,ranges[0].strftime("%Y%m%d")))
    plt.close()

def write_netcdf(final_path, ranges, heights, u_mean, v_mean, w_mean, lat, lon, alt):
    # create netCDF file
    filepath = os.path.join(final_path, "30min_winds_%s.nc" % ranges[0].strftime("%Y%m%d"))
    # create dir if doesn't exist yet
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    nc_file = netCDF4.Dataset(filepath, 'w',format='NETCDF4')
    nc_file.createDimension('time', len(ranges))
    nc_file.createDimension('height',len(heights))
    base_time = nc_file.createVariable('base_time','i')
    base_time.units = 'seconds since 1970-01-01 00:00:00 UTC'
    base_time[:] = netCDF4.date2num(ranges[0], base_time.units)
    time = nc_file.createVariable('time','d','time')
    basetime_string = ranges[0].strftime('%Y-%m-%d %H:%M:%S %Z')
    time.units = 'seconds since ' + basetime_string
    time[:] = netCDF4.date2num(ranges, time.units)
    height = nc_file.createVariable('height','f','height')
    height[:] = heights
    height.long_name = 'Height above instrument level'
    height.units = 'm'
    u_var = nc_file.createVariable('u','f',('time','height'))
    u_var[:,:] = u_mean
    u_var.long_name = 'Eastward component of wind vector'
    u_var.units = 'm/s' 
    v_var = nc_file.createVariable('v','f',('time','height'))
    v_var[:,:] = v_mean
    v_var.long_name = 'Northward component of wind vector'
    v_var.units = 'm/s' 
    w_var = nc_file.createVariable('w','f',('time','height'))
    w_var[:,:] = w_mean
    w_var.long_name = 'Vertical component of wind vector'
    w_var.units = 'm/s' 
    lat_var = nc_file.createVariable('lat','f')
    lat_var[:] = lat
    lon_var = nc_file.createVariable('lon','f')
    lon_var[:] = lon
    alt_var = nc_file.createVariable('alt','f')
    alt_var[:] = alt
    nc_file.close()

def main():
    args = parseArgs()
    vadset = VADSet.from_file(args.vadfile)

    ranges = create_time_ranges(vadset.stime[0].date())

    u_mean, v_mean, w_mean = process(vadset.height, ranges, vadset.stime, vadset.u, vadset.v, vadset.w)

    if (args.plot):
        plot(args.destdir, u_mean, v_mean, ranges, vadset.height)
    write_netcdf(args.destdir, ranges, vadset.height, u_mean, v_mean, w_mean, vadset.lat, vadset.lon, vadset.alt)

if __name__=="__main__":
    main()


def test_create_time_ranges():
    day = dt.date(2021, 6, 30)
    res = create_time_ranges(day)
    assert len(res) == 48
    assert res[0] == dt.datetime(2021, 6, 30, 00, 00, 00, 00, pytz.UTC)
    assert res[-1] == dt.datetime(2021, 6, 30, 23, 30, 00, 00, pytz.UTC)
             
