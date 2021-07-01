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

import Lidar_functions
import os
import sys
import argparse
import netCDF4
import datetime as dt
import numpy as np
import warnings
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)

def parseArgs():
    parser = argparse.ArgumentParser(description="Generate consensus averaged netcdfs from VAD files")
    parser.add_argument("vadfile", help="daily VAD file")
    parser.add_argument("destdir", help="directory to save averaged file to")
    parser.add_argument("date", help="date of file to create")
    parser.add_argument("--plot", help="create PNG plot w/ same filename as netcdf", dest="plot", default=False, action='store_true')
    return parser.parse_args()


args = parseArgs()
# get paths for ppi from command line input
vad_scan_path = args.vadfile
# get path for final nc file
final_path = args.destdir
# get str date
date = args.date

# get seconds for each set of 12 hourly plots/files
# needs to be time offset from midnight
hr_start = ['00']
secs = np.arange(0, 86400, 1800)
#secs.append(np.arange(0,43201,1800).tolist())
#secs.append(np.arange(43200,86401,1800).tolist())
datetime_var = dt.datetime.strptime(date,'%Y%m%d')
full_datetime = []
#for i in range(2):
datetimes = []
for s in secs:
    full_datetime.append(datetime_var + dt.timedelta(seconds=int(s)))
# get the needed variables from daily VAD file
vad_file = netCDF4.Dataset(vad_scan_path,'r')
lat = vad_file.variables['lat'][:]
lon = vad_file.variables['lon'][:]
alt = vad_file.variables['alt'][:]
heights = vad_file.variables['height'][:]
base_time = vad_file.variables['base_time'][:]
start_time = vad_file.variables['time'][:]
u = vad_file.variables['u'][:]
v = vad_file.variables['v'][:]
w = vad_file.variables['w'][:]

# loop through the 30 minute time steps to get all the winds
#for s,steps in enumerate(secs):
#print(s, steps)
u_mean = np.zeros((48,len(heights)))
v_mean = np.zeros((48,len(heights)))
w_mean = np.zeros((48,len(heights)))

# create plot
ax = None
if (args.plot):
    ticklabels = matplotlib.dates.DateFormatter("%H:%M")
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    fig.suptitle('SWEX 30 minute winds starting at %s %s:00:00 for 24 hours' % (date,hr_start[0]))
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('HH:MM UTC')
    ax.set_ylim(0,1500)
    ax.xaxis.set_major_formatter(ticklabels)

#for ind_start in range(len(secs)-1):
for idx, sec in enumerate(secs):
    start = sec
    end = start + 1800 
    thirty_min_ind = [i for i in range(len(start_time))\
                      if start_time[i] >= start and\
                      start_time[i] < end]

    if len(thirty_min_ind) == 0:
        u[idx,:] = np.nan
        v[idx,:] = np.nan
        w[idx,:] = np.nan
        continue

    u_all = np.array([u[i] for i in thirty_min_ind]) 
    v_all = np.array([v[i] for i in thirty_min_ind]) 
    w_all = np.array([w[i] for i in thirty_min_ind]) 

    for hgt in range(len(heights)):
    # run consensus averaging with a window of 5 m/s
        u_mean[idx,hgt] = Lidar_functions.consensus_avg(u_all[:,hgt],5)  
        v_mean[idx,hgt] = Lidar_functions.consensus_avg(v_all[:,hgt],5)  
        w_mean[idx,hgt] = Lidar_functions.consensus_avg(w_all[:,hgt],5)  

    vert_time = [full_datetime[idx]]*len(heights)
    if (args.plot):
        ax.barbs(vert_time,heights,u_mean[idx],v_mean[idx],barb_increments=dict(half=2.5,full=5,flag=10))

if (args.plot):
    plt.savefig('%s/30min_winds_%s.png' % (final_path,date))
    plt.close()

# create netCDF file
filepath = os.path.join(final_path, "30min_winds_%s" % date)
nc_file = netCDF4.Dataset(filepath, 'w',format='NETCDF4')
nc_file.createDimension('time',48)
nc_file.createDimension('height',len(heights))
base_time = nc_file.createVariable('base_time','i')
base_time[:] = datetime_var.timestamp()
base_time.units = 'seconds since 1970-01-01 00:00:00'
time = nc_file.createVariable('time','d','time')
time[:] = secs
time.units = 'seconds since basetime'
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

