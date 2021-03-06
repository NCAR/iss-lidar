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
import sys
import netCDF4
import datetime as dt
import numpy as np
import warnings
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)

# get paths for ppi from command line input
vad_scan_path = sys.argv[1]
# get path for final nc file
final_path = sys.argv[2]
# get str date
date = sys.argv[3]

# get seconds for each set of 12 hourly plots/files
# needs to be time offset from midnight
hr_start = ['00','12']
secs = []
secs.append(np.arange(0,43201,1800).tolist())
secs.append(np.arange(43200,86401,1800).tolist())
datetime_var = dt.datetime.strptime(date,'%Y%m%d')
full_datetime = []
for i in range(2):
  datetimes = []
  for s in range(24):
    datetimes.append(datetime_var + dt.timedelta(seconds=secs[i][s]))
  full_datetime.append(datetimes)

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
for s,steps in enumerate(secs):
  u_mean = np.zeros((24,len(heights)))
  v_mean = np.zeros((24,len(heights)))
  w_mean = np.zeros((24,len(heights)))
  # create plot
  ticklabels = matplotlib.dates.DateFormatter("%H:%M")
  fig,ax = plt.subplots(1,1,figsize=(10,8))
  fig.suptitle('SWEX 30 minute winds starting at %s %s:00:00 for 12 hours' % (date,hr_start[s]))
  ax.set_ylabel('Height (m)')
  ax.set_xlabel('HH:MM UTC')
  ax.set_ylim(0,1500)
  ax.xaxis.set_major_formatter(ticklabels)

  for ind_start in range(len(steps)-1):
    ind_end = ind_start+1
    thirty_min_ind = [i for i in range(len(start_time))\
                      if start_time[i] >= steps[ind_start] and\
                      start_time[i] < steps[ind_end]]
     
    if len(thirty_min_ind) == 0:
      u[ind_start,:] = np.nan
      v[ind_start,:] = np.nan
      w[ind_start,:] = np.nan
      continue

    u_all = np.array([u[i] for i in thirty_min_ind]) 
    v_all = np.array([v[i] for i in thirty_min_ind]) 
    w_all = np.array([w[i] for i in thirty_min_ind]) 

    for hgt in range(len(heights)):
      # run consensus averaging with a window of 5 m/s
      u_mean[ind_start,hgt] = Lidar_functions.consensus_avg(u_all[:,hgt],5)  
      v_mean[ind_start,hgt] = Lidar_functions.consensus_avg(v_all[:,hgt],5)  
      w_mean[ind_start,hgt] = Lidar_functions.consensus_avg(w_all[:,hgt],5)  

    vert_time = [full_datetime[s][ind_start]]*len(heights)
    ax.barbs(vert_time,heights,u_mean[ind_start],v_mean[ind_start],barb_increments=dict(half=2.5,full=5,flag=10))

  plt.savefig('%s/30min_winds_%s_%s.png' % (final_path,date,hr_start[s]))
  plt.close()

  # create netCDF file
  nc_file = netCDF4.Dataset('%s/30min_winds_%s_%s.nc' % (final_path,date,hr_start[s]),'w',format='NETCDF4')
  nc_file.createDimension('time',24)
  nc_file.createDimension('height',len(heights))
  base_time = nc_file.createVariable('base_time','i')
  base_time[:] = datetime_var.timestamp()
  base_time.units = 'seconds since 1970-01-01 00:00:00'
  time = nc_file.createVariable('time','d','time')
  time[:] = secs[s][:-1]
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
