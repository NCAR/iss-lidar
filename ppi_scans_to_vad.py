#!/opt/local/anaconda3/bin/python
# pylint: disable=C0103

"""
Created March 2021
Carol Costanza

Output VAD winds into ARM netCDF format from cfradial format
Program works for either 1 cfradial file or multiple within 1 day

EXAMPLE RUN FROM COMMAND LINE
./ppi_scans_to_vad.py 'path_to_cfradial' 'path_nc_file_dest' 'max_cnr'
"""
import sys
import warnings
import glob
import datetime as dt
import numpy as np
import Lidar_functions
warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)

# get paths for ppi from command line input
all_path_ppi = sys.argv[1]
# get path for final nc file
final_path = sys.argv[2]
# get cnr/snr threshold from command line input
max_cnr = float(sys.argv[3])

# get list of all the ppi files for a given day
ppi_files = glob.glob(all_path_ppi)
ppi_scans = sorted(list(ppi_files))

stime = []
etime = []
vr_all = []
mean_cnr = []

for ppi in ppi_scans:
    [cnr_ppi, ranges_ppi, vr, elevation, azimuth, str_start_ppi, str_end_ppi, lat, lon,
     alt] = Lidar_functions.read_cfradial(ppi)

    # for low elevation angles, VAD output isn't very helpful
    # NEED THIS IF STATEMENT IF THE LIST OF PPIs MIGHT USE A DIFFERENT # OF AZIMUTH ANGLES
    # For this case, SWEX ppis need to have 360 azimuth angles
    if elevation < 6 or len(azimuth) != 360:
        continue

    for x in range(len(azimuth)):
        for y in range(len(ranges_ppi)):
            if cnr_ppi[x, y] < max_cnr:
                vr[x, y] = np.nan

    vr_all.append(vr)
    mean_cnr.append(np.nanmean(cnr_ppi, axis=0))
    stime.append(dt.datetime.strptime(str_start_ppi, '%Y-%m-%d %H:%M:%S.%f').timestamp())
    etime.append(dt.datetime.strptime(str_end_ppi, '%Y-%m-%d %H:%M:%S.%f').timestamp())

if len(ppi_scans) > 1:
    filename_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y%m%d')
else:
    filename_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y%m%d_%H%M%S')

final_file_path = final_path + 'VAD_' + filename_time + '.nc'
VAD = Lidar_functions.ARM_VAD(vr_all, ranges_ppi, elevation, azimuth)
Lidar_functions.VAD_ARM_nc_format(VAD, mean_cnr, max_cnr, alt, lat, lon, stime, etime,
                                  final_file_path)
