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
import os
import sys
import warnings
import glob
import argparse
import datetime as dt
import numpy as np
import Lidar_functions
from ppi import PPI

warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)

def createParser():
    parser = argparse.ArgumentParser(description="Generate netCDF of VAD winds from PPI scans")
    parser.add_argument("--max_cnr", default=-22, type=float, help="threshold cnr below this value")
    parser.add_argument("ppifiles", help="ppi file(s) for input")
    parser.add_argument("destdir", help="directory to save VAD files to")
    return parser.parse_args()

def selectFiles(path):
    # get list of all the ppi files for a given day
    ppi_files = glob.glob(path)
    return sorted(list(ppi_files))

def process(ppi_files, max_cnr, final_path):
    stime = []
    etime = []
    vr_all = []
    mean_cnr = []
    for f in ppi_files:
        ppi = PPI(f)
        # for low elevation angles, VAD output isn't very helpful
        # NEED THIS IF STATEMENT IF THE LIST OF PPIs MIGHT USE A DIFFERENT # OF AZIMUTH ANGLES
        # For this case, SWEX ppis need to have 360 azimuth angles
        if ppi.elevation < 6 or len(ppi.azimuth) != 360:
            continue

        ppi.threshold_cnr(max_cnr)

        vr_all.append(ppi.vr)
        mean_cnr.append(np.nanmean(ppi.cnr, axis=0))
        stime.append(dt.datetime.strptime(ppi.str_start, '%Y-%m-%d %H:%M:%S.%f').timestamp())
        etime.append(dt.datetime.strptime(ppi.str_end, '%Y-%m-%d %H:%M:%S.%f').timestamp())

    if len(ppi_files) > 1:
        filename_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y%m%d')
    else:
        filename_time = dt.datetime.fromtimestamp(stime[0]).strftime('%Y%m%d_%H%M%S')
    final_file_name = 'VAD_' + filename_time + '.nc'
    final_file_path = os.path.join(final_path, final_file_name)

    VAD = Lidar_functions.ARM_VAD(vr_all, ppi.ranges, ppi.elevation, ppi.azimuth)
    VAD.create_ARM_nc(mean_cnr, max_cnr, ppi.alt, ppi.lat, ppi.lon, stime, etime, final_file_path)

def main():
    args = createParser()
    ppi_scans = selectFiles(args.ppifiles)
    process(ppi_scans, args.max_cnr, args.destdir)

if __name__=="__main__":
    main()
