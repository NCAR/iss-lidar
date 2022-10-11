#!/opt/local/anaconda3/bin/python
# pylint: disable=C0103

"""
Created March 2021
Carol Costanza

Output VAD winds into ARM netCDF format from cfradial format
Program works for either 1 cfradial file or multiple within 1 day
"""
import os
import warnings
import glob
import argparse
import numpy as np
from vad import VAD, VADSet
from ppi import PPI
from typing import List

warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)


def createParser():
    parser = argparse.ArgumentParser(description="Generate netCDF of VAD winds"
                                     " from PPI scans")
    parser.add_argument("--min_cnr", default=-22, type=float, help="threshold"
                        " cnr below this value")
    parser.add_argument("destdir", help="directory to save VAD files to")
    parser.add_argument("ppifiles", help="ppi file(s) for input", nargs='+')
    return parser.parse_args()


def selectFiles(path: str):
    # expand glob expression if it didn't get expanded in the shell
    ppi_files = glob.glob(path)
    return sorted(list(ppi_files))


def process(ppi_files: List[str], min_cnr: int):
    vads = []

    for f in ppi_files:
        ppi = PPI.fromFile(f)

        # for low elevation angles, VAD output isn't very helpful
        if ppi.elevation < 6:
            continue
        ppi.threshold_cnr(min_cnr)

        # generate VAD for this timestep
        vad = VAD.calculate_ARM_VAD(ppi)
        vads.append(vad)

    if not vads:
        # didn't successfully create any vads. can't continue processing.
        return

    vadset = VADSet.from_VADs(vads, min_cnr)
    return vadset


def save(vadset: VADSet, destdir: str, prefix: str = None):
    filename_time = vadset.stime[0].strftime('%Y%m%d')
    final_file_name = 'VAD_'
    if prefix:
        final_file_name += prefix + '_'
    final_file_name += filename_time + '.nc'
    final_file_path = os.path.join(destdir, final_file_name)
    vadset.to_ARM_netcdf(final_file_path)


def main():
    args = createParser()
    # try to allow either list of files created by shell glob or expression to
    # pass to python glob
    ppi_scans = args.ppifiles
    if len(args.ppifiles) == 1:
        ppi_scans = selectFiles(args.ppifiles[0])
    vadset = process(ppi_scans, args.min_cnr)
    save(vadset, args.destdir)


if __name__ == "__main__":
    main()
