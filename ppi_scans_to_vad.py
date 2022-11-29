#!/opt/local/anaconda3/bin/python
# pylint: disable=C0103

"""
Created March 2021
Carol Costanza

Output VAD winds into ARM netCDF format from cfradial format
Program works for either 1 cfradial file or multiple within 1 day
"""
import warnings
import glob
import argparse
import numpy as np
from vad import VADSet
from tools import create_filename

warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)


def create_parser():
    parser = argparse.ArgumentParser(description="Generate netCDF of VAD winds"
                                     " from PPI scans")
    parser.add_argument("--min_cnr", default=-22, type=float, help="threshold"
                        " cnr below this value")
    parser.add_argument("--threshold_config", default=None, help="path to "
                        "config file of post-VAD thresholding parameters")
    parser.add_argument("destdir", help="directory to save VAD files to")
    parser.add_argument("ppifiles", help="ppi file(s) for input", nargs='+')
    return parser.parse_args()


def select_files(path: str):
    # expand glob expression if it didn't get expanded in the shell
    ppi_files = glob.glob(path)
    return sorted(list(ppi_files))


def save(vadset: VADSet, destdir: str, prefix: str = None):
    final_file_path = create_filename(vadset.stime[0], destdir, "VAD", prefix)
    vadset.to_ARM_netcdf(final_file_path)


def main():
    args = create_parser()
    # try to allow either list of files created by shell glob or expression to
    # pass to python glob
    ppi_scans = args.ppifiles
    if len(args.ppifiles) == 1:
        ppi_scans = select_files(args.ppifiles[0])
    vadset = VADSet.from_PPIs(ppi_scans, args.min_cnr)
    if(args.threshold_config):
        # apply thresholds if a config is present
        vadset.load_thresholds(args.threshold_config)
        vadset.apply_thresholds()
    save(vadset, args.destdir)


if __name__ == "__main__":
    main()
