#!/opt/local/anaconda3/bin/python
#
# Created June 2021 Carol Costanza
#
# Output 30 minute consensus averaged VAD winds into ARM netCDF format from
# cfradial format Creates the 12 hour plot and netCDF with 30 minute winds
#
# EXAMPLE RUN FROM COMMAND LINE ./vad_to_30min_winds.py 'path_to_VAD_nc_file'
# 'path_30min_nc_file_dest' 'date'

import argparse
import numpy as np
import datetime as dt
import warnings
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from vad import VADSet
from consensus_set import ConsensusSet
from tools import create_filename

warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate consensus averaged"
                                     "netcdfs from VAD files")
    parser.add_argument("vadfile", help="daily VAD file")
    parser.add_argument("destdir", help="directory to save averaged file to")
    parser.add_argument("--plot",
                        help="create PNG plot w/ same filename as netcdf",
                        dest="plot", default=False, action='store_true')
    return parser.parse_args()


def plot(final_path: str, u_mean: np.ndarray, v_mean: np.ndarray,
         ranges: np.ndarray, heights: np.ndarray):
    ticklabels = matplotlib.dates.DateFormatter("%H:%M")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle('SWEX 30 minute winds starting at %s 00:00:00 for 24 hours'
                 % (ranges[0].strftime("%Y%m%d")))
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('HH:MM UTC')
    ax.set_ylim(0, 1500)
    ax.xaxis.set_major_formatter(ticklabels)
    # make times and heights 2d arrays
    times = np.repeat([np.array(ranges)],
                      u_mean.shape[-1], axis=0).swapaxes(1, 0)
    heights = np.repeat([heights], u_mean.shape[0], axis=0)
    ax.barbs(times, heights, u_mean, v_mean,
             barb_increments=dict(half=2.5, full=5, flag=10))
    plt.savefig('%s/30min_winds_%s.png'
                % (final_path, ranges[0].strftime("%Y%m%d")))
    plt.close()


def main():
    args = parse_args()
    vs = VADSet.from_file(args.vadfile)
    cs = ConsensusSet.from_VADSet(vs, 5, dt.timedelta(minutes=30))

    if (args.plot):
        plot(args.destdir, cs.u, cs.v, cs.stime, cs.height)

    fpath = create_filename(cs.stime[0], args.destdir, "30min_winds")
    cs.to_ARM_netcdf(fpath)


if __name__ == "__main__":
    main()
