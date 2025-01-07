#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103,E1101

"""
Decoder for raw .hpl files that are outputted by the Halo Photonics doppler
lidar. Many things are hard coded and should be documented with in-line
comments. Outputs decoded data to netcdf-4. May have to change the 'lookup'
variable to account for different header info. Though not originally coded to
decode RHI scans, it shouldn't be a problem as long as it is specified in the
lookup table


Original Author: Tyler Bell - University of Oklahoma (March 2017)
"""


import logging
import os
import argparse
from datetime import datetime, timedelta
import netCDF4
import numpy as np


# Key is name of scan type in header, value is name as it should appear
# in the filename
lookup = {'User file 4 - stepped': 'fp',
          'User file 3 - stepped': 'ppi',
          'User file 2 - stepped': 'fp',
          'User file 1 - stepped': 'ppi',
          'csm_Tail_7VT_Coplanar - csm': 'Lionstail',
          'csm_George - csm': 'George',
          'VAD - stepped': 'ppi',
          'Stare - stepped': 'fp',
          'Stare': 'fp'}

FILL_VALUE = -9999


def decode_header(header):
    """
    Takes in a list of lines from the raw hpl file. Separates them by
    tab and removes unnecessary text
    """
    new_header = {}

    for item in header:
        split = item.split('\t')
        # new_header[split[0].replace(':', '')] = split[1].replace("\r\n", "")
        new_header[split[0].replace(':', '')] = split[1].strip()

    return new_header


def _to_epoch(dt):
    return (dt - datetime(1970, 1, 1)).total_seconds()


def process_file(in_file, out_dir, prefix):
    """
    Processes a raw halo hpl file and turns it into a netcdf
    :param in_file:
    :param out_dir:
    :return:
    """

    # Read in the text file
    lines = []
    with open(in_file) as f:
        for line in f:
            lines.append(line)

    logging.debug("Decoding header")
    # Read in the header info
    header = decode_header(lines[0:11])

    ngates = int(header['Number of gates'])
    # nrays = int(header['No. of rays in file'])
    # Cant do this apparently. Not always correct (wtf)
    len_data = len(lines[17:])
    nrays = int(len_data / (ngates + 1))

    gate_length = float(header['Range gate length (m)'])
    start_time = datetime.strptime(header['Start time'], '%Y%m%d %H:%M:%S.%f')
    scan_type = lookup[header['Scan type']]

    logging.info("Processing file type: %s" % scan_type)
    logging.debug("Number of rays: %s" % nrays)

    logging.debug("Reading data")
    # Read in the actual data
    az = np.zeros(nrays)
    hour = np.zeros(nrays)
    elev = np.zeros(nrays)
    pitch = np.zeros(nrays)
    roll = np.zeros(nrays)
    rng = np.asarray([(gate + .5) * gate_length for gate in range(ngates)])

    vel = np.zeros((ngates, nrays))
    intensity = np.zeros((ngates, nrays))
    beta = np.zeros((ngates, nrays))

    try:
        for ray in range(nrays):
            # Get the scan info
            info = lines[ray * (ngates + 1) + 17].split()
            hour[ray] = float(info[0])
            az[ray] = float(info[1])
            elev[ray] = float(info[2])
            pitch[ray] = float(info[3])
            roll[ray] = float(info[4])

            for gate in range(ngates):
                data = lines[ray * (ngates + 1) + 17 + gate + 1].split()
                vel[gate, ray] = float(data[1])
                intensity[gate, ray] = float(data[2])
                beta[gate, ray] = float(data[3])

    except IndexError:
        logging.warning("Something went wrong with the indexing here...")

    logging.debug('Preparing to write netcdf')

    # Get the times and dates figured out for the netcdf
    time = []
    epoch = []
    for h in hour:
        dt = datetime(start_time.year, start_time.month,
                      start_time.day) + timedelta(hours=h)
        time.append(dt)
        epoch.append(_to_epoch(dt))

    time = np.asarray(time)
    epoch = np.asarray(epoch)
    base_time = _to_epoch(start_time)
    time_offset = epoch - base_time

    # Figure out netcdf attrs
    nc_attrs = {'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%S')}
    # None right now

    # Get the filename figured out
    if prefix is None:
        filename = start_time.strftime(
            "{type}_%Y%m%d_%H%M%S.nc".format(type=scan_type))
    else:
        filename = start_time.strftime(
            "{prefix}_{type}_%Y%m%d_%H%M%S.nc".format(type=scan_type,
                                                      prefix=prefix))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = os.path.join(out_dir, filename)

    # Write out the netcdf
    logging.info("Writing netcdf")
    logging.debug("Creating file: {}".format(filename))

    nc = netCDF4.Dataset(filename, "w", format="NETCDF4")

    # Create the dimensions
    nc.createDimension('time', size=None)
    nc.createDimension('range', size=len(rng))

    # Set the netcdf attributes
    logging.debug('Writing attributes')
    nc.setncatts(nc_attrs)

    logging.debug('Writing base_time')
    var = nc.createVariable('base_time', 'i8')
    var.setncattr('long_name', 'Time')
    var.setncattr('units', 'seconds since 1970-01-01 00:00:00 UTC')
    var[:] = base_time

    logging.debug('Writing time_offset')
    var = nc.createVariable('time_offset', 'i8', dimensions=('time',))
    var.setncattr('long_name', 'Time offset')
    var.setncattr('unis', 'seconds since base_time')
    var[:] = time_offset

    logging.debug('Writing epoch')
    var = nc.createVariable('epoch', 'i8', dimensions=('time',))
    var.setncattr('long_name', 'Epoch Time')
    var.setncattr('units', 'seconds since 1970-01-01 00:00:00 UTC')
    var[:] = epoch

    logging.debug('Writing hour')
    var = nc.createVariable('hour', 'f8', dimensions=('time',))
    var.setncattr('long_name', 'Hour of Day')
    var.setncattr('units', 'UTC')
    var[:] = hour

    logging.debug('Writing range')
    var = nc.createVariable('range', 'f8', dimensions=('range',))
    var.setncattr('long_name', 'height')
    var.setncattr('units', 'km AGL')
    var[:] = rng

    logging.debug('Writing azimuth')
    var = nc.createVariable('azimuth', 'f8', dimensions=('time', 'range'))
    var.setncattr('long_name', 'Azimuth Angle')
    var.setncattr('units', 'degrees')
    var[:] = np.tile(az, (len(rng), 1)).transpose()

    logging.debug('Writing elevation')
    var = nc.createVariable('elevation', 'f8', dimensions=('time', 'range'))
    var.setncattr('long_name', 'Elevation angle')
    var.setncattr('units', 'degrees above the horizon')
    var[:] = np.tile(elev, (len(rng), 1)).transpose()

    logging.debug('Writing pitch')
    var = nc.createVariable('pitch', 'f8', dimensions=('time', 'range'))
    var.setncattr('long_name', 'Instrument Pitch')
    var.setncattr('units', 'degrees')
    var[:] = np.tile(pitch, (len(rng), 1)).transpose()

    logging.debug('Writing roll')
    var = nc.createVariable('roll', 'f8', dimensions=('time', 'range'))
    var.setncattr('long_name', 'Instrument Roll')
    var.setncattr('units', 'degrees')
    var[:] = np.tile(roll, (len(rng), 1)).transpose()

    logging.debug('Writing velocity')
    var = nc.createVariable('velocity', 'f8', dimensions=('time', 'range'))
    var.setncattr('long_name', 'Doppler velocity')
    var.setncattr('units', 'm/s')
    var.setncattr('comment', 'Positive values are toward the radar')
    var[:] = vel.transpose()

    logging.debug('Writing intensity')
    var = nc.createVariable('intensity', 'f8', dimensions=('time', 'range'))
    var.setncattr('long_name', 'Intensity')
    var.setncattr('units', 'Unitless')
    var.setncattr('comment', 'This is computed as (SNR+1)')
    var[:] = intensity.transpose()

    logging.debug('Writing backscatter')
    var = nc.createVariable('backscatter', 'f8', dimensions=('time', 'range'))
    var.setncattr('long_name', 'Attenuated backscatter')
    var.setncattr('units', 'km^(-1) sr^(-1)')
    var[:] = (beta*1e3).transpose()

    logging.debug('Closing file')
    nc.close()

    logging.info("Netcdf successfully created!")

    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='in_files', nargs='*')
    parser.add_argument('-o', dest='out_dir')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-p', '--prefix', dest='prefix', default=None)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                            level=logging.INFO)

    for f in args.in_files:
        logging.info('Processing file %s' % f)
        filename = process_file(f, args.out_dir, args.prefix)
