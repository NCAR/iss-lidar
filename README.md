# iss-lidar
Lidar processing scripts for the NSF NCAR [EOL](https://www.eol.ucar.edu/) [Integrated Sounding System](https://www.eol.ucar.edu/observing_facilities/iss). These scripts are currently used to work with data from a Leosphere WindCube scanning wind lidar.

Currently this library performs two main functions: calculating velocity-azimuth display (VAD) winds from PPI scans, and calculating 30-minute consensus averaged winds from VAD winds.

This code was developed at EOL based on examples from Josh Gebauer, now at the University of Oklahoma. Subsequent development by Matt Paulus, Carol Ruchti, Bill Brown, Jacquie Witte, and Isabel Suhr (all EOL). Python package is maintained by Isabel Suhr.

## Installation

### Installing as a package
The iss-lidar package is available as a package on [PyPi](https://pypi.org/project/iss-lidar/), and can be installed with pip.

### Installing from a local checkout
It may be more convenient to clone this repo to get access to the iss-lidar code, especially if you are planning to modify the code, or use the convenience scripts for running processing. If you check out this repo locally, you will need to use a pip editable local install to make sure imports of iss_lidar work correctly:
```
pip install -e /path/to/lidar/repo
```

## Usage

### As a library
This package is designed as python library, so users can import specific objects or functions into their own code. The best example of this use case is in the tutorial given below.

#### Tutorial from LROSE 2025 workshop
In January 2025, a tutorial on how to use this package in combination with the [LROSE](http://wiki.lrose.net/index.php/Main_Page) software suite was created for the LROSE workshop at the AMS annual meeting. That jupyter notebook is now available as an example in the lrose-hub github repo [here](https://github.com/nsf-lrose/lrose-hub/blob/main/notebooks/LROSE_Lidar_tutorial.ipynb).

### Convenience scripts
This package includes two convenience scripts to facilitate data processing and exploration: `ppi_scans_to_vad` and `vad_to_consensus`. The usage of these scripts is given in their help output, i.e. `ppi-scans_to_vad --help`. When this repo is installed as a pip package (whether from PyPi or as a local install) these scripts will be automatically added to the user's path, or if the repo is cloned locally, the python scripts themselves can be called directly.
