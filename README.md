# iss-lidar
Lidar processing scripts for the NSF NCAR [EOL](https://www.eol.ucar.edu/) [Integrated Sounding System](https://www.eol.ucar.edu/observing_facilities/iss). These scripts are currently used to work with data from a Leosphere WindCube scanning wind lidar.

Currently this library performs two main functions: calculating velocity-azimuth display (VAD) winds from PPI scans, and calculating 30-minute consensus averaged winds from VAD winds.

This code was developed at EOL based on examples from Josh Gebauer, now at the University of Oklahoma. Subsequent development by Matt Paulus, Carol Ruchti, Bill Brown, Jacquie Witte, and Isabel Suhr (all EOL). Python package is maintained by Isabel Suhr.

## Installing as a package
The iss-lidar package is available as a package on [PyPi](https://pypi.org/project/iss-lidar/), and can be installed with pip.

## Installing from a local checkout
It may be more convenient to clone this repo to get access to the iss-lidar code, especially if you are planning to modify the code, or use the convenience scripts for running processing. If you check out this repo locally, you will need to use a pip local install to make sure imports of iss_lidar work correctly:
```
pip install -e /path/to/lidar/repo
```

## Tutorial from LROSE 2025 workshop
In January 2025, a tutorial on how to use this package in combination with the [LROSE](http://wiki.lrose.net/index.php/Main_Page) software suite was created for the LROSE workshop at the AMS annual meeting. That jupyter notebook is now available as an example in the lrose-hub github repo [here](https://github.com/nsf-lrose/lrose-hub/blob/main/notebooks/LROSE_Lidar_tutorial.ipynb).