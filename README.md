# ARPYS: python module for ARPES (**A**ngle **R**esolved **P**hoto**E**mission **S**pectroscopy) data analysis 

This repository consists of libraries, functions and tools related to ARPES 
data loading and analysis.
The software contained in this repository is distributed under the GNU 
General Public License v3+. See file 'COPYING' for more information.
The file 'LICENSE-3RD-PARTY.txt' covers the different licenses of libraries 
and other programs used by ARPYS.

## Requirements

The requirements are listed in `requirements.txt`.

## Installation

It is recommended to install with `pip`:
```
$ pip install arpys
```

## Documentation

Please find the documentation [here]<https://arpys.readthedocs.io/en/latest/>.

## Rough description of contents

The recommended way of using `arpys` currently is to make use of the classes 
in `dataloaders.py` (if a class for the beamline in question has already been 
implemented) to get the relevant data into a usable format in python. Then, 
one can use the functions provided in `postprocessing.py` (normalizations, 
background subtractions, etc.) on the so loaded data. Here's a simple example:
```
# Import the dataloaders and postprocessings
from arpys import dl, pp 

# Load the data (this requires an appropriate dataloader to be defined in 
# dataloaders.py. If it isn't, check the file to see how you should define it
# in your case.
D = dl.load_data('your_arpes_data_file.suffix')

# D is a Namespace object which stores the data array and some meta-data.
# In this example we're assuming the data to contain a single energy-k cut.
# arpys always loads data as 3d-arrays, however, so we need to take D.data[0]
# here.
data = D.data[0]
energies = D.xscale
angles = D.yscale

# Apply some background subtraction (use at your own discretion):
bg_subtracted = pp.subtract_bg_matt(data)

# Try taking the second derivative to make the bands more visible. This often
# requires smoothing first and is very susceptible to the various parameters.
from scipy.ndimage import filters
smoothened = filters.gaussian_filter(bg_subtracted, sigma=10)
dx = energies[1] - energies[0]
dy = angles[1] - angles[0]
second_derivative = pp.laplacian(smoothened, dx, dy)
```

postprocessing.py
-----------------

Library-like module that contains functions to process ARPES data, like 
normalizations, bg subtractions, derivative methods, etc.

dataloaders.py
--------------

Contains classes which handle reading of ARPES data from different beamlines 
(i.e. different data format and conventions) and passing it in a fixed, 
python-friendly format for use by other tools and scripts in this module.

utilities/
----------

A submodule that contains some custom python code that the original author 
used on his system and got incorporated into arpys.
arpys mostly needs the axes subclasses and some small helper functions from 
there.
This is actually just a copy of another module that is hosted at 
<git@github.com:kuadrat/kustom.git>.


## See also

This module had at some point exploded in size, offering many different 
tools, GUIs and command line interpreters to accomplish all kinds of 
things. The result was a long dependency list and complicated installations. In
an attempt to get things more streamlined and structured, the module has been 
stripped down to its bare essentials, outsourcing the graphical capabilities.

You can find PIT, a GUI for quick visualization of ARPES (and other) data 
[here](https://github.com/kuadrat/data_slicer). Use the
[corresponding plugin](https://github.com/kuadrat/ds_arpes_plugin) to use 
arpys data loading and postprocessing tools in conjunction with PIT.


================================================================================
Copyright (c) 2020 Kevin Kramer, Universität Zürich (kevin.kramer@uzh.ch)

