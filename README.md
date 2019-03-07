# ARPYS: python module for ARPES (**A**ngle **R**esolved **P**hoto**E**mission **S**pectroscopy) data analysis 

![](https://raw.githubusercontent.com/kuadrat/arpys/master/screenshots/pit_demo.gif)

This repository consists of libraries, programs and scripts related to ARPES 
data analysis.
The software contained in this repository is distributed under the GNU 
General Public License v3+. See file 'COPYING' for more information.
The file 'LICENSE-3RD-PARTY.txt' covers the different licenses of libraries 
and other programs used by ARPYS.

## Requirements

The requirements are listed in `requirements.txt`. Most notable are 
`pyqtgraph`, a nice library that's built on PyQt and allows fast real-time 
data visualization. Consequently, `arpys` requires `PyQt5` and its dependency 
`SIP`. Confer `INSTALLING.md` for more info.

## Installation

Please refer to the file `INSTALLING.md`

## Rough description of contents

The recommended way of using `arpys` currently is to make use of the classes 
in `dataloaders.py` (if the beamline in question has already been implemented) 
to get the relevant data into a usable format in python. Then, one can use the
functions provided in `postprocessing.py` (normalizations, background 
subtractions, etc.) on the so loaded data. Here's a simple example:
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

The tools that ship with `arpys` should be considered to be in an untested 
stage and used at your own discretion.

postprocessing.py
-----------------

Library-like module that contains functions to process ARPES data, like 
normalizations, bg subtractions, derivative methods, etc.

dataloaders.py
--------------

Contains classes which handle reading of ARPES data from different beamlines 
(i.e. different data format and conventions) and passing it in a fixed, 
python-friendly format for use by other tools and scripts in this module.

arpes_plot.py
-------------

Implements a commad-line interpreter that allows quick visualization of data 
and provides some basic postprocessing options (like cropping, normalization, 
angle-to-k conversion,...)
Can be used as a program from the command line (possibly after a `chmod 755 
arpes_plot.py`) by
```Bash
$ arpes_plot.py <path-to-data>
```

pit
---

The **P**ython **I**mage **T**ool: A graphical data analysis tool (in the 
making) based on the `pyqtgraph` module.

tools/
------

A set of little scripts and command-line tools for specific jobs.
Confer each tools respective documentation for more info.

#### gui.py

[Deprecated] A GUI which allows to take quick looks at cuts and maps. This is 
built with tkinter and matplotlib and, consequently, is rather slow.
In most cases `arpes_plot.py` should be used instead.

#### apc

A link to `arpes_plot.py`

#### bandcharacters.py

A tool to plot the band characters from a wien2k DFT calculation.

utilities/
----------

A submodule that contains some custom python code that the original author 
used on his system and got incorporated into arpys.
arpys mostly needs the axes subclasses and some small helper functions from 
there.
This is actually just a copy of another module that is hosted at 
<git@github.com:kuadrat/kustom.git>.

================================================================================
Copyright (c) 2018 Kevin Kramer, Universität Zürich (kevin.kramer@uzh.ch)

