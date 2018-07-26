# ARPYS: python module for ARPES (**A**ngle **R**esolved **P**hoto**E**mission **S**pectroscopy) data analysis 

This repository consists of libraries, programs and scripts related to ARPES 
data analysis.
The software contained in this repository is distributed under the GNU 
General Public License v3+. See file 'COPYING' for more information.
The file 'LICENSE-3RD-PARTY.txt' covers the different licenses of libraries 
and other programs used by ARPYS.

## Rough description of files

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
making) based on the pyqtgraph module.

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
`git@github.com:kuadrat/kustom.git`.

================================================================================
Copyright (c) 2018 Kevin Kramer, Universität Zürich (kevin.kramer@uzh.ch)

