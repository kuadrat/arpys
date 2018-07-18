# ARPYS: python module for ARPES (**A**ngle **R**esolved 
# **P**hoto**E**mission **S**pectroscopy) data analysis 

This repository consists of libraries, programs and scripts related to ARPES 
data analysis.

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
arpes_plot.py <path-to-data>
```

pit
---

The **P**ython **I**mage **T**ool: A graphical data analysis tool (in the 
making) based on the pyqtgraph module.

tools/
------

A set of little scripts and command-line tools for specific jobs.

### gui.py

A GUI which allows to take quick looks at cuts and maps. Only GUI and 
plotting functionality is defined in this file - all data analysis tools are 
used from other files in this module (dataloaders.py, postprocessing.py).


scripts/
--------

Rough scripts etc. for testing. Don't really need to be on the repository...

data/
-----

Contains some sample data for testing.

utilities/
----------

A submodule that contains some custom python code that the original author 
used on his system and got incorporated into arpys.
arpys mostly needs the axes subclasses and some small helper functions from 
there.

