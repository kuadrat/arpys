gui.py
------

A GUI which allows to take quick looks at cuts and maps. Only GUI and 
plotting functionality is defined in this file - all data analysis tools are 
used from other files in this module (dataloaders.py, postprocessing.py).


postprocessing.py
-----------------

Contains functions to process ARPES data, like normalizations, bg 
subtractions, derivative methods, etc.


dataloaders.py
--------------

Contains classes which handle reading of ARPES data and passing it in the 
right format to the GUI (or other clients).


cut.h5, map.h5
--------------

Example ARPES data used for testing.


All other files
---------------

Temporary files/scripts for testing.
