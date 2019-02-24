#!/usr/bin/python
description="""
Script that takes ARPES map data (3D) and slices it ath a specified index 
along a specified dimension to create cuts through the map and writes them to 
files.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from arpys import dataloaders as dl
from arpys import postprocessing as pp
from arpys.utilities import plotting as kplot

# Set up parser
# ==============================================================================

parser = argparse.ArgumentParser(description=description)

parser.add_argument('filename', type=str,
                    help='Name of the map to take a slice from')

args = parser.parse_args()

# Parameters
# ==============================================================================
filename = args.filename
filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00438.fits'

outfile = '20171215_00438_slice_x290.p'

z = 415
integrate = 10

slice_dim = 2
slice_ind = 60
slice_dim = 1
slice_ind = 290

figsize = (4, 8)

# Load the data
dataloader = dl.Dataloader_ALS()
ns = dataloader.load_data(filename)
print(ns.xscale)
data = ns.data
# Extract a Fermi surface map
fsm = pp.make_slice(data, d=0, i=z, integrate=integrate)

# Create a cut at the specified index
cut = pp.make_slice(data, d=slice_dim, i=slice_ind, integrate=1)

# Prepare plotting
fig = plt.figure(figsize=figsize, projection='cursor')

# Plot the map and indicate the location of the cut
ax1.pcolormesh(fsm)
fsm_max = fsm.shape[ slice_dim%2 ]
x = [0, fsm_max]
y = 2*[slice_ind]
if slice_dim != 1 :
    # Swap
    x, y = y, x

ax1.plot(x, y, 'r-', lw=1)

# Plot the cut
ax2.pcolormesh(cut)

import pickle

# Reshape data before outputting
x, y = cut.shape
new_data = cut.reshape([1, x, y])
ns.data = new_data
print(ns)

# Write an output file (binary)
with open(outfile, 'wb') as f :
    pickle.dump(ns, f)

plt.show()

