#!/usr/bin/python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace

import dataloaders as dl
import postprocessing as pp

# Set up parser
# ==============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('filename', type=str,
                   help='Name of the data file of which to apply ang2k \
                    conversion')

parser.add_argument('-s', '--shift', type=float, default=0,
                   help='How much to shift the angular scale (in its units).')

parser.add_argument('-l', '--lattice_constant', type=float, default=1,
                   help='Lattice constant in Angstrom.')

parser.add_argument('-v', '--vmax', type=float, default=1,
                   help='Colorbar scaling percentage (0-1).')

parser.add_argument('-c', '--cmap', type=str, default='plasma_r',
                   help='Name of matplotlib colormap.')

args = parser.parse_args()

# Load and process data
# ==============================================================================
ns = dl.load_data(args.filename)

data = ns.data
xscale = ns.xscale
yscale = ns.yscale
angles = ns.angles
theta = ns.theta
phi = ns.phi
hv = ns.hv
E_b = ns.E_b

shape = data.shape
# Quick fix for PSI "scans"
if shape[0] != 1 :
    print("Applying quick fix")
    data = data[0]
    data = data.reshape([1, shape[1], shape[2]])

#for key, val in ns.items() :
#    if key in ['theta', 'phi', 'hv', 'E_b'] :
#        print('{} {}'.format(key, val))
        
kx, ky = pp.angle_to_k(angles, theta, phi, hv, E_b, 
                       lattice_constant=args.lattice_constant, 
                       shift=args.shift, degrees=True)

# Plotting
# ==============================================================================
fig, (ax0, ax1, ax2) = plt.subplots(3)

vmax = args.vmax*data[0].max() 
cmap = args.cmap

ax0.plot(angles+args.shift, kx)
ax1.pcolormesh(angles+args.shift, yscale, data[0], vmin=0, vmax=vmax, cmap=cmap)
ax2.pcolormesh(kx, yscale, data[0], vmin=0, vmax=vmax, cmap=cmap)
ax2.grid(axis='x')
ax2.set_xlabel('units of pi/a')

plt.show()
