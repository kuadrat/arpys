#!/usr/bin/python
description="""
Simple tool to view ARPES map data.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec

from arpys import dataloaders as dl
from arpys import postprocessing as pp

from arpys.utilities import plotting as kplot

# Argument parsing
# ==============================================================================

parser = argparse.ArgumentParser(description=description)

parser.add_argument('filename',
                    type=str, 
                    help='Full or relative path to datafile.')

parser.add_argument('-I', '--integrate',
                    type=int, default=5,
                    help='Number of energy slices over which to integrate.')

parser.add_argument('-i', '--index',
                    type=int, default=0,
                    help='Starting index for mapmaking.')

parser.add_argument('-c', '--cutoff',
                    type=float, nargs=2,
                    help='''Amount of x-range outside of which the data 
                    is to be cut off in percent. Example: -c 25 75 cut off 
                    the first and last quarter of the image (along x) while 
                    -c 0 100 would just plot the full image.''')

parser.add_argument('-e', '--emin',
                    type=float, default=1.9,
                    help='Lower energy limit in (positive) eV.')

parser.add_argument('-E', '--emax',
                    type=float, default=0.1,
                    help='Upper energy limit in eV.')

parser.add_argument('-v', '--vmin',
                    type=float, default=0,
                    help='''Percentage of maximum of data at which to start the 
                    colormap. I.e. values below v_min*max(data) will be 
                    represented as 0 (or out of range) by the colormap.''')

parser.add_argument('-V', '--vmax',
                    type=float, default=1,
                    help='''Percentage of maximum of data at which to end the 
                    colormap. I.e. values above v_max*max(data) will be 
                    represented as out of range by the colormap.''')

parser.add_argument('-n', '--norm',
                    type=str, choices=['edc', 'None'], default='None',
                    help='''Select the type of normalization that is to be 
                    applied.''')

parser.add_argument('-S', '--shift',
                    default=False, action='store_true',
                    help='''Whether to shift each channel by its Fermi-level 
                    offset to create a flat Fermi edge. Only does anything if 
                    --fermi is nonzero.''')

parser.add_argument('-H', '--histogram',
                    type=int, default=0, 
                    help='''If a positive integer n is given, create a 
                    histogram with n bins showing the intensity distribution 
                    of the data.''')

parser.add_argument('-C', '--cmap',
                    type=str, default='magma',
                    help='Name of a matplotlib colormap.')

parser.add_argument('-B', '--colorbar',
                    default=False, action='store_true',
                    help='Toggle colorbar creation.')

parser.add_argument('--shading',
                    type=str, default='flat',
                    help='Matplotlib shading argument (flat or gouraud)')

parser.add_argument('--figsize',
                    type=int, default=[10, 5],
                    help='Size of figure in inches.')

parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Print additional information during execution.')

args = parser.parse_args()

# Prepare plotting
# ==============================================================================
fig = plt.figure(num=args.filename, figsize=args.figsize)
ax = fig.add_subplot(121, projection='cursor')

# Second figure for intensity selection
ifigsize = (4, 4)
#ifig = plt.figure(figsize=ifigsize)
iax = fig.add_subplot(122, projection='cursor')

# Load data
# ==============================================================================
ns = dl.load_data(args.filename)

data = ns.data
xscale = ns.xscale
yscale = ns.yscale

shape = data.shape
N_e = shape[0]
N_kx = shape[1]
N_ky = shape[2]

# Process data
# =========================================================================

# Create a map at specified index
# -------------------------------
dimension = 0
d0 = pp.make_slice(data, dimension, args.index, args.integrate)

# Prepare intensity selector plot
# -------------------------------
# Get the total intensity as a function of energy (i.e. integrate map at each 
# binding energy)
intensities = []
for i in range(N_e) :
    this_slice = data[i,:,:]
    intensity = sum(sum(this_slice))
    intensities.append(intensity)

# Create the x range for the energy-intensity plot
#energies = np.linspace(args.emin, args.emax, len(intensities))
energies = np.arange(len(intensities))

# Cut off unwanted artifacts at the sides
if args.cutoff :
    xmin, xmax = [int(N_k*x/100) for x in args.cutoff]
    d0 = d0[:,xmin:xmax]
    N_k = xmax - xmin

    xscale = xscale[xmin:xmax]

# Apply normalization
# -------------------
# NOTE: most pp functions return the data as a 3D array with the first dim 
# set to 0, thus the `[0]` after the function call.
if args.norm == 'edc' :
    d0 = pp.normalize_per_integrated_segment(d0, 0)[0]


# Plot data
# =========================================================================
vmin = args.vmin * d0.max()
vmax = args.vmax * d0.max()

kwargs = dict(
              vmin=vmin,
              vmax=vmax,
              cmap=args.cmap,
              shading=args.shading
             )
pcm = ax.pcolormesh(d0, **kwargs)

if args.colorbar :
    fig.colorbar(pcm, ax=ax)

# Plot the intensity distribution
# -------------------------------

ikwargs = dict(
               lw=1,
               ls='-',
               color='blue'
              )
iax.plot(energies, intensities, **ikwargs)

# Register slice selection 
def on_cursor_change(event) :
    try :
        i = iax.get_cursor()
    except AttributeError :
        return
    i = i[0]
    i = np.searchsorted(energies, i)
    print(i)
    d0 = pp.make_slice(data, dimension, i, args.integrate)
    # Clear current axes and plot next slice
    kwargs.update({'vmax': args.vmax*d0.max()})
    ax.clear()
    ax.pcolormesh(d0, **kwargs)
    ax.figure.canvas.draw()
    
fig.canvas.mpl_connect('button_release_event', on_cursor_change)

vmax_max_level = 1
vmax_step_level = 0.1
# Define and connect event handling
def on_key_press(event) :
    """ 
    React to button presses:
    up/down arrow : adjust colormap scaling
    """
    key = event.key
    # Incr./decr. the colormap scale
    if key == 'down' :
        if args.vmax < vmax_max_level :
            args.vmax += vmax_step_level
    elif key == 'up' :
        if args.vmax > vmax_step_level :
            args.vmax -= vmax_step_level

    # Clear the current axes and replot with new colorscale
    on_cursor_change(event)

# Connect the callback
fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()

