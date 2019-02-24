#!/usr/bin/python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace

import dataloaders as dl
import postprocessing as pp
#import kustom.plotting as kplot

# Set up parser
# ==============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('filename', type=str,
                   help='Name of the data file of which to apply ang2k \
                    conversion')

parser.add_argument('-x', '--xshift', type=float, default=0,
                   help='How much to shift the angular scale (in its units).')

parser.add_argument('-y', '--yshift', type=float, default=0,
                   help='How much to shift the angular scale (in its units).')

parser.add_argument('-l', '--lattice_constant', type=float, default=1,
                   help='Lattice constant in Angstrom.')

parser.add_argument('-i', '--index', type=int, default=0,
                   help='Index where to take the map.')

parser.add_argument('-I', '--integrate', type=int, default=5,
                   help='Number of slices to integrate when making the map.')

parser.add_argument('-t', '--theta', type=float, default=0,
                   help='Rotation angle.')

parser.add_argument('-v', '--vmax', type=float, default=1,
                   help='Colorbar scaling percentage (0-1).')

parser.add_argument('-c', '--cmap', type=str, default='magma_r',
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


mp = pp.make_slice(data, d=0, i=args.index, integrate=args.integrate)

f=[] 
xshifts = np.arange(2.5, 3.5, 0.05) 
xshifts = np.arange(-45, 46, 45) 
yshifts = np.arange(-90, 91, 45) 

def scan_xy_shifts(xshifts, yshifts, theta=theta, phi=phi, hv=hv, E_b=E_b, 
                   lattice_constant=args.lattice_constant, degrees=True) :
    pass

for xshift in xshifts :
    print(xshift)
    kx, foo = pp.angle_to_k(xscale, theta, phi, hv, E_b, 
                       lattice_constant=args.lattice_constant, 
                       shift=xshift, degrees=True)
    f.append([])
    for yshift in yshifts :
        ky, foo = pp.angle_to_k(yscale, theta=phi, phi=theta, hv=hv, E_b=E_b, 
                       lattice_constant=args.lattice_constant, 
                       shift=yshift, degrees=True)

        KX, KY, sym_m, overlap = pp.symmetrize_map(kx, ky, mp, clean=True, 
                                                   overlap=True, debug=False)
        print(overlap)
        f[-1].append(overlap)

fig, ax = plt.subplots()
ax.pcolormesh(f)
plt.show()
import sys
sys.exit()

if args.theta :
    kx, ky = pp.rotate_xy(kx, ky, args.theta)
    KX, KY = pp.rotate_xy(KX, KY, args.theta)

# Plotting
# ==============================================================================
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

try :
    cmap = plt.get_cmap(args.cmap)
except Exception as e :
    print(e)
    cmap = 'magma_r'

vmax1 = args.vmax*mp.max()
vmax2 = args.vmax*sym_mp.max()
kwargs1 = dict(vmin=0, vmax=vmax1)
kwargs2 = dict(vmin=0, vmax=vmax2)
for kwargs in [kwargs1, kwargs2] :
    kwargs.update(dict(cmap=cmap))

ax0.pcolormesh(mp, **kwargs1)
ax1.pcolormesh(kx, ky, mp, **kwargs1)
ax2.pcolormesh(KX, KY, sym_mp, **kwargs2)


# Define cornerpoints for diagonals 
c = 1
diag_kwargs = dict(color='gray', ls = '--', lw=1)

# Plot lines at sqrt(2)/2 which can help in orienting
rt2 = np.sqrt(2)/2
ax1.plot([-rt2, -rt2], [-1, 1], **diag_kwargs)
ax1.plot([rt2, rt2], [-1, 1], **diag_kwargs)

for ax in ax1, ax2 :
    # Plot some diagonals
    ax.plot([-c, c], [-c, c], **diag_kwargs)
    ax.plot([-c, c], [c, -c], **diag_kwargs)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.grid()

plt.show()
