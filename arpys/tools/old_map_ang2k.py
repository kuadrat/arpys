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

parser.add_argument('-c', '--cmap', type=str, default='bone_r',
                   help='Name of matplotlib colormap.')

parser.add_argument('-C', '--clean', default=True, action='store_false',
                   help='Whether to cut off unsymmetrized parts.')

parser.add_argument('-D', '--debug', default=False,
                   help='Toggle debug mode.')

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

kx, foo = pp.angle_to_k(xscale, theta, phi, hv, E_b, 
                       lattice_constant=args.lattice_constant, 
                       shift=args.xshift, degrees=True)

ky, foo = pp.angle_to_k(yscale, theta=phi, phi=theta, hv=hv, E_b=E_b, 
                       lattice_constant=args.lattice_constant, 
                       shift=args.yshift, degrees=True)

mp = pp.make_slice(data, d=0, i=args.index, integrate=args.integrate)

KX, KY, sym_mp = pp.symmetrize_map(kx, ky, mp, clean=args.clean, 
                                   debug=args.debug)

if 0 in sym_mp.shape :
    print('Symmetrization failed.')
    KX, KY, sym_mp = kx, ky, mp

if args.theta :
    kx, ky = pp.rotate_xy(kx, ky, args.theta)
    KX, KY = pp.rotate_xy(KX, KY, args.theta)

# Plotting
# ==============================================================================
figtitle = 'x: {}, y: {} - {}'.format(args.xshift, args.yshift, args.filename)
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, num=figtitle, 
                                    figsize=(15,5))

try :
    cmap = plt.get_cmap(args.cmap)
except Exception :
    cmap = 'bone_r'

vmax1 = args.vmax*mp.max()
#vmax2 = args.vmax*sym_mp.max()
vmax2 = sym_mp.max()
kwargs1 = dict(vmin=0, vmax=vmax1)
kwargs2 = dict(vmin=sym_mp.min(), vmax=vmax2)
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
