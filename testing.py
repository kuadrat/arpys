#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace

import dataloaders as dl
import postprocessing as pp

#filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00414.fits'
#filename = '/home/kevin/Documents/qmap/experiments/2017_12_ALS/LSCO22/20171217_00006.fits'

#als = dl.Dataloader_ALS()
#datadict = als.load_data(filename)

datadict = dl.load_data(filename)

data = datadict['data']
xscale = datadict['xscale']
yscale = datadict['yscale']
angles = datadict['angles']
theta = datadict['theta']
phi = datadict['phi']
hv = datadict['hv']
E_b = datadict['E_b']

for key, val in datadict.items() :
    if key in ['theta', 'phi', 'hv', 'E_b'] :
        print('{} {}'.format(key, val))
        
kx, ky = pp.angle_to_k(angles, theta, phi, hv, E_b, lattice_constant=3.78, 
                       shift=0, degrees=True)


fig, (ax0, ax1, ax2) = plt.subplots(3)

ax0.plot(angles, kx)
ax1.pcolormesh(xscale, yscale, data[0], vmin=0, vmax=data[0].max(), 
               cmap='plasma_r')
ax2.pcolormesh(kx, yscale, data[0], vmin=0, vmax=data[0].max(), 
               cmap='plasma_r')

p = np.sqrt(2)
ax2.set_xticks([-p, 0, p])
ax2.set_xticklabels(['-pi,-pi', '0,0', 'pi,pi'])

plt.show()
