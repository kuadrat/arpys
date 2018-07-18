#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace

import dataloaders as dl
import postprocessing as pp

# Switch between different files; 0: Bi2201, 1: LSCO
f = 2
if f == 0 :
    # Bi2201
    ef_index = 378
    dist = 30
    n = 190
    filename = '/home/kevin/Documents/qmap/experiments/2018_04_ADRESS/data/Bi2201_2/037_nodal_cut_LH_kz42.h5'
elif f == 1: 
    # LSCO
    ef_index = 537
    dist = 30
    n = 50
    filename = '/home/kevin/Documents/qmap/materials/LSCO22/170702_psi/LSCO22_1_0017.h5'
elif f == 2 :
    # Tl2201
    ef_index = 499
    dist = 20
    n = 40
    filename = '/home/kevin/Documents/qmap/materials/Tl2201/171025_psi/Tl_1_0006.h5'

D = dl.load_data(filename)
data = D.data[0]

norm_data, profile = pp.normalize_above_fermi(data, ef_index=ef_index, 
                                              dist=dist, n=n, dim=2, 
                                              profile=True) 
smoothed_profile = pp.smooth(profile, n_box=31, recursion_level=3)

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2, figsize=(12,8))
kwargs = dict(cmap='Greys_r')#, vmin=data.min(), vmax=data.max())
mesh1=ax1.pcolormesh(data, **kwargs)
#mesh2=ax2.pcolormesh(norm_data, **kwargs)
mesh2=ax2.pcolormesh(data/profile, **kwargs)
mesh3=ax3.pcolormesh(data/smoothed_profile, **kwargs)
plt.colorbar(mesh1, ax=ax1)
plt.colorbar(mesh2, ax=ax2)
plt.colorbar(mesh3, ax=ax3)

x = [0, len(data[0])]
ax1.plot(x, 2*[ef_index+dist], 'r', x, 2*[ef_index+dist+n], 'r', lw=1)

ax4.plot(profile)
#ax4.plot(pp.smooth(profile,15,3)+10)
ax4.plot(smoothed_profile)
#ax4.plot(pp.smooth(profile,15,9)+30)
plt.show()

