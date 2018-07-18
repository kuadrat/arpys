#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

from kustom.arpys import dl, pp
from kustom.functions import testarray

# Create some test-data
d = testarray(3, 5, 7).astype(float)
d = np.array([
    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
     [1, 5, 1, 1, 1, 5, 1],
     [1, 1, 4, 1, 4, 1, 1],
     [1, 2, 2, 2, 2, 2, 1],
     [5, 5, 5, 5, 5, 5, 5]],
    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
     [1, 4, 2, 1, 2, 4, 1],
     [1, 2, 3, 1, 3, 2, 1],
     [1, 1, 2, 3, 2, 1, 1],
     [4, 5, 5, 5, 5, 5, 4]],
    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
     [1, 5, 1, 1, 1, 5, 1],
     [1, 1, 4, 1, 4, 1, 1],
     [1, 2, 2, 2, 2, 2, 1],
     [5, 5, 5, 5, 5, 5, 5]]
]).astype(float)

# Retain a copy of the original
D = d.copy()

# Apply normalization to the map
#pp.apply_to_map(d, pp.normalize_per_integrated_segment, dim=2, fkwargs=dict(dim=0))
profile = pp.apply_to_map(d, pp.normalize_above_fermi, dim=0, output=True, fargs=[1], 
                fkwargs=dict(n=1, inverted=True, profile=True, dim=2))

print(profile)

# Plot stuff
fig = plt.figure()
nrows = 2
ncols = len(d) + 1
vmin = D.min()
vmax = D.max()
for z in range(ncols-1) :
    # Original data
    ax = fig.add_subplot(nrows, ncols, z+1)
    mesh0 = ax.pcolormesh(D[z], vmin=vmin, vmax=vmax)

    # Processed data
    ax = fig.add_subplot(nrows, ncols, ncols+z+1)
    mesh1 = ax.pcolormesh(d[z], vmin=d.min(), vmax=d.max())

cax0 = fig.add_subplot(nrows, ncols, ncols)
cax1 = fig.add_subplot(nrows, ncols, nrows*ncols)
plt.colorbar(mesh0, cax=cax0)
plt.colorbar(mesh1, cax=cax1)
plt.show()

