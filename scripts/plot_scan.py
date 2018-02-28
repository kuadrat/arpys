#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from dataloaders import Dataloader_PSI, Dataloader_ALS

filepath = '/home/kevin/Documents/qmap/analysis/171025_psi_Tl_1/data/'
filename = 'Tl_1_0018.h5'

# Load the data dict (with metadata)
D = Dataloader_PSI().load_data(filepath+filename)

d = D['data']

nrows=3
ncols=4

# Plotting
fig, axarray = plt.subplots(nrows=3, ncols=4)

for i in range(nrows) :
    for j in range(ncols) :
        ax = axarray[i,j]
        ax.pcolormesh(d[:,:,i+j], cmap='Greys')
        ax.set_title('{} {}'.format(i,j))
        ax.set_xticks([])
        ax.set_yticks([])

# Remove spacing between plots
fig.subplots_adjust(wspace=0, hspace=0)

plt.show()
