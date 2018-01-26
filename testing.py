#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace

import dataloaders as dl
import postprocessing as pp

#filename = '../als_sample/20160618_00116.fits'
#
#als = dl.Dataloader_ALS()
#d = als.load_data(filename)
#data = d['data']
#
#xscale = d['xscale']
#yscale = d['yscale']
#
#plt.pcolormesh(xscale, yscale, data[0,:,:])
#plt.show()

l, m, n = (10, 20, 10)
x = np.arange(l)
y = np.arange(m)
z = np.random.rand(n)
data = np.zeros((l, m, n))

for i in range(l) :
    for j in range(m) :
        for k in range(n) :
            data[i, j, k] = x[i] * np.sin(4*np.pi/m * y[j]) + 10*z[k]

#plt.figure()
#plt.pcolormesh(y, x, data[:,:,0])
#
#plt.figure()
#plt.pcolormesh(y, x, data[:,:,-1])

fig, axarray = plt.subplots(3)

s0 = pp.make_slice(data, 0, 5)
axarray[0].pcolormesh(s0)

s1 = pp.make_slice(data, 1, 5)
axarray[1].pcolormesh(s1)

s2 = pp.make_slice(data, 2, 5)
axarray[2].pcolormesh(s2)


plt.show()
