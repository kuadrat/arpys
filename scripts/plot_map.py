#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import dataloaders as dl
import postprocessing as pp

filename = '/home/kevin/qmap/materials/Bi2201/2017_12_ALS/20171215_00438.fits'
ddict = dl.load_data(filename)


z = 415
integrate = 10

data = ddict['data']
xscale = ddict['xscale']
yscale = ddict['yscale']
theta = ddict['theta']
phi = ddict['phi']
hv = ddict['hv']
E_b = ddict['E_b']

print(min(xscale), max(xscale))
print(min(yscale), max(yscale))

fsm = pp.make_slice(data, d=0, i=z, integrate=integrate)

shape = fsm.shape
kys = []
yshift = -72
yshift = 288
print(hv)
for y in yscale :
    kx, ky = pp.angle_to_k(xscale, theta, y+phi+yshift,
                           hv, E_b, lattice_constant=3.79,
                           shift=12)
    kys.append(ky)
kys = np.array(kys)


fig, (ax1, ax2) = plt.subplots(2)

ax1.pcolormesh(kx, kys, fsm)

# Rotate the map
#theta = np.pi/4
#
#R = np.array([[np.cos(theta), -np.sin(theta)],
#              [np.sin(theta),  np.cos(theta)]])
#KX, KY = np.meshgrid(kx, kys)
#nky = len(kys)
#nkx = len(kx)
#rkx = np.zeros([nky, nkx])
#rky = rkx.copy()
#for i in range(nky) :
#    for j in range(nkx) :
#        rotated = np.array([KX[i,j], KY[i,j]]).dot(R)
#        rkx[i,j] = rotated[0]
#        rky[i,j] = rotated[1]
rkx, rky = pp.rotate_xy(kx, kys)

ax2.pcolormesh(rkx, rky, fsm)
plt.show()
