""" Calculate and visualize k_z vs. hv. """
import matplotlib.pyplot as plt
import numpy as np

#_Parameters____________________________________________________________________

#lattice_constant = 9.798
lattice_constant = 9.302
#c_prime = lattice_constant/2

hv_max = 1000
#hv = np.arange(10, 1000, 1)
hv = np.arange(10, hv_max, 1)
hv_ticks = np.arange(10, hv_max, 5)

v0_0 = 0
v0_1 = 15

figsize = (8, 6)
nrow, ncol = 1, 1

line_kwargs = dict(lw=1)
colors = ['r', 'b']
grid_major_kwargs = dict(color=3*[0.3], lw=1)
grid_minor_kwargs = dict(color=3*[0.6], lw=0.5)
arrowprops = dict(arrowstyle='->', shrinkB=0)

label = r'$V_0=${} eV'

#_Maths_________________________________________________________________________

def kz(hv, v0) :
    return 0.5124 * np.sqrt(hv+v0)

# Calculate kz in units if 2 pi / c (distance Gamma-Z along k_z) for 
# different values of the inner potential
kz_0 = kz(hv, v0=v0_0) * lattice_constant / 2 /np.pi
kz_1 = kz(hv, v0=v0_1) * lattice_constant / 2 /np.pi

kz_max = max(kz_0.max(), kz_1.max())
kz_major_ticks = np.arange(0, kz_max, 2)
kz_minor_ticks = np.arange(0, kz_max, 1)

#_Plotting______________________________________________________________________

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(nrow, ncol, 1)

ax.plot(hv, kz_0, color=colors[0], label=label.format(v0_0), **line_kwargs)
ax.plot(hv, kz_1, color=colors[1], label=label.format(v0_1), **line_kwargs)

ax.set_xlim(hv[0], hv[-1]+1)
ax.set_ylim(0, kz_max)

ax.set_xticks(hv_ticks, minor=True)
ax.set_yticks(kz_major_ticks)
ax.set_yticks(kz_minor_ticks, minor=True)

ax.grid(which='major', **grid_major_kwargs)
ax.grid(which='minor', **grid_minor_kwargs)

ax.set_title(r'$|\mathbf{k}|=0.5124 \cdot \sqrt{h\nu+V_0}$')
ax.set_xlabel(r'$h\nu$ (eV)')
ax.set_ylabel(r'$|\mathbf{k}|$ ($2\pi/c$)')

ax.legend(loc='lower right')
# Annotations for the two curves
#i0 = np.argmin(np.abs(hv - 100))
#i1 = np.argmin(np.abs(hv - 100))
#ax.annotate('$V_0=10$ eV', [hv[i0], kz_10[i0]], [60, 9], 
#            arrowprops=arrowprops) 
#ax.annotate('$V_0=15$ eV', [hv[i1], kz_15[i1]], [110, 5], 
#            arrowprops=arrowprops) 

plt.show()

