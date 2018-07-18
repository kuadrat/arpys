#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

from kustom.arpys import pp

# Create some test data
N = 1000
x = np.arange(N)
y0 = pp.gaussian_step(x, mu=600, sigma=50, step_x=550, flip=True)
# Add signal at the high-energy end of the spectrum
y1 = y0 + pp.gaussian(x, mu=-50, sigma=75)


# Subtract Shirley
s0 = pp.subtract_bg_shirley(np.array([y0]), dim=1, normindex=400)[0]
s1 = pp.subtract_bg_shirley(np.array([y1]), dim=1, normindex=400)[0]

# Plot the 'raw data'
plt.plot(x, y0)
plt.plot(x, y1)

# Plot the subtracted EDC
plt.plot(x, s0)
plt.plot(x, s1)

plt.show()

