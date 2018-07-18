#!/usr/bin/python

import argparse
import matplotlib.pyplot as plt

import arpys as arp

# Prepare argument parsing
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('filename', default=None, type=str)
parser.add_argument('-m', '--min', default=0, type=int)
parser.add_argument('-M', '--max', default=-1, type=int)
parser.add_argument('-i', '--index', default=0, type=int)
parser.add_argument('-d', '--dim', default=0, type=int)
args = parser.parse_args()

# Load and process data
# ==============================================================================
D = arp.dl.load_data(args.filename)
data = D.data
print(data.shape)

if args.dim==0 :
    raw = data[args.index, args.min:args.max]
elif args.dim==1 :
    raw = data[args.index,: , args.min:args.max]

print(raw.shape)
subtracted = arp.pp.subtract_bg_shirley(raw, dim=args.dim)

# Plot
# ==============================================================================
fig, [ax1, ax2] = plt.subplots(2)
kwargs = dict(cmap='Blues')
ax1.pcolormesh(raw, **kwargs)
ax2.pcolormesh(subtracted, **kwargs)
plt.show()

