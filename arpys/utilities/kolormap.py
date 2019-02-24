#!/usr/bin/python
"""
http://www.workwithcolor.com/hsl-color-schemer-01.htm

cdict = {'red':     ((),
                     ()),
         'green':   ((),
                     ()),
         'blue':    ((),
                     ())}

white
FFFFFF  255,255,255
EBCCFF  235,204,255
D699FF  214,153,255
C266FF  194,102,255
AD33FF  173, 51,255
9900FF  153,  0,255
7A00CC  122,  0,204
5C0099   92,  0,153
3D0066   61,  0,102
1F0033   31,  0, 51
purple-black
000000    0,  0,  0
333300   51, 51,  0
666600  102,102,  0
999900  153,153,  0
CCCC00  204,204,  0
FFFF00  255,255,  0
FFFF33  255,255, 51
FFFF66  255,255,102
FFFF99  255,255,153
FFFFCC  255,255,204
white-yellow
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Put the colors in lists...
colors1 = ['F4EBFF',
           'DAB8FE',
           'BC80FC',
           '9739FB',
           '6904D6',
           '4A0395',
           '2A0256',
           '100120']

colors2 = ['1D1C01',
           '393701',
           '555202',
           '716D02',
           '8C8803',
           'A8A303',
           'C4BD04',
           'DFD805',
           'FAF318',
           'FFFFFF']

# ... and make a list of the lists
color_lists = [colors1, colors2]

# Specify the ranges for all color lists except the last one - the last one 
# is just 1 - sum(color_ranges)
color_ranges = [1]
last = 1. - sum(color_ranges)
color_ranges.append(last)

def hex_to_rgb(n) :
    """ Convert a color in hex notation (e.g. FFFFFF) to a normalized RGB 
    tuple (in this example (1., 1., 1.)). """
    # Split input string in three parts
    r, g, b = [ n[2*i:2*i+2] for i in range(3) ]

    rgb = [] 
    # Convert to int and normalize
    for i, c in enumerate([r, g, b]) :
        c = int(c, base=16)
        c /= 255.
        rgb.append(c)

    # Return the rgb tuple
    return tuple(rgb)

# Main loop
red = []
blue = []
green = []

# Loop over color lists
for i, color_list in enumerate(color_lists) :
    n_colors = len(color_list)
    # For the Interpolation value to reach the maximum of 1.0 we need to 
    # make bigger steps for the colors in the last color_list
    if i == len(color_lists)-1 :
        n_colors -= 1

    color_range = color_ranges[i]
    spanned_ranges = sum(color_ranges[:i])

    # Loop over colors in this color list
    for j, color in enumerate(color_list) :
        # Calculate the interpolation limit for this color
        s = spanned_ranges + j * color_range/n_colors

        # Convert to normalized RGB and create the cdict entries
        rgb = hex_to_rgb(color)
        #print(rgb)
        r = (s, rgb[0], rgb[0])
        g = (s, rgb[1], rgb[1])
        b = (s, rgb[2], rgb[2])

        # Add the cdict entries to their containers
        red.append(r)
        blue.append(b)
        green.append(g)

"""
for i, color in enumerate(colors2) :
    s = color_range1 + i * color_range2/(n_colors2-1)
    rgb = [i/255. for i in color]
    r = (s, rgb[0], rgb[0])
    g = (s, rgb[1], rgb[1])
    b = (s, rgb[2], rgb[2])

    red.append(r)
    blue.append(b)
    green.append(g)
"""
#print(red)
print(np.array(red)[:,0])

# Create the new colormap
cdict = dict(red=tuple([r for r in red]),
             blue=tuple([b for b in blue]),
             green=tuple([g for g in green]))
cmap = LinearSegmentedColormap('arpes', cdict)

if __name__ == '__main__' :
    # Create the test image data
    x = np.arange(1000)
    # no. of y points
    ny = 50
    ymin = 5
    # 5% of data range
    p = 0.05*(x.max() - x.min())
    # spatial frequency of modulation
    f = 60 * 2*np.pi/x.max()

    data = np.zeros([ny, len(x)])
    for i in range(ny) :
        if i > ymin :
            y = x + ( p*(i-ymin)*np.sin(x*f) )/(ny-ymin)
        else :
            y = x
        data[i] = y

    plt.pcolormesh(data, cmap=cmap)

    # Test with some map
    from kustom.arpys import dataloaders as dl
    from kustom.arpys import postprocessing as pp
    import kustom.plotting
    from matplotlib.colors import LightSource


    filename = \
    '/home/kevin/Documents/qmap/materials/LSCO22/170702_psi/LSCO22_1_0017.h5' 
    #'/home/kevin/Documents/qmap/materials/LSCO22/170702_psi/LSCO22_1_0040_reduced.p' 

    ddict = dl.load_data(filename)
    data = ddict['data']
    d = data[0]

    ls = LightSource(azdeg=270, altdeg=80)
    shaded = ls.hillshade(d, vert_exag=d.max())
    #shaded = ls.shade(d, cmap=cmap, vert_exag=d.max())
    cmap = plt.get_cmap('viridis')

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.pcolormesh(d, cmap=cmap)

    ax2 = fig.add_subplot(122)
    #ax2.pcolormesh(data[0], cmap='rainbow_light')
    #ax2.pcolormesh(shaded, cmap=cmap)
    ax2.imshow(shaded)
    plt.show()

    """
    colors1 = [(255,255,255),
               (235,204,255),
               (214,153,255),
               (194,102,255),
               (173, 51,255),
               (153,  0,255),
               (122,  0,204),
               ( 92,  0,153),
               ( 61,  0,102),
               ( 31,  0, 51)]

    colors2 = [(  0,  0,  0),
               ( 51, 51,  0),
               (102,102,  0),
               (153,153,  0),
               (204,204,  0),
               (255,255,  0),
               (255,255, 51),
               (255,255,102),
               (255,255,153),
               (255,255,204)]
    color_range1 = 0.2
    color_range2 = 1 - color_range1

    n_colors1 = len(colors1)
    n_colors2 = len(colors2)
    """
