"""
Convert some of the nicer matplotlib and kustom colormaps to pyqtgraph 
colormaps.
"""

import numpy as np
from matplotlib import cm
from matplotlib.pyplot import colormaps
from pyqtgraph import ColorMap

# Importing kplot registers custom colormaps as matplotlib colormaps
from arpys.utilities import plotting

class pit_cmap(ColorMap) :
    """ Simple subclass of :class: `<pyqtgraph.ColorMap>`. Adds vmax, 
    powerlaw normalization and a convenience function to change alpha.
    """
    alpha = 0.5
    vmax = 1
    gamma = 1

    def __init__(self, pos, color, gamma=1, **kwargs) :
        super().__init__(pos, color, **kwargs)
        # Retain a copy of the originally given positions
        self.original_pos = self.pos.copy()
        # Apply the powerlaw-norm
        self.set_gamma(gamma)

    def apply_transformations(self) :
        """ Recalculate the positions where the colormapping is defined by 
        applying (in sequence) alpha, then a linear map to the range 
        [0, vmax] and finally the powerlaw scaling: pos' = pos**gamma.
        """
        # Reset the cache in pyqtgraph.Colormap
        self.stopsCache = dict()

        # Apply alpha
        self.color[:,-1] = self.alpha

        # Linearly transform color values to the new range
        old_max = self.original_pos.max()
        old_min = self.original_pos.min()
        new_max = old_max * self.vmax
        m = (new_max - old_min) / (old_max - old_min)
        self.pos = m * (self.original_pos - old_max) + new_max

        # Apply a powerlaw norm to the positions
        self.pos = self.pos**self.gamma

    def set_gamma(self, gamma=1) :
        """ Set the exponent for the power-law norm that maps the colors to 
        values. I.e. the values where the colours are defined are mapped like 
        ``y=x**gamma``.
        """
        self.gamma = gamma
        self.apply_transformations()

    def set_alpha(self, alpha) :
        """ Set the value of alpha for the whole colormap to *alpha* where 
        *alpha* can be a float or an array of length ``len(self.color)``.
        """
        self.alpha = alpha
        self.apply_transformations()

    def set_vmax(self, vmax=1) :
        """ Set the relative (to the maximum of the data) maximum of the 
        colorscale. 
        """
        self.vmax = vmax
        self.apply_transformations()

def convert_matplotlib_to_pyqtgraph(matplotlib_cmap, alpha=0.5) :
    """ Take a matplotlib colormap and convert it to a pyqtgraph ColorMap.


    ===============  ===========================================================
    **Parameters**
    matplotlib_cmap  either a str representing the name of a matplotlib 
                     colormap or a :class: 
                     `<matplotlib.colors.LinearSegmentedColormap>` or :class: 
                     `<matplotlib.colors.ListedColormap>` instance.
    alpha            float or array of same length as there are defined 
                     colors in the matplotlib cmap; the alpha (transparency) 
                     value to be assigned to the whole cmap. matplotlib cmaps 
                     default to 1.
    **Returns**
    pyqtgraph_cmap   :class: `<pyqtgraph.ColorMap>`
    ===============  ===========================================================
    """
    # Get the colormap object if a colormap name is given 
    if isinstance(matplotlib_cmap, str) :
        matplotlib_cmap = cm.cmap_d[matplotlib_cmap]
    # Number of entries in the matplotlib colormap
    N = matplotlib_cmap.N
    # Create the mapping values in the interval [0, 1]
    values = np.linspace(0, 1, N)
    # Extract RGBA values from the matplotlib cmap
    indices = np.arange(N)
    rgba = matplotlib_cmap(indices)
    # Apply alpha
    rgba[:,-1] = alpha

    return pit_cmap(values, rgba)


# Convert all matplotlib colormaps to pyqtgraph ones and make them available 
# in the dict cmaps
cmaps = dict()
for name,cmap in cm.cmap_d.items() :
    cmaps.update({name: convert_matplotlib_to_pyqtgraph(cmap)})

# +---------+ #
# | Testing | # ================================================================
# +---------+ #
if __name__ == '__main__' :
    print(cmaps)
