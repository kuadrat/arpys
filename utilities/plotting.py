"""
Useful propcycle elements for plotting.
Usage:

from matplotlib import rc
rc("axes", prop_cycle=<insert your cycler here>)

"""

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from cycler import cycler
from itertools import islice
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import PolygonSelector


# +----------------+ #
# | Color palettes | # =========================================================
# +----------------+ #

# A colorblind friendly almost optimally distinct palette of colors
kolorful = ["#00599f", "#8f0033", "#d8c4f5", "#009757", "#cd690f", "#c653c1", \
            "#1990ff", "#222222"]

# Kevin's custom colors
k1 = ["#386cb0", "#7fc97f", "#fdc97f", "#e30278", "#751b6d", "#d8c4f5", \
      "#bf5b16", "#222222"]
k2 = ["#7b5db3", "#40b959", "#bb543e", "#c8ab42", "#0c214b", "#7ce1d1", 
      "#ad2e6e", "#161300"]


# +--------------+ #
# | Marker lists | # ===========================================================
# +--------------+ #

# A list of 8 different markers
markers8 = ["o", "v", "s", "^", "p", "<", "*", ">"]
markers4 = 2 * ["o", "v", "s", "^"]


# +-----------------+ #
# | Linestyle lists | # ========================================================
# +-----------------+ #

# A list of 4 different linestyles
linestyles4 = 2 * ["-", "--", "-.", ":"]


# +--------------+ #
# | Prop cyclers | # ===========================================================
# +--------------+ #

# A cycler that guarantees differentiability of 8 lines
markerlinecolorcycler = cycler("marker", markers8) + \
                        cycler("color", k1) + \
                        cycler("linestyle", linestyles4)

# Same as markerlinecolorcycler but with solid lines for all
markercolorcycler     = cycler("marker", markers8) + \
                        cycler("color", k1)


# +-----------+ #
# | Utilities | # ==============================================================
# +-----------+ #

def make_cycler(color = k2, **kwargs) : #lines = None, markers = None) :
    """ 
    Create a cycler from different elements. The kwargs need to be 
    :class: cycler keyword arguments.
    """
    result = cycler("color", color)
    for arg in kwargs :
        result += cycler(arg, kwargs[arg])
    return result


def advance_cycler(ax, n=1) :
    """
    Advance the state of a cycler by n steps.

    Inputs:
    -------
    ax      : matplotlib.axes._suplots.AxesSubplot instance; The subplot in 
    which to advance the cycler.
    n       : int; The number of steps to advance the cycler by

    Outputs:
    --------
    None
    """
    if n < 1 or type(n) != int :
        raise ValueError(
"Number of steps should be a positive integer, got {}.".format(n))

    for i in range(n) :
        next(ax._get_lines.prop_cycler)


def rewind_cycler(ax) :
    """
    Rewind the cycler to the last position that was used, i.e. the next line 
    will have the same colour as the last one that was drawn.
    Note: this is done very crudely - if you know the length of your cycler it
    might be better to just use the 'advance_cycler' method with argument 
    'len(cycler)-1'.

    Inputs:
    -------
    ax      : matplotlib.axes._suplots.AxesSubplot instance; The subplot in 
    which to advance the cycler.

    Outputs:
    --------
    None
    """
    cyc = ax._get_lines.prop_cycler
    # In order to get the length, iterate until you get the same result again
    # First, safe a starting value
    start = next(islice(cyc, 0, None), None)
    current = None
    length = 0
    while current != start :
        current = next(islice(cyc, 0, None), None)
        length += 1

    # Now advance the cycle by the length - 2 (-2 because we advanced one in 
    # the while loop above)
    advance_cycler(ax, length - 2)


def set_cycler(color=k2, **kwargs) :
    """ 
    Shorthand for setting the cycler. Uses kustom.plotting.make_cycler to 
    create a cycler according to given kwargs and imports and updates 
    matplotlibrc.  
    """
    # Define the cycler
    my_cycler = make_cycler(color=color, **kwargs)
    # Set the cycler in the matplotlibrc
    from matplotlib import rc
    rc("axes", prop_cycle=my_cycler)


def make_n_colors(n=8, cmap='plasma') :
    """
    Pick n equidistant colors from the matplotlib.cm colormap specified with 
    `cmap`. Returns a list of rgba color tuples.
    """
    # Load the colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap)

    # Create n points in the interval [0,1]
    from numpy import linspace
    points = linspace(0, 1, n)

    return [cmap(p) for p in points]


# +----------+ #
# | Colormap | # ===============================================================
# +----------+ #

# Rainbow ligth colormap from ALS
# ------------------------------------------------------------------------------

# Load the colormap data from file
filepath = '/home/kevin/bin/kustom/cmaps/rainbow_light.dat'
data = np.loadtxt(filepath)
colors = np.array([(i[0], i[1], i[2]) for i in data])

# Normalize the colors
colors /= colors.max()

# Build the colormap
rainbow_light = LinearSegmentedColormap.from_list('rainbow_light', colors, 
                                                  N=len(colors))
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)

# Hanin colormap: rainbow_light + viridis
# ------------------------------------------------------------------------------

# Load the colormap data from file
filepath = '/home/kevin/bin/kustom/cmaps/hanin.dat'
data = np.loadtxt(filepath)
colors = np.array([(i[0], i[1], i[2]) for i in data])

# Build the colormap
hanin = LinearSegmentedColormap.from_list('hanin', colors, 
                                                  N=len(colors))
cm.register_cmap(name='hanin', cmap=hanin)

# kocean colormap: ocean_r with different peak color
# ------------------------------------------------------------------------------

# Load the colormap data from file
filepath = '/home/kevin/bin/kustom/cmaps/kocean_red.dat'
data = np.loadtxt(filepath)
colors = np.array([(i[0], i[1], i[2], i[3]) for i in data]) #rgba

# Build the colormap
kocean = LinearSegmentedColormap.from_list('kocean', colors, N=len(colors))
cm.register_cmap(name='kocean', cmap=kocean)

# ARPES colormap
# ------------------------------------------------------------------------------

#from kustom.kolormap import cmap
#cm.register_cmap(name='arpes', cmap=cmap)

# Custom normalizations
# ------------------------------------------------------------------------------
class MidpointNorm(matplotlib.colors.Normalize) :
    """ A norm that maps the values between vmin and midpoint to the range 
    0-0.5 and the values from midpoint to vmax to 0.5-1. This is ideal for a 
    bivariate colormap where the data is split in two regions of interest of 
    different extents. 
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class DynamicNorm(matplotlib.colors.Normalize) :
    """ A norm which maps high density data regions to proportionally larger 
    intervals in color space. 
    """

    def __init__(self, vmin=None, vmax=None, n=None, bins=None, clip=False) :
        # Find the indices of where bins lie within the interval vmin-vmax
        # Fall back to vmin=0, vmax=len(n) in case vmin/vmax were not given 
        # (leading to TypeError) or have unreasonable values such that 
        # np.where did not yield a result (throwing an IndexError).
        try :
            imin = np.where(bins>=vmin)[0][0]
        except (IndexError, TypeError) :
            imin = 0
        try :
            imax = np.where(bins>=vmax)[0][0]
        except (IndexError, TypeError) :
            imax = len(n)

        self.n = n[imin:imax]

        # Normalize the histogram entries such that they sum to 1
        self.n = self.n/sum(self.n)
        # Create the 0-1 interval with levels proportional to the number of 
        # entries in n
        self.interval = []
        s = 0
        for i in self.n :
            self.interval.append(s)
            s += i
        self.interval.append(s)

        self.bins = bins[imin:imax+1]
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None) :
        return np.ma.masked_array(np.interp(value, self.bins, self.interval))


# +---------+ #
# | Cursors | # ================================================================
# +---------+ #

from matplotlib.axes import Axes
from matplotlib.projections import register_projection

class cursorax(Axes) :
    name='cursor'
    cursor_x = None
    cursor_y = None
    color = 'red'
    lw = 1

    def __init__(self, *args, **kwargs) :
        self._set_up_event_handling()
        super().__init__(*args, **kwargs)

    def get_xy_minmax(self) :
        """ Return the min and max for the x and y axes, depending on whether 
        xscale and yscale are defined. 
        """
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()

        return xmin, xmax, ymin, ymax

    def get_xy_scales(self) :
        """ Depending on whether we have actual data scales (self.xscale and 
        self.yscale are defined) or not, return arrays which represent data 
        coordinates. 
        """
        if self.xscale is None or self.yscale is None :
            shape = self.data.shape
            yscale = np.arange(0, shape[1], 1)
            xscale = np.arange(0, shape[2], 1)
        else :
            xscale = self.xscale
            yscale = self.yscale

        return xscale, yscale

    def get_cursor(self) :
        """ Return the cursor position. """
        try :
            return self.cursor_x._x[0], self.cursor_y._y[0]
        except AttributeError :
            return None, None

#    def snap_to(self, x, y) :
#        """ Return the closest data value to the given values of x and y. """
#        xscale, yscale = self.get_xy_scales()
#
#        # Find the index where element x/y would have to be inserted in the 
#        # sorted array.
#        self.xind = np.searchsorted(xscale, x)
#        self.yind = np.searchsorted(yscale, y)
#
#        # Find out whether the lower or upper 'neighbour' is closest
#        x_lower = xscale[self.xind-1]
#        y_lower = yscale[self.yind-1]
#        # NOTE In principle, these IndexErrors shouldn't occur. Try catch 
#        # only helps when debugging.
#        try :
#            x_upper = xscale[self.xind]
#        except IndexError :
#            x_upper = max(xscale)
#        try :
#            y_upper = yscale[self.yind]
#        except IndexError :
#            y_upper = max(yscale)
#
#        dx_upper = x_upper - x
#        dx_lower = x - x_lower
#        dy_upper = y_upper - y
#        dy_lower = y - y_lower
#
#        # Assign the exact data value and update self.xind/yind if necessary
#        if dx_upper < dx_lower :
#            x_snap = x_upper
#        else :
#            x_snap = x_lower
#            self.xind -= 1
#            
#        if dy_upper < dy_lower :
#            y_snap = y_upper
#        else :
#            y_snap = y_lower
#            self.yind -= 1
#
#        return x_snap, y_snap
#
#    def plot_cursors(self) :
#        """ Plot the cursors in the bottom left axis. """
#        # Delete current cursors (NOTE: this is dangerous if there are any 
#        # other lines in the plot)
#        ax = self.axes['map']
#        ax.lines = []
#
#        # Retrieve information about current data range
#        xmin, xmax, ymin, ymax = self.get_xy_minmax()
#       
#        xlimits = [xmin, xmax]
#        ylimits = [ymin, ymax]
#
#        # Initiate cursors in the center of graph if necessary
#        if self.cursor_xy is None :
#            x = 0.5 * (xmax + xmin)
#            y = 0.5 * (ymax + ymin)
#
#            # Keep a handle on cursor positions
#            self.cursor_xy = (x, y)
#        else : 
#            x, y = self.cursor_xy
#
#        # Make the cursor snap to actual data points
#        x, y = self.snap_to(x, y)
#
#        # Plot cursors and keep handles on them (need the [0] because plot() 
#        # returns a list of Line objects)
#        self.xcursor = ax.plot([x, x], ylimits, zorder=3, **cursor_kwargs)[0]
#        self.ycursor = ax.plot(xlimits, [y, y], zorder=3, **cursor_kwargs)[0]
#
    def _set_up_event_handling(self) :
        """ Define what happens when user clicks in the plot (move cursors to 
        clicked position) or presses an arrow key (move cursors in specified 
        direction [<- not implemented]). 
        """
        cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        pid = self.figure.canvas.mpl_connect('key_press_event', self.on_press)

    def on_click(self, event):
        # Stop if we're not in the right plot
        if event.inaxes != self :
            return
        #print('Clicked')
        # Don't do anything if someone else is drawing
        if not self.figure.canvas.widgetlock.available(self) :
            return

        # Get the x, y data of the click
        x, y = (event.xdata, event.ydata)

        # Remove the old cursor
        try :
            self.cursor_x.remove()
            self.cursor_y.remove()
        except AttributeError :
            pass
        except ValueError :
            pass

        # Get the extent of the axes
        xmin, xmax, ymin, ymax = self.get_xy_minmax()

        kwargs = {'color': self.color,
                  'lw': self.lw}

        # Plot new cursors
        self.cursor_x = self.plot([x, x], [ymin, ymax], **kwargs)[0]
        self.cursor_y = self.plot([xmin, xmax], [y, y], **kwargs)[0]

        # Reset the limits, because plotting the cursor likely changed them
        self.set_xlim((xmin, xmax))
        self.set_ylim((ymin, ymax))

        self.figure.canvas.draw()
        #self.figure.canvas.blit(self.bbox)

    def on_press(self, event):
        # Get the name of the pressed key and info on the current cursors
        #key = event.key
        #print(key)
        pass
#            x, y = self.cursor_xy
#            xmin, xmax, ymin, ymax = self.get_xy_minmax()
#
#            # Stop if no arrow key was pressed
#            if key not in ['up', 'down', 'left', 'right'] : return
#
#            # Move the cursor by one unit in data points
#            xscale, yscale = self.get_xy_scales()
#            dx = xscale[1] - xscale[0]
#            dy = yscale[1] - yscale[0]
#
#            # In-/decrement cursor positions depending on what button was 
#            # pressed and only if we don't leave the axis
#            if key == 'up' and y+dy <= ymax :
#                y += dy
#            elif key == 'down' and y-dy >= ymin :
#                y -= dy
#            elif key == 'right' and x+dx <= xmax :
#                x += dx
#            elif key == 'left' and x-dx >= xmin:
#                x -= dx
#
#            # Update the cursor position and redraw it
#            self.cursor_xy = (x, y)
#            self.plot_cursors()
#            # Now the cuts have to be redrawn as well
#            self.plot_cuts()
#            self.canvas.draw()

class cursorpolyax(cursorax) :
    """ 
    A cursorax that allows drawing of a draggable polygon-ROI. 

    By clicking on the plot, a cursor appears which behaves and can be 
    accessed the same way as in :class: `cursorax <kustom.plotting.cursorax>`.
    Additionally, hitting the `draw_key` (`d` by default) puts user in 
    `polygon-draw mode` where each subsequent click on the plot adds another 
    corner to a polygon until it is completed by clicking on the starting
    point again. 
    Once finished, each click just moves the cursor, as before. Hitting the 
    `remove_key` (`e` by default) removes the polygon from the plot.
    At the moment of the polygon's completion, the function :func: 
    `on_polygon_complete <kustom.plotting.cursorpolyax.on_polygon_complete>` 
    is executed. This function is a stub in the class definition and can be 
    overwritten/reassigned by the user to perform any action desired on 
    polygon completion.
    The vertices of the last completed polygon are present as an argument to 
    :func: `on_polygon_complete 
    <kustom.plotting.cursorpolyax.on_polygon_complete>` and can also be 
    accessed by :attr: `vertices` at any time.

    #The actual magic here is done by :class: `PolygonSelector 
    #<matplotlib.widgets.PolygonSelector>`. This class mostly provides a 
    #simple interface for ...

    Known bugs:
    -----------
    * Using :class: `PolygonSelector 
      <matplotlib.widgets.PolygonSelector>`'s default 'remove' key (Esc) 
      messes up reaction to :class: `cursorpolyax 
      <kustom.plotting.cursorpolyax`' keybinds.
    * Shift-dragging polygon makes the cursor jump.
    """
    # The name under which this class of axes will be accessible from matplotlib
    name = 'cursorpoly'

    poly = None
    vertices = None
    polylineprops = dict(color='r', lw='1')
    polymarkerprops = dict(marker='None')

    draw_key = 'd'
    remove_key = 'e'

    # Blitting leads to weird behaviour
    useblit = False

    #first_time_complete = True

    def __init__(self, *args, **kwargs) :
        """ 
        The super-class's __init__ method connects the event handling but is 
        going to use the definitions for :func: `on_click 
        <kustom.plotting.cursorpolyax.on_click>` and :func: `on_press 
        <kustom.plotting.cursorpolyas.on_press>` from this class. 
        """
        #cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        #pid = self.figure.canvas.mpl_connect('key_press_event', on_press)
        super().__init__(*args, **kwargs)

    def on_press(self, event) :
        """ 
        Handle keyboard press events. If the pressed key matches :attr: 
        `draw_key <cursorax.draw_key>` or :attr: `remove_key 
        <cursorpolyax.remove_key>` and the figure is not draw-locked, carry 
        out the respective operations.
        """
        # Don't do anything if someone else is drawing
        if not self.figure.canvas.widgetlock.available(self.poly) :
            return

        if event.key == self.remove_key :
            # Remove the polygon and release the lock
            self.remove_polygon()
            return

        elif event.key == self.draw_key :
            self.enter_draw_mode()

    def enter_draw_mode(self) :
        """ 
        Ensure that the next click after this fcn call will start drawing 
        the polygon. 
        """
        # Remove the previous polygon
        if self.poly and self.poly._polygon_completed :
            self.remove_polygon()

        # Reset the flag indicating the first completion of a new polygon
        #self.first_time_complete = True

        # Create a PolygonSelector object and attach the draw lock to 
        # it 
        self.poly = PolygonSelector(self, self._on_polygon_complete, 
                                    lineprops=self.polylineprops, 
                                    markerprops=self.polymarkerprops,
                                    useblit=self.useblit)
        self.figure.canvas.widgetlock(self.poly)

    def on_click(self, event) :
        """ 
        Handle mouse-click events. Just call the superclass' on_click 
        method, which positions the cursor at the clicked location. That 
        method check's itself whether the draw lock is free, so we don't get 
        cursor jumps while we're drawing a polygon. 
        """
        # Release the draw lock if the polygon has been completed. Otherwise, 
        # the cursor can't be repositioned.
        if self.poly and self.poly._polygon_completed :
            self.figure.canvas.widgetlock.release(self.poly)
        super().on_click(event)

    def remove_polygon(self) :
        """ 
        Make the polygon invisible, remove the reference to it (which 
        should cause the underlying :class: `PolygonSelector 
        <matplotlib.widgets.PolygonSelector>` object to be garbage collected) 
        and release the draw lock.
        """
        if not self.poly : return
        try :
            self.figure.canvas.widgetlock.release(self.poly)
        except :
            pass
        self.poly.set_visible(False)
        self.poly = None
        self.figure.canvas.draw()

    def _on_polygon_complete(self, vertices) :
        """ 
        Get a handle on the polygon's vertices and call the user-supplied 
        :func: `on_polygon_complete 
        <kustom.plotting.cursorpolyas.on_polygon_complete>`. 
        """
        self.vertices = vertices
        # Only do this the first time the polygon is completed NOTE If we 
        # only do this the first time the polygon is completed, the function 
        # wont be called when user moves the polygon with shift+drag. The 
        # drawback of the way it is now is that this function gets called 
        # every time the user clicks on the plot once the polygin has been 
        # created....
        #if self.first_time_complete :
        self.on_polygon_complete(vertices)
        #self.first_time_complete = False
        #self.figure.canvas.draw_idle()

    def on_polygon_complete(self, vertices) :
        """ This method should be overridden/redefined by user. """
        print(vertices)

# Register the cursorax upon import
register_projection(cursorax)
register_projection(cursorpolyax)

if __name__ == "__main__" :
    import matplotlib.pyplot as plt
    figsize=(10,1)
    fig1 = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(111, projection='cursor')
    ax1 = fig1.add_axes([0,0,1,0.5])
    #fig2 = plt.figure(figsize=figsize)
    ax2 = fig1.add_axes([0,0.5,1,0.5])
    ax2.set_xticks([])

    r = range(256)
    d = np.array([r])
    ax1.pcolormesh(d, cmap='viridis')
    ax2.pcolormesh(d, cmap='kocean')

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111, projection='cursorpoly')

    plt.show()

