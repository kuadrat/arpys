""" matplotlib pcolormesh equivalent in pyqtgraph (more or less) """

import logging

import pyqtgraph as pg
from numpy import clip, inf, ndarray, inf
from pyqtgraph import Qt as qt #import QtCore
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.widgets import PlotWidget, GraphicsView

from arpys.pit.utilities import TracedVariable

logger = logging.getLogger('pit.'+__name__)

class ImagePlot(pg.PlotWidget) :
    """
    A PlotWidget which mostly contains a single 2D image (intensity 
    distribution) or a 3D array (distribution of RGB values) as well as all 
    the nice pyqtgraph axes panning/rescaling/zooming functionality.

    ============================================================================
    *Signals*
    sig_image_changed  emitted whenever the image is updated
    sig_axes_changed   emitted when the axes are updated
    ============================================================================
    """
    image_item = None
    image_kwargs = {}
    xlim = None
    ylim = None
    sig_image_changed = qt.QtCore.Signal()
    sig_axes_changed = qt.QtCore.Signal()

    def __init__(self, image=None, parent=None, background='default', 
                 **kwargs) :
        """ Allows setting of the image upon initialization. 
        
        ==========  ============================================================
        image       np.ndarray or pyqtgraph.ImageItem instance; the image to be
                    displayed.
        parent      QtWidget instance; parent widget of this widget.
        background  str; confer PyQt documentation
        ==========  ============================================================
        """
        super().__init__(parent=parent, background=background, **kwargs) 
        if image is not None :
            self.set_image(image)

        self.sig_axes_changed.connect(self.fix_viewrange)

    def remove_image(self) :
        """ Removes the current image using the parent's :func: `removeItem` 
        function. 
        """
        if self.image_item is not None :
            self.removeItem(self.image_item)
        self.image_item = None

    def set_image(self, image, *args, **kwargs) :
        """ Expects both, np.arrays and pg.ImageItems as input and sets them 
        correctly to this PlotWidget's Image with `addItem`. Also makes sure 
        there is only one Image by deleting the previous image.

        ======  ================================================================
        image   np.ndarray or pyqtgraph.ImageItem instance; the image to be
                displayed.
        args    positional and keyword arguments that are passed on to :class:
        kwargs  `ImageItem <pyqtgraph.graphicsItems.ImageItem.ImageItem>`
        ======  ================================================================
        """
        # Convert array to ImageItem
        if isinstance(image, ndarray) :
            image = ImageItem(image, *args, **kwargs)
        # Throw an exception if image is not an ImageItem
        if not isinstance(image, ImageItem) :
            message = '''`image` should be a np.array or pg.ImageItem instance,
            not {}'''.format(type(image))
            raise TypeError(message)

        # Replace the image
        self.remove_image()
        self.image_item = image
        logger.debug('Setting image.')
        self.addItem(image)
        self._set_axes_scales()

        self.sig_image_changed.emit()

    def set_xscale(self, xscale, update=False) :
        """ Set the xscale of the plot. *xscale* is an array of the length 
        ``len(self.image_item.shape[0])``.
        """
        # Sanity check
        if len(xscale) != self.image_item.image.shape[0] :
            raise TypeError('Shape of xscale does not match data dimensions.')

        self.xscale = xscale
        # 'Autoscale' the image to the xscale
        self.xlim = (xscale[0], xscale[-1])

        if update :
            self._set_axes_scales()

    def set_yscale(self, yscale, update=False) :
        """ Set the yscale of the plot. *yscale* is an array of the length 
        ``len(self.image_item.image.shape[1])``.
        """
         # Sanity check
        if len(yscale) != self.image_item.image.shape[1] :
            raise TypeError('Shape of yscale does not match data dimensions.')

        self.yscale = yscale
        # 'Autoscale' the image to the xscale
        self.ylim = (yscale[0], yscale[-1])

        if update :
            self._set_axes_scales()

    def _set_axes_scales(self) :
        """ Transform the image such that it matches the desired x and y 
        scales.
        """
        # Get image dimensions and requested origin (x0,y0) and top right 
        # corner (x1, y1)
        logger.debug(('_set_axes_scales(): self.image_item.image.shape={}' + 
                     '').format(self.image_item.image.shape))
        nx, ny = self.image_item.image.shape
        if self.xlim is not None :
            x0, x1 = self.xlim
        else :
            x0, x1 = 0, nx-1
        if self.ylim is not None :
            y0, y1 = self.ylim
        else :
            y0, y1 = 0, ny-1
        # Calculate the scaling factors
        sx = (x1-x0)/nx
        sy = (y1-y0)/ny
        # Define a transformation matrix that scales and translates the image 
        # such that it appears at the coordinates that match our x and y axes.
        transform = qt.QtGui.QTransform()
        transform.scale(sx, sy)
        # Carry out the translation in scaled coordinates
        transform.translate(x0/sx, y0/sy)
        # Finally, apply the transformation to the imageItem
        self.image_item.setTransform(transform)

#        self.sig_axes_changed.emit()

    def get_limits(self) :
        """ Return ``[[x_min, x_max], [y_min, y_max]]``. """
         # Default to current viewrange but try to get more accurate values if 
        # possible
        [[x_min, x_max], [y_min, y_max]] = self.viewRange()
        if self.xlim is not None :
            x_min, x_max = self.xlim
        if self.ylim is not None :
            y_min, y_max = self.ylim

        logger.debug(('get_limits(): [[x_min, x_max], [y_min, y_max]] = '
                    + '[[{}, {}], [{}, {}]]').format(x_min, x_max, y_min, 
                                                     y_max))
        return [[x_min, x_max], [y_min, y_max]]

    def fix_viewrange(self) :
        """ Prevent zooming out by fixing the limits of the ViewBox. """
        [[x_min, x_max], [y_min, y_max]] = self.get_limits()
        self.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max,
                      maxXRange=x_max-x_min, maxYRange=y_max-y_min)

    def release_viewrange(self) :
        """ Undo the effects of :func: `fix_viewrange 
        <pit.imageplot.ImagePlot.fix_viewrange>`
        """
        self.setLimits(xMin=-inf,
                       xMax=inf,
                       yMin=-inf,
                       yMax=inf,
                       maxXRange=inf,
                       maxYRange=inf)

class CursorPlot(pg.PlotWidget) :
    """ Implements a simple, draggable scalebar represented by a line 
    (:class: `InfiniteLine <pyqtgraph.InfiniteLine>) on an axis (:class: 
    `PlotWidget <pyqtgraph.PlotWidget>).
    The current position of the slider is tracked with the :class: 
    `TracedVariable <arpys.pit.utilities.TracedVariable>` self.pos.
    """
    # The speed at which the slider moves on mousewheel scroll in units of 
    # % of total range
    wheel_sensitivity = 0.5

    def __init__(self, parent=None, background='default', **kwargs) :
        """ Initialize the slider and set up the visual tweaks to make a 
        PlotWidget look more like a scalebar.

        ==========  ============================================================
        parent      QtWidget instance; parent widget of this widget.
        background  str; confer PyQt documentation
        ==========  ============================================================
        """
        super().__init__(parent=parent, background=background, **kwargs) 

        # Hide the pyqtgraph auto-rescale button
        self.getPlotItem().buttonsHidden = True

        # The position of the slider is stored with a TracedVariable
        initial_pos = 0
        pos = TracedVariable(initial_pos)
        self.register_traced_variable(pos)

        # Set up the slider
        self.slider = pg.InfiniteLine(initial_pos, movable=True)
        self.slider.setPen((255,255,0,255))
        # Add a marker. Args are (style, position (from 0-1), size #NOTE 
        # seems broken
        #self.slider.addMarker('o', 0.5, 10)
        self.addItem(self.slider)

        # Disable mouse scrolling, panning and zooming for both axes
        self.setMouseEnabled(False, False)

        # Initialize range to [0, 1]
        self.set_bounds(initial_pos, initial_pos + 1)

        # Connect a slot (callback) to dragging and clicking events
        self.slider.sigDragged.connect(self.on_position_change)
        # sigMouseReleased seems to not work (maybe because sigDragged is used)
        #self.sigMouseReleased.connect(self.onClick)
        # The inherited mouseReleaseEvent is probably used for sigDragged 
        # already. Anyhow, overwriting it here leads to inconsistent behaviour.
        #self.mouseReleaseEvent = self.onClick

    def wheelEvent(self, event) :
        """ Override of the Qt wheelEvent method. Fired on mousewheel 
        scrolling inside the widget. Change the position of the 
        """
        # Get the relevant coordinate of the mouseWheel scroll
        delta = event.pixelDelta().y()
        if delta > 0 :
            sign = -1
        elif delta < 0 :
            sign = 1
        else :
            # It seems that in some cases delta==0
            sign = 0
        self.pos.set_value(self.pos.get_value() + sign * 
                           self.wheel_frames) 

    def register_traced_variable(self, traced_variable) :
        """ Set self.pos to the given TracedVariable instance and connect the 
        relevant slots to the signals. This can be used to share a 
        TracedVariable among widgets.
        """
        self.pos = traced_variable
        self.pos.sig_value_changed.connect(self.set_position)
        self.pos.sig_allowed_values_changed.connect(self.on_allowed_values_change)
        # Set the bounds to the current values of this TracedVar, if existent
        #self.on_allowed_values_changed()

    def on_position_change(self) :
        """ Callback for the :signal: `sigDragged 
        <pyqtgraph.InfiniteLine.sigDragged>`. Set the value of the 
        TracedVariable instance self.pos to the current slider position. 
        """
        current_pos = self.slider.value()
        # NOTE pos.set_value emits signal sig_value_changed which may lead to 
        # duplicate processing of the position change.
        self.pos.set_value(current_pos)

    def on_allowed_values_change(self) :
        """ Callback for the :signal: `sig_allowed_values_changed
        <pyqtgraph.utilities.TracedVariable.sig_allowed_values_changed>`. 
        With a change of the allowed values in the TracedVariable, we should 
        update our bounds accordingly.
        """
        # If the allowed values were reset, just exit
        if self.pos.allowed_values is None : return

        lower = self.pos.min_allowed
        upper = self.pos.max_allowed
        self.set_bounds(lower, upper)

    def set_position(self) :
        """ Callback for the :signal: `sig_value_changed 
        <arpys.pit.utilities.TracedVariable.sig_value_changed>`. Whenever the 
        value of this TracedVariable is updated (possibly from outside this 
        Scalebar object), put the slider to the appropriate position.
        """
        new_pos = self.pos.get_value()
        self.slider.setValue(new_pos)

    def set_bounds(self, lower, upper) :
        """ Set both, the displayed area of the axis as well as the the range 
        in which the slider (InfiniteLine) can be dragged to the interval 
        [lower, upper]. 
        """
        self.setXRange(lower, upper, padding=0.01)
        self.slider.setBounds([lower, upper])

        # When the bounds update, the mousewheelspeed should change accordingly
        self.wheel_frames = 0.01 * self.wheel_sensitivity * (upper-lower)

class Scalebar(CursorPlot) :
    """ Simple subclass of :class: `CursorPlot 
    <arpys.pit.imageview.CursorPlot>` that is intended to simulate a 
    scalebar. This is achieved by providing simply a long, flat plot without 
    any data and no y axis, but the same draggable slider as in CursorPlot.
    """
    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)

        # Aesthetics and other widget configs
        self.hideAxis('left')
        self.set_size(300, 50)

    def set_size(self, width, height) :
        """ Set this widgets size by setting minimum and maximum sizes 
        simultaneously to the same value. 
        """
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)

# Deprecated
class ImagePlotWidget(qt.QtGui.QWidget) :
    """ A widget that contains an :class: `ImagePlot3d 
    <arpys.pit.imageplot.ImagePlot3d> as its main element accompanied by a 
    scalebar representing the z dimension. 

    .. :deprecated:
    """

    default_geometry = (0, 50, 400, 400)

    def __init__(self, image=None, parent=None, background='default', 
                 **kwargs) :
        super().__init__(parent=parent)
        # Create the ImagePlot3d instance
        self.imagePlot3d = ImagePlot3d(image=image, parent=self, 
                                       background=background, **kwargs) 

        # Explicitly wrap methods and attributes from ImagePlot3d
        # NOTE: If you change this list, update the documentation above as 
        # well.
        for m in ['z', 'on_z_change', 'image_data', 'set_image',
                  'remove_image', 'update_image_slice'] :
            setattr(self, m, getattr(self.imagePlot3d, m))

        # Use a GridLayout to align the widgets
        self.build_layout()

        # Set the initial size and screen position
        self.setGeometry(*self.default_geometry)

        # Build and add the scalebar
        self.create_scalebar()

    def build_layout(self) :
        """ """
        self.layout = qt.QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)

        # Resoize imagePlot3d
        #self.imagePlot3d.resize( self.imagePlot3d.sizeHint() )
        #self.imagePlot3d.resize(400, 400)
        self.imagePlot3d.setMinimumSize(200,200)
        self.layout.addWidget(self.imagePlot3d, 0, 0, 3, 1)

    def create_scalebar(self) :
        self.pi = pg.PlotWidget()
        self.pi.setMaximumSize(10000, 100)
        self.layout.addWidget(self.pi, 4, 0, 1, 1)




