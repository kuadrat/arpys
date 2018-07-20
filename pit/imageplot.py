""" matplotlib pcolormesh equivalent in pyqtgraph (more or less) """

from numpy import clip, inf, ndarray, inf
import pyqtgraph as pg
from pyqtgraph import Qt as qt #import QtCore
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.widgets import PlotWidget, GraphicsView

from arpys.pit.utilities import TracedVariable
from arpys.pit.cursor import Cursor

class ImagePlot(pg.PlotWidget) :
    """
    A PlotWidget which mostly contains a single 2D image (intensity 
    distribution) or a 3D array (distribution of RGB values) as well as all 
    the nice pyqtgraph axes panning/rescaling/zooming functionality.
    """
    image = None
    sig_image_changed = qt.QtCore.Signal()

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

    def remove_image(self) :
        """ Removes the current image using the parent's :func: `removeItem` 
        function. 
        """
        if self.image is not None :
            self.removeItem(self.image)
        self.image = None

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
            raise Exception(message)

        # Replace the image
        self.remove_image()
        self.image = image
        self.addItem(image)

        self.sig_image_changed.emit()

    def fix_viewrange(self) :
        """ Prevent zooming out by fixing the limits of the ViewBox. """
        [[xMin, xMax], [yMin, yMax]] = self.viewRange()
        self.setLimits(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax,
                      maxXRange=xMax-xMin, maxYRange=yMax-yMin)

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

#class ImagePlot3d(ImagePlot, qt.QtGui.QGraphicsWidget):
class ImagePlot3d(ImagePlot):
    """
    Display one slice of a 3D dataset at a time. Allow scrolling through the 
    data with the keyboard.
    """

    def __init__(self, image=None, axes=(0,1), parent=None, 
                 background='default', **kwargs) :
        """ Call super() and connect signals. See doc of :func: ` __init__() 
        <arpys.pit.imageplot.ImagePlot.__init__> for explanation of 
        arguments.  
        """ 
        # Initialize self.z before the call to super() because super's 
        # __init__ will call self.set_image which accesses self.z
        z = TracedVariable()
        self.register_traced_variable(z)
        super().__init__(parent=parent, background=background, **kwargs) 
        if image is not None :
            self.set_image(image, axes)

    def register_traced_variable(self, traced_variable) :
        """ Set self.z to the given TracedVariable instance and connect the 
        relevant slots to the signals. 
        """
        self.z = traced_variable
        self.z.sig_value_changed.connect(self.on_z_change)

    def sizeHint(self) :
        # NOTE: Doesn't seem to do much.
        size = qt.QtCore.QSize(378, 310)
        return size

    def set_image(self, image, axes=(0,1), **image_kwargs) :
        """ As opposed to :class: `ImagePlot <pit.imageplot.ImagePlot>` (i.e.
        the parent), this only accepts 3D np.arrays. Also determines the z 
        range and resets the `z` value to 0.

        ============  ==========================================================
        image         3D np.ndarray; the data of which slices are to be 
                      displayed.
        axes          tuple of int; axis indices of the x and y axis to be 
                      used. For example if axis=(0,2) then the shape of 
                      *image* is assumed to be (x,z,y).
        image_kwargs  keyword arguments that are passed on to :class:
                      `ImageItem <pyqtgraph.graphicsItems.ImageItem.ImageItem>` 
                      when creating ImageItems from the data.
        ============  ==========================================================
        """
        # Test the shape of the input
        if image.ndim is not 3 :
            m = '`image` must have ndim==3, got {}.'.format(image.ndim)
            raise ValueError(m)

        # Get the axis indices
        self.xaxis, self.yaxis = axes
        # z is the remaining out of [0,1,2]
        l = [0,1,2]
        for a in axes :
            try :
                l.remove(a)
            except ValueError :
                m = '`axis` elements must be one of [0,1,2]. Got {}.'.format(a)
                raise ValueError(m)
        self.zaxis = l[0]

        # Determine the new ranges for z
        self.zmin = 0
        self.zmax = image.shape[self.zaxis] - 1

        # Update the allowed values for z
        self.z.set_allowed_values(range(self.zmin, self.zmax+1))

        self.image_data = image
        self.axes = axes
        self.image_kwargs = image_kwargs

        # Initialize z to 0, taking the first slice of the data
        self.z.set_value(0)
        self.update_image_slice()

        # Fix the scales to prevent zooming out
        self.fix_viewrange()
        
        self.sig_image_changed.emit()

    def update_image_slice(self) :
        """ Update the currently displayed image slice by deleting the old 
        `self.image` and using :func: `addItem 
        <pit.imageplot.ImagePlot3d.addItem>' to set the newly displayed image 
        according to the current value of `self.z`.
        """
        # Clear plot from the old ImageItem
        self.remove_image()

        # Extract the slice from the image data, depending on how our axes 
        # are defined
        z = self.z.get_value()
        if self.zaxis == 0 :
            image = self.image_data[z,:,:]
        elif self.zaxis == 1 :
            image = self.image_data[:,z,:]
        elif self.zaxis == 2 :
            image = self.image_data[:,:,z]

        # Convert to ImageItem and add
        self.image = ImageItem(image, **self.image_kwargs)
        self.addItem(self.image)

    def on_z_change(self, caller=None) :
        """ Callback to the :signal: `sig_z_changed`. Ensure self.z does not go 
        out of bounds and update the Image slice with a call to :func: 
        `update_image_slice <arpys.pit.imageplot.ImagePlot3d.update_image_slice>`.
        """
        # Ensure z doesn't go out of bounds
        z = self.z.get_value()
        clipped_z = clip(z, self.zmin, self.zmax)
        if z != clipped_z :
            # NOTE this leads to unnecessary signal emitting. Should avoid 
            # emitting the signal from inside a slot (function connected to 
            # that signal)
            self.z.set_value(clipped_z)
        self.update_image_slice()

    def keyPressEvent(self, event) :
        """ Handle keyboard input. """
        key = event.key()
        z = self.z.get_value()
        if key == qt.QtCore.Qt.Key_Right :
            z += 1
        elif key == qt.QtCore.Qt.Key_Left :
            z -= 1
        self.z.set_value(z)

class Scalebar(pg.PlotWidget) :
    """ Implements a simple, draggable scalebar represented by a line 
    (:class: `InfiniteLine <pyqtgraph.InfiniteLine>) on an axis (:class: 
    `PlotWidget <pyqtgraph.PlotWidget>).
    The current position of the slider is tracked with the :class: 
    `TracedVariable <arpys.pit.utilities.TracedVariable>` self.pos.
    """

    def __init__(self, parent=None, background='default', **kwargs) :
        """ Initialize the slider and set up the visual tweaks to make a 
        PlotWidget look more like a scalebar.

        ==========  ============================================================
        parent      QtWidget instance; parent widget of this widget.
        background  str; confer PyQt documentation
        ==========  ============================================================
        """
        super().__init__(parent=parent, background=background, **kwargs) 

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

        # Aesthetics and other widget configs
        self.hideAxis('left')
        self.setSize(300, 50)
        # Disable mouse scrolling, panning and zooming for both axes
        self.setMouseEnabled(False, False)

        # Initialize range to [0, 1]
        self.set_bounds(initial_pos, initial_pos + 1)

        # Set the speed at which the slider moves on mousewheel scroll
        # in units of % of total range
        self.wheel_sensitivity = 0.5

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

    def onClick(self, *args) :
        """ For testing. """
        print(args)
        print('Clicked')

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

        # When the bounds update, the mousewheelspeed should change accordingly
        self.wheel_frames = 0.01 * self.wheel_sensitivity * (upper-lower)

    def set_position(self) :
        """ Callback for the :signal: `sig_value_changed 
        <arpys.pit.utilities.TracedVariable.sig_value_changed>`. Whenever the 
        value of this TracedVariable is updated (possibly from outside this 
        Scalebar object), put the slider to the appropriate position.
        """
        new_pos = self.pos.get_value()
        self.slider.setValue(new_pos)

    def setSize(self, width, height) :
        """ Set this widgets size by setting minimum and maximum sizes 
        simultaneously to the same value. 
        """
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)

    def set_bounds(self, lower, upper) :
        """ Set both, the displayed area of the axis as well as the the range 
        in which the slider (InfiniteLine) can be dragged to the interval 
        [lower, upper]. 
        """
        self.setXRange(lower, upper, padding=0.01)
        self.slider.setBounds([lower, upper])

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




