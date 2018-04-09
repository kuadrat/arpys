""" matplotlib pcolormesh equivalent in pyqtgraph """

from numpy import clip, inf, ndarray
import pyqtgraph as pg
from pyqtgraph import Qt as qt #import QtCore
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.widgets import PlotWidget, GraphicsView

from .utilities import TracedVariable

class ImagePlot(pg.PlotWidget) :
    """
    A PlotWidget which mostly contains a single 2D image (intensity 
    distribution) or a 3D array (distribution of RGB values) as well as all 
    the nice pyqtgraph axes panning/rescaling/zooming functionality.
    """
    image = None

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
            self.setImage(image)

    def removeImage(self) :
        """ Removes the current image using the parent's :func: `removeItem` 
        function. 
        """
        if self.image is not None :
            self.removeItem(self.image)
        self.image = None

    def setImage(self, image, *args, **kwargs) :
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
        self.removeImage()
        self.image = image
        self.addItem(image)

    def getImage(self) :
        """ Getter method for self.image. """
        return self.image

#class ImagePlot3d(ImagePlot, qt.QtGui.QGraphicsWidget):
class ImagePlot3d(ImagePlot):
    """
    Display one slice of a 3D dataset at a time. Allow scrolling through the 
    data with the keyboard.
    """

    def __init__(self, image=None, parent=None, background='default', 
                 **kwargs) :
        """ Call super() and connect signals. See doc of :func: ` __init__() 
        <arpys.pit.imageplot.ImagePlot.__init__> for explanation of 
        arguments.  
        """ 
        # Initialize self.z before the call to super() because super's 
        # __init__ will call self.setImage which accesses self.z
        self.z = TracedVariable()
        self.z.sigValueChanged.connect(self.zChanged)
        super().__init__(image=image, parent=parent, background=background, 
                         **kwargs) 

    def sizeHint(self) :
        """ NOTE: Doesn't seem to do much. """
        size = qt.QtCore.QSize(378, 310)
        return size

    def setImage(self, image, axes=(0,1), **imageKwargs) :
        """ As opposed to :class: `ImagePlot <pit.imageplot.ImagePlot>` (i.e.
        the parent), this only accepts 3D np.arrays. Also determines the z 
        range and resets the `z` value to 0.

        ===========  ===========================================================
        image        3D np.ndarray; the data of which slices are to be 
                     displayed.
        axes         tuple of int; axis indices of the x and y axis to be 
                     used. For example if axis=(0,2) then the shape of 
                     `image` is assumed to be (x,z,y).
        imageKwargs  keyword arguments that are passed on to :class:
                     `ImageItem <pyqtgraph.graphicsItems.ImageItem.ImageItem>` 
                     when creating ImageItems from the data.
        ===========  ===========================================================
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

        self.image_data = image
        self.axes = axes
        self.imageKwargs = imageKwargs

        # Initialize z to 0, taking the first slice of the data
        self.z.set_value(0)
        self.updateImageSlice()
        
    def updateImageSlice(self) :
        """ Update the currently displayed image slice by deleting the old 
        `self.image` and using :func: `addItem 
        <pit.imageplot.ImagePlot3d.addItem>' to set the newly displayed image 
        according to the current value of `self.z`.
        """
        # Clear plot from the old ImageItem
        self.removeImage()

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
        self.image = ImageItem(image, **self.imageKwargs)
        self.addItem(self.image)

    def zChanged(self, caller=None) :
        """ Callback to the :signal: `sigZChanged`. Ensure self.z does not go 
        out of bounds and update the Image slice with a call to :func: 
        `updateImageSlice <arpys.pit.imageplot.ImagePlot3d.updateImageSlice>`.
        """
        # Ensure z doesn't go out of bounds
        z = self.z.get_value()
        clipped_z = clip(z, self.zmin, self.zmax)
        if z != clipped_z :
            # NOTE this leads to unnecessary signal emitting. Should avoid 
            # emitting the signal from inside a slot (function connected to 
            # that signal)
            self.z.set_value(clipped_z)
        self.updateImageSlice()

    def keyPressEvent(self, event) :
        """ Handle keyboard input. """
        key = event.key()
        z = self.z.get_value()
        if key == qt.QtCore.Qt.Key_Right :
            z += 1
        elif key == qt.QtCore.Qt.Key_Left :
            z -= 1
        self.z.set_value(z)

class ImagePlotWidget(qt.QtGui.QWidget) :
    """ A widget that contains an :class: `ImagePlot3d 
    <arpys.pit.imageplot.ImagePlot3d> as its main element accompanied by a 
    scalebar representing the z dimension. 
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
        for m in ['z', 'zChanged', 'image_data', 'setImage',
                  'removeImage', 'updateImageSlice'] :
            setattr(self, m, getattr(self.imagePlot3d, m))

        # Use a GridLayout to align the widgets
        self.buildLayout()

        # Set the initial size and screen position
        self.setGeometry(*self.default_geometry)

        # Build and add the scalebar
        self.createScalebar()

    def buildLayout(self) :
        """ """
        self.layout = qt.QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)

        # Resoize imagePlot3d
        #self.imagePlot3d.resize( self.imagePlot3d.sizeHint() )
        #self.imagePlot3d.resize(400, 400)
        self.imagePlot3d.setMinimumSize(200,200)
        self.layout.addWidget(self.imagePlot3d, 0, 0, 3, 1)

    def createScalebar(self) :
        self.pi = pg.PlotWidget()
        self.pi.setMaximumSize(10000, 100)
        self.layout.addWidget(self.pi, 4, 0, 1, 1)




