
import logging

import pyqtgraph as pg
from pyqtgraph import Qt as qt
from pyqtgraph import QtGui, Point
from pyqtgraph.functions import affineSlice

logger = logging.getLogger('pit.'+__name__)

class Cutline(qt.QtCore.QObject) :
    """ Wrapper class allowing easy adding and removing of :class: 
    `LineSegmentROI <pyqtgraph.LineSegmentROI>`s to a :class: `PlotWidget 
    <pyqtgraph.PlotWidget>`.
    It both
    has-a LineSegmentROI
    and
    has-a PlotWidget
    and handles interactions between the two.

    Needs to inherit from :class: `QObject <pyqtgraph.Qt.QtCore.QObject>` in 
    order to have signals.

    ==================  ========================================================
    *Signals*
    sig_region_changed  wraps the underlying :class: `LineSegmentROI 
                        <pyqtgraph.LineSegmentROI>`'s sigRegionChange. 
                        Emitted whenever the ROI is moved or changed.
    sig_initialized     emitted when a new :class: 
                        `LineSegmentROI <pyqtgraph.LineSegmentROI>` has been 
                        created and assigned as this :class: `Cutline`'s `roi`.
    ==================  ========================================================
    """

    sig_initialized = qt.QtCore.Signal()

    def __init__(self, plot_widget=None, orientation='horizontal', **kwargs) :
        super().__init__(**kwargs)
        if plot_widget :
            self.add_to_plot(plot_widget)
        self.orientation = orientation
        self.roi = None

    def add_to_plot(self, plot_widget) :
        """ Add this cutline to a :class: `PlotWidget <pyqtgraph.PlotWidget>`.
        This is effectively implemented by setting this :class: `Cutline 
        <arpys.pit.cutline.Cutline>`s plot attribute to the given *plot_widget*.
        """
        self.plot = plot_widget
        # Signal connection: whenever the viewRange changes, the cutline should 
        # be updated. Make sure to not accumulate connections by trying to 
        # disconnect first.
        try :
            self.plot.sigRangeChanged.disconnect(self.initialize)
        except TypeError :
            pass
        self.plot.sig_axes_changed.connect(self.initialize)

    def initialize(self, orientation=None) :
        """ Emits :signal: `sig_initialized`. """
        logger.debug('initialize()')
        # Change the orientation if one is given
        if orientation :
            self.orientation = orientation

        # Remove the old LineSegmentROI if necessary
        self.plot.removeItem(self.roi)

        # Put a new LineSegmentROI in the center of the plot in the right 
        # orientation
        lower_left, upper_right = self.calculate_endpoints()
        self.roi = pg.LineSegmentROI(positions=[lower_left, upper_right], 
                                     pen='m')
#        self.roi.setPos(lower_left)
        self.plot.addItem(self.roi, ignoreBounds=True)

        # Reconnect signal handling
        # Wrap the LineSegmentROI's sigRegionChanged
        self.sig_region_changed = self.roi.sigRegionChanged
#        self.plot.sig_axes_changed.connect(self.initialize)

        logger.info('Emitting sig_initialized.')
        self.sig_initialized.emit()

    def recenter(self) :
        """ Put the ROI in the center of the current plot. """
        logger.info('Recentering ROI.')
        lower_left, upper_right = self.calculate_endpoints()
#        self.roi.setPos(lower_left, update=False)
#        self.roi.setSize(upper_right)

    def calculate_endpoints(self) :
        """ Get sensible initial values for the endpoints of the :class: 
        LineSegmentROI from the :class: PlotWidget's current view range.  
        Depending on the state of `self.orientation` these endpoints 
        correspond either to a vertical or horizontal line centered at the 
        center of the plot and spanning exactly the whole plot range.

        Returns a tuple of len(2) lists: (lower_left, top_right) 
        corresponding to the  two endpoints.
        """
        # Get the current range of the plot
        [[xmin, xmax], [ymin, ymax]] = self.plot.get_limits()
        x = 0.5*(xmax+xmin)
        y = 0.5*(ymax+ymin)

        # Set the start and endpoint depending on the orientation
        if self.orientation is 'horizontal' :
            lower_left = [xmin, y]
            upper_right = [xmax, y]
        elif self.orientation is 'vertical' :
            lower_left = [x, ymin]
            upper_right = [x, ymax]

        logger.debug('lower_left: {}, upper_right: {}'.format(lower_left, 
                                                              upper_right))
        return lower_left, upper_right

    def flip_orientation(self) :
        """ Change the cutline's orientation from vertical to horitontal or 
        vice-versa and re-initialize it in the new orientation.
        """
        # Find out which orientation we're currently in and change 
        # accordingly 
        orientations = ['horizontal', 'vertical']
        # `i` will be the index of the orientation we currently don't have
        i = (orientations.index(self.orientation) + 1) % 2
        self.orientation = orientations[i]
        logger.info('New orientation: {}'.format(self.orientation))
        self.initialize()

    def get_array_region(self, *args, **kwargs) :
        """ Wrapper for :attr: `self.roi.getArrayRegion`. """
        return self.roi.getArrayRegion(*args, **kwargs)
 
class FooCursor(pg.ROI) :
    """ .. :deprecated:
    A straight cursor that can be dragged over a plot. Inherits from 
    :class: `ROI <pyqtgraph.ROI>` such that 
    functionalities like :func: `getArrayRegion 
    <pyqtgraph.ROI.getArrayRegion` are available.
    """

    def __init__(self, imagePlot, pos=None, orientation='vertical', 
                 **kwargs) :
        """ Connect to an ImagePlot instance and its :signal: sigImageChanged. """
        if pos is None:
            pos = [0,0]
        #ROI.__init__(self, pos, [1,1], **kwargs)
        super().__init__(pos=pos, size=[1,1], **kwargs)
        self.imagePlot = imagePlot

        # Get the first letter of the `orientation` argument in lowercase
        o = orientation[0].lower()
        if o == 'v' :
            self.orientation = 'vertical'
        elif o == 'h' :
            self.orientation = 'horizontal'
        else :
            msg = 'Orientation `{}` not understood.'.format(orientation)
            raise(ValueError(msg))

        self.updateEndpoints()

        #self.plotItem.sigRangeChanged.connect(self.updateEndpoints)
        self.imagePlot.sigImageChanged.connect(self.onParentImageChange)

    def translate(self, *args, **kwargs) :
        super().translate(*args, **kwargs)
        print(args, kwargs)
        print(self.pos())
        boundingRect = self.imagePlot.getViewBox().viewRect()
        print(boundingRect)
        print(self.endpoints)
        print()

    def onParentImageChange(self) :
        """ Update the bounds and endpoints of the cutline. """
        self.updateBounds()
        self.updateEndpoints()

    def updateBounds(self) :
        """ Set the region where this cutline can be moved around in to its 
        parents ViewBox. 
        """
        boundingRect = self.imagePlot.getViewBox().viewRect()
        self.maxBounds = boundingRect

    def updateEndpoints(self) :
        """ Set the endpoints to span either the full horizontal or vertical 
        region of the `parent` imagePlot.
        """
        [[xmin, xmax], [ymin, ymax]] = self.imagePlot.viewRange()
        if self.orientation == 'vertical' :
            point1 = (self.pos().x(), ymin)
            point2 = (self.pos().x(), ymax)
        elif self.orientation == 'horizontal' :
            point1 = (xmin, self.pos().y())
            point2 = (xmax, self.pos().y())
        self.setEndpoints(point1, point2)

    def setEndpoints(self, point1, point2) :
        """ Set endpoints as a list of :class: `Point <pyqtgraph.Point>` 
        objects, representing the start and endpoint of the line.
        """
        self.endpoints = [Point(point1), Point(point2)]

       
    def paint(self, p, *args) :
        """ Function required by one or another superclass. """
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)
        h1 = self.endpoints[0]
        h2 = self.endpoints[1]
        p.drawLine(h1, h2)
        #print(h1, h2)

    def boundingRect(self) :
        """ Function required by one or another superclass. """
        return self.shape().boundingRect()

    def shape(self) :
        """ Function required by one or another superclass. Define the shape 
        of this cutline. 
        """
        p = QtGui.QPainterPath()

        h1 = self.endpoints[0]
        h2 = self.endpoints[1]
        dh = h2-h1
        if dh.length() == 0:
            return p

        pxv = self.pixelVectors(dh)[1]
        if pxv is None:
            return p

        pxv *= 5

        p.moveTo(h1+pxv)
        p.lineTo(h2+pxv)
        p.lineTo(h2-pxv)
        p.lineTo(h1-pxv)
        p.lineTo(h1+pxv)

        return p

    def getArrayRegion(self, data, img, axes=(0,1), order=1, 
                       returnMappedCoords=False, **kwds):
        """
        Use the position of this ROI relative to an imageItem to pull a slice 
        from an array.

        Since this pulls 1D data from a 2D coordinate system, the return value 
        will have ndim = data.ndim-1

        See ROI.getArrayRegion() for a description of the arguments.
        """
        imgPts = [self.mapToItem(img, h) for h in self.endpoints]
        rgns = []
        coords = []
        d = Point(imgPts[1] - imgPts[0])
        o = Point(imgPts[0])
        rgn = affineSlice(data, shape=(int(d.length()),), 
                          vectors=[Point(d.norm())], origin=o, axes=axes, 
                          order=order, returnCoords=returnMappedCoords, **kwds)
        return rgn
                            
#class Cursor(pg.LineSegmentROI) :
#    """ A straight cursor that can be dragged over a plot. Inherits from 
#    :class: `LineSegmentROI <pyqtgraph.LineSegmentRoi>` such that 
#    functionalities like :func: `getArrayRegion 
#    <pyqtgraph.LineSegmentRoi.getArrayRegion` are available.
#    """
#
#    def __init__(self, *args, **kwargs) :
#        super().__init__(*args, **kwargs)
#
#        # Remove all handles
#        for handle in self.getHandles() :
#            self.removeHandle(handle)
#            break
#        print(self.getHandles())
#
#    def removeHandle(self, handle):
#        """Remove a handle from this ROI. Argument may be either a Handle 
#        instance or the integer index of the handle."""
#        index = self.indexOfHandle(handle)
#        
#        handle = self.handles[index]['item']
#        self.handles.pop(index)
#        handle.disconnectROI(self)
#        #if len(handle.rois) == 0:
#        #    self.scene().removeItem(handle)
#        self.stateChanged()
#
