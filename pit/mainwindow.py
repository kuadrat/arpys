"""
The `view` part of the Image Tool. Basically implements the layout of one PIT 
window.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

# Absolute imports are better than relative ones, apparently
from kustom.arpys import dataloaders as dl
#from kustom.arpys.pit.widgets.kimageview.KImageView import *
from kustom.arpys.pit.imageplot import *

appStyle="""
QMainWindow{
    background-color: black;
    }
"""

class MainWindow(QtGui.QMainWindow) :
    
    title = 'Python Image Tool'
    size = (800, 800)

    def __init__(self, filename=None, background='default') :
        super().__init__()
        self.setStyleSheet(appStyle)

        self.initUI()
        if filename is not None :
            self.prepare_data(filename)

    def setImage(self, image, *args, **kwargs) :
        """ Wraps the underlying ImagePlot3d's setImage method.
        See :func: `<arpys.pit.imageplot.ImagePlot3d.setImage>`.
        """
        self.main_plot.setImage(image, *args, **kwargs)

    def prepare_data(self, filename) :
        """ Load the specified data and prepare some parts of it (caching).
        @TODO Maybe add a 'loading...' notification.
        """
        self.D = dl.load_data(filename)
        ## Transpose since pg.ImageView expects column-major image data
        self.data = np.moveaxis(self.D.data, 0, -1)
        self.data = self.D.data

        self.setImage(self.data, axes=(1,2))

    def initUI(self) :
        # Set the window title
        self.setWindowTitle(self.title)
        self.resize(*self.size)

        # Create a "central widget" and its layout
        self.central_widget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Create the 3D (main) and cut ImagePlots 
        self.main_plot = ImagePlot3d()
        #self.main_imv.getHistogramWidget().hide()

        # Creat the scalebar for the z dimension and connect to main_plot's z
        self.zscale = Scalebar()
        self.zscale.registerTracedVar(self.main_plot.z)

        # Add ROIs to the main ImageView
        #shape = self.data.shape
        #x = shape[1]
        #y = shape[2]
        #x_2, y_2 = (int(0.5*i) for i in (x, y))
        #x_endpoints = [[0, y_2], [x, y_2]]
        #y_endpoints = [[x_2, 0], [x_2, y]]
        #self.xroi = pg.LineSegmentROI(x_endpoints, pen='r')
        #self.yroi = pg.LineSegmentROI(y_endpoints, pen='b')
        #self.main_plot.addItem(self.xroi)
        #self.main_plot.addItem(self.yroi)

        # Connect signal handling
        #self.xroi.sigRegionChanged.connect(self.update)
        #self.yroi.sigRegionChanged.connect(self.update)

        # Put the data in the main IMV and take the first cuts
        #self.main_plot.setImage(self.data, autoRange=True)
        self.update()

        # Align all the gui elements
        self.align()
        self.show()

    def align(self) :
        """ Align all the GUI elements in the QLayout. """
        # Get a short handle
        l = self.layout

        # Main (3D) ImageView in bottom left
        l.addWidget(self.main_plot, 1, 0)

        # Scalebar
        l.addWidget(self.zscale, 2, 0)

        # Xcut and Ycut above, to the right of Main
#        l.addWidget(self.xcut_imv, 0, 0)
#        l.addWidget(self.ycut_imv, 1, 1)

    def update(self) :
        """ Take cuts of the data along the ROIs. """
        axes = (1,2)
        #xcut = self.xroi.getArrayRegion(self.data, self.main_plot.imageItem, 
        #                                axes=axes)
        # Convert np.array `xcut` to an ImageItem set it as `xcut_plot`'s Image
        #xcut_image = pg.ImageItem(xcut)
        #self.xcut_plot.setImage(xcut)

        # Same for y
        #ycut = self.yroi.getArrayRegion(self.data, self.main_plot.imageItem, 
        #                                axes=axes)
        #ycut_image = pg.ImageItem(ycut)
        #self.ycut_plot.setImage(ycut)

if __name__ == '__main__' :
    app = QtGui.QApplication([])
    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00428.fits'
    main_window = MainWindow()
    main_window.prepare_data(filename)
    app.exec_()

