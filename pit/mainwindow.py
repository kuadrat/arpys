"""
The `view` part of the Image Tool. Basically implements the layout of one PIT 
window.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

# Absolute imports are better than relative ones, apparently
from kustom.arpys import dataloaders as dl
#from kustom.arpys.pit.widgets.kimageview.KImageView import *
from kustom.arpys.pit.imageplot import *
from kustom.arpys.pit.cursor import Cursor

appStyle="""
QMainWindow{
    background-color: black;
    }
"""

class MainWindow(QtGui.QMainWindow) :
    
    title = 'Python Image Tool'
    # width, height
    size = (1200, 800)
    data = None

    def __init__(self, filename=None, background='default') :
        super().__init__()
        self.setStyleSheet(appStyle)

        self.initUI()
        
        # Connect signal handling
        self.main_plot.sigImageChanged.connect(self.onImageChange)

        if filename is not None :
            self.prepare_data(filename)

    def setImage(self, image, *args, **kwargs) :
        """ Wraps the underlying ImagePlot3d's setImage method.
        See :func: `<arpys.pit.imageplot.ImagePlot3d.setImage>`.
        """
        self.main_plot.setImage(image, *args, **kwargs)

    def prepareData(self, filename) :
        """ Load the specified data and prepare some parts of it (caching).
        @TODO Maybe add a 'loading...' notification.
        """
        self.D = dl.load_data(filename)
        ## Transpose since pg.ImageView expects column-major image data
        #self.data = np.moveaxis(self.D.data, 0, -1)
        self.data = self.D.data

        self.setImage(self.data, axes=(1,2))

        # Put the data in the main IMV and take the first cuts
        self.update()

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
        self.cut_plot = ImagePlot()

        # Creat the scalebar for the z dimension and connect to main_plot's z
        self.zscale = Scalebar()
        self.zscale.registerTracedVar(self.main_plot.z)

        # Add ROI to the main ImageView
        self.roi = Cursor(self.main_plot)
        self.onImageChange()

        # Align all the gui elements
        self.align()
        self.show()

        #self.defineKeys()

    def defineKeys() :
        pass

    def onImageChange(self) :
        """ Recenter the ROI. """
        self.roi.initialize()

        # Reconnect signal handling
        self.roi.roi.sigRegionChanged.connect(self.update)

    def align(self) :
        """ Align all the GUI elements in the QLayout. """
        # Get a short handle
        l = self.layout

        # Main (3D) ImageView in bottom left
        l.addWidget(self.main_plot, 0, 0)

        # Scalebar
        l.addWidget(self.zscale, 1, 0)

        # Xcut and Ycut above, to the right of Main
        l.addWidget(self.cut_plot, 0, 1)

    def update(self) :
        """ Take cuts of the data along the ROIs. """
        axes = (1,2)
        try :
            cut = self.roi.getArrayRegion(self.data, self.main_plot.getImage(), 
                                        axes=axes)
        except Exception as e :
            print(e)
            return

        # Convert np.array `xcut` to an ImageItem set it as `xcut_plot`'s Image
        cut_image = pg.ImageItem(cut)
        self.cut_plot.setImage(cut)

    def keyPressEvent(self, event) :
        """ Define all responses to keyboard presses. """
        key = event.key()
        print(key, type(key))
        if key == QtCore.Qt.Key_R :
            print('is R')
            self.roi.flipOrientation()
        else :
            print('not R')
            event.ignore()
            return
        event.accept()

if __name__ == '__main__' :
    app = QtGui.QApplication([])
    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00428.fits'
    #filename = '/home/kevin/Documents/qmap/materials/Bi2201/2018_06_SIS/20180609_0007.h5'
    #filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00398.fits'
    #filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00399.fits'
    main_window = MainWindow()
    main_window.prepareData(filename)
    app.exec_()

