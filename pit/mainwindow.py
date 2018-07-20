"""
The `view` part of the Image Tool. Basically implements the layout of one PIT 
window.
"""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtGui, QtCore
from qtconsole.rich_ipython_widget import RichIPythonWidget, RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

# Absolute imports are better than relative ones, apparently
import arpys as arp
from arpys import dl, pp
#from kustom.arpys.pit.widgets.kimageview.KImageView import *
from arpys.pit.cmaps import cmaps
from arpys.pit.cursor import Cursor
from arpys.pit.imageplot import *
from arpys.pit.utilities import TracedVariable

app_style="""
QMainWindow{
    background-color: black;
    }
"""

console_style = """
color: rgb(255, 255, 255);
background-color: rgb(0, 0, 0);
border: 1px solid rgb(50, 50, 50);
"""

DEFAULT_CMAP = 'kocean'

class EmbedIPython(RichJupyterWidget):
    """ Some voodoo to get an ipython console in a Qt application. """
    def __init__(self, **kwarg):
        super(RichJupyterWidget, self).__init__()
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push(kwarg)
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

class MainWindow(QtGui.QMainWindow) :
    
    title = 'Python Image Tool'
    # width, height
    size = (1200, 800)
    data = None

    def __init__(self, filename=None, background='default') :
        super().__init__()
        # Aesthetics
        self.setStyleSheet(app_style)
        self.set_cmap(DEFAULT_CMAP)

        self.initUI()
        
        # Connect signal handling
        self.main_plot.sig_image_changed.connect(self.on_image_change)

        if filename is not None :
            self.prepare_data(filename)

    def get_data(self) :
        """ Convenience `getter` method. Allows writing `self.get_data()` 
        instead of ``self.data.get_value()``. 
        """
        return self.data.get_value()

    def set_data(self, data) :
        """ Convenience `setter` method. Allows writing `self.set_data(d)` 
        instead of ``self.data.set_value(d)``. 
        """
        self.data.set_value(data)

    def set_image(self, image=None, *args, **kwargs) :
        """ Wraps the underlying ImagePlot3d's set_image method.
        See :func: `<arpys.pit.imageplot.ImagePlot3d.set_image>`.
        """
        if image is None :
            image = self.main_plot.image_data
        self.main_plot.set_image(image, *args, lut=self.lut, **kwargs)

    def prepareData(self, filename) :
        """ Load the specified data and prepare some parts of it (caching).
        @TODO Maybe add a 'loading...' notification.
        """
        self.D = dl.load_data(filename)
        self.data = TracedVariable(self.D.data)
        def on_data_change() :
            """ Callback for change of self.data. """
            self.set_image(self.get_data(), axes=(1,2))
            self.update()
        self.data.sig_value_changed.connect(on_data_change)

        self.set_image(self.get_data(), axes=(1,2))

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
        # The python console
        namespace = dict(pit=self, pg=pg, arp=arp, dl=dl, pp=pp)
#        self.console = pyqtgraph.console.ConsoleWidget(namespace=namespace)
        self.console = EmbedIPython(**namespace)
        self.console.kernel.shell.run_cell('%pylab qt')
        self.console.setStyleSheet(console_style)

        # Creat the scalebar for the z dimension and connect to main_plot's z
        self.zscale = Scalebar()
        self.zscale.register_traced_variable(self.main_plot.z)

        # Add ROI to the main ImageView
        self.roi = Cursor(self.main_plot)
        self.on_image_change()

        # Align all the gui elements
        self.align()
        self.show()

        #self.defineKeys()

    def defineKeys() :
        pass

    def on_image_change(self) :
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
        # Xcut and Ycut above, to the right of Main
        l.addWidget(self.cut_plot, 0, 1)
        # Scalebar
        l.addWidget(self.zscale, 1, 0)
        # Console
        l.addWidget(self.console, 1, 1)

    def update(self) :
        """ Take cuts of the data along the ROI. """
        axes = (1,2)
        try :
            cut = self.roi.getArrayRegion(self.get_data(), 
                                          self.main_plot.image, 
                                          axes=axes)
        except Exception as e :
            print(e)
            return

        # Convert np.array `xcut` to an ImageItem and set it as `xcut_plot`'s 
        # Image
        cut_image = pg.ImageItem(cut)
        self.cut_plot.set_image(cut, lut=self.lut)

    def set_cmap(self, cmap) :
        """ Set the colormap to *cmap* where *cmap* is one of the names 
        registered in `<arpys.pit.cmaps>` which includes all matplotlib and 
        kustom cmaps.
        """
        try :
            self.cmap = cmaps[cmap]
        except KeyError :
            print('Invalid colormap name. Use one of: ')
            print(cmaps.keys())
        self.lut = self.cmap.getLookupTable()
        self.redraw_plots()

    def set_alpha(self, alpha) :
        """ Set the alpha value of the currently used cmap. *alpha* can be a 
        single float or an array of length ``len(self.cmap.color)``.
        """
        self.cmap.set_alpha(alpha)
        self.lut = self.cmap.getLookupTable()
        self.redraw_plots()

    def redraw_plots(self) :
        """ Redraw plotted data to reflect changes in data or its colors. """
        try :
            # Redraw main plot
            self.set_image()
            # Redraw cut plot
            self.update()
        except AttributeError :
            # In some cases (namely initialization) the mainwindow is not 
            # defined yet
            pass

    def keyPressEvent(self, event) :
        """ Define all responses to keyboard presses. """
        pass
        #key = event.key()
        #print(key, type(key))
        #if key == QtCore.Qt.Key_R :
        #    print('is R')
        #    self.roi.flipOrientation()
        #else :
        #    print('not R')
        #    event.ignore()
        #    return
        #event.accept()

if __name__ == '__main__' :
    app = QtGui.QApplication([])
#    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00428.fits'
    #filename = '/home/kevin/Documents/qmap/materials/Bi2201/2018_06_SIS/20180609_0007.h5'
    #filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00398.fits'
    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00399.fits'
#    filename = '/home/kevin/qmap/experiments/2018_07_CASSIOPEE/CaMnSb/S3_FSM_fine_hv75_T65'

    main_window = MainWindow()
    main_window.prepareData(filename)
    app.exec_()

