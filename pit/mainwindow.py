"""
The `controller` part of the Image Tool - though the separation is not very 
strict. Keeps track of the data and the applied postprocessing and 
visualization options but also sets up the geometry and appearance of GUI 
elements.
"""

import logging

import numpy as np
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtGui, QtCore
from qtconsole.rich_ipython_widget import RichIPythonWidget, RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

import arpys as arp
from arpys import dl, pp
from arpys.pit.cmaps import cmaps
from arpys.pit.cutline import Cutline
from arpys.pit.imageplot import *
from arpys.pit.utilities import TracedVariable

# +----------------+ #
# | Set up logging | # =========================================================
# +----------------+ #

logger = logging.getLogger('pit')
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s][%(name)s]%(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False

# +------------------------+ #
# | Appearance definitions | # =================================================
# +------------------------+ #

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

# +-----------------------+ #
# | Main class definition | # ==================================================
# +-----------------------+ #

class MainWindow(QtGui.QMainWindow) :
    """ (Currently) The main window of PIT. Defines the basic GUI layouts and 
    acts as the controller, keeping track of the data and handling the 
    communication between the different GUI elements. 
    """
    
    title = 'Python Image Tool'
    # width, height
    size = (1200, 800)
    data = None
    axes = (1,2)
    z = TracedVariable()
    Z_AXIS_INDEX = 0

    def __init__(self, filename=None, background='default') :
        super().__init__()
        # Aesthetics
        self.setStyleSheet(app_style)
        self.set_cmap(DEFAULT_CMAP)

        self.init_UI()
        
        # Connect signal handling
        self.cutline.sig_initialized.connect(self.on_cutline_initialized)

        if filename is not None :
            self.prepare_data(filename)

    def init_UI(self) :
        """ Initialize the elements of the user interface. """
        # Set the window title
        self.setWindowTitle(self.title)
        self.resize(*self.size)

        # Create a "central widget" and its layout
        self.central_widget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Create the 3D (main) and cut ImagePlots 
        self.main_plot = ImagePlot()
        self.cut_plot = ImagePlot()

        # Set up the python console
        namespace = dict(pit=self, pg=pg, arp=arp, dl=dl, pp=pp)
#        self.console = pyqtgraph.console.ConsoleWidget(namespace=namespace)
        self.console = EmbedIPython(**namespace)
        self.console.kernel.shell.run_cell('%pylab qt')
        self.console.setStyleSheet(console_style)

        # Create the scalebar for the z dimension and connect to main_plot's z
        self.zscale = Scalebar()
        self.zscale.register_traced_variable(self.z)

        # Add ROI to the main ImageView
        self.cutline = Cutline(self.main_plot)
        self.cutline.initialize()

        # Align all the gui elements
        self.align()
        self.show()

        #self.define_keys()

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

    def define_keys() :
        pass

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

    def prepare_data(self, filename) :
        """ Load the specified data and prepare the corresponding z range. 
        Then display the newly loaded data.
        """
        logger.debug('prepare_data()')

        self.D = dl.load_data(filename)
        self.data = TracedVariable(self.D.data)

        self.update_z_range()
        self.prepare_scales()
        
        # Connect signal handling so changes in data are immediately reflected
        self.z.sig_value_changed.connect(self.on_z_change)
        self.data.sig_value_changed.connect(self.redraw_plots)

        self.update_main_plot()
        self.set_scales()

    def prepare_scales(self) :
        """ Create a list containing the three original x-, y- and z-scales. """
        # Define the scales in the initial view. The data is arranged as 
        # (z,y,x) and we initially display (y,x)
        xscale = self.D.yscale
        yscale = self.D.xscale
        zscale = self.D.zscale
        self.scales = np.array([zscale, yscale, xscale])
        # Avoid undefined axes scales and replace them with len(1) sequences
        for i,scale in enumerate(self.scales) :
            if scale is None :
                self.scales[i] = np.array([0])

    def set_scales(self) :
        """ Set the x- and y-scales of the plots. The :class: `ImagePlot 
        <arpys.pit.imageplot.ImagePlor>` object takes care of keeping the 
        scales as they are, once they are set.
        """
        xscale = self.scales[2]
        yscale = self.scales[1]
        logger.debug(('set_scales(): len(xscale), len(yscale)={}, ' +
                      '{}').format(len(xscale), len(yscale)))
        self.main_plot.set_xscale(xscale)
        self.main_plot.set_yscale(yscale, update=True)
        self.main_plot.fix_viewrange()
        self.cutline.initialize()

    def update_z_range(self) :
        """ When new data is loaded or the axes are rolled, the limits and 
        allowed values along the z dimension change.
        """
        # Determine the new ranges for z
        self.zmin = 0
        self.zmax = self.get_data().shape[self.Z_AXIS_INDEX] - 1

        self.z.set_allowed_values(range(self.zmin, self.zmax+1))
        self.z.set_value(self.zmin)

    def on_z_change(self, caller=None) :
        """ Callback to the :signal: `sig_z_changed`. Ensure self.z does not go 
        out of bounds and update the Image slice with a call to :func: 
        `update_main_plot <arpys.pit.imageplot.ImagePlot.update_main_plot>`.
        """
        # Ensure z doesn't go out of bounds
        z = self.z.get_value()
        clipped_z = clip(z, self.zmin, self.zmax)
        if z != clipped_z :
            # NOTE this leads to unnecessary signal emitting. Should avoid 
            # emitting the signal from inside a slot (slot: function 
            # connected to that signal)
            self.z.set_value(clipped_z)
        self.update_main_plot()

    def set_image_data(self) :
        """ Get the right (possibly integrated) slice out of *self.data*, 
        apply postprocessings and store it in *self.image_data*. 
        """
        z = self.z.get_value()
        self.image_data = self.get_data()[z,:,:]

    def set_image(self, image=None, *args, **kwargs) :
        """ Wraps the underlying ImagePlot3d's set_image method.
        See :func: `<arpys.pit.imageplot.ImagePlot3d.set_image>`. *image* can 
        be *None* i.e. in order to just update the plot with a new colormap.
        """
        if image is None :
            image = self.image_data
        self.main_plot.set_image(image, *args, lut=self.lut, **kwargs)

    def update_main_plot(self, **image_kwargs) :
        """ Change the *self.main_plot*`s currently displayed
        `image_item <arpys.pit.imageplot.ImagePlot.image_item>` to the slice 
        of *self.data* corresponding to the current value of *self.z*.
        """
        logger.debug(('update_main_plot(): ' + 
                      'Z_AXIS_INDEX={}').format(self.Z_AXIS_INDEX))

        self.set_image_data()

        logger.debug('self.image_data.shape={}'.format(self.image_data.shape))

        if image_kwargs != {} :
            self.image_kwargs = image_kwargs

        # Add image to main_plot
        self.set_image(self.image_data, **image_kwargs)

    def update_cut(self) :
        """ Take a cut of *self.data* along *self.cutline*. This is used to 
        update only the cut plot without affecting the main plot.
        """
        logger.debug('update_cut')
        try :
            cut = self.cutline.get_array_region(self.get_data(), 
                                            self.main_plot.image_item, 
                                            axes=self.axes)
        except Exception as e :
            logger.error(e)
            return

        # Convert np.array *cut* to an ImageItem and set it as *xcut_plot*'s 
        # Image
        cut_image = pg.ImageItem(cut)
        self.cut_plot.set_image(cut, lut=self.lut)

    def redraw_plots(self, image=None) :
        """ Redraw plotted data to reflect changes in data or its colors. """
        try :
            # Redraw main plot
            self.set_image(image, axes=self.axes)
            # Redraw cut plot
            self.update_cut()
        except AttributeError :
            # In some cases (namely initialization) the mainwindow is not 
            # defined yet
            pass

    def on_cutline_initialized(self) :
        """ Need to reconnect the signal to the cut_plot. And directly update 
        the cut_plot.
        """
        self.cutline.sig_region_changed.connect(self.update_cut)
        self.update_cut()

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

    def roll_axes(self) :
        """ """
        data = self.get_data()
        self.set_data(np.moveaxis(data, [0,1,2], [2,0,1]))
        self.scales = np.roll(self.scales, 1)
        self.update_z_range()
#        self.redraw_plots()
        self.set_scales()

    def keyPressEvent(self, event) :
        """ Define all responses to keyboard presses. """
        key = event.key()
        logger.debug('keyPressEvent(): key={}'.format(key))
        if key == QtCore.Qt.Key_Right :
            self.z.set_value(self.z.get_value() + 1)
        elif key == QtCore.Qt.Key_Left :
            self.z.set_value(self.z.get_value() - 1)
        #if key == QtCore.Qt.Key_R :
        #    print('is R')
        #    self.cutline.flip_orientation()
        #else :
        #    print('not R')
        #    event.ignore()
        #    return
        #event.accept()

if __name__ == '__main__' :
    app = QtGui.QApplication([])
#    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00428.fits'
#    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2018_06_SIS/20180609_0007.h5'
    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00398.fits'
#    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2017_12_ALS/20171215_00399.fits'
#    filename = '/home/kevin/qmap/experiments/2018_07_CASSIOPEE/CaMnSb/S3_FSM_fine_hv75_T65'
#    filename = '/home/kevin/qmap/materials/Bi2201/2018_06_SIS/0025.h5'

    logger.info(filename)
    main_window = MainWindow()
    main_window.prepare_data(filename)
    app.exec_()

