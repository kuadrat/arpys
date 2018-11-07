"""
The `controller` part of the Image Tool - though the separation is not very 
strict. Keeps track of the data and the applied postprocessing and 
visualization options but also sets up the geometry and appearance of GUI 
elements.
"""

import logging
from copy import copy

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
from arpys.pit.utilities import TracedVariable, indexof

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

class PITDataHandler() :
    """ Object that keeps track of a set of ARPES data and allows 
    manipulations on it. In a Model-View-Controller framework this could be 
    seen as the Model, while :class: `MainWindow 
    <arpys.pit.mainwidow.MainWindow>` would be the View part.
    """
    # np.array that contains the 3D data
    data = None
    scales = np.array([[0, 1], [0, 1], [0, 1]])
    # Indices of *data* that are displayed in the main plot 
    axes = (1,2)
    # How often we have rolled the axes from their initial positions
    n_rolls = 0 #unused
    # Index along the z axis at which to produce a slice
    z = TracedVariable(name='z')

    def __init__(self, main_window) :
        self.main_window = main_window

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
        self.data = TracedVariable(self.D.data, name='data')

        # Retain a copy of the original datadict so that we can reset later
        self.original_D = copy(self.D)

        self.prepare_scales()
        self.on_z_dim_change()
        
        # Connect signal handling so changes in data are immediately reflected
        self.z.sig_value_changed.connect(self.on_z_change)
        self.data.sig_value_changed.connect(self.on_data_change)

        self.main_window.update_main_plot()
        self.main_window.set_scales()

    def update_z_range(self) :
        """ When new data is loaded or the axes are rolled, the limits and 
        allowed values along the z dimension change.
        """
        # Determine the new ranges for z
        self.zmin = 0
        self.zmax = self.get_data().shape[0] - 1

        self.z.set_allowed_values(range(self.zmin, self.zmax+1))
        self.z.set_value(self.zmin)

    def reset_data(self) :
        """ Put all data and metadata into its original state, as if it was 
        just loaded from file.
        """
        logger.debug('reset_data()')
        self.D = copy(self.original_D)
        self.set_data(self.D.data)
        self.prepare_scales()
        self.main_window.set_scales()
        # Redraw the integrated intensity plot
        self.on_z_dim_change()

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

    def on_data_change(self) :
        """ Update self.main_window.image_data and replot. """
        logger.debug('on_data_change()')
        self.main_window.update_image_data()
        self.main_window.redraw_plots()
        # Also need to recalculate the intensity plot
        self.on_z_dim_change()

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
        self.main_window.update_main_plot()

    def update_edc_mdc(self) :
        """ Update the EDC and MDC plots. """
        logger.debug('on_crosshair_move()')

        # Get shorthands for plots
        ep = self.main_window.edc_plot
        mp = self.main_window.mdc_plot
        for plot in [ep, mp] :
            # Remove the old integrated intensity curve
            try :
                old = plot.listDataItems()[0]
                plot.removeItem(old)
            except IndexError :
                pass

        # EDC
        i_edc = int( min(mp.pos.get_value(), mp.pos.allowed_values.max()-1))
#        i_edc = indexof(mp.pos.get_value(), self.scales[1])
        logger.debug('mp.pos.get_value()={}; i_edc: {}'.format(mp.pos.get_value(), i_edc))
        edc = self.cut_data[i_edc]
        y = np.arange(len(edc))
        ep.plot(edc, y)

        # MDC
        i_mdc = int( min(ep.pos.get_value(), ep.pos.allowed_values.max()-1)) 
#        i_mdc = indexof(ep.pos.get_value(), self.scales[0])
        logger.debug('ep.pos.get_value()={}; i_mdc: {}'.format(ep.pos.get_value(), i_mdc))
        mdc = self.cut_data[:,i_mdc]
        x = np.arange(len(mdc))
        mp.plot(x, mdc)

        # This update is only necessary for the EDC plot as its range is 
        # variable. NOTE Leads to infinite recursion
#        ep.pos.set_value(i_mdc)

    def on_z_dim_change(self) :
        """ Called when either completely new data is loaded or the dimension 
        from which we look at the data changed (e.g. through :func: `roll_axes 
        <arpys.pit.mainwindow.PITDataHandler.roll_axes>`).
        Update the z range and the integrated intensity plot.
        """
        logger.debug('on_z_dim_change()')
        self.update_z_range()

        # Get a shorthand for the integrated intensity plot
        ip = self.main_window.integrated_plot
        # Remove the old integrated intensity curve
        try :
            old = ip.listDataItems()[0]
            ip.removeItem(old)
        except IndexError :
            pass

        # Calculate the integrated intensity and plot it
        self.calculate_integrated_intensity()
        ip.plot(self.integrated)

        # Also display the actual data values in the top axis
        zscale = self.scales[0]
        zmin = zscale[0]
        zmax = zscale[-1]
        ip.set_secondary_axis(zmin, zmax)

    def calculate_integrated_intensity(self) :
        self.integrated = self.get_data().sum(1).sum(1)

    def roll_axes(self, i=1) :
        """ Change the way we look at the data cube. While initially we see 
        an Y vs. X slice in the main plot, roll it to Z vs. Y. A second call 
        would roll it to X vs. Z and, finally, a third call brings us back to 
        the original situation.
        ..: *i* : int; either 1 or 2. Number of dimensions to roll.
        """
        logger.debug('roll_axes()')
        data = self.get_data()
        if i==1 :
            res = [2,0,1]
        elif i==2 :
            res = [1,2,0]
        self.n_rolls = (self.n_rolls + 1) % 3
        self.set_data(np.moveaxis(data, [0,1,2], res))
        # Setting the data triggers a call to self.redraw_plots()
        self.scales = np.roll(self.scales, i)
        self.on_z_dim_change()
        self.main_window.set_scales()

class MainWindow(QtGui.QMainWindow) :
    """ (Currently) The main window of PIT. Defines the basic GUI layouts and 
    acts as the controller, keeping track of the data and handling the 
    communication between the different GUI elements. 
    """
    
    title = 'Python Image Tool'
    # width, height in pixels
    size = (1200, 800)

    # Plot transparency alpha
    alpha = 0.5
    # Plot powerlaw normalization exponent gamma
    gamma = 1

    def __init__(self, filename=None, background='default') :
        super().__init__()
        self.data_handler = PITDataHandler(self)

        # Aesthetics
        self.setStyleSheet(app_style)
        self.set_cmap(DEFAULT_CMAP)

        self.init_UI()
        
        # Connect signal handling
        self.cutline.sig_initialized.connect(self.on_cutline_initialized)

        if filename is not None :
            self.data_handler.prepare_data(filename)

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
        self.main_plot = ImagePlot(name='main_plot')
        self.cut_plot = CrosshairImagePlot(name='cut_plot')

        # Create the EDC and MDC plots (though, technically, they only 
        # display MDCs or EDCs under certain circumstances, and just 
        # intensities along a line in the general case)
        self.edc_plot = CursorPlot(name='edc_plot', orientation='horizontal')
        self.mdc_plot = CursorPlot(name='mdc_plot')
        self.edc_plot.register_traced_variable(self.cut_plot.pos[0])
        self.mdc_plot.register_traced_variable(self.cut_plot.pos[1])
        for traced_variable in self.cut_plot.pos :
            traced_variable.sig_value_changed.connect(
                self.data_handler.update_edc_mdc)

        self.cut_plot.sig_image_changed.connect(self.data_handler.update_edc_mdc)

        # Set up the python console
        namespace = dict(pit=self.data_handler, mw=self, pg=pg, arp=arp, 
                         dl=dl, pp=pp)
#        self.console = pyqtgraph.console.ConsoleWidget(namespace=namespace)
        self.console = EmbedIPython(**namespace)
        self.console.kernel.shell.run_cell('%pylab qt')
        self.console.setStyleSheet(console_style)

        # Create the integrated intensity plot
        self.integrated_plot = CursorPlot(name='z selector')
        self.integrated_plot.register_traced_variable(self.data_handler.z)

        # Add ROI to the main ImageView
        self.cutline = Cutline(self.main_plot)
        self.cutline.initialize()

        # Scalebars. Scalebar1 is for the `gamma` value
        scalebar1 = Scalebar()
        self.gamma_values = np.concatenate((np.linspace(0.1, 1, 50), 
                                            np.linspace(1.1, 10, 50)))
        scalebar1.pos.set_value(0.5)
        scalebar1.pos.sig_value_changed.connect(self.on_gamma_slider_move)
        # Label the scalebar
        gamma_label = pg.TextItem('γ', anchor=(0.5, 0.5))
        gamma_label.setPos(0.5, 0.5)
        scalebar1.addItem(gamma_label)
        self.scalebar1 = scalebar1

        # Align all the gui elements
        self.align()
        self.show()

        #self.define_keys()

    def align(self) :
        """ Align all the GUI elements in the QLayout. 
        
          0   1   2   3   4
        +---+---+---+---+---+
        |       |       | e | 0
        + main  |  cut  | d +
        |       |       | c | 1
        +-------+-------+---+
        |       |  mdc  |   | 2
        +   z   +-------+---+
        |       |  console  | 4
        +---+---+---+---+---+
        
        (Units of subdivision [sd])
        """
        # subdivision 
        sd = 3
        # Get a short handle
        l = self.layout
        # addWIdget(row, column, rowSpan, columnSpan)
        # Main (3D) ImageView in top left
        l.addWidget(self.main_plot, 0, 0, 2*sd, 2*sd)
        # Cut to the right of Main
        l.addWidget(self.cut_plot, 0, 2*sd, 2*sd, 2*sd)
        # EDC and MDC plots
#        l.addWidget(self.edc_plot, 0, 4*sd, 2*sd, 1*sd)
        l.addWidget(self.edc_plot, 0, 4*sd, 2*sd, 2)
        l.addWidget(self.mdc_plot, 2*sd, 2*sd, 1*sd, 2*sd)
        # Integrated z-intensity plot
        l.addWidget(self.integrated_plot, 2*sd, 0, 2*sd, 2*sd)
        # Console
        l.addWidget(self.console, 3*sd, 2*sd, 1*sd, 3*sd)

        # Scalebars
        l.addWidget(self.scalebar1, 2*sd, 4*sd, 1, 1*sd)

        nrows = 4*sd
        ncols = 5*sd
        # Need to manually set all row- and columnspans as well as min-sizes
        for i in range(nrows) :
            l.setRowMinimumHeight(i, 50)
            l.setRowStretch(i, 1)
        for i in range(ncols) :
            l.setColumnMinimumWidth(i, 50)
            l.setColumnStretch(i, 1)

    def define_keys() :
        pass

    def set_scales(self) :
        """ Set the x- and y-scales of the plots. The :class: `ImagePlot 
        <arpys.pit.imageplot.ImagePlot>` object takes care of keeping the 
        scales as they are, once they are set.
        """
        xscale = self.data_handler.scales[2]
        yscale = self.data_handler.scales[1]
        zscale = self.data_handler.scales[0]
        logger.debug(('set_scales(): len(xscale), len(yscale)={}, ' +
                      '{}').format(len(xscale), len(yscale)))
        self.main_plot.set_xscale(xscale)
        self.main_plot.set_yscale(yscale, update=True)
        self.main_plot.fix_viewrange()

        # Kind of a hack to get the crosshair to the right position...
        self.cut_plot.sig_axes_changed.emit()
#        self.cut_plot.set_xscale(zscale, update=True)
        # yscale depends on our cutline
        #self.cut_plot.set_yscale(yscale, update=True)
#        self.cut_plot.fix_viewrange()
        self.cutline.initialize()

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
        of *self.data_handler.data* corresponding to the current value of 
        *self.z*.
        """
        logger.debug('update_main_plot()')

        self.update_image_data()

        logger.debug('self.image_data.shape={}'.format(self.image_data.shape))

        if image_kwargs != {} :
            self.image_kwargs = image_kwargs

        # Add image to main_plot
        self.set_image(self.image_data, **image_kwargs)

    def update_cut(self) :
        """ Take a cut of *self.data_handler.data* along *self.cutline*. This 
        is used to update only the cut plot without affecting the main plot.
        """
        logger.debug('update_cut()')
        try :
            cut = self.cutline.get_array_region(self.data_handler.get_data(), 
                                            self.main_plot.image_item, 
                                            axes=self.data_handler.axes)
        except Exception as e :
            logger.error(e)
            return

        self.data_handler.cut_data = cut
        # Convert np.array *cut* to an ImageItem and set it as *cut_plot*'s 
        # Image
        cut_image = pg.ImageItem(cut)
        self.cut_plot.set_image(cut, lut=self.lut)

    def update_image_data(self) :
        """ Get the right (possibly integrated) slice out of *self.data*, 
        apply postprocessings and store it in *self.image_data*. 
        Skip this if the z value should be out of range, which can happen if 
        the image data changes and the z scale hasn't been updated yet.
        """
        logger.debug('update_image_data()')
        z = self.data_handler.z.get_value()
        try :
            self.image_data = self.data_handler.get_data()[z,:,:]
        except IndexError :
            logger.debug(('update_image_data(): z index {} out of range for '
                          'data of length {}.').format(
                             z, self.image_data.shape[0]))

    def redraw_plots(self, image=None) :
        """ Redraw plotted data to reflect changes in data or its colors. """
        logger.debug('redraw_plots()')
        try :
            # Redraw main plot
            self.set_image(image, axes=self.data_handler.axes)
            # Redraw cut plot
            self.update_cut()
        except AttributeError as e :
            # In some cases (namely initialization) the mainwindow is not 
            # defined yet
            logger.debug('AttributeError: {}'.format(e))

    def on_cutline_initialized(self) :
        """ Need to reconnect the signal to the cut_plot. And directly update 
        the cut_plot.
        """
        self.cutline.sig_region_changed.connect(self.update_cut)
        self.update_cut()

    def on_gamma_slider_move(self) :
        """ When the user moves the gamma slider, update gamma. """
        ind = min(int(100*self.scalebar1.pos.get_value()), len(self.gamma_values)-1)
        gamma = self.gamma_values[ind]
        self.set_gamma(gamma)

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
        # Since the cmap changed it forgot our settings for alpha and gamma
        self.cmap.set_alpha(self.alpha)
        self.cmap.set_gamma(self.gamma)
        self.cmap_changed()

    def set_alpha(self, alpha) :
        """ Set the alpha value of the currently used cmap. *alpha* can be a 
        single float or an array of length ``len(self.cmap.color)``.
        """
        self.alpha = alpha
        self.cmap.set_alpha(alpha)
        self.cmap_changed()

    def set_gamma(self, gamma=1) :
        """ Set the exponent for the power-law norm that maps the colors to 
        values. I.e. the values where the colours are defined are mapped like 
        ``y=x**gamma``.
        """
        self.gamma = gamma
        self.cmap.set_gamma(gamma)
        self.cmap_changed()
        # Additionally, we need to update the slider position. We need to 
        # hack a bit to avoid infinite signal loops: avoid emitting of the 
        # signal and update the slider position by hand with a call to 
        # scalebar1.set_position().
        self.scalebar1.pos._value = indexof(gamma, self.gamma_values)/100
        self.scalebar1.set_position()
    
    def cmap_changed(self) :
        """ Recalculate the lookup table and redraw the plots such that the 
        changes are immediately reflected.
        """
        self.lut = self.cmap.getLookupTable()
        self.redraw_plots()

    def keyPressEvent(self, event) :
        """ Define all responses to keyboard presses. """
        key = event.key()
        logger.debug('keyPressEvent(): key={}'.format(key))
#        if key == QtCore.Qt.Key_Right :
#            self.data_handler.z.set_value(self.data_handler.z.get_value() + 1)
#        elif key == QtCore.Qt.Key_Left :
#            self.data_handler.z.set_value(self.data_handler.z.get_value() - 1)
        if key == QtCore.Qt.Key_R :
            self.cutline.flip_orientation()
        else :
            event.ignore()
            return
        # If any if-statement matched, we accepted the event
        event.accept()

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
    main_window.data_handler.prepare_data(filename)
    app.exec_()

