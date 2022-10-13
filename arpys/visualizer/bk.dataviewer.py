import os

from pyqtgraph.Qt import QtGui, QtCore
from data_slicer.imageplot import ImagePlot, CursorPlot

import arpys.visualizer.datahandler as dh

class UtilitiesPanel(QtGui.QWidget):
    pass

class Viewer(QtGui.QMainWindow) :

    def __init__(self) :
        super().__init__()
        if index is not None :
            self.title = os.path.basename(index)
        else :
            self.title = 'ARPES 3D data viewer'
        self.initUI()

    def initUI(self) :
        """ Initialize the user interface. """
        self.setWindowTitle(self.title)
        # Create a "central widget" and its layout
        self.layout = QtGui.QGridLayout()
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Create the different widgets
        # Create the 3D (main) ImagePlot
        self.main_plot = ImagePlot(name='main_plot')
        # Create cut plot along x
        self.cut_x = ImagePlot(name='cut_x')
        # Create cut of cut_x
        self.plot_x = CursorPlot(name='plot_x')
        # Create cut plot along y
        self.cut_y = ImagePlot(name='cut_y', orientation='vertical')
        # Create cut of cut_y
        self.plot_y = CursorPlot(name='plot_y', orientation='vertical')
        # Create the integrated intensity plot
        self.plot_z = CursorPlot(name='plot_z', z_plot=True)

        self.util_panel = QtGui.QWidget()

        # Align GUI elements and show
        self.align()
        self.show()

    def align(self) :
        """ Define the geometry of this window and place all sub-widgets.

              0   1   2   3
            +---+---+---+---+
            |utilities  |bts| 0
            +---+---+---+---+
            | mdc x |       | 1
            +-------+  edc  |
            | cut x |       | 2
            +-------+-------+
            |       | c | m | 3
            | main  |   |   |  
            |       | y | y | 4
            +---+---+---+---+

            bts: buttons
            (Units of subdivision [sd])
        """
        # subdivision
        sd = 1
        # Get a short handle
        l = self.layout
        # addWidget(row, column, rowSpan, columnSpan)
        # utilities bar
        l.addWidget(self.util_panel,    0,      0,          2 * sd, 5 * sd)
        # X cut and mdc
        l.addWidget(self.plot_x,        2 * sd, 0,          1 * sd, 2 * sd)
        l.addWidget(self.cut_x,         3 * sd, 0,          1 * sd, 2 * sd)
        # Main plot
        l.addWidget(self.main_plot,     4 * sd, 0,          2 * sd, 2 * sd)
        # Y cut and mdc
        l.addWidget(self.cut_y,         4 * sd, 2,          2 * sd, 1 * sd)
        l.addWidget(self.plot_y,        4 * sd, 3 * sd,     2 * sd, 1 * sd)
        # EDC (integrated)
        l.addWidget(self.plot_z,        2 * sd, 2 * sd,     2 * sd, 2 * sd)

        nrows = 6 * sd
        ncols = 4 * sd
        # Need to manually set all row- and columnspans as well as min-sizes
        for i in range(nrows):
            l.setRowMinimumHeight(i, 50)
            # l.setRowMaximumHeight(i, 51)
            l.setRowStretch(i, 1)
        for i in range(ncols):
            l.setColumnMinimumWidth(i, 50)
            # l.setColumnMaximumWidth(i, 51)
            l.setColumnStretch(i, 1)


class Viewer3D(Viewer) :
    pass
