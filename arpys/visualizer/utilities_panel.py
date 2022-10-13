from matplotlib.pyplot import colormaps
from pyqtgraph.Qt import QtCore, QtGui

util_panel_style = """
QFrame{
    margin:5px; 
    border:1px solid rgb(150,150,150);
    }
QLabel{
    color: rgb(246, 246, 246);
    border:1px solid rgb(64, 64, 64);
    }
QCheckBox{
    color: rgb(246, 246, 246);
    }
"""

DEFAULT_CMAP = 'coolwarm'

bold_font = QtGui.QFont()
bold_font.setBold(True)

class UtilitiesPanel(QtGui.QWidget):
    """ Utilities panel on the top. Mostly just creating and aligning the stuff, signals and callbacks are
    handled in 'MainWindow' """
    # TODO Fermi edge fitting
    # TODO k-space conversion
    # TODO ROI
    # TODO multiple plotting tool
    # TODO logbook!
    # TODO curvature along specific directions
    # TODO smoothing along specific directions
    # TODO colorscale

    def __init__(self, main_window, name=None, dim=3):

        super().__init__()

        self.mw = main_window
        self.layout = QtGui.QGridLayout()
        self.tabs = QtGui.QTabWidget()
        self.tabs_visible = True
        self.dim = dim

        self.close_button = QtGui.QPushButton('close')
        self.save_button = QtGui.QPushButton('save')
        self.slicer_button = QtGui.QPushButton('open slicer')
        self.hide_button = QtGui.QPushButton('hide tabs')

        self.buttons = QtGui.QWidget()
        self.buttons_layout = QtGui.QGridLayout()
        self.buttons_layout.addWidget(self.close_button,    1, 0)
        self.buttons_layout.addWidget(self.save_button,     2, 0)
        self.buttons_layout.addWidget(self.slicer_button,   3, 0)
        self.buttons_layout.addWidget(self.hide_button,     4, 0)
        self.buttons.setLayout(self.buttons_layout)

        if name is not None:
            self.name = name
        else:
            self.name = 'Unnamed'

        self.initUI()

    def initUI(self):

        self.setStyleSheet(util_panel_style)
        momentum_labels_width = 80
        energy_labels_width = 80
        self.tabs_rows_span = 4
        self.tabs_cols_span = 9

        self.align()

        if self.dim == 2:
            self.energy_vert_value.setFixedWidth(energy_labels_width)
            self.momentum_hor_value.setFixedWidth(momentum_labels_width)
        elif self.dim == 3:
            self.energy_main_value.setFixedWidth(energy_labels_width)
            self.energy_hor_value.setFixedWidth(energy_labels_width)
            self.energy_vert_value.setFixedWidth(energy_labels_width)
            self.momentum_hor_value.setFixedWidth(momentum_labels_width)
            self.momentum_vert_value.setFixedWidth(momentum_labels_width)

        self.layout.addWidget(self.tabs,            0, 0, self.tabs_rows_span, self.tabs_cols_span)
        self.layout.addWidget(self.buttons,         0, self.tabs_cols_span + 1)
        self.setLayout(self.layout)

        # file options
        self.file_show_md_button.clicked.connect(self.show_metadata_window)
        self.file_add_md_button.clicked.connect(self.add_metadata)
        self.file_remove_md_button.clicked.connect(self.remove_metadata)
        self.file_sum_datasets_sum_button.clicked.connect(self.sum_datasets)
        self.file_sum_datasets_reset_button.clicked.connect(self.reset_summation)
        self.file_jn_button.clicked.connect(self.open_jupyter_notebook)

        # connect callbacks
        self.hide_button.clicked.connect(self.hidde_tabs)

        self.setup_cmaps()
        self.setup_gamma()
        self.setup_colorscale()
        self.setup_bin_z()

    def align(self):

        self.set_sliders_tab()
        self.set_image_tab()
        self.set_axes_tab()
        if self.dim == 3:
            self.set_orientate_tab()
        self.set_file_tab()

    def hidde_tabs(self):
        self.tabs_visible = not self.tabs_visible
        self.tabs.setVisible(self.tabs_visible)
        if self.tabs_visible:
            self.hide_button.setText('hide tabs')
        else:
            self.hide_button.setText('show tabs')

    def set_image_tab(self):

        max_w = 80
        # create elements
        self.image_tab = QtGui.QWidget()
        itl = QtGui.QGridLayout()
        self.image_colors_label = QtGui.QLabel('Colors')
        self.image_colors_label.setFont(bold_font)
        self.image_cmaps_label = QtGui.QLabel('cmaps:')
        self.image_cmaps = QtGui.QComboBox()
        self.image_invert_colors = QtGui.QCheckBox('invert colors')
        self.image_gamma_label = QtGui.QLabel('gamma:')
        self.image_gamma = QtGui.QDoubleSpinBox()
        self.image_gamma.setRange(0.05, 10)
        self.image_colorscale_label = QtGui.QLabel('color scale:')
        self.image_colorscale = QtGui.QDoubleSpinBox()

        self.image_pmesh = QtGui.QCheckBox('pmesh')

        self.image_other_lbl = QtGui.QLabel('Normalize')
        self.image_other_lbl.setFont(bold_font)
        self.image_normalize_edcs = QtGui.QCheckBox('normalize by each EDC')

        self.image_BZ_contour_lbl = QtGui.QLabel('BZ contour')
        self.image_BZ_contour_lbl.setFont(bold_font)
        self.image_show_BZ = QtGui.QCheckBox('show')
        self.image_symmetry_label = QtGui.QLabel('symmetry:')
        self.image_symmetry = QtGui.QSpinBox()
        self.image_symmetry.setRange(4, 6)
        self.image_rotate_BZ_label = QtGui.QLabel('rotate:')
        self.image_rotate_BZ = QtGui.QDoubleSpinBox()
        self.image_rotate_BZ.setRange(-90, 90)
        self.image_rotate_BZ.setSingleStep(0.5)

        self.image_2dv_lbl = QtGui.QLabel('Open in 2D viewer')
        self.image_2dv_lbl.setFont(bold_font)
        self.image_2dv_cut_selector_lbl = QtGui.QLabel('select cut')
        self.image_2dv_cut_selector = QtGui.QComboBox()
        self.image_2dv_cut_selector.addItems(['vertical', 'horizontal'])
        self.image_2dv_button = QtGui.QPushButton('Open')

        self.image_smooth_lbl = QtGui.QLabel('Smooth')
        self.image_smooth_lbl.setFont(bold_font)
        self.image_smooth_n_lbl = QtGui.QLabel('box size:')
        self.image_smooth_n = QtGui.QSpinBox()
        self.image_smooth_n.setValue(3)
        self.image_smooth_n.setRange(3, 50)
        self.image_smooth_n.setMaximumWidth(max_w)
        self.image_smooth_rl_lbl = QtGui.QLabel('recursion:')
        self.image_smooth_rl = QtGui.QSpinBox()
        self.image_smooth_rl.setValue(3)
        self.image_smooth_rl.setRange(1, 20)
        self.image_smooth_button = QtGui.QPushButton('Smooth')

        self.image_curvature_lbl = QtGui.QLabel('Curvature')
        self.image_curvature_lbl.setFont(bold_font)
        self.image_curvature_method_lbl = QtGui.QLabel('method:')
        self.image_curvature_method = QtGui.QComboBox()
        curvature_methods = ['2D', '1D (EDC)', '1D (MDC)']
        self.image_curvature_method.addItems(curvature_methods)
        self.image_curvature_a_lbl = QtGui.QLabel('a:')
        self.image_curvature_a = QtGui.QDoubleSpinBox()
        self.image_curvature_a.setRange(0, 1e10)
        self.image_curvature_a.setSingleStep(0.0001)
        self.image_curvature_a.setValue(100.)
        self.image_curvature_a.setMaximumWidth(max_w)
        self.image_curvature_button = QtGui.QPushButton('Do curvature')

        sd = 1
        # addWidget(widget, row, column, rowSpan, columnSpan)
        row = 0
        itl.addWidget(self.image_colors_label,          row * sd, 0)
        itl.addWidget(self.image_cmaps_label,           row * sd, 1)
        itl.addWidget(self.image_cmaps,                 row * sd, 2)
        itl.addWidget(self.image_invert_colors,         row * sd, 3)
        itl.addWidget(self.image_pmesh,                 row * sd, 4)

        row = 1
        itl.addWidget(self.image_gamma_label,           row * sd, 1)
        itl.addWidget(self.image_gamma,                 row * sd, 2)
        itl.addWidget(self.image_colorscale_label,      row * sd, 3)
        itl.addWidget(self.image_colorscale,            row * sd, 4)

        row = 2
        itl.addWidget(self.image_other_lbl,             row * sd, 0)
        itl.addWidget(self.image_normalize_edcs,        row * sd, 1, 1, 2)

        if self.dim == 2:
            row = 3
            itl.addWidget(self.image_smooth_lbl,        row * sd, 0)
            itl.addWidget(self.image_smooth_n_lbl,      row * sd, 1)
            itl.addWidget(self.image_smooth_n,          row * sd, 2)
            itl.addWidget(self.image_smooth_rl_lbl,     row * sd, 3)
            itl.addWidget(self.image_smooth_rl,         row * sd, 4)
            itl.addWidget(self.image_smooth_button,     row * sd, 5, 1, 2)

            row = 4
            itl.addWidget(self.image_curvature_lbl,         row * sd, 0)
            itl.addWidget(self.image_curvature_method_lbl,  row * sd, 1)
            itl.addWidget(self.image_curvature_method,      row * sd, 2)
            itl.addWidget(self.image_curvature_a_lbl,       row * sd, 3)
            itl.addWidget(self.image_curvature_a,           row * sd, 4)
            itl.addWidget(self.image_curvature_button,      row * sd, 5, 1, 2)

        if self.dim == 3:
            row = 3
            itl.addWidget(self.image_BZ_contour_lbl,    row * sd, 0)
            itl.addWidget(self.image_symmetry_label,    row * sd, 1)
            itl.addWidget(self.image_symmetry,          row * sd, 2)
            itl.addWidget(self.image_rotate_BZ_label,   row * sd, 3)
            itl.addWidget(self.image_rotate_BZ,         row * sd, 4)
            itl.addWidget(self.image_show_BZ,           row * sd, 5)

            row = 4
            itl.addWidget(self.image_2dv_lbl,               row, 0, 1, 2)
            itl.addWidget(self.image_2dv_cut_selector_lbl,  row, 2)
            itl.addWidget(self.image_2dv_cut_selector,      row, 3)
            itl.addWidget(self.image_2dv_button,            row, 4)

            # dummy item
            dummy_lbl = QtGui.QLabel('')
            itl.addWidget(dummy_lbl, 5, 0, 1, 7)

        self.image_tab.layout = itl
        self.image_tab.setLayout(itl)
        self.tabs.addTab(self.image_tab, 'Image')

    def set_sliders_tab(self):

        self.sliders_tab = QtGui.QWidget()
        vtl = QtGui.QGridLayout()
        max_lbl_w = 40
        bin_box_w = 50
        coords_box_w = 70

        if self.dim == 2:
            # binning option
            self.bins_label = QtGui.QLabel('Integrate')
            self.bins_label.setFont(bold_font)
            self.bin_y = QtGui.QCheckBox('bin EDCs')
            self.bin_y_nbins = QtGui.QSpinBox()
            self.bin_z = QtGui.QCheckBox('bin MDCs')
            self.bin_z_nbins = QtGui.QSpinBox()

            # cross' hairs positions
            self.positions_momentum_label = QtGui.QLabel('Momentum sliders')
            self.positions_momentum_label.setFont(bold_font)
            self.energy_vert_label = QtGui.QLabel('E:')
            self.energy_vert = QtGui.QSpinBox()
            self.energy_vert_value = QtGui.QLabel('eV')
            self.momentum_hor_label = QtGui.QLabel('kx:')
            self.momentum_hor = QtGui.QSpinBox()
            self.momentum_hor_value = QtGui.QLabel('deg')

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            col = 0
            vtl.addWidget(self.bins_label,                0 * sd, col, 1, 3)
            vtl.addWidget(self.bin_y,                     1 * sd, col * sd)
            vtl.addWidget(self.bin_y_nbins,               1 * sd, (col+1) * sd)
            vtl.addWidget(self.bin_z,                     2 * sd, col * sd)
            vtl.addWidget(self.bin_z_nbins,               2 * sd, (col+1) * sd)

            col = 3
            vtl.addWidget(self.positions_momentum_label,  0 * sd, col, 1, 3)
            vtl.addWidget(self.energy_vert_label,         1 * sd, col)
            vtl.addWidget(self.energy_vert,               1 * sd, (col+1) * sd)
            vtl.addWidget(self.energy_vert_value,         1 * sd, (col+2) * sd)
            vtl.addWidget(self.momentum_hor_label,        2 * sd, col)
            vtl.addWidget(self.momentum_hor,              2 * sd, (col+1) * sd)
            vtl.addWidget(self.momentum_hor_value,        2 * sd, (col+2) * sd)

            # dummy lbl
            dummy_lbl = QtGui.QLabel('')
            vtl.addWidget(dummy_lbl, 0, 6, 5, 2)

        elif self.dim == 3:
            # binning option
            self.bin_z = QtGui.QCheckBox('bin E')
            self.bin_z_nbins = QtGui.QSpinBox()
            self.bin_z_nbins.setMaximumWidth(bin_box_w)
            self.bin_x = QtGui.QCheckBox('bin kx')
            self.bin_x_nbins = QtGui.QSpinBox()
            self.bin_x_nbins.setMaximumWidth(bin_box_w)
            self.bin_y = QtGui.QCheckBox('bin ky')
            self.bin_y_nbins = QtGui.QSpinBox()
            self.bin_y_nbins.setMaximumWidth(bin_box_w)
            self.bin_zx = QtGui.QCheckBox('bin E (kx)')
            self.bin_zx_nbins = QtGui.QSpinBox()
            self.bin_zx_nbins.setMaximumWidth(bin_box_w)
            self.bin_zy = QtGui.QCheckBox('bin E (ky)')
            self.bin_zy_nbins = QtGui.QSpinBox()
            self.bin_zy_nbins.setMaximumWidth(bin_box_w)

            # cross' hairs positions
            self.positions_energies_label = QtGui.QLabel('Energy sliders')
            self.positions_energies_label.setFont(bold_font)
            self.energy_main_label = QtGui.QLabel('main:')
            self.energy_main_label.setMaximumWidth(max_lbl_w)
            self.energy_main = QtGui.QSpinBox()
            self.energy_main.setMaximumWidth(coords_box_w)
            self.energy_main_value = QtGui.QLabel('eV')
            self.energy_hor_label = QtGui.QLabel('kx:')
            self.energy_hor_label.setMaximumWidth(max_lbl_w)
            self.energy_hor = QtGui.QSpinBox()
            self.energy_hor.setMaximumWidth(coords_box_w)
            self.energy_hor_value = QtGui.QLabel('eV')
            self.energy_vert_label = QtGui.QLabel('ky:')
            self.energy_vert_label.setMaximumWidth(max_lbl_w)
            self.energy_vert = QtGui.QSpinBox()
            self.energy_vert.setMaximumWidth(coords_box_w)
            self.energy_vert_value = QtGui.QLabel('eV')

            self.positions_momentum_label = QtGui.QLabel('Momentum sliders')
            self.positions_momentum_label.setFont(bold_font)
            self.momentum_hor_label = QtGui.QLabel('ky:')
            self.momentum_hor_label.setMaximumWidth(max_lbl_w)
            self.momentum_hor = QtGui.QSpinBox()
            self.momentum_hor.setMaximumWidth(coords_box_w)
            self.momentum_hor_value = QtGui.QLabel('deg')
            self.momentum_vert_label = QtGui.QLabel('kx:')
            self.momentum_vert_label.setMaximumWidth(max_lbl_w)
            self.momentum_vert = QtGui.QSpinBox()
            self.momentum_vert.setMaximumWidth(coords_box_w)
            self.momentum_vert_value = QtGui.QLabel('deg')

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            col = 0
            vtl.addWidget(self.positions_energies_label, 0 * sd, col * sd, 1, 3)
            vtl.addWidget(self.energy_main_label, 1 * sd, col * sd)
            vtl.addWidget(self.energy_main, 1 * sd, (col + 1) * sd)
            vtl.addWidget(self.energy_main_value, 1 * sd, (col + 2) * sd)
            vtl.addWidget(self.energy_hor_label, 2 * sd, col * sd)
            vtl.addWidget(self.energy_hor, 2 * sd, (col + 1) * sd)
            vtl.addWidget(self.energy_hor_value, 2 * sd, (col + 2) * sd)
            vtl.addWidget(self.energy_vert_label, 3 * sd, col * sd)
            vtl.addWidget(self.energy_vert, 3 * sd, (col + 1) * sd)
            vtl.addWidget(self.energy_vert_value, 3 * sd, (col + 2) * sd)

            col = 3
            vtl.addWidget(self.positions_momentum_label, 0 * sd, col * sd, 1, 3)
            vtl.addWidget(self.momentum_vert_label, 1 * sd, col * sd)
            vtl.addWidget(self.momentum_vert, 1 * sd, (col + 1) * sd)
            vtl.addWidget(self.momentum_vert_value, 1 * sd, (col + 2) * sd)
            vtl.addWidget(self.momentum_hor_label, 2 * sd, col * sd)
            vtl.addWidget(self.momentum_hor, 2 * sd, (col + 1) * sd)
            vtl.addWidget(self.momentum_hor_value, 2 * sd, (col + 2) * sd)

            col = 6
            vtl.addWidget(self.bin_z, 0 * sd, col * sd)
            vtl.addWidget(self.bin_z_nbins, 0 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_x, 1 * sd, col * sd)
            vtl.addWidget(self.bin_x_nbins, 1 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_y, 2 * sd, col * sd)
            vtl.addWidget(self.bin_y_nbins, 2 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_zx, 3 * sd, col * sd)
            vtl.addWidget(self.bin_zx_nbins, 3 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_zy, 4 * sd, col * sd)
            vtl.addWidget(self.bin_zy_nbins, 4 * sd, (col + 1) * sd)

        self.sliders_tab.layout = vtl
        self.sliders_tab.setLayout(vtl)
        self.tabs.addTab(self.sliders_tab, 'Volume')

    def set_axes_tab(self):
        self.axes_tab = QtGui.QWidget()
        atl = QtGui.QGridLayout()
        box_max_w = 100
        lbl_max_h = 30

        self.axes_energy_main_lbl = QtGui.QLabel('Energy correction')
        self.axes_energy_main_lbl.setFont(bold_font)
        self.axes_energy_main_lbl.setMaximumHeight(lbl_max_h)
        self.axes_energy_Ef_lbl = QtGui.QLabel('Ef (eV):')
        # self.axes_energy_Ef_lbl.setMaximumWidth(max_lbl_w)
        self.axes_energy_Ef = QtGui.QDoubleSpinBox()
        self.axes_energy_Ef.setMaximumWidth(box_max_w)
        self.axes_energy_Ef.setRange(-5000., 5000)
        self.axes_energy_Ef.setDecimals(6)
        # self.axes_energy_Ef.setMinimumWidth(100)
        self.axes_energy_Ef.setSingleStep(0.001)

        self.axes_energy_hv_lbl = QtGui.QLabel('h\u03BD (eV):')
        # self.axes_energy_hv_lbl.setMaximumWidth(max_w)
        self.axes_energy_hv = QtGui.QDoubleSpinBox()
        self.axes_energy_hv.setMaximumWidth(box_max_w)
        self.axes_energy_hv.setRange(-2000., 2000)
        self.axes_energy_hv.setDecimals(4)
        self.axes_energy_hv.setSingleStep(0.001)

        self.axes_energy_wf_lbl = QtGui.QLabel('wf (eV):')
        # self.axes_energy_wf_lbl.setMaximumWidth(max_w)
        self.axes_energy_wf = QtGui.QDoubleSpinBox()
        self.axes_energy_wf.setMaximumWidth(box_max_w)
        self.axes_energy_wf.setRange(0, 5)
        self.axes_energy_wf.setDecimals(4)
        self.axes_energy_wf.setSingleStep(0.001)

        self.axes_energy_scale_lbl = QtGui.QLabel('scale:')
        self.axes_energy_scale = QtGui.QComboBox()
        self.axes_energy_scale.addItems(['binding', 'kinetic'])

        self.axes_momentum_main_lbl = QtGui.QLabel('k-space conversion')
        self.axes_momentum_main_lbl.setFont(bold_font)
        self.axes_momentum_main_lbl.setMaximumHeight(lbl_max_h)
        self.axes_gamma_x_lbl = QtGui.QLabel('\u0393 x0:')
        self.axes_gamma_x = QtGui.QSpinBox()
        self.axes_gamma_x.setRange(0, 5000)

        self.axes_transform_kz = QtGui.QCheckBox('Transform to kz')

        # self.axes_conv_hv_lbl = QtGui.QLabel('h\u03BD (eV):')
        # self.axes_conv_hv = QtGui.QDoubleSpinBox()
        # self.axes_conv_hv.setMaximumWidth(box_max_w)
        # self.axes_conv_hv.setRange(-2000., 2000.)
        # self.axes_conv_hv.setDecimals(3)
        # self.axes_conv_hv.setSingleStep(0.001)
        #
        # self.axes_conv_wf_lbl = QtGui.QLabel('wf (eV):')
        # self.axes_conv_wf = QtGui.QDoubleSpinBox()
        # self.axes_conv_wf.setMaximumWidth(box_max_w)
        # self.axes_conv_wf.setRange(0, 5)
        # self.axes_conv_wf.setDecimals(3)
        # self.axes_conv_wf.setSingleStep(0.001)

        self.axes_conv_lc_lbl = QtGui.QLabel('a (\u212B):')
        self.axes_conv_lc = QtGui.QDoubleSpinBox()
        self.axes_conv_lc.setMaximumWidth(box_max_w)
        self.axes_conv_lc.setRange(0, 10)
        self.axes_conv_lc.setDecimals(4)
        self.axes_conv_lc.setSingleStep(0.001)
        self.axes_conv_lc.setValue(3.1416)

        self.axes_conv_lc_op_lbl = QtGui.QLabel('c (\u212B):')
        self.axes_conv_lc_op = QtGui.QDoubleSpinBox()
        self.axes_conv_lc_op.setMaximumWidth(box_max_w)
        self.axes_conv_lc_op.setRange(0, 10)
        self.axes_conv_lc_op.setDecimals(4)
        self.axes_conv_lc_op.setSingleStep(0.001)
        self.axes_conv_lc_op.setValue(3.1416)

        self.axes_slit_orient_lbl = QtGui.QLabel('Slit:')
        self.axes_slit_orient = QtGui.QComboBox()
        self.axes_slit_orient.addItems(['horizontal', 'vertical', 'deflection'])
        self.axes_copy_values = QtGui.QPushButton('Copy from \'Orientate\'')
        self.axes_do_kspace_conv = QtGui.QPushButton('Convert')
        self.axes_reset_conv = QtGui.QPushButton('Reset')

        if self.dim == 2:

            self.axes_angle_off_lbl = QtGui.QLabel('angle offset:')
            self.axes_angle_off = QtGui.QDoubleSpinBox()
            self.axes_angle_off.setMaximumWidth(box_max_w)
            self.axes_angle_off.setDecimals(4)
            self.axes_angle_off.setSingleStep(0.0001)

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            row = 0
            atl.addWidget(self.axes_energy_main_lbl,    row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_energy_scale_lbl,   row * sd, 4 * sd)
            atl.addWidget(self.axes_energy_scale,       row * sd, 5 * sd)
            atl.addWidget(self.axes_energy_Ef_lbl,      (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_energy_Ef,          (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_energy_hv_lbl,      (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_energy_hv,          (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_energy_wf_lbl,      (row + 1) * sd, 4 * sd)
            atl.addWidget(self.axes_energy_wf,          (row + 1) * sd, 5 * sd)

            row = 2
            atl.addWidget(self.axes_momentum_main_lbl,  row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_gamma_x_lbl,        (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_gamma_x,            (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_angle_off_lbl,      (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_angle_off,          (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_conv_lc_lbl,        (row + 1) * sd, 4 * sd)
            atl.addWidget(self.axes_conv_lc,            (row + 1) * sd, 5 * sd)
            # atl.addWidget(self.axes_conv_hv_lbl,        (row + 1) * sd, 4 * sd)
            # atl.addWidget(self.axes_conv_hv,            (row + 1) * sd, 5 * sd)

            row = 4
            # atl.addWidget(self.axes_conv_wf_lbl,        row * sd, 0 * sd)
            # atl.addWidget(self.axes_conv_wf,            row * sd, 1 * sd)
            atl.addWidget(self.axes_slit_orient_lbl,    row * sd, 0 * sd)
            atl.addWidget(self.axes_slit_orient,        row * sd, 1 * sd)
            atl.addWidget(self.axes_do_kspace_conv,     row * sd, 2 * sd, 1, 2)
            atl.addWidget(self.axes_reset_conv,         row * sd, 4 * sd, 1, 2)

            # # dummy item
            # self.axes_massage_lbl = QtGui.QLabel('')
            # atl.addWidget(self.axes_massage_lbl, 6, 0, 1, 9)

        elif self.dim == 3:

            self.axes_gamma_y_lbl = QtGui.QLabel('\u0393 y0')
            self.axes_gamma_y = QtGui.QSpinBox()
            self.axes_gamma_y.setRange(0, 5000)

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            row = 0
            atl.addWidget(self.axes_energy_main_lbl,    row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_energy_scale_lbl,   row * sd, 4 * sd)
            atl.addWidget(self.axes_energy_scale,       row * sd, 5 * sd)
            atl.addWidget(self.axes_energy_Ef_lbl,      (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_energy_Ef,          (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_energy_hv_lbl,      (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_energy_hv,          (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_energy_wf_lbl,      (row + 1) * sd, 4 * sd)
            atl.addWidget(self.axes_energy_wf,          (row + 1) * sd, 5 * sd)

            row = 2
            atl.addWidget(self.axes_momentum_main_lbl,  row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_gamma_x_lbl,        (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_gamma_x,            (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_gamma_y_lbl,        (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_gamma_y,            (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_transform_kz,       (row + 1) * sd, 4 * sd, 1, 2)

            row = 4
            atl.addWidget(self.axes_conv_lc_lbl,        row * sd, 0 * sd)
            atl.addWidget(self.axes_conv_lc,            row * sd, 1 * sd)
            atl.addWidget(self.axes_conv_lc_op_lbl,     row * sd, 2 * sd)
            atl.addWidget(self.axes_conv_lc_op,         row * sd, 3 * sd)
            atl.addWidget(self.axes_slit_orient_lbl,    row * sd, 4 * sd)
            atl.addWidget(self.axes_slit_orient,        row * sd, 5 * sd)

            row = 5
            atl.addWidget(self.axes_copy_values,        row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_do_kspace_conv,     row * sd, 2 * sd, 1, 2)
            atl.addWidget(self.axes_reset_conv,         row * sd, 4 * sd, 1, 2)

            # # dummy item
            # self.axes_massage_lbl = QtGui.QLabel('')
            # atl.addWidget(self.axes_massage_lbl, 5, 0, 1, 9)

        self.axes_tab.layout = atl
        self.axes_tab.setLayout(atl)
        self.tabs.addTab(self.axes_tab, 'Axes')

    def set_orientate_tab(self):

        self.orientate_tab = QtGui.QWidget()
        otl = QtGui.QGridLayout()

        self.orientate_init_cooradinates_lbl = QtGui.QLabel('Give initial coordinates')
        self.orientate_init_cooradinates_lbl.setFont(bold_font)
        self.orientate_init_x_lbl = QtGui.QLabel('scanned axis:')
        self.orientate_init_x = QtGui.QSpinBox()
        self.orientate_init_x.setRange(0, 1000)
        self.orientate_init_y_lbl = QtGui.QLabel('slit axis:')
        self.orientate_init_y = QtGui.QSpinBox()
        self.orientate_init_y.setRange(0, 1000)

        self.orientate_find_gamma = QtGui.QPushButton('Find \t \u0393')
        self.orientate_copy_coords = QtGui.QPushButton('Copy from \'Volume\'')

        self.orientate_find_gamma_message = QtGui.QLineEdit('NOTE: algorythm will process the main plot image.')
        self.orientate_find_gamma_message.setReadOnly(True)

        self.orientate_lines_lbl = QtGui.QLabel('Show rotatable lines')
        self.orientate_lines_lbl.setFont(bold_font)
        self.orientate_hor_line = QtGui.QCheckBox('horizontal line')
        self.orientate_hor_line
        self.orientate_ver_line = QtGui.QCheckBox('vertical line')
        self.orientate_angle_lbl = QtGui.QLabel('rotation angle (deg):')
        self.orientate_angle = QtGui.QDoubleSpinBox()
        self.orientate_angle.setRange(-180, 180)
        self.orientate_angle.setSingleStep(0.5)

        self.orientate_info_button = QtGui.QPushButton('info')

        sd = 1
        # addWidget(widget, row, column, rowSpan, columnSpan)
        row = 0
        otl.addWidget(self.orientate_init_cooradinates_lbl,   row * sd, 0 * sd, 1, 2)
        otl.addWidget(self.orientate_init_x_lbl,              (row + 1) * sd, 0 * sd)
        otl.addWidget(self.orientate_init_x,                  (row + 1) * sd, 1 * sd)
        otl.addWidget(self.orientate_init_y_lbl,              (row + 1) * sd, 2 * sd)
        otl.addWidget(self.orientate_init_y,                  (row + 1) * sd, 3 * sd)

        row = 2
        otl.addWidget(self.orientate_find_gamma,              row * sd, 0 * sd, 1, 2)
        otl.addWidget(self.orientate_copy_coords,             row * sd, 2 * sd, 1, 2)
        otl.addWidget(self.orientate_find_gamma_message,      (row + 1) * sd, 0 * sd, 1, 4)

        col = 4
        otl.addWidget(self.orientate_lines_lbl,               0 * sd, col * sd, 1, 2)
        otl.addWidget(self.orientate_hor_line,                1 * sd, col * sd)
        otl.addWidget(self.orientate_ver_line,                1 * sd, (col + 1) * sd)
        otl.addWidget(self.orientate_angle_lbl,               2 * sd, col * sd)
        otl.addWidget(self.orientate_angle,                   2 * sd, (col + 1) * sd)
        otl.addWidget(self.orientate_info_button,                    3 * sd, (col + 1) * sd)

        # dummy lbl
        dummy_lbl = QtGui.QLabel('')
        otl.addWidget(dummy_lbl, 4, 0, 2, 8)

        self.orientate_tab.layout = otl
        self.orientate_tab.setLayout(otl)
        self.tabs.addTab(self.orientate_tab, 'Orientate')

        self.set_orientation_info_window()

    def set_file_tab(self):

        self.file_tab = QtGui.QWidget()
        ftl = QtGui.QGridLayout()

        self.file_add_md_lbl = QtGui.QLabel('Edit entries')
        self.file_add_md_lbl.setFont(bold_font)
        self.file_md_name_lbl = QtGui.QLabel('name:')
        self.file_md_name = QtGui.QLineEdit()
        self.file_md_value_lbl = QtGui.QLabel('value:')
        self.file_md_value = QtGui.QLineEdit()
        self.file_add_md_button = QtGui.QPushButton('add/update')
        self.file_remove_md_button = QtGui.QPushButton('remove')

        self.file_show_md_button = QtGui.QPushButton('show metadata')

        self.file_sum_datasets_lbl = QtGui.QLabel('Sum data sets')
        self.file_sum_datasets_lbl.setFont(bold_font)
        self.file_sum_datasets_fname_lbl = QtGui.QLabel('file name:')
        self.file_sum_datasets_fname = QtGui.QLineEdit('Only *.h5 files')
        self.file_sum_datasets_sum_button = QtGui.QPushButton('sum')
        self.file_sum_datasets_reset_button = QtGui.QPushButton('reset')

        self.file_jn_main_lbl = QtGui.QLabel('Jupyter')
        self.file_jn_main_lbl.setFont(bold_font)
        self.file_jn_fname_lbl = QtGui.QLabel('file name:')
        self.file_jn_fname = QtGui.QLineEdit(self.mw.title.split('.')[0])
        self.file_jn_button = QtGui.QPushButton('open in jn')

        self.file_mdc_fitter_lbl = QtGui.QLabel('MDC fitter')
        self.file_mdc_fitter_lbl.setFont(bold_font)
        self.file_mdc_fitter_button = QtGui.QPushButton('Open')

        self.file_edc_fitter_lbl = QtGui.QLabel('EDC fitter')
        self.file_edc_fitter_lbl.setFont(bold_font)
        self.file_edc_fitter_button = QtGui.QPushButton('Open')

        sd = 1
        # addWidget(widget, row, column, rowSpan, columnSpan)
        row = 0
        ftl.addWidget(self.file_add_md_lbl,                     row * sd, 0 * sd, 1, 2)
        ftl.addWidget(self.file_show_md_button,                 row * sd, 8 * sd, 1, 2)

        row = 1
        ftl.addWidget(self.file_md_name_lbl,                    row * sd, 0 * sd)
        ftl.addWidget(self.file_md_name,                        row * sd, 1 * sd, 1, 3)
        ftl.addWidget(self.file_md_value_lbl,                   row * sd, 4 * sd)
        ftl.addWidget(self.file_md_value,                       row * sd, 5 * sd, 1, 3)
        ftl.addWidget(self.file_add_md_button,                  row * sd, 8 * sd)
        ftl.addWidget(self.file_remove_md_button,               row * sd, 9 * sd)

        row = 2
        ftl.addWidget(self.file_sum_datasets_lbl,               row * sd, 0 * sd, 1, 2)
        # ftl.addWidget(self.file_sum_datasets_fname_lbl,         row * sd, 2 * sd)
        ftl.addWidget(self.file_sum_datasets_fname,             row * sd, 2 * sd, 1, 6)
        ftl.addWidget(self.file_sum_datasets_sum_button,        row * sd, 8 * sd)
        ftl.addWidget(self.file_sum_datasets_reset_button,      row * sd, 9 * sd)

        row = 3
        ftl.addWidget(self.file_jn_main_lbl,                    row * sd, 0 * sd, 1, 2)
        # ftl.addWidget(self.file_jn_fname_lbl,                   row * sd, 2 * sd)
        ftl.addWidget(self.file_jn_fname,                       row * sd, 2 * sd, 1, 6)
        ftl.addWidget(self.file_jn_button,                      row * sd, 8 * sd)

        if self.dim == 2:
            row = 4
            ftl.addWidget(self.file_mdc_fitter_lbl,             row * sd, 0, 1, 2)
            ftl.addWidget(self.file_mdc_fitter_button,          row * sd, 2)
            ftl.addWidget(self.file_edc_fitter_lbl,             row * sd, 4, 1, 2)
            ftl.addWidget(self.file_edc_fitter_button,          row * sd, 6)

        # dummy lbl
        # dummy_lbl = QtGui.QLabel('')
        # ftl.addWidget(dummy_lbl, 4, 0, 1, 9)

        self.file_tab.layout = ftl
        self.file_tab.setLayout(ftl)
        self.tabs.addTab(self.file_tab, 'File')

    def setup_cmaps(self):
        cm = self.image_cmaps
        for cmap in colormaps():
            cm.addItem(cmap)
        cm.setCurrentText(DEFAULT_CMAP)

    def setup_gamma(self):

        g = self.image_gamma
        g.setRange(0, 10)
        g.setValue(1)
        g.setSingleStep(0.05)

    def setup_colorscale(self):

        cs = self.image_colorscale
        cs.setRange(0, 2)
        cs.setValue(1)
        cs.setSingleStep(0.1)

    def setup_bin_z(self):

        bz = self.bin_z_nbins
        bz.setRange(0, 100)
        bz.setValue(0)

    def set_orientation_info_window(self):
        self.orient_info_window = QtGui.QWidget()
        oiw = QtGui.QGridLayout()

        self.oi_window_lbl = QtGui.QLabel('pyta -> beamline coordinates translator')
        self.oi_window_lbl.setFont(bold_font)
        self.oi_beamline_lbl = QtGui.QLabel('Beamline')
        self.oi_beamline_lbl.setFont(bold_font)
        self.oi_azimuth_lbl = QtGui.QLabel('Azimuth (clockwise)')
        self.oi_azimuth_lbl.setFont(bold_font)
        self.oi_analyzer_lbl = QtGui.QLabel('Analyzer (-> +)')
        self.oi_analyzer_lbl.setFont(bold_font)
        self.oi_scanned_lbl = QtGui.QLabel('Scanned (-> +)')
        self.oi_scanned_lbl.setFont(bold_font)

        entries = [['SIS (SLS, SIStem)',    'phi -> -',     'theta -> +',   'tilt -> -'],
                   ['SIS (SLS, SES)',       'phi -> +',     'theta -> -',   'tilt -> -'],
                   ['Bloch (MaxIV)',        'azimuth -> +', 'tilt -> -',    'polar -> -'],
                   ['CASSIOPEE (SOLEIL)',   '-',            '-',            '-'],
                   ['I05 (Diamond)',        '-',            '-',            '-'],
                   ['UARPES (SOLARIS)',     '-',            '-',            '-'],
                   ['APE (Elettra)',        '-',            '-',            '-'],
                   ['ADDRES (SLS)',         '-',            '-',            '-'],
                   ['-',                    '-',            '-',            '-'],
                   ['-',                    '-',            '-',            '-']]
        labels = {}

        sd = 1
        row = 0
        oiw.addWidget(self.oi_beamline_lbl,     row * sd, 0 * sd)
        oiw.addWidget(self.oi_azimuth_lbl,      row * sd, 1 * sd)
        oiw.addWidget(self.oi_analyzer_lbl,     row * sd, 2 * sd)
        oiw.addWidget(self.oi_scanned_lbl,      row * sd, 3 * sd)

        for entry in entries:
            row += 1
            labels[str(row)] = {}
            labels[str(row)]['beamline'] = QtGui.QLabel(entry[0])
            labels[str(row)]['azimuth'] = QtGui.QLabel(entry[1])
            labels[str(row)]['azimuth'].setAlignment(QtCore.Qt.AlignCenter)
            labels[str(row)]['analyzer'] = QtGui.QLabel(entry[2])
            labels[str(row)]['analyzer'].setAlignment(QtCore.Qt.AlignCenter)
            labels[str(row)]['scanned'] = QtGui.QLabel(entry[3])
            labels[str(row)]['scanned'].setAlignment(QtCore.Qt.AlignCenter)

            oiw.addWidget(labels[str(row)]['beamline'],    row * sd, 0 * sd)
            oiw.addWidget(labels[str(row)]['azimuth'],     row * sd, 1 * sd)
            oiw.addWidget(labels[str(row)]['analyzer'],    row * sd, 2 * sd)
            oiw.addWidget(labels[str(row)]['scanned'],     row * sd, 3 * sd)

        self.orient_info_window.layout = oiw
        self.orient_info_window.setLayout(oiw)

    def set_metadata_window(self, dataset):

        self.md_window = QtGui.QWidget()
        mdw = QtGui.QGridLayout()

        attribute_name_lbl = QtGui.QLabel('Attribute')
        attribute_name_lbl.setFont(bold_font)
        attribute_value_lbl = QtGui.QLabel('Value')
        attribute_value_lbl.setFont(bold_font)
        attribute_value_lbl.setAlignment(QtCore.Qt.AlignCenter)
        attribute_saved_lbl = QtGui.QLabel('user saved')
        attribute_saved_lbl.setFont(bold_font)
        attribute_saved_lbl.setAlignment(QtCore.Qt.AlignCenter)

        dataset = vars(dataset)
        entries = {}

        sd = 1
        row = 0
        mdw.addWidget(attribute_name_lbl,   row * sd, 0 * sd)
        mdw.addWidget(attribute_value_lbl,  row * sd, 1 * sd)

        row = 1
        for key in dataset.keys():
            if key == 'ekin' or key == 'saved':
                continue
            elif key == 'data':
                s = dataset[key].shape
                value = '(' + str(s[0]) + ',  ' + str(s[1]) + ',  ' + str(s[2]) + ')'
                entries[str(row)] = {}
                entries[str(row)]['name'] = QtGui.QLabel(key)
                entries[str(row)]['value'] = QtGui.QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'xscale':
                value = '({:.2f}  :  {:.2f})'.format(dataset[key][0], dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QtGui.QLabel(key)
                entries[str(row)]['value'] = QtGui.QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'yscale':
                value = '({:.4f}  :  {:.4f})'.format(dataset[key][0], dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QtGui.QLabel(key)
                entries[str(row)]['value'] = QtGui.QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'zscale':
                value = '({:.4f}  :  {:.4f})'.format(dataset[key][0], dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QtGui.QLabel(key)
                entries[str(row)]['value'] = QtGui.QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'kxscale':
                if not (dataset[key] is None):
                    value = '({:.3f}  :  {:.3f})'.format(dataset[key][0], dataset[key][-1])
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QtGui.QLabel(key)
                    entries[str(row)]['value'] = QtGui.QLabel(str(value))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QtGui.QLabel(key)
                    entries[str(row)]['value'] = QtGui.QLabel(str(dataset[key]))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'kyscale':
                if not (dataset[key] is None):
                    value = '({:.3f}  :  {:.3f})'.format(dataset[key][0], dataset[key][-1])
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QtGui.QLabel(key)
                    entries[str(row)]['value'] = QtGui.QLabel(str(value))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QtGui.QLabel(key)
                    entries[str(row)]['value'] = QtGui.QLabel(str(dataset[key]))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            else:
                entries[str(row)] = {}
                entries[str(row)]['name'] = QtGui.QLabel(key)
                entries[str(row)]['value'] = QtGui.QLabel(str(dataset[key]))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)

            mdw.addWidget(entries[str(row)]['name'],    row * sd, 0 * sd)
            mdw.addWidget(entries[str(row)]['value'],   row * sd, 1 * sd)
            row += 1

        if 'saved' in dataset.keys():
            mdw.addWidget(attribute_saved_lbl,   row * sd, 0 * sd, 1, 2)
            for key in dataset['saved'].keys():
                row += 1
                entries[str(row)] = {}
                entries[str(row)]['name'] = QtGui.QLabel(key)
                if key == 'kx' or key == 'ky' or key == 'k':
                    value = '({:.2f}  :  {:.2f})'.format(dataset['saved'][key][0], dataset['saved'][key][-1])
                    entries[str(row)]['value'] = QtGui.QLabel(str(value))
                else:
                    entries[str(row)]['value'] = QtGui.QLabel(str(dataset['saved'][key]))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)

                mdw.addWidget(entries[str(row)]['name'],    row * sd, 0 * sd)
                mdw.addWidget(entries[str(row)]['value'],   row * sd, 1 * sd)

        self.md_window.layout = mdw
        self.md_window.setLayout(mdw)

    def show_metadata_window(self):

        self.set_metadata_window(self.mw.data_set)
        title = self.mw.title + ' - metadata'
        self.info_box = InfoWindow(self.md_window, title)
        self.info_box.setMinimumWidth(350)
        self.info_box.show()

    def add_metadata(self):

        name = self.file_md_name.text()
        value = self.file_md_value.text()
        try:
            value = float(value)
        except ValueError:
            pass

        if name == '':
            empty_name_box = QtGui.QMessageBox()
            empty_name_box.setIcon(QtGui.QMessageBox.Information)
            empty_name_box.setText('Attribute\'s name not given.')
            empty_name_box.setStandardButtons(QtGui.QMessageBox.Ok)
            if empty_name_box.exec() == QtGui.QMessageBox.Ok:
                return

        message = 'Sure to add attribute \'{}\' with value <{}> (type: {}) to the file?'.format(
            name, value, type(value))
        sanity_check_box = QtGui.QMessageBox()
        sanity_check_box.setIcon(QtGui.QMessageBox.Question)
        sanity_check_box.setText(message)
        sanity_check_box.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
        if sanity_check_box.exec() == QtGui.QMessageBox.Ok:
            if hasattr(self.mw.data_set, name):
                attr_conflict_box = QtGui.QMessageBox()
                attr_conflict_box.setIcon(QtGui.QMessageBox.Question)
                attr_conflict_box.setText(f'Data set already has attribute \'{name}\'.  Overwrite?')
                attr_conflict_box.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
                if attr_conflict_box.exec() == QtGui.QMessageBox.Ok:
                    setattr(self.mw.data_set, name, value)
            else:
                dl.update_namespace(self.mw.data_set, [name, value])
        else:
            return

    def remove_metadata(self):

        name = self.file_md_name.text()

        if not hasattr(self.mw.data_set, name):
            no_attr_box = QtGui.QMessageBox()
            no_attr_box.setIcon(QtGui.QMessageBox.Information)
            no_attr_box.setText(f'Attribute \'{name}\' not found.')
            no_attr_box.setStandardButtons(QtGui.QMessageBox.Ok)
            if no_attr_box.exec() == QtGui.QMessageBox.Ok:
                return

        message = 'Sure to remove attribute \'{}\' from the data set?'.format(name)
        sanity_check_box = QtGui.QMessageBox()
        sanity_check_box.setIcon(QtGui.QMessageBox.Question)
        sanity_check_box.setText(message)
        sanity_check_box.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
        if sanity_check_box.exec() == QtGui.QMessageBox.Ok:
            delattr(self.mw.data_set, name)
        else:
            return

    def sum_datasets(self):

        if self.dim == 3:
            no_map_box = QtGui.QMessageBox()
            no_map_box.setIcon(QtGui.QMessageBox.Information)
            no_map_box.setText('Summing feature works only on cuts.')
            no_map_box.setStandardButtons(QtGui.QMessageBox.Ok)
            if no_map_box.exec() == QtGui.QMessageBox.Ok:
                return

        file_path = self.mw.fname[:-len(self.mw.title)] + self.file_sum_datasets_fname.text()
        org_dataset = dl.load_data(self.mw.fname)

        try:
            new_dataset = dl.load_data(file_path)
        except FileNotFoundError:
            no_file_box = QtGui.QMessageBox()
            no_file_box.setIcon(QtGui.QMessageBox.Information)
            no_file_box.setText('File not found.')
            no_file_box.setStandardButtons(QtGui.QMessageBox.Ok)
            if no_file_box.exec() == QtGui.QMessageBox.Ok:
                return

        try:
            check_result = self.check_conflicts([org_dataset, new_dataset])
        except AttributeError:
            not_h5_file_box = QtGui.QMessageBox()
            not_h5_file_box.setIcon(QtGui.QMessageBox.Information)
            not_h5_file_box.setText('Cut is not an SIStem *h5 file.')
            not_h5_file_box.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
            if not_h5_file_box.exec() == QtGui.QMessageBox.Cancel:
                return
            else:
                pass

        if check_result == 0:
            data_mismatch_box = QtGui.QMessageBox()
            data_mismatch_box.setIcon(QtGui.QMessageBox.Information)
            data_mismatch_box.setText('Data sets\' shapes don\'t match.\nConnot proceed.')
            data_mismatch_box.setStandardButtons(QtGui.QMessageBox.Ok)
            if data_mismatch_box.exec() == QtGui.QMessageBox.Ok:
                return

        check_result_box = QtGui.QMessageBox()
        check_result_box.setMinimumWidth(600)
        check_result_box.setMaximumWidth(1000)
        check_result_box.setIcon(QtGui.QMessageBox.Information)
        check_result_box.setText(check_result)
        check_result_box.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
        if check_result_box.exec() == QtGui.QMessageBox.Ok:
            self.mw.org_dataset = org_dataset
            self.mw.data_set.data += new_dataset.data
            self.mw.data_set.n_sweeps += new_dataset.n_sweeps
            d = np.swapaxes(self.mw.data_set.data, 1, 2)
            self.mw.data_handler.set_data(d)
            self.mw.update_main_plot()
        else:
            return

    def reset_summation(self):

        if self.mw.org_dataset is None:
            no_summing_yet_box = QtGui.QMessageBox()
            no_summing_yet_box.setIcon(QtGui.QMessageBox.Information)
            no_summing_yet_box.setText('No summing done yet.')
            no_summing_yet_box.setStandardButtons(QtGui.QMessageBox.Ok)
            if no_summing_yet_box.exec() == QtGui.QMessageBox.Ok:
                return

        reset_summation_box = QtGui.QMessageBox()
        reset_summation_box.setMinimumWidth(600)
        reset_summation_box.setMaximumWidth(1000)
        reset_summation_box.setIcon(QtGui.QMessageBox.Question)
        reset_summation_box.setText('Want to reset summation?')
        reset_summation_box.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
        if reset_summation_box.exec() == QtGui.QMessageBox.Ok:
            self.mw.data_set.data = self.mw.org_dataset.data
            self.mw.data_set.n_sweeps = self.mw.org_dataset.n_sweeps
            d = np.swapaxes(self.mw.data_set.data, 1, 2)
            self.mw.data_handler.set_data(d)
            self.mw.update_main_plot()
        else:
            return

    def open_jupyter_notebook(self):
        file_path = self.mw.fname[:-len(self.mw.title)] + self.file_jn_fname.text() + '.ipynb'
        template_path = os.path.dirname(os.path.abspath(__file__)) + '/'
        template_fname = 'template.ipynb'
        self.edit_file((template_path + template_fname), file_path)

        # Open jupyter notebook as a subprocess
        openJupyter = "jupyter notebook"
        subprocess.Popen(openJupyter, shell=True, cwd=self.mw.fname[:-len(self.mw.title)])

    def edit_file(self, template, new_file_name):
        os.system('touch ' + new_file_name)

        templ_file = open(template, 'r')
        templ_lines = templ_file.readlines()
        templ_file.close()

        new_lines = []

        # writing to file
        for line in templ_lines:
            if 'path = ' in line:
                line = '    "path = \'{}\'\\n",'.format(self.mw.fname[:-len(self.mw.title)])
            if 'fname = ' in line:
                line = '    "fname = \'{}\'\\n",'.format(self.mw.title)
            if 'slit_idx, e_idx =' in line:
                if self.dim == 2:
                    line = '    "slit_idx, e_idx = {}, {}\\n",'.format(
                        self.momentum_hor.value(), self.energy_vert.value())
                elif self.dim == 3:
                    line = '    "scan_idx, slit_idx, e_idx = {}, {}, {}\\n",'.format(
                        self.momentum_vert.value(), self.momentum_hor.value(), self.energy_vert.value())
            new_lines.append(line)

        new_file = open(new_file_name, 'w')
        new_file.writelines(new_lines)
        new_file.close()

    @staticmethod
    def check_conflicts(datasets):

        labels = ['fname', 'data', 'T', 'hv', 'polarization', 'PE', 'FE', 'exit', 'x', 'y', 'z', 'theta', 'phi', 'tilt',
                  'lens_mode', 'acq_mode', 'e_start', 'e_stop', 'e_step']
        to_check = [[] for _ in range(len(labels))]

        to_check[0] = ['original', 'new']
        for ds in datasets:
            to_check[1].append(ds.data.shape)
            to_check[2].append(ds.temp)
            to_check[3].append(ds.hv)
            to_check[4].append(ds.polarization)
            to_check[5].append(ds.PE)
            to_check[6].append(ds.FE)
            to_check[7].append(ds.exit_slit)
            to_check[8].append(ds.x)
            to_check[9].append(ds.y)
            to_check[10].append(ds.z)
            to_check[11].append(ds.theta)
            to_check[12].append(ds.phi)
            to_check[13].append(ds.tilt)
            to_check[14].append(ds.lens_mode)
            to_check[15].append(ds.acq_mode)
            to_check[16].append(ds.zscale[0])
            to_check[17].append(ds.zscale[-1])
            to_check[18].append(wp.get_step(ds.zscale))

        # check if imporatnt stuff match
        check_result = []
        for idx, lbl in enumerate(labels):
            if lbl == 'fname' or lbl == 'data':
                check_result.append(True)
            # temperature
            elif lbl == 'T':
                err = 1
                par = array(to_check[idx])
                to_compare = ones(par.size) * par[0]
                check_result.append(allclose(par, to_compare, atol=err))
            # photon energy
            elif lbl == 'hv':
                err = 0.1
                par = array(to_check[idx])
                to_compare = ones(par.size) * par[0]
                check_result.append(allclose(par, to_compare, atol=err))
            # e_min of analyzer
            elif lbl == 'e_start':
                err = to_check[-1][0]
                par = array(to_check[idx])
                to_compare = ones(par.size) * par[0]
                check_result.append(allclose(par, to_compare, atol=err))
            # e_max of analyzer
            elif lbl == 'e_stop':
                err = to_check[-1][0]
                par = array(to_check[idx])
                to_compare = ones(par.size) * par[0]
                check_result.append(allclose(par, to_compare, atol=err))
            elif lbl == 'e_step':
                err = to_check[-1][0]
                par = array(to_check[idx])
                to_compare = ones(par.size) * par[0]
                check_result.append(allclose(par, to_compare, atol=err * 0.1))
            else:
                check_result.append(to_check[idx][0] == to_check[idx][1])

        if not (to_check[1][0] == to_check[1][1]):
            return 0

        if array(check_result).all():
            message = 'Everything match! We\'re good to go!'
        else:
            message = 'Some stuff doesn\'t match...\n\n'
            dont_match = where(array(check_result) == False)
            for idx in dont_match[0]:
                try:
                    message += '{} \t\t {:.3f}\t  {:.3f} \n'.format(str(labels[idx]), to_check[idx][0], to_check[idx][1])
                except TypeError:
                    message += '{} \t\t {}\t  {} \n'.format(
                        str(labels[idx]), str(to_check[idx][0]), str(to_check[idx][1]))
                except ValueError:
                    message += '{} \t\t {}\t  {} \n'.format(
                        str(labels[idx]), str(to_check[idx][0]), str(to_check[idx][1]))
            message += '\nSure to proceed?'

        return message



