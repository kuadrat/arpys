from pyqtgraph.Qt import QtGui, QtCore

import arpys.dataloaders as dl

class DetailsView(QtGui.QWidget) :
    """ Display metadata of an ARPES scan. """
    # String to indicate a lack of a value
    PLACEHOLDER = '-'
    # Sizes of LineEdits
    LEN1 = 50
    LEN2 = 75
    number_fmt = '{:.2f}'
    number_fmt_exp = '{:.2E}'

    def __init__(self) :
        super().__init__()
        self.data = None
        # Create a bold font
        self.bold_font = QtGui.QFont()
        self.bold_font.setBold(True)
        self.line_edits = {}
        # Create UI elements
        self.create_dataloader_dropdown()
        self.create_scan_info()
        # Manipulator
        labels = ['x', 'y', 'z', 'T [K]', 'alpha', 'beta', 'gamma', 'p [mbar]']
        names = ['x_pos', 'y_pos', 'z_pos', 'temperature'] + 3*[None] + ['pressure']
        self.manipulator_layout = self.create_grid(labels, names)
        # Analyzer
        labels = ['E[0]', 'E[-1]', 'step', 'PE', 'lens mode', 'acq mode', 
                  'sweeps', 'DT']
        names = ['e0', 'e1', 'de', 'PE', 'lens_mode', 'acq_mode', 'n_sweeps', 
                 'dwell_time']
        self.analyzer_layout = self.create_grid(labels, names)
        # Beamline
        labels = ['hv', 'exit_slit', 'polarization']
        names = 3*[None]
        self.beamline_layout = self.create_grid(labels, names)

        self.align()
        self.setLayout(self.layout)

    def create_dataloader_dropdown(self) :
        """ Dropdown listing all available dataloaders. """
        # Create a dropdown and add its items
        self.dl_dropdown = QtGui.QComboBox()
        self.dl_dropdown.addItem('All')
        for dataloader in dl.all_dls :
            self.dl_dropdown.addItem(dataloader.name)

        # Put the dropdown in a layout and add a label
        label = QtGui.QLabel('Data loader:')
        label.setFont(self.bold_font)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.dl_dropdown)
        layout.addStretch()
        self.dropdown_layout = layout

    def create_scan_info(self) :
        """ Scan type and start-stop-step information.
        In terms of layout, stack two horizontal layouts in a vertical 
        one. The top one just contains the scan type, the bottom one the 
        start-stop-step info.
        """
        vlayout = QtGui.QVBoxLayout()
        # Top row
        top = QtGui.QHBoxLayout()
        label, line_edit = self.make_entry('Scan type', self.LEN2, name='scan_type')
        top.addWidget(label)
        top.addWidget(line_edit)
        top.addStretch()
        # Bottom row
        bottom = QtGui.QHBoxLayout()
        for label in ['start', 'stop', 'step'] :
            label, line_edit = self.make_entry(label, self.LEN1)
            bottom.addWidget(label)
            bottom.addWidget(line_edit)
        bottom.addStretch()
        # Combine and finish
        vlayout.addLayout(top)
        vlayout.addLayout(bottom)
        self.scan_info_layout = vlayout

    def create_grid(self, labels, names) :
        """ Create QLabels and QLineEdits for each (label, name) pair in 
        *labels* and *names*, arranged in a simple grid. 
        """
        layout = QtGui.QGridLayout()
        # Get enough rows
        n_rows = (len(labels)+1)//2
        n_cols = 4
        for i,label in enumerate(labels) :
            qlabel, line_edit = self.make_entry(label, self.LEN2, names[i])
            row = i%n_rows
            col = i//n_rows * n_cols/2
            layout.addWidget(qlabel, row, col)
            layout.addWidget(line_edit, row, col+1)
        return layout

    def make_entry(self, label, max_length, name=None) :
        """ Create a Read-Only QLineEdit with maximum width *max_width* and 
        an associated QLabel with text `*label*:` (i.e. a semicolon is 
        automatically attached).
        Return them as: QLabel, QLineEdit and add the QLineEdit to 
        <dict>`self.line_edits`.
        The optional argument *name* can be specified to register the created 
        line_edit under a different key in <dict>`self.line_edits`. If no 
        *name* is supplied, use *label*.
        """
        qlabel = QtGui.QLabel(label + ':')
        line_edit = QtGui.QLineEdit(self.PLACEHOLDER)
        line_edit.setMaximumWidth(max_length)
        line_edit.setReadOnly(True)
        if name is not None :
            key = name
        else :
            key = label
        self.line_edits.update({key: line_edit})
        return qlabel, line_edit

    def add_section(self, title, layout) :
        """ Add *title* and *layout* to the global QVBoxLayout. """
        subtitle = QtGui.QLabel(title)
        subtitle.setFont(self.bold_font)
        self.layout.addWidget(subtitle)
        self.layout.addLayout(layout)

    def align(self) :
        """ Define widget geometry. """
        self.layout = QtGui.QVBoxLayout()
        # Title
        title = QtGui.QLabel('Metadata')
        title.setFont(self.bold_font)
        self.layout.addWidget(title)
        # Dropdown
        self.layout.addLayout(self.dropdown_layout)
        # Scan info
        self.layout.addLayout(self.scan_info_layout)
        # Manipulator
        self.add_section('Manipulator', self.manipulator_layout)
        # Analyzer
        self.add_section('Analyzer', self.analyzer_layout)
        # Beamline
        self.add_section('Beamline', self.beamline_layout)
        self.layout.addStretch()

    def load_metadata(self, path) :
        """ Use an :class:`~<arpys.dataloaders.Dataloader>` instance to load 
        ARPES data at *path*.
        """
        selected_loader = self.dl_dropdown.currentText()
        try:
            if selected_loader == 'All':
                data = dl.load_data(path, metadata_only=True)
            else:
                for loader in dl.all_dls :
                    if loader.name == selected_loader :
                        data = loader().load_metadata(path)
                        break
        except Exception as e:
            print('Couldn\'t load data {}.'.format(path))
            raise(e)
            return
        # Add some extra info
        E = data.zscale
        data.e0 = E[0]
        data.e1 = E[-1]
        data.de = E[1] - E[0]
        self.data = data

    def update_details(self, path) :
        """ Put all found metadata into the respective LineEdit fields. """
        self.load_metadata(path)
        if self.data is None :
            print('Could not load data.')
            return
        else :
            # Create a shorthand
            data = self.data
        for key, line_edit in self.line_edits.items() :
            # Try extracting the value
            try :
                value = data[key]
            except AttributeError :
                print('Attribute not found: {}'.format(key))
                continue
            # Skip "None"
            if value is None : continue
            # Try formatting as a number first
            try :
                txt = self.number_fmt.format(value)
            except ValueError :
                txt = '{}'.format(value)
            except TypeError :
                txt = '{}'.format(value.decode('utf-8'))
            # If we got zero, it might be worth displaying in scientific 
            # notation
            if txt == self.number_fmt.format(0.0) :
                txt = self.number_fmt_exp.format(value)
            line_edit.setText(txt)

