#!/usr/bin/python

#import matplotlib.backends.backend_tkagg as tkagg
import numpy as np
import os
import re
import tkinter as tk
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib.pyplot import get_cmap
from tkinter.filedialog import askopenfilename, asksaveasfilename

from arpys.dataloaders import *
import arpys.dataloaders as dl
import arpys.utilities.plotting as kplot
import arpys.postprocessing as pp

# +------------+ #
# | Parameters | # =============================================================
# +------------+ #

# Spacing between the plots in inches (I think)
PLOT_SPACING = 0.02

# Backgroundcolor of plot
BGCOLOR = "black"

# Plot resolution when saving to png
DPI = 150

# Length of sliders
SLIDER_LENGTH = 200

# Number of entries allowed in colormap sliders (starts from 0)
CM_SLIDER_RESOLUTION = 49

# Postprocessing functions dicts
MAP = { 'map' : None }

SUBTRACTORS = { 'matt' : pp.subtract_bg_matt,
                'fermi' : pp.subtract_bg_fermi }

NORMALIZERS = {
    'horizontal' : lambda data : pp.normalize_per_segment(data, dim=0),
    'vertical' : lambda data : pp.normalize_per_segment(data, dim=1),
    'int. EDC' : lambda data : pp.normalize_per_integrated_segment(data, dim=0),
    'int. MDC' : lambda data : pp.normalize_per_integrated_segment(data, dim=1)
}

DERIVATIVES = { 'laplacian' : pp.laplacian,
                'curvature' : pp.curvature }

# Add 'Off' to all postprocessor dicts
PP_DICTS = [ MAP, SUBTRACTORS, NORMALIZERS, DERIVATIVES ]
for D in PP_DICTS :
    D.update({'Off' : None})

# Available matplotlib colormaps
CMAPS = ['Blues', 'viridis', 'Greys', 'bone', 'summer', 'plasma', 'inferno', 'magma', \
         'nipy_spectral', 'BrBG', 'YlGnBu', 'PuBuGn', 'PuRd', 'rainbow_light']
# Size of the plot in inches (includes labels and title etc)
HEIGHT = 6.5
WIDTH = 1.2*HEIGHT
FIGSIZE = (WIDTH, HEIGHT)
VFIGSIZE = FIGSIZE

# Set the font size for plot labels, titles, etc.
FONTSIZE = 10
rcParams.update({'font.size': FONTSIZE})

# The status of the GUI when nothing else is going on
DEFAULT_STATUS = 'Idle'

# Initialize some default data
STARTUP_DATA = np.array([ np.array([ np.array([k*(i+j) +(1-k)*(i-j) for i in 
                                    range(100)]) for j in range(90)]) for k in 
                                   range(2)])

# Style of the cursors and cuts
cursor_kwargs = dict(linewidth=1,
                     color='r')
cut_kwargs = dict(linewidth=1,
                  color='white')
intensity_kwargs = dict(linewidth=1,
                        color='white')
intensity_cursor_kwargs = dict(linewidth=1,
                               color='r')

# Filename endings which indicate that the file is clearly not an ARPES data 
# file (which typically end in .h5, .fits, .pickle)
SKIPPABLE = ['txt', 'zip', 'png', 'pdf']

# +------------+ #
# | GUI Object | # =============================================================
# +------------+ #

class Gui :
    """
    A tkinter GUI to quickly visualize ARPES data, i.e. cuts and maps. Should 
    be built in a modular fashion such that any data reader can be 'plugged in'.

    data      : 3D array; the raw data from the selected file. This is 
                expected to be in the shape (z, x, y) (z may be equal to 1)
    pp_data   : 3D array; the postprocessed data to be displayed.
    cmaps     : list of str; a list of available matplotlib colormap names.
    xscale, yscale, zscale
              : 1D arrays; the x, y and z-axis data.
    cursor_xy : tuple; current location of the cursor in the bottom left plot.
    dpi       : int; resolution at which to save .png`s.
    """
    data = STARTUP_DATA
    pp_data = STARTUP_DATA.copy()
    cmaps = CMAPS
    xscale = None
    yscale = None
    zscale = None
    cursor_xy = None
    dpi = DPI
    main_mesh = None
    vmain_mesh = None
    cut1 = None
    cut2 = None

    def __init__(self, master, filename=None) :
        """ This init function mostly just calls all 'real' initialization 
        functions where the actual work is outsourced to. """
        # Create the main container/window
        frame = tk.Frame(master)

        # Define some elements
        self._set_up_load_button(master)
        self._set_up_pp_selectors(master)
        self._set_up_plots(master)
        self._set_up_colormap_sliders(master)
        self._set_up_z_slider(master)
        self._set_up_integration_range_selector(master)
        self._set_up_status_label(master)

        # Align all elements
        self._align()

        # Load the given file
        if filename :
            self.filepath.set(filename)
            self.load_data()

        # The setup of event handling requires there to be some data already
        self._set_up_event_handling()

    def _align(self) :
        """ Use the grid() layout manager to align the elements of the GUI. 
        At this stage of development the grid has 12 columns of unequal size.
        """
        # The plot takes up the space of PLOT_COLUMNSPAN widgets
        PLOT_COLUMNSPAN = 8
        PLOT_ROWSPAN = 3
        N_PATH_FIELD = PLOT_COLUMNSPAN

        # 'Load file' elements
        LOADROW = 0
        c = 0
        self.browse_button.grid(row=LOADROW, column=c, sticky='ew')
        c += 1
        self.load_button.grid(row=LOADROW, column=c, sticky='ew')
        c += 1
        self.decrement_button.grid(row=LOADROW, column=c, sticky='ew')
        c += 1
        self.increment_button.grid(row=LOADROW, column=c, sticky='ew')
        c += 1
        self.path_field.grid(row=LOADROW, column=c, columnspan=N_PATH_FIELD, 
                             sticky='ew') 

        # Postprocessing selectors
        PPROW = LOADROW + 1
        c = 0
        for lst in self.radiobuttons :
            r = 0
            for btn in lst :
                btn.grid(row=PPROW+r, column=c, sticky='ew')
                r += 1
            c += 1

        # Plot & colormap sliders & selector
        PLOTROW = PPROW + max([len(lst) for lst in self.radiobuttons])
        PLOTCOLUMN = 1
        self.canvas.get_tk_widget().grid(row=PLOTROW, column=PLOTCOLUMN, 
                                         rowspan=PLOT_ROWSPAN, columnspan=PLOT_COLUMNSPAN)
        right_of_plot = PLOT_COLUMNSPAN + PLOTCOLUMN
        self.cm_min_slider.grid(row=PLOTROW, column=right_of_plot + 1)
        self.cm_max_slider.grid(row=PLOTROW, column=right_of_plot + 2)
        self.cmap_dropdown.grid(row=PLOTROW + 1, column=right_of_plot + 1)
        self.invert_cmap_checkbutton.grid(row=PLOTROW + 1, column=right_of_plot + 2)

        # Save png button
        self.save_button.grid(row=PLOTROW + 2, column=right_of_plot + 1)

        # z slider, integration range selector
        self.z_slider.grid(row=PLOTROW, column=0)
        self.integration_range_entry.grid(row=PLOTROW+1, column=0)

        # Put the status label at the very bottom left
        STATUSROW = PLOTROW + PLOT_ROWSPAN + 1
        self.status_label.grid(row=STATUSROW, column=0, columnspan=10,
                               sticky='ew')

    def _set_up_load_button(self, master) :
        """ Add a button which opens a filebrowser to choose the file to load
        and a textbox (Entry widget) where the filepath can be changed.
        """
        # Define the Browse button
        self.browse_button = tk.Button(master, text='Browse',
                                     command=self.browse)

        # and the Load button
        self.load_button = tk.Button(master, text='Load',
                                     command=self.load_data)
        
        # and the entry field which holds the path to the current file
        self.filepath = tk.StringVar()
        self.path_field = tk.Entry(master, textvariable=self.filepath)

        # Also add inc and decrement buttons
        self.increment_button = tk.Button(master, text='>',
                                          command=lambda : self.increment(1)) 
        self.decrement_button = tk.Button(master, text='<',
                                          command=lambda : self.increment(-1)) 

        # Add a 'save' button for creating png s
        self.save_button = tk.Button(master, text='Save png', 
                                     command=self.save_plot)

    def _set_up_pp_selectors(self, master) :
        """ Create radiobuttons for the selction of postprocessing methods. 
        The order of the pp methods in all lists is:
            0) Make map
            1) BG subtraction
            2) Normalization
            3) derivative
        """
        # Create control variables to hold the selections and store them in a 
        # list for programmatic access later on
        self.map = tk.StringVar()
        self.subtractor = tk.StringVar()
        self.normalizer = tk.StringVar()
        self.derivative = tk.StringVar()
        self.selection = [self.map,
                          self.subtractor,
                          self.normalizer,
                          self.derivative]
        # Create sets of radiobuttons and set all to default value 'Off'
        self.radiobuttons = []
        for i, D  in enumerate(PP_DICTS) :
            variable = self.selection[i]
            variable.set('Off')
            self.radiobuttons.append([])
            for key in D :
                rb = tk.Radiobutton(master, text=key, variable=variable,
                                   value=key, command=self.process_data,
                                   indicatoron=0)
                self.radiobuttons[i].append(rb)

    def _set_up_plots(self, master) :
        """ Take care of all the matplotlib stuff for the plot. """
        fig = Figure(figsize=FIGSIZE)
        fig.patch.set_alpha(0)
        ax_cut1 = fig.add_subplot(221)
        ax_cut2 = fig.add_subplot(224)
        ax_map = fig.add_subplot(223)#, sharex=ax_cut1, sharey=ax_cut2)
        ax_energy = fig.add_subplot(222)

        # Virtual figure and ax for creation of png's
        self.vfig = Figure(figsize=VFIGSIZE)
        vax = self.vfig.add_subplot(111)
        self.vcanvas = FigureCanvasTkAgg(self.vfig, master=master)
        
        # Remove padding between min and max of data and plot border
        ax_cut2.set_ymargin(0)

        # Move ticks to the other side
        ax_energy.xaxis.tick_top()
        ax_energy.yaxis.tick_right()

        ax_cut1.xaxis.tick_top()
        ax_cut2.yaxis.tick_right()

        # Get a handle on all axes through the dictionary self.axes
        self.axes = {'cut1': ax_cut1,
                     'cut2': ax_cut2,
                     'map': ax_map,
                     'energy': ax_energy,
                     'vax': vax}

        # Set bg color
        for ax in self.axes.values() :
            ax.set_facecolor(BGCOLOR)

        # Remove spacing between plots
        fig.subplots_adjust(wspace=PLOT_SPACING, hspace=PLOT_SPACING)

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.show()

    def _set_up_colormap_sliders(self, master) :
        """ Add the colormap adjust sliders, set their starting position and add 
        its binding such that it only triggers upon release. Also, couple 
        them to the variables vmin/max_index.
        Then, also create a dropdown with all available cmaps and a checkbox 
        to invert the cmap.
        """
        self.vmin_index = tk.IntVar()
        self.vmax_index = tk.IntVar()
        cm_slider_kwargs = { 'showvalue' : 0,
                             'to' : CM_SLIDER_RESOLUTION, 
                             'length':SLIDER_LENGTH }
        self.cm_min_slider = tk.Scale(master, variable=self.vmin_index,
                                      label='Min', **cm_slider_kwargs)
        self.cm_min_slider.set(CM_SLIDER_RESOLUTION)

        self.cm_max_slider = tk.Scale(master, variable=self.vmax_index,
                                      label='Max', **cm_slider_kwargs)
        self.cm_max_slider.set(0)

        # StringVar to keep track of the cmap and whether it's inverted
        self.cmap = tk.StringVar()
        self.invert_cmap = tk.StringVar()
        # Default to the first cmap
        self.cmap.set(self.cmaps[0])
        self.invert_cmap.set('')

        # Register callbacks for colormap-range change
        for var in [self.vmin_index, self.vmax_index, self.cmap, 
                    self.invert_cmap] :
            var.trace('w', self.redraw_mesh)

        # Create the dropdown menu, populated with all strings in self.cmaps
        self.cmap_dropdown = tk.OptionMenu(master, self.cmap, *self.cmaps)

        # And a button to invert
        self.invert_cmap_checkbutton = tk.Checkbutton(master, text='Invert',
                                                      variable=self.invert_cmap,
                                                      onvalue='_r',
                                                      offvalue='')

    def _set_up_z_slider(self, master) :
        """ Create a Slider which allows to select the z value of the data.
        This value is stored in the DoubleVar self.z """
        self.z = tk.IntVar()
        self.z.set(0)
        self.zmax = tk.IntVar()
        self.zmax.set(1)
        self.z_slider = tk.Scale(master, variable=self.z, label='z', 
                                 to=self.zmax.get(), showvalue=1, 
                                 length=SLIDER_LENGTH) 
        self.z_slider.bind('<ButtonRelease-1>', self.process_data)

    def _set_up_integration_range_selector(self, master) :
        """ Create widgets that will allow setting the integration range when 
        creating a map. """
        self.integrate = tk.IntVar()
        self.integration_range_entry = tk.Entry(master, width=3,
                                           textvariable=self.integrate)

    def _set_up_status_label(self, master) :
        """ Create a label which can hold informative text about the current
        state of the GUI or success/failure of certain operations. This text 
        is held in the StringVar self.status.
        """
        self.status = tk.StringVar()
        # Initialize the variable with the default status
        self.update_status()
        self.status_label = tk.Label(textvariable=self.status, justify=tk.LEFT,
                                    anchor='w')

    def redraw_mesh(self, *args, **kwargs) :
        """ Efficiently redraw the pcolormesh without having to redraw the 
        axes, ticks, labels, etc. """
        # Get the new colormap parameters and apply them to the pcolormesh
        cmap = self.get_cmap()
        vmin, vmax = self.vminmax(self.pp_data)
        mesh = self.main_mesh
        mesh.set_clim(vmin=vmin, vmax=vmax)
        mesh.set_cmap(cmap)

        # Redraw the mesh
        ax = self.axes['map']
        ax.draw_artist(mesh)

        # Cursors need to be redrawn - the blitting happens there
        self.redraw_cursors()

        # Also redraw the cuts if they are pcolormeshes
        if self.map.get() != 'Off' :
            self.redraw_cuts()

    def redraw_cuts(self, *args, **kwargs) :
        """ Efficiently redraw the cuts (meshes or lines, depending on state 
        of `self.map`) without having to redraw the axes, ticks, labels, etc. 
        """
        axes = [self.axes['cut1'], self.axes['cut2']]
        data = [self.cut1, self.cut2]
        artists = [self.cut1_plot, self.cut2_plot] 
        cmap = self.get_cmap()
        for i,ax in enumerate(axes) :
            artist = artists[i]
            if self.map.get() != 'Off' :
                vmin, vmax = self.vminmax(data[i])
                artist.set_clim(vmin=vmin, vmax=vmax)
                artist.set_cmap(cmap)
            ax.draw_artist(artist)
            self.canvas.blit(ax.bbox)

    def redraw_cursors(self, *args, **kwargs) :
        """ Efficiently redraw the cursors in the bottom left plot without 
        having to redraw the axes, ticks, labels, etc. """
        ax = self.axes['map']
        #self.canvas.restore_region(self.bg_mesh)
        ax.draw_artist(self.xcursor)
        ax.draw_artist(self.ycursor)
        self.canvas.blit(ax.bbox)

    def update_z_slider(self, state) :
        """ Change the relief of the z slider to indicate that it is 
        inactive/active. Also update the z slider range"""
        if state == 'active' :
            config = dict(sliderrelief='raised', showvalue=1)
        else :
            config = dict(sliderrelief='flat', showvalue=0)
        self.zmax.set( len(self.data) - 1) 
        config.update(dict(to=self.zmax.get()))
        self.z_slider.config(**config)

    def get_filename(self) :
        """ Return the filename (without path) of the currently selected 
        file. """
        return self.filepath.get().split('/')[-1]

    def increment(self, plusminus) :
        # Get the path and the name of the current file
        filepath = self.filepath.get()
        split = filepath.split('/')
        # If just a filename is given, assume the current directory
        if filepath[0] is not '/' :
            path = './'
        else :
            #path = '/' + '/'.join(split[:-1]) + '/'
            path = '/'.join(split[:-1]) + '/'

        old_filename = split[-1]
        
        # Get a list of the files in the current directory
        dir_content = sorted( os.listdir(path) )
        dir_size = len(dir_content)

        # Get the index of the current file in the directory 
        index = dir_content.index(old_filename)

        # Raise/lower the index until a not-obiously skippable entry is found
        while True :
            index += plusminus
            # Cycle through the list
            index %= dir_size
            new_filename = dir_content[index]
            suffix = new_filename.split('.')[-1]
            if suffix not in SKIPPABLE :
                self.filepath.set(path+new_filename)
                break

    def update_status(self, status=DEFAULT_STATUS) :
        """ Update the status StringVar with the current time and the given
        status argument. 
        """
        now = datetime.now().strftime('%H:%M:%S')
        new_status = '[{}] {}'.format(now, status)
        self.status.set(new_status)

    def browse(self) :
        """ Open a filebrowser dialog and put the selected file into  
        self.path_field. 
        """
        # If the file entry field already contains a path to a file use it
        # as the default file for the browser
        old_filepath = self.filepath.get() 
        if old_filepath :
            default_file = old_filepath
            initialdir = None
        else :
            default_file = None
            initialdir = '/home/kevin/qmap/'

        # Open a browser dialog
        new_filepath = askopenfilename(initialfile=default_file,
                                       initialdir=initialdir)

        # Update the path only if a selection was made
        if new_filepath != "" :
            self.filepath.set(new_filepath)

    def load_data(self) :
        """ Load data from the file currently selected by self.filepath. And 
        reset and prepare several things such that the GUI is able to handle 
        the new data properly. 
        """
        # Show the user that something is happening
        self.update_status('Loading data...')

        # Try to load the data with the given dataloader
        try :
            ns = dl.load_data(self.filepath.get())
        except Exception as e :
            print(e)
            self.update_status('Failed to load data.')
            # Leave the function
            return 1

        # Extract the fields from the namespace
        self.data = ns.data
        self.xscale = ns.xscale
        self.yscale = ns.yscale
        try :
            self.zscale = ns.zscale
        except KeyError :
            # Set zscale to None
            self.zscale = None
            pass

        # Notify user of success
        self.update_status('Loaded data: {}.'.format(self.get_filename())) 

        # Update the max z value/toggle z slider (should better be handled in 
        # background by tkinter)
        if self.data.shape[0] == 1 :
            self.update_z_slider('disabled')
        else :
            self.update_z_slider('active')

        # Initiate new cursors
        self.cursor_xy = None

        # The postprocessing also creates copy of the raw data and replots
        self.process_data()

    def process_data(self, event=None) :
        """ Apply all the selected postprocessing routines in the following 
        order:
            0) Make map
            1) bg subtraction
            2) normalization
            3) derivative
        """
        # Retain a copy of the raw data
        self.pp_data = self.data.copy()
        z = self.z.get()

        # Make a map if necessary
        if self.map.get() != 'Off' :
            integrate = self.integrate.get()
            self.pp_data = pp.make_slice(self.pp_data, d=0, i=z, integrate=integrate)
            #self.pp_data = pp.make_map(self.pp_data, z, integrate=integrate)
            
            # Need to reshape
            shape = self.pp_data.shape
            self.pp_data = self.pp_data.reshape(1, shape[0], shape[1])


        # Apply all pp, unless they are set to 'Off' (None)
        for i, D in enumerate(PP_DICTS) :
            pp_operation = D[self.selection[i].get()]
            if pp_operation :
                self.pp_data = pp_operation(self.pp_data)
                #self.pp_data[z,:,:] = pp_operation(self.pp_data[z,:,:])

        # Replot
        self.plot_data()

    def plot_intensity(self) :
        """ Plot the binding energy distribution in the top right if we have 
        a map. """
        # Clear the current distribution
        ax = self.axes['energy']
        ax.clear()

        # Write the value of the energy in the upper right plot
        z = self.z.get()
        if self.zscale is not None :
            z_val = self.zscale[z]
        else :
            z_val = z
        ax.text(0.1, 0.05, z_val, color='red', transform=ax.transAxes)

        # Nothing else to do if we don't have a map
        if self.map.get() == 'Off' : 
            return

        # Get the energies and number of energies
        if self.zscale is not None :
            energies = self.zscale
        else :
            energies = np.arange(len(self.data))
        N_e = len(energies)
        
        # Get the intensities
        intensities = []
        for i in range(N_e) :
            this_slice = self.data[i,:,:]
            intensity = sum( sum(this_slice) )
            intensities.append(intensity)

        # Plot energy distribution
        ax.plot(energies, intensities, **intensity_kwargs)

        # Plot a cursor indicating the current value of z
        y0 = min(intensities)
        y1 = max(intensities)
        ylim = [y0, y1]
        ax.plot(2*[z_val], ylim, **intensity_cursor_kwargs)
        ax.set_ylim(ylim)

    def calculate_cuts(self) :
        """ """
        if self.map.get() != 'Off' :
            # Create a copy of the original map (3D) data
            data = self.data.copy()
            # Slice and dice it
            self.cut1 = pp.make_slice(data, d=1, i=self.yind, integrate=1)
            self.cut2 = pp.make_slice(data, d=2, i=self.xind, integrate=1)
        else :
            z = self.z.get()
            self.cut1 = self.pp_data[z, self.yind, :]
            self.cut2 = self.pp_data[z, :, self.xind]

    def plot_cuts(self) :
        """ Plot cuts of whatever is in the bottom left ('map') axis along 
        the current positions of the cursors. 
        """
        self.calculate_cuts()

        # Clear the current cuts
        for ax in ['cut1', 'cut2'] :
            self.axes[ax].clear()

        # Get the right xscale/yscale information
        xscale, yscale = self.get_xy_scales()

        if self.map.get() != 'Off' :
            kwargs = dict(cmap=self.get_cmap())

            # Ensure zscale is defined
            if self.zscale is None :
                zscale = np.arange(self.cut1.shape[0])
            else :
                zscale = self.zscale

            # Plot x cut in upper left
            vmin, vmax = self.vminmax(self.cut1)
            kwargs.update(dict(vmin=vmin, vmax=vmax))
            self.cut1_plot = self.axes['cut1'].pcolormesh(xscale, zscale, 
                                                      self.cut1, **kwargs)
            # Plot y cut in lower right (rotated by 90 degrees)
            vmin, vmax = self.vminmax(self.cut2)
            kwargs.update(dict(vmin=vmin, vmax=vmax))
            self.cut2_plot = self.axes['cut2'].pcolormesh(zscale, yscale, 
                                                      self.cut2.T, **kwargs)
        else :
            # Plot the x cut in the upper left
            self.cut1_plot = self.axes['cut1'].plot(xscale, self.cut1, 
                                                    **cut_kwargs)[0]
            # Plot the y cut in the lower right
            self.cut2_plot = self.axes['cut2'].plot(self.cut2, yscale, 
                                                    **cut_kwargs)[0]

            # Make sure the cut goes over the full range of the plot
            #self.axes['cut2'].set_ymargin(0) # For some reason this doesn't work
            ymin = min(yscale)
            ymax = max(yscale)
            self.axes['cut2'].set_ylim([ymin, ymax])

    def get_plot_args_and_kwargs(self) :
        """ Prepare args and kwargs for plotting, depending on the 
        circumstances. """
        # Add x and y scales to args if available
        args = []
        if self.xscale is not None and self.yscale is not None :
            args.append(self.xscale)
            args.append(self.yscale)

        # Use z=0 in case of a map (as pp_data is of length 1 along this 
        # dimension as a result of pp_make_cut())
        if self.map.get() != 'Off' :
            z = 0
        else :
            z = self.z.get()

        args.append(self.pp_data[z,:,:])

        vmin, vmax = self.vminmax(self.pp_data)
        kwargs = dict(cmap=self.get_cmap(), vmin=vmin, 
                      vmax=vmax)
        return args, kwargs

    def plot_data(self, event=None, *args, **kwargs) :
        """ Update the colormap range and (re)plot the data. """
        # Remove old plots
        for ax in self.axes.values() :
            ax.clear()

        args, kwargs = self.get_plot_args_and_kwargs()
        # Do the actual plotting with just defined args and kwargs
        ax = self.axes['map']
        self.main_mesh = ax.pcolormesh(*args, **kwargs)

        self.bg_mesh = self.canvas.copy_from_bbox(ax.bbox)

        # Update the cursors (such that they are above the pcolormesh) and cuts
        self.plot_cursors()
        self.plot_cuts()
        self.plot_intensity()
        self.canvas.draw()

    def get_cmap(self) :
        """ Build the name of the colormap by combining the value stored in 
        `self.cmap` (the basename of the colormap) and `self.invert_cmap` 
        (either empty string or '_r', which is the suffix for inverted cmaps 
        in matplotlib) """
        # Build the name of the cmap
        cmap_name = self.cmap.get() + self.invert_cmap.get()
        # Make sure this name exists in the list of cmaps. Otherwise reset to 
        # default cmap
        try :
            cmap = get_cmap(cmap_name)
        except ValueError :
            # Notify user
            message = \
            'Colormap {} not found. Using default instead.'.format(cmap_name)
            self.update_status(message)
            # Set default
            cmap = get_cmap(self.cmaps[0])

        return cmap

    def vminmax(self, data) :
        """ Helper function that returns appropriate values for vmin and vmax
        for a given set of data. """
        # Note: vmin_index goes from 100 to 0 and vice versa for vmax_index.
        # This is to turn the sliders upside down.
        # Crude method to avoid unreasonable colormap settings
        if self.vmin_index.get() < self.vmax_index.get() :
            self.vmin_index.set(CM_SLIDER_RESOLUTION)

        # Split the data value range into equal parts
        #drange = np.linspace(self.pp_data.min(), data.max(), 
        #                     CM_SLIDER_RESOLUTION + 1)
        drange = np.linspace(data.min(), data.max(), 
                             CM_SLIDER_RESOLUTION + 1)

        # Get the appropriate vmin and vmax values from the data
        vmin = drange[CM_SLIDER_RESOLUTION - self.vmin_index.get()]
        vmax = drange[CM_SLIDER_RESOLUTION - self.vmax_index.get()]

        return vmin, vmax

    def get_xy_minmax(self) :
        """ Return the min and max for the x and y axes, depending on whether 
        xscale and yscale are defined. 
        """
        xscale, yscale =  self.get_xy_scales()

        xmin = min(xscale)
        xmax = max(xscale)
        ymin = min(yscale)
        ymax = max(yscale)

        return xmin, xmax, ymin, ymax

    def get_xy_scales(self) :
        """ Depending on whether we have actual data scales (self.xscale and 
        self.yscale are defined) or not, return arrays which represent data 
        coordinates. 
        """
        if self.xscale is None or self.yscale is None :
            shape = self.data.shape
            yscale = np.arange(0, shape[1], 1)
            xscale = np.arange(0, shape[2], 1)
        else :
            xscale = self.xscale
            yscale = self.yscale

        return xscale, yscale

    def snap_to(self, x, y) :
        """ Return the closest data value to the given values of x and y. """
        xscale, yscale = self.get_xy_scales()

        # Find the index where element x/y would have to be inserted in the 
        # sorted array.
        self.xind = np.searchsorted(xscale, x)
        self.yind = np.searchsorted(yscale, y)

        # Find out whether the lower or upper 'neighbour' is closest
        x_lower = xscale[self.xind-1]
        y_lower = yscale[self.yind-1]
        # NOTE In principle, these IndexErrors shouldn't occur. Try-except 
        # only helps when debugging.
        try :
            x_upper = xscale[self.xind]
        except IndexError :
            x_upper = max(xscale)
        try :
            y_upper = yscale[self.yind]
        except IndexError :
            y_upper = max(yscale)

        dx_upper = x_upper - x
        dx_lower = x - x_lower
        dy_upper = y_upper - y
        dy_lower = y - y_lower

        # Assign the exact data value and update self.xind/yind if necessary
        if dx_upper < dx_lower :
            x_snap = x_upper
        else :
            x_snap = x_lower
            self.xind -= 1
            
        if dy_upper < dy_lower :
            y_snap = y_upper
        else :
            y_snap = y_lower
            self.yind -= 1

        return x_snap, y_snap

    def plot_cursors(self) :
        """ Plot the cursors in the bottom left axis. """
        # Delete current cursors (NOTE: this is dangerous if there are any 
        # other lines in the plot)
        ax = self.axes['map']
        ax.lines = []

        # Retrieve information about current data range
        xmin, xmax, ymin, ymax = self.get_xy_minmax()
       
        xlimits = [xmin, xmax]
        ylimits = [ymin, ymax]

        # Initiate cursors in the center of graph if necessary
        if self.cursor_xy is None :
            x = 0.5 * (xmax + xmin)
            y = 0.5 * (ymax + ymin)

            # Keep a handle on cursor positions
            self.cursor_xy = (x, y)
        else : 
            x, y = self.cursor_xy

        # Make the cursor snap to actual data points
        x, y = self.snap_to(x, y)

        # Plot cursors and keep handles on them (need the [0] because plot() 
        # returns a list of Line objects)
        self.xcursor = ax.plot([x, x], ylimits, zorder=3, **cursor_kwargs)[0]
        self.ycursor = ax.plot(xlimits, [y, y], zorder=3, **cursor_kwargs)[0]

    def _set_up_event_handling(self) :
        """ Define what happens when user clicks in the plot (move cursors to 
        clicked position) or presses an arrow key (move cursors in specified 
        direction). 
        """
        def on_click(event):
            event_ax = event.inaxes
            if event_ax == self.axes['map'] :
                self.cursor_xy = (event.xdata, event.ydata)
                self.plot_cursors()
                # Also update the cuts
                self.plot_cuts()
                self.canvas.draw()
            elif event_ax == self.axes['energy'] and \
                 self.map.get() != 'Off' :
                if self.zscale is not None :
                    z = np.where(self.zscale > event.xdata)[0][0]
                else :
                    z = int(event.xdata)
                self.z.set(z)
                # Since z changed we need to apply the whole data processing
                # and replot
                self.process_data()
                
        def on_press(event):
            # Get the name of the pressed key and info on the current cursors
            key = event.key
            x, y = self.cursor_xy
            xmin, xmax, ymin, ymax = self.get_xy_minmax()

            # Stop if no arrow key was pressed
            if key not in ['up', 'down', 'left', 'right'] : return

            # Move the cursor by one unit in data points
            xscale, yscale = self.get_xy_scales()
            dx = xscale[1] - xscale[0]
            dy = yscale[1] - yscale[0]

            # In-/decrement cursor positions depending on what button was 
            # pressed and only if we don't leave the axis
            if key == 'up' and y+dy <= ymax :
                y += dy
            elif key == 'down' and y-dy >= ymin :
                y -= dy
            elif key == 'right' and x+dx <= xmax :
                x += dx
            elif key == 'left' and x-dx >= xmin:
                x -= dx

            # Update the cursor position and redraw it
            self.cursor_xy = (x, y)
            #self.plot_cursors()
            self.redraw_cursors()
            # Now the cuts have to be redrawn as well
            self.calculate_cuts()
            #self.plot_cuts()
            self.redraw_cuts()

        cid = self.canvas.mpl_connect('button_press_event', on_click)
        pid = self.canvas.mpl_connect('key_press_event', on_press)

        # Inititate the cursors
        self.plot_cursors()

    def save_plot(self) :
        """ Save a png image of the currently plotted data (only what is in 
        bottom left) """
        # Plot the same thing into a virtual figure such that a png can be 
        # created
        args, kwargs = self.get_plot_args_and_kwargs()
        self.vmain_mesh = self.axes['vax'].pcolormesh(*args, **kwargs)

        # Open a filebrowser where user can select a place to store the result
        filename = asksaveasfilename(filetypes=[('PNG', '*.png')])
        if filename :
            self.vfig.savefig(filename, transparent=True, dpi=self.dpi)
            self.update_status('Saved file {}.'.format(filename))
        else :
            self.update_status('Saving file aborted.')

# +------+ #
# | Main | # ===================================================================
# +------+ #

def start_gui(filename=None) :
    """ Initialize the Tk object, give it a title, attach the Gui (App) to it 
    and start the mainloop. 
    """
    root = tk.Tk()
    root.title('ARPES visualizer')
    gui = Gui(root, filename=filename)
    gui.plot_data()
    root.mainloop()

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', default=None)
    args = parser.parse_args()
    
    start_gui(args.filename)

