#!/usr/bin/python
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#import matplotlib.backends.backend_tkagg as tkagg
from matplotlib import rcParams
import numpy as np
import re
import tkinter as tk
from tkinter.filedialog import askopenfilename

from dataloaders import *
import postprocessing as pp
import kustom.plotting as kplot

# +------------+ #
# | Parameters | # =============================================================
# +------------+ #

# Spacing between the plots in inches (I think)
PLOT_SPACING = 0.02

# Backgroundcolor of plot
BGCOLOR = "black"

# Length of sliders
SLIDER_LENGTH = 200

# Number of entries allowed in colormap sliders (starts from 0)
CM_SLIDER_RESOLUTION = 99

# Data loader objects
DATALOADERS = { 'PSI' : Dataloader_PSI(),
                'ALS' : Dataloader_ALS(),
                'Pickle': Dataloader_Pickle() }

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
CMAPS = ['viridis', 'Greys', 'Greys_r', 'bone', 'summer', 'plasma', 'inferno', 'magma', \
         'nipy_spectral', 'BrBG', 'Wistia', 'rainbow_light']
# Size of the plot in inches (includes labels and title etc)
HEIGHT = 6.5
WIDTH = 1.2*HEIGHT
FIGSIZE = (WIDTH, HEIGHT)

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
cursor_kwargs = {'linewidth': 1,
                 'color': 'r'}
cut_kwargs = {'linewidth': 1,
              'color': 'white'}

# +------------+ #
# | GUI Object | # =============================================================
# +------------+ #

class Gui :
    """
    A tkinter GUI to quickly visualize ARPES data, i.e. cuts and maps. Should 
    be built in a modular fashion such that any data reader can be 'plugged in'.
    """
    data = STARTUP_DATA
    pp_data = STARTUP_DATA.copy()
    dataloaders = DATALOADERS
    cmaps = CMAPS
    xscale = None
    yscale = None
    cursor_xy = None

    def __init__(self, master, filename=None, dataloader='PSI') :
        """ This init function mostly just calls all 'real' initialization 
        functions where the actual work is outsourced to. """
        # Create the main container/window
        frame = tk.Frame(master)

        # Set the initial dataloader
        self.dataloader = dataloader

        # Define some elements
        self._set_up_load_button(master)
        self._set_up_pp_selectors(master)
        self._set_up_dataloader_selector(master)
        self._set_up_plots(master)
        self._set_up_colormap_sliders(master)
        self._set_up_z_slider(master)
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
        N_PATH_FIELD = PLOT_COLUMNSPAN - 1

        # 'Load file' elements
        LOADROW = 0
        self.dataloader_dropdown.grid(row=LOADROW, column=0, sticky='ew')
        self.browse_button.grid(row=LOADROW, column=1, sticky='ew')
        self.load_button.grid(row=LOADROW, column=2, sticky='ew')
        self.decrement_button.grid(row=LOADROW, column=3, sticky='ew')
        self.increment_button.grid(row=LOADROW, column=4, sticky='ew')
        self.path_field.grid(row=LOADROW, column=5, columnspan=N_PATH_FIELD, 
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

        # z slider
        self.z_slider.grid(row=PLOTROW, column=0)

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

        # Finally, also add inc and decrement buttons
        self.increment_button = tk.Button(master, text='>',
                                          command=lambda : self.increment('+')) 
        self.decrement_button = tk.Button(master, text='<',
                                          command=lambda : self.increment('-')) 

    def _set_up_pp_selectors(self, master) :
        """ Create radiobuttons for the selction of postprocessing methods. 
        The order of the pp methods in all lists is:
            0) Make map
            1) BG subtraction
            2) Normalization
            3) derivative
        """
        # Create IntVars to hold the selections and store them in a list for
        # programmatic access later on
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

    def _set_up_dataloader_selector(self, master) :
        """ Create a dropdown containing all possible dataloaders. """
        # Get the list of possible options for the dropdown
        options = list(self.dataloaders.keys())

        # Initialize the associated StringVar
        self.dataloader_selection = tk.StringVar()
        self.dataloader_selection.set(self.dataloader)

        # Create the dropdown
        self.dataloader_dropdown = tk.OptionMenu(master, self.dataloader_selection,
                                                 *options)
        
    def _set_up_plots(self, master) :
        """ Take care of all the matplotlib stuff for the plot. """
        fig = Figure(figsize=FIGSIZE)
        fig.patch.set_alpha(0)
        ax_cut1 = fig.add_subplot(221)
        ax_cut2 = fig.add_subplot(224)
        ax_map = fig.add_subplot(223)#, sharex=ax_cut1, sharey=ax_cut2)
        ax_energy = fig.add_subplot(222)
        
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
                     'energy': ax_energy}

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
        Then, also create a dropdown with all available cmaps.
        """
        self.vmin_index = tk.IntVar()
        self.vmax_index = tk.IntVar()
        cm_slider_kwargs = { 'showvalue' : 0,
                             'to' : CM_SLIDER_RESOLUTION, 
                             'length':SLIDER_LENGTH }
        self.cm_min_slider = tk.Scale(master, variable=self.vmin_index,
                                      label='Min', **cm_slider_kwargs)
        self.cm_min_slider.set(CM_SLIDER_RESOLUTION)
        self.cm_min_slider.bind('<ButtonRelease-1>', self.plot_data)

        self.cm_max_slider = tk.Scale(master, variable=self.vmax_index,
                                      label='Max', **cm_slider_kwargs)
        self.cm_max_slider.set(0)
        self.cm_max_slider.bind('<ButtonRelease-1>', self.plot_data)

        # StringVar to keep track of the cmap
        self.cmap = tk.StringVar()
        # Default to the first cmap
        self.cmap.set(self.cmaps[0])
        # Replot whenever the variable changes
        self.cmap.trace("w", self.plot_data)

        # Create the dropdown menu, populated with all strings in self.cmaps
        self.cmap_dropdown = tk.OptionMenu(master, self.cmap, *self.cmaps)

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

    def get_filename(self) :
        """ Return the filename (without path) of the currently selected 
        file. """
        return self.filepath.get().split('/')[-1]

    def increment(self, plusminus) :
        """ Raise or lower the filename by one. This assumes a filename of 
        the form <FILENAME>XXXX<SUFFIX> where XXXX is the four digit ID 
        number of the file which is being in- or decremented.
        """
        old_filename = self.filepath.get()

        # Search from the back of the string (thus reversed) for a sequence 
        # of four digits
        found = re.search('\d\d\d\d', old_filename[::-1])

        if not found : 
            message = 'Could not find sequence of four digits in filename.'
            self.update_status(message)
            return

        # Because the string was inverted, start end stop indexes are swapped 
        # and negative
        start = -found.end()
        end = -found.start()

        # Get the old string and value and raise/lower it
        old_string = old_filename[start:end]
        n = int(old_string)

        if plusminus == '+' :
            n += 1
        elif plusminus == '-' :
            # Don't lower when we are already at 0000
            if n > 0 :
                n -= 1
            elif n <= 0 :
                return

        # Convert back to string of four digits
        new_string = '{:>04.0f}'.format(n)

        # Split the filepath and replace the correct parts
        prefix = old_filename[:start]
        tail = old_filename[start:]
        new_tail = tail.replace(old_string, new_string)

        self.filepath.set(prefix + new_tail)

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
        else :
            default_file = None

        # Open a browser dialog
        new_filepath = askopenfilename(initialfile=default_file)

        # Update the path only if a selection was made
        if new_filepath != "" :
            self.filepath.set(new_filepath)

    def load_data(self) :
        """ Load data from the file currently selected by self.filepath. And 
        reset and prepare several things such that the GUI is able to handle 
        the new data properly. 
        """
        # Get the chose dataloader from the selection
        dataloader = self.dataloader_selection.get()

        # Show the user that something is happening
        self.update_status('Loading {} data...'.format(dataloader))

        # Try to load the data with the given dataloader
        try :
            datadict = \
            self.dataloaders[dataloader].load_data(self.filepath.get())
        except Exception as e :
            print(e)
            self.update_status('Failed to load {} data.'.format(dataloader))
            # Leave the function
            return 1

        # Extract the fields from the datadict
        self.data = datadict['data']
        self.xscale = datadict['xscale']
        self.yscale = datadict['yscale']

        # Notify user of success
        self.update_status('Loaded {} data: {}.'.format(dataloader, 
                                                        self.get_filename()))

        # Update the max z value
        self.zmax.set( len(self.data) - 1) 
        self.z_slider.config(to=self.zmax.get())

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
            integrate = 5
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

    def plot_cuts(self) :
        """ Plot cuts of whatever is in the bottom left ('map') axis along 
        the current positions of the cursors. 
        """
        # Clear the current cuts
        for ax in ['cut1', 'cut2'] :
            self.axes[ax].clear()

        # Get the right xscale/yscale information
        xscale, yscale = self.get_xy_scales()

        # Set z depending on whether we deal with a map or not
        if self.map.get() != 'Off' :
            # Create a copy of the original map (3D) data
            data = self.data.copy()
            # Slice and dice it
            xcut = pp.make_slice(data, d=1, i=self.yind, integrate=1)
            ycut = pp.make_slice(data, d=2, i=self.xind, integrate=1)

            # Plot x cut in upper left
            self.axes['cut1'].pcolormesh(xcut)
            # Plot y cut in lower right
            self.axes['cut2'].pcolormesh(ycut)
        else :
            z = self.z.get()
            xcut = self.pp_data[z, self.yind, :]
            ycut = self.pp_data[z, :, self.xind]

            # Plot the x cut in the upper left
            self.axes['cut1'].plot(xscale, xcut, **cut_kwargs)

            # Plot the y cut in the lower right
            self.axes['cut2'].plot(ycut, yscale, **cut_kwargs)

            # Make sure the cut goes over the full range of the plot
            #self.axes['cut2'].set_ymargin(0) # For some reason this doesn't work
            ymin = min(yscale)
            ymax = max(yscale)
            self.axes['cut2'].set_ylim([ymin, ymax])

    def plot_data(self, event=None, *args, **kwargs) :
        """ Update the colormap range and (re)plot the data. """
        # Note: vmin_index goes from 100 to 0 and vice versa for vmax_index.
        # This is to turn the sliders upside down.
        # Crude method to avoid unreasonable colormap settings
        if self.vmin_index.get() < self.vmax_index.get() :
            self.vmin_index.set(CM_SLIDER_RESOLUTION)

        # Split the data value range into equal parts
        drange = np.linspace(self.pp_data.min(), self.pp_data.max(), 
                             CM_SLIDER_RESOLUTION + 1)

        # Get the appropriate vmin and vmax values from the data
        vmin = drange[CM_SLIDER_RESOLUTION - self.vmin_index.get()]
        vmax = drange[CM_SLIDER_RESOLUTION - self.vmax_index.get()]

        # Remove old plots
        for ax in self.axes.values() :
            ax.clear()

        # Prepare args and kwargs for plotting, depending on the circumstances
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
        kwargs = dict(cmap=self.cmap.get(), vmin=vmin, vmax=vmax)

        # Do the actual plotting with just defined args and kwargs
        self.axes['map'].pcolormesh(*args, **kwargs)

        # Update the cursors (such that they are above the pcolormesh) and cuts
        self.plot_cursors()
        self.plot_cuts()
        self.canvas.draw()

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
        # NOTE In principle, these IndexErrors shouldn't occur. Try catch 
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
            # Stop if we're not in the right plot
            if event.inaxes != self.axes['map'] :
                return

            self.cursor_xy = (event.xdata, event.ydata)
            self.plot_cursors()
            # Also update the cuts
            self.plot_cuts()
            self.canvas.draw()

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
            self.plot_cursors()
            # Now the cuts have to be redrawn as well
            self.plot_cuts()
            self.canvas.draw()

        cid = self.canvas.mpl_connect('button_press_event', on_click)
        pid = self.canvas.mpl_connect('key_press_event', on_press)

        # Inititate the cursors
        self.plot_cursors()

def start_gui(filename=None, dataloader='PSI') :
    """ Initialize the Tk object, give it a title, attach the Gui (App) to it 
    and start the mainloop. 
    """
    root = tk.Tk()
    root.title('Data visualizer')
    gui = Gui(root, filename=filename, dataloader=dataloader)
    gui.plot_data()
    root.mainloop()


# +------+ #
# | Main | # ===================================================================
# +------+ #

if __name__ == '__main__' :
    start_gui()
