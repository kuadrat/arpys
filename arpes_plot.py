#!/usr/bin/python

"""
arpes_plot.py
A tool to plot ARPES data from the command line.

TODO:
    - open new files from within APC
    - functionality for maps/3D data
      > maps need different normalization and bg subtraction routines
    - fit functions to EDCs
    - shirley bg, and other norms and bgs (`above_fermi`)
    - Fermi level detection
    - find a way for iPython to be non-blocking
    - re-open figure with `do_plot` if window has been closed >or< close app 
      on window close.
    - record sequence of commands to replay (ideally on a new file)
"""

import argparse
import cmd2 as cmd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, rcParams
from screeninfo import get_monitors

import kustom.arpys.dataloaders as dl
import kustom.arpys.postprocessing as pp
# Need to import kustom.plotting to have projection='cursor' available
from kustom import plotting as kplot

# +--------------------------+ #
# | Parameters and constants | # ===============================================
# +--------------------------+ #

# Shortands for final variables (Update the respective docstrings if you 
# change these)
INT_EDC_NORM = 'integrated_edc'
FERMI_NORM = 'above_fermi'
NO_NORM = 'off'

FERMI_BG = 'above_fermi'
PROFILE_BG = 'profile'
NO_BG = 'off'

DEFAULT_CMAP = 'Blues'

# Command category names
VISUAL = 'Plot'
PROCESSING = 'Data Processing'
ANALYSIS = 'Data analysis'

# List of registered matplotlib colormaps
CMAPS = [name for name in cm.datad]

# +-----+ #
# | CLI | # ====================================================================
# +-----+ #

class APCmd(cmd.Cmd) :
    """ (A)RPES (P)lot (C)o(m)man(d) line interpreter. """
    #_Initial_parameters________________________________________________________
    # Porcessing
    normalization = NO_NORM
    bg = NO_BG
    cmap = DEFAULT_CMAP
    vmax = 1
    z = 0
    integrate = 0
    crop_x_above = None
    crop_x_below = None
    crop_y_above = None
    crop_y_below = None
    lattice_constant = 1
    convert_ang2k = False
    convert_y_ang2k = False
    kx_shift = 0
    ky_shift = 0
    # Visual
    grid_on = False
    xlabel = ''
    ylabel = ''
    title = ''

    #_Cmd2_configuration________________________________________________________
    debug = True
    locals_in_py = True
    allow_cli_args = False

    def __init__(self, ax, D, filename, *args, **kwargs) :
        """
        ========  ==============================================================
        ax        :class: `Axes <matplotlib.axes.Axes>` object; the ax on 
                  which the plot lives.
        D         :class: `Namespace <argparse.Namespace>` object; the 
                  Namepsace with the data and metadata of an ARPES file as it 
                  would be returned by a :module: `dataloader 
                  <kustom.arpys.dataloaders>`.
        filename  str; name of the file to be opened.
        ========  ==============================================================
        """
        super().__init__(*args, use_ipython=True, **kwargs)

        # Define the prompt and welcome messages
        self.prompt = self.colorize('[APC] ', 'cyan')
        bold = [self.colorize(i, 'bold') for i in ['APC', 'A', 'P', 'C']]
        self.intro = ('Welcome to {}, the {}RPES {}lot {}ommand line '
                      'interpreter.').format(*bold)

        self.ax = ax
        self.D = D
        self.filename = filename

        # Get a handle on the data and retain a copy of the original
        self.original_data = D.data.copy()
        self.data = D.data.copy()
        # Also remmeber whether or not this is 3D data
        self.is_three_d = True if D.data.shape[0] > 1 else False
        # Look for x- and y- scales
        self.original_X = D.xscale.copy()
        self.original_Y = D.yscale.copy()
        self.X = D.xscale.copy()
        self.Y = D.yscale.copy()
        try :
            self.Z = D.zscale.copy()
        except AttributeError :
            self.Z = None

        # Set up path-completion for respective do_xxx commands by defining 
        # complete_xxx
        self.complete_save = self.path_complete
        self.complete_png = self.path_complete

        # Plot the initial data
        self.plot()

        # Plot the energy (z scale) distribution in 3D datasets
        if self.is_three_d :
            self.do_integrate('')

    def move_right(self) :
        """ Move the last created figure to the right of the `main` figure. """
        # Get the size of the main figure in inches and convert to pixels
        size = self.ax.figure.get_size_inches()
        dpi = rcParams['figure.dpi']
        # Add the x location of the main window
        x0 = int( self.ax.figure.canvas.manager.window.geometry().split('+')[1] )
        width = int(x0 + size[0]*dpi)

        # Get the figure manager of the most recently created (?) figure and 
        # use it to move that figure
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry('+{}+{}'.format(width, 0))

    def get_axes(self) :
        """ 
        Return two lists, one with the window-titles (`figure numbers`, 
        though they are usually strings) of all figures open in APC and a 
        second one with the corresponding axes instaces.

        Returns
        -------
        names, axes  :: list, list; see description above.
        """
        axes = [plt.figure(i).axes[0] for i in plt.get_fignums()]
        names = [ax.figure.canvas.get_window_title() for ax in axes]
        return names, axes

    @cmd.with_category(VISUAL)
    def do_copy(self, arg) :
        """ 
        Create a copy of the current plot in the current state for easy 
        comparison with different settings. All created copies (as well as 
        all matplotlib figures opened in APC) are accessible from iPython via 
        `self.get_axes()`.
        """
        if arg=='' :
            arg = 'Copy of {}'.format(self.filename)
        fig, ax = plt.subplots(num=arg)
        self.plot(ax)
        self.move_right()

    """ Define the parser for :func: `do_norm`. """
    norm_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    norm_parser.add_argument('method', nargs='?', default=INT_EDC_NORM, 
                             choices=[NO_NORM, INT_EDC_NORM, FERMI_NORM],
                             help='Name of normalization method.')
                                 
    @cmd.with_category(PROCESSING)
    @cmd.with_argparser(norm_parser)
    def do_norm(self, args) :
        """ 
        Apply normalization to the data. Available options:
                    off: no normalization.
                int_edc: normalize each EDC by its integral.
            above_fermi: normalize each EDC by the mean in a defined region above the
                         Fermi level.
        """
        method = args.method

        # Apply the correct normalization
        self.poutput('Applying norm `{}`. \n'.format(method))
        self.normalization = method
        self.plot()

    """ Define the parser for :func: `do_bg` """
    bg_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bg_parser.add_argument('method', nargs='?', choices=[NO_BG, PROFILE_BG, FERMI_BG], 
                           default='off', help=('Name of the background '
                                                'subtraction method.'))
        
    @cmd.with_category(PROCESSING)
    @cmd.with_argparser(bg_parser)
    def do_bg(self, args) :
        """ Apply bg subtraction to the data. """
        method = args.method

        # Apply the selected BG subtraction
        self.poutput('Applying background subtraction `{}`.'.format(method))
        self.bg = method
        self.plot()

    """ Define the parser for :func: `do_ang2k`. NOTE: By using this parser 
    also for :func: `do_y_ang2k` we get the bug that the help message 
    displays the name `y_ang2k` in both cases. """
    a2k_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a2k_parser.add_argument('-s', '--shift', type=float, default=0,
                            help='Linear shift to apply.')
    a2k_parser.add_argument('-l', '--lattice_constant', type=float,
                            help='Lattice constant of the crystal.')

    @cmd.with_category(PROCESSING)
    @cmd.with_argparser(a2k_parser)
    def do_ang2k(self, args) :
        """ 
        Carry out an angle-to-k-space conversion with the given parameters. 
        """
        # Save the relevant k-space conversion parameters
        if args.lattice_constant :
            self.lattice_constant = args.lattice_constant
        self.kx_shift = args.shift

        self.convert_ang2k = True

        self.plot()

    @cmd.with_category(PROCESSING)
    @cmd.with_argparser(a2k_parser)
    def do_y_ang2k(self, args) :
        """ 
        Carry out an angle-to-k-space conversion for the y-axis with the 
        given parameters. 
        """
        # Save the relevant k-space conversion parameters
        if args.lattice_constant :
            self.lattice_constant = args.lattice_constant
        self.ky_shift = args.shift

        self.convert_y_ang2k = True

        self.plot()

    @cmd.with_category(PROCESSING)
    def do_shift_y(self, arg) :
        """ Shift the y scale by a fixed number. If no argument is given and 
        a Fermi index is defined, shift y such that 0 lies at the Fermi 
        index. 
        """
        if arg == '' :
            try :
                ef_index = self.D.ef_index
            except AttributeError :
                self.poutput('No Fermi index present.')
                return
            # Get the current index of 0
            #zero_index = np.argmin(np.abs(self.Y))
            arg = -self.Y[ef_index]

        self.Y += float(arg)
        self.plot()

    """ Define the parser for :func: `do_z`. 
    Call the first argument `i` instead of `z` to make help message more 
    comprehensive. """
    z_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    z_parser.add_argument('i', default=0, type=int, nargs='?',
                          help='Index of slice along z dimension.')
    z_parser.add_argument('integrate', default=0, type=int, nargs='?',
                          help='Number of slices to integrate over.')
    z_parser.add_argument('-f', '--fermi', action='store_true',
                          help=('Take a slice at the fermi index, if it is'
                                'present. If not, just take a slice at '
                                'z-value 0.'))

    @cmd.with_category(PROCESSING)
    @cmd.with_argparser(z_parser)
    def do_z(self, args) :
        """ Select the slice (z dimension). """
        if args.fermi :
            try :
                self.z = self.D.ef_index
            except AttributeError :
                self.poutput('No Fermi index present.')
                return
            # Allow the user to still supply a number of slices to integrate 
            # by treating the first number given as the usual `integrate`
            self.integrate = args.i
        else :
            self.z = args.i
            self.integrate = args.integrate
        self.plot()

    """ Define the parser for :func: `do_ef`. """
    ef_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ef_parser.add_argument('ef_index', type=int, nargs='?', default=None,
                           help=('If given, this value is stored as the index' 
                                 'of the Fermi level.'))

    @cmd.with_category(PROCESSING)
    @cmd.with_argparser(ef_parser)
    def do_ef(self, args) :
        """ 
        Get (default) or set (if argument is given) the index of the Fermi 
        level.
        """
        ef_index = args.ef_index
        if ef_index is not None :
            dl.update_namespace(self.D, ('ef_index', ef_index))
            self.poutput('Stored Fermi index {}.'.format(ef_index))
            return

        # Else
        try :
            self.poutput('Fermi index is: {}.'.format(self.D.ef_index))
        except AttributeError :
            self.poutput('No Fermi index has been defined yet.')

    """ Define the parser for :func: `do_cut` """
    cut_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cut_parser.add_argument('dimension', type=str, 
                            choices=['0', '1', '2', 'x', 'y', 'z'],
                            help=('Dimension along which to cut. The '
                                  'definitions are as in :func: `make_slice '
                                  '<kustom.arpys.postprocessing.make_slice>`, '
                                  'but can also given as `x`, `y` or `z`.'
                                  'E.g. 0 or `z` corresponds to the first '
                                  'dimension in the array ([X,:,:]) which '
                                  'is the `logical` z axis; 1 or `y` '
                                  '([:,X,:]) is the y- and 2 or `x` ([:,:,X]) '
                                  'the x-axis.'))
    cut_parser.add_argument('value', type=float,
                            help=('Value along dimension `dimension` (in that '
                                  'dimension`s units) at which to take the cut.'))
    cut_parser.add_argument('-i', '--integrate', type=float, default=0,
                            help=('The resulting cut will be the sum within '
                                  'the region `value +- integrate`.'))

    @cmd.with_category(ANALYSIS)
    @cmd.with_argparser(cut_parser)
    def do_cut(self, args) :
        """
        Cut along the specified dimension and at the specified point and 
        present the result in a new figure.
        Reminder: All matplotlib figures opened in APC) are accessible from 
        iPython via `self.get_axes()`.
        """
        # Get the right axes scales
        d = args.dimension
        if d in ['0', 'z'] :
            d = 0
            scale = self.Z
            x = self.X
            y = self.Y
        elif d in ['1', 'y'] :
            d = 1
            scale = self.Y
            x = self.X
            y = self.Z
        elif d in ['2', 'x'] :
            d = 2
            scale = self.X
            x = self.Y
            y = self.Z

        # Determine the index at which the requested value is reached
        index = np.argmin( np.abs( scale - args.value ) )

        # Convert `args.integrate` from axis units to pixels
        units_per_pixel = abs(scale[1]-scale[0])
        integrate = int( args.integrate/units_per_pixel )

        # Extract the cut
        if self.is_three_d :
            cut = pp.make_slice(self.data, d, index, integrate)
        else :
            # Artificially add a dimension for compatibility with :func: 
            # `make_slice <kustom.arpys.postprocessing.make_slice>`
            cut = pp.make_slice(np.array([self.sliced]), d, index, integrate) 

        # Plot it
        num = len( self.get_axes()[0] )
        fig = plt.figure(num='Fig. {}: Cut along dim {}'.format(num, d))
        ax = fig.add_subplot(111)
        if self.is_three_d :
            ax.pcolormesh(x, y, cut)
        else :
            ax.plot(x, cut[0])
        self.move_right()

    @cmd.with_category(ANALYSIS)
    def do_integrate(self, args) :
        """ Show the angle integrated EDC. """
        # Prepare the figure and axes
        self.angintfig = plt.figure(num='{} | Integrated'.format(self.filename),
                                    figsize=(4,4))
        self.angintax = self.angintfig.add_subplot(111, projection='cursor')

        # Handle cases for 3D and 2D data differently
        if self.is_three_d :
            # Angle-integrate the energy distributions
            integrated = self.data.sum(1).sum(1)

            # Prepare for index to energy conversion
            indices = np.arange(self.data.shape[0])
            energies = self.D.zscale 
            try :
                m = (energies.max()-energies.min()) / \
                     (indices.max()-indices.min())
            except AttributeError :
                pass

            # Define a function for index to energy conversion on-the-fly
            ind_to_energy = lambda x : (x-indices.min())*m + energies.min()

            # Connect event handling
            def on_click(event) :
                """ Print index and value of the clicked spot and set z to 
                that index. """
                self.poutput(('Clicked at pixel: {:} - z units: '
                              '{:}').format(int(event.xdata), 
                                            ind_to_energy(event.xdata)))
                self.do_z(str(int(event.xdata)))
            self.angintfig.canvas.mpl_connect('button_press_event', on_click)

            # Plot
            self.angintax.plot(indices, integrated)

        # Case 2D data
        else :
            integrated = self.data[0].sum(1)
            indices = range(len(integrated))

            # Plot
            self.angintax.plot(integrated, self.Y)

        # Move this figure to the right of the main figure
        self.move_right()

    """ Define the parser for :func: `do_cut`. This one has some 
    sub-commands, so there's quite a bit more code."""
    cursor_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    crsr_sbprsrs = cursor_parser.add_subparsers(dest='command')

    # `get` subcommand
    crsr_sbprsr_get = crsr_sbprsrs.add_parser('get', 
                                              help=('Print the cursor '
                                                    'coordinates.')) 
    # `cut` subcommand
    crsr_sbprsr_cut = crsr_sbprsrs.add_parser('cut', 
                                              help=('Produce a cut at the '
                                                    'current cursor position.'))
    crsr_sbprsr_cut.add_argument('axis', nargs='?', choices=['x', 'y', 'both'],
                                default='both',
                                help='Select along which axis to take the cut.')
    # `crop` subcommand
    crsr_sbprsr_crop = crsr_sbprsrs.add_parser('crop', 
                                               help=('Cut off parts of the '
                                                     'data relative to the'
                                                     'current cursor '
                                                     'position.'))
    crsr_sbprsr_crop.add_argument('direction', nargs='+', 
                                  choices=['above', 'below', 'left', 'right', 
                                           'a', 'b', 'l', 'r'],
                                  help=('Choose which part of the data to crop. '
                                        'You can specify several options at '
                                        'once.'))

    @cmd.with_category(ANALYSIS)
    @cmd.with_argparser(cursor_parser)
    def do_cursor(self, args) :
        """ 
        By default read out the cursor position on the main window. For 
        additional functionality see `help` of the respective subcommands.
        """
        x, y = self.ax.get_cursor()
        if x is None :
            self.poutput('No cursor present. Click main plot to create one.')
            return

        command = args.command

        # `get`: print the current cursor position
        if command in [None, 'get'] :
            self.poutput('Cursor at {:.4f} | {:.4f}.'.format(x, y))

        # `cut`: create cuts of the data along the specified axis
        elif command == 'cut' :
            if args.axis is not 'x' :
                self.do_cut('y {}'.format(y))
            if args.axis is not 'y' :
                self.do_cut('x {}'.format(x))

        # `crop`: crop the data
        elif command == 'crop' :
            d = args.direction
            if 'a' in d or 'above' in d :
                #self.crop_data(1, y, 1)
                self.crop_y_above = y
            elif 'b' in d or 'below' in d :
                #self.crop_data(1, y, -1)
                self.crop_y_below = y
            if 'l' in d or 'left' in d :
                #self.crop_data(2, x, -1)
                self.crop_x_above = x
            elif 'r' in d or 'right' in d :
                #self.crop_data(2, x, 1)
                self.crop_x_below = x

        self.plot()

    def crop_data(self, dimension, value, direction=1) :
        """ 
        Cut off parts of the data above (`direction`=1) or below 
        (`direction`=-1) the given `value` along the specified `dimension`. 
        Following the convention from :module: `postprocessing 
        <kustom.arpys.postprocessing>`, dimension 1 stands for the y- and 2 
        for the x-axis.
        """
        # Crop y axis
        if dimension==1 :
            scale = self.Y
        # Crop x axis
        elif dimension==2 :
            scale = self.X

        # Find the index of the given value
        index = np.argmin( np.abs( scale - value ) )

        # Minor correction to get all numbers when negative array indexing
        if direction == -1 and index > 0 :
            index -= 1

        # Crop the scale. The second `[::direction]` is to revert the order 
        # back to original if it has been reversed by a `direction`=-1.
        # If `direction` is +1, this part does nothing - and doesn't have to.
        scale = scale[:index:direction][::direction]

        # The awkward code may become clearer with above remark
        if dimension==1 :
            self.Y = scale
            self.data = self.data[:,:index:direction,:][:,::direction,:]
        elif dimension==2 :
            self.X = scale
            self.data = self.data[:,:,:index:direction][:,:,::direction]

    @cmd.with_category(VISUAL)
    def do_grid(self, arg=None) :
        """ 
        Toggle the plot grid. If no argument is given, change the current 
        state from on to off or vice versa. Otherwise, accepted arguments are 
        `on`, `1` or `off`, `0`, `none`.
        """
        if arg.lower() in ['on', '1'] :
            self.grid_on = 1
        elif arg.lower() in ['off', '0', 'none'] :
            self.grid_on = 0
        elif arg in ['', None] :
            self.grid_on = (self.grid_on+1)%2
        else :
            self.poutput('Invalid input: `{}`.'.format(arg))
            return

        self.poutput('Toggling grid.')
        self.plot()

    @cmd.with_category(VISUAL)
    def do_cscale(self, arg) :
        """ Adjust the maximum of the colorscale to arg*max(data). """
        # Default case
        if arg=='' :
            arg=1
        self.vmax = float(arg)
        self.plot()

    """ Define the parser for :func: `do_cmap`. This is only to provide 
    tab-completion on colormap names. """
    cmap_parser = \
    argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmap_parser.add_argument('cmap', choices=CMAPS, default=DEFAULT_CMAP,
                             help='Name of a matplotlib colormap.')

    @cmd.with_category(VISUAL)
    @cmd.with_argparser(cmap_parser)
    def do_cmap(self, args) :
        """ Select the colormap. Expects a matplotlib colormap name. """
        self.cmap = args.cmap
        self.plot()

    @cmd.with_category(VISUAL)
    def do_xlabel(self, arg) :
        """ Set the label of the x axis. """
        self.xlabel = arg
        self.plot()

    @cmd.with_category(VISUAL)
    def do_ylabel(self, arg) :
        """ Set the label of the y axis. """
        self.ylabel = arg
        self.plot()

    @cmd.with_category(VISUAL)
    def do_title(self, arg) :
        """ Set the title of the figure. """
        self.title = arg
        self.plot()

    @cmd.with_category(VISUAL)
    def do_close_all(self, arg) :
        """ Close all plots except the main one. """
        for fig in plt.get_fignums()[1:] :
            plt.close(fig)

    def do_name(self, arg) :
        """ Print the name of the currently opened file. """
        self.poutput(self.filename)

    def do_test(self, arg) :
        self.poutput(arg)

    @cmd.with_category(VISUAL)
    def do_plot(self, arg) :
        """ Replot the data. """
        self.poutput('Plotting.')
        self.plot()

    def do_reset(self, args) :
        """ Retrieve the original uncropped, unprocessed data. """
        self.data = self.original_data.copy()
        self.X = self.original_X.copy()
        self.Y = self.original_Y.copy()
        self.crop_x_above = None
        self.crop_y_above = None
        self.crop_x_below = None
        self.crop_y_below = None
        self.normalization = NO_NORM
        self.bg = NO_BG

        self.plot()

    def apply_cropping(self) :
        """ 
        Crop the data using the information stored in 
        `self.crop_{x/y}_{above/below}`.
        """
        crop_values = [self.crop_x_above,
                       self.crop_x_below,
                       self.crop_y_above,
                       self.crop_y_below]
        dimension = [2, 2, 1, 1]
        direction = [-1, 1, 1, -1]

        for i, crop_value in enumerate(crop_values) :
            if crop_value is not None :
                self.crop_data(dimension[i], crop_value, direction[i])

    def apply_normalization(self) :
        """ 
        Apply the currently selected normalization method (as stored in 
        `self.normalization`) to the data `self.data`. This is just a series 
        of if-else blocks. The actual magic is defined in :module: 
        `postprocessing <kustom.arpys.postprocessing>`.
        """
        if self.normalization == NO_NORM :
            return
        elif self.normalization == INT_EDC_NORM :
            self.sliced = pp.normalize_per_integrated_segment(self.sliced, dim=1)
        else :
            self._warn_not_implemented(self.normalization)

    def apply_bg_subtraction(self) :
        """ 
        Apply the currently selected background subtraction method (as stored 
        in `self.bg`) to the data `self.data`. This is just a series of 
        if-else blocks. The actual magic is defined in :module: 
        `postprocessing <kustom.arpys.postprocessing>`.
        """
        if self.bg == NO_BG :
            return
        elif self.bg == PROFILE_BG :
            # TODO Make profile accessible
            self.sliced, profile = pp.subtract_bg_matt(self.sliced, profile=True)
        else :
            self._warn_not_implemented(self.bg)

    def apply_kspace_conversion(self) :
        """ Carry out the angle to k-space conversions, if requested. """
        # Shorthand for self.D
        D = self.D

        # Calculate kx
        if self.convert_ang2k :
            self.X, foo = pp.angle_to_k(D.angles, D.theta, D.phi, D.hv, D.E_b,
                                        lattice_constant=self.lattice_constant,
                                        shift=self.kx_shift,
                                        degrees=True)

        # Calculate kx
        if self.convert_y_ang2k :
            self.Y, foo = pp.angle_to_k(D.yscale, theta=D.phi, phi=D.theta, 
                                        hv=D.hv, E_b=D.E_b,
                                        lattice_constant=self.lattice_constant,
                                        shift=self.ky_shift,
                                        degrees=True)

    def apply_plot_formatting(self) :
        """ Apply grid, x- and y-labels, title, etc. to the figure. """
        if self.grid_on :
            self.ax.grid(self.grid_on)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

    def plot(self, ax=None) :
        """ Replot the data applying the currently selected postporcessings. """
        # Reset to original data and generate a 2D slice from it
        self.data = self.original_data.copy()
        self.X = self.original_X.copy()
        self.Y = self.original_Y.copy()
        self.apply_cropping()
        self.sliced = pp.make_map(self.data, self.z, self.integrate)

        # Postprocessing
        self.apply_kspace_conversion()
        self.apply_normalization()
        self.apply_bg_subtraction()
        
        if ax is None :
            ax = self.ax
        # Plot
        ax.clear()
        ax.pcolormesh(self.X, self.Y, self.sliced, 
                      vmax=self.vmax*self.sliced.max(), cmap=self.cmap)

        # Matplotlib stuff
        self.apply_plot_formatting()

    @cmd.with_category(PROCESSING)
    def do_save(self, savename) :
        """ 
        Save the data as-is in a python pickle file under the given filename. 
        If no name is given, use the original filename but with a .p suffix 
        instead.
        """
        # Create default savename if necessary
        if savename == '' :
            savename = '.'.join(self.filename.split('.')[:-1] + ['p'])

        # Update the namespace
        # TODO Allow saving of a whole map instead of just a slice
        attributes = (('data', np.array([self.sliced])),
                      ('xscale', self.X),
                      ('yscale', self.Y))

        dl.update_namespace(self.D, *attributes)

        # Save the file. A confirmation prompt appears, if file under given 
        # name already exists.
        dl.dump(self.D, savename)
                  
    @cmd.with_category(VISUAL)
    def do_png(self, savename) :
        """
        Save the figure as a png file under the given name. By default just 
        uses the data filename with a png extension.
        """
        # Create default savename if necessary
        if savename == '' :
            savename = '.'.join(self.filename.split('.')[:-1] + ['png'])

        self.ax.figure.savefig(savename)
        self.poutput('Saved figure as {}.'.format(savename))

    def _warn_not_implemented(self, fcn) :
        """ Notify the user that feature or function `fcn` has not yet been 
        implemented. 
        """
        message = 'WARNING: Not yet implemented: {}.'.format(fcn)
        self.poutput(self.colorize(message, 'red'))

# +------+ #
# | Main | # ===================================================================
# +------+ #
    
if __name__ == '__main__' :

    # Set up the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str,
                        help='Path to/name of ARPES data file.')

    args = parser.parse_args()
    D = dl.load_data(args.filename)
    
    # Plot the data
    plt.ion()
    fig = plt.figure(num=args.filename)
    ax = fig.add_subplot(111, projection='cursor')

    # Move the figure to the right monitor, if it exists
    mngr = plt.get_current_fig_manager()
    monitors = get_monitors()
    #height = int(monitors[0].height / 2)
    height = 0
    if len(monitors) > 1 :
        width = monitors[0].width
    else :
        width = 0
    mngr.window.wm_geometry('+{}+0'.format(width))

    # Instantiate the CLI object
    apcmd = APCmd(ax, D, args.filename)

    """
    # If data is 3D, initiate a second window with the intensity distribution
    N_z = D.data.shape[0]
    if N_z > 1 :
        # Integrate the intensity at each energy
        intensities = D.data.sum(1).sum(1)

        indices = np.arange(N_z)
        energies = D.zscale 
        try :
            m = (energies.max()-energies.min()) / (indices.max()-indices.min())
        except AttributeError :
            pass

        ind_to_energy = lambda x : (x-indices.min())*m + energies.min()

        ifig = plt.figure(figsize=(4, 4), dpi=rcParams['figure.dpi'], 
                          num='Intensity; ' + args.filename)
        iax = ifig.add_subplot(111, projection='cursor')
        iax.plot(indices, intensities)

        # Add energy scale on top
        #iax_top = iax.twiny()
        #xticks = iax.get_xticks()
        #iax_top.set_xlim( [D.zscale.min(), D.zscale.max()] )

        # Connect event handling
        def on_click(event) :
            print(('Clicked at pixel: {:} - z units: '
                   '{:}').format(int(event.xdata), ind_to_energy(event.xdata)))
            apcmd.do_z(str(int(event.xdata)))

        ifig.canvas.mpl_connect('button_press_event', on_click)

        # Move the new window right of the main window
        apcmd.move_right()
    """

    # Start the CLI
    apcmd.cmdloop()

    plt.show()

