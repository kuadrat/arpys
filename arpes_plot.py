#!/usr/bin/python

"""
arpes_plot.py
A tool to plot ARPES data from the command line.
"""

import argparse
import cmd2 as cmd
import matplotlib.pyplot as plt
import numpy as np

import kustom.arpys.dataloaders as dl
import kustom.arpys.postprocessing as pp

# +--------------------------+ #
# | Parameters and constants | # ===============================================
# +--------------------------+ #

# Shortands for final variables
INT_EDC = 'integrated_edc'
ABOVE_FERMI = 'above_fermi'
NONE = 'no_norm'

# Kevin's screen dimensions
WIDTH = 1600
HEIGHT = 1200

plt.ion()

# +-----+ #
# | CLI | # ====================================================================
# +-----+ #

class APCmd(cmd.Cmd) :
    """ (A)RPES (P)lot (C)o(m)man(d) line interpreter.
    """
    #_Initial_parameters________________________________________________________
    normalization = NONE
    vmax = 1

    #_Cmd2_configuration________________________________________________________
    locals_in_py = True
    allow_cli_args = False

    def __init__(self, ax, D, filename, *args, **kwargs) :
        """
        ==  ====================================================================
        ax  :class: `Axes <matplotlib.axes.Axes>` object; the ax on which the 
            plot lives.
        D   :class: `Namespace <argparse.Namespace>` object; the Namepsace 
            with the data and metadata of an ARPES file as it would be 
            returned by a :module: `dataloader <kustom.arpys.dataloaders>`.
        ==  ====================================================================
        """
        super().__init__(*args, use_ipython=True, **kwargs)

        # Define the prompt and welcome messages
        self.prompt = self.colorize('[APC] ', 'cyan')
        bold = [self.colorize(i, 'bold') for i in ['APC', 'A', 'P', 'C']]
        self.intro = 'Welcome to {}, the {}RPES {}lot {}ommand line \
interpreter.'.format(*bold)

        self.ax = ax
        self.D = D
        self.filename = filename

        # Get a handle on the data and retain a copy of the original
        self.original_data = D.data
        self.data = D.data.copy()
        # Look for x- and y- scales
        self.X = D.xscale.copy()
        self.Y = D.yscale.copy()

    def parse(self, args, default=None) :
        """ Parse the supplied arguments by simply splitting on whitespace. 
        Returns the list of user-supplied arguments. If no user arguments 
        were supplied, the args are set to a list containing just the element 
        `default`.

        =======  ===============================================================
        args     str; space-seperated sequence of arguments.
        default  str; if `args` is an empty string, the result will be the list
                 [`default`].
        =======  ===============================================================
        """
        args = args.split()
        if len(args) is 0 :
            args = [default]
        return args

    def do_norm(self, args) :
        """ Apply normalization to the data. """
        args = self.parse(args, default='int')
        method = args[0]

        # Retain the original user input for later
        original_input = method

        # Define all recognized command names for the different methods
        methods = {
                   INT_EDC: ['1', 'int', 'int_edc'],
                   ABOVE_FERMI: ['2', 'above', 'fermi', 'above_fermi'],
                   NONE: ['0', 'none', 'no', 'n', 'off']
                  }
        
        # Convert input to lowercase
        method = method.lower()
        
        # Try to understand the user supplied method name
        recognized = False
        for name in methods :
            if method in [name] + methods[name] :
                method = name
                # Recognized! Leave the loop
                recognized = True
                break

        # Print a help text
        if not recognized :
            self.poutput('Normalization method name `{}` not understood. Try one of \
these:\n'.format(original_input))
            self.poutput('{:<20}  {:<}'.format('Method name', 'Aliases'))
            self.poutput(80*'=')
            for name in methods :
                self.poutput('{:<20}  {:<}'.format(name, '  '.join(methods[name])))
            return

        # Apply the correct normalization
        self.poutput('Applying norm {}. \n'.format(method))
        
        #if method==INT_EDC :
           #self.data = pp.normalize_per_integrated_segment(self.data, dim=1)
        self.normalization = method
        self.plot()

    """ Define the parser for :func: `do_ang2k` """
    a2k_parser = argparse.ArgumentParser()
    a2k_parser.add_argument('-s', '--shift', type=float, default=0,
                            help='Linear shift to apply.')
    a2k_parser.add_argument('-l', '--lattice_constant', type=float,
                            help='Lattice constant of the crystal.')

    @cmd.with_argparser(a2k_parser)
    def do_ang2k(self, args) :
        """ Carry out an angle-to-k-space conversion with the given 
        parameters. 
        """
        # Shorthand for self.D
        D = self.D

        # Calculate kx
        self.X, foo = pp.angle_to_k(D.angles, D.theta, D.phi, D.hv, D.E_b,
                                    lattice_constant=args.lattice_constant,
                                    shift=args.shift,
                                    degrees=True)
        self.plot()

    def do_shift_y(self, arg) :
        """ Shift the y scale by a fixed number. """
        self.Y += float(arg)
        self.plot()

    def do_grid(self, arg=None) :
        """ Toggle the plot grid. """
        # ax.grid() only accepts 'on' or 'off'
        if arg in [True, 1] :
            arg = 'on'
        elif arg in [False, 0] :
            arg = 'off'

        self.poutput('Toggling grid.')
        if arg in ['', None] :
            self.ax.grid()
        else :
            self.ax.grid(arg)

    def do_cscale(self, arg) :
        """ Adjust the maximum of the colorscale to arg*max(data). """
        self.vmax = float(arg)
        self.plot()

    def do_name(self, arg) :
        """ Print the name of the currently opened file. """
        self.poutput(self.filename)

    def do_test(self, arg) :
        self.poutput(arg)

    def do_plot(self, arg) :
        """ Replot the data. """
        self.poutput('Plotting.')
        self.plot()

    def plot(self) :
        """ Replot the data applying the currently selected postporcessings. """
        # Reset to original data
        self.data = self.original_data.copy()

        # Normalization
        if self.normalization == NONE :
            pass
        elif self.normalization == INT_EDC :
            self.data = pp.normalize_per_integrated_segment(self.data, dim=1)

        # Plot
        self.ax.clear()
        self.ax.pcolormesh(self.X, self.Y, self.data[0], 
                           vmax=self.vmax*self.data[0].max())

    def do_save(self, savename) :
        """ Save the data as-is in a python pickle file under the given 
        filename. If no name is given, use the original filename but with a 
        .p suffix instead.
        """
        # Create default savename if necessary
        if savename == '' :
            savename = '.'.join(self.filename.split('.')[:-1] + ['p'])

        # Update the namespace
        attributes = (('xscale', self.X),
                      ('yscale', self.Y))

        dl.update_namespace(self.D, *attributes)

        # Save the file. A confirmation prompt appears, if file under given 
        # name already exists.
        dl.dump(self.D, savename)

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
    fig, ax = plt.subplots(num=args.filename)
    ax.pcolormesh(D.xscale, D.yscale, D.data[0])

    # Move the figure to the right screen
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+{}+0".format(WIDTH))

    #plt.show(block=False)

    # Start the CLI
    apcmd = APCmd(ax, D, args.filename)
    apcmd.cmdloop()

    plt.show()

