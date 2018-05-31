#!/usr/bin/python

"""
arpes_plot.py
A tool to plot ARPES data from the command line.
"""

import cmd2 as cmd
import matplotlib.pyplot as plt
import numpy as np

import kustom.arpys.postprocessing as pp

plt.ion()

class APCmd(cmd.Cmd) :
    """ (A)RPES (P)lot (C)o(m)man(d) line interpreter.
    """
    def __init__(self, ax, D, *args, **kwargs) :
        """
        ==  ====================================================================
        ax  :class: `Axes <matplotlib.axes.Axes>` object; the ax on which the 
            plot lives.
        D   :class: `Namespace <argparse.Namespace>` object; the 
            Namepsace with the data and metadata of an ARPES file as it would 
            be returned by a :module: `dataloader <kustom.arpys.dataloaders>`.
        ==  ====================================================================
        """
        super().__init__(*args, **kwargs)
        self.ax = ax
        self.D = D
        # Get a handle on the data and retain a copy of the original
        self.original_data = D.data
        self.data = D.data.copy()

    def parse(self, args, default=None) :
        """ Parse the supplied arguments by simply splitting on whitespace. 
        Returns the list of user-supplied arguments. If no user arguments 
        were supplied, the args are set to a list containing just the element 
        `default`.
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
        INT_EDC = 'integrated_edc'
        ABOVE_FERMI = 'above_fermi'
        NONE = 'no_norm'
        methods = {
                   INT_EDC: ['1', 'int', 'int_edc'],
                   ABOVE_FERMI: ['2', 'above', 'fermi', 'above_fermi'],
                   NONE: ['0', 'none', 'no', 'n']
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
            print('Normalization method name `{}` not understood. Try one of \
these:\n'.format(original_input))
            print('{:<20}  {:<}'.format('Method name', 'Aliases'))
            print(80*'=')
            for name in methods :
                print('{:<20}  {:<}'.format(name, '  '.join(methods[name])))
            return

        # Apply the correct normalization
        print('Applying norm {}. \n'.format(method))
        
        if method==INT_EDC :
           self.data = pp.normalize_per_integrated_segment(self.data, dim=1)
        elif method==NONE :
            self.data = self.original_data.copy()
        self.plot()

    def do_grid(self, arg=None) :
        """ Toggle the plot grid. """
        # ax.grid() only accepts 'on' or 'off'
        if arg in [True, 1] :
            arg = 'on'
        elif arg in [False, 0] :
            arg = 'off'

        print('Toggling grid.')
        if arg in ['', None] :
            self.ax.grid()
        else :
            self.ax.grid(arg)

    def do_test(self, arg, *args) :
        print(arg, args)

    def plot(self) :
        """ Replot the current `self.data`. """
        self.ax.pcolormesh(self.data[0])
    
if __name__ == '__main__' :

    from dataloaders import load_data

    filename = '/home/kevin/Documents/qmap/materials/Bi2201/2018_04_ADRESS/034_nodal_cut_LH_kz43.h5'
    D = load_data(filename)
    
    fig, ax = plt.subplots()

    ax.pcolormesh(D.data[0])

    plt.show(block=False)

    apcmd = APCmd(ax, D)
    apcmd.cmdloop()

    plt.show()

