#!/usr/bin/python
"""
Provides several Dataloader objects which open different kinds of data files 
- typically acquired at different sources (i.e. beamlines at various 
synchrotrons) - and crunch them into the same shape and form.
The output form is an argparse.Namespace object like this:
```
    Namespace(data,
              xscale,
              yscale,
              zscale,
              angles,
              theta,
              phi,
              E_b,
              hv)
```
Where the entries are as follows:

======  ========================================================================
data    np.array of shape (z,y,x); this has to be a 3D array even for 2D data 
        - z=1 in that case. x, y and z are the lengths of the x-, y- and 
        z-scales, respectively. The convention (z,y,x) is used over (x,y,z) 
        as a consequence of matplotlib.pcolormesh transposing the data when 
        plotting.
xscale  np.array of shape(x); the x axis corresponding to the data.
yscale  np.array of shape(y); the y axis corresponding to the data.
zscale  np.array of shape(z); the z axis corresponding to the data.
angles  1D np.array; corresponding angles on the momentum axis of the 
        analyzer. Depending on the beamline (analyzer slit orientation) this 
        is expressed as theta or tilt. Usually coincides with either of x-, 
        y- and zscales.
theta   float or 1D np.array; the value of theta (or tilt in rotated analyzer 
        slit orientation). Is mostly used for angle-to-k conversion.
phi     float; value of the azimuthal angle phi. Mostly used for angle-to-k 
        conversion.
E_b     float; typical binding energy for the electrons represented in the 
        data. In principle there is not just a single binding energy but as 
        this is only used in angle-to-k conversion, where the typical 
        variations of the order <10eV doen't matter it suffices to give an 
        average or maximum value.
hv      float or 1D np.array; the used photon energy in the scan(s). In Case 
        of a hv-scan, this obviously coincides with one of the x-, y- or 
        z-scales.
======  ========================================================================

Note that any change in the output structure has consequences for all 
programs and routines that receive from a dataloader (which is pretty much 
everything in this module) and previously pickled files.
"""

import h5py
import numpy as np
import os
import pickle
import pyfits
from argparse import Namespace
from errno import ENOENT
from igor import binarywave
from warnings import catch_warnings, simplefilter

# Fcn to build the x, y (, z) ranges (maybe outsource this fcn definition)
def start_step_n(start, step, n) :
    """ 
    Return an array that starts at value `start` and goes `n` 
    steps of `step`. 
    """
    end = start + n*step
    return np.linspace(start, end, n)

class Dataloader() :
    """ 
    Base dataloader class (interface) from which others inherit some 
    methods (specifically the __repr__() function). 
    The `date` attribute should indicate the last date that this specific 
    dataloader worked properly for files of its type (as beamline filetypes 
    may vary with time).
    """
    name = 'Base'
    date = ''

    def __init__(self, *args, **kwargs) :
        pass

    def __repr__(self, *args, **kwargs) :
        return '<class Dataloader_{}>'.format(self.name)

    def print_m(self, *messages) :
        """ Print message to console, adding the dataloader name. """
        s = '[Dataloader {}]'.format(self.name)
        print(s, *messages)

class Dataloader_Pickle(Dataloader) :
    """ 
    Load data that has been saved using python's `pickle` module. ARPES 
    pickle files are assumed to just contain the data namespace the way it 
    would be returned by any Dataloader of this module. 
    """
    name = 'Pickle'

    def load_data(self, filename) :
        # Open the file and get a handle for it
        with open(filename, 'rb') as f :
            filedata = pickle.load(f)
        return filedata

class Dataloader_ALS(Dataloader) :
    """ 
    Object that allows loading and saving of ARPES data from the MAESTRO
    beamline at ALS, Berkely, which is in .fits format. 
    """
    name = 'ALS'
    # A factor required to reach a reasonable result for the k space 
    # coordinates 
    k_stretch = 1.05

    # Scanmode aliases
    CUT = 'null'
    MAP = 'Slit Defl'
    HV = 'mono_eV'
    DOPING = 'Sorensen Program'

    def __init__(self, work_func=4) :
        # Assign a value for the work function
        self.work_func = work_func

    def load_data(self, filename) :
        # Open the file
        hdulist = pyfits.open(filename)

        # Access the BinTableHDU
        self.bintable = hdulist[1]

        # Try to fix FITS standard violations
        self.bintable.verify('fix')

        # Get the header to extract metadata from
        header = hdulist[0].header

        # Find out what scan mode we're dealing with
        scanmode = header['NM_0_0']
        self.print_m('Scanmode: ', scanmode)

        # Find out whether we are in fixed or swept (dithered) mode which has 
        # an influence on how the data is stored in the fits file.
        swept = True if ('SSSW0' in header) else False
        
        # Case normal cut
        if scanmode == self.CUT :
            data = self.load_cut()
        # Case FSM
        elif scanmode == self.MAP :
            data = self.load_map(swept)
        # Case hv scan
        elif scanmode == self.HV :
            data = self.load_hv_scan()
        # Case doping scan
        elif scanmode == self.DOPING :
            #data = self.load_doping_scan()
            data = self.load_hv_scan()
        else :
            raise(IndexError('Couldn\'t determine scan type'))

        nz, ny, nx = data.shape

        # Determine whether the file uses the SS or SF prefix for metadata
        if 'SSX0_0' in header :
            pre = 'SS'
        elif 'SFX0_0' in header :
            pre = 'SF'
        else :
            raise(IndexError('Neither SSX0_0 nor SFX0_0 appear in header.'))

        # Starting pixel (energy scale)
        e0 = header[pre+'X0_0']
        # Ending pixel (energy scale)
        e1 = header[pre+'X1_0']
        # Pixels per eV
        ppeV = header[pre+'PEV_0']
        # Zero pixel (=Fermi level?)
        fermi_level = header[pre+'KE0_0']

        # Create the energy (y) scale
        energy_binning = 2
        energies_in_px = np.arange(e0, e1, energy_binning)
        energies = (energies_in_px - fermi_level)/ppeV

        # Starting and ending pixels (angle or ky scale)
        # DEPRECATED
        #y0 = header[pre+'Y0_0']
        #y1 = header[pre+'Y1_0']

        # Use this arbitrary seeming conversion factor (found in Denys' 
        # script) to get from pixels to angles
        #deg_per_px = 0.193 / 2
        deg_per_px = 0.193 * self.k_stretch
        #angle_binning = 3
        angle_binning = 2

        #angles_in_px = np.arange(y0, y1, angle_binning)
        #n_y = int((y1-y0)/angle_binning)
        #angles_in_px = np.arange(0, n_y, 1)
        angles_in_px = np.arange(0, ny, 1)
        # NOTE
        # Actually, I would think that we have to multiply by 
        # `angle_binning` to get the right result. That leads to 
        # unreasonable results, however, and division just gives a 
        # perfect looking output. Maybe the factor 0.193/2 from ALS has 
        # changed with the introduction of binning?
        #angles = angles_in_px * deg_per_px / angle_binning
        angles = angles_in_px * deg_per_px  / angle_binning

        # Case cut
        if scanmode == self.CUT :
            # NOTE x and y scale may need to be swapped
            yscale = energies
            xscale = angles
            zscale = None
        # Case map
        elif scanmode == self.MAP :
            x0 = header['ST_0_0']
            x1 = header['EN_0_0']
            #n_x = header['N_0_0']
            #xscale = np.linspace(x0, x1, n_x) * self.k_stretch
            xscale = np.linspace(x0, x1, nx) #* self.k_stretch
            yscale = angles
            zscale = energies
        # Case hv scan
        elif scanmode == self.HV :
            z0 = header['ST_0_0']
            z1 = header['EN_0_0']
            #nz = header['N_0_0']
            angles_in_px = np.arange(0, nx, 1)
            angles = angles_in_px * deg_per_px  / angle_binning
            xscale = angles
            yscale = energies
            zscale = np.linspace(z0, z1, nz)
        # Case doping
        elif scanmode == self.DOPING :
            z0 = header['ST_0_0']
            z1 = header['EN_0_0']
            # Not sure how the sqrt2 comes in
            angles_in_px = np.arange(0, nx, 1) * np.sqrt(2)
            angles = angles_in_px * deg_per_px  / angle_binning 
            xscale = angles
            # For some reason the energy determination from file fails here...
            yscale = np.arange(ny, dtype=float)
            zscale = np.linspace(z0, z1, nz)

        # For the binding energy, just take a min value as its variations 
        # are small compared to the photon energy
        E_b = energies.min()

        # Extract some additional metadata (mostly for angles->k conversion)
        # TODO Special cases for different scan types
        # Get relevant metadata from header
        theta = header['LMOTOR3']
        # beta in LMOTOR4
        phi = header['LMOTOR5']

        # The photon energy may vary in the case that this is a hv_scan
        hv = header['MONO_E']

        # In recent ALS data the above determined x, y and z scales did not 
        # match the actual shape of the data...
        self.print_m(*data.shape)
        self.print_m(nx, ny, nz)
        for scale in [xscale, yscale, zscale] :
            try :
                self.print_m(len(scale))
            except Exception :
                self.print_m('length problem')

        # NOTE angles==xscale and E_b can be extracted from yscale, thus it 
        # is not really necessary to pass them on here.
        res = Namespace(
               data = data,
               xscale = xscale,
               yscale = yscale,
               zscale = zscale,
               angles = angles,
               theta = theta,
               phi = phi,
               E_b = E_b,
               hv = hv
        )
        return res
    
    def load_cut(self) :
        """ Read data from a 'cut', which is just one slice of energy vs k. """
        # Access the actual data which is in the last element of the last 
        # element of the bin table...
        fields = self.bintable.data[-1]
        data = fields[-1]

        # Reshape towards GUI expectations
        x, y = data.shape
        data = data.reshape(1, x, y)

        return data

    def load_map(self, swept) :
        """ 
        Read data from a 'map', i.e. several energy vs k slices and bring 
        them in the right shape for the gui, which is 
        (energy, k_parallel, k_perpendicular)
        """
        bt_data = self.bintable.data
        n_slices = len(bt_data)

        first_slice = bt_data[0][-1]
        # The shape of the returned array must be (energy, kx, ky)
        # (n_slices corresponds to ky)
        if swept :
            n_energy, n_kx = first_slice.shape    
            data = np.zeros([n_energy, n_kx, n_slices])
            for i in range(n_slices) :
                this_slice = bt_data[i][-1]
                data[:,:,i] = this_slice
        else :
            n_kx, n_energy = first_slice.shape
            data = np.zeros([n_energy, n_kx, n_slices])
            for i in range(n_slices) :
                this_slice = bt_data[i][-1]
                # Reshape the slice to (energy, kx)
                this_slice = this_slice.transpose()
                data[:,:,i] = this_slice

        # Convert to numpy array
        data = np.array(data)

        return data

    def load_hv_scan(self) :
        """ 
        Read data from a hv scan, i.e. a series of energy vs k cuts 
        (shape (n_kx, n_energy), each belonging to a different photon energy 
        hv. The returned shape in this case must be 
        (photon_energies, energy, k)
        The same is used for doping scans, where the output shape is
        (doping, energy, k)
        """
        bt_data = self.bintable.data
        n_hv = len(bt_data)
        # = header['N_0_0']

        first_cut = bt_data[0][-1]
        n_kx, n_energy = first_cut.shape # should all be accessible from header

        data = np.zeros([n_hv, n_kx, n_energy])
        for i in range(n_hv) :
            this_slice = bt_data[i][-1]
            data[i,:,:] = this_slice

        # Convert to numpy array
        data = np.array(data)

        return data

class Dataloader_SIS(Dataloader) :
    """ 
    Object that allows loading and saving of ARPES data from the SIS 
    beamline at PSI which is in hd5 format. 
    """
    name = 'SIS'
    # Number of cuts that need to be present to assume the data as a map 
    # instead of a series of cuts
    min_cuts_for_map = 10

    def __init__(self, filename=None) :
        pass

    def load_file(self, filename) :
        """ Load and store the full h5 file. """
        # Load the hdf5 file
        self.datfile = h5py.File(filename, 'r')

    def load_data(self, filename) :
        """ 
        Extract and return the actual 'data', i.e. the recorded map/cut. 
        Also return labels which provide some indications what the data means.
        """
        # Note: x and y are a bit confusing here as the hd5 file has a 
        # different notion of zero'th and first dimension as numpy and then 
        # later pcolormesh introduces yet another layer of confusion. The way 
        # it is written now, though hard to read, turns out to do the right 
        # thing and leads to readable code from after this point.

        self.load_file(filename)

        # Extract the actual dataset and some metadata
        h5_data = self.datfile['Electron Analyzer/Image Data']
        attributes = h5_data.attrs

        # Convert to array and make 3 dimensional if necessary
        data = np.array(h5_data)
        shape = data.shape
        # How the data needs to be arranged depends on the scan type: cut, 
        # map, hv scan or a sequence of cuts
        # Case cut
        if len(shape) == 2 :
            x = shape[0]
            y = shape[1]
            # Make data 3D
            data = data.reshape(1, x, y)
            N_E = y
            # Extract the limits
            xlims = attributes['Axis1.Scale']
            ylims = attributes['Axis0.Scale']
            elims = ylims
        # shape[2] should hold the number of cuts. If it is reasonably large, 
        # we have a map. Otherwise just a sequence of cuts.
        # Case map
        elif shape[2] > self.min_cuts_for_map :
            print('Dataloader PSI. Is a map?')
            x = shape[1]
            y = shape[2]
            N_E = shape[0]
            # Extract the limits
            xlims = attributes['Axis2.Scale']
            ylims = attributes['Axis1.Scale']
            elims = attributes['Axis0.Scale']
        # Case sequence of cuts
        else :
            x = shape[0]
            y = shape[1]
            N_E = y
            z = shape[2]
            # Reshape data
            #new_data = np.zeros([z, x, y])
            #for i in range(z) :
            #    cut = data[:,:,i]
            #    new_data[i] = cut
            #data = new_data
            data = np.rollaxis(data, 2, 0)
            # Extract the limits
            xlims = attributes['Axis1.Scale']
            ylims = attributes['Axis0.Scale']
            elims = ylims

        # Construct x, y and energy scale (x/ylims[1] contains the step size)
        xscale = self.make_scale(xlims, y)
        yscale = self.make_scale(ylims, x)
        energies = self.make_scale(elims, N_E)

        # Extract some data for ang2k conversion
        metadata = self.datfile['Other Instruments']
        theta = metadata['Theta'][0]
        #theta = metadata['Tilt'][0]
        phi = metadata['Phi'][0]
        hv = attributes['Excitation Energy (eV)']
        angles = xscale
        E_b = min(energies)

        res = Namespace(
               data = data,
               xscale = xscale,
               yscale = yscale,
               zscale = energies,
               angles = angles,
               theta = theta,
               phi = phi,
               E_b = E_b,
               hv = hv
        )
        return res

    def make_scale(self, limits, nstep) :
        """ 
        Helper function to construct numbers starting from limits[0] 
        and going in steps of limits[1] for nstep steps.
        """
        start = limits[0]
        step = limits[1]
        end = start + (nstep+1)*step
        return np.linspace(start, end, nstep)

class Dataloader_ADRESS(Dataloader) :
   """ ADRESS beamline at SLS, PSI. """
   name = 'ADRESS'

   def load_data(self, filename) :
        h5file = h5py.File(filename, 'r')
        # The actual data is in the field: 'Matrix'
        matrix = h5file['Matrix']

        # The scales can be extracted from the matrix' attributes
        scalings = matrix.attrs['IGORWaveScaling']
        units = matrix.attrs['IGORWaveUnits']
        info = matrix.attrs['IGORWaveNote']

        # Convert `units` and `info`, which is a bytestring of ASCII 
        # characters, to lists of strings
        metadata = info.decode('ASCII').split('\n')
        units = [b.decode('ASCII') for b in units[0]]

        # Put the data into a numpy array and convert to float
        data = np.array(matrix, dtype=float)
        shape = data.shape

        # 'IGORWaveUnits' contains a list of the form 
        # ['', 'degree', 'eV', units[3]]. The first three elements should 
        # always be the same, but the third one may vary or not even exist. 
        # Use this to determine the scan type.
        # Convert to np.array bring it in the shape the gui expects, which is 
        # [energy/hv/1, kparallel/"/kparallel, kperpendicular/"/energy] and 
        # prepare the x,y,z # scales
        # Note: `scalings` is a 3/4 by 2 array where every line contains a 
        # (step, start) pair
        if len(units) == 3 :
        # Case cut
            # Make data 3-dimensional by adding an empty dimension
            data = data.reshape(1, shape[0], shape[1])
            data = np.rollaxis(data, 2, 1)
            # Shape has changed                                   
            shape = data.shape
            xstep, xstart = scalings[1]
            ystep, ystart = scalings[2]
            zscale = None
        else :
        # Case map or hv scan (or...?)
            data = np.rollaxis(data, 1, 0)
            # Shape has changed                                   
            shape = data.shape
            xstep, xstart = scalings[3]
            ystep, ystart = scalings[1]
            zstep, zstart = scalings[2]
            zscale = start_step_n(zstart, zstep, shape[0])

        # Binned data may contain some sort of scaling 
        xscale = start_step_n(xstart, xstep, shape[2])
        yscale = start_step_n(ystart, ystep, shape[1])

        def convert_raw(raw) :
            """
            Try to convert a string which is expected to be either just a 
            decimal number or a MATLAB-like range expression 
            (START:STEP:STOP) to either a float or an np.array. 
            """
            if ':' in raw :
                # raw is of the form start:step:end
                start, step, end = [float(n) for n in raw.split(':')]
                res = np.arange(start, end*step, step)
            else :
                res = float(raw)
            return res

        hv_raw = metadata[0].split('=')[-1]
        hv = convert_raw(hv_raw)
        theta_raw = metadata[8].split('=')[-1]
        theta = convert_raw(theta_raw)
        phi_raw = metadata[10].split('=')[-1]
        phi = convert_raw(phi_raw)
        angles = xscale
        # For the binding energy just take the minimum of the energies
        E_b = yscale.min()

        res = Namespace(
               data = data,
               xscale = xscale,
               yscale = yscale,
               zscale = zscale,
               angles = angles,
               theta = theta,
               phi = phi,
               E_b = E_b,
               hv = hv
        )
        return res

class Dataloader_CASSIOPEE(Dataloader) :
    """ CASSIOPEE beamline at SOLEIL synchrotron, Paris. """
    name = 'CASSIOPEE'
    date = '18.07.2018'

    def load_data(self, filename) :
        """ 
        Single cuts are stored as two files: One file contians the data and 
        the other the metadata. Maps, hv scans and other `external 
        loop`-scans are stored as a directory containing these two files for 
        each cut/step of the external loop. Thus, this dataloader 
        distinguishes between directories and single files and changes its 
        behaviour accordingly.
        """
        if os.path.isfile(filename) :
            return self.load_from_file(filename)
        else :
            if not filename.endswith('/') : filename += '/'
            return self.load_from_dir(filename)

    def load_from_dir(self, dirname) :
        """
        Load 3D data from a directory as it is output by the IGOR macro used 
        at CASSIOPEE. The dir is assumed to contain two files for each cut:

            BASENAME_INDEX_i.txt     -> beamline related metadata
            BASENAME_INDEX_ROI1_.txt -> data and analyzer related metadata

        To be more precise, the assumptions made on the filenames in the 
        directory are:
            - the INDEX is surrounded by underscores (`_`) and appears after 
              the first underscore.
            - the string `ROI` appears in the data filename.
        """
        # Get the all filenames in the dir
        all_filenames = os.listdir(dirname)
        # Remove all non-data files
        filenames = []
        for name in all_filenames :
            if 'ROI' in name :
                filenames.append(name)

        # Get metadata from first file in list
        skip, energy, angles, hv = self.get_metadata(dirname+filenames[0]) 

        # Get the data from each cut separately. This happens in the order 
        # they appear in os.listdir() which is usually not what we want -> a 
        # reordering is necessary later.
        unordered = {}
        i_min = np.inf
        i_max = -np.inf
        for name in filenames :
            # Keep track of the min and max indices in the directory
            i = int(name.split('_')[1])
            if i < i_min : i_min = i
            if i > i_max : i_max = i
            
            # Get the data of cut i
            this_cut = np.loadtxt(dirname+name, skiprows=skip+1)[:,1:]
            unordered.update({i: this_cut})

        # Properly rearrange the cuts
        data = []
        for i in range(i_min, i_max) :
            data.append(unordered[i])
        data = np.array(data)
        # For a map, we expect output of the form (Energy, k_para, k_perp), 
        # currently it is (tilt, energy, theta) -> reshape with moveaxis
        data = np.moveaxis(data, 0, 1)

        res = Namespace(
            data = data,
            xscale = angles,
            yscale = np.arange(i_min, i_max),
            zscale = energy,
            angles = angles,
            theta = 1,
            phi = 1,
            E_b = 0,
            hv = hv)
        return res

    def load_from_file(self, filename) :
        """ 
        Load just a single cut. However, at CASSIOPEE they output .ibw files 
        if the cut does not belong to a scan...
        """
        if filename.endswith('.ibw') :
            return self.load_from_ibw(filename)
        else :
            return self.load_from_txt(filename)

    def load_from_ibw(self, filename) :
        """
        Load scan data from an IGOR binary wave file. Luckily someone has 
        already written an interface for this (the python `igor` package).
        """
        wave = binarywave.load(filename)['wave']
        data = np.array([wave['wData']])

        # The `header` contains some metadata
        header = wave['wave_header']
        nDim = header['nDim']
        steps = header['sfA']
        starts = header['sfB']

        # Construct the x and y scales from start, stop and n
        yscale = start_step_n(starts[0], steps[0], nDim[0])
        xscale = start_step_n(starts[1], steps[1], nDim[1])

        # Convert `note`, which is a bytestring of ASCII characters that 
        # contains some metadata, to a list of strings
        note = wave['note']
        note = note.decode('ASCII').split('\r')

        # Now the extraction fun begins. Most lines are of the form 
        # `Some-kind-of-name=some-value`
        metadata = dict()
        for line in note :
            # Split at '='. If it fails, we are not in a line that contains 
            # useful information
            try :
                name, val = line.split('=')
            except ValueError :
                continue
            # Put the pair in a dictionary for later access
            metadata.update({name: val})
#            # Now check if the name is any of the keywords we are interested in
#            if name=='Excitation Energy' :
#                hv = val
#            elif name=='Detector First X-Channel' :
#                x0 = val
#            elif name=='Detector Last X-Channel' :
#                x1 = val
#            elif name=='Detector First Y-Channel' :
#                y0 = val
#            elif name=='Detector Last Y-Channel' :
#                y1 = val
#            elif :
        
        hv = metadata['Excitation Energy']
        res = Namespace(
                data = data,
                xscale = xscale,
                yscale = yscale,
                zscale = None,
                angles = xscale,
                theta = 0,
                phi = 0,
                E_b = 0,
                hv = hv)
        return res

    def load_from_txt(self, filename) :
        i, energy, angles, hv = self.get_metadata(filename)
        data0 = np.loadtxt(filename, skiprows=i+1)
        # The first column in the datafile contains the angles
        angles_from_data = data0[:,0]
        data = np.array([data0[:,1:]])

        res = Namespace(
            data = data,
            xscale = angles,
            yscale = energy,
            zscale = None,
            angles = angles,
            theta = 1,
            phi = 1,
            E_b = 0,
            hv = hv)

        return res

    def get_metadata(self, filename) :
        """ 
        Extract some of the metadata stored in a CASSIOPEE output text file. 
        Also try to detect the line number below which the data starts (for 
        np.loadtxt's skiprows.

        Returns:
        ======  ================================================================
        i       int; last line number still containing metadata.
        energy  1D np.array; energy (y-axis) values.
        angles  1D np.array; angle (x-axis) values.
        hv      float; photon energy for this cut.
        ======  ================================================================
        """
        with open(filename, 'r') as f :
            for i,line in enumerate(f.readlines()) :
                if line.startswith('Dimension 1 scale=') :
                    energy = line.split('=')[-1].split()
                    energy = np.array(energy, dtype=float)
                elif line.startswith('Dimension 2 scale=') :
                    angles = line.split('=')[-1].split()
                    angles = np.array(angles, dtype=float)
                elif line.startswith('Excitation Energy') :
                    hv = float(line.split('=')[-1])
                elif line.startswith('inputA') :
                    # this seems to be the last line before the data
                    break
        return i, energy, angles, hv

# +-------+ #
# | Tools | # ==================================================================
# +-------+ #

# List containing all reasonably defined dataloaders
all_dls = [
           Dataloader_SIS,
           Dataloader_ADRESS,
           Dataloader_ALS,
           Dataloader_CASSIOPEE,
           Dataloader_Pickle
          ]

# Function to try all dataloaders in all_dls
def load_data(filename, exclude=None) :
    """
    Try to load some dataset 'filename' by iterating through `all_dls` 
    and appliyng the respective dataloader's load_data method. If it works: 
    great. If not, try with the next dataloader. 
    Collects and prints all raised exceptions in case that no dataloader 
    succeeded.
    """ 
    # Sanity check: does the given path even exist in the filesystem?
    if not os.path.exists(filename) :
        raise FileNotFoundError(ENOENT, os.strerror(ENOENT), filename) 

    # If only a single string is given as exclude, pack it into a list
    if exclude is not None and type(exclude)==str :
        exclude = [exclude]
    
    # Keep track of all exceptions in case no loader succeeds
    exceptions = dict()

    # Suppress warnings
    with catch_warnings() :
        simplefilter('ignore')
        for dataloader in all_dls :
            # Instantiate a dataloader object
            dl = dataloader()

            # Skip to the next if this dl is excluded (continue brings us 
            # back to the top of the loop, starting with the next element)
            if exclude is not None and dl.name in exclude : 
                continue

            # Try loading the data
            try :
                namespace = dl.load_data(filename)
            except Exception as e :
                # Temporarily store the exception
                exceptions.update({dl : e})
                # Try the next dl
                continue

            # Reaching this point must mean we succeeded
            return namespace

    # Reaching this point means something went wrong. Print all exceptions.
    for dl in exceptions :
        print(dl)
        e = exceptions[dl]
        print('Exception {}: {}'.format(type(e), e))

    raise Exception('Could not load data {}.'.format(filename))

# Function to create a python pickle file from a data namespace
def dump(D, filename, force=False) :
    """ 
    Wrapper for :func: `pickle.dump()`. Does not overwrite if a file of 
    the given name already exists, unless :param: `force` is True.

    ========  ==================================================================
    D         argparse.Namespace; the namespace holding the data and 
              metadata. The format is the same as what is returned by a 
              dataloader.
    filename  str; name of the output file to create.
    force     boolean; if True, overwrite existing file.
    ========  ==================================================================
    """
    # Check if file already exists
    if not force and os.path.isfile(filename) :
        question = 'File {} exists. Overwrite it? (y/N)'.format(filename)
        answer = input(question)
        # If the answer is anything but a clear affirmative, stop here
        if answer.lower() not in ['y', 'yes'] :
            return

    with open(filename, 'wb') as f :
        pickle.dump(D, f)

    message = 'Wrote to file {}.'.format(filename)
    print(message)

def update_namespace(D, *attributes) :
    """ 
    Add arbitrary attributes to a :class: `Namespace <argparse.Namespace>`.

    ==========  ================================================================
    D           argparse.Namespace; the namespace holding the data and 
                metadata. The format is the same as what is returned by a 
                dataloader.
    attributes  tuples or len(2) lists; (name, value) pairs of the attributes 
                to add. Where `name` is a str and value any python object.
    ==========  ================================================================
    """
    for name, attribute in attributes :
        D.__dict__.update( {name: attribute} )

def add_attributes(filename, *attributes) :
    """ Add arbitrary attributes to an argparse.Namespace that is stored as a 
    python pickle file. Simply opens the file, updates the namespace with 
    :func: `update_namespace <arpys.dataloaders.update_namespace>` and writes 
    back to file.

    ==========  ================================================================
    filename    str; name of the file to update.
    attributes  tuples or len(2) lists; (name, value) pairs of the attributes 
                to add. Where `name` is a str and value any python object.
    ==========  ================================================================
    """
    dataloader = Dataloader_Pickle()
    D = dataloader.load_data(filename)
    
    update_namespace(D, *attributes)
    
    dump(D, filename, force=True)
    
# +---------+ #
# | Testing | # ================================================================
# +---------+ #

if __name__ == '__main__' :
#    sis = Dataloader_SIS()
#    path = '/home/kevin/qmap/experiments/2017_10_PSI/Tl2201/Tl_1_0003.h5'
#    datadict = sis.load_data(path)
#    print(datadict['data'].shape)
#    print(datadict['xscale'].shape)
#    print(datadict['yscale'].shape)

#    adress = Dataloader_ADRESS()
#    path = '/home/kevin/qmap/experiments/2018_03_PSI/Tl2201/003_quickmap_540eV.h5'
#    ns = adress.load_data(path)
#
#    path = '/home/kevin/qmap/experiments/2018_03_PSI/Tl2201/002_quick_kz_350to800.h5'
#    ns = adress.load_data(path)
#
#    path = '/home/kevin/qmap/experiments/2018_03_PSI/Tl2201/014_HSscan_nodal_428eV.h5'
#    ns = adress.load_data(path)
    
#    D = Namespace(a=0, b=1)
#    dump(D, 'foo.p')

#    add_attributes('foo.p', ('c', 2), ['d', 'three'])

#    dl = Dataloader_Pickle()
#    D = dl.load_data('foo.p')
#    print(D)

    D = load_data('/home/kevin/qmap/experiments/2018_07_CASSIOPEE/S1_FSM/FS_1_ROI1_.txt')
    import matplotlib.pyplot as plt

    plt.pcolormesh(D.xscale, D.yscale, D.data[0])
    plt.show()

