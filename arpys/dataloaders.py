#!/usr/bin/python
"""
Provides several Dataloader objects which open different kinds of data files 
- typically acquired at different sources (i.e. beamlines at various 
synchrotrons) - and crunch them into the same shape and form.
The output form is an :class:`argparse.Namespace` object like this::

    Namespace(data,
              xscale,
              yscale,
              zscale,
              angles,
              theta,
              phi,
              E_b,
              hv)

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

import ast
import os
import pickle
import re
import zipfile
from argparse import Namespace
from errno import ENOENT
from warnings import catch_warnings, simplefilter, warn

import h5py
import numpy as np
import astropy.io.fits as pyfits
from igor import binarywave

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
    methods (specifically the ``__repr__()`` function). 
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

class Dataloader_i05(Dataloader) :
    """
    Dataloader object for the i05 beamline at the Diamond Light Source.
    """
    name = 'i05'

    def load_data(self, filename) :
        # Read file with h5py reader
        infile = h5py.File(filename, 'r')  

        data = np.array(infile['/entry1/analyser/data']).T
        angles = np.array(infile['/entry1/analyser/angles'])
        energies = np.array(infile['/entry1/analyser/energies'])
        hv = np.array(infile['/entry1/instrument/monochromator/energy'])
        
        zscale = energies
        yscale = angles

        # Find which xscale is appropriate
        #"""
        #sapolar : map
        #salong  : combined x & y scan along beam
        #saperp  : combined x & y scan perpendicular to beam
        #"""
        #for z_name in ['salong', 'saperp'] :
        #    index = '/entry1/analyser/{}'.format(z_name)
        #    try :
        #        zscale = np.array(infile[index])
        #    except KeyError :
        #        continue

        # Check if we have a scan
        if data.shape[2] == 1 :
            xscale = energies
            zscale = np.array([0])
            data = data.T
        else :
            # Otherwise, extract third dimension from scan command
            command = infile['entry1/scan_command'][()]

            # Special case for 'pathgroup'
            if command.split()[1] == 'pathgroup' :
                self.print_m('is pathgroup')
                # Extract points from a ([polar, x, y], [polar, x, y], ...) 
                # tuple
                points = command.split('(')[-1].split(')')[0]
                tuples = points.split('[')[1:]
                xscale = []
                for t in tuples :
                    point = t.split(',')[0]
                    xscale.append(float(point))
                xscale = np.array(xscale)

                # Now, if this was a scan with varying centre_energy, the 
                # zscale contains a list of energies...
                # for now, just take the first one
#                zscale = zscale[0]

            # Special case for 'scangroup'
            elif command.split()[1] == 'scan_group' :
                self.print_m('is scan_group')
                # Extract points from a ([polar, x, y], [polar, x, y], ...) 
                # tuple
                points = command.split('((')[-1].split('))')[0]
                points = '((' + points + '))'
                xscale = np.array(ast.literal_eval(points))[:,0]

                # Now, if this was a scan with varying centre_energy, the 
                # zscale contains a list of energies...
                # for now, just take the first one
                zscale = zscale[0]

           # "Normal" case
            else :
                start_stop_step = command.split()[2:5]
                start, stop, step = [float(s) for s in start_stop_step] 
                xscale = np.arange(start, stop+0.5*step, step)

        # What we usually call theta is tilt in this beamline
        theta = infile['entry1/instrument/manipulator/satilt'][0]
        phi = infile['entry1/instrument/manipulator/sapolar'][0]

        # Take the mean of the given binding energies as an estimate
        try :
            E_b = -np.mean(infile['entry1/analyser/binding_energies'])
        except KeyError :
            E_b = -np.mean(infile['entry1/analyser/energies'])

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

class Dataloader_ALS(Dataloader) :
    """ 
    Object that allows loading and saving of ARPES data from the MAESTRO
    beamline at ALS, Berkely, in the newer .h5 format
    Organization of the ALS h5 file (June, 2018)::

        /-0D_Data
        | |
        | +-Cryostat_A
        | +-Cryostat_B
        | +-Cryostat_C
        | +-Cryostat_D
        | +-I0_NEXAFS
        | +-IG_NEXAFS
        | +-X                       <--- only present for xy scans (probably)
        | +-Y                       <--- "
        | +-Sorensen Program        <--- only present for dosing scans
        | +-time
        | 
        +-1D_Data
        | |
        | +-Swept_SpectraN          <--- Not always present
        |
        +-2D_Data
        | |
        | +-Swept_SpectraN          <--- Usual location of data. There can be 
        |                                several 'Swept_SpectraN', each with an 
        |                                increasing value of N. The relevant data 
        |                                seems to be in the highest numbered 
        |                                Swept_Spectra.
        |
        +-Comments
        | |
        | +-PreScan
        |
        +-Headers
          |
          +-Beamline
          | |
          | +-[...]
          | +-EPU_POL               <--- Polarization (Integer encoded)
          | +-BL_E                  <--- Beamline energy (hv)
          | +-[...]
          | 
          +-Computer
          +-DAQ_Swept
          | |
          | +-[...]
          | +-SSPE_0                <--- Pass energy (eV)
          | +-[...]
          | |
          +-FileFormat
          +-Low_Level_Scan
          +-Main
          +-Motors_Logical
          +-Motors_Logical_Offset
          +-Motors_Physical
          +-Motors_Sample           <--- Contains sample coordinates (xyz & angles)
          | |
          | +-[...]
          | +-SMOTOR3               <--- Theta
          | +-SMOTOR5               <--- Phi
          | +-[...]
          |
          +-Motors_Sample_Offset
          +-Notebook
    """
    name = 'ALS'

    def get(self, group, field_name) :
        """
        Return the value of property *field_name* from h5File group *group*.
        *field_name* must be a bytestring (e.g. ``b'some_string'``)!
        This also returns a bytes object, remember to cast it correctly.
        Returns *None* if *field_name* was not found in *group*.
        """
        for entry in group[()] :
            if entry[1].strip() == field_name :
                return entry[2]

    def load_data(self, filename) :
        h5file = h5py.File(filename, 'r')
        # The relevant data seems to be the highest numbered "Swept_SpectraN" 
        # entry in "2D_Data". Find the highest N:
        Ns = [int(spectrum[-1]) for spectrum in list(h5file['2D_Data'])]
        i = np.argmax(Ns)
        path = '2D_Data/Swept_Spectra{}'.format(Ns[i])

        # Convert the data to a numpy array
        self.print_m('Converting data to numpy array (this may take a '
                     'while)...')
        data = np.array(h5file[path])

        # Get hv
        hv = self.get(h5file['Headers/Beamline/'], b'BL_E')
        hv = float(hv)

        # Build x-, y- and zscales
        # TODO Detect cuts (i.e. factual 2D Datasets)
        nz, ny, nx = data.shape
        # xscale is the outer loop (if present)
        x_group = h5file['Headers/Low_Level_Scan']
        x_start = self.get(x_group, b'ST_0_0')
        x_end = self.get(x_group, b'EN_0_0')
        if x_start is not None :
            xscale = np.linspace(float(x_start), float(x_end), nx)
        else :
            xscale = np.arange(nx)
        # y- and zscale are the axis of one spectrum (i.e. pixels and energies)
        attributes = h5file[path].attrs
        y_start, z_start = attributes['scaleOffset']
        y_step, z_step = attributes['scaleDelta']
        yscale = start_step_n(y_start, y_step, ny)
        zscale = start_step_n(z_start, z_step, nz)
#        zscale, yscale, xscale = [np.arange(s) for s in shape]

        # TODO pixel to angle conversion
        yscale *= 0.193/2

        # Get theta and phi
        theta = float(self.get(h5file['Headers/Motors_Sample'], b'SMOTOR3'))
        phi = float(self.get(h5file['Headers/Motors_Sample'], b'SMOTOR5'))
        
        res = Namespace(data=data,
                        xscale=xscale,
                        yscale=yscale,
                        zscale=zscale,
                        angles=yscale,
                        theta=theta,
                        phi=phi,
                        E_b=0,
                        hv=hv)
        return res

class Dataloader_ALS_fits(Dataloader) :
    """ 
    Object that allows loading and saving of ARPES data from the MAESTRO
    beamline at ALS, Berkely, which is in .fits format. 
    """
    name = 'ALS .fits'
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
#        self.print_m(*data.shape)
#        self.print_m(nx, ny, nz)
        scales = [zscale, yscale, xscale]
        for i,scale in enumerate(scales) :
            try :
                length = len(scale)
            except Exception :
                self.print_m('length problem')
                continue
#            self.print_m(length)
            n = data.shape[i]
            if length != n :
                self.print_m(('Shape mismatch in dim {}: {} != {}. Setting ' +
                              'scale to arange.').format(i, length, n))
                scales[i] = np.arange(n)
        zscale, yscale, xscale = scales

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

        if filename.endswith('h5') :
            return self.load_h5(filename)
        elif filename.endswith('zip') :
            return self.load_zip(filename)
        else :
            raise NotImplementedError('File suffix not supported.')

    def load_zip(self, filename) :
        """ Load and store a deflector mode file from SIS-ULTRA. """
        # Prepare metadata key-value pairs for the different metadata files
        # and their expected types
        keys1 = [
                 ('width', 'n_energy', int),
                 ('height', 'n_x', int),
                 ('depth', 'n_y', int),
                 ('first_full', 'first_energy', int),
                 ('last_full', 'last_energy', int),
                 ('widthoffset', 'start_energy', float),
                 ('widthdelta', 'step_energy', float),
                 ('heightoffset', 'start_x', float),
                 ('heightdelta', 'step_x', float), 
                 ('depthoffset', 'start_y', float),
                 ('depthdelta', 'step_y', float)
                ]
        keys2 = [('Excitation Energy', 'hv', float)]

        # Load the zipfile
        with zipfile.ZipFile(filename, 'r') as z :
            # Get the created filename from the viewer
            with z.open('viewer.ini') as viewer :
                file_id = self.read_viewer(viewer)
            # Get most metadata from a metadata file
            with z.open('Spectrum_' + file_id + '.ini') as metadata_file :
                M = self.read_metadata(keys1, metadata_file)
            # Get additional metadata from a second metadata file...
            with z.open(file_id + '.ini') as metadata_file2 :
                M2 = self.read_metadata(keys2, metadata_file2)
            # Extract the binary data from the zipfile
            with z.open('Spectrum_' + file_id + '.bin') as f :
                data_flat = np.frombuffer(f.read(), dtype='float32')
        # Put the data back into its actual shape
        data = np.reshape(data_flat, (int(M.n_y), int(M.n_x), int(M.n_energy)))
        # Cut off unswept region
        data = data[:,:,M.first_energy:M.last_energy+1]
        # Put into shape (energy, other angle, angle along analyzer)
        data = np.moveaxis(data, 2, 0)

        # Create axes
        xscale = start_step_n(M.start_x, M.step_x, M.n_x)
        yscale = start_step_n(M.start_y, M.step_y, M.n_y)
        energies = start_step_n(M.start_energy, M.step_energy, M.n_energy)
        energies = energies[M.first_energy:M.last_energy+1]

        res = Namespace(
            data = data,
            xscale = xscale,
            yscale = yscale,
            zscale = energies,
            hv = M2.hv
        )
        return res
        
    def read_viewer(self, viewer) :
        """ Extract the file ID from a SIS-ULTRA deflector mode output file. """
        for line in viewer.readlines() :
            l = line.decode('UTF-8')
            if l.startswith('name') :
                # Make sure to split off unwanted whitespace
                return l.split('=')[1].split()[0]

    def read_metadata(self, keys, metadata_file) :
        """ Read the metadata from a SIS-ULTRA deflector mode output file. """
        # List of interesting keys and associated variable names
        metadata = Namespace()
        for line in metadata_file.readlines() :
            # Split at 'equals' sign
            tokens = line.decode('utf-8').split('=')
            for key, name, dtype in keys :
                if tokens[0] == key :
                    # Split off whitespace or garbage at the end
                    value = tokens[1].split()[0]
                    # And cast to right type
                    value = dtype(value)
                    metadata.__setattr__(name, value)
        return metadata

    def load_h5(self, filename) :
        """ Load and store the full h5 file and extract relevant information. """
        # Load the hdf5 file
        self.datfile = h5py.File(filename, 'r')
        # Extract the actual dataset and some metadata
        h5_data = self.datfile['Electron Analyzer/Image Data']
        attributes = h5_data.attrs

        # Convert to array and make 3 dimensional if necessary
        shape = h5_data.shape
        # Access data chunk-wise, which is much faster.
        # This improvement has been contributed by Wojtek Pudelko and makes data 
        # loading from SIS Ultra orders of magnitude faster!
        if len(shape) == 3:
            data = np.zeros(shape)
            for i in range(shape[2]):
                data[:, :, i] = h5_data[:, :, i]
        else:
            data = np.array(h5_data)
        # How the data needs to be arranged depends on the scan type: cut, 
        # map, hv scan or a sequence of cuts
        # Case cut
        if len(shape) == 2 :
            x = shape[0]
            y = shape[1]
            # Make data 3D
            data = data.reshape(1, x, y)
#            N_E = y
            N_E = 1
            # Extract the limits
            xlims = attributes['Axis1.Scale']
            ylims = attributes['Axis0.Scale']
#            elims = ylims
            elims = [1, 1]
        # shape[2] should hold the number of cuts. If it is reasonably large, 
        # we have a map. Otherwise just a sequence of cuts.
        # Case map
        elif shape[2] > self.min_cuts_for_map :
            self.print_m('Is a map?')
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
        xscale = start_step_n(*xlims, y)
        yscale = start_step_n(*ylims, x)
        energies = start_step_n(*elims, N_E)
#        xscale = self.make_scale(xlims, y)
#        yscale = self.make_scale(ylims, x)
#        energies = self.make_scale(elims, N_E)

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

#    def make_scale(self, limits, nstep) :
#        """ 
#        Helper function to construct numbers starting from limits[0] 
#        and going in steps of limits[1] for nstep steps.
#        """
#        start = limits[0]
#        step = limits[1]
#        end = start + (nstep+1)*step
#        return np.linspace(start, end, nstep)

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

    # Possible scantypes
    HV = 'hv'
    FSM = 'FSM'

    def load_data(self, filename) :
        """ 
        Single cuts are stored as two files: One file contians the data and 
        the other the metadata. Maps, hv scans and other *external 
        loop*-scans are stored as a directory containing these two files for 
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
        at CASSIOPEE. The dir is assumed to contain two files for each cut::

            BASENAME_INDEX_i.txt     -> beamline related metadata
            BASENAME_INDEX_ROI1_.txt -> data and analyzer related metadata

        To be more precise, the assumptions made on the filenames in the 
        directory are:

            * the INDEX is surrounded by underscores (`_`) and appears after 
              the first underscore.
            * the string ``ROI`` appears in the data filename.
        """
        # Get the all filenames in the dir
        all_filenames = os.listdir(dirname)
        # Remove all non-data files
        filenames = []
        for name in all_filenames :
            if 'ROI' in name :
                filenames.append(name)

        # Get metadata from first file in list
        skip, energy, angles = self.get_metadata(dirname+filenames[0]) 

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
        for i in range(i_min, i_max+1) :
            data.append(unordered[i])
        data = np.array(data)

        # Get the z-axis from the metadata files
        scantype, outer_loop, hv = self.get_outer_loop(dirname, filenames) 
        self.print_m('Scantype: {}'.format(scantype))
        if scantype == self.HV :
            yscale = energy
            zscale = outer_loop
            hv = outer_loop
        elif scantype == self.FSM :
            yscale = outer_loop
            zscale = energy
            # For a map, we expect output of the form (Energy, k_para, 
            # k_perp), currently it is (tilt, energy, theta) -> reshape 
            # with moveaxis
            data = np.moveaxis(data, 0, 1)
        else :
            yscale = energy
            zscale = np.arange(data.shape[0])
        xscale = angles

        res = Namespace(
            data = data,
            xscale = xscale,
            yscale = yscale,
            zscale = zscale,
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
        
        # NOTE Unreliable hv
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
        i, energy, angles = self.get_metadata(filename)
        self.print_m('Loading from txt')
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
            hv = 1)
        return res

    def get_metadata(self, filename) :
        """ 
        Extract some of the metadata stored in a CASSIOPEE output text file. 
        Also try to detect the line number below which the data starts (for 
        np.loadtxt's skiprows.)

        **Returns**

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
                    # NOTE this hv does not reflect the actually used hv
#                    hv = float(line.split('=')[-1])
                    pass
                elif line.startswith('inputA') or line.startswith('[Data') :
                    # this seems to be the last line before the data
                    break
        return i, energy, angles

    def get_outer_loop(self, dirname, filenames) :
        """
        Try to determine the scantype and the corresponding z-axis scale from 
        the additional metadata textfiles. These follow the assumptions made 
        in :meth:`self.load_from_dir 
        <arpys.dataloaders.Dataloader_CASSIOPEE.load_from_dir>`. 
        Additionally, the MONOCHROMATOR section must come before the 
        UNDULATOR section as in both sections we have a key `hv` but only the 
        former makes sense.
        Return a string for the scantype, the extracted z-scale and the value 
        for hv for non-hv-scans (scantype, zscale, hvs[0]) or (None, 
        None, hvs[0]) in case of failure.
        """
        # Step 1) Extract metadata from metadata file
        # Prepare containers
        indices, xs, ys, zs, thetas, phis, tilts, hvs = ([], [], [], [], [], 
                                                         [], [], [])
        containers = [indices, xs, ys, zs, thetas, phis, tilts, hvs] 
        for name in filenames :
            # Get the index of the file
            index = int(name.split('_')[1])

            # Build the metadata-filename by substituting the ROI part with i
            metafile = re.sub(r'_ROI.?_', '_i', name)

            # The values are separated from the names by a `:`
            splitchar = ':'

            # Read in the file
            with open(dirname + metafile, 'r') as f :
                for line in f.readlines() :
                    if line.startswith('x (mm)') :
                        x = float(line.split(splitchar)[-1])
                    elif line.startswith('y (mm)') :
                        y = float(line.split(splitchar)[-1])
                    elif line.startswith('z (mm)') :
                        z = float(line.split(splitchar)[-1])
                    elif line.startswith('theta (deg)') :
                        theta = float(line.split(splitchar)[-1])
                    elif line.startswith('phi (deg)') :
                        phi = float(line.split(splitchar)[-1])
                    elif line.startswith('tilt (deg)') :
                        tilt = float(line.split(splitchar)[-1])
                    elif line.startswith('hv (eV)') :
                        hv = float(line.split(splitchar)[-1])
                    elif line.startswith('UNDULATOR') :
                        break
            # NOTE The order of this list has to match the order of the 
            # containers
            values = [index, x, y, z, theta, phi, tilt, hv]
            for i,container in enumerate(containers) :
                container.append(values[i])

        # Step 2) Check which parameters vary to determine scantype
        if hvs[1] != hvs[0] :
            scantype = self.HV
            zscale = hvs
        elif thetas[1] != thetas[0] :
            scantype = self.FSM
            zscale = thetas
        else :
            scantype = None
            zscale = None

        # Step 3) Put zscale in order and return
        if zscale is not None :
            zscale = np.array(zscale)[np.argsort(indices)]

        return scantype, zscale, hvs[0]

# +-------+ #
# | Tools | # ==================================================================
# +-------+ #

# List containing all reasonably defined dataloaders
all_dls = [
           Dataloader_SIS,
           Dataloader_ADRESS,
           Dataloader_i05,
           Dataloader_CASSIOPEE,
           Dataloader_ALS,
           Dataloader_ALS_fits,
           Dataloader_Pickle
          ]

# Function to try all dataloaders in all_dls
def load_data(filename, exclude=None, suppress_warnings=False) :
    """
    Try to load some dataset *filename* by iterating through `all_dls` 
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
        if suppress_warnings :
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

            # Reaching this point must mean we succeeded. Print warnings from 
            # this dataloader, if any occurred
            print('[arpys]Loaded data with {}.'.format(dl))
            try :
                print(dl, ': ', exceptions[dl])
            except KeyError :
                pass
            
            return namespace

    # Reaching this point means something went wrong. Print all exceptions.
    for dl in exceptions :
        print('[arpys]', dl)
        e = exceptions[dl]
        print('[arpys]Exception {}: {}'.format(type(e), e))

    raise Exception('Could not load data {}.'.format(filename))

# Function to create a python pickle file from a data namespace
def dump(D, filename, force=False) :
    """ Wrapper for :func:`pickle.dump`. Does not overwrite if a file of 
    the given name already exists, unless *force* is True.

    **Parameters**

    ========  ==================================================================
    D         python object to be stored.
    filename  str; name of the output file to create.
    force     boolean; if True, overwrite existing file.
    ========  ==================================================================
    """
    # Check if file already exists
    if not force and os.path.isfile(filename) :
        question = 'File <{}> exists. Overwrite it? (y/N)'.format(filename)
        answer = input(question)
        # If the answer is anything but a clear affirmative, stop here
        if answer.lower() not in ['y', 'yes'] :
            return

    with open(filename, 'wb') as f :
        pickle.dump(D, f)

    message = 'Wrote to file <{}>.'.format(filename)
    print(message)

def load_pickle(filename) :
    """ Shorthand for loading python objects stored in pickle files.

    **Parameters**

    ========  ==================================================================
    filename  str; name of file to load.
    ========  ==================================================================
    """
    with open(filename, 'rb') as f :
        return pickle.load(f)

def update_namespace(D, *attributes) :
    """ Add arbitrary attributes to a :class:`Namespace <argparse.Namespace>`.

    **Parameters**

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
    :func:`update_namespace <arpys.dataloaders.update_namespace>` and writes 
    back to file.
  
    **Parameters**

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
    D = Dataloader_SIS().load_data('/home/kevin/qmap/experiments/2020_09_SIS/PrFeP_3P/PrFeP_3P_0001.zip')
    print(D.data.shape)
    print(D.xscale.shape)
