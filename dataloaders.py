"""
Provides several Dataloader objects which open different kinds of data files 
- typically acquired at different sources - and crunch them into the same 
shape and form.
"""

import h5py
import numpy as np
import pickle
import pyfits

class Dataloader() :
    """ Base dataloader class (interface) from which others inherit some 
    methods (specifically the __repr__() function). """
    name = 'Base'

    def __init__(self, *args, **kwargs) :
        pass

    def __repr__(self, *args, **kwargs) :
        return '<class Dataloader_{}>'.format(self.name)

class Dataloader_Pickle(Dataloader) :
    """ Load data that has been saved using python's `pickle` module. Usually 
    this will be just rare arrays, so determin the shape and such from the 
    array itself. 
    """
    name = 'Pickle'

    def load_data(self, filename) :
        # Print a warning
        import warnings
        warnings.warn('Pickle dataloader is in experimental stage.')

        # Open the file and get a handle for it
        with open(filename, 'rb') as f :
            filedata = pickle.load(f)

        # Check if we are dealing with an array
        if type(filedata) != np.ndarray :
            raise TypeError

        # Get the dimensions of the array
        shape = filedata.shape
        x = shape[0]
        y = shape[1]

        # Create x and y scales from shape
        xscale = np.arange(x)
        yscale = np.arange(y)

        # Bring the data in the right shape
        data = filedata.reshape(1, x, y)

        # Create and return the datadict
        res = {
                'data': data,
                'xscale': yscale,
                'yscale': xscale
        }
        return res

class Dataloader_ALS(Dataloader) :
    """ Object that allows loading and saving of ARPES data from the  
    beamline at ALS, Berkely which is in .fits format. 
    """
    name = 'ALS'

    def __init__(self, work_func=4) :
        # Assign a value for the work function
        self.work_func = work_func

    def load_data(self, filename) :
        # Open the file
        hdulist = pyfits.open(filename)

        # Access the BinTableHDU
        self.bintable = hdulist[1]

        # Get the header to extract metadata from
        header = hdulist[0].header

        # Find out what scan mode we're dealing with
        scanmode = header['NM_0_0']

        # Find out whether we are in fixed or swept (dithered) mode which has 
        # an influence on how the data is stored in the fits file.
        swept = True if ('SSSW0' in header) else False
        
        # Case normal cut
        if scanmode == 'null' :
            data = self.load_cut()
        # Case FSM
        elif scanmode == 'Slit Defl' :
            data = self.load_map(swept)
        # Case hv scan
        elif scanmode == 'mono_eV' :
            data = self.load_hv_scan()
        else :
            raise(IndexError('Couldn\'t determine scan type'))

        if scanmode != 'Slit Defl' :
            # Starting pixel (energy scale)
            p0 = header['SSX0_0']
            # Ending pixel (energy scale)
            p1 = header['SSX1_0']
            # Pixels per eV
            ppeV = header['SSPEV_0']
            # Zero pixel (=Fermi level?)
            fermi_level = header['SSKE0_0']

            # Create the energy (y) scale
            # NOTE: Not sure whether these energies are appropriate
            energy_binning = 2
            energies_in_px = np.arange(p0, p1, energy_binning)
            energies = (energies_in_px - fermi_level)/ppeV

            # Starting and ending pixels (angle scale)
            p0 = header['SSY0_0']
            p1 = header['SSY1_0']

            # Use this arbitrary seeming conversion factor (found in Denys' 
            # script) to get from pixels to angles
            angle_binning = 2
            angles_in_px = np.arange(p0, p1, angle_binning)
            c1 = 0.193
            c2 = 4
            angles = angles_in_px * c1/c2

            xscale = angles
            yscale = energies
            # For the binding energy, just take a min value as its variations 
            # are small compared to the photon energy
            E_b = energies.min()
        else :
            xscale = None
            yscale = None
            angles = None

        # Extract some additional metadata (mostly for angles->k conversion)
        # TODO Special cases for different scan types
        # Get relevant metadata from header
        theta = header['LMOTOR3']
        phi = header['LMOTOR5']

        # The photon energy may vary in the case that this is a hv_scan
        hv = header['MONO_E']

        # NOTE angles==xscale and E_b can be extracted from yscale, thus it 
        # is not really necessary to pass them on here.
        res = {
               'data': data,
               'xscale': xscale,
               'yscale': yscale,
               'angles': angles,
               'theta': theta,
               'phi': phi,
               'E_b': E_b,
               'hv': hv
              }
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
        """ Read data from a 'map', i.e. several energy vs k slices and bring 
        them in the right shape for the gui, which is 
        (energy, k_parallel, k_perpendicular)
        """
        bt_data = self.bintable.data
        n_slices = len(bt_data)

        first_slice = bt_data[0][-1]
        if swept :
            n_energy, n_kx = first_slice.shape    
            data = np.zeros([n_energy, n_kx, n_slices])
            for i in range(n_slices) :
                this_slice = bt_data[i][-1]
                data[:,:,i] = this_slice
        else :
            n_kx, n_energy = first_slice.shape
            
            # The shape of the returned array must be (energy, kx, ky)
            # (n_slices corresponds to ky)
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
        """ Read data from a hv scan, i.e. a series of energy vs k cuts 
        (shape (n_kx, n_energy), each belonging to a different photon energy 
        hv. The returned shape in this case must be 
        (photon_energies, energy, k) 
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

class Dataloader_PSI(Dataloader) :
    """ Object that allows loading and saving of ARPES data from the SIS 
    beamline at PSI which is in hd5 format. 
    """
    name = 'PSI'
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
        """ Extract and return the actual 'data', i.e. the recorded map/cut. 
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
            new_data = np.zeros([z, x, y])
            for i in range(z) :
                cut = data[:,:,i]
                new_data[i] = cut
            data = new_data
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

        res = {
               'data': data,
               'xscale': xscale,
               'yscale': yscale,
               'angles': angles,
               'theta': theta,
               'phi': phi,
               'E_b': E_b,
               'hv': hv
              }

        return res

    def make_scale(self, limits, nstep) :
        """ Helper function to construct numbers starting from limits[0] 
        and going in steps of limits[1] for nstep steps.
        """
        start = limits[0]
        step = limits[1]
        end = start + (nstep+1)*step
        return np.linspace(start, end, nstep)

# +-------+ #
# | Tools | # ==================================================================
# +-------+ #

# List containing all reasonably defined dataloaders
all_dls = [
           Dataloader_PSI,
           Dataloader_ALS
          ]

# Function to try all dataloaders in all_dls
def load_data(filename, exclude=None) :
    """ Pass """
    # If only a single string is given as exclude, pack it into a list
    if exclude is not None and type(exclude)==str :
        exclude = [exclude]
    
    # Keep track of all exceptions in case no loader succeeds
    exceptions = dict()

    for dataloader in all_dls :
        # Instantiate a dataloader object
        dl = dataloader()

        # Skip to the next if this dl is excluded (continue brings us back to 
        # the top of the loop, starting with the next element)
        if exclude is not None and dl.name in exclude : 
            continue

        # Try loading the data
        try :
            datadict = dl.load_data(filename)
        except Exception as e :
            # Temporarily store the exception
            exceptions.update({dl : e})
            # Try the next dl
            continue

        # Reaching this point must mean we succeeded
        return datadict

    # Reaching this point means something went wrong. Print all exceptions.
    for dl in exceptions :
        print(dl)
        print(exceptions[dl])

    raise Exception('Could not load data {}.'.format(filename))

        
        
