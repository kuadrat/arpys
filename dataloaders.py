import h5py
import numpy as np
import pickle
import pyfits

class Dataloader_Pickle() :
    """ Load data that has been saved using python's `pickle` module. Usually 
    this will be just rare arrays, so determin the shape and such from the 
    array itself. """

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

class Dataloader_ALS() :
    """ Object that allows loading and saving of ARPES data from the  
    beamline at ALS, Berkely which is in .fits format. """

    def load_data(self, filename) :
        # Open the file
        hdulist = pyfits.open(filename)

        # Access the BinTableHDU
        self.bintable = hdulist[1]

        # Get the header to extract metadata from
        header = hdulist[0].header

        # Find out what scan mode we're dealing with
        scanmode = hdulist[0].header['NM_0_0']

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
            energies_in_px = np.arange(p0, p1, 2)
            energies = (energies_in_px - fermi_level)/ppeV

            # Starting and ending pixels (angle scale)
            p0 = header['SSY0_0']
            p1 = header['SSY1_0']

            # Use this arbitrary seeming conversion factor (found in Denys' 
            # script) to get from pixels to angles
            angles_in_px = np.arange(p0, p1, 2)
            angles = angles_in_px * 0.193/2

            xscale = angles
            yscale = energies
        else :
            xscale = None
            yscale = None

        res = {
               'data': data,
               'xscale': xscale,
               'yscale': yscale
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

class Dataloader_PSI() :
    """ Object that allows loading and saving of ARPES data from the SIS 
    beamline at PSI which is in hd5 format. """

    def __init__(self, filename=None) :
        """ Allow user to (optionally) already load a file on initialization. 
        :UNUSED:
        """
        if filename is not None :
            self.load_file(filename)
            

    def load_file(self, filename) :
        """ Load and store the full h5 file. """
        # Load the hdf5 file
        self.data = self.datfile = h5py.File(filename, 'r')

    def load_data(self, filename) :
        """ Extract and return the actual 'data', i.e. the recorded map/cut. 
        Also return labels which provide some indications what the data means.
        """
        self.load_file(filename)

        # Extract the actual dataset
        h5_data = self.datfile['Electron Analyzer/Image Data']

        # Convert to array and make 3 dimensional if necessary
        data = np.array(h5_data)
        shape = data.shape
        if len(shape) == 2 :
            x = shape[0]
            y = shape[1]
            data = data.reshape(1, x, y)
        else :
            x = shape[1]
            y = shape[2]

        # Get x and y axis scales
        # Note: x and y are a bit confusing here as the hd5 file has a 
        # different notion of zero'th and first dimension as numpy and then 
        # later pcolormesh introduces yet another layer of confusion. The way 
        # it is written now, though hard to read, turns out to do the right 
        # thing and leads to readable code from after this point.
        attributes = h5_data.attrs

        # Special case if we are dealing with a map (3 dimensional data)
        if 'Axis2.Scale' in attributes :
            xlims = attributes['Axis2.Scale']
            ylims = attributes['Axis1.Scale']
        else :
            xlims = attributes['Axis1.Scale']
            ylims = attributes['Axis0.Scale']

        xscale = np.linspace(xlims[0], xlims[1], y)
        yscale = np.linspace(ylims[0], ylims[1], x)

        res = {
               'data': data,
               'xscale': xscale,
               'yscale': yscale
              }

        return res
