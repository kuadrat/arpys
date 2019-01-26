#!/usr/bin/python
""" 
Contains different tools to post-process (ARPES) data. 
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patheffects import withStroke
from scipy import ndimage

from arpys.utilities import constants

# +------------+ #
# | Decorators | # ============================================================
# +------------+ #

#def dc(f) :
#    """ Decorator which converts input to a numpy array. """
#    def decorated(data, *args, **kwargs) :
#        data = np.array(data)
#        return f(data, *args, **kwargs)
#    return decorated
#
#def reshape(D) :
#    """ Decorator factory which converts the input data array either to shape
#    (l x m) (D=2) or to (l x m x 1) (D=3)
#    """
#    def decorator(f) :
#        def decorated(data, *args, **kwargs) :
#            # Find outher whether we have a (l x m) (d=2) or a (l x m x 1) 
#            # (d=3) array
#            shape = data.shape
#            d = len(shape)
#
#            # Convert to shape (l x m x 1) 
#            if d == D :
#                l = shape[0]
#                m = shape[1]
#                if D == 2 :
#                    data = data.reshape(l, m)
#                elif D == 3 :
#                    data = data.reshape(l, m, 1)
#
#            return f(data, *args, **kwargs)
#        return decorated
#    return decorator

# +------------------+ #
# | ARPES processing | # ======================================================
# +------------------+ #

def make_slice(data, d, i, integrate=0) :
    """ Create a slice out of the 3d data (l x m x n) along dimension d 
    (0,1,2) at index i. Optionally integrate around i.

    *Parameters*
    ============================================================================
    data       array-like; map data of the shape (l x m x n) where l 
               corresponds to the number of energy values
    d          int, d in (0, 1, 2); dimension along which to slice
    i          int, 0 <= i < data.size[d]; The index at which to create the slice
    integrate  int, 0 <= integrate < |i - n|; the number of slices above 
               and below slice i over which to integrate
    ============================================================================

    *Returns*
    ============================================================================
    res        np.array; Slice at index with dimensions shape[:d] + shape[d+1:]
               where shape = (l, m, n).
    ============================================================================
    """
    # Get the relevant dimensions
    shape = data.shape
    try :
        n_slices = shape[d]
    except IndexError :
        print('d ({}) can only be 0, 1 or 2 and data must be 3D.'.format(d))
        return

    output_shape = shape[:d] + shape[d+1:]

    # Set the integration indices and adjust them if they go out of scope
    start = i - integrate
    stop = i + integrate + 1
    if start < 0 :
        warnings.warn(
        'i - integrate ({}) < 0, setting start=0'.format(start))       
        start = 0
    if stop > n_slices :
        warning = ('i + integrate ({}) > n_slices ({}), setting '
                   'stop=n_slices').format(stop, n_slices)       
        warnings.warn(warning)
        stop = n_slices

    # Initialize data container and fill it with data from selected slices
    sliced = np.zeros(output_shape)
    if d == 0 :
        for i in range(start, stop, 1) :
            sliced += data[i, :, :]
    elif d == 1 :
        for i in range(start, stop, 1) :
            sliced += data[:, i, :]
    elif d == 2 :
        for i in range(start, stop, 1) :
            sliced += data[:, :, i]

    return sliced

def make_map(data, i, integrate=0) :
    """ Create a 'top view' slice for FSM data.
    If the values of i or integrate are bigger than what is possible, they 
    are automatically reduced to the maximum possible.

    *Parameters*
    ============================================================================
    data       array-like; map data of the shape (l x m x n) where l 
               corresponds to the number of energy values
                
    i          int, 0 <= i < n; The index at which to create the slice
    integrate  int, 0 <= integrate < |i - n|; the number of slices above 
               and below slice i over which to integrate
    ============================================================================

    *Returns*
    ============================================================================
    res        np.array; Map at given energy with dimensions (m x n)
    ============================================================================

    .. :see also:
        `make_slice <arpys.postprocessing.make_slice>`. `make_map` is 
        basically a special case of `make_slice`.
    """
    # Prepare the start and stop indices for slicing of the map data
    l, m, n = data.shape
    if i >= l :
        warnings.warn('i ({}) >= l ({}), setting i=l-1.'.format(i, l))       
        i = l-1

    start = i - integrate
    stop = i + integrate + 1
    if start < 0 :
        warnings.warn(
        'i - integrate ({}) < 0, setting start=0.'.format(start))       
        start = 0
    if stop > l :
        warnings.warn(
        'i + integrate ({}) > l ({}), setting stop=l.'.format(stop, l))       
        stop = l

    # Initialize data container and fill it with data from selected slices
    fsm = np.zeros([m, n])
    for i in range(start, stop, 1) :
        fsm += data[i, :, :]

    return fsm

# +---------------+ #
# | Normalization | # ========================================================
# +---------------+ #

def normalize_globally(data, minimum=True) :
    """ The simplest approach: normalize the whole dataset by the global min- 
    or maximum.

    *Parameters*
    ============================================================================
    data     array-like; the input data of arbitrary dimensionality
    minimum  boolean; if True, use the min, otherwise the max function
    ============================================================================

    *Returns*
    ============================================================================
    res      np.array; normalized version of input data
    ============================================================================
    """
    # Select whether to use the minimum or the maximum for normalization
    min_or_max = 'min' if minimum else 'max'

    # Retrieve the global maximum
    m = data.__getattribute__(min_or_max)()

    # Return the normalized values
    return data/m

def convert_data(data) :
    """ Helper function to convert data to the right shape. """
    # Find out whether we have a (m x n) (d=2) or a (1 x m x n) (d=3) array
    shape = data.shape
    d = len(shape)

    # Convert to shape (1 x m x n) 
    if d == 2 :
        m = shape[0]
        n = shape[1]
        data = data.reshape(1, m, n)
    elif d == 3 :
        m = shape[1]
        n = shape[2]
    else :
        raise ValueError('Could not bring data with shape {} into right \
                         form.'.format(shape))
    return data, d, m, n

def convert_data_back(data, d, m, n) :
    """ Helper function to convert data back to the original shape which is 
    determined by the values of d, m and n (outputs of :func: convert_data).
    """
    if d == 2 :
        data = data.reshape(m, n)
    return data

def normalize_per_segment(data, dim=0, minimum=False) :
    """ Normalize each column/row by its respective max value.

    *Parameters*
    ============================================================================
    data     array-like; the input data with shape (m x n) or 
             (1 x m x n)
    dim      int; along which dimension to normalize (0 or 1)
    minimum  boolean; if True, use the min, otherwise the max function
    ============================================================================

    *Returns*
    ============================================================================
    res      np.array; normalized version of input data in same shape
    ============================================================================
    """
    # Select whether to use the minimum or the maximum for normalization
    min_or_max = 'min' if minimum else 'max'

    # Convert data if necessary
    data, d, m, n = convert_data(data)

    # Determine the length of the dimension along which to normalize
    length = data.shape[dim+1]

    # Get a reference to the respective row and divide it by its max 
    # value. This changes the values in the data array.
    for i in range(length) :
        if dim == 0 :
            row = data[:,i,:] 
        elif dim == 1 :
            row = data[:,:,i] 

        row /= row.__getattribute__(min_or_max)()

    # Convert back to original shape, if necessary
    convert_data_back(data, d, m, n)

    return data

def normalize_per_integrated_segment(data, dim=0, profile=False, 
                                     in_place=True) :
    """ Normalize each MDC/EDC by its integral.

    *Parameters*
    ============================================================================
    data      array-like; the input data with shape (m x n) or 
              (1 x m x n).
    dim       int; along which dimension to normalize (0 or 1)
    profile   boolean; if True return a tuple (res, norm) instead of just 
                  res.
    in_place  boolean; whether or not to update the input data in place. 
              This can be used if one is only interested in the 
              normalization profile and does not want to spend 
              computation time with actually changing the data (as might 
              be the case when processing FSMs). If this is False `data` 
              will not be in the output.
              TODO This doesn't make sense.
    ============================================================================

    *Returns*
    ============================================================================
    res       np.array; normalized version of input data in same shape. 
              Only given if `in_place` is True.
    norms     np.array; 1D array of length X for dim=0 and Y for dim=1 of 
              normalization factors for each channel. Only given if 
              `profile` is True.
    ============================================================================
    """
    # Convert data if necessary
    data, d, m, n = convert_data(data)

    # Determine the length of the dimension along which to normalize
    length = data.shape[dim+1]

    # Prepare a container for the normalization factors
    if profile :
        norms = np.zeros(length)

    # Get a reference to the respective row and divide it by its max 
    # value.
    for i in range(length) :
        if dim == 0 :
            row = data[:,i,:] 
        elif dim == 1 :
            row = data[:,:,i] 

        # Integate
        integral = sum(row[0])

        if profile : 
            norms[i] = integral

        # Nothing else to do if `in_place` is False
        if not in_place :
            continue

        # Update the data values in-place
        # Suppress the warnings printed when division by zero happens in rows 
        # of zeros
        with np.errstate(invalid='raise') :
            try :
                # Using augmented assignment (row/=integral instead of 
                # row=row/integral) would lead to nan's being written into 
                # row, as the array is updated in-place - even though the 
                # exception is caught!
                row = row / integral
            except FloatingPointError as e :
                # This error occurs when all values in the row and, 
                # consequently, its integral are 0. Just leave the row 
                # unchanged in this case 
                #print(e)
                pass

        # Copy the values back into the original data array
        if dim == 0 :
            data[:,i,:] = row
        elif dim == 1 :
            data[:,:,i] = row

    # Convert back to original shape, if necessary
    data = convert_data_back(data, d, m, n)

    # Prepare the output
    return_value = []
    if in_place :
        return_value.append(data)
    if profile :
        return_value.append(norms)

    # If just one object is given, avoid returning a len(1) tuple
    if len(return_value) == 1 :
        return_value = return_value[0]
    else :
        tuple(return_value)

    return return_value

def norm_int_edc(data, profile=False) :
    """ 
    Shorthand for :func: `normalize_per_integrated_segment 
    <arpys.postprocessing.normalize_per_integrated_segment>` with 
    arguments
    `dim=1, profile=False, in_place=True`.
    Returns the normalized array.
    """
    return normalize_per_integrated_segment(data, dim=1, profile=profile)

def normalize_above_fermi(data, ef_index, n_pts=10, dist=0, inverted=False, 
                          dim=1, profile=False, in_place=True) : 
    """ Normalize data to the mean of the n_pts smallest values above the Fermi 
    level.

    *Parameters*
    ============================================================================
    data      array-like; data of shape (m x n) or (1 x m x n)
    ef_index  int; index of the Fermi level in the EDCs
    n         int; number of points above the Fermi level to average over
    dist      int; distance from Fermi level before starting to take points 
              for the normalization. The points taken correspond to 
              EDC[ef_index+d:ef_index+d+n] (in the non-inverted case)
    dim       either 1 or 2; 1 if EDCs have length n, 2 if EDCs have length m
    inverted  boolean; this should be set to True if higher energy values 
              come first in the EDCs
    profile   boolean; if True, the list of normalization factors is returned
              additionally
    ============================================================================

    *Returns*
    ============================================================================
    data      array-like; normalized data of same shape as input data
    profile   1D-array; only returned as a tuple with data (`data, profile`) 
              if argument `profile` was set to True. Contains the 
              normalization profile, i.e. the normalization factor for each 
              channel. Its length is m if dim==2 and l if dim==1.
    ============================================================================
    """
    # Prevent input data from being overwritten by creating a copy and 
    # convert data shape if necessary
    if not in_place :
        data = data.copy()
    data, d, m, n = convert_data(data)

    # Create a mini-function (with lambda) which extracts the right part of 
    # the data, depending on the user input
    if dim==1 :
        get_edc = lambda k : data[0,k]
        n_edcs = m
    elif dim==2 :
        get_edc = lambda k : data[0,:,k]
        n_edcs = n
    else :
        message = '[arpys.pp.normalize_above_fermi]`dim` should be 1 or 2, \
        got {}'.format(dim)
        raise ValueError(message)

    # Create the right mini-function for the extraction of points above the 
    # Fermi level
    if inverted :
        get_above_fermi = lambda edc : edc[:ef_index-dist][::-1]
    else :
        get_above_fermi = lambda edc : edc[ef_index+dist:]

    if profile :
        norms = []

    # Iterate over EDCs
    for k in range(n_edcs) :
        edc = get_edc(k)
        above_ef = get_above_fermi(edc)

        norm = np.mean(above_ef[:n_pts])
        if profile :
            norms.append(norm)

        # This syntax updates the data in-place
        edc /= norm
#    with np.errstate(invalid='raise') :
#        try :
#            # This should update the data in-place
#            edc /= norm
#        except FloatingPointError as e :
#            print('Passing...')
#            pass
    data = convert_data_back(data, d, m, n)

    if profile :
        return data, np.array(norms)
    else :
        return data

def norm_to_smooth_mdc(data, mdc_index, integrate, dim=1, n_box=15, 
                       recursion_level=1) :
    """
    Normalize a cut to a smoothened average MDC of the intensity above the 
    Fermi level.

    *Parameters*
    ============================================================================
    data       array of shape (1 x m x n) or (m x n);
    mdc_index  int; index in data at which to take the mdc
    integrate  int; number of MDCs above and below `mdc_index` over which to 
               integrate
    dim        either 1 or 2; 1 if EDCs have length n, 2 if EDCs have length m
    n_box      int; box size of linear smoother. Confer :func: 
               `<arpys.postprocessing.smooth>`
    recursion_level
               int; number of times to iteratively apply the smoother. Confer
               :func: `<arpys.postprocessing.smooth>`
    ============================================================================

    *Returns*
    ============================================================================
    result     normalized data in same shape
    ============================================================================
    """
    data, d, m, n = convert_data(data)
    i0 = mdc_index - integrate
    i1 = mdc_index + integrate + 1
    if dim==1 :
        mdc = np.sum(data[0,i0:i1], 0)
    elif dim==2 :
        mdc = np.sum(data[0,:,i0:i1], 1)
    
    # Smoothen MDC with a linear box smoother
    smoothened = smooth(mdc, n_box, recursion_level)

    # Build an array of the same shape as the original data to allow fast 
    # array-division
    if dim==1 :
        divisor = [smoothened for i in range(m)]
    if dim==2 :
        divisor = [smoothened for i in range(n)]
        divisor = np.array(divisor).T

    # Add empty dimension to be in right shape (1xlxm)
    divisor = np.array([divisor])
    data /= divisor

    return convert_data_back(data, d, m, n)

# +----------------+ #
# | BG subtraction | # ========================================================
# +----------------+ #

def subtract_bg_fermi(data, n_pts=10, ef=None, ef_index=None) :
    """ Use the mean of the counts above the Fermi level as a background and 
    subtract it from every channel/k. If no fermi level or index of the fermi
    level is specified, do the same as <func> subtract_bg_matt() but along EDCs
    instead of MDCs.

    *Parameters*
    ============================================================================
    data   array-like; the input data with shape (m x n) or (1 x m x n) 
           containing momentum in y (n momentum points (?)) and energy along x 
           (m energy points) (plotting amazingly inverts x and y)
    n_pts  int; number of smallest points to take in order to determine bg
    ============================================================================

    *Returns*
    ============================================================================
    res    np.array; bg-subtracted  version of input data in same shape
    ============================================================================
    """
    # Reshape input
    shape = data.shape
    d = len(shape)
    if d == 3 :
        m = shape[1] 
        n = shape[2]
        data = data.reshape([m, n])
    else :
        m = shape[0]
        n = shape[1]

    if ef_index == None :
    # TODO elif ef:
        if ef == None :
            ef_index = 0

    # Loop over every k
    for k in range(m) :
        edc = data[k]
        above_ef = edc[ef_index:]

        # Order the values in the edc by magnitude
        ordered = sorted(above_ef)

        # Average over the first (smallest) n_pts points to get the bg
        bg = np.mean(ordered[:n_pts])

        # Subtract the background (this updates the data array in place)
        edc -= bg

    if d==3 :
        data = data.reshape([1, m, n])
         
    return data

def subtract_bg_matt(data, n_pts=5, profile=False) :
    """ Subtract background following the method in C.E.Matt's 
    "High-temperature Superconductivity Restrained by Orbital Hybridisation".
    Use the mean of the n_pts smallest points in the spectrum for each energy 
    (i.e. each MDC).

    *Parameters*
    ============================================================================
    data     array-like; the input data with shape (m x n) or (1 x m x n) 
             containing momentum in y (n momentum points) and energy along x 
             (m energy points) (plotting amazingly inverts x and y)
    n_pts    int; number of smallest points to take in each MDC in order to 
             determine bg
    profile  boolean; if True, a list of the background values for each MDC 
             is returned additionally.
    ============================================================================

    *Returns*
    ============================================================================
    res      np.array; bg-subtracted version of input data in same shape
    profile  1D-array; only returned as a tuple with data (`data, profile`) 
             if argument `profile` was set to True. Contains the 
             background profile, i.e. the background value for each MDC.
    ============================================================================
    """
    # Prevent original data from being overwritten by retaining a copy
    data = data.copy()

    # Reshape input
    shape = data.shape
    d = len(shape)
    if d == 3 :
        m = shape[1] 
        n = shape[2]
        data = data.reshape([m,n])
    else :
        m = shape[0]
        n = shape[1]

    # Determine the number of energies l
    #    shape = data.shape
    #    l = shape[1] if len(shape)==3 else shape[0]
    if profile :
        bgs = []

    # Loop over every energy
    for e in range(m) :
        mdc = data[e, :]

        # Order the values in the mdc by magnitude
        ordered = sorted(mdc)

        # Average over the first (smallest) n_pts points to get the bg
        bg = np.mean(ordered[:n_pts])
        if profile :
            bgs.append(bg)

        # Subtract the background (this updates the data array in place)
        mdc -= bg
     
    if d == 3 :
        data = data.reshape([1, m, n])

    if profile :
        return data, np.array(bgs)
    else :
        return data

def subtract_bg_shirley(data, dim=0, profile=False, normindex=0) :
    """ Use an iterative approach for the background of an EDC as described in
    DOI:10.1103/PhysRevB.5.4709. Mathematically, the value of the EDC after 
    BG subtraction for energy E EDC'(E) can be expressed as follows :

                               E1
                               /
        EDC'(E) = EDC(E) - s * | EDC(e) de
                               /
                               E

    where EDC(E) is the value of the EDC at E before bg subtraction, E1 is a 
    chosen energy value (in our case the last value in the EDC) up to which 
    the subtraction is applied and s is chosen such that EDC'(E0)=EDC'(E1) 
    with E0 being the starting value of the bg subtraction (in our case the 
    first value in the EDC).

    In principle, this is an iterative method, so it should be applied 
    repeatedly, until no appreciable change occurs through an iteration. In 
    practice this convergence is reached in 4-5 iterations at most and even a 
    single iteration may suffice.

    *Parameters*
    ============================================================================
    data     np.array; input data with shape (m x n) or (1 x m x n) containing 
             an E(k) cut
    dim      int; either 0 or 1. Determines whether the input is aranged as 
             E(k) (n EDCs of length m, dim=0) or k(E) (m EDCs of length n, dim=1) 
    profile  boolean; if True, a list of the background values for each MDC 
             is returned additionally.
    normindex  TESTING
    ============================================================================

    *Returns*
    ============================================================================
    data     np.array; has the same dimensions as the input array.
    profile  1D-array; only returned as a tuple with data (`data, profile`) 
             if argument `profile` was set to True. Contains the 
             background profile, i.e. the background value for each MDC.
    ============================================================================
    """
    # Prevent original data from being overwritten by retaining a copy
    data = data.copy()

    data, d, m, n = convert_data(data)
    
    if dim == 0 :
        nk = n
        ne = m
        get_edc = lambda k : data[0,:,k]
    elif dim == 1 :
        nk = m
        ne = n
        get_edc = lambda k : data[0,k]

    # Take shirley bg from the angle-averaged EDC
    average_edc = np.mean(data[0], dim+1) 

    # Calculate the "normalization" prefactor
    s = np.abs(average_edc[normindex] - average_edc[-1]) / average_edc.sum()

    # Prepare a function that sums the EDC from a given index upwards
    sum_from = np.frompyfunc(lambda e : average_edc[e:].sum(), 1, 1)
    indices = np.arange(ne)
    bg = s*sum_from(indices).astype(float)

    # Subtract the bg profile from each EDC
    for k in range(nk) :
        edc = get_edc(k)
        # Update data in-place
        edc -= bg

    data = convert_data_back(data, d, m, n)
    if profile :
        return data, bg
    else :
        return data

def subtract_bg_shirley_old(data, dim=0, normindex=0) :
    """ Use an iterative approach for the background of an EDC as described in
    DOI:10.1103/PhysRevB.5.4709. Mathematically, the value of the EDC after 
    BG subtraction for energy E EDC'(E) can be expressed as follows :

                               E1
                               /
        EDC'(E) = EDC(E) - s * | EDC(e) de
                               /
                               E

    where EDC(E) is the value of the EDC at E before bg subtraction, E1 is a 
    chosen energy value (in our case the last value in the EDC) up to which 
    the subtraction is applied and s is chosen such that EDC'(E0)=EDC'(E1) 
    with E0 being the starting value of the bg subtraction (in our case the 
    first value in the EDC).

    In principle, this is an iterative method, so it should be applied 
    repeatedly, until no appreciable change occurs through an iteration. In 
    practice this convergence is reached in 4-5 iterations at most and even a 
    single iteration may suffice.

    *Parameters*
    ============================================================================
    data : np.array; input data with shape (l x m) or (1 x l x m) containing 
           an E(k) cut
    dim  : int; either 0 or 1. Determines whether the input is aranged as 
           E(k) (m EDCs of length l, dim=0) or k(E) (l EDCs of length m, dim=1) 
    normindex : TESTING
    ============================================================================

    *Returns*
    ============================================================================
    data : np.array; has the same dimensions as the input array.
    ============================================================================
    """
    # Prevent original data from being overwritten by retaining a copy
    data = data.copy()

    data, d, l, m = convert_data(data)
    
    if dim == 0 :
        nk = m
        ne = l
        get_edc = lambda k : data[0,:,k]
    elif dim == 1 :
        nk = l
        ne = m
        get_edc = lambda k : data[0,k]

    indices = np.arange(ne)
    for k in range(nk) :
        edc = get_edc(k)

        # Calculate the "normalization" prefactor
        s = (edc[normindex] - edc[-1]) / edc.sum()

        # Prepare a function that sums the EDC from a given index upwards
        sum_from = np.frompyfunc(lambda e : edc[e:].sum(), 1, 1)

        # Update data in-place
        edc -= s * sum_from(indices).astype(float)

    data = convert_data_back(data, d, l, m)
    return data

def subtract_bg_kaminski(data) :
    """ 
    *Unfinished*
    Use the method of Kaminski et al. (DOI: 10.1103/PhysRevB.69.212509) 
    to subtract background. The principle is as follows:
    A lorentzian + a linear background y(x) = ax + b is fitted to momentum
    distribution curves. One then uses the magnitude of the linear component 
    at every energy as the background at that energy for a given k point. 

    *Parameters*
    ============================================================================
    data     array-like; the input data with shape (l x m) or 
             (l x m x 1) containing momentum in y (m momentum points) 
             and energy along x (l energy points) (plotting amazingly 
             inverts x and y)
    ============================================================================

    *Returns*
    """
    pass
    ## Get the number of energies
    #shape = data.shape
    #l = shape[0]

    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(1)
    ## For every energy, create an mdc
    #for e in range(500,600) :
    #    mdc = data[e]
    #    ax.plot(mdc, lw=1, color='k')

def apply_to_map(data, func, dim=1, output=True, fargs=(), fkwargs={}) :
    """ 
    **Untested**
    Apply a postprocessing function `func` which is designed to be 
    applied to an `energy vs k` cut to each cut of a map. 

    *Parameters*
    =======  ===================================================================
    data     array; 3D array of shape (l x m x n) representing the data.
    func     function; a function that can be applied to 2D data
    dim      int; dimension along which to apply the function `func`:
             0 - l cuts
             1 - m cuts
             2 - n cuts
    output   boolean; if *True*, collect the output of every application of 
             `func` on each slice in a list `results` and return it. Can be 
             set to *False* if no return is needed in order to save some memory.
    fargs    tuple; positional arguments to be passed on to `func`.
    fkwargs  dict; keyword arguments to be passed on to `func`.
    =======  ===================================================================

    *Returns*
    =======  ===================================================================
    returns  list; contains the return value of every call to `func` that was 
             made in the order they were made.
    =======  ===================================================================
    """
    n_cuts = data.shape[dim]
    if dim == 0 :
        get_cut = lambda i : data[i]
    elif dim == 1 :
        get_cut = lambda i : data[:,i]
    elif dim == 2  :
        get_cut = lambda i : data[:,:,i]

    if output :
        results = []

    for i in range(n_cuts) :
        cut = get_cut(i)
        result = func(cut, *fargs, **fkwargs)
        if output :
            results.append(result)

    if output :
        return results

# +--------------------------------+ #
# | Derivatieves/Data manipulation | # ========================================
# +--------------------------------+ #

def _derivatives(data, dx, dy) :
    """ Helper function to caluclate first and second partial derivatives of 
    2D data.

    *Parameters*
    ============================================================================
    data  array-like; the input data with shape (m x n) or (1 x m x n)
    dx    float; distance at x axis
    dy    float; distance at y axis
    ============================================================================

    *Returns*
    ============================================================================
    grad_x, grad_y, grad2_x, grad2_y
          np.arrays; first and second derivatives of data along x/y
          in the same shape as the input array
    ============================================================================
    """
    # The `axis` arguments change depending on the shape of the input data
    d = len(data.shape)
    axis0 = 0 if d==2 else 1

    # Calculate the derivatives in both directions
    grad_x = np.gradient(data, dx, axis=axis0)
    grad_y = np.gradient(data, dy, axis=axis0+1)

    # And the second derivatives...
    grad2_x = np.gradient(grad_x, dx, axis=axis0)
    grad2_y = np.gradient(grad_y, dy, axis=axis0+1)

    return grad_x, grad_y, grad2_x, grad2_y

def laplacian(data, dx=1, dy=1, a=None) :
    """ Apply the second derivative (Laplacian) to the data.

    *Parameters*
    ============================================================================
    data  array-like; the input data with shape (m x n) or (1 x m x n)
    dx    float; distance at x axis
    dy    float; distance at y axis
    a     float; scaling factor for between x and y derivatives.  
          Should be close to dx/dy. 
    ============================================================================

    *Returns*
    ============================================================================
    res   np.array; second derivative of input array in same dimensions
    ============================================================================
 
    """
    # Get the partial derivatives
    grad_x, grad_y, grad2_x, grad2_y = _derivatives(data, dx, dy)

    # Get a default value for a
    a = dx/dy if a == None else a

    # Return the laplacian
    return a * grad2_x + grad2_y

def curvature(data, dx=1, dy=1, cx=1, cy=1) :
    """ Apply the curvature method (DOI: 10.1063/1.3585113) to the data.
 
    *Parameters*
    ============================================================================
    data  array-like; the input data with shape (m x n) or (1 x m x n)
    dx    float; distance at x axis
    dy    float; distance at y axis
    ============================================================================

    *Returns*
    ============================================================================
    res   np.array; curvature of input array in same dimensions
    ============================================================================
    """
    # Get the partial derivatives
    grad_x, grad_y, grad2_x, grad2_y = _derivatives(data, dx, dy)
    
    # We also need the mixed derivative
    axis = 1 if len(data.shape)==2 else 2
    grad_xy = np.gradient(grad_x, dy, axis=axis)

    # And the squares of first derivatives
    grad_x2 = grad_x**2
    grad_y2 = grad_y**2

    # Build the nominator
    nominator  = (1 + cx*grad_x2)*cy*grad2_y
    nominator += (1 + cy*grad_y2)*cx*grad2_x
    nominator -= 2 * cx * cy * grad_x * grad_y * grad_xy

    # And the denominator
    denominator = (1 + cx*grad_x2 + cy*grad_y2)**(1.5)

    # Return the curvature
    return nominator / denominator

# +------------------------------------------+ #
# | Fermi level detection/curve manipulation | # ==============================
# +------------------------------------------+ #

def smooth(x, n_box, recursion_level=1) :
    """ Implement a linear midpoint smoother: Move an imaginary 'box' of size 
    'n_box' over the data points 'x' and replace every point with the mean 
    value of the box centered at that point.
    Can be called recursively to apply the smoothing n times in a row 
    by setting 'recursion_level' to n.

    At the endpoints, the arrays are assumed to continue by repeating their 
    value at the start/end as to minimize endpoint effects. I.e. the array 
    [1,1,2,3,5,8,13] becomes [1,1,1,1,2,3,5,8,13,13,13] for a box with 
    n_box=5.  

    *Parameters*
    ============================================================================
    x      1D array-like; the data to smooth
    n_box  int; size of the smoothing box (i.e. number of points around the 
           central point over which to take the mean).  Should be an odd 
           number - otherwise the next lower odd number is taken.
    recursion_level
           int; equals the number of times the smoothing is applied.
    ============================================================================

    *Returns*
    ============================================================================
    res    np.array; smoothed data points of same shape as input.
    ============================================================================
    """
    # Ensure odd n_box
    if n_box%2 == 0 :
        n_box -= 1

    # Make the box. Sum(box) should equal 1 to keep the normalization (?)
    box = np.ones(n_box) / n_box

    # Append some points to reduce endpoint effects
    n_append = int(0.5*(n_box-1))
    left = [x[0]]
    right = [x[-1]]
    y = np.array(n_append*left + list(x) + n_append*right)

    # Let numpy do the work
    #smoothened = np.convolve(x, box, mode='same')
    smoothened = np.convolve(y, box, mode='valid')

    # Do it again (enter next recursion level) or return the result
    if recursion_level == 1 :
        return smoothened
    else :
        return smooth(smoothened, n_box, recursion_level - 1)

def smooth_derivative(x, n_box=15, n_smooth=3) :
    """ Apply linear smoothing to some data, take the derivative of the 
    smoothed curve, smooth that derivative and take the derivative again. 
    Finally, apply a last round of smoothing.
    
    *Parameters*
    ============================================================================
    Same as in `func:arpys.postprocessing.smooth`.
    `n_smooth` corresponds to `recursion_level`.
    ============================================================================
    """
    # Smoothing 1
    res = smooth(x, n_box, n_smooth)
    # Derivative 1
    res = np.gradient(res)
    # Smoothing 2
    res = smooth(res, n_box, n_smooth)
    # Derivative 2
    res = np.gradient(res)
    # Smoothing 3
    res = smooth(res, n_box, n_smooth)
    return res

def zero_crossings(x, direction=0) :
    """ Return the indices of the points where the data in x crosses 0, going 
    from positive to negative values (direction = -1), vice versa 
    (direction=1) or both (direction=0). This is detected simply by a change 
    of sign between two subsequent points.

    *Parameters*
    ============================================================================
    x          1D array-like; data in which to find zero crossings
    direction  int, one of (-1, 0, 1); see above for explanation
    ============================================================================

    *Returns*
    ============================================================================
    crossings  list; list of indices of the elements just before the crossings
    ============================================================================
    """
    # Initiate the container
    crossings = []

    for i in range(len(x) - 1) :
        # Get the signs s0 and s1 of two neighbouring values i0 and i1
        i0 = x[i]
        i1 = x[i+1]
        s0 = np.sign(i0)
        s1 = np.sign(i1)
        if direction == -1 :
            # Only detect changes from + to -
            if s0 == 1 and s1 == -1 :
                crossings.append(i)
        elif direction == 1 :
            # Only detect changes from - to +
            if s0 == -1 and s1 == 1 :
                crossings.append(i)
        elif direction == 0 :
            # Detect any sign change
            if s0*s1 == -1 :
                crossings.append(i)

    return crossings

def old_detect_fermi_level(edc, n_box, n_smooth, orientation=1) :
    """ This routine is more useful for detecting local extrema, not really 
    for detecting steps. 
    """
    smoothdev = smooth_derivative(edc, n_box, n_smooth)
    crossings = zero_crossings(smoothdev[::orientation])
    return crossings

def detect_step(signal, n_box=15, n_smooth=3) :
    """ Try to detect a the biggest, clearest step in a signal by smoothing 
    it and looking at the maximum of the first derivative.
    """
    smoothened = smooth(signal, n_box, n_smooth)
    grad = np.gradient(smoothened)
    step_index = np.argmax(np.abs(grad))
    return step_index

def fermi_fit_func(E, E_F, sigma, a, b, T=10) :
    # Basic Fermi Dirac distribution at given T
    y = fermi_dirac(E, E_F, T) 
    
    # Add a linear contribution to the 'below-E_F' part
    y += (a*E+b)*step_function(E, E_F, flip=True)

    # Convolve with instrument resolution
    y = gaussian_filter(y, sigma)
    return y

def fit_fermi_dirac(energies, edc, e_0, T=10, sigma0=10, a0=0, b0=-0.1) :
    """ Try fitting a Fermi Dirac distribution convoluted by a Gaussian 
    (simulating the instrument resolution) plus a linear component on the 
    side with E<E_F to a given energy distribution curve.

    *Parameters*
    ========  ==================================================================
    energies  1D array of float; energy values.
    edc       1D array of float; corresponding intensity counts.
    e_0       float; starting guess for the Fermi energy. The fitting 
              procedure is quite sensitive to this.
    T         float; temperature.
    sigma0    float; starting guess for the standard deviation of the Gaussian.
    a0        float; starting guess for the slope of the linear component.
    b0        float; starting guess for the linear offset.
    ========  ==================================================================

    *Returns*
    ========  ==================================================================
    p         list of float; contains the fit results for [E_F, sigma, a, b].
    res_func  callable; the fit function with the optimized parameters. With 
              this you can just do res_func(E) to get the value of the 
              Fermi-Dirac distribution at energy E.
    ========  ==================================================================
    """
    # Normalize the EDC
    edc = edc/edc.max()

    # Initial guess and bounds for parameters
    p0 = [e_0, sigma0, a0, b0]
    de = 1
    lower = [e_0-de, 0, -10, -1]
    upper = [e_0+de, 100, 10, 1]

    def fit_func(E, E_F, sigma, a, b) :
        """ Wrapper around fermi_fit_func whta fixes T. """
        return fermi_fit_func(E, E_F, sigma, a, b, T=T)

    # Carry out the fit
    p, cov = curve_fit(fit_func, energies, edc, p0=p0, bounds=(lower, upper)) 

    res_func = lambda x : fit_func(x, *p) 
    return p, res_func

def fit_gold(D, e_0=None, T=10) :
    """ Apply a Fermi-Dirac fit to all EDCs of an ARPES Gold spectrum. 
    
    *Parameters*
    ===  =======================================================================
    D    argparse.Namespace object; ARPES data and metadata as is created by a
         :class: `Dataloader <arpys.dataloaders.Dataloader>` object.
    e_0  float; starting guess for the Fermi energy in the energy units 
         provided in *D*. If this is not given, a starting guess will be 
         estimated by detecting the step in the integrated spectrum using :func:
         `detect_step <arpys.postprocessing.detect_step>`.

    T    float; Temperature
    ===  =======================================================================

    *Returns*
    ============  ==============================================================
    fermi_levels  list; Fermi energy for each EDC.
    sigmas        list; standard deviations of insrtument resolution Gaussian 
                  for each EDC.
    functions     list of callables; functions of energy that produce the fit 
                  for each EDC.
    ============  ==============================================================
    """
    # Extract data
    gold = D.data[0]
    n_pixels, n_energies = gold.shape
    energies = D.xscale
    pixels = np.arange(n_pixels)

    # If no hint for the Fermi energy is given, try to detect it from the 
    # gradient of the integrated spectrum
    if e_0 is None :
        integrated = gold.sum(0)
        step_index = detect_step(integrated)
        e_0 = energies[step_index]

    params = []
    functions = []
    for i,edc in enumerate(gold) :
        p, res_func = fit_fermi_dirac(energies, edc, e_0, T=T)
        params.append(p)
        functions.append(res_func)

    # Prepare the results
    params = np.array(params)
    fermi_levels = params[:,0]
    sigmas = params[:,1]

    return fermi_levels, sigmas, functions

# +------------------------------------------------+ #
# | Conversion from angular coordinates to k space | # =========================
# +------------------------------------------------+ #

def angle_to_k(angles, theta, phi, hv, E_b, work_func=4, c1=0.5124, 
               shift=0, lattice_constant=1, degrees=True) :
    """ Convert the angular information you get from beamline data into 
    proper k-space coordinates (in units of pi/a, a being the lattice 
    constant) using the formula:

        k_x = c1 * sqrt(hv - e*phi + E_B) * sin(theta_a + theta_m)
        k_y = c1 * sqrt(hv - e*phi + E_B) * sin(phi_a + phi_m)

    with:
             c1 : numeric constant
             hv : photon energy (eV)
          e*phi : work function (eV)
            E_B : electron binding energy (eV)
        theta_a : in-plane angle along analyzer slit
        theta_m : in-plane manipulator angle
          phi_a : angle along analyzer slit in vertical configuration
          phi_m : tilt (?)

    Confer the sheet 'Electron momentum calculations' from ADDRESS beamline 
    for more information on the used notation.
        
    *Parameters*
    ============================================================================
    angles     1D array; angles (in degree) to be converted into k space
               (currently equal theta_a)
    theta      float; corresponds to theta_m 
    phi        float; corresponds to phi_m
    hv         float; photon energy in eV
    E_B        float; binding energy of electrons in eV
    work_func  float; corresponds to e*phi (eV)
    c1         float; numeric constant. Shouldn't really be changed 
    shift      float; shift in units of `angle` to get zero at right 
               position 
    lattice_constant    
               float; lattice constant a in Angstrom used to convert to 
               units of pi/a
    degrees    bool; allows giving the angles in either degrees or radians
    ============================================================================

    *Returns*
    ============================================================================
    kx, ky     tuple of floats; kx and ky coordinates in units of pi/a
    ============================================================================

    """
    # Precalculate the prefactor (*lattice_constant to get lattice constant 
    # units)
    prefactor = c1 * np.sqrt(hv - work_func + E_b) * lattice_constant / np.pi

    # Apply a shift to the angles
    angles = angles.copy() + shift

    # Convert all angles from degrees to radians
    if degrees :
        conversion = np.pi/180
        # Leave the original array unchanged
        angles = angles.copy()
        angles *= conversion
        theta *= conversion
        phi *= conversion

    # kx and ky in inverse Angstrom or lattice constant units if 
    # lattice_constant!=1
    kx = prefactor * np.sin((theta + angles))
    ky = prefactor * np.sin(phi) 

    return kx, ky

def new_a2k(thetas, tilts, hv, work_func=4, E_b=0, dtheta=0, dtilt=0, 
            lattice_constant=1, orientation='horizontal', 
            photon_momentum=True, alpha=20) :
    """ Cleaner implementation of angle to k conversion, particularly more 
    suitable for maps. 
    Confer docstring of :func: `angle_to_k
    <arpys.postprocessing.angle_to_k>` for now.
    """
    # c0 = sqrt(2*&m_e)/hbar
    c0 = 0.5124
    # c1 is the angle to radian conversion
    c1 = np.pi/180
    
    prefactor = c0 * np.sqrt(hv - E_b - work_func)
    # Convert to units of pi/a
    prefactor *= lattice_constant/np.pi

    kx = prefactor * np.sin(c1*(thetas+dtheta))
    ky = prefactor * np.cos(c1*(tilts+dtilt))

    if photon_momentum :
        # TODO Change this in other slit orientation
        # TODO Use actual manipulator values instead of dtheta and dtilt
        # c2 is the eV to Angstrom conversion
        c2 = 2*lattice_constant/12400
        kx -= c2*np.cos(c1*(alpha+dtheta))
        ky += c2*np.sin(c1*(alpha+dtheta))*np.sin(c1*(dtilt))

    o = orientation.lower()[0]
    if o == 'h' :
        return kx, ky
    elif o == 'v' :
        return ky, kx
    else :
        raise ValueError('Orientation not understood: {}.'.format(orientation))

def a2k(D, lattice_constant, dtheta=0, dtilt=0) :
    """
    Shorthand angle to k conversion that takes the output of a `Dataloader 
    <arpys.dataloaders.Dataloader>` object as input and passes all necessary 
    information on to the actual converter (`new_a2k 
    <arpys.postprocessing.new_a2k>`).
    """
    kx, ky = new_a2k(D.xscale, D.yscale, hv=D.hv, 
                     lattice_constant=lattice_constant, dtheta=dtheta, 
                     dtilt=dtilt)
    return kx, ky

def alt_a2k(angle, dtilt, dtheta, dphi, hv, work_func, orientation='horizontal') :
    """ 
    *Unfinished*
    Alternative angle-to-k conversion approach using rotation matrices. 
    Determine the norm of the k vector from the kinetic energy using the 
    relation:
               sqrt( 2*m_e*hv )
        k_F = ------------------
                     hbar
    Then initiate a k vector in the direction measured and rotate it with the 
    given tilt, theta and phi angles.
    
    """
    pass


# +---------+ #
# | Fitting | # ================================================================
# +---------+ #

def step_function_core(x, step_x=0, flip=False) :
    """ Implement a perfect step function f(x) with step at `step_x`:
    
            / 0   if x < step_x
            |
    f(x) = {  0.5 if x = step_x
            |
            \ 1   if x > step_x

    *Parameters*
    ============================================================================
    x       array; x domain of function
    step_x  float; position of the step
    flip    boolean; Flip the > and < signs in the definition
    ============================================================================
    """
    sign = -1 if flip else 1
    if sign*x < sign*step_x :
        result = 0
    elif x == step_x :
        result = 0.5
    elif sign*x > sign*step_x :
        result = 1
    return result

def step_function(x, step_x=0, flip=False) :
    """ np.ufunc wrapper for step_function_core. Confer corresponding 
    documentation. 
    """
    res = \
    np.frompyfunc(lambda x : step_function_core(x, step_x, flip), 1, 1)(x)
    return res.astype(float)

def step_core(x, step_x=0, flip=False) :
    """ Implement a step function f(x) with step at `step_x`:

                / 0 if x < step_x
        f(x) = {
                \ 1 if x >= step_x

    Confer also :func: `step_function_core`.
    """
    sign = -1 if flip else 1
    if sign*x < sign*step_x :
        result = 0
    elif sign*x >= sign*step_x :
        result = 1
    return result

def step_ufunc(x, step_x=0, flip=False) :
    """ np.ufunc wrapper for step_core. Confer corresponding 
    documentation. 
    """
    res = \
    np.frompyfunc(lambda x : step_core(x, step_x, flip), 1, 1)(x)
    return res.astype(float)

def lorentzian(x, a=1, mu=0, gamma=1) :
    """ Implement a Lorentzian curve f(x) given by the expression


                        a
           ----------------------------
    f(x) =                         2
                      /   /  x-mu \  \
            pi*gamma*( 1+( ------- )  )
                      \   \ gamma /  /


    *Parameters*
    ============================================================================
    x      array; variable at which to evaluate f(x)
    a      float; amplitude (maximum value of curve)
    mu     float; mean of curve (location of maximum)
    gamma  float; half-width at half-maximum (HWHM) of the curve
    ============================================================================

    *Returns*
    ============================================================================
    res    array containing the value of the Lorentzian at every point of 
           input x 
    ============================================================================
    """
    return a/( np.pi*gamma*( 1 + ((x-mu)/gamma)**2 ) )

def gaussian(x, a=1, mu=0, sigma=1) :
    """ Implement a Gaussian bell curve f(x) given by the expression

                                     2
                      1    / (x-mu) \
    f(x) = a * exp( - - * ( -------  ) )
                      2    \ sigma  /

    *Parameters*
    ============================================================================
    x      array; variable at which to evaluate f(x)
    a      float; amplitude (maximum value of curve)
    mu     float; mean of curve (location of maximum)
    sigma  float; standard deviation (`width` of the curve)
    ============================================================================

    *Returns*
    ============================================================================
    res    array containing the value of the Gaussian at every point of 
           input x 
    ============================================================================
    """
    return a * np.exp(-0.5 * (x-mu)**2 / sigma**2)

#def merge_functions(f, g, x0) :
#    """ Return a function F(x) which is defined by:
#                / f(x) if x < x0
#        F(x) = {
#                \ g(x) if x >= x0
#    """
#    def core_func(x, fparams, gparams) :
#        if x < x0 :
#            return f(x, **fparams)
#        else :
#            return g(x, **gparams)
#
#    # Convert to np.ufunc and ensure the returned dtype is float
#    core_func = np.frompyfunc(core_func, 1, 1)
#    def F(x, fparams, gparams) :
#        return core_func(x, fparams, gparams).astype(float)
#    return F

def gaussian_step(x, step_x=0, a=1, mu=0, sigma=1, flip=False, after_step=None) :
    """ Implement (as a broadcastable np.ufunc) a sort-of convolution of a 
    step-function with a Gaussian bell curve, defined as follows :

            / g(x, a, mu, sigma) if x < step_x
    f(x) = {
            \ after_step         if x >= step_x

    where g(x) is the :func: `gaussian`.

    *Parameters*
    ============================================================================
    x           array; x domain of function
    step_x      float; position of the step
    a           float; prefactor of the Gaussian
    mu          float; mean of the Gaussian
    sigma       float; standard deviation of the Gaussian
    flip        boolean; Flip the > and < signs in the definition
    after_step  float; if not None, set a constant value that is assumed after 
                the step. Else assume the value of the Gaussian at the step_x.   
    ============================================================================
    """
    # If no specific step height is given, use the value of the Gaussian at 
    # x=step_x
    if after_step is None :
        after_step = gaussian(step_x, a, mu, sigma)

    # Define the function core
    def core_function(X) :
        sign = -1 if flip else 1
        if sign*X < sign*step_x :
            return gaussian(X, a, mu, sigma)
        else :
            return after_step
    # Convert to a numpy ufunc on the fly, such that arrays can be accepted 
    # as input
    result = np.frompyfunc(core_function, 1, 1)(x)
    # Convert to float
    return result.astype(float)

def fermi_dirac(E, mu=0, T=4.2) :
    """ Return the Fermi Dirac distribution with chemical potential *mu* at 
    temperature *T* for energy *E*. The Fermi Dirac distribution is given by

                     1
    n(E) = ----------------------
            exp((E-mu)/(kT)) + 1

    and assumes values from 0 to 1.
    """
    kT = constants.k_B * T / constants.eV
    res = 1 / (np.exp((E-mu)/kT) + 1)
    return res

# +---------+ #
# | Various | # ================================================================
# +---------+ #

def rotation_matrix(theta) :
    """ Return the 2x2 rotation matrix for an angle theta (in degrees). """
    # Build the rotation matrix (convert angle to radians first)
    t = np.pi * theta/180.
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    return R

def rotate_XY(X, Y, theta=45) :
    """ Rotate a coordinate mesh of (n by m) points by angle *theta*. *X* and 
    *Y* hold the x and y components of the coordinates respectively, as if 
    generated through :func: `np.meshgrid()`.

    *Parameters*
    =====  =====================================================================
    X      n by m array; x components of coordinates.
    Y      n by m array; y components of coordinates.
    theta  float; rotation angle in degrees
    =====  =====================================================================

    *Returns*
    ===  =======================================================================
    U,V  n by m arrays; U (V) contains the x (y) components of the rotated 
    coordinates. These can be used as arguments to :func: pcolormesh()
    ===  =======================================================================

    .. :see also: `<arpes.postprocessing.rotate_xy>`
    """
    # Create the rotation matrix
    R = rotation_matrix(theta)

    # In order to take the dot product we need to flatten the meshgrids and 
    # organize them correctly
    U, V = np.dot(R, [X.ravel(), Y.ravel()])

    return U, V

def rotate_xy(x, y, theta=45) :
    """ Rotate the x and y cooridnates of rectangular 2D data by angle theta.

    *Parameters*
    =====  =====================================================================
    x      1D array of length n; x values of the rectangular grid
    y      1D array of length m; y values of the rectangular grid
    theta  float; rotation angle in degrees
    =====  =====================================================================

    *Returns*
    ===  =======================================================================
    U,V  n by m arrays; U (V) contains the x (y) components of the rotated 
         coordinates. These can be used as arguments to :func: pcolormesh() 
    ===  =======================================================================

    .. :see also: `<arpes.postprocessing.rotate_XY>`
    """
    # Create a coordinate mesh
    X, Y = np.meshgrid(x, y)
    # rotate_XY does the rest of the work
    return rotate_XY(X, Y, theta)

def symmetrize_around(data, p0, p1) :
    """ TODO p0, p1: indices of points """
    # Data has to be 2D and dimensions (y, x) (which is flipped by pcolormesh)
    ny, nx = data.shape

    # Create coordinate arrays
    original_x = np.arange(nx, dtype=float)
    original_y = np.arange(ny, dtype=float)

    # Build the linear function that defines the line through the selected 
    # points
    m = (p1[1] - p0[1]) / (p1[0] - p0[0])
    print('P0: ', p0, '\nP1: ', p1)
    print('m: ', m)
    y0 = p1[1] - m*p1[0]
    x0 = -y0/m
    def line(x) :
        return m*x +y0
    def inverse_line(y) :
        return (y - y0)/m

    # Find the intersect of the line with the data boundary
    ileft = [0, line(0)]
    iright = [nx, line(nx)]
    ibot = [inverse_line(0), 0]
    itop = [inverse_line(ny), ny]

    # Determine which of these intersects are actually in the visible range
    # (there are 6 possible cases)
    if ibot[0] >= 0 and ibot[0] < nx :
        if ileft[1] > 0 and ileft[1] <= ny :
            i0 = ileft
            i1 = ibot
        elif itop[0] > 0 and itop[0] <= nx :
            i0 = ibot
            i1 = itop
        elif iright[1] > 0 and iright[1] <= ny :
            i0 = ibot
            i1 = iright
    elif ileft[1] >= 0 and ileft[1] < ny :
        if itop[0] > 0 and itop[0] <= nx :
            i0 = ileft
            i1 = itop
        else :
            i0 = ileft
            i1 = iright
    else :
        i0 = itop
        i1 = iright

    # Find the middle point of the line
    line_center = [0.5*(i1[i] + i0[i]) for i in range(2)]

    # Build the mirror matrix
    phi = np.arctan(1/m)
    T = np.array([[np.cos(2*phi), np.sin(2*phi)],
                  [np.sin(2*phi),-np.cos(2*phi)]])

    # Transform the coordinates
    # 1) Shift origin to the central point of the part of the mirror line 
    # that lies within the data range
    xshift = line_center[0]
    x = original_x - xshift
    yshift = line_center[1]
    y = original_y - yshift
    print('xshift, yshift: ', xshift, yshift)
    print('original_x.min(), original_x.max(): ', original_x.min(), original_x.max())
    print('x.min(), x.max(): ', x.min(), x.max())
    print('y.min(), y.max(): ', y.min(), y.max())
    # 2 Create coordinate vectors
    X, Y = [a.flatten() for a in np.meshgrid(x, y)]
    v0 = np.stack((X, Y))
    # 3) Take the matrix product
    v1 = T.dot(v0)
    # 4) Shift origin back
    xt = v1[0] + xshift
    yt = v1[1] + yshift
    print('xt.min(), xt.max(): ', xt.min(), xt.max())
    print('yt.min(), yt.max(): ', yt.min(), yt.max())

    transformed = ndimage.map_coordinates(data, [yt, xt]).reshape(ny, nx)
#    transformed = ndimage.map_coordinates(data, [xt, yt]).reshape(ny, nx)
    return transformed[::-1,::-1]
#    return transformed

def symmetrize_rectangular(data, i, k=None) :
    """ Symmetrize a piece of rectangular *data* around column *i* by simply 
    mirroring the data at column *i* and overlaying it in the correct position. 
    
    *Parameters*
    ====  ======================================================================
    data  array of shape (ny, nx0); data to be symmetrized.
    i     int; index along x (0 <= i < nx0) around which to symmetrize.
    k     array of length nx0; the original k values (x scale of the data). 
          If this is given, the new, expanded k values will be calculated and 
          returned.
    ====  ======================================================================

    *Returns*
    ======  ====================================================================
    result  array of shape (ny, nx1); the x dimension has expanded.
    sym_k   array of length nx1; the expanded k values (x scale to the 
            resulting data). If *k* is None, *sym_k* will also be None.
    ======  ====================================================================

    Here's a graphical explanation for the coordinates used in the code for 
    the case i < nx0/2. If i > nx0/2 we flip the data first such that we can 
    apply the same procedure and coordinates.
    ```
    Original image:

            +----------+
            |   |      |
            |          |
            |   |      |
            |          |
            |   |      |
            +----------+

    Mirrored image:

         x----------x
         |      |   |
         |          |
         |      |   |
         |          |
         |      |   |
         x----------x

    Overlay both images:
        
         x--+-------x--+
         |  |   |   |  |
         |  |       |  |
         |  |   |   |  |
         |  |       |  |
         |  |   |   |  |
         x--+-------x--+
         ^  ^   ^   ^  ^
         0  |   i  nx0 |
            |          |
         nx0-2*i      nx1 = 2*(nx0-i)
    ```
    """
    # Flip the image if i>nx0/2
    ny, nx0 = data.shape
    if i > nx0/2 :
        flipped = True
        data = data[:,::-1]
        i = nx0-i
    else :
        flipped = False

    # Define relevant coordinates
    i0 = nx0 - 2*i

    # Determine the new data dimensions and prepare the new data container
    nx1 = 2*(nx0 - i)
    result = np.zeros([ny, nx1])

    # Fill the mirrored data into the new container
    result[:,:nx0] += data[:,::-1]

    # Overlay the original data
    result[:,i0:] += data

    # "Normalize" the overlap region
    result[:,i0:nx0] /= 2

    # Create the new k values
    if k is not None :
        sym_k = np.arange(nx1, dtype=float)
        dn = nx1 - nx0
        dk = k[1] - k[0]
        if flipped :
            # Extend to the right
            sym_k[:nx0] = k
            start = k[-1]
            stop = start + dn*dk
            sym_k[nx0:] = np.arange(start, stop, dk)
        else :
            # Extend to the left
            sym_k[dn:] = k
            start = k[0]
            stop = start - dn*dk
            sym_k[:dn] = np.arange(start, stop, -dk)[::-1]
    else :
        sym_k = None

    return result, sym_k

def symmetrize_map(kx, ky, mapdata, clean=False, overlap=False, n_rot=4, 
                   debug=False) :
    """ Apply all valid symmetry operations (rotation around 90, 180, 270 
    degrees, mirror along x=0, y=0, y=x and y=-x axis) to a map and sum their 
    results together in order to get a symmetric picture. 
    The `clean` option allows to automatically cut off unsymmetrized parts and 
    returns a data array of reduced size, containing only the points that 
    could get fully symmetrized. In this case, kx and ky are also trimmed to 
    the right size.
    This functions implements a couple of optimizations, leading to slightly 
    more complicated but faster running code.

    *Parameters*
    ===========  ===============================================================
    kx           n length array
    ky           m length array
    mapdata      (m x n) array (counterintuitive to kx and ky but consistent 
                 with pcolormesh)
    clean        boolean; toggle whether or not to cut off unsymmetrized parts
    ===========  ===============================================================

    *Returns*
    ===========  ===============================================================
    kx, ky       if `clean` is False, these are the same as the input kx and 
                 ky. If `clean` is True the arrays are cut to the right size
    symmetrized  2D array; the symmetrized map. Either it has shape (m x n) 
                 `clean` is False) or smaller, depending on how much could 
                 be symmetrized.
    ===========  ===============================================================
    """
    # Create data index ranges
    M, N = mapdata.shape
    ms = range(M)
    ns = range(N)

    # Sort kxy and retain a copy of the original data
    kx.sort()
    ky.sort()
    symmetrized = mapdata.copy()

    # DEBUG: create a highly visible artifact in the data
    if debug:
        mapdata[int(M/2)+15:int(M/2)+25, int(N/2)+15:int(N/2)+25] = \
                                                      2*symmetrized.max()

    # Define all symmetry operations, starting with the rotations
    transformations = []
    # Rotate n_rot-1 times
    theta0 = 360./n_rot
    for i in range(1, n_rot) :
        theta = i*theta0
        R = rotation_matrix(theta)
        transformations.append(R)
    
    # Define the mirror transformation matrices
    # Mirror along x=0 axis
    Mx = np.array([[ 1,  0],
                   [ 0, -1]])
    # Mirror along y=0 axis
    My = np.array([[-1,  0],
                   [ 0,  1]])
    # Mirrors along diagonals y=x and y=-x
    # The diagonal reflections are a combinaton of rotation by 90 deg and 
    # reflections
    R90 = rotation_matrix(90)
    Mxy = Mx.dot(R90)
    Myx = My.dot(R90)
    for T in Mx, My, Mxy, Myx :
        transformations.append(T)

    # Create original k vectors and index arrays (index arrays make np.array 
    # assignments significantly faster compared to assignments in a loop)
    K = []
    MS = []
    NS = []
    for i,x in enumerate(kx) :
        for j,y in enumerate(ky) :
            K.append([x, y])
            MS.append(j)
            NS.append(i)
    # Transpose to get a `np.dot()-able` shape
    K = np.array(K).transpose()
    MS = np.array(MS, dtype=int)
    NS = np.array(NS, dtype=int)

    # Initialize bottom left and top right k vectors
    if clean :
        kxmin = -np.inf
        kxmax = np.inf
        kymin = -np.inf
        kymax = np.inf

    if overlap :
        product = 0

    for T in transformations :
        # Transform all k vectors at once
        KV = np.dot(T, K)

        # Initialize new index arrays. searchsorted finds at which index one 
        # would have to insert a given value into the given array -> MS_, NS_ 
        # are lists of indices
        MS_ = np.searchsorted(ky, KV[1])
        NS_ = np.searchsorted(kx, KV[0])
        # Set out-of-bounds regions to the actual indices
        where_m = (MS_ >= M) | (MS_ <= 0)
        where_n = (NS_ >= N) | (NS_ <= 0)
        MS_[where_m] = 0#MS[where_m]
        NS_[where_n] = 0#NS[where_n]

        # Add the transformed map to the original (only where we actually had 
        # overlap) (`~` is the bitwise not operator)
        w = ~where_m & ~where_n
        symmetrized[MS[w], NS[w]] += mapdata[MS_[w], NS_[w]]

        if overlap :
            product += sum( mapdata[MS, NS] * mapdata[MS_, NS_] )

        # Keep track of min and max k vectors
        if clean :
            # Get the still in-bound (ib) kx and ky components. `~` is the 
            # bitwise `not` operator
            kxs_ib = KV[0,~where_n]
            kys_ib = KV[1,~where_m]
            if len(kxs_ib) is not 0 :
                new_kxmin = kxs_ib.min()
                new_kxmax = kxs_ib.max()
                kxmin = new_kxmin if new_kxmin > kxmin else kxmin
                kxmax = new_kxmax if new_kxmax < kxmax else kxmax
            if len(kys_ib) is not 0 :
                new_kymin = kys_ib.min()
                new_kymax = kys_ib.max()
                kymin = new_kymin if new_kymin > kymin else kymin
                kymax = new_kymax if new_kymax < kymax else kymax

    # End of T loop

    # Cut off unsymmetrized parts of the data and kx, ky
    if clean :
        x0 = np.searchsorted(kx, kxmin)
        x1 = np.searchsorted(kx, kxmax)
        y0 = np.searchsorted(ky, kymin)
        y1 = np.searchsorted(ky, kymax)
        symmetrized = symmetrized[y0:y1, x0:x1]
        kx = kx[x0:x1]
        ky = ky[y0:y1]

    if overlap :
        return kx, ky, symmetrized, product
    else :
        return kx, ky, symmetrized

def plot_cuts(data, dim=0, zs=None, labels=None, max_ppf=16, max_nfigs=4, 
              **kwargs) :
    """ Plot all (or only the ones specified by `zs`) cuts along dimension `dim` 
    on separate subplots onto matplotlib figures.

    *Parameters*
    =========  =================================================================
    data       3D np.array with shape (z,y,x); the data cube.
    dim        int; one of (0,1,2). Dimension along which to take the cuts.
    zs         1D np.array; selection of indices along dimension `dim`. Only 
               the given indices will be plotted.
    labels     1D array/list of length z. Optional labels to assign to the 
               different cuts
    max_ppf    int; maximum number of *p*lots *p*er *f*igure.
    max_nfigs  int; maximum number of figures that are created. If more would 
               be necessary to display all plots, a warning is issued and 
               only every N'th plot is created, where N is chosen such that 
               the whole 'range' of plots is represented on the figures. 
    kwargs     dict; keyword arguments passed on to :func: `pcolormesh 
               <matplotlib.axes._subplots.AxesSubplot.pcolormesh>`. 
               Additionally, the kwarg `gamma` for power-law color mapping 
               is accepted.
    =========  =================================================================
    """
    # Create a list of all indices in case no list (`zs`) is given
    if zs is None :
        zs = np.arange(data.shape[dim])

    # The total number of plots and figures to be created
    n_plots = len(zs)
    n_figs = int( np.ceil(n_plots/max_ppf) )
    nth = 1
    if n_figs > max_nfigs :
        # Only plot every nth plot
        nth = round(n_plots/(max_ppf*max_nfigs))
        # Get the right English suffix depending on the value of nth
        if nth <= 3 :
            suffix = ['st', 'nd', 'rd'][nth-1]
        else :
            suffix = 'th'
        warnings.warn((
        'Number of necessary figures n_figs ({0}) > max_nfigs ({1}).' +
        'Setting n_figs to {1} and only plotting every {2}`{3} cut.').format( 
            n_figs, max_nfigs, nth, suffix))
        n_figs = max_nfigs
        n_plots = max_ppf*n_figs

    # If we have just one figure, make the subplots as big as possible by 
    # setting the number of subplots per row (ppr) to a reasonable value
    if n_figs == 1 :
        ppr = int( np.ceil(np.sqrt(n_plots)) )
    else :
        ppr = int( np.ceil(np.sqrt(max_ppf)) )

    # Depending on the dimension we need to extract the cuts differently. 
    # Account for this by moving the axes
    x = np.arange(len(data.shape))
    data = np.moveaxis(data, x, np.roll(x, dim))

    # Extract additional kwarg from kwargs
    if 'gamma' in kwargs :
        gamma = kwargs.pop('gamma')
    else :
        gamma = 1

    # Define the beginnings of the plot in figure units
    margins = dict(left=0, right=1, bottom=0, top=1)

    figures = []
    for i in range(n_figs) :
        # Create the figure with pyplot 
        fig = plt.figure()
        start = i*ppr*ppr
        stop = (i+1)*ppr*ppr
        # Iterate over the cuts that go on this figure
        for j,z in enumerate(zs[start:stop]) :
            # Try to extract the cut and create the axes 
            cut_index = z*nth
            try :
                cut = data[cut_index]
            except IndexError :
                continue
            ax = fig.add_subplot(ppr, ppr, j+1)
            ax.pcolormesh(cut, norm=PowerNorm(gamma=gamma), **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if labels is not None :
                labeltext = str(labels[cut_index])
            else :
                labeltext = str(cut_index)
            label = ax.text(0, 0, labeltext, size=10)
            label.set_path_effects([withStroke(linewidth=2, foreground='w', 
                                               alpha=0.5)])

        fig.subplots_adjust(hspace=0.01, wspace=0.01, **margins)
        figures.append(fig)

    return figures

# +---------+ #
# | Testing | # ================================================================
# +---------+ #

if __name__ == '__main__' :    
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit

    import arpys as arp
    filename = ('/home/kevin/Documents/qmap/experiments/2019_01_I05/' +
                'Ca327_1_A/i05-95358.nxs')
    D = arp.dl.load_data(filename)
    data = D.data[0]
    energies = D.xscale
    pixels = np.arange(len(D.yscale))

    T = 10

    levels, sigmas, funcs = fit_gold(D, T=T)
    print(np.mean(levels), np.mean(sigmas))

    offset = 0.5

    nrow = 1
    ncol = 2

    fig = plt.figure()

    ax0 = fig.add_subplot(nrow, ncol, 1)
    ax0.pcolormesh(energies, pixels, data)

    ax1 = fig.add_subplot(nrow, ncol, 2)
    for j,i in enumerate(range(5, 500, 50)) :
        edc = D.data[0][i]
        edc = edc/edc.max()
        f = funcs[i]
        ax1.plot(energies, edc+j*offset, 'k-',
                 energies, f(energies)+j*offset, 'r-')

    plt.show()


