#!/usr/bin/python
""" 
Contains different tools to post-process (ARPES) data. 
"""

import numpy as np
import warnings

# +------------+ #
# | Decorators | # ============================================================
# +------------+ #

def array_input(f) :
    """ Decorator which converts input to a numpy array. """
    def decorated(data, *args, **kwargs) :
        data = np.array(data)
        return f(data, *args, **kwargs)
    return decorated

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

@array_input
def make_slice(data, d, i, integrate=0) :
    """ Create a slice out of the 3d data (l x m x n) along dimension d 
    (0,1,2) at index i. Optionally integrate around i.

    Parameters
    ----------
    data        : array-like; map data of the shape (l x m x n) where l 
                  corresponds to the number of energy values
    d           : int, d in (0, 1, 2); dimension along which to slice
    i           : int, 0 <= i < data.size[d]; The index at which to create the slice
    integrate   : int, 0 <= integrate < |i - n|; the number of slices above 
                  and below slice i over which to integrate

    Returns
    -------
    res         : np.array; Slice at index with dimensions shape[:d] + shape[d+1:]
                  where shape = (l, m, n).
    """
    # Get the relevant dimensions
    shape = data.shape
    try :
        n_slices = shape[d]
    except IndexError :
        print('d ({}) can only be 0, 1 or 2'.format(d))
        return

    output_shape = shape[:d] + shape[d+1:]

    # Set the integration indices and adjust them if they go out of scope
    start = i - integrate
    stop = i + integrate + 1
    if start < 0 :
        warnings.warn(
        'i - integrate ({}) < 0, setting start=0'.format(start))       
        start = 0
    if stop >= n_slices :
        warnings.warn(
        'i + integrate ({}) >= n_slices ({}), setting \
            stop=n_slices'.format(stop, n_slices))       
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

@array_input
def make_map(data, i, integrate=0) :
    """ Create a 'top view' slice for FSM data.
    If the values of i or integrate are bigger than what is possible, they 
    are automatically reduced to the maximum possible.

    Parameters
    ----------
    data        : array-like; map data of the shape (l x m x n) where l 
                  corresponds to the number of energy values
                
    i           : int, 0 <= i < n; The index at which to create the slice
    integrate   : int, 0 <= integrate < |i - n|; the number of slices above 
                  and below slice i over which to integrate

    Returns
    -------
    res         : np.array; Map at given energy with dimensions (m x n)
    """
    # Prepare the start and stop indices for slicing of the map data
    l, m, n = data.shape
    if i >= l :
        warnings.warn('i ({}) >= l ({}), setting i=l'.format(i, l))       
        i = l

    start = i - integrate
    stop = i + integrate + 1
    if start < 0 :
        warnings.warn(
        'i - integrate ({}) < 0, setting start=0'.format(start))       
        start = 0
    if stop >= l :
        warnings.warn(
        'i + integrate ({}) >= l ({}), setting stop=l'.format(stop, l))       
        stop = l

    # Initialize data container and fill it with data from selected slices
    fsm = np.zeros([m, n])
    for i in range(start, stop, 1) :
        fsm += data[i, :, :]

    return fsm

# +---------------+ #
# | Normalization | # ========================================================
# +---------------+ #

@array_input
def normalize_globally(data, minimum=True) :
    """ The simplest approach: normalize the whole dataset by the global min- 
    or maximum.

    Parameters
    ----------
    data        : array-like; the input data of arbitrary dimensionality
    minimum     : boolean; if True, use the min, otherwise the max function

    Returns
    -------
    res         : np.array; normalized version of input data
    """
    # Select whether to use the minimum or the maximum for normalization
    min_or_max = 'min' if minimum else 'max'

    # Retrieve the global maximum
    m = data.__getattribute__(min_or_max)()

    # Return the normalized values
    return data/m

def convert_data(data) :
    """ Helper function to convert data to the right shape. """
    # Find out whether we have a (l x m) (d=2) or a (1 x l x m) (d=3) array
    shape = data.shape
    d = len(shape)

    # Convert to shape (1 x l x m) 
    if d == 2 :
        l = shape[0]
        m = shape[1]
        data = data.reshape(1, l, m)
    else :
        l = None
        m = None

    return data, d, l, m

def convert_data_back(data, d, l, m) :
    """ Helper function to convert data back to the original shape which is 
    determined by the values of d, l and m (outputs of :func: convert_data).
    """
    if d == 2 :
        data = data.reshape(l, m)

@array_input
def normalize_per_segment(data, dim=0, minimum=False) :
    """ Normalize each column/row by its respective max value.

    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (1 x l x m)
    dim         : int; along which dimension to normalize (0 or 1)
    minimum     : boolean; if True, use the min, otherwise the max function

    Returns
    -------
    res         : np.array; normalized version of input data in same shape
    """
    # Select whether to use the minimum or the maximum for normalization
    min_or_max = 'min' if minimum else 'max'

    # Convert data if necessary
    data, d, l, m = convert_data(data)

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
    convert_data_back(data, d, l, m)

    return data

@array_input
def normalize_per_integrated_segment(data, dim=0) :
    """ Normalize each MDC/EDC by its integral.

    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (1 x l x m)
    dim         : int; along which dimension to normalize (0 or 1)

    Returns
    -------
    res         : np.array; normalized version of input data in same shape
    """
    # Convert data if necessary
    data, d, l, m = convert_data(data)

    # Determine the length of the dimension along which to normalize
    length = data.shape[dim+1]

    # Get a reference to the respective row and divide it by its max 
    # value.
    for i in range(length) :
        if dim == 0 :
            row = data[:,i,:] 
        elif dim == 1 :
            row = data[:,:,i] 

        integral = sum(row[0])
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
                # consequently, # its integral are 0. Just leave the row 
                # unchanged in this case 
                print(e)
                pass

        # Copy the values back into the original data array
        if dim == 0 :
            data[:,i,:] = row
        elif dim == 1 :
            data[:,:,i] = row

    # Convert back to original shape, if necessary
    convert_data_back(data, d, l, m)

    return data

# +----------------+ #
# | BG subtraction | # ========================================================
# +----------------+ #

@array_input
def subtract_bg_fermi(data, n=10, ef=None, ef_index=None) :
    """ Use the mean of the counts above the Fermi level as a background and 
    subtract it from every channel/k. If no fermi level or index of the fermi
    level is specified, do the same as <func> subtract_bg_matt() but along EDCs
    instead of MDCs.

    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (1 x l x m) containing momentum in y (m momentum points) 
                  and energy along x (l energy points) (plotting amazingly 
                  inverts x and y)
    n           : int; number of smallest points to take in order to determine
                  bg

    Returns
    -------
    res         : np.array; bg-subtracted  version of input data in same shape
    """
    # Reshape input
    shape = data.shape
    if len(shape)==3 :
        l = shape[1] 
        m = shape[2]
        data = data.reshape([l,m])
    else :
        l = shape[0]
        m = shape[1]

    if ef_index == None :
    # TODO elif ef:
        if ef == None :
            ef_index = 0

    # Loop over every k
    for k in range(l) :
        edc = data[k,ef_index:]

        # Order the values in the edc by magnitude
        ordered = sorted(edc)

        # Average over the first (smallest) n points to get the bg
        bg = np.mean(ordered[:n])

        # Subtract the background (this updates the data array in place)
        edc -= bg
         
    return data

@array_input
def subtract_bg_matt(data, n=5) :
    """ Subtract background following the method in C.E.Matt's 
    "High-temperature Superconductivity Restrained by Orbital Hybridisation".
    Use the mean of the n smallest point in the spectrum for each energy.

    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (1 x l x m) containing momentum in y (m momentum points) 
                  and energy along x (l energy points) (plotting amazingly 
                  inverts x and y)
    n           : int; number of smallest points to take in order to determine
                  bg

    Returns
    -------
    res         : np.array; bg-subtracted  version of input data in same shape
    """
    # Reshape input
    shape = data.shape
    if len(shape)==3 :
        l = shape[1] 
        m = shape[2]
        data = data.reshape([l,m])
    else :
        l = shape[0]
        m = shape[1]

# Determine the number of energies l
#    shape = data.shape
#    l = shape[1] if len(shape)==3 else shape[0]

    # Loop over every energy
    for e in range(l) :
        mdc = data[e, :]

        # Order the values in the mdc by magnitude
        ordered = sorted(mdc)

        # Average over the first (smallest) n points to get the bg
        bg = np.mean(ordered[:n])

        # Subtract the background (this updates the data array in place)
        mdc -= bg
     
    return data

def subtract_bg_gold(data) :
    # @TODO Or should this be normalize?
    pass

@array_input
def subtract_bg_kaminski(data) :
    """ Use the method of Kaminski et al. (DOI: 10.1103/PhysRevB.69.212509) 
    to subtract background. The principle is as follows:
    A lorentzian + a linear background y(x) = ax + b is fitted to momentum
    distribution curves. One then uses the magnitude of the linear component 
    at every energy as the background at that energy for a given k point. 

    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (l x m x 1) containing momentum in y (m momentum points) 
                  and energy along x (l energy points) (plotting amazingly 
                  inverts x and y)

    Returns
    -------
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

# +--------------------------------+ #
# | Derivatieves/Data manipulation | # ========================================
# +--------------------------------+ #

def _derivatives(data, dx, dy) :
    """ Helper function to caluclate first and second partial derivatives of 
    2D data.

    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (1 x l x m)
    dx          : float; distance at x axis
    dy          : float; distance at y axis

    Returns
    -------
    grad_x, grad_y, grad2_x, grad2_y
                : np.arrays; first and second derivatives of data along x/y
                  in the same shape as the input array
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

@array_input
def laplacian(data, dx=1, dy=1, a=None) :
    """ Apply the second derivative (Laplacian) to the data.

    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (1 x l x m)
    dx          : float; distance at x axis
    dy          : float; distance at y axis
    a           : float; scaling factor for between x and y derivatives.  
                  Should be close to dx/dy. 

    Returns
    -------
    res         : np.array; second derivative of input array in same dimensions
 
    """
    # Get the partial derivatives
    grad_x, grad_y, grad2_x, grad2_y = _derivatives(data, dx, dy)

    # Get a default value for a
    a = dx/dy if a == None else a

    # Return the laplacian
    return a * grad2_x + grad2_y

def curvature(data, dx=1, dy=1, cx=1, cy=1) :
    """ Apply the curvature method (DOI: 10.1063/1.3585113) to the data.
 
    Parameters
    ----------
    data        : array-like; the input data with shape (l x m) or 
                  (1 x l x m)
    dx          : float; distance at x axis
    dy          : float; distance at y axis

    Returns
    -------
    res         : np.array; second derivative of input array in same dimensions
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
    by setting with 'recursion_level' to n.

    Parameters
    ----------
    x           : 1D array-like; the data to smooth
    n_box       : int; size of the smoothing box (i.e. number of points 
                  around the central point over which to take the mean). 
                  Should be an odd number.
    recursion_level
                : int; equals the number of times the smoothing is applied.

    Returns
    -------
    res         : np.array; smoothed data points
    """
    # Make the box. Sum(box) should equal 1 to keep the normalization (?)
    box = np.ones(n_box) / n_box

    # Let numpy do the work
    smoothened = np.convolve(x, box, mode='same')

    # Do it again (enter next recursion level) or return the result
    if recursion_level == 1 :
        return smoothened
    else :
        return smooth(smoothened, n_box, recursion_level - 1)

def smooth_derivative(x, n_box, n_smooth) :
    """ Apply linear smoothing to some data, take the derivative of the 
    smoothed curve, smooth that derivative and take the derivative again. 
    Finally, apply a last round of smoothing.
    
    Parameters
    ----------
    Same as in `func:arpys.postprocessing.smooth`.
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

def zero_crossings(x, direction=1) :
    """ Return the indices of the points where the data in x crosses 0, going 
    from positive to negative values (direction = -1), vice versa 
    (direction=1) or both (direction=0). This is detected simply by a change 
    of sign between two subsequent points.

    :NOTE: The functionality with `direction` is not yet implemented!

    Parameters
    ----------
    x           : 1D array-like; data in which to find zero crossings
    direction   : int, one of (-1, 0, 1); see above for explanation

    Returns
    -------
    crossings   : list; list of indices of the elements just before the 
                  crossings
    """
    # Initiate the container
    crossings = []

    for i in range(len(x) - 1) :
        i0 = x[i]
        i1 = x[i+1]
        if np.sign(i0) == 1 and np.sign(i1) == -1 :
            crossings.append(i)

    return crossings

# +------------------------------------------------+ #
# | Conversion from angular coordinates to k space | # =========================
# +------------------------------------------------+ #

def angle_to_k(angles, theta, phi, hv, E_b, work_func=4, c1=0.5124, 
               shift=0, lattice_constant=1, degrees=True) :
    """ Convert the angular information you get from beamline data into 
    proper k-space coordinates using the formula:
        pass


    Parameters
    ----------
    angles      : 1D array; angles (in degree) to be converted into k space
                  (currently equal theta_m
        pass
    """
    # Precalculate the prefactor (*lattice_constant to get lattice constant 
    # units)
    prefactor = c1 * np.sqrt(hv - work_func + E_b) * lattice_constant / np.pi

    # Apply a shift to the angles
    angles += shift

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


     
