"""
A module that provides custom functions.
"""
import numpy as np

def testarray(*dims) :
    """
    Return a multidimensional array with increasing integer entries.

    **Parameters**

    ====  ======================================================================
    dims  tuple of int; the lengths along each dimension.
    ====  ======================================================================
    """
    #ndim = len(dims)
    # Find the number of entries the resulting array will have
    n = 1
    for d in dims :
        n *= d
    
    # Create the base, 1-dimensional sequence
    res = np.arange(n)

    # Bring it to the desired dimension
    return res.reshape(dims)

def describe(obj) :
    """ Exploratory introspection function that outputs information on 
    allattributes of an object.
    Tends to print a lot of output, so it might be advised to pipe or send it 
    to a file over which one can easily search.
    """
    for key in dir(obj) :
        try :
            val = getattr(obj, key)
        except AttributeError :
            continue
        if callable(val) :
            help(val)
        else :
            print('{k} => {v!r}'.format(k=key, v=val))
        print(80*'-')

def polynomial(n=1) :
    """ Return a polynomial function of order *n*. 
    
    **Parameters**

    =  =========================================================================
    n  positive integer; the degree of the returned polynomial function.
    =  =========================================================================

    **Returns**

    =  =========================================================================
    p  callable; a function that takes an independent variable *x* as its 
       first argument and *n* more arguments representing the coefficients of 
       the *n*'th order polynomial in increasing order.
    =  =========================================================================
    """
    def p(x, *a) :
        """ Polynomial function of order *n* of the variable *x* with 
        parameters *a*.
        *a* is a list of length *n* corresponding to the coefficients of of 
        the different terms in increasing order.
        """
        return sum([a[i]*x**i for i in range(n)])
    return p
        
def multiple_exponential(t, i=0, *A_TAU) :
    """
    Recursively build a function of the form::

        f(t) = a1 * exp(-t/tau1) + a2 * exp(-t/tau2) + ... + an * exp(-t/taun)

    and return its value at *t*.
    *A_TAU* must have length 2n, where n is the number of exponentials
    you want in the sum. t can be any argument that could be passed to the 
    np.exp function.

    The reason some of this seems weirdly implemented is because this function
    has been made compatible with scipy.optimize.curve_fit which unpacks 
    parameters before passing them on to the fit function.
    Even now, some care has to be taken when passing arguments to this function
    directly - they probably have to be unpacked with the * operator, e.g.::

        >>> multiple_exponential(np.array([1,2,3]), 0, *[0.5, 1, 0.5, 1])

    **Parameters**

    =====  =====================================================================
    t      array_like; input (x-values) to the resulting function.
    i      int; iteration level
    A_TAU  list of length 2n; the first n entries are amplitude parameters 
           for the different terms and the last n entries are the decay times 
           of each term
    =====  =====================================================================
    """
    order = int(len(A_TAU)/2)
    i = int(i)
    if i < order-1 :
        return A_TAU[i] * np.exp(-t/A_TAU[order+i]) + multiple_exponential(t, i+1, *A_TAU)
    else :
        return A_TAU[i] * np.exp(-t/A_TAU[order+i])

def stretched_exponential(t, A, tau, h) :
    """
    Implement stretched exponentail decay:

        f(t) = A * exp( - (t/tau)^(1/h) )
    """
    return A * np.exp(-(t/tau)**(1./h))

def lorentzian(x, a0, xmax, gamma) :
    r"""
    .. math::
        a_0 \cdot \frac{\Gamma^2}{(x-x_\mathrm{max})^2 + \Gamma^2}

    **Parameters**

    =====  =====================================================================
    x      float or array; the variable.
    a0     float; normalization factor.
    xmax   float; peak position of the Lorentzian in *x* units. 
    gamma  float; peak width in *x* units.
    =====  =====================================================================
    """
    return a0 * gamma*2 / ((x - xmax)**2 + gamma**2)

def chi2(y, f, sigma, normalize=True) :
    """
    Return the chi-square value for a series of measured values y and 
    uncertainties sigma that are described by model f::

                        y_i - f_i
        chi2 = sum_i( (-----------)^2 )
                         sigma_i

    Can be normalized by the number of entries.

    **Parameters**

    =========  =================================================================
    y          np.array; values of the data which is to be described
    f          np.array; values predicted by the model
    sigma      np.array or float; uncertainties on values y
    normalize  boolean; whether to divide the result by the length of y or not
    =========  =================================================================
    """
    res = sum( ( (y-f)/sigma )**2 )
    res = res/len(y) if normalize else res
    return res

def indexof(value, array) :
    """ 
    Return the first index of the value in the array closest to the given 
    *value*.

    Example::

        >>> a = np.array([1, 0, 0, 2, 1])
        >>> indexof(0, a)
        1
        >>> indexof(0.9, a)
        0
    """
    return np.argmin(np.abs(array - value))

def sort_together(*lists) :
    """ 
    Sort the given lists by the elements in the first list.

    Example::

        >>> l0 = [3, 1, 2]
        >>> l1 = ['b', 'c', 'a']
        >>> l2 = [11, 22, 33]
        >>> sort_together(l0, l1, l2)
        [[1, 2, 3], ['c', 'a', 'b'], [22, 33, 11]]
        >>> sort_together(l1, l0, l2)
        [['a', 'b', 'c'], [2, 3, 1], [33, 11, 22]]
    """
    return [list(x) for x in zip(*sorted(zip(*lists)))]

