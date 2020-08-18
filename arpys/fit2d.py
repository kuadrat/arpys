"""
Implementation of the 2D ARPES spectra fitting procedure as outlined by Li et 
al. in `Coherent organization of electronic correlations as a mechanism to 
enhance and stabilize high-Tc cuprate superconductivity` 
(DOI: 10.1038/s41467-017-02422-2).
"""
import multiprocessing
from datetime import datetime

import numpy as np
import scipy.integrate as integrate

from arpys import pp
import arpys.utilities.constants as const

# Boltzmann constant in eV/K
K_B_IN_eV_PER_K = 1.38064852e-23 / 1.6021766208e-19
K_B_IN_eV_PER_K = const.k_B / const.eV

def im_sigma_factory(lamb=1, T=0, i_step=0, e_step=0, w_step=0.01, i_gauss=0, 
                     e_gauss=0, w_gauss=0.01, offset=0) :
    """
    Factory to create functions that represent the imaginary part of the 
    self-energy. Confer the documentation of :func:`im_sigma 
    <arpys.fit2d.im_sigma>` for explanations of the parameters.
    The factory pattern is used here because a function for 
    im_sigma is needed in the Kramers-Kroning relations that give re_sigma.  

    Returns a function of the energy.

    :see also: 
        :func:`im_sigma <arpys.fit2d.im_sigma>`
        :func:`re_sigma <arpys.fit2d.re_sigma>`
    """
    def im_sigma(E) :
        res = lamb * np.sqrt(E**2 + (np.pi*K_B_IN_eV_PER_K*T)**2)
        res += i_step / (np.exp((E-e_step)/w_step) + 1)
        res += i_gauss * np.exp(-(E-e_gauss)**2/(2*w_gauss**2))
        return res + offset
    return im_sigma

def im_sigma(E, lamb=1, T=0, i_step=0, e_step=0, w_step=0.1, i_gauss=0, 
             e_gauss=0, w_gauss=0.1, offset=0) :
    """
    Imaginary part of the self-energy. It is parametrized as follows::

        im_sigma(E) = lamb * sqrt(E^2 + (pi*k*T)^2)

                                 i_step
                      + ----------------------------
                         exp((E-e_step)/w_step) + 1

                      + i_gauss * exp(-(E-e_gauss)^2 /(2*w_gauss))

                      + offset

    **Parameters**

    =======  ===================================================================
    E        float or 1d-array; binding energy/argument to the self-energy in eV
    lamb     float; coefficient of "standard" self-energy term
    T        float; temperature in K
    i_step   float; coefficient of step function
    e_step   float; energy at which step occurs
    w_step   float; width of the step function
    i_gauss  float; coefficient of Gaussian
    e_gauss  float; center energy of Gaussian
    w_gauss  float; width of Gaussian (sigma)
    offset   float; constant additive offset
    =======  ===================================================================

    All energies are given in eV.

    **Constants**

    ==  ========================================================================
    pi  3.14159...
    k   Boltzmann constant: 8.6173e-05 eV/K
    ==  ========================================================================
    """
    # Create the actual function using the factory
    kwargs = dict(lamb=lamb, T=T, i_step=i_step, e_step=e_step, 
                  w_step=w_step, i_gauss=i_gauss, e_gauss=e_gauss, 
                  w_gauss=w_gauss, offset=offset)
    im_sigma = im_sigma_factory(**kwargs)
    return im_sigma(E)

def re_sigma(E, im_sig, e0=-5, e1=5) :
    """
    Calculate the real part of the self-energy from its imaginary part using 
    the Kramers-Kroning relation::

                             e1
                             /  im_sigma(E')
        re_sigma(E) = 1/pi * |  ------------ dE'
                             /     E' - E
                            e0

    **Parameters**

    ======  ====================================================================
    E       float; energy at which to evaluate the self-energy
    im_sig  func; function for the imaginary part of the self-energy
    e0      float; lower integration bound
    e1      float; upper integration bound
    ======  ====================================================================
    """
    def integrand(e) :
        return im_sig(e) / (e-E)
    # The points [E, 0] are tricky. Give those as a hint to `quad`.
    # `full_output=True` suppresses annoying warnings
    res = integrate.quad(integrand, e0, e1, points=[E, 0], 
                          full_output=True)[0] / np.pi
    return res
    
def self_energy_factory(im_kwargs=dict(), re_kwargs=dict()) :
    """
    Combine real and imaginary parts of the self-energy to yield the full, 
    complex self-energy.

    Returns a function of the energy.

    **Parameters**

    =========  =================================================================
    im_kwargs  dict; keyword arguments to :func:`im_sigma_factory 
               <arpys.fit2d.im_sigma_factory>`
    re_kwargs  dict; keyword arguments to :func:`re_sigma 
               <arpys.fit2d.re_sigma_factory>`
    =========  =================================================================

    Confer respective documentations for further explanations on the 
    parameters.  
    """
    im_sig = im_sigma_factory(**im_kwargs)
    def self_energy(E) :
        re = re_sigma(E, im_sig, **re_kwargs)
        im = im_sig(E)
        return complex(re, im)
    return self_energy

def g11(k, E, sig, band, gap) :
    """
    Return the complex electron removal portion of the Green's function in 
    the Nambu-Gorkov formalism::

                                  E - sig(E) + band(k)
        g11(k, E) = -------------------------------------------------
                     (E-sig(E))^2 - band(k)^2 - gap*(1-Re(sig(E))/E)

    **Parameters**

    ====  ======================================================================
    k     array of length 2; k vector (in-plane) at which to evaluate g11
    E     float; energy at which to evaluate g11
    sig   func; function that returns the complex self-energy at E
    band  func; function that returns the bare band at k
    gap   func; function that returns the superconducting gap at k
    ====  ======================================================================
    """
    # Precalculate some values
    sig_E = sig(E)
    re_sig = sig_E.real
    e_diff = E - sig_E
    band_k = band(k)

    nominator = e_diff + band_k
    denominator = e_diff**2 - band_k**2 - gap(k) * (1-re_sig/E)

    return nominator/denominator

def g11_alt(k, E, sig, band, gap) :
    """
    Variation of :func:`g11 <arpys.fit2d.g11>` which takes precalculated 
    values of `sig`, `band` and `gap`.

    **Parameters**

    ====  ======================================================================
    k     array of length 2; k vector (in-plane) at which to evaluate g11
    E     float; energy at which to evaluate g11
    sig   complex; value of the complex self-energy at E
    band  float; value of the bare band at this k
    gap   float; value of the gap at this k
    ====  ======================================================================
    """
    # Precalculate some values
    re_sig = sig.real
    e_diff = E - sig

    nominator = e_diff + band
    denominator = e_diff**2 - band**2 - gap * (1-re_sig/E)

    return nominator/denominator

def arpes_intensity(k, E, i0, im_kwargs, re_kwargs, band, gap) :
    """
    Return the expected ARPES intensity at point (E,k) as modeled by::

                             g11(k, E)
        I_ARPES = i0 * (-Im -----------) * f(E, T)
                                pi

    Note that no broadening is applied.

    **Parameters**

    =========  =================================================================
    k          array of length 2; k vector (in-plane) at which to evaluate
    E          float; energy at which to evaluate ARPES intensity
    i0         float; global amplitude multiplier
    im_kwargs  dict; kwargs to :func:`im_sigma_factory 
               <arpys.fit2d.im_sigma_factory>`
    re_kwargs  dict; kwargs to :func: `re_sigma <arpys.fit2d.re_sigma_factory>`
    band       func; function that returns the bare band at k
    gap        func; function that returns the superconducting gap at k
    =========  =================================================================
    """
    # Try extracting T from im_kwargs
    try :
        T = im_kwargs['T']
    except KeyError :
        # Otherwise revert to default value
        T = 15

    # Calculate the self-energy
    self_energy = self_energy_factory(im_kwargs, re_kwargs)

    # Calculate the spectral function as the imaginary part of the Green's 
    # function
    A = -1/np.pi * g11(k, E, self_energy, band, gap).imag

    return i0 * A * pp.fermi_dirac(E, T=T)

def compute_self_energy_parallel(self_energy_func, n_proc, E) :
    """ Calculate the self energy more efficiently by splitting the work to 
    several subprocesses. Since multiprocessing.Pool cannot handle local 
    functions and lambdas, we have to do the job splitting by hand.

    **Parameters**

    ================  ==========================================================
    self_energy_func  func; function of E that returns the complex self-energy.
    n_proc            int; number of subprocesses to spawn (should be smaller 
                      or equal to the number of availabel cpus)
    E                 1d-array; energies at which to evaluate the self-energy.
    ================  ==========================================================

    **Returns**

    =============  =============================================================
    self_energies  1d-array of same length as *E*;
    =============  =============================================================

    This function simply splits the evaluation::

        self_energies = [self_energy_func(e) for e in E]

    into *n_proc* separate parts::

        self_energies = [self_energy_func(e) for e in E[i0:i1]] + 
                        [self_energy_func(e) for e in E[i1:i2]] + 
                        [self_energy_func(e) for e in E[i2:i3]] + 
                        ...

    all of which can be evaluated simultaneously by a different subprocess.
    """
    # Calculate the index ranges which will split up our energy array
    indices = [int(i*len(E)/n_proc) for i in range(n_proc+1)]
    # ind contains (i0, i1) tuples representing start and stop index for 
    # every sub-range
    ind = [(indices[i], indices[i+1]) for i in range(n_proc)]

    # Keep a list of spawned processes. Their outputs will be stored in a 
    # multiprocessing.Queue.
    processes = []
    outputs = multiprocessing.Queue()
    
    def loop(i, ind) :
        """ Calculate the self_energy for all energy values *E[i0:i1]* 
        where *ind*=(i0, i1). Write the result, along with the 
        corresponding subprocess index *i* into the Queue *outputs*. 
        """
        i0, i1 = ind
        self_energies = [self_energy_func(e) for e in E[i0:i1]]
        outputs.put((i, self_energies))

    # Create *n_proc* subprocesses that each calculate a portion of the 
    # self-energies.
    for i in range(n_proc) :
        p = multiprocessing.Process(target=loop, args=(i, ind[i]))
        processes.append(p)
    # Start all subprocesses...
    [p.start() for p in processes]
    # ...and wait until the last one has completed.
    [p.join() for p in processes]
    # Extract and stitch together the results into a numpy array.
    res = [outputs.get() for p in processes]
#    print([(r[0], len(r[1])) for r in res])
    res.sort()
    lists = [r[1] for r in res]
    self_energies = []
    for l in lists :
        self_energies += l
    return np.array(self_energies)

def arpes_intensity_alt(k, E, i0, im_kwargs, re_kwargs, band, gap, n_proc=1) :
    """
    Alternative implementation of :func:`arpes_intensity 
    <arpys.fit2d.arpes_intensity>` that is hopefully a bit faster.

    Return the expected ARPES intensity at point (E,k) as modeled by::

                             g11(k, E)
        I_ARPES = i0 * (-Im -----------) * f(E, T)
                                pi

    Note that no broadening is applied.

    **Parameters**

    =========  =================================================================
    k          2D array of shape (2,nk); k vectors (in-plane)
    E          array of length ne; energies at which to evaluate ARPES intensity
    i0         float; global amplitude multiplier
    im_kwargs  dict; kwargs to :func: `im_sigma_factory 
               <arpys.fit2d.im_sigma_factory>`
    re_kwargs  dict; kwargs to :func: `re_sigma <arpys.fit2d.re_sigma_factory>`
    band       func; function that returns the bare band at k
    gap        func; function that returns the superconducting gap at k
    =========  =================================================================

    **Returns**

    =========  =================================================================
    intensity  2D array of shape (ne, nk);
    =========  =================================================================
    """
    # Try extracting T from im_kwargs
    try :
        T = im_kwargs['T']
    except KeyError :
        # Otherwise revert to default value
        T = 15

#    start_prep = datetime.now()
    # Build the self-energy function and evaluate it
    self_energy_func = self_energy_factory(im_kwargs, re_kwargs)
    if n_proc==1 :
        self_energies = np.array([self_energy_func(e) for e in E])
    else :
        self_energies = compute_self_energy_parallel(self_energy_func, 
                                                     n_proc, E)
#    print('Preparations done in: {}'.format(datetime.now()-start_prep))

    # Precalculate as much as possible
    ne = len(E)
    nk = k.shape[1]
    intensity = np.zeros([ne, nk])
    fermi_dirac = pp.fermi_dirac(E, T=T)

    # Calculate the spectral function as the imaginary part of the Green's 
    # function
#    start_loop = datetime.now()
    for i in range(nk) :
        this_k = k[:,i]
        this_band = band(this_k)
        this_gap = gap(this_k)
        for j,e in enumerate(E) :
            fdj = fermi_dirac[j]
            a = g11_alt(this_k, e, self_energies[j], this_band, this_gap)
            intensity[j,i] += fdj * i0*a.imag/np.pi
#    print('Loop finished in: {}'.format(datetime.now()-start_loop))

    return intensity

def band_factory(bottom, m_e=1) :
    """
    Create a function that represents a parabolic band with band bottom at 
    energy `bottom`.

    **Parameters**

    ======  ====================================================================
    bottom  float; energy of the band bottom in eV, measured from the Fermi 
            level.
    m_e     float; effective electron mass in units of electron rest mass. Tunes
            the opening of the parabola.
    ======  ====================================================================

    **Returns**

    ====  ======================================================================
    band  func; a function of a length 2 array `k` that returns the energy of a 
          band at given `k`. `k` should be given in inverse Angstrom.
    ====  ======================================================================
    """
    def band(k) :
        """ Return thr energy of a parabolic band at given `k`.
        `k` has to be an array of length 2 containing the parallel and 
        perpendicular components of the in-plane wave vector given in inverse 
        Angstrom (1 Angstrom = 1e-10 m).
        """
        # Unit conversion to eV
        conversion = 1e20/const.eV
        k2 = k[0]**2 + k[1]**2
        return conversion * const.h**2 * k2 / (4*np.pi**2*m_e*const.m_e) + bottom
    return band

#class SpectrumFit() :
#    """
#    The `SpectrumFit` object handles parameters and parametrizations as well 
#    as the calculations necessary in fitting ARPES spectra to a model as 
#    suggested in Li et al. [*].
#
#    [*] DOI: 10.1038/s41467-017-02422-2
#    """
#    # Default values
#    default_values = dict(i0 = 1,
#                          m_e = 1,
#                          band_bottom = -0.5,
#                          T = 1,
#                          lamb = 0,
#                          i_step = 0,
#                          e_step = 0,
#                          w_step = 1e-3,
#                          i_gauss = 0,
#                          e_gauss = 0,
#                          w_gauss = 1e-3,
#                          offset = 0
#                         )
#
#    def __init__(self, **kwargs) :
#        """ """
#        for key,val in kwargs.items() :
#            self.__setattr__(key, val)


if __name__=="__main__" :
    import matplotlib.pyplot as plt
    from datetime import datetime

    im_kwargs = dict(T=15,
                     lamb=0.05,
                     i_step=0,
                     e_step=-0.1,
                     i_gauss=0,
                     e_gauss=-0.25,
                     offset = 0.05)
    re_kwargs = dict()

    i0 = 1

    nk = 80
    kmin = 0
    kmax = 1
    ks = np.array([np.linspace(kmin, kmax, nk), np.zeros(nk)])
    ne = 100
    emin = -1.2
    emax = 0.1
    es = np.linspace(emax, emin, ne)

    def gap(k) :
        return 0.0

    def band(k) :
#        return 0*k[0]+0.8*emin
        return (emax-emin)/(kmax-kmin)**2*k[0]**2 + 0.8*emin

    band_bottom = 0.8*emin
    m_e = 1
    band = band_factory(band_bottom, m_e)
    band2 = band_factory(0.5*emin, 5*m_e)

    self_energy = self_energy_factory(im_kwargs, re_kwargs)

    intensity = np.zeros([ne, nk])
    real_part = np.zeros([ne, nk])

    tstart = datetime.now()

    # Precalculate as much as possible
    self_energies = np.array([self_energy(e) for e in es])
    fd = pp.fermi_dirac(es, T=im_kwargs['T'])

    for i in range(nk) :
        k = ks[:,i]
        this_band = band(k)
        this_band2 = band2(k)
        this_gap = gap(k)
        for j,e in enumerate(es) :
#            intensity[j,i] = -arpes_intensity(k, e, i0, im_kwargs, re_kwargs, 
#                                              band, gap)
            fdj = fd[j]
            a = g11_alt(k, e, self_energies[j], this_band, this_gap)
            intensity[j,i] += i0*a.imag/np.pi * fdj
            real_part[j,i] += i0*a.real/np.pi * fdj

            a2 = g11_alt(k, e, self_energies[j], this_band2, this_gap)
            intensity[j,i] += i0*a2.imag/np.pi * fdj
            real_part[j,i] += i0*a2.real/np.pi * fdj

    tend = datetime.now()
    print(tend-tstart)

    k_abs = np.sqrt(ks[0]**2 + ks[1]**2)

    fig = plt.figure()
    ax_im = fig.add_subplot(121)
    ax_re = fig.add_subplot(122)
    b = []
    b2 = []
    for k in ks.T :
        b.append(band(k))
        b2.append(band2(k))
    for ax in [ax_im, ax_re] :
        ax.plot(k_abs, b, 'r--', dashes=(3,10), lw=1)
        ax.plot(k_abs, b2, 'k--', dashes=(3,10), lw=1)
        ax.set_ylim([emin, emax])
    ax_im.pcolormesh(k_abs, es, intensity)
    ax_re.pcolormesh(k_abs, es, real_part)
    """
    b = band_factory(-2)
    ks = np.arange(-2, 2, 0.05)
    res = [b(np.array([k, 0])) for k in ks]
    plt.plot(ks, res)
    """

    plt.show()

