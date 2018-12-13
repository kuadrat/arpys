
import numpy as np
import scipy.integrate as integrate

import postprocessing as pp

# Boltzmann constant in eV/K
K_B_IN_eV_PER_K = 1.38064852e-23 / 1.6021766208e-19

def im_sigma_factory(lamb=1, T=0, i_step=0, e_step=0, w_step=0.01, i_gauss=0, 
                     e_gauss=0, w_gauss=0.01, offset=0) :
    """
    Factory to create functions that represent the imaginary part of the 
    self-energy. Confer the documentation of :func: `im_sigma 
    <arpys.2dfit.im_sigma>` for explanations of the parameters.
    The factory pattern is used here because a function for 
    im_sigma is needed in the Kramers-Kroning relations that give re_sigma.  

    Returns a function of the energy.

    ..:see also: :func: `im_sigma <arpys.2dfit.im_sigma>`
    ..:see also: :func: `re_sigma <arpys.2dfit.re_sigma>`
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
    Imaginary part of the self-energy. It is parametrized as follows:

        im_sigma(E) = lamb * sqrt(E^2 + (pi*k*T)^2)

                                 i_step
                      + ----------------------------
                         exp((E-e_step)/w_step) + 1

                      + i_gauss * exp(-(E-e_gauss)^2 /(2*w_gauss))

                      + offset

    *Parameters*
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

    *Constants*
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
    the Kramers-Kroning relation:

                             e1
                             /  im_sigma(E')
        re_sigma(E) = 1/pi * |  ------------ dE'
                             /     E' - E
                            e0

    *Parameters*
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

    *Parameters*
    =========  =================================================================
    im_kwargs  dict; keyword arguments to :func: `im_sigma_factory 
               <arpys.2dfit.im_sigma_factory>`
    re_kwargs  dict; keyword arguments to :func: `re_sigma 
               <arpys.2dfit.re_sigma_factory>`
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
    the Nambu-Gorkov formalism:

                                  E - sig(E) + band(k)
        g11(k, E) = -------------------------------------------------
                     (E-sig(E))^2 - band(k)^2 - gap*(1-Re(sig(E))/E)

    *Parameters*
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
    Variation of :func: `g11 <arpys.2dfit.g11>` which takes precalculated 
    values of `band` and `gap`.

    *Parameters*
    ====  ======================================================================
    k     array of length 2; k vector (in-plane) at which to evaluate g11
    E     float; energy at which to evaluate g11
    sig   func; function that returns the complex self-energy at E
    band  float; value of the bare band at this k
    gap   float; value of the gap at this k
    ====  ======================================================================
    """
    # Precalculate some values
    sig_E = sig(E)
    re_sig = sig_E.real
    e_diff = E - sig_E

    nominator = e_diff + band
    denominator = e_diff**2 - band**2 - gap * (1-re_sig/E)

    return nominator/denominator

def arpes_intensity(k, E, i0, im_kwargs, re_kwargs, band, gap) :
    """
    Return the expected ARPES intensity at point (E,k) as modeled by:

                             g11(k, E)
        I_ARPES = i0 * (-Im -----------) * f(E, T)
                                pi

    Note that no broadening is applied.

    *Parameters*
    =========  =================================================================
    k          array of length 2; k vector (in-plane) at which to evaluate
    E          float; energy at which to evaluate ARPES intensity
    i0         float; global amplitude multiplier
    im_kwargs  dict; kwargs to :func: `im_sigma_factory 
               <arpys.2dfit.im_sigma_factory>`
    re_kwargs  dict; kwargs to :func: `re_sigma <arpys.2dfit.re_sigma_factory>`
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

if __name__=="__main__" :
    import matplotlib.pyplot as plt
    from datetime import datetime

    im_kwargs = dict(T=15,
                     lamb=0.0,
                     i_step=0,
                     e_step=-0.1,
                     e_gauss=-0.25,
                     i_gauss=0,
                     offset = 0.08)
    re_kwargs = dict()

    i0 = 1

    nk = 30
    kmin = 0
    kmax = 1
    ks = np.array([np.linspace(kmin, kmax, nk), np.zeros(nk)])
    ne = 10
    emin = -0.5
    emax = 0.1
    es = np.linspace(emax, emin, ne)

    def gap(k) :
        return 0

    def band(k) :
#        return 0*k[0]+0.8*emin
        return (emax-emin)/(kmax-kmin)**2*k[0]**2 + 0.8*emin

    self_energy = self_energy_factory(im_kwargs, re_kwargs)

    intensity = np.zeros([ne, nk])

    tstart = datetime.now()
    for i in range(nk) :
        k = ks[:,i]
#        this_band = band(k)
#        this_gap = gap(k)
        for j,e in enumerate(es) :
            intensity[j,i] = -arpes_intensity(k, e, i0, im_kwargs, re_kwargs, 
                                              band, gap)
#            a = g11_alt(k, e, self_energy, this_band, this_gap)
#            intensity[j,i] += i0*a.imag/np.pi
    tend = datetime.now()
    print(tend-tstart)

    k_abs = np.sqrt(ks[0]**2 + ks[1]**2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_abs, band(ks), 'r-')
    ax.pcolormesh(k_abs, es, intensity)

    plt.show()

