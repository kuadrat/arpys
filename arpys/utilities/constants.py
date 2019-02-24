#!/usr/bin/python
"""
A database like file containing numerical values of various natural constants
in SI units. Alphabetically ordered by name (not by variable name).
"""

# Constants
#==============================================================================

# Avogadro constant [1]
N_A = 6.022e23

# Bohr's Magneton [J/T]
mu_bohr = 9.274009994e-24

# Bohr radius [m]
a0 = 5.2917721067e-11

# Boltzmann constant [J/K]
k_B = 1.38064852e-23

# Dielectric constant in vacuum [C / V / m]
eps0 = 8.854e-12

# Electronvolt [J]
eV = 1.6021766208e-19

# Electron mass [kg]
m_e = 9.10938356e-31

# Planck's constant [J * s]
h = 6.626070040e-34

# Speed of light [m / s]
c = 299792458.

# Universal Gas constant [J / K / mol]
R = 8.3144598


# Dependent constants
#==============================================================================

# Electronvolt-nanometer conversion for light; 1239.84197
eV_nm_conversion = h*c/eV*1e9


# Utilities
#==============================================================================

def convert_eV_nm(eV_or_nm) :
    """
    Convert between electronvolt and nanometers for electromagnetic waves.
    The conversion follows from E = h*c/lambda and is simply:

        nm_or_eV = 1239.84193 / eV_or_nm

    Inputs
    ------
    eV_or_nm    : float; value in electronvolts or nanometers to be converted.


    Outputs
    -------
    nm_or_eV    : float; if eV were given, this is the corresponding value in
                  nanometers, or vice versa.
    """
    nm_or_eV = eV_nm_conversion / eV_or_nm
    return nm_or_eV

