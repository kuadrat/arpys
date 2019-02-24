#!/usr/bin/python
description = '''
Use the result from a wien2k DFT calculation to create band-character 
plots. All information is extracted from the wien2k files `CASE.qtl` and 
`CASE.klist_band`.
'''
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# +------------+ #
# | Parameters | # =============================================================
# +------------+ #

# Rydberg energy
ryd = 13.605693

# Plotting kwargs
band_kwargs = dict(color='grey', linestyle='-', linewidth=1, zorder=1)
fermi_kwargs = dict(color='black', linestyle=':', linewidth=1, zorder=2)

# Factor for thickness of band characters
character_scale = 12
character_color = 'red'

# Minimum character required to be considered worthy for plotting
#min_character = character_scale * 0.01
min_character = 0.01

# Define some special points (kx, ky, kz in units of pi/lattice constant)
# Assuming tetragonal lattice
special_points = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.5]
    ]
special_names = [
    'Gamma',
    'X',
    'M',
    'Z',
    'R',
    'A'
    ]

# +-----------+ #
# | Functions | # ==============================================================
# +-----------+ #

def read_klist_band(klist_band) :
    """
    Read a CASE.klist_band file and extract the k vectors from it. 

    Parameters
    ----------
    klist_band  : str; path/filename of the CASE.klist_band file.

    Returns
    -------
    ks          : list; list of [kx, ky, kz] lists in units of pi/(a,b,c).
    """
    ks = []
    with open(klist_band) as f :
        for line in f :
            l = line.split()
            try :
                k = [ float(i)/float(l[3]) for i in l[0:3] ]
            except ValueError :
                # Some lines start with the name of the special point
                k = [ float(i)/float(l[4]) for i in l[1:4] ]
            ks.append(k)

    # The last line in CASE.klist_bands usually contains `END` which leads to 
    # an empty list being appendes to `ks`. Pop it off.
    if ks[-1] == [] : ks.pop()

    return ks

def read_qtl(qtl) :
    """ Read a wien2k CASE.qtl file (generated from `lapw2 -band -qtl`) and 
    extract energy eigenvalues, orbital characters and additional information.

    Parameters
    ----------
    qtl             : str; path/filename of CASE.qtl file.

    Returns
    -------
    data            : argparse.Namespace; a container which enables easy 
                      access to the numerous variables returned by this 
                      function, as described in the following.

    data.energy     : list of floats; energy eigenvalues in eV. Has length 
                      n_k * n_bands.

    data.n_bands    : int; number of distinct bands found.

    data.n_k        : int; number of k values in the k-path.

    data.n_atoms    : int; number of nonequivalent atoms in the compund.

    data.n_orbitals : list of ints; contains the number of distinct orbitals 
                      for every atom.

    data.orbitals   : list of lists of str; the w2k names of the orbitals for 
                      every atom. 0, 1, 2, 3 stand for s, p, d, f

    data.multiplicities
                    : list of int; the multiplicities of each inequivalent 
                      atom.  

    data.character  : list of lists of lists of floats; for every atom, for 
                      every energy value, there is a list of orbital 
                      characters for every orbital in that atom at that 
                      energy. Access via character[atom][energy][orbital].

    a, b, c         : floats; lattice parameters in units of Bohr radii

    E_F             : float; Fermi energy in units of Rydberg (13.6 eV).
    """
    # Initialize containers
    energy = []
    n_bands = 1
    orbitals = []
    character = []
    multiplicities = []

    with open(args.infile, 'r') as f :
        # Read the preamble, i.e. read up to the point of te first BAND
        for line in f :
            if line.startswith(' BAND') : 
                break
            elif line.startswith(' LATTICE') :
                # Extract lattice constants (in units of a_Bohr) and Fermi 
                # energy (in Rydberg)
                l = line.split()
                a, b, c = [float(i) for i in l[2:5]]
                E_F = float(l[-1]) 
            elif line.startswith(' JATOM') :
                # Extract info on the multiplicities and the present orbitals 
                # for the atoms in the compound
                l = line.split()
                multiplicities.append(int(l[3]))
                this_atoms_orbitals = l[-1].split(',')
                orbitals.append(this_atoms_orbitals)

        # Number of atoms and number of orbitals for each atom
        n_atoms = len(multiplicities)
        n_orbitals = [len(O) for O in orbitals]

        # Create sublists for every atom, plus one for unassigne electrons
        for i in range(n_atoms + 1) :
            character.append([])

        # Read the beands (the file object remembers where we left the last 
        # for-loop)
        for line in f :
            # Skip the `BAND X` lines, but use them to count the number of bands
            if line.startswith(' BAND') :
                n_bands += 1
                continue

            l = line.split()
            atom = int(l[1])
            # Every (n_atoms+1)'th line just has the percentage of unassigned 
            # electrons
            if atom == n_atoms+1 :
                # Store the energy (shifted by Fermi level and converted to eV) 
                # and unassigned contributions
                energy.append((float(l[0]) - E_F)*ryd)
                character[-1].append(float(l[2]))
            else :
                # In the normal case, store the characters of all orbitals for 
                # this atom
                chars = [float(char) for char in l[2:]]
                character[atom-1].append(chars)
                
    # Number of k points per band
    n_k = int( len(energy)/n_bands )

    # Build and return the Namespace object
    data = argparse.Namespace()
    data.energy = energy
    data.n_bands = n_bands
    data.n_k = n_k
    data.n_atoms = n_atoms
    data.multiplicities = multiplicities
    data.n_orbitals = n_orbitals
    data.orbitals = orbitals
    data.character = character
    data.a = a
    data.b = b
    data.c = c
    data.E_F = E_F

    return data

def find_special_points(ks, special_points=special_points, 
                        special_names=special_names) :
    """ Look for special points in a list of [kx, ky, kz] k-points and 
    collect them in a list of (index, name) tuples, if they exist.

    Parameters
    ----------
    ks      : list of [kx, ky, kz] lists; k-points representing a path in 
              reciprocal space.
    special_points
            : list of [kx, ky, kz] lists; special points to identify in `ks`. 
              Should have the same units as the points in `ks`.
    special_names
            : list of str; names of the points in `special_points`.

    Returns
    -------
    special_ks : list of (index, name) tuples; indices in `ks` where the 
                 special points with the respective names lie.
    """
    special_ks = []
    # Try to detect the special points in our ks
    for i,k  in enumerate(ks) :
        # Retain a copy of the original
        K = list(k)
        for j,point in enumerate(special_points) :
            # Rotate around 90 degrees to cover equivalent points
            for _ in range(4) :
                K = [ -K[1], K[0], K[2] ]
                if K == point :
                    name = special_names[j]
                    special_ks.append( (i, name) )
                    break

    return special_ks

def write_character_file(name, ks, energies, characters, n_bands, n_k) :
    """ Create a file containing columns of kx, ky, kz, E and the character 
    of a specific band (the name of which should be reflected in the 
    filename). """
    fmt = '{: .10f} '
    s = '{:>10} ' + 5*fmt + '\n'
    header = '#{:>9}' + 5*'{:>14}' + '\n'
    header = header.format('k index', 'kx (pi/a)', 'ky (pi/b)', 'kz (pi/c)', 
                           'E (eV)', 'character %')
    with open(name, 'w+') as f :
        f.write(header)
        for band in range(n_bands) :
            f.write('# Band {}\n'.format(band))
            for k in range(n_k) :
                i = band*n_k + k
                line = s.format(k, ks[k][0], ks[k][1], ks[k][2], energies[i], 
                                characters[i])
                f.write(line)

# +------+ #
# | Main | # ===================================================================
# +------+ #

if __name__ == '__main__' :
    # +---------------+ #
    # | Set up parser | # ======================================================
    # +---------------+ #

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('infile', type=str,
                        help='The CASE.qtl file, containing all the \
                        information we need.')

    parser.add_argument('-e', '--emin', type=float, default=-2,
                        help='Minimum energy (in eV, measured from E_F) for \
                        plot.')

    parser.add_argument('-E', '--emax', type=float, default=0.1,
                        help='Maximum energy (in eV, measured from E_F) for \
                        plot.')

    parser.add_argument('-d', '--savedir', type=str, default='.',
                        help='Path to directory where png`s and .dat files \
                        will be stored.')

    parser.add_argument('-o', '--output', default=False, action='store_true',
                        help='Toggle creation of .dat files.')

    parser.add_argument('-p', '--png', default=True, action='store_true',
                        help='Toggle creation of .png files.')

    parser.add_argument('-D', '--dpi', type=int, default=150,
                        help='dpi value for the output png.')

    parser.add_argument('-v', action='count', default=0,
                        help='Increase verbosity.')

    args = parser.parse_args()

    # +-------------------+ #
    # | Read in k vectors | # ==================================================
    # +-------------------+ #

    # Build the filename of the CASE.klist_band file
    case = args.infile.split('.')[0]
    klist_band = case + '.klist_band'
    if args.v : print('Reading klist file {}.'.format(klist_band))
    ks = read_klist_band(klist_band)

    # +--------------+ #
    # | Read in data | # =======================================================
    # +--------------+ #

    if args.v : print('Reading header of {}.'.format(args.infile))
    data = read_qtl(args.infile)
    energy = data.energy
    n_bands = data.n_bands
    n_k = data.n_k
    n_atoms = data.n_atoms
    multiplicities = data.multiplicities
    n_orbitals = data.n_orbitals
    orbitals = data.orbitals
    character = data.character
    a = data.a
    b = data.b
    c = data.c
    E_F = data.E_F

    # Print some info on the read data
    if args.v :
        message = '''
    a, b, c = {}, {}, {} (units of Bohr radius)
    E_F = {} Ry
    n_bands = {}
    n_atoms = {}
    n_orbitals = {}
    orbitals:
'''.format(a, b, c, E_F, n_bands, n_atoms, n_orbitals)
        for atom in range(n_atoms) :
            message += 'atom {}: '.format(atom+1)
            for orbital in range(n_orbitals[atom]) :
                message += orbitals[atom][orbital] + ' '
            message += '\n'
        message += '''
    len(energy) = {}
    len(energy)/n_bands = {}
        '''.format(len(energy), n_k)
        print(message)

    # +----------------+ #
    # | Analyze k-path | # =========================================================
    # +----------------+ #

    special_ks = find_special_points(ks, special_points, special_names)
    if args.v :
        for i,name in special_ks :
            print('k {} ({}) is equivalent to {}'.format(i, ks[i], name))
    # The number of k points in each section is already proportional to the 
    # section length, if the klist_band file has been generated by xcrysden

    # +------+ #
    # | Plot | # ===============================================================
    # +------+ #

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = range(n_k)

    # Plot band characters
    if args.v : print('Plotting bands/writing files...')
    # Iterate over all atoms and orbitals
    for atom in range(n_atoms) :
        characters = np.array(character[atom])
        for j,orbital in enumerate(orbitals[atom]) :
            # Get the character - account for multiplicity
            char = character_scale*characters[:,j]
            if j!=0 :
                multiplicity = multiplicities[atom]
                char *= multiplicity

            # Build the atom-orbital name
            name = '{}_{}'.format(atom+1, orbital)

            save_this = False
            for i in range(n_bands) :
                band_start = i * n_k
                band_end = (i+1) * n_k
                E = energy[band_start:band_end]
                # Skip this band if it isn't even visible in the final plot
                if max(E) < args.emin or min(E) > args.emax : continue

                # Plot the raw bands
                if args.png :
                    ax.plot(x, E, **band_kwargs)

                this_char = char[band_start:band_end]

                # Nothing to do if no character visible in our view
                if this_char.max() < character_scale*min_character : continue
                
                # If we reach this point at least once, save the fig
                save_this = True

                # Some magic to create line thickness proportional to the 
                # band character
                if args.png :
                    points = np.array([x, E]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], 
                                              axis=1) 
                    lc = LineCollection(segments, linewidths=this_char,
                                        color=character_color)
                    ax.add_collection(lc)

            # If there is nothing to show, move on to the next orbital
            if not save_this : continue

            print(name)

            # Write the k-character data to file for access by other 
            # programs
            if args.output :
                filename = args.savedir+'/'+name+'.dat'
                write_character_file(filename, ks, energy, char, n_bands, 
                                     n_k) 
                
            if args.png :
                # Plot Fermi level
                ax.plot((x[0], x[-1]), (0, 0), **fermi_kwargs)

                # Ticks, limits and title
                ax.set_title(name)
                ax.set_xticks([sp[0] for sp in special_ks])
                ax.set_xticklabels([sp[1] for sp in special_ks])
                ax.set_ylim([args.emin, args.emax])

                # Save the figure
                outfilename = '{}/{}.png'.format(args.savedir, name)
                fig.savefig(outfilename, format='png', dpi=args.dpi)

                # Clear the figure for the next band character
                ax.clear()

    #plt.show()

