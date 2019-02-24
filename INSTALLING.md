ARPYS installation instructions
===============================

Most of arpys'' dependencies (as listed in `requirements.txt`) can be fetched 
by `pip`. However, the GUI that arpys comes with (the *P*ython *I*mage *T*ool 
PIT) makes use of Qt5, which depends on SIP.
As of this writing (February 2019) SIP seemingly cannot be installed via pip.
The first step in installing arpys is therefore the compilation of SIP from 
source.
Once that is done, the rest of the installation can be carried out with a 
simply `pip install arpys`.


Installing SIP
--------------

1. Download the source for SIP from [here](https://www.riverbankcomputing.com/software/sip/download "SIP").
 
2. Extract the archive and enter the extracted directory (probably called 
`sip-4.19.14` or similar).

3. Run `python configure.py` to prepare for the compilation.

4. Run `make` to compile the code.

5. Run `make install` (potentially requires `sudo`) to get the files in the 
right places.

You should now have SIP installed. Try a `pip freeze` to see if it shows up.

Installing ARPYS
----------------

If SIP is installed, you *should* be able to just do `pip install arpys`.

An alternative possibility to this is to clone the [github 
repo](https://github.com/kuadrat/arpys) and (after SIP has been installed) to 
do `python setup.py install` from the top-level directory.
