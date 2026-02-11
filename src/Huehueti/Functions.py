"""Transform units functions"""
import numpy as np
import pytensor
import pytensor.tensor as pt

# Set configuration
floatX = pytensor.config.floatX

def m2flux(m_true):
    """Convert relative magnitud value to flux.

        F = F0 * 10^(-m/2.5)

    Units:
        flux: [erg/s/cm^2] 
        mag : [1]
        ZP_nu : [erg/s/cm^2] <- svo2.cab.inta-csic.es/
    """

    ZeroPoints = {
        'GAIA/GAIA3.G' : 2.5e-9,         # g
        'GAIA/GAIA3.Gbp' : 4.08e-9,      # bp
        'GAIA/GAIA3.Grp' : 1.27e-9,      # rp
        '2MASS/2MASS.J' : 3.13e-10,      # J
        '2MASS/2MASS.H' : 1.13e-10,      # H
        '2MASS/2MASS.Ks' : 4.28e-11,     # K
        'PAN-STARRS/PS1.g' : 5.05e-9,    # gmag
        'PAN-STARRS/PS1.r' : 2.47e-9,    # rmag
        'PAN-STARRS/PS1.i' : 1.36e-9,    # imag
        'PAN-STARRS/PS1.y' : 7.05e-10,	 # ymag
        'PAN-STARRS/PS1.z' : 9.01e-10    # zmag
    }

    F0 = np.array([ZeroPoints[x]  for x in ZeroPoints]).astype(floatX)

    return F0*10**(-0.4*m_true)

def flux2m(flux_obs):
    """Convert flux value to relative magnitud.

        m = -2.5 log (F/F0)
    
    Units:
        flux: [erg/s/cm^2] 
        mag : [1]
        ZP_nu : [erg/s/cm^2] <- svo2.cab.inta-csic.es/
    """

    ZeroPoints = {
        'GAIA/GAIA3.G' : 2.5e-9,         # g
        'GAIA/GAIA3.Gbp' : 4.08e-9,      # bp
        'GAIA/GAIA3.Grp' : 1.27e-9,      # rp
        '2MASS/2MASS.J' : 3.13e-10,      # J
        '2MASS/2MASS.H' : 1.13e-10,      # H
        '2MASS/2MASS.Ks' : 4.28e-11,     # K
        'PAN-STARRS/PS1.g' : 5.05e-9,    # gmag
        'PAN-STARRS/PS1.r' : 2.47e-9,    # rmag
        'PAN-STARRS/PS1.i' : 1.36e-9,    # imag
        'PAN-STARRS/PS1.y' : 7.05e-10,	 # ymag
        'PAN-STARRS/PS1.z' : 9.01e-10    # zmag
    }

    F0 = np.array([ZeroPoints[x]  for x in ZeroPoints]).astype(floatX)

    return -2.5*np.log10(flux_obs/F0)

def absolute_to_apparent(M, distance,n_bands):
    """Convert absolute magnitude to relative magnitude.
    m = 

    Units: 
        distance : [pc]
        M : [1]
        m : [1] 
    """
    distance_v = pt.stack([distance for _ in range(M.shape[1])], axis=1)
    return M + 5.*pt.log10(distance_v) - 5.0

def distance2parallax(distance):
    """Convert distance to parallax.

    Units:
        distance: kiloparsecs (kpc)
        parallax: milliarcseconds (mas)
    """
    return 1/distance

def apparent_to_absolute(m, distance):
    """Convert apparent to absolute
    M = m - 5*log10(distance) + 5
    """
    return m - 5.*np.log10(distance) + 5.



