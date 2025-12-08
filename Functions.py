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
    distance_v = pt.stack([distance for _ in range(n_bands)], axis=1)
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


def relu(x, alpha=0):
    """
    Compute the element-wise rectified linear activation function.

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : `scalar or tensor, optional`
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.

    Notes
    -----
    This is numerically equivalent to ``pt.switch(x > 0, x, alpha * x)``
    (or ``pt.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.

    """
    if alpha == 0:
        return 0.5 * (x + abs(x))
    else:
        # We can't use 0.5 and 1 for one and half.  as if alpha is a
        # numpy dtype, they will be considered as float64, so would
        # cause upcast to float64.
        alpha = pt.as_tensor_variable(alpha)
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * abs(x)
