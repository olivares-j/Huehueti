"""Probabilistic models used by Huehueti.

This module defines Model_v0 (the baseline model) and Model_v1 (an extended model
with spectroscopy / abundance support). Both classes subclass pymc.Model and
compose observed data, prior distributions, latent variables, and likelihoods.
The code relies on the MLP wrapper (mlp callable) to produce model-predicted
absolute photometry as a function of age and a per-source parameter theta.
"""
import sys
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc import Model

def absolute_to_apparent(M, distance):
    """Convert absolute magnitude to relative magnitude.
    m = 

    Units: 
        distance : [pc]
        M : [1]
        m : [1] 
    """
    distance_v = pt.stack([distance for _ in range(M.shape.eval()[1])], axis=1)
    return M + 5.*pt.log10(distance_v) - 5.0


def apparent_to_absolute(m, distance):
    """Convert apparent to absolute
    M = m - 5*log10(distance) + 5
    """
    return m - 5.*np.log10(distance) + 5.


def ccm89_for_gaia(Av):
	""" Optimized for Gaia bands in order,  G_BPmag, Gmag, and G_RPmag with fixed Rv to 3.1"""
	# waves_invm = 1.e4/np.array([5050.0,6230.0,7730.0]) # Angstroms to Inverse Microns

	# """ccm89 a, b parameters for 1.1 < x < 3.3 (optical)"""
	# y = waves_invm - 1.82
	# a = ((((((0.329990*y - 0.77530)*y + 0.01979)*y + 0.72085)*y - 0.02427)*y - 0.50447)*y + 0.17699)*y + 1.0
	# b = ((((((-2.09002*y + 5.30260)*y - 0.62251)*y - 5.38434)*y + 1.07233)*y + 2.28305)*y + 1.41338)*y
	# abrv = (a + b / Rv)

	# return Av * (a + b / Rv)

	# a = np.array([ 1.01577189,0.94036655, 0.80497438])
	# b = np.array([  0.28589221,-0.21954552, -0.51975359])
	# abrv = np.array([ 1.10799518, 0.86954541, 0.63731193])
	abrv = pytensor.shared(np.array([ 1.10799518,0.86954541, 0.63731193]))
	result = pt.outer(Av, abrv)
	
	return result

# import extinction
# Avs = np.array([1.0,2.,3.,4.,5.])
# for Av in Avs:
# 	print(extinction.ccm89(np.array([5050.0,6230.0,7730.0]), Av,3.1))
# print(ccm89_for_gaia(pytensor.shared(Avs)).eval())
# sys.exit()

def combine_component_fluxes(mag1, mag2):
	"""
	Combine component magnitudes into unresolved-binary magnitudes.

	Parameters
	----------
	mag1 : pytensor.tensor.TensorVariable
	    Primary-star magnitudes. Shape (..., n_bands).

	mag2 : pytensor.tensor.TensorVariable
	    Secondary-star magnitudes. Shape (..., n_bands).

	Returns
	-------
	mag_combined : pytensor.tensor.TensorVariable
	    Combined unresolved-binary magnitudes.
	    Shape (..., n_bands).

	Notes
	-----
	Magnitudes are assumed to use the same photometric system
	and zero points for both components.
	"""

	# Magnitude -> relative flux
	flux1 = pt.pow(10.0, -0.4 * mag1)
	flux2 = pt.pow(10.0, -0.4 * mag2)

	# Add component fluxes
	flux_combined = flux1 + flux2

	# Combined flux -> magnitude
	mag_combined = -2.5 * pt.log10(flux_combined)

	return mag_combined

class Model_base(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		mlp_mass : None,
		mlp_logL : None,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)
		n_bands = len(photometry_names)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			age_lower = np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6)
			age_upper = np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower = age_lower,
						upper = age_upper,
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=age_lower,
						upper=age_upper,
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#================================================================================

		#====================== logL =============================
		z = pm.Normal("z", 0, 1,dims="source_id")
		u = pm.math.sigmoid(z)

		log_lum_min = mlp_phot.logL_limits["lower"]["intercept"] + log_age*mlp_phot.logL_limits["lower"]["slope"]
		log_lum_max = mlp_phot.logL_limits["upper"]["intercept"] + log_age*mlp_phot.logL_limits["upper"]["slope"]

		log_lum = pm.Deterministic("log_lum",dims="source_id",
						var=log_lum_min + u * (log_lum_max - log_lum_min)
						)
		#===================================================================

		#------------------ Mass -----------------------------
		mass = pm.Deterministic("mass",
						var=mlp_mass(log_age,log_lum,n_stars),
						dims="source_id")
		#-----------------------------------------------------

		#===================== Photometry =================================================		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance

		abs_phot = mlp_phot(log_age,log_lum,n_stars)
		
		photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance),
						)
		#------------------------------------------------------------------------------------

		#-------------- Likelihood -----------------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=photometry_sd[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#------------------------------------------------------
		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=pytensor.tensor.reshape(tef,(n_stars,1)),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================

class Model_base_dispersion(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		mlp_mass : None,
		mlp_logl : None,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)
		n_bands = len(photometry_names)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			age_lower = np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6)
			age_upper = np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower = age_lower,
						upper = age_upper,
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=age_lower,
						upper=age_upper,
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#=====================================================================================

		#====================== logL =============================
		z = pm.Normal("z", 0, 1,dims="source_id")
		u = pm.math.sigmoid(z)

		log_lum_min = mlp_phot.logL_limits["lower"]["intercept"] + log_age*mlp_phot.logL_limits["lower"]["slope"]
		log_lum_max = mlp_phot.logL_limits["upper"]["intercept"] + log_age*mlp_phot.logL_limits["upper"]["slope"]

		log_lum = pm.Deterministic("log_lum",dims="source_id",
						var=log_lum_min + u * (log_lum_max - log_lum_min)
						)
		#===================================================================

		#------------------ Mass -----------------------------
		mass = pm.Deterministic("mass",
						var=mlp_mass(log_age,log_lum,n_stars),
						dims="source_id")
		#-----------------------------------------------------

		#===================== Photometry =================================================		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance

		abs_phot = mlp_phot(log_age,log_lum,n_stars)
		
		photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance),
						)
		#------------------------------------------------------------------------------------

		#--------------- Intrinsic dispersion --------------------------
		if prior["dispersion"]["family"] == "Gamma":
			photometric_dispersion = pm.Gamma("photometric_dispersion",
							alpha=2.0,
							beta=prior["dispersion"]["beta"],
							dims="photometry_names")
		elif prior["dispersion"]["family"] == "Exponential":
			photometric_dispersion = pm.Exponential("photometric_dispersion",
							lam=prior["dispersion"]["lambda"],
							dims="photometry_names")

		sigma_photometry = pt.sqrt(photometry_sd**2 + pt.broadcast_to(photometric_dispersion**2,
						shape=(n_stars, n_bands)))
		#---------------------------------------------------------------


		#-------------- Likelihood -----------------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=sigma_photometry[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#------------------------------------------------------
		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=pytensor.tensor.reshape(tef,(n_stars,1)),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================

class Model_binaries(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		mlp_mass : None,
		mlp_logl : None,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)
		n_bands = len(photometry_names)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			age_lower = np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6)
			age_upper = np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower = age_lower,
						upper = age_upper,
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=age_lower,
						upper=age_upper,
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#=====================================================================================

		#====================== logL =============================================================================
		z = pm.Normal("z", 0, 1,dims="source_id")
		u = pm.math.sigmoid(z)

		log_lum_min = mlp_phot.logL_limits["lower"]["intercept"] + log_age*mlp_phot.logL_limits["lower"]["slope"]
		log_lum_max = mlp_phot.logL_limits["upper"]["intercept"] + log_age*mlp_phot.logL_limits["upper"]["slope"]

		log_lum_one = pm.Deterministic("log_lum",dims="source_id",
						var=log_lum_min + u * (log_lum_max - log_lum_min)
						)
		#==========================================================================================================

		#======================= Masses ==================================================
		y = pm.Normal("y", 0, 1,dims="source_id")
		v = pm.math.sigmoid(y)

		mass_min = mlp_logl.covariate_limits["lower"]["a"]*(log_age**2) + \
				   mlp_logl.covariate_limits["lower"]["b"]*(log_age) + \
				   mlp_logl.covariate_limits["lower"]["c"]


		mass_one = pm.Deterministic("mass",
						var=mlp_mass(log_age,log_lum_one,n_stars),
						dims="source_id")

		mass_two = pm.Deterministic("mass_secondary",
						var=mass_min + v * (mass_one-mass_min),
						dims="source_id")

		mass_ratio = pm.Deterministic("mass_ratio",
						var=mass_two/mass_one,
						dims="source_id")
		#-------------------------------------------------------
		#=================================================================================

		#=================== Absolute photometry ========================
		log_lum_two = mlp_logl(log_age,mass_two,n_stars)
		abs_phot_one = mlp_phot(log_age,log_lum_one,n_stars)
		abs_phot_two = mlp_phot(log_age,log_lum_two,n_stars)
		abs_phot = combine_component_fluxes(abs_phot_one, abs_phot_two)
		#================================================================

		#===================== Photometry =================================================		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance
		photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance),
						)
		#------------------------------------------------------------------------------------

		#----------------------- Likelihood -----------------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=photometry_sd[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#---------------------------------------------------------------
		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=pytensor.tensor.reshape(tef,(n_stars,1)),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================

class Model_binaries_dispersion(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		mlp_mass : None,
		mlp_logl : None,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)
		n_bands = len(photometry_names)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			age_lower = np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6)
			age_upper = np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower = age_lower,
						upper = age_upper,
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=age_lower,
						upper=age_upper,
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#=====================================================================================

		#====================== logL =============================================================================
		z = pm.Normal("z", 0, 1,dims="source_id")
		u = pm.math.sigmoid(z)

		log_lum_min = mlp_phot.logL_limits["lower"]["intercept"] + log_age*mlp_phot.logL_limits["lower"]["slope"]
		log_lum_max = mlp_phot.logL_limits["upper"]["intercept"] + log_age*mlp_phot.logL_limits["upper"]["slope"]

		log_lum_one = pm.Deterministic("log_lum",dims="source_id",
						var=log_lum_min + u * (log_lum_max - log_lum_min)
						)
		#==========================================================================================================

		#======================= Masses ==================================================
		y = pm.Normal("y", 0, 1,dims="source_id")
		v = pm.math.sigmoid(y)

		mass_min = mlp_logl.covariate_limits["lower"]["a"]*(log_age**2) + \
				   mlp_logl.covariate_limits["lower"]["b"]*(log_age) + \
				   mlp_logl.covariate_limits["lower"]["c"]


		mass_one = pm.Deterministic("mass",
						var=mlp_mass(log_age,log_lum_one,n_stars),
						dims="source_id")

		mass_two = pm.Deterministic("mass_secondary",
						var=mass_min + v * (mass_one-mass_min),
						dims="source_id")

		mass_ratio = pm.Deterministic("mass_ratio",
						var=mass_two/mass_one,
						dims="source_id")
		#-------------------------------------------------------
		#=================================================================================

		#=================== Absolute photometry ========================
		log_lum_two = mlp_logl(log_age,mass_two,n_stars)
		abs_phot_one = mlp_phot(log_age,log_lum_one,n_stars)
		abs_phot_two = mlp_phot(log_age,log_lum_two,n_stars)
		abs_phot = combine_component_fluxes(abs_phot_one, abs_phot_two)
		#================================================================

		#===================== Photometry =================================================		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance
		photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance),
						)
		#------------------------------------------------------------------------------------

		#--------------- Intrinsic dispersion --------------------------
		if prior["dispersion"]["family"] == "Gamma":
			photometric_dispersion = pm.Gamma("photometric_dispersion",
							alpha=2.0,
							beta=prior["dispersion"]["beta"],
							dims="photometry_names")
		elif prior["dispersion"]["family"] == "Exponential":
			photometric_dispersion = pm.Exponential("photometric_dispersion",
							lam=prior["dispersion"]["lambda"],
							dims="photometry_names")

		sigma_photometry = pt.sqrt(photometry_sd**2 + pt.broadcast_to(photometric_dispersion**2,
						shape=(n_stars, n_bands)))
		#---------------------------------------------------------------


		#-------------- Likelihood -----------------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=sigma_photometry[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#------------------------------------------------------
		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=pytensor.tensor.reshape(tef,(n_stars,1)),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================


class Model_linear_dispersion(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)
		n_bands = len(photometry_names)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			age_lower = np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6)
			age_upper = np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower = age_lower,
						upper = age_upper,
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=age_lower,
						upper=age_upper,
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#=====================================================================================

		#====================== logL =============================
		z = pm.Normal("z", 0, 1,dims="source_id")
		u = pm.math.sigmoid(z)

		log_lum_min = mlp_phot.logL_limits["lower"]["intercept"] + log_age*mlp_phot.logL_limits["lower"]["slope"]
		log_lum_max = mlp_phot.logL_limits["upper"]["intercept"] + log_age*mlp_phot.logL_limits["upper"]["slope"]

		log_lum = pm.Deterministic("log_lum",dims="source_id",
						var=log_lum_min + u * (log_lum_max - log_lum_min)
						)
		#===================================================================

		#===================== Photometry =================================================		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance

		abs_phot,mini = mlp_phot(log_age,log_lum,n_stars)
		
		photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance),
						)
		#------------------------------------------------------------------------------------

		#------------------ Mass -----------------------------
		mass = pm.Deterministic("mass",mini,dims="source_id")
		#-----------------------------------------------------

		#--------------------------- Intrinsic dispersion ------------------------------------------
		photometric_intercept = pm.Normal("photometric_intercept",
						mu=0.0,
						sigma = prior["linear_dispersion"]["sigma_intercept"],
						dims="photometry_names")
		photometric_slope = pm.Normal("photometric_slope",
						mu=0.0,
						sigma = prior["linear_dispersion"]["sigma_slope"],
						dims="photometry_names")

		phot_dsp = photometric_intercept[None,:] + photometric_slope[None,:] * z[:,None]

		if prior["linear_dispersion"]["family"] == "Exponential":
			photometric_dispersion = pm.math.exp(phot_dsp)
		elif prior["linear_dispersion"]["family"] == "SoftPlus":
			photometric_dispersion = pt.math.softplus(phot_dsp)

		sigma_photometry = pt.sqrt(photometry_sd**2 + photometric_dispersion**2)
		#----------------------------------------------------------------------------------------------


		#-------------- Likelihood -----------------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=sigma_photometry[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#------------------------------------------------------
		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=pytensor.tensor.reshape(tef,(n_stars,1)),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================

class Model_outliers(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower=np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6),
						upper=np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6),
						upper=np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#=====================================================================================

		#====================== logL =============================
		log_lum = pm.Uniform('log_lum',dims="source_id",
						lower = mlp_phot.domain["logL"][0],
						upper = mlp_phot.domain["logL"][1]
						)
		#=========================================================

		abs_phot,mini = mlp_phot(log_age,log_lum,n_stars)

		#------------------ Mass -----------------------------
		mass = pm.Deterministic("mass",mini,dims="source_id")
		#-----------------------------------------------------

		#===================== Photometry =================================================
		if prior["outliers"]["family"] == "Exponential":			
			#-------------------------------------
			shift_scale = pm.Uniform("shift_scale",
						lower=0.0,
						upper=prior["outliers"]["scale"]
						)
			
			shift_photometry = pm.Exponential('shift_photometry',
						scale=shift_scale,
						dims=("source_id","photometry_names")
						)

			#--------------------- True value ---------------------------------------------------
			photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance)-shift_photometry)
			#------------------------------------------------------------------------------------

		elif prior["outliers"]["family"] == "Normal":
			#-------------------------------------
			shift_scale = pm.Uniform("shift_scale",
						lower=0.0,
						upper=prior["outliers"]["scale"]
						)
			
			shift_photometry = pm.Normal('shift_photometry',
						mu=0.0,
						sigma=shift_scale,
						dims=("source_id","photometry_names")
						)

			#--------------------- True value ---------------------------------------------------
			photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance)+shift_photometry)
			#------------------------------------------------------------------------------------
		else:
			sys.exit("Unrecognized family for outliers model")

		#-------------- Likelihood --------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=photometry_sd[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#-----------------------------------------------------

		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=pytensor.tensor.reshape(tef,(n_stars,1)),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================

class Model_shift(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)
		n_bands = len(photometry_names)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			age_lower = np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6)
			age_upper = np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower = age_lower,
						upper = age_upper,
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=age_lower,
						upper=age_upper,
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#=====================================================================================

		#====================== logL =============================
		log_lum = pm.Uniform('log_lum',dims="source_id",
						lower = mlp_phot.domain["logL"][0],
						upper = mlp_phot.domain["logL"][1]
						)
		#=========================================================

		abs_phot,mini = mlp_phot(log_age,log_lum,n_stars)

		#------------------ Mass -----------------------------
		mass = pm.Deterministic("mass",mini,dims="source_id")
		#-----------------------------------------------------

		#===================== Photometry =================================================
		if prior["shift"]["outliers"]["family"] == "Exponential":			
			#-------------------------------------
			shift_scale = pm.Exponential("shift_scale",
						scale=prior["outliers"]["scale"]
						)
			
			shift_photometry = pm.Exponential('shift_photometry',
						scale=shift_scale,
						dims=("source_id","photometry_names")
						)

			#--------------------- True value ---------------------------------------------------
			photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance)-shift_photometry)
			#------------------------------------------------------------------------------------

		elif prior["shift"]["outliers"]["family"] == "Normal":
			#-------------------------------------
			shift_scale = pm.Exponential("shift_scale",
						scale=prior["outliers"]["scale"]
						)
			
			shift_photometry = pm.Normal('shift_photometry',
						mu=0.0,
						sigma=shift_scale,
						dims=("source_id","photometry_names")
						)

			#--------------------- True value ---------------------------------------------------
			photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance)+shift_photometry)
			#------------------------------------------------------------------------------------
		else:
			sys.exit("Unrecognized family for outliers model")


		#--------------- Intrinsic dispersion --------------------------
		if prior["shift"]["dispersion"]["family"] == "Gamma":
			photometric_dispersion = pm.Gamma("photometric_dispersion",
							alpha=2.0,
							beta=prior["dispersion"]["beta"],
							dims="photometry_names")
		elif prior["shift"]["dispersion"]["family"] == "Exponential":
			photometric_dispersion = pm.Exponential("photometric_dispersion",
							lam=prior["dispersion"]["lambda"],
							dims="photometry_names")

		sigma_photometry = pt.sqrt(photometry_sd**2 + pt.broadcast_to(photometric_dispersion**2,
						shape=(n_stars, n_bands)))
		#---------------------------------------------------------------

		#-------------- Likelihood -----------------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=sigma_photometry[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#------------------------------------------------------
		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=pytensor.tensor.reshape(tef,(n_stars,1)),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================


class Model_base_extinction(Model):
	"""
	Model with extinction.

	Key model pieces:
	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		parameters : dict,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_names : None,
		astrometry_mu : None,
		astrometry_sd : None,
		astrometry_ix : None,
		photometry_names : None,
		photometry_mu : None,
		photometry_sd : None,
		photometry_ix : None,
		spectroscopy_names : None,
		spectroscopy_mu : None,
		spectroscopy_sd : None,
		spectroscopy_ix : None,
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
		- parameters: dict specifying the model parameters to be inferred.
		- prior: dict specifying priors for age, distance, dispersions, etc.
		- identifiers: array of source IDs (used for coords/dims)
		- astrometry_mu, astrometry_sd: observed astrometric values and errors
		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
		- astrometric_names, photometric_names: lists of observable names used to set coords
		"""
		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		if photometry_names is not None:
			self.add_coord("photometry_names",values=photometry_names)
		if astrometry_names is not None:
			self.add_coord("astrometry_names",values=astrometry_names)
		if spectroscopy_names is not None:
			self.add_coord("spectroscopy_names",values=spectroscopy_names)

		n_stars = len(identifiers)

		#===================== Age ======================================================
		if parameters["age"] is None:
			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
			if prior["age"]["family"] == 'TruncatedNormal':
				age = pm.TruncatedNormal("age",
						mu = prior["age"]['mu'],
						sigma = prior["age"]['sigma'],
						lower=np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6),
						upper=np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6),
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform("age", 
						lower=np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6),
						upper=np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6)
						)
			else: 
				raise KeyError('Unknown logAge prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

		log_age = pt.log10(age*1.e6)
		#===============================================================================

		#================ Distance =====================================================
		if parameters["distance"] is None:
			#--------------- Distance_mu --------------------------------------
			# distance_mu is the cluster-level (global) distance prior
			if prior['distance_mu']['family'] == "Gaussian":
				distance_mu = pm.Normal('distance_mu', 
						mu = prior['distance_mu']['mu'],
						sigma = prior['distance_mu']['sigma'])
			elif prior['distance_mu']['family'] == 'Uniform':
				distance_mu = pm.Uniform('distance_mu',
						lower = prior['distance_mu']['lower'],
						upper = prior['distance_mu']['upper'])
			else: 
				raise KeyError('Unknown distance_mu prior distribution')
			#--------------------------------------------------------------

			#------------------- Distance_sd --------------------------------------
			# distance_sd controls the per-source scatter around central distance
			if prior["distance_sd"]["family"] == 'Exponential':
				distance_sd = pm.Exponential('distance_sd',
						scale=prior["distance_sd"]["scale"])
			elif prior["distance_sd"]["family"] == 'Gamma':
				distance_sd = pm.Gamma('distance_sd',
						alpha=2.0,
						beta=prior["distance_sd"]["beta"])
			else:
				raise KeyError('Unknown distance_sd distribution')
			#---------------------------------------------------------------------
			
			#------------- Distances -------------------------------
			distance = pm.Normal("distance",
						mu=distance_mu,
						sigma=distance_sd,
						dims="source_id")
			#----------------------------------------------------------------
		else:
			distance = pm.Deterministic("distance",
						var=pytensor.shared(parameters["distance"]),
						dims="source_id")
		#=====================================================================================

		#====================== logL ==============================
		log_lum = pm.Uniform('log_lum',dims="source_id",
						lower = mlp_phot.domain["logL"][0],
						upper = mlp_phot.domain["logL"][1]
						)
		#==========================================================

		#==================== Extinction ============================
		#------------------- Prior -----------------------------
		if prior["extinction"]["family"] == "Uniform":
			Av = pm.Uniform("Av",
						lower=prior["extinction"]["lower"],
						upper=prior["extinction"]["upper"],
						dims="source_id")
		elif prior["extinction"]["family"] == "TruncatedNormal":
			Av = pm.TruncatedNormal('Av',
						mu=prior["extinction"]["mu"],
						sigma=prior["extinction"]["sigma"],
						lower = prior["extinction"]['lower'],
						upper = prior["extinction"]['upper'],
						dims="source_id")
		elif prior["extinction"]["family"] == "Exponential":
			Av = pm.Exponential('Av',
						scale=prior["extinction"]["sigma"],
						dims="source_id")
		else:
			sys.exit("Unsupported extinction family")
		#===========================================================
		
		#===================== Photometry =================================================		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance

		abs_phot,mini = mlp_phot(log_age,log_lum,n_stars)

		#------------------ Mass -----------------------------
		mass = pm.Deterministic("mass",mini,dims="source_id")
		#-----------------------------------------------------

		photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
						var=absolute_to_apparent(abs_phot,distance) +
						ccm89_for_gaia(Av)
						)
		#------------------------------------------------------------------------------------

		#-------------- Likelihood ----------------------------
		obs_photometry = pm.Normal('obs_photometry', 
						mu=photometry[photometry_ix], 
						sigma=photometry_sd[photometry_ix],
						observed=photometry_mu[photometry_ix])
		#-----------------------------------------------------
		#======================================================================================

		#===================== Astrometry ===============================================
		if astrometry_mu is not None:
			#------------ True value --------------------------------------------------
			astrometry = pm.Deterministic("astrometry",
						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
						dims=("source_id","astrometry_names"))
			#---------------------------------------------------------------------------

			#----------- Likelihood --------------------------
			obs_astrometry = pm.Normal('obs_astrometry',
						mu=astrometry[astrometry_ix], 
						sigma=astrometry_sd[astrometry_ix], 
						observed=astrometry_mu[astrometry_ix])
			#----------------------------------------------------
		#================================================================================

		#=================== Spectroscopy =============================================
		if spectroscopy_mu is not None:
			#---------------- True values -----------------------------------
			spectroscopy = pm.Deterministic('spectroscopy',
						var=mlp_teff(age, mass, n_stars),
						dims=("source_id","spectroscopy_names"))
			#----------------------------------------------------------------

			#-------------- Likelihood ---------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
						mu=spectroscopy[spectroscopy_ix], 
						sigma=spectroscopy_sd[spectroscopy_ix],
						observed=spectroscopy_mu[spectroscopy_ix])
			#-----------------------------------------------
		#=================================================================================


# class Model_outliers_extinction(Model):
# 	"""
# 	Model with extinction and outliers.

# 	Key model pieces:
# 	- Global parameters: age, distance_central, distance_sd, photometric_dispersion
# 	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
# 	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
# 	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
# 	"""
	
# 	def __init__(self,
# 		mlp_phot,
# 		parameters : dict,
# 		prior : dict,
# 		identifiers : np.ndarray,
# 		astrometry_names : None,
# 		astrometry_mu : None,
# 		astrometry_sd : None,
# 		astrometry_ix : None,
# 		photometry_names : None,
# 		photometry_mu : None,
# 		photometry_sd : None,
# 		photometry_ix : None,
# 		spectroscopy_names : None,
# 		spectroscopy_mu : None,
# 		spectroscopy_sd : None,
# 		spectroscopy_ix : None,
# 		):
# 		"""Construct Model_v0.

# 		Parameters (high level)
# 		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
# 		- parameters: dict specifying the model parameters to be inferred.
# 		- prior: dict specifying priors for age, distance, dispersions, etc.
# 		- identifiers: array of source IDs (used for coords/dims)
# 		- astrometry_mu, astrometry_sd: observed astrometric values and errors
# 		- astrometry_ix: indices of finite astrometric measurements (used to mask likelihood)
# 		- photometry_mu, photometry_sd, photometry_ix: analogous for photometry
# 		- astrometric_names, photometric_names: lists of observable names used to set coords
# 		"""
# 		# Initialize parent Model (name empty) and register coords for ArviZ/InferenceData
# 		super().__init__(name="", model=None)
# 		self.add_coord("source_id",values=identifiers)
# 		if photometry_names is not None:
# 			self.add_coord("photometry_names",values=photometry_names)
# 		if astrometry_names is not None:
# 			self.add_coord("astrometry_names",values=astrometry_names)
# 		if spectroscopy_names is not None:
# 			self.add_coord("spectroscopy_names",values=spectroscopy_names)

# 		n_stars = len(identifiers)

# 		#===================== Age ======================================================
# 		if parameters["age"] is None:
# 			# Age prior can be either TruncatedNormal or Uniform as provided by caller.
# 			if prior["age"]["family"] == 'TruncatedNormal':
# 				age = pm.TruncatedNormal("age",
# 						mu = prior["age"]['mu'],
# 						sigma = prior["age"]['sigma'],
# 						lower=np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6),
# 						upper=np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6),
# 						)
# 			elif prior["age"]["family"] == 'Uniform':
# 				age = pm.Uniform("age", 
# 						lower=np.pow(10,mlp_phot.domain["logAge"][0])/np.pow(10,6),
# 						upper=np.pow(10,mlp_phot.domain["logAge"][1])/np.pow(10,6),)
# 			else: 
# 				raise KeyError('Unknown logAge prior distribution')
# 		else:
# 			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

# 		log_age = pt.log10(age*1.e6)
# 		#===============================================================================

# 		#================ Distance =====================================================
# 		if parameters["distance"] is None:
# 			#--------------- Distance_mu --------------------------------------
# 			# distance_mu is the cluster-level (global) distance prior
# 			if prior['distance_mu']['family'] == "Gaussian":
# 				distance_mu = pm.Normal('distance_mu', 
# 						mu = prior['distance_mu']['mu'],
# 						sigma = prior['distance_mu']['sigma'])
# 			elif prior['distance_mu']['family'] == 'Uniform':
# 				distance_mu = pm.Uniform('distance_mu',
# 						lower = prior['distance_mu']['lower'],
# 						upper = prior['distance_mu']['upper'])
# 			else: 
# 				raise KeyError('Unknown distance_mu prior distribution')
# 			#--------------------------------------------------------------

# 			#------------------- Distance_sd --------------------------------------
# 			# distance_sd controls the per-source scatter around central distance
# 			if prior["distance_sd"]["family"] == 'Exponential':
# 				distance_sd = pm.Exponential('distance_sd',
# 						scale=prior["distance_sd"]["scale"])
# 			elif prior["distance_sd"]["family"] == 'Gamma':
# 				distance_sd = pm.Gamma('distance_sd',
# 						alpha=2.0,
# 						beta=prior["distance_sd"]["beta"])
# 			else:
# 				raise KeyError('Unknown distance_sd distribution')
# 			#---------------------------------------------------------------------
			
# 			#------------- Distances -------------------------------
# 			distance = pm.Normal("distance",
# 						mu=distance_mu,
# 						sigma=distance_sd,
# 						dims="source_id")
# 			#----------------------------------------------------------------
# 		else:
# 			distance = pm.Deterministic("distance",
# 						var=pytensor.shared(parameters["distance"]),
# 						dims="source_id")
# 		#=====================================================================================

# 		#====================== logL ==============================
# 		log_lum = pm.Uniform('log_lum',dims="source_id",
# 						lower = mlp_phot.domain["logL"][0],
# 						upper = mlp_phot.domain["logL"][1]
# 						)
# 		#==========================================================

# 		#==================== Extinction ============================
# 		#------------------- Prior -----------------------------
# 		if prior["extinction"]["family"] == "Uniform":
# 			Av = pm.Uniform("Av",
# 						lower=prior["extinction"]["lower"],
# 						upper=prior["extinction"]["upper"],
# 						dims="source_id")
# 		elif prior["extinction"]["family"] == "TruncatedNormal":
# 			Av = pm.TruncatedNormal('Av',
# 						mu=prior["extinction"]["mu"],
# 						sigma=prior["extinction"]["sigma"],
# 						lower = prior["extinction"]['lower'],
# 						upper = prior["extinction"]['upper'],
# 						dims="source_id")
# 		elif prior["extinction"]["family"] == "Exponential":
# 			Av = pm.Exponential('Av',
# 						scale=prior["extinction"]["sigma"],
# 						dims="source_id")
# 		else:
# 			sys.exit("Unsupported extinction family")
# 		#===========================================================
		
# 		#===================== Photometry =================================================
# 		#--------------------- True value ---------------------------------------------------
# 		photometry = pm.Deterministic('photometry',dims=("source_id","photometry_names"),
# 						var=absolute_to_apparent(mlp_phot(log_age,log_lum,n_stars),distance) +
# 						ccm89_for_gaia(Av)
# 						)
# 		#------------------------------------------------------------------------------------

# 		if prior["outliers"]["family"] == "StudentT":
# 			nu = pm.Gamma("nu",
# 						alpha=2.0,
# 						beta=prior["outliers"]["beta"],
# 						dims="photometry_names")

# 			#-------------- Likelihood --------------------
# 			obs_photometry = pm.StudentT('obs_photometry',
# 						nu=pt.tile(nu,n_stars),
# 						mu=photometry[photometry_ix], 
# 						sigma=photometry_sd[photometry_ix],
# 						observed=photometry_mu[photometry_ix])
# 			#-----------------------------------------------------
# 		elif prior["outliers"]["family"] == "SkewNormal":
# 			alpha = pm.Normal("alpha",
# 						mu=0.0,
# 						sigma=prior["outliers"]["scale"],
# 						dims="photometry_names"
# 						)

# 			#-------------- Likelihood --------------------
# 			obs_photometry = pm.SkewNormal('obs_photometry',
# 						alpha=pt.tile(alpha,n_stars),
# 						mu=photometry[photometry_ix], 
# 						sigma=photometry_sd[photometry_ix],
# 						observed=photometry_mu[photometry_ix],
# 						dims=("source_id","photometry_names")
# 				)
# 			#-----------------------------------------------------
# 		else:
# 			sys.exit("Unrecognized outliers family")
# 		#======================================================================================

# 		#===================== Astrometry ===============================================
# 		if astrometry_mu is not None:
# 			#------------ True value --------------------------------------------------
# 			astrometry = pm.Deterministic("astrometry",
# 						var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
# 						dims=("source_id","astrometry_names"))
# 			#---------------------------------------------------------------------------

# 			#----------- Likelihood --------------------------
# 			obs_astrometry = pm.Normal('obs_astrometry',
# 						mu=astrometry[astrometry_ix], 
# 						sigma=astrometry_sd[astrometry_ix], 
# 						observed=astrometry_mu[astrometry_ix])
# 			#----------------------------------------------------
# 		#================================================================================

# 		#=================== Spectroscopy =============================================
# 		if spectroscopy_mu is not None:
# 			#---------------- True values -----------------------------------
# 			spectroscopy = pm.Deterministic('spectroscopy',
# 						var=mlp_teff(age, mass, n_stars),
# 						dims=("source_id","spectroscopy_names"))
# 			#----------------------------------------------------------------

# 			#-------------- Likelihood ---------------------
# 			obs_spectroscopy = pm.Normal('obs_spectroscopy',
# 						mu=spectroscopy[spectroscopy_ix], 
# 						sigma=spectroscopy_sd[spectroscopy_ix],
# 						observed=spectroscopy_mu[spectroscopy_ix])
# 			#-----------------------------------------------
# 		#=================================================================================