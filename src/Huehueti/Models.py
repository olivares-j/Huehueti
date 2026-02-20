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



class Model_v0(Model):
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
		mlp_teff,
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
				age = pm.TruncatedNormal('age',
						mu = prior['age']['mu'],
						sigma = prior['age']['sigma'],
						lower = prior['age']['lower'],
						upper = prior['age']['upper'],
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform('age', 
						lower = prior['age']['lower'],
						upper = prior['age']['upper'],)
			else: 
				raise KeyError('Unknown age prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))
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
								pytensor.shared(parameters["distance"]),
								dims="source_id")
		#=====================================================================================

		#====================== Mass ============================================================
		mass = pm.Uniform('mass',dims="source_id",
							lower=mlp_phot.mass_domain[0],
							upper=mlp_phot.mass_domain[1],
							initval=np.full(shape=n_stars,fill_value=mlp_phot.mass_domain[0]+1e-3)
							)
		#======================================================================================

		#===================== Photometry =================================================
		# # Photometric dispersion is a per-band parameter (dims="photometric_names")
		# if parameters["photometric_dispersion"] is None:
		# 	if prior["photometric_dispersion"]["family"] == 'Exponential':
		# 		photometric_dispersion = pm.Exponential('photometric_dispersion',
		# 								scale=prior["photometric_dispersion"]["sigma"],
		# 								dims="photometric_names")
		# 	elif prior["photometric_dispersion"]["family"] == 'Gamma':
		# 		photometric_dispersion = pm.Gamma('photometric_dispersion',
		# 								alpha=2.0,
		# 								beta=prior["photometric_dispersion"]["beta"],
		# 								dims="photometric_names")
		# 	else:
		# 		raise KeyError('Unknown photometric_dispersion distribution')
		# else:
		# 	photometric_dispersion = pm.Deterministic("photometric_dispersion",
		# 								pytensor.shared(parameters["photometric_dispersion"]))
		#---------------------------------------------------------------------------------

		# Combine observed photometric uncertainties with model photometric dispersion
		# photometry_sigma = pm.math.sqrt(photometry_sd**2 + photometric_dispersion**2)
		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance
		photometry = pm.Deterministic('photometry',
							var=absolute_to_apparent(mlp_phot(age, mass, n_stars),distance),
							dims=("source_id","photometry_names"))
		#------------------------------------------------------------------------------------

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


class Model_v1(Model):
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
		mlp_teff,
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
		"""Construct Model_v1.

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
				age = pm.TruncatedNormal('age',
						mu = prior['age']['mu'],
						sigma = prior['age']['sigma'],
						lower = prior['age']['lower'],
						upper = prior['age']['upper'],
						)
			elif prior["age"]["family"] == 'Uniform':
				age = pm.Uniform('age', 
						lower = prior['age']['lower'],
						upper = prior['age']['upper'],)
			else: 
				raise KeyError('Unknown age prior distribution')
		else:
			age = pm.Deterministic("age",pytensor.shared(parameters["age"]))
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
								pytensor.shared(parameters["distance"]),
								dims="source_id")
		#=====================================================================================

		#====================== Mass ============================================================
		mass = pm.Uniform('mass',dims="source_id",
							lower=mlp_phot.mass_domain[0],
							upper=mlp_phot.mass_domain[1],
							initval=np.full(shape=n_stars,fill_value=mlp_phot.mass_domain[0]+1e-3)
							)
		#======================================================================================

		#====================== Outliers distribution =========================================
		# Pb   = parameters[3] # The probability of being an outlier
		# Yb   = parameters[4] # The mean position of the outlier distribution
		# sd_b = parameters[5] # The variance of the outlier distribution
		# sd_m = parameters[6] # The variance added to the photometry

		# #--------------- Extra ---------------------------------------------
		# prior_Pb   = st.dirichlet(alpha=hyper["alpha_Pb"])
		# prior_Yb   = st.norm(loc=np.mean(o_phot),scale=5.00*np.std(o_phot))
		# prior_sd_b = st.gamma(a=2.0,scale=hyper["beta_sd_b"])
		# prior_sd_m = st.gamma(a=2.0,scale=hyper["beta_sd_m"])
		#-------------------------------------------------------------------
		#======================================================================================

		#===================== Photometry =================================================
		# # Photometric dispersion is a per-band parameter (dims="photometric_names")
		# if parameters["photometric_dispersion"] is None:
		# 	if prior["photometric_dispersion"]["family"] == 'Exponential':
		# 		photometric_dispersion = pm.Exponential('photometric_dispersion',
		# 								scale=prior["photometric_dispersion"]["sigma"],
		# 								dims="photometric_names")
		# 	elif prior["photometric_dispersion"]["family"] == 'Gamma':
		# 		photometric_dispersion = pm.Gamma('photometric_dispersion',
		# 								alpha=2.0,
		# 								beta=prior["photometric_dispersion"]["beta"],
		# 								dims="photometric_names")
		# 	else:
		# 		raise KeyError('Unknown photometric_dispersion distribution')
		# else:
		# 	photometric_dispersion = pm.Deterministic("photometric_dispersion",
		# 								pytensor.shared(parameters["photometric_dispersion"]))
		#---------------------------------------------------------------------------------

		# Combine observed photometric uncertainties with model photometric dispersion
		# photometry_sigma = pm.math.sqrt(photometry_sd**2 + photometric_dispersion**2)
		
		#--------------------- True value ---------------------------------------------------
		# Convert absolute photometry to apparent magnitudes using per-source distance
		photometry = pm.Deterministic('photometry',
							var=absolute_to_apparent(mlp_phot(age, mass, n_stars),distance),
							dims=("source_id","photometry_names"))
		#------------------------------------------------------------------------------------

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