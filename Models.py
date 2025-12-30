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
from pymc import Model

from Functions import absolute_to_apparent


class Model_v0(Model):
	"""
	Baseline model for photometry and parallax.

	Key model pieces:
	- Global parameters: age, distance_central, distance_dispersion, photometric_dispersion
	- Per-source latent variables: theta (uniform prior), distance (Normal around central),
	  derived astrometry (parallax) and deterministic predicted photometry (via MLP).
	- Likelihoods: Normal for astrometry (parallax) and Normal for photometry.
	"""
	
	def __init__(self,
		mlp_phot,
		mlp_mass,
		prior : dict,
		identifiers : np.ndarray,
		astrometry_mu : np.ndarray,
		astrometry_sd : np.ndarray,
		astrometry_ix : np.ndarray,
		photometry_mu : np.ndarray,
		photometry_sd : np.ndarray,
		photometry_ix : np.ndarray,
		astrometric_names : list,
		photometric_names : list
		):
		"""Construct Model_v0.

		Parameters (high level)
		- mlp: callable MLP(age, theta, n_stars) -> (mass, absolute_photometry)
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
		self.add_coord("astrometric_names",values=astrometric_names)
		self.add_coord("photometric_names",values=photometric_names)

		n_stars = len(identifiers)

		#===================== Global parameters =====================================
		#---------------------- Age ------------------------
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
		#-----------------------------------------------------

		#--------------- Astrometry -----------------------------------
		# distance_central is the cluster-level (global) distance prior
		if prior['distance']['family'] == "Gaussian":
			distance_central = pm.Normal('distance_central', 
					mu = prior['distance']['mu'],
					sigma = prior['distance']['sigma'])
		elif prior['distance']['family'] == 'Uniform':
			distance_central = pm.Uniform('distance_central',
					lower = prior['distance']['lower'],
					upper = prior['distance']['upper'])
		else: 
			raise KeyError('Unknown distance prior distribution')

		# distance_dispersion controls the per-source scatter around central distance
		if prior["distance_dispersion"]["family"] == 'Exponential':
			distance_dispersion = pm.Exponential('distance_dispersion',
									scale=prior["distance_dispersion"]["scale"])
		elif prior["distance_dispersion"]["family"] == 'Gamma':
			distance_dispersion = pm.Gamma('distance_dispersion',
									alpha=2.0,
									beta=prior["distance_dispersion"]["beta"])
		else:
			raise KeyError('Unknown astrometric_dispersion distribution')
		#----------------------------------------------------------------------------

		#--------------- Photometry hyper-parameters ------------------------------------------------------
		# Photometric dispersion is a per-band parameter (dims="photometric_names")
		if prior["photometric_dispersion"]["family"] == 'Exponential':
			photometric_dispersion = pm.Exponential('photometric_dispersion',
									scale=prior["photometric_dispersion"]["sigma"],
									dims="photometric_names")
		elif prior["photometric_dispersion"]["family"] == 'Gamma':
			photometric_dispersion = pm.Gamma('photometric_dispersion',
									alpha=2.0,
									beta=prior["photometric_dispersion"]["beta"],
									dims="photometric_names")
		else:
			raise KeyError('Unknown photometric_dispersion distribution')
		#---------------------------------------------------------------------------------
		#======================================================================================

		#====================== Source parameters =======================================
		# Theta: per-source parameter used as input to the MLP (domain typically in (0,1])
		theta = pm.Uniform("theta",lower = 1e-5, upper = 1.0,dims="source_id")
		#-----------------------------------------------------------------------
		#================================================================================

		#===================== True values =========================================
		# Per-source distance drawn from the global distance_central and dispersion
		distance = pm.Normal("distance",
								mu=distance_central,sigma=distance_dispersion,
								dims="source_id")
		# Convert distance (pc) to parallax (mas) as a deterministic variable for comparison
		astrometry = pm.Deterministic("astrometry",
							var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
							dims=("source_id","astrometric_names"))

		#-------------- Neural Network -----------------------
		# Use the provided MLP callable to predict mass and absolute photometry for each source.
		absolute_photometry = mlp_phot(age, theta, n_stars)
		mass_logTeff_logg   = mlp_mass(age, theta,n_stars)
		#-----------------------------------------------------

		# Expose mass, Teff as a deterministic per-source variable in the model
		mass = pm.Deterministic('mass',mass_logTeff_logg[:,0],dims="source_id")
		# teff = pm.Deterministic('teff',mass_logTeff_logg[:,1],dims="source_id")
		
		# Convert absolute photometry to apparent magnitudes using per-source distance
		photometry = pm.Deterministic('photometry',
							absolute_to_apparent(absolute_photometry,distance,11),
							dims=("source_id","photometric_names"))
		#=================================================================================

		#================== Addition of uncertainties ====================================
		# Combine observed photometric uncertainties with model photometric dispersion
		photometry_sigma = pm.math.sqrt(photometry_sd**2 + photometric_dispersion**2)
		#=================================================================================

		#===================== Likelihood =============================
		# Use the provided index masks (astrometry_ix, photometry_ix) to restrict likelihood
		# application only to measured (finite) entries.
		#---------------- Parallax --------------------------
		obs_astrometry = pm.Normal('obs_astrometry', 
			mu=astrometry[astrometry_ix], 
			sigma=astrometry_sd[astrometry_ix], 
			observed=astrometry_mu[astrometry_ix])
		#----------------------------------------------------

		#-------------- Photometric likelihood --------------------
		obs_photometry = pm.Normal('obs_photometry', 
			mu=photometry[photometry_ix], 
			sigma=photometry_sigma[photometry_ix],
			observed=photometry_mu[photometry_ix])
		#-----------------------------------------------------
		#===============================================================

class Model_v1(Model):
	"""
	Chronos baseline model for photometry and Lithium abundance (longer form).

	This class is an extended/experimental model which supports:
	- per-source Teff priors
	- separate age variables for photometry and LDB (age_iso, age_ldb)
	- mixture model for photometric outliers
	- optional spectroscopy modeling (true_spectroscopy)
	"""
	
	def __init__(self,
		mlp,
		mlp_ldb,
		prior : dict,
		identifiers : np.ndarray,
		parallax_mu : np.ndarray,
		parallax_sd : np.ndarray,
		parallax_ix : np.ndarray,
		photometry_mu : np.ndarray,
		photometry_sd : np.ndarray,
		photometry_ix : np.ndarray,
		spectroscopy_mu : np.ndarray,
		spectroscopy_sd : np.ndarray,
		spectroscopy_ix : np.ndarray,
		photometric_names : list,
		spectroscopic_names : list
		):
		"""Construct Model_v1.

		This initializer follows the same pattern as Model_v0 but includes more
		components for spectroscopy and outlier handling. Many parts are optional
		and guarded by the presence of corresponding observed arrays (e.g. spectroscopy_mu).
		"""
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		self.add_coord("photometric_names",values=photometric_names)
		self.add_coord("spectroscopic_names",values=spectroscopic_names)

		# determine which observation types are being used (None implies unused)
		use_spectroscopy = spectroscopy_mu is not None
		use_photometry = photometry_mu is not None
		n_stars = len(identifiers)

		#===================== Prior specification =====================================
		#-------------------------- Teff --------------------------
		# Teff can be a per-source Gaussian or Uniform prior depending on prior dict.
		if prior["Teff"]['family'] == 'Gaussian':
			tefs = pm.Normal("tefs", 
					mu = prior["Teff"]['mu'],
					sigma = prior["Teff"]['sigma'],
					initval=np.repeat(prior["Teff"]["initval"],n_stars),
					dims="source_id")
		elif prior["Teff"]['family'] == 'Uniform':
			tefs = pm.Uniform("tefs",
					lower = prior["Teff"]['lower'], 
					upper = prior["Teff"]['upper'],
					initval=np.repeat(prior["Teff"]["initval"],n_stars),
					dims="source_id")
		else: 
			raise KeyError('Unknown age prior distribution')
		#------------------------------------------------------
		
		# --------------- Photometry --------------------------------
		if use_photometry:
			#---------------------- Age ------------------------ 
			if prior["age"]["family"] == 'Gaussian':
				age_iso = pm.TruncatedNormal('age_iso',
						mu = prior['age']['mu'],
						sigma = prior['age']['sigma'],
						lower=0.0,
						initval=prior["age"]["initval"])
			elif prior["age"]["family"] == 'Uniform':
				age_iso = pm.Uniform('age_iso',
						upper = prior['age']['upper'], 
						lower = prior['age']['lower'],
						initval=prior["age"]["initval"])
			else: 
				raise KeyError('Unknown age prior distribution')
			#-----------------------------------------------------

			#------------ Distance (per-source prior) ----------------------------
			if prior['distance']['family'] == 'Gaussian':
				distance = pm.Normal('distance', 
						mu = prior['distance']['mu'],
						sigma = prior['distance']['sigma'],
						initval=np.repeat(prior["distance"]["initval"],n_stars),
						dims="source_id")
			elif prior['distance']['family'] == 'Uniform':
				distance = pm.Uniform('distance',
						lower = prior['distance']['lower'],
						upper = prior['distance']['upper'],
						initval=np.repeat(prior["distance"]["initval"],n_stars),
						dims="source_id")
			else: 
				raise KeyError('Unknown distance prior distribution')
			#---------------------------------------------------------

			#-------------- Photometric dispersion hyperparameters --------------------------------------
			if prior["photometric_dispersion"]["family"] == 'Exponential':
				photometric_dispersion = pm.Exponential('photometric_dispersion',
										scale=prior["photometric_dispersion"]["sigma"],
										initval=prior["photometric_dispersion"]["initval"],
										dims="photometric_names")
			else:
				raise KeyError('Unknown photometric_dispersion distribution')
			#----------------------------------------------------------------

			#------------- Outlier modeling (mixture) --------------------------------------------
			photometric_outliers_weights = pm.Dirichlet("photometric_outliers_weights",
									a=prior["photometric_outliers"]["weights"])
			photometric_outliers_mu = pm.Uniform("photometric_outliers_mu",
									lower=prior["photometric_outliers"]["lower"],
									upper=prior["photometric_outliers"]["upper"],
									# dims="photometric_names"
									)
			photometric_outliers_sd = pm.Gamma("photometric_outliers_sd",
									alpha=2,
									beta=prior["photometric_outliers"]["beta"],
									# dims="photometric_names"
									)
			#------------------------------------------------------------
		
		#---------------------- Spectroscopy--------------------------------------------
		if use_spectroscopy:
			# Age variable specific to LDB/spectroscopy modeling
			if prior["age"]["family"] == 'Gaussian':
				age_ldb = pm.TruncatedNormal('age_ldb',
						mu = prior['age']['mu'],
						sigma = prior['age']['sigma'],
						lower=0.0,
						initval=prior["age"]["initval"])
			elif prior["age"]["family"] == 'Uniform':
				age_ldb = pm.Uniform('age_ldb',
						upper = prior['age']['upper'], 
						lower = prior['age']['lower'],
						initval=prior["age"]["initval"])
			else: 
				raise KeyError('Unknown age prior distribution')
			# Note: spectroscopic dispersion and outlier components are shown
			# commented-out in the original code. They can be enabled if needed.
		#------------------------------------------------------------------------------
		#================================================================================

		#===================== Transformations =========================================
		#------------ True photometry --------------------------------
		if use_photometry:
			# Convert distance to parallax deterministically
			parallax = pm.Deterministic('parallax', 
								1000./distance,
								dims="source_id")

			# Compute true photometry via an mlp_iso function (must be provided)
			true_photometry = pm.Deterministic('true_photometry',
								mlp_iso(age_iso,tefs,distance,n_stars),
								dims=("source_id","photometric_names"))
		#------------------------------------------------------------------

		#------------- True spectroscopy ---------------------
		if use_spectroscopy:
			# Compute true spectroscopic predictions using provided mlp_ldb callable
			true_spectroscopy = pm.Deterministic("true_spectroscopy",
								mlp_ldb(age_ldb,tefs,n_stars),
								dims=("source_id","spectroscopic_names"))
		#-----------------------------------------------------
		#=================================================================================

		#================== Intrinsic dispersion & Outliers ================================
		#----------------- Photometry mixture model ----------------------------
		if use_photometry:
			photometric_components = [
				pm.Normal.dist(mu=true_photometry,
							sigma=photometric_dispersion),
				pm.Normal.dist(mu=photometric_outliers_mu, 
							sigma=photometric_outliers_sd),
				]
			photometry = pm.Mixture('photometry', 
							w=photometric_outliers_weights, 
							comp_dists=photometric_components,
							dims=("source_id","photometric_names"))

			# Alternative (simple) photometry model (commented in original):
			# photometry = pm.Normal("photometry",
			# 					mu=true_photometry,
			# 					sigma=photometric_dispersion,
			# 					dims=("source_id","photometric_names"))
		#------------------------------------------------------------

		#------------------- Spectroscopy -----------------------------
		# (spectroscopy model is commented out in the original code)
		#=================================================================================

		#===================== Likelihood =============================
		# Apply the observation likelihoods only to measured entries using the provided masks.
		if use_photometry:
			#---------------- Parallax --------------------------
			obs_parallax = pm.Normal('obs_parallax', 
				mu=parallax[parallax_ix], 
				sigma=parallax_sd[parallax_ix], 
				observed=parallax_mu[parallax_ix])
			#----------------------------------------------------

			#-------------- Photometric likelihood --------------------
			obs_photometry = pm.Normal('obs_photometry', 
				mu=photometry[photometry_ix], 
				sigma=photometry_sd[photometry_ix],
				observed=photometry_mu[photometry_ix])
			#-----------------------------------------------------

		if use_spectroscopy:
			#------------- Spectroscopic likelihood (e.g., Lithium) ------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
				mu=true_spectroscopy[spectroscopy_ix], 
				sigma=spectroscopy_sd[spectroscopy_ix],
				observed=spectroscopy_mu[spectroscopy_ix])
		#===============================================================