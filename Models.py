import sys
import numpy as np
import pymc as pm
import pytensor
from pymc import Model

from Functions import absolute_to_apparent


class Model_v0(Model):
	"""
	Baseline model.
	"""
	
	def __init__(self,
		mlp,
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
		"""Input observed variables.
		
		Parameters
		----------
		data : pd.Dataframe
			Dataframe containing observed parallax (parallax), photometry, and Li abundance.
		"""
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		self.add_coord("astrometric_names",values=astrometric_names)
		self.add_coord("photometric_names",values=photometric_names)

		n_stars = len(identifiers)

		#===================== Global parameters =====================================
		#---------------------- Age ------------------------ 
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

		#--------------- Photometry ------------------------------------------------------
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
		#---------------- Theta-------------------------------------------------
		theta = pm.Uniform("theta",lower = 1e-5, upper = 1.0,dims="source_id")
		#-----------------------------------------------------------------------
		#================================================================================

		#===================== True values =========================================
		distance = pm.Normal("distance",
								mu=distance_central,sigma=distance_dispersion,
								dims="source_id")
		astrometry = pm.Deterministic("astrometry",
							var=pytensor.tensor.reshape(1000./distance,(n_stars,1)),
							dims=("source_id","astrometric_names"))

		#-------------- Neural Network -----------------------
		mass,absolute_photometry = mlp(age,theta, n_stars)
		#-----------------------------------------------------

		mass = pm.Deterministic('mass',mass,dims="source_id")
		
		photometry = pm.Deterministic('photometry',
							absolute_to_apparent(absolute_photometry,distance,11),
							dims=("source_id","photometric_names"))
		#=================================================================================

		#================== Addition of uncertainties ====================================
		photometry_sigma = photometry_sd**2 + photometric_dispersion**2
		#=================================================================================

		#===================== Likelihood =============================
		#- The true and observed are contracted to the not NaN values
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
	Chronos baseline model for photometry and Lithium abundance.
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
		"""Input observed variables.
		
		Parameters
		----------
		data : pd.Dataframe
			Dataframe containing observed parallax (parallax), photometry, and Li abundance.
		"""
		super().__init__(name="", model=None)
		self.add_coord("source_id",values=identifiers)
		self.add_coord("photometric_names",values=photometric_names)
		self.add_coord("spectroscopic_names",values=spectroscopic_names)

		use_spectroscopy = spectroscopy_mu is not None
		use_photometry = photometry_mu is not None
		n_stars = len(identifiers)

		#===================== Prior specification =====================================
		#-------------------------- Teff --------------------------
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

			#------------ Distance ----------------------------
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

			#-------------- Dispersion --------------------------------------
			if prior["photometric_dispersion"]["family"] == 'Exponential':
				photometric_dispersion = pm.Exponential('photometric_dispersion',
										scale=prior["photometric_dispersion"]["sigma"],
										initval=prior["photometric_dispersion"]["initval"],
										dims="photometric_names")
			else:
				raise KeyError('Unknown photometric_dispersion distribution')
			#----------------------------------------------------------------

			#------------- Outliers --------------------------------------------
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
			#---------------------- Age ------------------------ 
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
			#-----------------------------------------------------

			#---------------------- Dispersion -----------------------------------
			# if prior["spectroscopic_dispersion"]["family"] == 'exponential':
			# 	spectroscopic_dispersion = pm.Exponential('spectroscopic_dispersion',
			# 						scale=prior["spectroscopic_dispersion"]["sigma"],
			# 						dims="spectroscopic_names")
			# else:
			# 	raise KeyError('Unknown spectroscopic_dispersion distribution')
			#----------------------------------------------------------------------

			# #---------------------- Outliers -----------------------------------------------
			# spectroscopic_outliers_weights = pm.Dirichlet("spectroscopic_outliers_weights",
			# 							a=prior["spectroscopic_outliers"]["weights"])
			# spectroscopic_outliers_mu = pm.Normal("spectroscopic_outliers_mu",
			# 							mu=prior["spectroscopic_outliers"]["mu"],
			# 							sigma=prior["spectroscopic_outliers"]["sigma"],
			# 							dims="spectroscopic_names")
			# spectroscopic_outliers_sd = pm.Gamma("spectroscopic_outliers_sd",
			# 							alpha=2,
			# 							beta=prior["spectroscopic_outliers"]["beta"],
			# 							dims="spectroscopic_names")
			# #-----------------------------------------------------------------------------
		#------------------------------------------------------------------------------
		#================================================================================

		#===================== Transformations =========================================
		#------------ True photometry --------------------------------
		if use_photometry:
			parallax = pm.Deterministic('parallax', 
								1000./distance,
								dims="source_id")

			true_photometry = pm.Deterministic('true_photometry',
								mlp_iso(age_iso,tefs,distance,n_stars),
								dims=("source_id","photometric_names"))
		#------------------------------------------------------------------

		#------------- True spectroscopy ---------------------
		if use_spectroscopy:
			true_spectroscopy = pm.Deterministic("true_spectroscopy",
								mlp_ldb(age_ldb,tefs,n_stars),
								dims=("source_id","spectroscopic_names"))
		#-----------------------------------------------------
		#=================================================================================

		#================== Intrinsic dispersion & Outliers ================================
		#----------------- Photometry ----------------------------
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

			# photometry = pm.Normal("photometry",
			# 					mu=true_photometry,
			# 					sigma=photometric_dispersion,
			# 					dims=("source_id","photometric_names"))
		#------------------------------------------------------------

		#------------------- Spectroscopy -----------------------------
		# if use_spectroscopy:
			# spectroscopic_components = [
			# 	pm.Normal.dist(mu=true_spectroscopy,
			# 				sigma=spectroscopic_dispersion),
			# 	pm.Normal.dist(mu=spectroscopic_outliers_mu, 
			# 				sigma=spectroscopic_outliers_sd)
			# 	]

			# spectroscopy = pm.Mixture("spectroscopy",
			# 				w=spectroscopic_outliers_weights,
			# 				comp_dists=spectroscopic_components, 
			# 				dims=("source_id","spectroscopic_names"))
			# spectroscopy = pm.Normal("spectroscopy",
			# 				mu=true_spectroscopy,
			# 				sigma=spectroscopic_dispersion,
			# 				dims=("source_id","spectroscopic_names"))
		#-------------------------------------------------------------
		#=================================================================================

		#===================== Likelihood =============================
		#- The true and observed are contracted to the not NaN values
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
			#------------- Lithium likelihood ------------------
			obs_spectroscopy = pm.Normal('obs_spectroscopy', 
				mu=true_spectroscopy[spectroscopy_ix], 
				sigma=spectroscopy_sd[spectroscopy_ix],
				observed=spectroscopy_mu[spectroscopy_ix])
		#===============================================================