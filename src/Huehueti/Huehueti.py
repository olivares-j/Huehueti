# (Full modified Huehueti.py content with type hints and comments)
import sys
import os
import numpy as np
import pandas as pn
import pymc as pm
import arviz as az
import dill
import xarray
import seaborn as sns
from typing import Optional, Dict, Any, List, Sequence, Union

#---------------- Matplotlib -------------------------------------
# Use the PDF backend so scripts can run headless and produce vector output.
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#------------------------------------------------------------------

# Local model/utility imports.
from Models import absolute_to_apparent,Model_v0
from MLPs import MLP_phot, MLP_teff

# Configure pandas display to show all columns when printing summaries.
pn.set_option('display.max_columns', None)


class Huehueti:
	"""
	Main high-level class wrapping the data ingestion, model setup,
	fitting (with PyMC), and visualization for the project.

	Primary responsibilities:
	- Read and preprocess input catalog (load_data)
	- Initialize neural-network isochrone (MLP) and probabilistic model (setup)
	- Run variational initialization and MCMC sampling (run)
	- Load and analyze saved traces (load_trace, convergence)
	- Produce a variety of diagnostic and scientific plots (plot_* methods)
	- Export summary statistics (save_statistics)
	"""
	def __init__(self,
		dir_out: str,
		file_mlp_phot: str,
		file_mlp_teff: str,
		observables: dict,
		hyperparameters: dict,
		absolute_photometry: list
	) -> None:
		# Output directory and path to MLP file
		self.dir_out = dir_out
		self.file_mlp_phot = file_mlp_phot
		self.file_mlp_teff = file_mlp_teff
		self.absolute_photometry = absolute_photometry

		# File paths for common outputs produced by the class methods.
		# These are constructed relative to dir_out.
		self.file_ids           = self.dir_out+"/Identifiers.csv"
		self.file_obs           = self.dir_out+"/Observations.nc"
		self.file_chains        = self.dir_out+"/Chains.nc"
		self.file_start         = self.dir_out+"/Initialization.pkl"
		self.file_prior         = self.dir_out+"/Prior.nc"
		self.file_vi_loss       = self.dir_out+"/Initializations.png"
		self.file_init_true     = self.dir_out+"/initial_true.csv"
		self.file_trace_globals = self.dir_out+"/Global_traces.pdf"
		self.file_trace_sources = self.dir_out+"/Sources_traces.pdf"
		self.file_pos           = self.dir_out+"/Posterior.pdf"
		self.file_cpp           = self.dir_out+"/Comparison_prior_posterior.pdf"
		self.file_prd           = self.dir_out+"/Predictions.pdf"
		self.file_cmd           = self.dir_out+"/Color-magnitude_diagram.png"
		self.file_hrd           = self.dir_out+"/Hertzprung-Russell_diagram.png"
		self.file_sts_src       = self.dir_out+"/Sources_statistics.csv"
		self.file_sts_glb       = self.dir_out+"/Global_statistics.csv"

		# A list containing every filter and its uncertainty and some metadata.
		default_observables = {
		"identifiers":"source_id",
		"photometry":['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
		"photometry_error":['phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error'],
		# "astrometry":["parallax"],
		# "astrometry_error":["parallax_error"],
		# "spectroscopy":["teff"],
		# "spectroscopy_error":["teff_error"]
		"astrometry":[],
		"astrometry_error":[],
		"spectroscopy":[],
		"spectroscopy_error":[]
		}

		for arg,val in default_observables.items():
			if not arg in observables:
				observables[arg] = val

		self.observables = observables
		print("The following observables will be used:")
		for k,v in self.observables.items():
			print("\t{0} : {1}".format(k,v))

		self.hyperparameters = hyperparameters

		# Primary identifier column name
		self.id_name = observables["identifiers"]

		# Names expected for mean values and uncertainties used throughout the code.
		self.names_mu = sum([
			observables["astrometry"],
			observables["photometry"],
			observables["spectroscopy"],
			],[])
		self.names_sd = sum([
			observables["astrometry_error"],
			observables["photometry_error"],
			observables["spectroscopy_error"]
			],[])

		self.observables_names = sum([
			self.names_mu,
			self.names_sd,
			],[])

		self.columns = sum([
			[self.id_name],
			self.observables_names,
			list(hyperparameters.values())
			],[])

		

		# Helpers grouping photometric and astrometric names into dicts for convenience.
		self.photometric_names = {
			"values":self.observables["photometry"],
			"errors":self.observables["photometry_error"]}
		self.astrometric_names = {
			"values":self.observables["astrometry"],
			"errors":self.observables["astrometry_error"]}
		self.spectroscopic_names = {
			"values":self.observables["spectroscopy"],
			"errors":self.observables["spectroscopy_error"]}

	def _isochrone_photometric_limits(self, distance: float) -> np.ndarray:
		"""
		Compute apparent magnitude limits for each photometric band based on the
		absolute magnitude lower limits encoded in the MLP. This is used to
		reject sources that are brighter (i.e. have lower absolute magnitude)
		than the isochrone model domain.

		Parameters
		----------
		distance : float
			Distance in parsecs used to transform absolute magnitude limits to apparent.
		"""
		try:
			# The MLP is serialized with dill and contains a key "phot_min"
			# which stores the minimum absolute magnitude (brightest) the NN covers.
			with open(self.file_mlp_phot, 'rb') as file:
				mlp = dill.load(file)
				phot_min = mlp["phot_min"]
		except FileNotFoundError as error:
			# Re-raise with a clearer message for the caller.
			raise FileNotFoundError("The isochrone model cannot be found. Please, provide a valid path.") from error
		else:
			# Convert absolute magnitude limit to apparent magnitude at given distance:
			# m = M + 5 log10(d) - 5
			phot_lim = phot_min + 5.0*np.log10(distance) - 5.0
			return phot_lim

	def load_data(self, 
		file_data: str,
		fill_nan: str = "max",
	) -> None:
		"""
		Read dataset CSV and perform preprocessing.

		Parameters
		----------
		file_data : str
			Path to input CSV file containing the columns defined in self.columns.
		fill_nan : str, optional
			Method to fill missing error values ("max" or "mean"). Default "max".
		source_parameters : dictionary
			If set, mapping from parameters to columns in file_data.
		"""
		assert fill_nan in ["max","mean"],"Error: fill_nan can only be max or mean"

		# ----------------Read dataset --------------------
		# Only read the expected columns for efficiency.
		df = pn.read_csv(file_data,usecols=self.columns)
		# Use the source_id as the DataFrame index.
		df.set_index("source_id",inplace=True)
		#--------------------------------------------------

		for key,value in self.hyperparameters.items():
			df.rename(mapper={value:key},inplace=True)
		
		#-------- Filter out sources brighter than limit ------------------------------------------------
		# Estimate an average distance from parallax to compute an apparent magnitude
		# limit for rejection. 1000/parallax to convert mas->pc (parallax must be in mas).
		distance = df["distance"].mean() if "distance" in df.columns else np.mean(1000/df["parallax"])
		phot_lim = self._isochrone_photometric_limits(distance=distance).to_numpy()

		# For each star, keep it if ALL photometric bands are fainter than the phot_lim
		# or if the measurement is missing (NaN). The mask_valid ends up boolean per-source.
		mask_valid = df.loc[:,self.observables["photometry"]].apply(
			lambda x : x >= phot_lim ,axis=1) | df.loc[:,self.observables["photometry"]].isnull()
		mask_valid = mask_valid.apply(lambda x: np.all(x),axis=1)
		df = df.loc[mask_valid]
		#-------------------------------------------------------------------------------------------------

		#-------------- Replace zeros and missing values for error columns -------------------------
		# Replace zero errors with NaN (zeros often mean missing or bad data), then fill
		# with either the column max or mean depending on fill_nan policy.
		for value,error in zip(self.names_mu,self.names_sd):
			# Replace 0.0 in error columns by NaN so we can fill with a sensible value.
			df[error] = df[error].replace(0.0,np.nan)
			# Choose replacement strategy
			replace = df[error].max() if fill_nan == "max" else df[error].mean()
			df[error] = df[error].fillna(value=replace)

			# If the band value itself is missing (NaN) then set the error back to NaN.
			# This protects from producing spurious error bars for missing measurements.
			df.loc[np.isnan(df[value]),error] = np.nan
		#-------------------------------------------------------------------------

		# Save processed data into the object
		self.data = df.loc[:,self.observables_names]
		self.df_hyp = df.loc[:,self.hyperparameters.keys()]

		print("Summary of input data:")
		print(self.data.describe())

		print("Summary of input hyperparameters:")
		print(self.df_hyp.describe())

		#----- Track ID -----------------
		self.ID = self.data.index.values
		#--------------------------------

		#----- Save identifiers to CSV for downstream tools --------------------------
		df = pn.DataFrame(self.ID,columns=[self.id_name])
		df.to_csv(path_or_buf=self.file_ids,index=False)
		#------------------------------------------------

		#-------- Observations to Arviz InferenceData (observed_data) ---------------
		# This transforms the DataFrame into an xarray.Dataset with a MultiIndex
		# so it can be attached as observed_data for ArviZ/InferenceData workflows.
		df = pn.DataFrame(self.data,
			columns=["obs"],
			index=pn.MultiIndex.from_product(
			iterables=[[0],[0],self.ID,self.names_mu],
			names=['chain', 'draw','source_id','observable']))
		xdata = xarray.Dataset.from_dataframe(df)
		observed = az.InferenceData(observed_data=xdata)
		az.to_netcdf(observed,self.file_obs)
		#------------------------------------------------------

	def setup(self,
		parameters: Dict[str,float],
		prior: Dict[str, Any]
	) -> None:
		"""
		Initialize the MLP and probabilistic model.

		Parameters
		----------
		parameters: dict,
			Global parameters to be set or inferred (None)
		prior : dict
			Prior specification passed to Model_v0.
		starting_points : optional initial values (currently unused)
		"""
		#----------------- Initialize NNs -------------------
		self.mlp_phot = MLP_phot(file_mlp=self.file_mlp_phot)
		self.mlp_teff = MLP_teff(file_mlp=self.file_mlp_teff)
		#----------------------------------------------------

		assert self.mlp_phot.targets == self.absolute_photometry,KeyError("Absolute bands do not correspond to PARSEC mlp ones")
		assert "age" in parameters.keys(), KeyError("The age parameter needs to be set either to float value or None to infer it")

		#------------- Parameters and Prior verification --------------------
		if parameters["age"] is None:
			assert "age"  in prior.keys(), KeyError('Please, provide a prior for the age parameter')
			assert prior["age"]["family"]  in ["TruncatedNormal","Uniform"], KeyError("Unknown family of age prior")
			assert prior["age"]["lower"] >= self.mlp_phot.age_domain[0],"Error at the lower limit of the age prior! Verify mlp_phot file"
			assert prior["age"]["upper"] <= self.mlp_phot.age_domain[1],"Error at the upper limit of the age prior! Verify mlp_phot file"
		elif isinstance(parameters["age"],float()):
			print("The age parameter will be fixed to: {0} Myr".format(parameters["age"]))
		else:
			KeyError("The age parameter can only be None or float")

		if "distance" in self.hyperparameters.keys():
			parameters["distance"] = self.df_hyp["distance"].to_numpy()

			print("The distance parameter will be fixed to the following values:")
			print(parameters["distance"])
		else:
			assert ["distance_mu","distance_sd"] in parameters.keys(),KeyError("If the distance is not a hyperparameter its hyperparameters must be provided")
			assert isinstance(parameters["distance_mu"],(None,float)),KeyError("The distance_mu parameter can only be None or float")
			assert isinstance(parameters["distance_sd"],(None,float)),KeyError("The distance_sd parameter can only be None or float")
			if parameters["distance_mu"] is None:
				assert "distance_mu" in prior.keys(), KeyError('Please, provide a prior for distance_mu parameter')
				assert prior["distance_mu"]["family"] in ["Gaussian","Uniform"], KeyError("Unknown family of distance_mu prior")
			if parameters["distance_sd"] is None:
				assert "distance_sd" in prior.keys(), KeyError('Please, provide a prior for distance_sd parameter')
				assert prior["distance_sd"]["family"] in ["Exponential","Gamma"], KeyError("Unknown family of distance_sd prior")

		if "photometric_dispersion" in parameters.keys():
			assert "photometric_dispersion" in prior.keys(), KeyError('Please, provide a prior for photometric_dispersion parameter')
			assert prior["photometric_dispersion"]["family"] in ["Exponential","Gamma"], KeyError("Unknown family of photometric_dispersion prior")
		#--------------------------------------------------------------------------------------------------------------------------------------------


		identifiers = self.data.index.values
		assert len(identifiers) > 1,"The number of photometric sources is less than two!"
		
		#---------------------- Data arrangement ---------------------------------------------
		if len(self.astrometric_names["values"])>0:
			astrometry_names = self.astrometric_names["values"]
			astrometry_mu = self.data.loc[:,self.astrometric_names["values"]].to_numpy()
			astrometry_sd = self.data.loc[:,self.astrometric_names["errors"]].to_numpy()
			astrometry_ix = np.where(np.isfinite(astrometry_mu))

			assert np.isfinite(astrometry_mu[astrometry_ix]).all(),"Error: NaN in astrometry_mu"
			assert np.isfinite(astrometry_sd[astrometry_ix]).all(),"Error: NaN in astrometry_sd"
		else:
			astrometry_names = None
			astrometry_mu = None
			astrometry_sd = None
			astrometry_ix = None

		if len(self.photometric_names["values"])>0:
			photometry_names = self.photometric_names["values"]
			photometry_mu = self.data.loc[:,self.photometric_names["values"]].to_numpy()
			photometry_sd = self.data.loc[:,self.photometric_names["errors"]].to_numpy()
			photometry_ix = np.where(np.isfinite(photometry_mu))

			assert np.isfinite(photometry_mu[photometry_ix]).all(),"Error: NaN in photometry_mu"
			assert np.isfinite(photometry_sd[photometry_ix]).all(),"Error: NaN in photometry_sd"
		else:
			photometry_names = None
			photometry_mu = None
			photometry_sd = None
			photometry_ix = None

		if len(self.spectroscopic_names["values"])>0:
			spectroscopy_names = self.spectroscopic_names["values"] 
			spectroscopy_mu = self.data.loc[:,self.spectroscopic_names["values"]].to_numpy()
			spectroscopy_sd = self.data.loc[:,self.spectroscopic_names["errors"]].to_numpy()
			spectroscopy_ix = np.where(np.isfinite(spectroscopy_mu))

			assert np.isfinite(spectroscopy_mu[spectroscopy_ix]).all(),"Error: NaN in spectroscopy_mu"
			assert np.isfinite(spectroscopy_sd[spectroscopy_ix]).all(),"Error: NaN in spectroscopy_sd"
		else:
			spectroscopy_names = None
			spectroscopy_mu = None
			spectroscopy_sd = None
			spectroscopy_ix = None
		#---------------------------------------------------------------------------------

		self.Model = Model_v0(
						mlp_phot=self.mlp_phot,
						mlp_teff=self.mlp_teff,
						parameters = parameters,
						prior = prior,
						identifiers = identifiers,
						astrometry_names = astrometry_names,
						astrometry_mu = astrometry_mu,
						astrometry_sd = astrometry_sd,
						astrometry_ix = astrometry_ix,
						photometry_names = photometry_names,
						photometry_mu = photometry_mu,
						photometry_sd = photometry_sd,
						photometry_ix = photometry_ix,
						spectroscopy_names = spectroscopy_names,
						spectroscopy_mu = spectroscopy_mu,
						spectroscopy_sd = spectroscopy_sd,
						spectroscopy_ix = spectroscopy_ix
						)
		for key,value in self.Model.initial_point().items():
			assert np.isfinite(value).all(),"Initial point error at {0}\n{1}".format(key,value)
		#-------------------------------------------------------------------------------------

	def plot_pgm(self, file: Optional[str] = None) -> None:
		"""
		Render and save a graphical model (PGM) from the PyMC model.
		"""
		file = file if file is not None else self.dir_out+"model_graph.png"
		graph = pm.model_to_graphviz(self.Model)
		graph.render(outfile=file,format="png")

	def run(self,
		tuning_iters: int = 3000,
		sample_iters: int = 2000,
		target_accept: float = 0.65,
		chains: int = 2,
		cores: int = None,
		step: Optional[Any] = None,
		step_size: Optional[float] = None,
		init_method: str = "advi",
		init_iters: int = int(1e6),
		init_absolute_tol: float = 5e-10,
		prior_predictive: bool = True,
		prior_iters: int = 2000,
		progressbar: bool = True,
		nuts_sampler: str = "numpyro",
		nuts_sampler_kwargs: Optional[Dict] = None,
		random_seed: Optional[int] = None) -> None:
		"""
		Performs the variational initialization (ADVI) followed by MCMC sampling.
		"""

		# Only run sampling if posterior file does not already exist.
		if not os.path.exists(self.file_chains):
			if not os.path.exists(self.file_start):
				#================== Optimization with variational inference ============================================
				if init_method.lower() == "advi":
					print("Finding initial positions with ADVI method")
					vi = pm.ADVI(model=self.Model)
				elif init_method.lower() == "fullrank_advi":
					print("Finding initial positions with FullRankADVI method")
					vi = pm.FullRankADVI(model=self.Model)
				elif init_method.lower() == "svgd":
					print("Finding initial positions with SVGD method")
					vi = pm.SVGD(
						n_particles=100,
						jitter=1,
						# obj_optimizer=pm.sgd(learning_rate=0.01),
						model=self.Model)
				else:
					sys.exit("Unrecognized VI method")

				
				# Convergence callbacks used to stop ADVI when parameter changes are small.
				cnv_abs = pm.callbacks.CheckParametersConvergence(
						tolerance=init_absolute_tol,
						diff="absolute",ord=None)
				tracker = pm.callbacks.Tracker(
					  	mean=vi.approx.mean.eval,
						std=vi.approx.std.eval)

				approx = vi.fit(
					n=init_iters,
					callbacks=[cnv_abs,tracker],
					progressbar=True)

				#------------- Plot the ADVI loss (last init_plot_iters iterations) ----------------
				fig = plt.figure(figsize=(16, 9))
				mu_ax = fig.add_subplot(221)
				std_ax = fig.add_subplot(222)
				hist_ax = fig.add_subplot(212)
				mu_ax.plot(tracker["mean"])
				mu_ax.set_title("Mean track")
				std_ax.plot(tracker["std"])
				std_ax.set_title("Std track")
				hist_ax.plot(vi.hist)
				hist_ax.set_yscale("log")
				hist_ax.set_title("Negative ELBO track")
				plt.savefig(self.file_vi_loss)
				plt.close()
				#-----------------------------------------------------------

				# Save initialization to disk so future runs can reuse it.
				with open(self.file_start, "wb") as out_file:
					dill.dump(approx, out_file)
			else:
				assert nuts_sampler.lower() != init_method.lower(),("Error: "+
				"To sample with the same method as the initialization "+
				"please remove file:\n {0}".format(self.file_start))

				with open(self.file_start, 'rb') as in_strm:
					approx = dill.load(in_strm)

			#----------- Extract values needed for sampler ---------------------
			mu_point = approx.mean.eval()
			sd_point = approx.std.eval()

			random_seed_list = pm.util._get_seeds_per_chain(random_seed, chains)
			approx_sample = approx.sample(
				draws=chains, 
				random_seed=random_seed_list[0],
				return_inferencedata=False
				)
			initial_points = [approx_sample[i] for i in range(chains)]
			#--------------------------------------------------------------------

			#=================== Sampling ==================================================
			if nuts_sampler.lower() == init_method.lower():
				print("WARNING: Sampling posterior with {0}".format(nuts_sampler.upper()))
				traces = []
				for chain in np.arange(chains):
					print("sampling chain: {0}".format(chain))
					vi.refine(
						n=tuning_iters,
						progressbar=True)
					vi.approx.hist = vi.hist
					approx = vi.approx
					tr = approx.sample(
						draws=sample_iters, 
						random_seed=None,
						return_inferencedata=True)
					traces.append(tr)
				trace = az.concat(traces,dim="chain")

			elif nuts_sampler == "numpyro":
				#---------- Posterior sampling using default sampler backend (e.g., numpyro) -----------
				trace = pm.sample(
					draws=sample_iters,
					initvals=initial_points,
					nuts_sampler=nuts_sampler,
					tune=tuning_iters,
					chains=chains, 
					progressbar=progressbar,
					target_accept=target_accept,
					discard_tuned_samples=True,
					return_inferencedata=True,
					nuts_sampler_kwargs=nuts_sampler_kwargs,
					model=self.Model
					)

			elif nuts_sampler == "pymc":
				#--------------- Prepare step with a diagonal quadpotential -----------------
				potential = pm.step_methods.hmc.quadpotential.QuadPotentialDiagAdapt(
							n=len(mu_point),
							initial_mean=mu_point,
							initial_diag=sd_point**2, 
							initial_weight=10)

				step = pm.NUTS(
						potential=potential,
						model=self.Model,
						target_accept=target_accept
						)

				print("Sampling the model ...")

				#---------- Posterior sampling with a configured step ----------
				trace = pm.sample(
					draws=sample_iters,
					initvals=initial_points,
					step=step,
					nuts_sampler=nuts_sampler,
					tune=tuning_iters,
					chains=chains, 
					cores=cores,
					progressbar=progressbar,
					discard_tuned_samples=True,
					return_inferencedata=True,
					nuts_sampler_kwargs=nuts_sampler_kwargs,
					model=self.Model
					)
			else:
				trace = pm.sample(
					draws=sample_iters,
					initvals=initial_points,
					nuts_sampler=nuts_sampler,
					tune=tuning_iters,
					chains=chains, 
					cores=cores,
					progressbar=progressbar,
					discard_tuned_samples=True,
					return_inferencedata=True,
					nuts_sampler_kwargs=nuts_sampler_kwargs,
					model=self.Model
					)
			

			#--------- Save posterior samples to disk in ArviZ NetCDF representation ------------
			print("Saving posterior samples ...")
			az.to_netcdf(trace,self.file_chains)
			#-------------------------------------
			del trace
			#================================================================================

		# Optionally sample prior predictive draws and save them for later model checking.
		if prior_predictive and not os.path.exists(self.file_prior):
			#-------- Prior predictive -------------------
			print("Sampling prior predictive ...")
			prior_pred = pm.sample_prior_predictive(
						samples=prior_iters,
						model=self.Model)
			print("Saving prior predictive ...")
			az.to_netcdf(prior_pred,self.file_prior)
			#---------------------------------------------

		print("Sampling done!")
		

	def load_trace(self, file_chains: Optional[str] = None,chains=None) -> None:
		'''
		Loads a previously saved sampling of the model (ArviZ InferenceData from NetCDF).

		After loading, this method organizes variable names into convenient categories
		(self.source_variables, self.global_variables, etc.) used by plotting and summary methods.
		'''
		file_chains = self.file_chains if (file_chains is None) else file_chains

		# If IDs not present in-memory, read saved identifiers file.
		if not hasattr(self,"ID"):
			#----- Load identifiers ------
			self.ID = pn.read_csv(self.file_ids).to_numpy().flatten()

		print("Loading existing samples ... ")

		#---------Load posterior ---------------------------------------------------
		try:
			posterior = az.from_netcdf(file_chains)
			if chains is not None:
				posterior = posterior.sel(chain=chains)
		except ValueError:
			# Fatal error if file cannot be parsed / is incompatible.
			sys.exit("ERROR at loading {0}".format(file_chains))
		#------------------------------------------------------------------------

		#----------- Load prior (if exists) and extend posterior with prior group ----
		try:
			prior = az.from_netcdf(self.file_prior)
		except:
			prior = None
			self.ds_prior = None
		
		if prior is not None:
			posterior.extend(prior)
			self.ds_prior = posterior.prior
		#-------------------------------------------------------------------------

		# Store the combined InferenceData
		self.trace = posterior

		#--------- Extract posterior dataset and ensure it exists --------------------
		try:
			self.ds_posterior = self.trace.posterior
		except ValueError:
			sys.exit("There is no posterior in trace")
		#------------------------------------------------------------------------

		#------- Build lists of variables grouped by use (source-level vs global) -----------
		source_variables = list(filter(lambda x: (("mass" in x)
											or ("distance" in x)
											or ("astrometry" in x)
											or ("photometry" in x)
											or ("spectroscopy" in x)
											), 
											self.ds_posterior.data_vars))
		global_variables = list(filter(lambda x: (("age" in x)
											or (x == "distance_mu")
											or (x == "distance_sd")
											or ("astrometric_" in x)
											or ("photometric_" in x)
											or ("spectroscopic_" in x)
											),
											self.ds_posterior.data_vars))
	
		# Default groups used for plotting/summaries.
		trace_variables = global_variables.copy()
		source_sts_variables = source_variables.copy()
		global_sts_variables = global_variables.copy()
		global_cpp_var = global_variables.copy()
		source_prd_var = source_variables.copy()

		#----------- Case specific variable filtering -------------
		tmp_src     = source_variables.copy()
		tmp_sts_src = source_variables.copy()
		tmp_sts_glb = global_variables.copy()
		tmp_cpp   = global_variables.copy()
		tmp_prd   = source_variables.copy()

		
		for var in tmp_sts_src:
			if not (
				(var == "mass") or 
				(var == "distance") or
				("astrometry" in var) or 
				("photometry" in var) or
				("spectroscopy" in var)
				):
				source_sts_variables.remove(var)

		# Keep only relevant global statistics variables: age, distance central/dispersion and photometric hyperparams
		for var in tmp_sts_glb:
			if not (
				("age" in var) or 
				(var == "distance_mu") or 
				(var == "distance_sd") or
				("photometric_" in var)
				):
				global_sts_variables.remove(var)

		# Variables to compare prior/posterior (cpp): only keep age, distance central/dispersion and photometric_*
		for var in tmp_cpp:
			if not (("age" in var)
				or (var == "distance_mu")
				or (var == "distance_sd")
				or ("photometric_" in var)):
				global_cpp_var.remove(var)

		for var in tmp_prd:
			if (
				("astrometry" in var)
				or ("photometry" in var)
				or ("spectroscopy" in var)
				):
				# keep (pass)
				pass
			else:
				source_prd_var.remove(var)

		# Store computed groupings on the object for later use by plotting functions.
		self.source_variables  = source_variables
		self.global_variables  = global_variables
		self.trace_variables   = trace_variables
		self.cpp_variables     = global_cpp_var
		self.prd_variables     = source_prd_var
		self.global_sts_variables  = global_sts_variables
		self.source_sts_variables  = source_sts_variables

	def convergence(self) -> None:
		"""
		Compute and print basic MCMC convergence diagnostics using ArviZ:
		- rhat (Gelman-Rubin)
		- ess (effective sample size)
		Also prints the mean step size per chain from sample_stats.
		"""
		print("Computing convergence statistics ...")
		rhat  = az.rhat(self.ds_posterior)
		ess   = az.ess(self.ds_posterior)

		print("Gelman-Rubin statistics:")
		for var in self.ds_posterior.data_vars:
			print("{0} : {1:2.4f}".format(var,np.mean(rhat[var].values)))

		print("Effective sample size:")
		for var in self.ds_posterior.data_vars:
			print("{0} : {1:2.4f}".format(var,np.mean(ess[var].values)))

		if hasattr(self.trace,"sample_stats"):
			print("Step size:")
			# The trace object has sample_stats; average over draws to report mean step_size per chain.
			for i,val in enumerate(self.trace.sample_stats["step_size"].mean(dim="draw")):
				print("Chain {0}: {1:3.8f}".format(i,val))
			print("LogP:")
			# The trace object has sample_stats; average over draws to report mean logP per chain.
			for i,val in enumerate(self.trace.sample_stats["lp"].mean(dim="draw")):
				print("Chain {0}: {1:3.8f}".format(i,val))

	def plot_chains(self,
		file_trace_sources: Optional[str] = None,
		file_trace_globals: Optional[str] = None,
		IDs: Optional[Sequence[Union[str,int]]] = None,
		divergences: Union[str, bool] = 'bottom', 
		figsize: Optional[tuple] = None, 
		lines: Optional[Any] = None, 
		combined: bool = False,
		compact: bool = False,
		legend: bool = True,
		plot_kwargs: Optional[Dict[str,Any]] = None, 
		hist_kwargs: Optional[Dict[str,Any]] = None, 
		trace_kwargs: Optional[Dict[str,Any]] = None,
		fontsize_title: int = 16) -> None:
		"""
		Plot trace plots for global parameters and (optionally) per-source parameters.
		"""
		# If no IDs provided and there are no global variables, nothing to plot.
		if IDs is None and len(self.global_variables) == 0:
			return

		print("Plotting traces ...")

		file_trace_globals = file_trace_globals if (file_trace_globals is not None) else self.file_trace_globals
		file_trace_sources = file_trace_sources if (file_trace_sources is not None) else self.file_trace_sources

		# If specific source IDs requested, create a PDF with one page per source.
		if IDs is not None:
			pdf = PdfPages(filename=file_trace_sources)
			#--------- Loop over ID in list ---------------
			for i,ID in enumerate(IDs):
				# Validate provided ID is present in the dataset.
				id_in_IDs = np.isin(ID,self.ID)
				print(id_in_IDs)
				if not np.any(id_in_IDs) :
					sys.exit("{0} {1} is not valid. Use strings".format(self.id_name,ID))
				idx = np.where(id_in_IDs)[0]
				coords = {"source_id":ID}
				plt.figure(0)
				axes = az.plot_trace(self.ds_posterior,
						var_names=self.source_variables,
						coords=coords,
						figsize=figsize,
						lines=lines, 
						combined=combined,
						compact=compact,
						legend=legend,
						plot_kwargs=plot_kwargs, 
						hist_kwargs=hist_kwargs, 
						trace_kwargs=trace_kwargs)

				# Tweak axis labels to include units where appropriate.
				for ax in axes:
					# --- Set units in parameters ------------------------------
					title = ax[0].get_title()
					if "mass" in title:
						ax[0].set_xlabel("$M_\\odot$")
					if "distance" in title:
						ax[0].set_xlabel("pc")
					#-----------------------------------------------------------
					ax[1].set_xlabel("Iterations")
				
				plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,hspace=0.5,wspace=0.1)
				plt.gcf().suptitle(self.id_name +" "+str(ID),fontsize=fontsize_title)

				#-------------- Save fig --------------------------
				pdf.savefig(bbox_inches='tight')
				plt.close(0)
			pdf.close()

		# Produce a PDF containing trace plots for all global variables.
		pdf = PdfPages(filename=file_trace_globals)
		for var_name in self.global_variables:
			axes = az.plot_trace(self.ds_posterior,
					var_names=var_name,
					figsize=figsize,
					lines=lines, 
					combined=combined,
					compact=compact,
					legend=legend,
					plot_kwargs=plot_kwargs, 
					hist_kwargs=hist_kwargs, 
					trace_kwargs=trace_kwargs,
					# labeller=az.labels.NoVarLabeller()
					)
			for ax in axes:
				# Add readable units to axis labels where applicable
				title = ax[0].get_title()
				if "age" in title:
					ax[0].set_xlabel("Myr")
				if "photometric_dispersion" in title:
					ax[0].set_xlabel("magnitude")
				#-----------------------------------------------------------
				ax[1].set_xlabel("Iteration")

			#-------------- Save fig --------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close(1)
		pdf.close()

	def plot_cpp(self,
		file_cpp: Optional[str] = None,
		figsize: Optional[tuple] = None,
	) -> None:
		"""
		Plot comparison of prior and posterior distributions for a set of global variables.
		"""
		print("Plotting prior and posterior ...")
		file_cpp = file_cpp if file_cpp is not None else self.file_cpp

		pdf = PdfPages(filename=file_cpp)
		for var in self.cpp_variables:
			plt.figure(0,figsize=figsize)
			ax = az.plot_dist_comparison(self.trace,var_names=var)
			pdf.savefig(bbox_inches='tight')
			plt.close(0)
		pdf.close()


	def plot_posterior(self,
		file_posterior: Optional[str] = None,
		figsize: Optional[tuple] = None,
	) -> None:
		"""
		Plot marginal posterior distributions for global variables.
		"""
		print("Plotting posterior ...")
		file_pos = file_posterior if file_posterior is not None else self.file_pos

		pdf = PdfPages(filename=file_pos)
		for var in self.global_variables:
			plt.figure(0,figsize=figsize)
			ax = az.plot_posterior(self.ds_posterior,var_names=var)
			pdf.savefig(bbox_inches='tight')
			plt.close(0)
		pdf.close()

	def plot_predictions(self,
		file_plots: Optional[str] = None,
		figsize: Optional[tuple] = None,
		fmt_121: str = "k:"
	) -> None:
		"""
		Plot predicted vs observed values for per-source predicted quantities (e.g., predicted photometry).
		"""
		print("Plotting posterior predictive distributions ...")
		file_plots = file_plots if file_plots is not None else self.file_prd

		pdf = PdfPages(filename=file_plots)

		for case in self.prd_variables:
			# Build the column list (value + error) for this case (photometry or astrometry)
			columns = sum([self.observables[case],self.observables[case+"_error"]],[])
			# Mapper from observable->error column name for renaming predicted sd later
			mapper = {key: value for key,value in zip(
							self.observables[case],
							self.observables[case+"_error"])}
			df_obs = self.data.loc[:,columns].copy()

			# Convert posterior draws for the variable into a DataFrame and reformat.
			df_prd = self.trace.posterior[case].to_dataframe().unstack()
			# After unstacking the first level is chain/draw; drop it to get source_id and variable.
			df_prd.columns = df_prd.columns.droplevel(level=0)

			# Compute mean and std across posterior draws for each source
			df_prd_mu = df_prd.groupby("source_id").mean()
			df_prd_sd = df_prd.groupby("source_id").std()
			# Rename predicted sd columns to match the observed error column names
			df_prd_sd.rename(columns=mapper,inplace=True)
			# Join mean and sd and prefix columns with "pred_"
			df_prd = df_prd_mu.join(df_prd_sd)
			df_prd.rename(columns=lambda x: "pred_"+str(x),inplace=True)

			# Join observed and predicted summaries into a single DataFrame
			df = df_obs.join(df_prd)

			# For each observable, plot observed vs predicted with errorbars.
			for var,err in zip(
				self.observables[case],
				self.observables[case+"_error"]):
				plt.figure(0,figsize=figsize)
				ax = plt.gca()
				ax.errorbar(x=df[var], y=df["pred_"+var]-df[var], 
						xerr=df[err],
						yerr=df["pred_"+err], 
						fmt='.',
						capsize=2,
						color="black", 
						ecolor='gray', 
						zorder=1)
				ax.set_xlabel("Observed {0}".format(var))
				ax.set_ylabel("$\\Delta$ (P-O) {0}".format(var))
				pdf.savefig(bbox_inches='tight')
				plt.close(0)
		pdf.close()



	def plot_cmd(self,
		file_plot: Optional[str] = None,
		figsize: Optional[tuple] = None,
		cmd: Dict[str, Any] = {"magnitude":"g","color":["g","rp"]},
		n_samples: int = 10,
		n_points: int = 100,
		scatter_palette: str = "dark",
		lines_color: str = "orange",
		alpha: float = 1.0,
		dpi: int = 600,
	) -> None:
		"""
		Plot a color-magnitude diagram (CMD) showing observed points, predicted points,
		and sampled isochrones drawn from the posterior distribution.
		"""
		print("Plotting CMD ...")

		msg_n = "The required n_samples {0} is larger than those in the posterior.".format(n_samples)

		# Ensure we have enough posterior draws to sample unique ages.
		assert n_samples <= self.ds_posterior.sizes["draw"], msg_n

		#----- Draw ages from posterior and prepare a theta grid used by MLP -----------
		ages = np.random.choice(self.trace.posterior["age"].values.flatten(),
					size=n_samples,replace=False)
		# Use mean distance from posterior for converting absolute->apparent.
		distance = np.mean(self.trace.posterior["distance"].values.flatten())
		mass     = np.linspace(self.mlp_phot.mass_domain[0],self.mlp_phot.mass_domain[1],n_points)

		dfs_smp = []
		# For each sampled age, query MLP to produce absolute photometry and convert to apparent.
		for age in ages:
			absolute_photometry = self.mlp_phot(age,mass,n_points)
			photometry = absolute_to_apparent(absolute_photometry,distance)
			df_tmp = pn.DataFrame(
					data=photometry.eval(),
					columns=self.observables["photometry"])
			# The sampled isochrone points are indexed by star and age for plotting.
			df_tmp.index.name = "star"
			df_tmp["age"] = age
			df_tmp.set_index("age",append=True,inplace=True)
			dfs_smp.append(df_tmp)
		df_smp = pn.concat(dfs_smp,ignore_index=False)
		#------------------------------------------------------------------------------

		#-------------- Photometric data: compute posterior predicted mean per source -------
		df_pht = self.trace.posterior["photometry"].to_dataframe().unstack("photometry_names")
		df_pht.columns = df_pht.columns.droplevel(level=0)
		df_pos = df_pht.groupby("source_id").mean()
		#----------------------------------------------------------------------------------------

		#---------------- Color and magnitude  ------------------------
		# Use a set to avoid duplicated columns if magnitude also in color list
		columns = list(set(sum([[cmd["magnitude"]],cmd["color"]],[])))
		#--------------------------------------------------------------

		#---------- Dataframes ------------------
		df_obs = self.data.loc[:,columns].copy()
		df_prd = df_pos.loc[:,columns].copy()
		df_smp = df_smp.loc[:,columns].copy()
		#-----------------------------------------

		#------------------- Concatenate observed and predicted for common scatter plotting -----------------
		df_obs["Origin"] = "Observed"
		df_prd["Origin"] = "Predicted"
		df_all = pn.concat([df_obs,df_prd],
					ignore_index=True) #Otherwise seaborn scatterplot may trip on duplicated indices
		#------------------------------------------

		#------------------------ Compute color (color = band1 - band2) ------------------------
		df_all["color"] = df_all.apply(lambda x: 
						x[cmd["color"][0]] - 
						x[cmd["color"][1]],axis=1)
		df_smp["color"] = df_smp.apply(lambda x: 
						x[cmd["color"][0]] - 
						x[cmd["color"][1]],axis=1)
		#---------------------------------------------------------

		file_plot = file_plot if (file_plot is not None) else self.file_cmd
		#----------------- Produce CMD --------------------------------------------
		plt.figure(0,figsize=figsize)
		ax = sns.scatterplot(data=df_all,
						x="color",
						y=cmd["magnitude"],
						palette=sns.color_palette(scatter_palette,n_colors=2),
						hue="Origin",
						style="Origin",
						s=10,
						zorder=0)
		# Overplot sampled isochrone lines colored by age (but not showing legend)
		sns.lineplot(data=df_smp,
						x="color",
						y=cmd["magnitude"],
						palette=sns.color_palette([lines_color], n_samples),
						hue="age",
						legend=False,
						alpha=alpha,
						sort=False,
						zorder=1,
						ax=ax)
		ax.set_xlabel("{0} - {1} {2}".format(
			cmd["color"][0],cmd["color"][1],"[mag]"))
		ax.set_ylabel("{0} {1}".format(cmd["magnitude"],"[mag]"))
		ax.invert_yaxis()  # Magnitudes increase downward in plots
		ax.set_title("Apparent photometry")
		plt.savefig(file_plot,bbox_inches='tight',dpi=dpi)
		plt.close(0)

	def plot_hrd(self,
		file_plot: Optional[str] = None,
		figsize: Optional[tuple] = None,
		magnitude: str = "phot_g_mean_mag",
		n_samples: int = 10,
		n_points: int = 100,
		scatter_palette: str = "dark",
		lines_color: str = "orange",
		alpha: float = 1.0,
		dpi: int = 600,
	) -> None:
		"""
		Plot a Hertzprung-Russell diagram (HRD) showing observed points, predicted points,
		and sampled isochrones drawn from the posterior distribution.
		"""
		print("Plotting HRD ...")

		msg_n = "The required n_samples {0} is larger than those in the posterior.".format(n_samples)

		# Ensure we have enough posterior draws to sample unique ages.
		assert n_samples <= self.ds_posterior.sizes["draw"], msg_n

		#----- Draw ages from posterior and prepare a theta grid used by MLP -----------
		ages = np.random.choice(self.trace.posterior["age"].values.flatten(),
					size=n_samples,replace=False)
		mass     = np.linspace(self.mlp_phot.mass_domain[0],self.mlp_phot.mass_domain[1],n_points)

		dfs_smp = []
		# For each sampled age, query MLP to produce absolute photometry and convert to apparent.
		for age in ages:
			teff = self.mlp_teff(age,mass,n_points)
			absolute_photometry = self.mlp_phot(age,mass,n_points)
			photometry = absolute_to_apparent(absolute_photometry,distance)
			df_tmp = pn.DataFrame(
					data=photometry.eval(),
					columns=self.observables["photometry"])
			# The sampled isochrone points are indexed by star and age for plotting.
			df_tmp.index.name = "star"
			df_tmp["age"] = age
			df_tmp["teff"] = teff.flatten().eval()
			df_tmp.set_index("age",append=True,inplace=True)
			dfs_smp.append(df_tmp)
		df_smp = pn.concat(dfs_smp,ignore_index=False)
		#------------------------------------------------------------------------------

		#-------------- Photometric data: compute posterior predicted mean per source -------
		df_pht = self.trace.posterior["photometry"].to_dataframe().unstack("photometry_names")
		df_pht.columns = df_pht.columns.droplevel(level=0)
		df_pht = df_pht.groupby("source_id").mean()
		#----------------------------------------------------------------------------------------

		#-------------- Spectroscopic data: compute posterior predicted mean per source -------
		df_spc = self.trace.posterior["spectroscopy"].to_dataframe().unstack("spectroscopy_names")
		df_spc.columns = df_spc.columns.droplevel(level=0)
		df_spc = df_spc.groupby("source_id").mean()
		#----------------------------------------------------------------------------------------

		#---------------- Color and magnitude  ------------------------
		# Use a set to avoid duplicated columns if magnitude also in color list
		columns = [magnitude,"teff"]
		#--------------------------------------------------------------

		#---------- Dataframes ------------------
		df_obs = self.data.loc[:,columns].copy()
		df_prd = df_pht.join(df_spc).loc[:,columns].copy()
		df_smp = df_smp.loc[:,columns].copy()
		#-----------------------------------------

		#------------------- Concatenate observed and predicted for common scatter plotting -----------------
		df_obs["Origin"] = "Observed"
		df_prd["Origin"] = "Predicted"
		df_all = pn.concat([df_obs,df_prd],
					ignore_index=True) #Otherwise seaborn scatterplot may trip on duplicated indices
		#------------------------------------------

		file_plot = file_plot if (file_plot is not None) else self.file_hrd
		#----------------- Produce CMD --------------------------------------------
		plt.figure(0,figsize=figsize)
		ax = sns.scatterplot(data=df_all,
						x="teff",
						y=magnitude,
						palette=sns.color_palette(scatter_palette,n_colors=2),
						hue="Origin",
						style="Origin",
						s=10,
						zorder=0)
		# Overplot sampled isochrone lines colored by age (but not showing legend)
		sns.lineplot(data=df_smp,
						x="teff",
						y=magnitude,
						palette=sns.color_palette([lines_color], n_samples),
						hue="age",
						legend=False,
						alpha=alpha,
						sort=False,
						zorder=1,
						ax=ax)
		ax.set_xlabel("Teff [K]")
		ax.set_ylabel("{0} {1}".format(magnitude,"[mag]"))
		ax.invert_yaxis()  # Magnitudes increase downward in plots
		ax.set_title("Apparent photometry")
		plt.savefig(file_plot,bbox_inches='tight',dpi=dpi)
		plt.close(0)

	
	def save_statistics(self,
		file_globals: Optional[str] = None,
		file_sources: Optional[str] = None,
		hdi_prob: float = 0.95,
		stat_focus: str = "mean",
		kind: str = "stats") -> None:
		'''
		Compute and save summary statistics for sources and global parameters.
		'''
		file_sources = file_sources if file_sources is not None else self.file_sts_src
		file_globals = file_globals if file_globals is not None else self.file_sts_glb

		print("Computing statistics ...")

		#-------------- Source statistics ----------------------------
		dfs = []
		for case in self.source_sts_variables:
			if case in ["distance","mass","teff"]:
				df_tmp  = az.summary(self.ds_posterior,
							var_names=case,
							stat_focus = stat_focus,
							hdi_prob=hdi_prob,
							round_to=5,
							extend=True,
							kind=kind
							)
				df_tmp.set_index(self.ID,inplace=True)
				df_tmp.columns = pn.MultiIndex.from_product([[case], df_tmp.columns])
				df_tmp = df_tmp.stack(sort=False,future_stack=False)
				df_tmp = df_tmp.rename_axis(index=[self.id_name,"statistic"])
				dfs.append(df_tmp)
			elif case == "astrometry":
				for var in self.observables[case]:
					df_tmp  = az.summary(self.ds_posterior,
								var_names=case,
								coords={"astrometry_names":var},
								stat_focus = stat_focus,
								hdi_prob=hdi_prob,
								round_to=5,
								extend=True,
								kind=kind
								)
					df_tmp.set_index(self.ID,inplace=True)
					df_tmp.columns = pn.MultiIndex.from_product([[var], df_tmp.columns])
					df_tmp = df_tmp.stack(sort=False,future_stack=False)
					df_tmp = df_tmp.rename_axis(index=[self.id_name,"statistic"])
					dfs.append(df_tmp)
			elif case == "photometry":
				for var in self.observables[case]:
					df_tmp  = az.summary(self.ds_posterior,
								var_names=case,
								coords={"photometry_names":var},
								stat_focus = stat_focus,
								hdi_prob=hdi_prob,
								round_to=5,
								extend=True,
								kind=kind
								)
					df_tmp.set_index(self.ID,inplace=True)
					df_tmp.columns = pn.MultiIndex.from_product([[var], df_tmp.columns])
					df_tmp = df_tmp.stack(sort=False,future_stack=False)
					df_tmp = df_tmp.rename_axis(index=[self.id_name,"statistic"])
					dfs.append(df_tmp)
			else:
				sys.exit("Unrecognized case: {0}".format(case))

		df_source = pn.concat(dfs,axis=1,ignore_index=False)
		#---------- Save source data frame to CSV ----------------------
		df_source.to_csv(path_or_buf=file_sources)
		#-------------- Global statistics ----------------------------------
		if len(self.global_sts_variables) > 0:
			df_global = az.summary(self.ds_posterior,
							var_names=self.global_sts_variables,
							stat_focus=stat_focus,
							hdi_prob=hdi_prob,
							round_to=5,
							extend=True
							)
			df_global.to_csv(path_or_buf=file_globals,index_label="Parameter")
		#---------------------------------------------------------------------
			


if __name__ == "__main__":

	# Example run when executed as a script. These defaults assume a certain
	# directory layout (data/, mlps/, outputs/) relative to the current working dir.

	age,distance,n_stars,seed = 120,136,10,1
	dir_base    = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC_tests/"
	dir_mlps    = "/home/jolivares/Models/PARSEC/Gaia_EDR3_15-400Myr/MLPs/"

	dir_inputs  = dir_base + "inputs/"
	dir_outputs = dir_base + "outputs/"
	base_name   = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"
	file_data = dir_inputs  + base_name.format(age,distance,n_stars,seed)+".csv"
	dir_out   = dir_outputs + base_name.format(age,distance,n_stars,seed)+"_fixed_distance_numpyro_new/"

	file_mlp_phot = dir_mlps + "Phot_l7_s512/mlp.pkl"
	file_mlp_teff = dir_mlps + "Teff_l16_s512/mlp.pkl"

	os.makedirs(dir_out,exist_ok=True)

	absolute_photometry = ['Gmag', 'G_BPmag', 'G_RPmag']
	observables = {
	"photometry":['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
	"photometry_error":['phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error'],
	}

	parameters = {"age":None}
	hyperparameters = {"distance":"distance"}

	# Example priors to be passed to Huehueti.setup
	prior = {
		'age' : {
			'family' : 'Uniform',
			'mu'    : 120.,
			'sigma' : 20.,
			'lower' : 20,
			'upper' : 200,
			},
		'distance_mu' : {
			'family' : 'Gaussian',
			'mu' : 136.,
			'sigma' : 5.
			},
		"distance_sd":{
			"family": "Exponential",
			"scale" : 5.
			},
		"photometric_dispersion":{
			"family": "Exponential",
			"sigma" : 0.01,
			"beta"  : 100.0
			},
		"astrometric_outliers":{
			"weights" : [90,10],
			"lower"   : 50.0,
			"upper"   : 150.0,
			"beta"    : 1/20.
		},
		"photometric_outliers":{
			"weights" : [90,10],
			"lower"   : 0.0,
			"upper"   : 30.0,
			"beta"    : 1/20.
		},
	}

	hue = Huehueti(
		dir_out = dir_out, 
		file_mlp_phot=file_mlp_phot,
		file_mlp_teff=file_mlp_teff,
		observables=observables,
		absolute_photometry=absolute_photometry,
		hyperparameters = hyperparameters,
		)
	hue.load_data(
		file_data = file_data
		)
	hue.setup(
		parameters = parameters, 
		prior = prior
		)
	# hue.plot_pgm()
	hue.run(
		init_method="advi",
		# init_method="fullrank_advi",
		init_iters=int(5e5),
		# nuts_sampler="advi",
		# nuts_sampler="fullrank_advi",
		nuts_sampler="numpyro",
		# tuning_iters=int(1e4),
		tuning_iters=int(2e3),
		sample_iters=int(2e3),
		prior_iters=int(2e3),
		chains=4,
		cores=4)
	hue.load_trace()#chains=[0,2,3])
	hue.convergence()
	hue.plot_chains()
	hue.plot_posterior()
	hue.plot_cpp()
	hue.plot_predictions()
	hue.plot_cmd(cmd={"magnitude":"phot_g_mean_mag","color":["phot_g_mean_mag","phot_rp_mean_mag"]})
	# hue.plot_hrd(magnitude="phot_g_mean_mag")
	hue.save_statistics()