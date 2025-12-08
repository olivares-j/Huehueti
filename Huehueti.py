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
from Models import Model_v0,Model_v1
from MLPs import MLP
from Functions import absolute_to_apparent, apparent_to_absolute

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
		file_mlp: str,
	) -> None:
		# Output directory and path to MLP file
		self.dir_out = dir_out
		self.file_mlp = file_mlp

		# File paths for common outputs produced by the class methods.
		# These are constructed relative to dir_out.
		self.file_ids           = self.dir_out+"/Identifiers.csv"
		self.file_obs           = self.dir_out+"/Observations.nc"
		self.file_chains        = self.dir_out+"/Chains.nc"
		self.file_start         = self.dir_out+"/Initialization.pkl"
		self.file_prior         = self.dir_out+"/Prior.nc"
		self.file_advi_loss     = self.dir_out+"/Initializations.png"
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
		observables = {
		"identifiers":"source_id",
		"absolute":['Gmag', 'G_BPmag', 'G_RPmag', 'gP1mag', 'rP1mag', 'iP1mag', 'zP1mag','yP1mag', 'Jmag', 'Hmag', 'Ksmag'],
		"photometry":['g', 'bp', 'rp','gmag','rmag','imag','ymag','zmag','Jmag','Hmag','Kmag'],
		"photometry_error":['g_error','bp_error','rp_error', 'e_gmag', 'e_rmag', 'e_imag', 'e_ymag', 'e_zmag', 'e_Jmag', 'e_Hmag', 'e_Kmag' ],
		"photometry_units":["[mag]","[mag]","[mag]","[mag]","[mag]","[mag]","[mag]","[mag]","[mag]","[mag]","[mag]"],
		"astrometry":["parallax"],
		"astrometry_error":["parallax_error"],
		"astrometry_units":["[mas]"]
		}

		# Primary identifier column name
		self.id_name = observables["identifiers"]
		self.observables = observables

		# Names expected for mean values and uncertainties used throughout the code.
		self.names_mu = sum([
			observables["astrometry"],
			observables["photometry"]
			],[])
		self.names_sd = sum([
			observables["astrometry_error"],
			observables["photometry_error"]
			],[])
		self.columns = sum([
			[self.id_name],
			self.names_mu,
			self.names_sd
			],[])

		self.observables = observables

		# Helpers grouping photometric and astrometric names into dicts for convenience.
		self.photometric_names = {"values":observables["photometry"],"errors":observables["photometry_error"]}
		self.astrometric_names = {"values":observables["astrometry"],"errors":observables["astrometry_error"]}

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
			with open(self.file_mlp, 'rb') as file:
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
		n_stars: Optional[int] = None
	) -> None:
		"""
		Read dataset CSV and perform preprocessing.

		Parameters
		----------
		file_data : str
			Path to input CSV file containing the columns defined in self.columns.
		fill_nan : str, optional
			Method to fill missing error values ("max" or "mean"). Default "max".
		n_stars : int or None
			If set, only keep the last n_stars rows (the code uses iloc[-n_stars:]).
		"""
		assert fill_nan in ["max","mean"],"Error: fill_nan can only be max or mean"

		# ----------------Read dataset --------------------
		# Only read the expected columns for efficiency.
		df = pn.read_csv(file_data,usecols=self.columns)
		# Use the source_id as the DataFrame index.
		df.set_index("source_id",inplace=True)
		#--------------------------------------------------
		
		#-------- Filter out sources brighter than limit ------------------------------------------------
		# Estimate an average distance from parallax to compute an apparent magnitude
		# limit for rejection. 1000/parallax to convert mas->pc (parallax must be in mas).
		phot_lim = self._isochrone_photometric_limits(distance=np.mean(1000/df["parallax"])).to_numpy()

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

		print("Summary of input data:")
		print(df.describe())

		# If n_stars is provided, keep only the last n_stars rows.
		if n_stars is not None:
			print(50*">")
			print("Using only these sources:")
			df = df.iloc[-n_stars:]
			print(df.describe())
			print(50*"<")

		# Save processed data into the object
		self.data = df

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
		prior: Dict[str, Any],
		starting_points: Optional[Any] = None,
	) -> None:
		"""
		Initialize the MLP and probabilistic model.

		Parameters
		----------
		prior : dict
			Prior specification passed to Model_v0.
		starting_points : optional initial values (currently unused)
		"""
		#----------------- Initialize NNs ----------------
		self.mlp = MLP(file_mlp=self.file_mlp)
		#--------------------------------------------------

		#------------ Prior verification ------------------------
		assert "age"  in prior.keys(), KeyError('Please, provide a prior for the age')
		assert "distance" in prior.keys(), KeyError('Please, provide a prior for distance')

		assert prior["age"]["family"]  in ["TruncatedNormal","Uniform"], KeyError("Unknown family of age prior")
		assert prior["distance"]["family"] in ["Gaussian","Uniform"], KeyError("Unknown family of distance prior")
		assert prior["distance_dispersion"]["family"] in ["Exponential","Gamma"], KeyError("Unknown family of distance_dispersion prior")
		assert prior["photometric_dispersion"]["family"] in ["Exponential","Gamma"], KeyError("Unknown family of photometric_dispersion prior")
		#--------------------------------------------------------------------------------------------------------

		#--------------------- Verify that the input to the neural network conforms to its training -------------------------------
		assert prior["age"]["lower"] >= self.mlp.age_domain[0],"Error at the lower limit of the age prior! Verify mlp file"
		assert prior["age"]["upper"] <= self.mlp.age_domain[1],"Error at the upper limit of the age prior! Verify mlp file"
		#---------------------------------------------------------------------------------------------------------------------------
		
		#---------------------- Data arrangement ---------------------------------------------
		astrometry_mu = self.data.loc[:,self.astrometric_names["values"]].to_numpy()
		astrometry_sd = self.data.loc[:,self.astrometric_names["errors"]].to_numpy()
		astrometry_ix = np.where(np.isfinite(astrometry_mu))
		photometry_mu = self.data.loc[:,self.photometric_names["values"]].to_numpy()
		photometry_sd = self.data.loc[:,self.photometric_names["errors"]].to_numpy()
		photometry_ix = np.where(np.isfinite(photometry_mu))
		n_stars = photometry_mu.shape[0]
		assert n_stars > 1,"The number of photometric sources is less than two!"

		n_bands = len(self.photometric_names["values"])
		identifiers = self.data.index.values

		self.Model = Model_v0(
						mlp=self.mlp,
						prior = prior,
						identifiers = identifiers,
						astrometry_mu = astrometry_mu,
						astrometry_sd = astrometry_sd,
						astrometry_ix = astrometry_ix,
						photometry_mu = photometry_mu,
						photometry_sd = photometry_sd,
						photometry_ix = photometry_ix,
						astrometric_names = self.observables["astrometry"],
						photometric_names = self.observables["photometry"]
						)
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
		cores: int = 2,
		step: Optional[Any] = None,
		step_size: Optional[float] = None,
		init_method: str = "advi+adapt_diag",
		init_iters: int = int(1e6),
		init_absolute_tol: float = 5e-3,
		init_relative_tol: float = 1e-5,
		init_plot_iters: int = int(1e4),
		init_refine: bool = False,
		prior_predictive: bool = True,
		prior_iters: int = 2000,
		progressbar: bool = True,
		nuts_sampler: str = "numpyro",
		random_seed: Optional[int] = None) -> None:
		"""
		Performs the variational initialization (ADVI) followed by MCMC sampling.
		"""
		#------- Step_size ----------
		if step_size is None:
			step_size = 1.e-1
		#---------------------------

		# Only run sampling if posterior file does not already exist.
		if not os.path.exists(self.file_chains):
			#================== Optimization (ADVI) =============================================
			# If an initialization file exists, read it (it saves starting points).
			if os.path.exists(self.file_start):
				print("Reading initial positions ...")
				in_file = open(self.file_start, "rb")
				approx = dill.load(in_file)
				in_file.close()
				start = approx["initial_points"][0]
			else:
				approx = None
				start = None #self.starting_points
				print("Finding initial positions ...")

				# Run ADVI if there is no saved initialization or if user requested refine.
			if approx is None or (approx is not None and init_refine):

				# Prepare random seeds per chain (PyMC utility).
				random_seed_list = pm.util._get_seeds_per_chain(random_seed, chains)
				# Convergence callbacks used to stop ADVI when parameter changes are small.
				cb = [pm.callbacks.CheckParametersConvergence(
						tolerance=init_absolute_tol, diff="absolute",ord=None),
					  pm.callbacks.CheckParametersConvergence(
						tolerance=init_relative_tol, diff="relative",ord=None)]

				approx = pm.fit(
					start=start,
					random_seed=random_seed_list[0],
					n=init_iters,
					method="advi",
					model=self.Model,
					callbacks=cb,
					progressbar=True,
					#test_optimizer=pm.adagrad#_window
					)

				#------------- Plot the ADVI loss (last init_plot_iters iterations) ----------------
				plt.figure()
				plt.plot(approx.hist[-init_plot_iters:])
				plt.xlabel("Last {0} iterations".format(init_plot_iters))
				plt.ylabel("Average Loss")
				plt.savefig(self.file_advi_loss)
				plt.close()
				#-----------------------------------------------------------

				# Sample initial points from the fitted variational approximation for each chain.
				approx_sample = approx.sample(
					draws=chains, 
					random_seed=random_seed_list[0],
					return_inferencedata=False
					)

				initial_points = [approx_sample[i] for i in range(chains)]
				# Extract mean/std used to build HMC mass matrix/potential if needed.
				sd_point = approx.std.eval()
				mu_point = approx.mean.get_value()
				approx = {
					"initial_points":initial_points,
					"mu_point":mu_point,
					"sd_point":sd_point
					}

				# Save initialization to disk so future runs can reuse it.
				out_file = open(self.file_start, "wb")
				dill.dump(approx, out_file)
				out_file.close()

			#----------- Extract values needed for sampler ---------------------
			mu_point = approx["mu_point"]
			sd_point = approx["sd_point"]
			initial_points = approx["initial_points"]
			#-------------------------------------------

			#=================== Sampling ==================================================
			if nuts_sampler == "pymc":
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
					nuts_sampler_kwargs={"step_size":step_size},
					model=self.Model
					)
			else:
				#---------- Posterior sampling using default sampler backend (e.g., numpyro) -----------
				trace = pm.sample(
					draws=sample_iters,
					initvals=initial_points,
					nuts_sampler=nuts_sampler,
					tune=tuning_iters,
					chains=chains, 
					cores=cores,
					progressbar=progressbar,
					target_accept=target_accept,
					discard_tuned_samples=True,
					return_inferencedata=True,
					#nuts_sampler_kwargs={"step_size":step_size},
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
		

	def load_trace(self, file_chains: Optional[str] = None) -> None:
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
											), 
											self.ds_posterior.data_vars))
		global_variables = list(filter(lambda x: (("age" in x)
											or (x == "distance_central")
											or (x == "distance_dispersion")
											or ("astrometric_" in x) 
											or ("photometric_" in x)
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

		# Remove "true" variants from source_variables (we want the inferred quantities)
		for var in tmp_src:
			if (
				("true" in var)
				):
				source_variables.remove(var)

		# Keep only relevant source statistics variables: mass, distance, astrometry and photometry (not "true")
		for var in tmp_sts_src:
			if not (
				(var == "mass") or 
				(var == "distance") or
				("astrometry" in var) or 
				(("photometry" in var)
					and not ("true" in var))
				):
				source_sts_variables.remove(var)

		# Keep only relevant global statistics variables: age, distance central/dispersion and photometric hyperparams
		for var in tmp_sts_glb:
			if not (
				("age" in var) or 
				(var == "distance_central") or 
				(var == "distance_dispersion") or
				("photometric_" in var) 
				):
				global_sts_variables.remove(var)

		# Variables to compare prior/posterior (cpp): only keep age, distance central/dispersion and photometric_*
		for var in tmp_cpp:
			if not (("age" in var)
				or (var == "distance_central")
				or (var == "distance_dispersion")
				or ("photometric_" in var)):
				global_cpp_var.remove(var)

		# For posterior predictions keep photometry true values (predictions) and drop others
		for var in tmp_prd:
			if (
				(("photometry" in var) 
					and not ("true" in var)) 
				or ("astrometry" in var)
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

		print("Step size:")
		# The trace object has sample_stats; average over draws to report mean step_size per chain.
		for i,val in enumerate(self.trace.sample_stats["step_size"].mean(dim="draw")):
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
					# compact=compact,
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
			for var,err,unit in zip(
				self.observables[case],
				self.observables[case+"_error"],
				self.observables[case+"_units"]):
				plt.figure(0,figsize=figsize)
				ax = plt.gca()
				ax.errorbar(x=df[var], y=df["pred_"+var], 
						xerr=df[err],
						yerr=df["pred_"+err], 
						fmt='.',
						capsize=2,
						color="black", 
						ecolor='gray', 
						zorder=1)
				lim = ax.get_xlim()
				# Plot the one-to-one line for reference
				ax.plot(lim, lim,fmt_121, zorder=0)
				ax.set_xlabel("Observed {0} {1}".format(var,unit))
				ax.set_ylabel("Predicted {0} {1}".format(var,unit))
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
		theta    = np.linspace(self.mlp.theta_domain[0],self.mlp.theta_domain[1],n_points)

		dfs_smp = []
		# For each sampled age, query MLP to produce absolute photometry and convert to apparent.
		for age in ages:
			mass,absolute_photometry = self.mlp(age,theta,n_points)
			photometry = absolute_to_apparent(absolute_photometry,distance,11)
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
		df_pht = self.trace.posterior["photometry"].to_dataframe().unstack("photometric_names")
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
			cmd["color"][0],cmd["color"][1],
			self.observables["photometry_units"][0]))
		ax.set_ylabel("{0} {1}".format(cmd["magnitude"],
			self.observables["photometry_units"][0]))
		ax.invert_yaxis()  # Magnitudes increase downward in plots
		ax.set_title("Apparent photometry")
		plt.savefig(file_plot,bbox_inches='tight',dpi=dpi)
		plt.close(0)

	# The HRD plotting function is present but commented out in the original file.
	# If needed it can be uncommented and updated with the same style of comments.

	
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
			if case in ["mass","distance"]:
				df_tmp  = az.summary(self.ds_posterior,
							var_names=case,
							stat_focus = stat_focus,
							hdi_prob=hdi_prob,
							round_to=5,
							extend=True,
							kind=kind
							)
				df_tmp.set_index(self.ID,inplace=True)
				df_tmp.rename(columns=lambda x:case+"_"+x,inplace=True)
				dfs.append(df_tmp)
			elif case == "astrometry":
				for var in self.observables[case]:
					df_tmp  = az.summary(self.ds_posterior,
								var_names=case,
								coords={"astrometric_names":var},
								stat_focus = stat_focus,
								hdi_prob=hdi_prob,
								round_to=5,
								extend=True,
								kind=kind
								)
					df_tmp.set_index(self.ID,inplace=True)
					df_tmp.rename(columns=lambda x:var+"_"+x,inplace=True)
					dfs.append(df_tmp)
			elif case == "photometry":
				for var in self.observables[case]:
					df_tmp  = az.summary(self.ds_posterior,
								var_names=case,
								coords={"photometric_names":var},
								stat_focus = stat_focus,
								hdi_prob=hdi_prob,
								round_to=5,
								extend=True,
								kind=kind
								)
					df_tmp.set_index(self.ID,inplace=True)
					df_tmp.rename(columns=lambda x:var+"_"+x,inplace=True)
					dfs.append(df_tmp)
			else:
				sys.exit("Unrecognized case: {0}".format(case))

		df_source = pn.concat(dfs,axis=1,ignore_index=False)
		#---------- Save source data frame to CSV ----------------------
		df_source.to_csv(path_or_buf=file_sources,
						index_label=self.id_name)
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
	nuts_sampler = "numpyro"
	n_stars = None

	dir_data = os.getcwd() + "/data/"
	dir_out  = os.getcwd() + "/outputs/PARSEC_v0_test_8/"
	dir_mlps = os.getcwd() + "/mlps/"

	os.makedirs(dir_out,exist_ok=True)

	file_data      = dir_data + "Pleiades.csv"
	file_mlp       = dir_mlps + "PARSEC_10x96/mlp.pkl"
	file_posterior = dir_out  + "Chains.nc"
	file_prior     = dir_out  + "Prior.nc"

	# Example priors to be passed to Huehueti.setup
	priors = {
		'age' : {
			'family' : 'Uniform',
			'mu'    : 120.,
			'sigma' : 80.,
			'lower' : 20,
			'upper' : 200,
			"initval" : 120
			},
		'distance' : {
			'family' : 'Gaussian',
			'mu' : 135.,
			'sigma' : 10.,
			},
		"distance_dispersion":{
			"family": "Exponential",
			"scale" : 10.,
			"initval" : 10.
			},
		"photometric_dispersion":{
			"family": "Exponential",
			"sigma" : 0.2,
			"initval" : 0.2
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


	hue = Huehueti(dir_out = dir_out, file_mlp=file_mlp)
	hue.load_data(file_data = file_data, n_stars = n_stars)
	hue.setup(prior = priors)
	# hue.plot_pgm()
	hue.run(
		init_iters=int(1e6),
		init_refine=True,
		tuning_iters=5000,
		sample_iters=2000)
	hue.load_trace()
	hue.convergence()
	hue.plot_chains()#IDs=[69945814454871680,64979732749686016,65247704349267584])
	hue.plot_posterior()
	hue.plot_cpp()
	hue.plot_predictions()
	hue.plot_cmd()
	hue.plot_hrd()
	hue.save_statistics()
````markdown name=DEVELOPER.md
```markdown
# Huehueti — Developer Notes

This short developer guide documents the expected input CSV format, required columns, units, missing-value handling, and quick commands to run Huehueti.

## Required CSV columns (header names)
The loader expects the following columns in the input CSV (exact names):

- Identifier
  - `source_id` (string or integer) — unique ID for each source (becomes index)
- Astrometry
  - `parallax` (float) — mas
  - `parallax_error` (float) — mas
- Photometry (values)
  - `g`, `bp`, `rp`, `gmag`, `rmag`, `imag`, `ymag`, `zmag`, `Jmag`, `Hmag`, `Kmag` (magnitudes)
- Photometry (errors)
  - `g_error`, `bp_error`, `rp_error`, `e_gmag`, `e_rmag`, `e_imag`, `e_ymag`, `e_zmag`, `e_Jmag`, `e_Hmag`, `e_Kmag` (magnitudes)

These names are defined in `Huehueti.observables`. If your catalog uses different names, either rename the columns or update `Huehueti.py`'s observables mapping accordingly.

## Units
- Photometry: magnitudes (`[mag]`)
- Parallax: milliarcseconds (`[mas]`)

## Missing data / error handling
- Zero error values in error columns are treated as missing and replaced by NaN before filling.
- By default missing error values are filled with the maximum value of the error column (use `fill_nan="mean"` in `load_data` to use the mean instead).
- If a photometric value is missing (NaN), its corresponding error is set back to NaN.

## Photometric limits filtering
- The loader filters out sources that are brighter than the isochrone model limits. The function `_isochrone_photometric_limits` reads `phot_min` from the serialized MLP (`file_mlp`) and converts it to apparent magnitudes using an average distance estimate (1000 / parallax mean).

## Files and directories
- Default directories used in the example `__main__`:
  - Data: `./data/` (e.g., `data/Pleiades.csv`)
  - MLPs: `./mlps/` (e.g., `mlps/PARSEC_10x96/mlp.pkl`)
  - Outputs: `./outputs/…/`
- Key outputs created by Huehueti (in `dir_out`):
  - `Identifiers.csv`, `Observations.nc`, `Chains.nc`, `Prior.nc`
  - Plots: `Posterior.pdf`, `Comparison_prior_posterior.pdf`, `Predictions.pdf`, `Color-magnitude_diagram.png`, etc.
  - Statistics: `Sources_statistics.csv`, `Global_statistics.csv`

## Quick run (example)
1. Place your input CSV at `data/YourCatalog.csv`.
2. Ensure you have an MLP dill file (e.g. `mlps/PARSEC_10x96/mlp.pkl`) with the `phot_min` and domain attributes consumed by the code.
3. Run:
   ```bash
   python Huehueti.py