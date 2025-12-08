import sys
import os
import numpy as np
import pandas as pn
import pymc as pm
import arviz as az
import dill
import xarray
import seaborn as sns

#---------------- Matplotlib -------------------------------------
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#------------------------------------------------------------------

from Models import Model_v0,Model_v1
from MLPs import MLP_iso
from Functions import absolute_to_apparent, apparent_to_absolute

pn.set_option('display.max_columns', None)

class Huehueti:
	"""docstring for ClassName"""
	def __init__(self,
		dir_out : str,
		file_mlp : str,
		):
		self.dir_out = dir_out
		self.file_mlp = file_mlp

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

		# A list containing every filter and its uncertainty
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

		self.id_name = observables["identifiers"]
		self.observables = observables

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

		self.photometric_names = {"values":observables["photometry"],"errors":observables["photometry_error"]}
		self.astrometric_names = {"values":observables["astrometry"],"errors":observables["astrometry_error"]}

	def _isochrone_photometric_limits(self,distance):
		"""Filter out those stars having absolute magnitude lower than the model limits"""
		try:
			with open(self.file_mlp, 'rb') as file:
				mlp = dill.load(file)
				phot_min = mlp["phot_min"]
		except FileNotFoundError as error:
			raise FileNotFoundError("The isochrone model cannot be found. Please, provide a valid path.") from error
		else:
			# Get apparent magnitud and parallax from our dataset
			phot_lim = phot_min + 5.0*np.log10(distance) - 5.0
			return phot_lim

	def load_data(self, 
		file_data : str, 
		fill_nan : str = "max",
		n_stars=None
		):
		"""Read dataset file and sample nStars from the generated dataframe
		
		Parameters
		----------
		filename : str
			Dataset file name. It must be at data folder.
		nStars : int
			Number of stars to select from data. Default is None, therefore the full dataset is taken.
		sortPho : bool
			If nStars is given, sort sampled Photometry (ascending order) and take nStars (most brilliant). If False,
			just sample nStars at random.
		"""
		assert fill_nan in ["max","mean"],"Error: fill_nan can only be max or mean"

		# ----------------Read dataset --------------------
		df = pn.read_csv(file_data,usecols=self.columns)
		df.set_index("source_id",inplace=True)
		#--------------------------------------------------
		
		#-------- Filter out sources brighter than limit ------------------------------------------------
		phot_lim = self._isochrone_photometric_limits(distance=np.mean(1000/df["parallax"])).to_numpy()
		mask_valid = df.loc[:,self.observables["photometry"]].apply(
			lambda x : x >= phot_lim ,axis=1) | df.loc[:,self.observables["photometry"]].isnull()
		mask_valid = mask_valid.apply(lambda x: np.all(x),axis=1)
		df = df.loc[mask_valid]
		#-------------------------------------------------------------------------------------------------

		#-------------- Replace zeros and missing values -------------------------
		for value,error in zip(self.names_mu,self.names_sd):
			df[error] = df[error].replace(0.0,np.nan)
			replace = df[error].max() if fill_nan == "max" else df[error].mean()
			df[error] = df[error].fillna(value=replace)

			#---- if band is missing error as well -----
			df.loc[np.isnan(df[value]),error] = np.nan
			#-------------------------------------------
		#-------------------------------------------------------------------------

		print("Summary of input data:")
		print(df.describe())

		if n_stars is not None:
			print(50*">")
			print("Using only these sources:")
			df = df.iloc[-n_stars:]
			print(df.describe())
			print(50*"<")

		self.data = df

		#----- Track ID -----------------
		self.ID = self.data.index.values
		#--------------------------------

		#----- Save identifiers --------------------------
		df = pn.DataFrame(self.ID,columns=[self.id_name])
		df.to_csv(path_or_buf=self.file_ids,index=False)
		#------------------------------------------------

		#-------- Observations to InferenceData ---------------
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
		prior : dict,
		starting_points = None,
		):
		"""
		Fill sd NaNs with zeros to avoid error while sampling
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

	def plot_pgm(self,file=None):
		file = file if file is not None else self.dir_out+"model_graph.png"
		graph = pm.model_to_graphviz(self.Model)
		graph.render(outfile=file,format="png")

	def run(self,
		tuning_iters : int = 3000,
		sample_iters : int = 2000,
		target_accept : float = 0.65,
		chains : int = 2,
		cores : int = 2,
		step=None,
		step_size=None,
		init_method : str = "advi+adapt_diag",
		init_iters : int = int(1e6),
		init_absolute_tol : float = 5e-3,
		init_relative_tol : float = 1e-5,
		init_plot_iters : int = int(1e4),
		init_refine : bool = False,
		prior_predictive : bool = True,
		prior_iters : int = 2000,
		progressbar : bool = True,
		nuts_sampler : str = "numpyro",
		random_seed=None):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		tuning_iters (integer):    Number of burning iterations.
		"""

		#------- Step_size ----------
		if step_size is None:
			step_size = 1.e-1
		#---------------------------

		if not os.path.exists(self.file_chains):
			#================== Optimization =============================================
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

			if approx is None or (approx is not None and init_refine):

				random_seed_list = pm.util._get_seeds_per_chain(random_seed, chains)
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

				#------------- Plot Loss ----------------------------------
				plt.figure()
				plt.plot(approx.hist[-init_plot_iters:])
				plt.xlabel("Last {0} iterations".format(init_plot_iters))
				plt.ylabel("Average Loss")
				plt.savefig(self.file_advi_loss)
				plt.close()
				#-----------------------------------------------------------

				approx_sample = approx.sample(
					draws=chains, 
					random_seed=random_seed_list[0],
					return_inferencedata=False
					)

				initial_points = [approx_sample[i] for i in range(chains)]
				sd_point = approx.std.eval()
				mu_point = approx.mean.get_value()
				approx = {
					"initial_points":initial_points,
					"mu_point":mu_point,
					"sd_point":sd_point
					}

				out_file = open(self.file_start, "wb")
				dill.dump(approx, out_file)
				out_file.close()

				# #------------------ Save initial point ------------------------------
				# df = pn.DataFrame(data=initial_points[0]["{0}D::true".format(self.D)],
				# 	columns=self.names_mu)
				# df.to_csv(self.file_init_true,index=False)
				# df = pn.DataFrame(data=initial_points[0]["{0}D::source".format(self.D)],
				# 	columns=self.names_coords)
				# df.to_csv(self.dir_out+"/initial_source.csv",index=False)
				# #---------------------------------------------------------------------

			#----------- Extract ---------------------
			mu_point = approx["mu_point"]
			sd_point = approx["sd_point"]
			initial_points = approx["initial_points"]
			#-------------------------------------------

			#================================================================================

			#=================== Sampling ==================================================
			if nuts_sampler == "pymc":
				#--------------- Prepare step ---------------------------------------------
				# Only valid for nuts_sampler == "pymc". 
				# The other samplers adapt steps independently.
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
				#----------------------------------------------------------------------------

				print("Sampling the model ...")

				#---------- Posterior -----------
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
				#--------------------------------
			else:
				#---------- Posterior -----------
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
				#--------------------------------

			#--------- Save with arviz ------------
			print("Saving posterior samples ...")
			az.to_netcdf(trace,self.file_chains)
			#-------------------------------------
			del trace
			#================================================================================

		
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
		


	def load_trace(self,file_chains=None):
		'''
		Loads a previously saved sampling of the model
		'''

		file_chains = self.file_chains if (file_chains is None) else file_chains

		if not hasattr(self,"ID"):
			#----- Load identifiers ------
			self.ID = pn.read_csv(self.file_ids).to_numpy().flatten()

		print("Loading existing samples ... ")

		#---------Load posterior ---------------------------------------------------
		try:
			posterior = az.from_netcdf(file_chains)
		except ValueError:
			sys.exit("ERROR at loading {0}".format(file_chains))
		#------------------------------------------------------------------------

		#----------- Load prior -------------------------------------------------
		try:
			prior = az.from_netcdf(self.file_prior)
		except:
			prior = None
			self.ds_prior = None
		
		if prior is not None:
			posterior.extend(prior)
			self.ds_prior = posterior.prior
		#-------------------------------------------------------------------------

		self.trace = posterior

		#---------Load posterior ---------------------------------------------------
		try:
			self.ds_posterior = self.trace.posterior
		except ValueError:
			sys.exit("There is no posterior in trace")
		#------------------------------------------------------------------------

		#------- Variable names -----------------------------------------------------------
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
	
		trace_variables = global_variables.copy()
		source_sts_variables = source_variables.copy()
		global_sts_variables = global_variables.copy()
		global_cpp_var = global_variables.copy()
		source_prd_var = source_variables.copy()

		#----------- Case specific variables -------------
		tmp_src     = source_variables.copy()
		tmp_sts_src = source_variables.copy()
		tmp_sts_glb = global_variables.copy()
		tmp_cpp   = global_variables.copy()
		tmp_prd   = source_variables.copy()

		for var in tmp_src:
			if (
				("true" in var)
				):
				source_variables.remove(var)

		for var in tmp_sts_src:
			if not (
				(var == "mass") or 
				(var == "distance") or
				("astrometry" in var) or 
				(("photometry" in var)
					and not ("true" in var))
				):
				source_sts_variables.remove(var)

		for var in tmp_sts_glb:
			if not (
				("age" in var) or 
				(var == "distance_central") or 
				(var == "distance_dispersion") or
				("photometric_" in var) 
				):
				global_sts_variables.remove(var)

		for var in tmp_cpp:
			if not (("age" in var)
				or (var == "distance_central")
				or (var == "distance_dispersion")
				or ("photometric_" in var)):
				global_cpp_var.remove(var)

		for var in tmp_prd:
			if (
				(("photometry" in var) 
					and not ("true" in var)) 
				or ("astrometry" in var)
				):
				pass
			else:
				source_prd_var.remove(var)

		#----------------------------------------------------

		self.source_variables  = source_variables
		self.global_variables  = global_variables
		self.trace_variables   = trace_variables
		self.cpp_variables     = global_cpp_var
		self.prd_variables     = source_prd_var
		self.global_sts_variables  = global_sts_variables
		self.source_sts_variables  = source_sts_variables

		# print("source_variables: ",self.source_variables)
		# print("global_variables: ",self.global_variables)
		# print("trace_variables: ",self.trace_variables)
		# print("cpp_variables: ",self.cpp_variables)
		# print("prd_variables: ",self.prd_variables)
		# print("global_sts_variables",self.global_sts_variables)
		# print("source_sts_variables",self.source_sts_variables)
		# sys.exit()

	def convergence(self):
		"""
		Analyse the chains.		
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
		for i,val in enumerate(self.trace.sample_stats["step_size"].mean(dim="draw")):
			print("Chain {0}: {1:3.8f}".format(i,val))

	def plot_chains(self,
		file_trace_sources=None,
		file_trace_globals=None,
		IDs=None,
		divergences='bottom', 
		figsize=None, 
		lines=None, 
		combined=False,
		compact=False,
		plot_kwargs=None, 
		hist_kwargs=None, 
		trace_kwargs=None,
		fontsize_title=16):
		"""
		This function plots the trace. Parameters are the same as in pymc3
		"""
		if IDs is None and len(self.global_variables) == 0:
			return

		print("Plotting traces ...")

		file_trace_globals = file_trace_globals if (file_trace_globals is not None) else self.file_trace_globals
		file_trace_sources = file_trace_sources if (file_trace_sources is not None) else self.file_trace_sources

		if IDs is not None:
			pdf = PdfPages(filename=file_trace_sources)
			#--------- Loop over ID in list ---------------
			for i,ID in enumerate(IDs):
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

				for ax in axes:
					# --- Set units in parameters ------------------------------
					title = ax[0].get_title()
					if "mass" in title:
						ax[0].set_xlabel("$M_\\odot$")
					if "distance" in title:
						ax[0].set_xlabel("pc")
					#-----------------------------------------------------------
					ax[1].set_xlabel("Iterations")
					# ax[0].set_title(None)
					# ax[1].set_title(None)
				
				plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,hspace=0.5,wspace=0.1)
				plt.gcf().suptitle(self.id_name +" "+str(ID),fontsize=fontsize_title)

					
				#-------------- Save fig --------------------------
				pdf.savefig(bbox_inches='tight')
				plt.close(0)
			pdf.close()

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
				# --- Set units in parameters ------------------------------
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
		file_cpp=None,
		figsize=None,
		):
		"""
		This function plots the prior and posterior distributions.
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
		file_posterior=None,
		figsize=None,
		):
		"""
		This function plots the prior and posterior distributions.
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
		file_plots : str = None,
		figsize : str = None,
		fmt_121 : str = "k:"
		):
		"""
		This function plots the posterior predictive distributions.
		"""

		print("Plotting posterior predictive distributions ...")
		file_plots = file_plots if file_plots is not None else self.file_prd

		pdf = PdfPages(filename=file_plots)

		for case in self.prd_variables:
			columns = sum([self.observables[case],self.observables[case+"_error"]],[])
			mapper = {key: value for key,value in zip(
							self.observables[case],
							self.observables[case+"_error"])}
			df_obs = self.data.loc[:,columns].copy()

			df_prd = self.trace.posterior[case].to_dataframe().unstack()
			df_prd.columns = df_prd.columns.droplevel(level=0)

			df_prd_mu = df_prd.groupby("source_id").mean()
			df_prd_sd = df_prd.groupby("source_id").std()
			df_prd_sd.rename(columns=mapper,inplace=True)
			df_prd = df_prd_mu.join(df_prd_sd)
			df_prd.rename(columns=lambda x: "pred_"+str(x),inplace=True)

			df = df_obs.join(df_prd)

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
				ax.plot(lim, lim,fmt_121, zorder=0)
				ax.set_xlabel("Observed {0} {1}".format(var,unit))
				ax.set_ylabel("Predicted {0} {1}".format(var,unit))
				pdf.savefig(bbox_inches='tight')
				plt.close(0)
		pdf.close()



	def plot_cmd(self,
		file_plot=None,
		figsize=None,
		cmd : dict = {"magnitude":"g","color":["g","rp"]},
		n_samples : int = 10,
		n_points : int = 100,
		scatter_palette = "dark",
		lines_color = "orange",
		alpha=1.0,
		dpi=600,
		):
		"""
		This function plots the model.
		"""
		print("Plotting CMD ...")

		msg_n = "The required n_samples {0} is larger than those in the posterior.".format(n_samples)

		assert n_samples <= self.ds_posterior.sizes["draw"], msg_n

		#----- Samples from the posterior distribution -----------------------------------
		ages = np.random.choice(self.trace.posterior["age"].values.flatten(),
					size=n_samples,replace=False)
		distance = np.mean(self.trace.posterior["distance"].values.flatten())
		theta    = np.linspace(self.mlp.theta_domain[0],self.mlp.theta_domain[1],n_points)

		dfs_smp = []
		for age in ages:
			mass,absolute_photometry = self.mlp(age,theta,n_points)
			photometry = absolute_to_apparent(absolute_photometry,distance,11)
			df_tmp = pn.DataFrame(
					data=photometry.eval(),
					columns=self.observables["photometry"])
			df_tmp.index.name = "star"
			df_tmp["age"] = age
			df_tmp.set_index("age",append=True,inplace=True)
			dfs_smp.append(df_tmp)
		df_smp = pn.concat(dfs_smp,ignore_index=False)
		#------------------------------------------------------------------------------

		#-------------- Photometric data -------------------------------------------------------
		df_pht = self.trace.posterior["photometry"].to_dataframe().unstack("photometric_names")
		df_pht.columns = df_pht.columns.droplevel(level=0)
		df_pos = df_pht.groupby("source_id").mean()
		#----------------------------------------------------------------------------------------

		#---------------- Color and magnitude  ------------------------
		columns = list(set(sum([[cmd["magnitude"]],cmd["color"]],[])))
		#--------------------------------------------------------------

		#---------- Dataframes ------------------
		df_obs = self.data.loc[:,columns].copy()
		df_prd = df_pos.loc[:,columns].copy()
		df_smp = df_smp.loc[:,columns].copy()
		#-----------------------------------------

		#------------------- Join -----------------
		df_obs["Origin"] = "Observed"
		df_prd["Origin"] = "Predicted"
		df_all = pn.concat([df_obs,df_prd],
					ignore_index=True) #Otherwise scatterplot wont work on duplicated index labels
		#------------------------------------------


		#------------------------ Color ------------------------
		df_all["color"] = df_all.apply(lambda x: 
						x[cmd["color"][0]] - 
						x[cmd["color"][1]],axis=1)
		df_smp["color"] = df_smp.apply(lambda x: 
						x[cmd["color"][0]] - 
						x[cmd["color"][1]],axis=1)
		#---------------------------------------------------------

		file_plot = file_plot if (file_plot is not None) else self.file_cmd
		#----------------- CMD --------------------------------------------
		plt.figure(0,figsize=figsize)
		ax = sns.scatterplot(data=df_all,
						x="color",
						y=cmd["magnitude"],
						palette=sns.color_palette(scatter_palette,n_colors=2),
						hue="Origin",
						style="Origin",
						s=10,
						zorder=0)
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
		ax.invert_yaxis()
		ax.set_title("Apparent photometry")
		plt.savefig(file_plot,bbox_inches='tight',dpi=dpi)
		plt.close(0)

	# def plot_hrd(self,
	# 	file_plot=None,
	# 	figsize=None,
	# 	magnitude: dict = {"G":"g"},
	# 	n_samples : int = 10,
	# 	n_points : int = 100,
	# 	scatter_palette = "dark",
	# 	lines_color = "orange",
	# 	alpha=1.0,
	# 	dpi=600,
	# 	):
	# 	"""
	# 	This function plots the model.
	# 	"""
	# 	print("Plotting HRD ...")
	# 	abs_mag = list(magnitude.keys())[0]
	# 	app_mag = list(magnitude.values())[0]

	# 	msg_n = "The required n_samples {0} is larger than those in the posterior.".format(n_samples)

	# 	assert n_samples <= self.ds_posterior.sizes["draw"], msg_n

	# 	#----- Samples from the posterior distribution -----------------------------------
	# 	ages = np.random.choice(self.trace.posterior["age"].values.flatten(),
	# 				size=n_samples,replace=False)
	# 	mass = np.linspace(self.mlp.mass_domain[0],self.mlp.mass_domain[1],n_points)
	# 	teff = np.linspace(self.mlp.teff_domain[0],self.mlp.teff_domain[1],n_points)

	# 	dfs_smp = []
	# 	for age in ages:
	# 		df_tmp = pn.merge(
	# 				left=pn.DataFrame(data={"age":age,"teff":teff,"mass":mass}),
	# 				right=pn.DataFrame(
	# 				data=self.mlp(age,mass,teff,10.0,n_points).eval(),
	# 				columns=self.observables["absolute"]),
	# 				left_index=True,right_index=True)
	# 		df_tmp.index.name = "source_id"
	# 		df_tmp.reset_index(inplace=True)
	# 		df_tmp.set_index(["age","source_id"],inplace=True)
	# 		df_tmp = df_tmp.loc[:,["teff",abs_mag]]
	# 		dfs_smp.append(df_tmp)
	# 	df_smp = pn.concat(dfs_smp,ignore_index=False)
	# 	#------------------------------------------------------------------------------

	# 	#-------------- Preparing Data -------------------------------------------------------
	# 	df_pht = self.trace.posterior["photometry"].to_dataframe().unstack("photometric_names")
	# 	df_pht.columns = df_pht.columns.droplevel(level=0)
	# 	df_dst = self.trace.posterior["distance"].to_dataframe()
	# 	df_tff = self.trace.posterior["teff"].to_dataframe()
	# 	df_tmp = pn.concat([df_tff,df_dst,df_pht],axis=1,join="inner")
	# 	df_tmp[abs_mag] = df_tmp.apply(
	# 		lambda x: apparent_to_absolute(x[app_mag],x["distance"]),axis=1)	
	# 	df_prd = pn.merge(
	# 				left=df_tmp.groupby("source_id").mean(),
	# 				right=df_tmp.groupby("source_id").std(),
	# 				left_index=True,right_index=True,
	# 				suffixes=["","_sd"])
	# 	df_prd = df_prd.loc[:,["teff",abs_mag]]
	# 	#----------------------------------------------------------------------------------------

	# 	#---------------------- Plot ---------------------------------------
	# 	file_plot = file_plot if (file_plot is not None) else self.file_hrd
	# 	plt.figure(0,figsize=figsize)
	# 	ax = sns.scatterplot(data=df_prd,
	# 					x="teff",
	# 					y=abs_mag,
	# 					s=10,
	# 					zorder=0)
	# 	sns.lineplot(data=df_smp,
	# 					x="teff",
	# 					y=abs_mag,
	# 					palette=sns.color_palette([lines_color], n_samples),
	# 					hue="age",
	# 					legend=False,
	# 					alpha=alpha,
	# 					zorder=1,
	# 					ax=ax)
	# 	ax.set_xlabel("Teff [K]")
	# 	ax.set_ylabel("{0} [mag]".format(abs_mag))
	# 	ax.invert_yaxis()
	# 	ax.invert_xaxis()
	# 	ax.set_title("Hertzprung-Russell diagram")
	# 	plt.savefig(file_plot,bbox_inches='tight',dpi=dpi)
	# 	plt.close(0)
	# 	#------------------------------------------------------------------

	
	def save_statistics(self,
		file_globals=None,
		file_sources=None,
		hdi_prob=0.95,
		stat_focus="mean",
		kind="stats"):
		'''
		Saves the statistics to a csv file.
		Arguments:
		
		'''
		file_sources = file_sources if file_sources is not None else self.file_sts_src
		file_globals = file_globals if file_globals is not None else self.file_sts_glb

		print("Computing statistics ...")

		#-------------- Source statistics ----------------------------
		dfs = []
		for case in self.source_sts_variables:
			if case in ["mass","distance"]:
				# for var in self.observables[case]:
				df_tmp  = az.summary(self.ds_posterior,
							var_names=case,
							# coords={"astrometric_names":var},
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
		#--------------------------------------------------------------

		#---------- Save source data frame ----------------------
		df_source.to_csv(path_or_buf=file_sources,
						index_label=self.id_name)
		#-------------------------------------------------

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