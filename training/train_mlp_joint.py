import sys
import os
import dill
import random
import numpy as np
import optuna
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from mlp_model import create_custom_model, compile_model, evaluate_gradient,learning_rate_scheduler
from mlp_model import analyze_residuals

os.environ["PYTHONHASHSEED"] = "42"
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# age_range = "1-21Myr"
# age_range = "11-21Myr"
# age_range = "20-220Myr"
age_range = "200-600Myr"
# age_range = "600-1000Myr"

# age_step = "0.025myr"
# age_step = "0.05myr"
# age_step = "0.1myr"
age_step = "0.5myr"
# age_step = "1myr"


#------------- Input data ---------------------------
max_label = 1 # label >1 are evolved stars that we do not need
features = ["logAge","logL"]
targets = ["G_BPmag","Gmag","G_RPmag"]
n_features = len(features)
n_targets = len(targets)
#----------------------------------------------------

# --------------- Model properties --------------------------------
list_of_num_layers = [3,4,5]# Number of hidden layers
seeds = [0] # Seeds for the MLP initializers
activation_layers = "sigmoid" # Activation functions for each hidden layer
activation_output = "linear"  # Activation function for the output layer
loss_function = "mae"
metric = "root_mean_squared_error"
#--------------------------------------------------------------------------

#--------- Fixed Hyperparameters ------------------------------------
optimization_trials = 50
epochs = int(5e2)
lr_decay_function = "InverseTimeDecay"			
lr_decay_steps = int(1e3)
# batch_size = 90971
# lr_initial = 9.0e-2
# lr_end   = 1e-4
# lr_decay_rate = 9.5e1
# lr_boundaries = [50,100,500,1000,2000,2500]
# lr_values = [1e-1,1e-2,1e-3,7e-4,5e-4,3e-4,1e-4]
# lr_power = 1e-1 # Only if decay function is PolynomialDecay
beta_1 = 0.90  # Adam optimizer beta1 default 0.90
beta_2 = 0.999 # Adam optimizer beta2 default 0.999
clipnorm = 1.0 # The norm of the gradients does not goes larger than this value
validation_split = 0.2 # 20% of dataset used for validation
seed_split = 0
verbose = 0
critical_weight = 1.0
logL_crw_lower = 0.2
logL_crw_upper = 1.2
#------------------------------------------------------------------

#--------------- Directories and files ---------------------------------------------
dir_base  = "/home/jolivares/Models/PARSEC/{0}/".format(age_range)
# Remove the # from the row contain the header in the input file
file_iso  = dir_base + "Gaia_EDR3_{0}.dat".format(age_step) # Input file
dir_mlps  = dir_base + "Optuna_{0}_epochs_{1:1.0e}_trials_{2}_{3}/".format(
lr_decay_function,epochs,optimization_trials,age_step)
file_mtrs  = dir_mlps + "Metrics.png"
file_grds  = dir_mlps + "Gradients.png"
base_fld  = "l{0}"+ "/" #"_logL<4.3_wgt_{0}_{1}-{2}/".format(critical_weight,logL_crw_lower,logL_crw_upper)
base_sed  = "seed_{0}/"
base_dat  = "{0}data.csv"
base_fit  = "{0}fit.csv"
base_grd  = "{0}gradients.csv"
base_opt  = "{0}optuna_study_with_{1}_trials.pkl"
base_mtr  = "{0}metric.csv"
base_mlp  = "{0}mlp.pkl"
base_plt_opt  = "{0}study.png"
base_plt_prm  = "{0}study_params.png"
base_plt_lss  = "{0}loss.png"
base_plt_mtr  = "{0}metric.png"
base_plt_grd  = "{0}gradients.png"
base_res      = "{0}residuals.csv"
base_plt_res  = "{0}residuals.png"
base_plt_res2d = "{0}residuals2d.png"
#------------------------------------------------------------------------------------
#----------------- Domain of tuned parameters -------------------------
dict_btsz   = {}
dict_lysz   = {}
dict_lr_dcr = {}
dict_lr_itl = {}

match age_range:
	case "1-21Myr":
		#------------------ 3 layers -------------------------
		dict_btsz[3]   = {"value":None,"low":1,"high":100}
		dict_lysz[3]   = {"value":None,"low":50,"high":150}
		dict_lr_dcr[3] = {"value":None,"low":1e-3,"high":5e-2}
		dict_lr_itl[3] = {"value":None,"low":1e-3,"high":4e-2}
		#-----------------------------------------------------

		#------------------ 4 layers -------------------------
		dict_btsz[4]   = {"value":None,"low":1,"high":150}
		dict_lysz[4]   = {"value":None,"low":10,"high":200}
		dict_lr_dcr[4] = {"value":None,"low":1e-3,"high":1e-1}
		dict_lr_itl[4] = {"value":None,"low":1e-3,"high":5e-2}
		#-----------------------------------------------------

		#------------------ 5 layers -------------------------
		dict_btsz[5]   = {"value":None,"low":80,"high":200}
		dict_lysz[5]   = {"value":None,"low":100,"high":300}
		dict_lr_dcr[5] = {"value":None,"low":1e-3,"high":3e-1}
		dict_lr_itl[5] = {"value":None,"low":1e-3,"high":2e-2}
		#-----------------------------------------------------

	case "11-21Myr":
		#------------------ 3 layers -------------------------
		dict_btsz[3]   = {"value":None,"low":1,"high":50}
		dict_lysz[3]   = {"value":None,"low":100,"high":300}
		dict_lr_dcr[3] = {"value":None,"low":3e-2,"high":8e-2}
		dict_lr_itl[3] = {"value":None,"low":1e-3,"high":2e-2}
		#-----------------------------------------------------

		#------------------ 4 layers -------------------------
		dict_btsz[4]   = {"value":None,"low":1,"high":50}
		dict_lysz[4]   = {"value":None,"low":100,"high":300}
		dict_lr_dcr[4] = {"value":None,"low":1e-3,"high":1e-1}
		dict_lr_itl[4] = {"value":None,"low":1e-3,"high":3e-2}
		#-----------------------------------------------------

		#------------------ 5 layers -------------------------
		dict_btsz[5]   = {"value":None,"low":1,"high":50}
		dict_lysz[5]   = {"value":None,"low":100,"high":200}
		dict_lr_dcr[5] = {"value":None,"low":5e-2,"high":15e-2}
		dict_lr_itl[5] = {"value":None,"low":1e-3,"high":2e-2}
		#-----------------------------------------------------

		#------------------ 6 layers -------------------------
		dict_btsz[6]   = {"value":None,"low":1,"high":100}
		dict_lysz[6]   = {"value":None,"low":100,"high":300}
		dict_lr_dcr[6] = {"value":None,"low":5e-2,"high":2e-1}
		dict_lr_itl[6] = {"value":None,"low":1e-4,"high":1e-2}
		#-----------------------------------------------------

	case "20-220Myr":
		#------------------ 3 layers -------------------------
		dict_btsz[3]   = {"value":None,"low":10,"high":150}
		dict_lysz[3]   = {"value":None,"low":10,"high":300}
		dict_lr_dcr[3] = {"value":None,"low":1e-3,"high":2e-1}
		dict_lr_itl[3] = {"value":None,"low":1e-3,"high":5e-2}
		#-----------------------------------------------------

		#------------------ 4 layers -------------------------
		dict_btsz[4]   = {"value":None,"low":10,"high":200}
		dict_lysz[4]   = {"value":None,"low":100,"high":300}
		dict_lr_dcr[4] = {"value":None,"low":1e-3,"high":4e-1}
		dict_lr_itl[4] = {"value":None,"low":1e-3,"high":3e-2}
		#-----------------------------------------------------

		#------------------ 5 layers -------------------------
		dict_btsz[5]   = {"value":None,"low":10,"high":150}
		dict_lysz[5]   = {"value":None,"low":100,"high":250}
		dict_lr_dcr[5] = {"value":None,"low":1e-3,"high":5e-1}
		dict_lr_itl[5] = {"value":None,"low":1e-3,"high":3e-2}
		#-----------------------------------------------------

	case "200-600Myr":
		#------------------ 3 layers -------------------------
		dict_btsz[3]   = {"value":None,"low":2,"high":100}
		dict_lysz[3]   = {"value":None,"low":10,"high":200}
		dict_lr_dcr[3] = {"value":None,"low":1e-1,"high":4e-1}
		dict_lr_itl[3] = {"value":None,"low":5e-3,"high":3e-2}
		#-----------------------------------------------------

		#------------------ 4 layers -------------------------
		dict_btsz[4]   = {"value":None,"low":10,"high":100}
		dict_lysz[4]   = {"value":None,"low":10,"high":200}
		dict_lr_dcr[4] = {"value":None,"low":1e-2,"high":3e-1}
		dict_lr_itl[4] = {"value":None,"low":1e-3,"high":2e-2}
		#-----------------------------------------------------

		#------------------ 5 layers -------------------------
		dict_btsz[5]   = {"value":None,"low":10,"high":100}
		dict_lysz[5]   = {"value":None,"low":10,"high":200}
		dict_lr_dcr[5] = {"value":None,"low":1e-2,"high":7e-1}
		dict_lr_itl[5] = {"value":None,"low":1e-3,"high":2e-2}
		#-----------------------------------------------------
#-----------------------------------------------------------------------------

os.makedirs(dir_mlps,exist_ok=True)

#------------- Load data ----------------------------
df_iso = pd.read_csv(file_iso,
					# skiprows=13,
					delimiter=r"\s+",
					header="infer",
					comment="#")
df_iso = df_iso.loc[df_iso["label"]<= max_label]
df_iso = df_iso.loc[:,sum([features,targets],[])]
if age_range == "11-21Myr":
	df_iso = df_iso.loc[df_iso["logL"] < 4.3]
print(df_iso.describe())
df_idx = df_iso.copy()
df_idx.set_index(features,inplace=True)
#--------------------------------------------------

#----------------------- Domains ------------------------------------
phot_min = df_iso[targets].min()
domain = {}
for feature in features:
	domain[feature] = [df_iso[feature].min(),df_iso[feature].max()]
#--------------------------------------------------------------------

#------------- Fit logL limits as function of logAge ------------------
x = []
y_max = []
y_min = []
for log_age,tmp in df_iso.groupby("logAge").__iter__():
	x.append(log_age)
	y_max.append(tmp["logL"].max())
	y_min.append(tmp["logL"].min())

x = np.array(x)
y_max = np.array(y_max)
y_min = np.array(y_min)

def linear(x,a,b):
	return a*x + b
logL_upper_par,_ = curve_fit(linear,xdata=x,ydata=y_max)
logL_lower_par,_ = curve_fit(linear,xdata=x,ydata=y_min)

# print(logL_lower_par)
# print(logL_upper_par)
# print(linear(8.5,*logL_lower_par),linear(8.5,*logL_upper_par))

# plt.scatter(x,y_min,s=2,c="black")
# plt.scatter(x,y_max,s=2,c="black")
# plt.plot(x,linear(x,*logL_lower_par),c="red",lw=2)
# plt.plot(x,linear(x,*logL_upper_par),c="red",lw=2)
# plt.show()
# sys.exit()
#-----------------------------------------------------------------

#------------------- Sample weight ---------------------------------
sample_weight = np.ones(len(df_iso))

critical = (
    (df_iso["logL"] > logL_crw_lower) | (df_iso["logL"] < logL_crw_upper)
)

sample_weight[critical] = critical_weight
#---------------------------------------------------------------------------------------

#------- Transformations standardize inputs and outputs --------------------
def forward_transform(df_ori,mu,sd,features):
	df_trn = df_ori.copy()
	# for col in df_trn.columns:
	for col in features:
		df_trn[col] = (df_trn[col] - iso_mu[col])/iso_sd[col]

	return df_trn

def backward_transform(df_trn,mu,sd,features):
	df_ori = df_trn.copy()
	# for col in df_ori.columns:
	for col in features:
		df_ori[col] = (df_ori[col]*iso_sd[col]) + iso_mu[col]
	
	return df_ori

iso_mu = df_iso.mean(axis=0)
iso_sd = df_iso.std(axis=0)
df_trn = forward_transform(df_iso,iso_mu,iso_sd,features=features)
# df_new = backward_transform(df_trn,iso_mu,iso_sd,features=features)
# pd.testing.assert_frame_equal(df_iso,df_new)
#-------------------------------------------------------------------

#-------------- Split dataset --------------------------------------
x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
							df_trn.loc[:,features],
							df_trn.loc[:,targets],
							sample_weight,
							test_size=validation_split,
							random_state=seed_split
							)
#-------------------------------------------------------------------

#--------------------------------------------------------------------------------------
mtrs = []
for num_layers in list_of_num_layers:
	print("Working on NN with {0} layers".format(num_layers))
	dir_case   = dir_mlps  + base_fld.format(num_layers)
	os.makedirs(dir_case,exist_ok=True)
	df_trn.to_csv(base_dat.format(dir_case))
	print(dir_case)

	#----- Extract layer specific ranges ------------
	tmp_lr_itl = dict_lr_itl[num_layers]
	tmp_lr_dcr = dict_lr_dcr[num_layers]
	tmp_btsz   = dict_btsz[num_layers]
	tmp_lysz  = dict_lysz[num_layers]

	#========================= Optuna ============================================
	if not os.path.exists(base_opt.format(dir_case,optimization_trials)):

		#------------------objective function -----------------------------------------
		def objective(trial):
			if tmp_lr_itl["value"] is None:
				lr_initial = trial.suggest_float("lr_initial",
					low=tmp_lr_itl["low"],
					high=tmp_lr_itl["high"],
					log=False)
			else:
				lr_initial = tmp_lr_itl["value"]

			if tmp_lr_dcr["value"] is None:
				lr_decay_rate = trial.suggest_float("lr_decay_rate",
					low=tmp_lr_dcr["low"],
					high=tmp_lr_dcr["high"],
					log=False)
			else:
				lr_decay_rate = tmp_lr_dcr["value"]

			if tmp_btsz["value"] is None:
				batch_size = trial.suggest_int(
					name="batch_size",
					low=tmp_btsz["low"],
					high=tmp_btsz["high"]
					)
			else:
				batch_size = tmp_btsz["value"]

			if tmp_lysz["value"] is None:
				layer_size = trial.suggest_int(
					name="layer_size",
					low=tmp_lysz["low"],
					high=tmp_lysz["high"],
					)
			else:
				layer_size = tmp_lysz["value"]

			#--------------- Instantiate model -----------------------
			model = create_custom_model(
					input_shape=n_features,
					output_shape=n_targets,  
					num_layers=num_layers,
					size_layers=layer_size,
					activation_layers=activation_layers, 
					activation_output=activation_output,
					seed=seeds[0])
			#------------------------------------------------------------

			#--------------- Learning rate Scheduler --------------------------
			lr_schedule = learning_rate_scheduler(
						lr_decay_function=lr_decay_function,
						initial_learning_rate=lr_initial,
						decay_steps=lr_decay_steps,
						decay_rate = lr_decay_rate,
						# alpha=lr_alpha,
						# end_learning_rate=lr_final,
						# power=lr_power,
						# boundaries=lr_boundaries,
						# values=lr_values
						)
			#---------------------------------------------------------------

			#-------------- Compile model ------------------------------------------
			compiled_model = compile_model(model=model,
						lr_schedule=lr_schedule,
						beta_1=beta_1,
						beta_2=beta_2,
						loss=loss_function,
						metrics=[metric],
						clipnorm=clipnorm
						)
			#----------------------------------------------------------------------

			#------------ Fit ------------------------------
			fit = compiled_model.fit(
						x=x_train.to_numpy(),
						y=y_train.to_numpy(),
						validation_data=(x_val,y_val),
						sample_weight=w_train,
						epochs=epochs,
						batch_size=batch_size,
						verbose=verbose,
						# callbacks=[early_stopping]
						)
			#----------------------------------------------

			return fit.history["val_loss"][-1]

		# ---- Run optimization ----
		study = optuna.create_study(direction="minimize")
		study.optimize(objective, n_trials=optimization_trials)

		with open(base_opt.format(dir_case,optimization_trials), "wb") as file:
			dill.dump(study, file)

		#-------------- Corner plot --------------------------------
		df = study.trials_dataframe(multi_index=False)
		df.drop(columns=["number"],inplace=True)
		fg = sns.PairGrid(df, hue="value",corner=True)
		fg.map_lower(sns.scatterplot)
		fg.map_diag(sns.histplot,hue=None)

		# Best trial parameters
		best_params = study.best_trial.params

		# Variables plotted (same order as PairGrid)
		vars_ = fg.x_vars

		# Overlay best trial
		for i, yvar in enumerate(vars_):
			for j, xvar in enumerate(vars_):
				ax = fg.axes[i, j]
				if ax is None:
					continue

				# Diagonal: histogram
				if i == j:
					if xvar.startswith("params_"):
						pname = xvar.replace("params_", "")
						if pname in best_params:
							ax.axvline(
								best_params[pname],
								color="red",
								linestyle="--",
								linewidth=2,
								zorder=10,
							)

				# Lower triangle: scatter
				elif i > j:
					xname = xvar.replace("params_", "")
					yname = yvar.replace("params_", "")

					if xname in best_params and yname in best_params:
						ax.scatter(
							best_params[xname],
							best_params[yname],
							color="red",
							s=100,
							marker="o",
							edgecolor="black",
							linewidth=1,
							zorder=10,
							label="Best trial" if (i == 1 and j == 0) else None,
						)

		fg.add_legend()
		fg.savefig(base_plt_prm.format(dir_case))
		plt.close()
		#-----------------------------------------------------------
	else:
		with open(base_opt.format(dir_case,optimization_trials), "rb") as file:
			study = dill.load(file)

	# ---- Best results ----
	best_trial = study.best_trial
	print("Best trial:")
	print("  Value:", best_trial.value)
	print("  Params:")
	for key, value in best_trial.params.items():
		print(f"    {key}: {value}")

	if "lr_initial" in best_trial.params:
		lr_initial = best_trial.params["lr_initial"]
	else:
		lr_initial = tmp_lr_itl["value"]

	if "lr_decay_rate"in best_trial.params:
		lr_decay_rate = best_trial.params["lr_decay_rate"]
	else:
		lr_decay_rate = tmp_lr_dcr["value"]

	if "batch_size"in best_trial.params:
		batch_size = best_trial.params["batch_size"]
	else:
		batch_size = tmp_btsz["value"]

	if "layer_size"in best_trial.params:
		layer_size = best_trial.params["layer_size"]
	else:
		layer_size = tmp_lysz["value"]


	print("lr_initial: {0}".format(lr_initial))
	print("lr_decay_rate: {0}".format(lr_decay_rate))
	print("batch_size: {0}".format(batch_size))
	print("layer_size: {0}".format(layer_size))
	#================================================================

	#================= Loop over seeds =================================================
	fits = []
	grds = []
	for seed in seeds:
		dir_seed = dir_case + base_sed.format(seed)
		os.makedirs(dir_seed,exist_ok=True)

		
		file_mlp = base_mlp.format(dir_seed)
		file_fit = base_fit.format(dir_seed)
		file_mtr = base_mtr.format(dir_seed)
		file_res = base_res.format(dir_seed)
		file_plt_lss = base_plt_lss.format(dir_seed)
		file_plt_res = base_plt_res.format(dir_seed)
		file_plt_res2d = base_plt_res2d.format(dir_seed)

		if not os.path.exists(file_mlp):
			print("Fitting optimal NN of {0} layers with seed {1}".format(
				num_layers,seed))
			#--------------- Instantiate model -----------------------
			seeded_model = create_custom_model(
					input_shape=n_features,
					output_shape=n_targets,  
					num_layers=num_layers,
					size_layers=layer_size,
					activation_layers=activation_layers, 
					activation_output=activation_output,
					seed=seed)
			#------------------------------------------------------------
			
			#-------------- Compile model ---------------------------
			optimal_model = compile_model(model=seeded_model,
						lr_schedule=learning_rate_scheduler(
							lr_decay_function=lr_decay_function,
							initial_learning_rate=lr_initial,
							decay_steps=lr_decay_steps,
							decay_rate =lr_decay_rate,
							# alpha=lr_alpha,
							# end_learning_rate=lr_final,
							# power=lr_power,
							# boundaries=lr_boundaries,
							# values=lr_values
							),
						beta_1=beta_1,
						beta_2=beta_2,
						loss=loss_function,
						metrics=[metric],
						clipnorm=clipnorm
						)
			#--------------------------------------------------------
		
			#------------ Fit ------------------------------
			fit = optimal_model.fit(
						x=x_train.to_numpy(),
						y=y_train.to_numpy(),
						validation_data=(x_val,y_val),
						sample_weight=w_train,
						epochs=epochs,
						batch_size=batch_size,
						verbose=0,
						# callbacks=[early_stopping]
						)
			#----------------------------------------------

			#-------------------- Join losses --------------------
			df_fit_trn = pd.DataFrame(data={
				"loss":fit.history["loss"],
				"metric":fit.history[metric],
				"Case":"Train",
				"Iteration":np.arange(len(fit.history["loss"]))
				})
			df_fit_vld = pd.DataFrame(data={
				"loss":fit.history["val_loss"],
				"metric":fit.history["val_{0}".format(metric)],
				"Case":"Validation",
				"Iteration":np.arange(len(fit.history["val_loss"]))
				})
			df_fit = pd.concat([df_fit_trn,df_fit_vld],
								ignore_index=True)
			df_fit["num_layers"] = num_layers
			df_fit["layer_size"] = layer_size
			df_fit["seed"] = seed
			df_fit.to_csv(file_fit)
			#----------------------------------------------------

			#--------- Save for general plot--------------------------------------------
			df_mtr = pd.DataFrame(data={
								"validation":[fit.history["val_{0}".format(metric)][-1]],
								"training":[fit.history["{0}".format(metric)][-1]],
								"num_layers":[num_layers],
								"layer_size":[layer_size],
								"seed":[seed]
								})
			# for feature in features:
			# 	df_mtr["min_grad_"+feature] = df_grd["grad_"+feature].abs().min()
			df_mtr.to_csv(file_mtr,index=False)
			#----------------------------------------------------------------------------

			# ---------------- Residual diagnostics -----------------------------
			# Evaluate on the held-out validation sample in original coordinates.
			cov_res = analyze_residuals(
				model=optimal_model,
				x_data=x_val,
				y_data=y_val,
				df_original=df_iso,
				features=features,
				targets=targets,
				case="Validation",
				file_res=file_res,
				file_plt_res=file_plt_res,
				file_plt_res2d=file_plt_res2d
				)
			# -------------------------------------------------------------------

			mlp = {
				"features":features,
				"targets":targets,
				"num_layers":num_layers,
				"size_layers":layer_size,
				"mu_transform":iso_mu,
				"sd_transform":iso_sd,
				"logL_lower_par":logL_lower_par,
				"logL_upper_par":logL_upper_par,
				"weights":optimal_model.get_weights(),
				"phot_min":phot_min,
				"domain":domain,
				"seed":seed,
				"val_{0}".format(metric):fit.history["val_{0}".format(metric)][-1],
				"trn_{0}".format(metric):fit.history["{0}".format(metric)][-1],
				"cov_res":cov_res
				}
			with open(file_mlp, "wb") as file:
				dill.dump(mlp, file)

		else:
			print("Reading optimal NN of {0} layers with seed {1}".format(
				num_layers,seed))
			df_mtr = pd.read_csv(file_mtr)
			df_fit = pd.read_csv(file_fit)

			# Reconstruct the saved ANN so residual diagnostics can also
			# be generated when the model was fitted in an earlier run.
			with open(file_mlp, "rb") as file:
				mlp = dill.load(file)

			if "cov_res" not in mlp.keys():

				optimal_model = create_custom_model(
					input_shape=n_features,
					output_shape=n_targets,
					num_layers=num_layers,
					size_layers=mlp["size_layers"],
					activation_layers=activation_layers,
					activation_output=activation_output,
					seed=mlp["seed"]
				)
				optimal_model.set_weights(mlp["weights"])

				cov_res = analyze_residuals(
					model=optimal_model,
					x_data=x_val,
					y_data=y_val,
					df_original=df_iso,
					features=features,
					targets=targets,
					case="Validation",
					file_res=file_res,
					file_plt_res=file_plt_res,
					file_plt_res2d=file_plt_res2d
					)
				mlp["cov_res"] = cov_res

				with open(file_mlp, "wb") as file:
					dill.dump(mlp, file)

		if not os.path.exists(file_plt_lss):
			#------------ Plot Loss --------------------------
			fig, ax = plt.subplots(1, 1, figsize=(16, 8))
			ax = sns.lineplot(data=df_fit,
								x="Iteration",
								y="loss",
								style="Case",
								hue="seed",
								legend=True,
								)
			ax.set_xlabel("Iteration")
			ax.set_ylabel("Loss")
			ax.set_yscale('log')
			ax.set_ylim(bottom=1e-4,top=1e-1)
			fig.savefig(file_plt_lss)
			plt.close()
			#------------------------------------------------------

		fits.append(df_fit)
		mtrs.append(df_mtr)
		# grds.append(df_grd)

	df_fit = pd.concat(fits,ignore_index=False)
		
#------------- Plots as function of layers size ------
df_mtr = pd.concat(mtrs)

#------------ RMS ----------------------------------------
df_tmp = pd.melt(df_mtr,
	id_vars=["num_layers","layer_size","seed"], 
	value_vars=['training', 'validation'],
	var_name='Case',
	value_name='value')

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax = sns.scatterplot(data=df_tmp,
					x="num_layers",
					y="value",
					style="Case",
					hue="seed",
					palette="tab10",
					legend=True,
					zorder=0)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_xlabel("Number of layers")
ax.set_ylabel("Metric {0} [mag]".format(metric))
ax.set_yscale("log")
ax.set_ylim(bottom=1e-4,top=1e-1)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(file_mtrs)
plt.close()
#--------------------------------------------------------
