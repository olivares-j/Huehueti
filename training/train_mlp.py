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
from mlp_model import create_custom_model, compile_model, compute_gradients, learning_rate_scheduler

os.environ["PYTHONHASHSEED"] = "42"
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#------------- Input data ---------------------------
max_label = 1 # label >1 are evolved stars that we do not need
features = ["logAge","logL","logTe"]
targets  = ['Mini']
n_features = len(features)
n_targets = len(targets)
#----------------------------------------------------

# --------------- Model properties --------------------------------
list_of_num_layers = [2] # Number of hidden layers
list_of_size_layers = [5,10,15,20,25,30,35,40]# Units in each hidden layer
seeds = [0] # Seeds for the MLP initializers
activation_layers = "sigmoid" # Activation functions for each hidden layer
activation_output = "linear"  # Activation function for the output layer
loss_function = "mae"
metric = "root_mean_squared_error"
#--------------------------------------------------------------------------

#--------- Fixed Hyperparameters ------------------------------------
optimization_trials = 50
epochs = int(1e3)
lr_decay_function = "InverseTimeDecay"			
lr_decay_steps = epochs
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
#---------------------------------------------------------------------------------

# #------------ Optimized parameters up to 40 Neurons -----------------------------------
# dict_lr_itl = {"value":None,"low":0.05,"high":0.2}
# dict_lr_dcr = {"value":None,"low":5.,"high":50.}
# dict_btsz   = {"value":90971,"options":[1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90971]}
# #--------------------------------------------------------------------------------------------------------

#------------ Optimized parameters from 50 to  Neurons -----------------------------------
dict_lr_itl = {"value":None,"low":0.1,"high":0.5}
dict_lr_dcr = {"value":None,"low":0.1,"high":20.}
dict_btsz   = {"value":10000,"options":[1000,5000,10000,15000,20000,30000,40000,50000,60000,70000,80000,90971]}
#--------------------------------------------------------------------------------------------------------


#--------------- Directories and files ---------------------------------------------
dir_base  = "/home/jolivares/Models/PARSEC/Gaia_EDR3/50-150Myr/Kroupa/"
# Remove the # from the row contain the header in the input file
file_iso  = dir_base + "output_0.1myr.dat" # Input file
dir_mlps  = dir_base + "Optuna_{0}_lrin_{1}_lrdr_{2}_bs_{3}_epochs_{4:1.0e}/".format(
lr_decay_function,dict_lr_itl["value"],dict_lr_dcr["value"],dict_btsz["value"],epochs)
file_mtr  = dir_mlps + "Metric_{0}_l{1}.png"
base_fld  = "{0}_l{1}_s{2}/"
base_dat  = "{0}data.csv"
base_fit  = "{0}fit.csv"
base_opt  = "{0}optuna_study_with_{1}_trials.pkl"
base_mtr  = "{0}metric.csv"
base_mlp  = "{0}mlp.pkl"
base_plt_opt  = "{0}study.png"
base_plt_prm  = "{0}study_params.png"
base_plt_lss  = "{0}loss.png"
base_plt_mtr  = "{0}metric.png"
base_plt_grd  = "{0}gradients.png"
#------------------------------------------------------------------------------------
os.makedirs(dir_mlps,exist_ok=True)

#------------- Load data ----------------------------
df_iso = pd.read_csv(file_iso,
					skiprows=13,
					delimiter=r"\s+",
					header="infer",
					comment="#")
df_iso = df_iso.loc[df_iso["label"]<= max_label]
df_iso = df_iso.loc[:,sum([features,targets],[])]
print(df_iso.describe())
#--------------------------------------------------

#----------------------- Domains ------------------------------------
phot_min = df_iso[targets].min()
domain = {}
for feature in features:
	domain[feature] = [df_iso[feature].min(),df_iso[feature].max()]
#--------------------------------------------------------------------

#------------------- Sample weight ---------------------------------
variable = df_iso[features[1]].to_numpy()
hist, edges = np.histogram(variable, bins=100)
bin_ids = np.digitize(variable, edges[:-1])
weights = 1 / hist[bin_ids-1]
sample_weight = weights/np.mean(weights)
# sample_weight = np.ones_like(weights)
# print(np.sum(sample_weight))
# print(sample_weight.shape)
# print(sample_weight.min(),sample_weight.max())
# ax = sns.histplot(data=df_iso,x=features[1],
# 	color="tab:blue",stat="density",bins=200,element="step",fill=False)
# sns.histplot(data=df_iso,x=features[1],weights=sample_weight,ax=ax,
# 	color="tab:green",stat="density",bins=200,element="step",fill=False)
# plt.show()
# sys.exit()
#---------------------------------------------------------------------------------------

#------- Transformations standardize inputs and outputs --------------------
def forward_transform(df_ori,mu,sd):
	df_trn = df_ori.copy()
	# for col in df_trn.columns:
	for col in features:
		df_trn[col] = (df_trn[col] - iso_mu[col])/iso_sd[col]

	return df_trn

def backward_transform(df_trn,mu,sd):
	df_ori = df_trn.copy()
	# for col in df_ori.columns:
	for col in features:
		df_ori[col] = (df_ori[col]*iso_sd[col]) + iso_mu[col]
	
	return df_ori

iso_mu = df_iso.mean(axis=0)
iso_sd = df_iso.std(axis=0)
df_trn = forward_transform(df_iso,iso_mu,iso_sd)
# df_new = backward_transform(df_trn,iso_mu,iso_sd)
# pd.testing.assert_frame_equal(df_iso,df_new)
df_trn.to_csv(base_dat.format(dir_mlps))
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
# Loop over number of layers, number of hidden neurons and initializer seeds 
for num_layers in list_of_num_layers:
	mtrs = []
	for size_layers in list_of_size_layers:
		print("Working on NN with {0} layers and {1} neurons".format(num_layers,size_layers))
		dir_case   = dir_mlps  + base_fld.format(targets[0],num_layers,size_layers)
		os.makedirs(dir_case,exist_ok=True)			

		#========================= Optuna ============================================
		if not os.path.exists(base_opt.format(dir_case,optimization_trials)):

			#------------------objective function -----------------------------------------
			def objective(trial):
				#--------------- Instantiate model -----------------------
				model = create_custom_model(
						input_shape=n_features,
						output_shape=n_targets,  
						num_layers=num_layers,
						size_layers=size_layers,
						activation_layers=activation_layers, 
						activation_output=activation_output,
						seed=seeds[0])
				#------------------------------------------------------------
				#----------------------------------------------------
				# Hyperparameters related to training
				# epochs = trial.suggest_int("epochs", 20, 100)
				# Early stopping (also tunable if desired)
				# patience = trial.suggest_int("patience", 5, 15)
				# early_stopping = keras.callbacks.EarlyStopping(
				# monitor="val_loss",
				# patience=patience,
				# restore_best_weights=True
				# )
				if dict_lr_itl["value"] is None:
					lr_initial = trial.suggest_float("lr_initial",
						low=dict_lr_itl["low"],
						high=dict_lr_itl["high"],
						log=False)
				else:
					lr_initial = dict_lr_itl["value"]

				if dict_lr_dcr["value"] is None:
					lr_decay_rate = trial.suggest_float("lr_decay_rate",
						low=dict_lr_dcr["low"],
						high=dict_lr_dcr["high"],
						log=False)
				else:
					lr_decay_rate = dict_lr_dcr["value"]

				if dict_btsz["value"] is None:
					batch_size = trial.suggest_categorical(
						name="batch_size",
						choices=dict_btsz["options"]
						)
				else:
					batch_size = dict_btsz["value"]


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
							verbose=0,
							# callbacks=[early_stopping]
							)
				#----------------------------------------------

				return fit.history["val_{0}".format(metric)][-1]

			# ---- Run optimization ----
			study = optuna.create_study(direction="minimize")
			study.optimize(objective, n_trials=optimization_trials)

			#------------------ Plot study --------------------------------------------
			df = study.trials_dataframe(multi_index=True)
			df.rename(columns={"value":"metric"},inplace=True)
			df.set_index("metric",inplace=True)
			df = df.loc[:,[("params",name) for name in study.best_params.keys()]]
			df = df.stack(future_stack=True).reset_index(level=0)
			df.rename(columns={"params":"value"},inplace=True)
			df.index.name = "Parameter"
			df.reset_index(inplace=True)

			# --- Create FacetGrid with rows from MultiIndex ---
			g = sns.FacetGrid(df,
					col='Parameter',
					margin_titles=True,
					sharey=True,
					sharex=False
					)

			g.map_dataframe(sns.lineplot, x='value', y='metric')
			g.set(yscale="log")
			g.savefig(base_plt_opt.format(dir_case))

			df = study.trials_dataframe(multi_index=False)
			df.rename(columns={"value":"metric"},inplace=True)
			df.set_index("metric",inplace=True)
			names = list(study.best_params.keys())
			fig, ax = plt.subplots(1, 1, figsize=(16, 8))
			ax = sns.scatterplot(data=df,
				x="params_{0}".format(names[0]), 
				y="params_{0}".format(names[1]),
				hue="metric",
				size="metric",
				palette="magma",
				ax=ax
				)
			fig.savefig(base_plt_prm.format(dir_case))
			plt.close()
			#---------------------------------------------------------------

			with open(base_opt.format(dir_case,optimization_trials), "wb") as file:
				dill.dump(study, file)
		else:
			with open(base_opt.format(dir_case,optimization_trials), "rb") as file:
				study = dill.load(file)

			# #------------------ Plot study --------------------------------------------
			# df = study.trials_dataframe(multi_index=False)
			# df.rename(columns={"value":"metric"},inplace=True)
			# df.set_index("metric",inplace=True)
			# names = list(study.best_params.keys())
			# fig, ax = plt.subplots(1, 1, figsize=(16, 8))
			# ax = sns.scatterplot(data=df,
			# 	x="params_{0}".format(names[0]), 
			# 	y="params_{0}".format(names[1]),
			# 	hue="metric",
			# 	size="metric",
			# 	palette="magma",
			# 	ax=ax
			# 	)
			# fig.savefig(base_plt_prm.format(dir_case))
			# plt.close()
			# #---------------------------------------------------------------

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
			lr_initial = dict_lr_itl["value"]

		if "lr_decay_rate"in best_trial.params:
			lr_decay_rate = best_trial.params["lr_decay_rate"]
		else:
			lr_decay_rate = dict_lr_dcr["value"]

		if "batch_size"in best_trial.params:
			batch_size = best_trial.params["batch_size"]
		else:
			batch_size = dict_btsz["value"]


		print("lr_initial: {0}".format(lr_initial))
		print("lr_decay_rate: {0}".format(lr_decay_rate))
		print("batch_size: {0}".format(batch_size))
		#================================================================

		#================= Loop over seeds =================================================
		fits = []
		for seed in seeds:
			dir_seed = dir_case + "seed_{0}/".format(seed)
			os.makedirs(dir_seed,exist_ok=True)

			if not os.path.exists(base_mlp.format(dir_seed)):
				print("Fitting optimal NN of {0} layers and {1} neurons with seed {2}".format(
					num_layers,size_layers,seed))
				#--------------- Instantiate model -----------------------
				seeded_model = create_custom_model(
						input_shape=n_features,
						output_shape=n_targets,  
						num_layers=num_layers,
						size_layers=size_layers,
						activation_layers=activation_layers, 
						activation_output=activation_output,
						seed=seed)
				#------------------------------------------------------------
				
				#-------------- Compile model ------------------------------------------
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
				#----------------------------------------------------------------------

			
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

				# names, norms = compute_gradients(
				# 				model=optimal_model,
				# 				x=df_trn[features].to_numpy(),
				# 				y=df_trn[targets].to_numpy())

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
				df_fit["size_layers"] = size_layers
				df_fit["seed"] = seed
				#----------------------------------------------------

				#--------- Save ------------------------------------
				df_fit.to_csv(base_fit.format(dir_seed))

				#--------- Save for general plot--------------------------------------------
				df_mtr = pd.DataFrame(data={
									"validation":[fit.history["val_{0}".format(metric)][-1]],
									"training":[fit.history["{0}".format(metric)][-1]],
									"num_layers":[num_layers],
									"size_layers":[size_layers],
									"seed":[seed]
									})
				df_mtr.to_csv(base_mtr.format(dir_seed),index=False)
				#----------------------------------------------------------------------------

				mlp = {
					"features":features,
					"targets":targets,
					"num_layers":num_layers,
					"size_layers":size_layers,
					"mu_transform":iso_mu,
					"sd_transform":iso_sd,
					"weights":optimal_model.get_weights(),
					"phot_min":phot_min,
					"domain":domain,
					"seed":seed,
					"val_{0}".format(metric):fit.history["val_{0}".format(metric)][-1],
					"trn_{0}".format(metric):fit.history["{0}".format(metric)][-1]
					}
				with open(base_mlp.format(dir_seed), "wb") as file:
					dill.dump(mlp, file)

			else:
				print("Reading optimal NN of {0} layers and {1} neurons with seed {2}".format(
					num_layers,size_layers,seed))
				df_mtr = pd.read_csv(base_mtr.format(dir_seed))
				df_fit = pd.read_csv(base_fit.format(dir_seed))
				#-----------------------------------------------
			fits.append(df_fit)
			mtrs.append(df_mtr)

		df_fit = pd.concat(fits,ignore_index=False)

		if not os.path.exists(base_plt_lss.format(dir_case)):
			#------------ Plot Loss --------------------------
			fig, ax = plt.subplots(1, 1, figsize=(16, 8))
			ax = sns.lineplot(data=df_fit,
								x="Iteration",
								y="loss",
								hue="Case",
								legend=True,
								hue_order=["Train","Validation"])
			ax.set_xlabel("Iteration")
			ax.set_ylabel("Loss")
			ax.set_yscale('log')
			ax.set_ylim(bottom=1e-3,top=1e-1)
			fig.savefig(base_plt_lss.format(dir_case))
			plt.close()
			#------------------------------------------------------

		if not os.path.exists(base_plt_mtr.format(dir_case)):
			#------------ Plot Metric --------------------------
			fig, ax = plt.subplots(1, 1, figsize=(16, 8))
			ax = sns.lineplot(data=df_fit,
								x="Iteration",
								y="metric",
								hue="Case",
								legend=True,
								hue_order=["Train","Validation"])
			ax.set_xlabel("Iteration")
			ax.set_ylabel("Metric {0}".format(metric))
			ax.set_yscale('log')
			ax.set_ylim(bottom=1e-3,top=1e-1)
			fig.savefig(base_plt_mtr.format(dir_case))
			plt.close()
			#------------------------------------------------------

		# #---------------- Plot gradients ------------------------
		# plt.figure(figsize=(10,4))
		# plt.bar(range(len(norms)), norms)
		# plt.xticks(range(len(norms)), names, rotation=90)
		# plt.ylabel("Gradient L2 norm")
		# plt.title("Gradient norms per parameter")
		# plt.tight_layout()
		# plt.savefig(base_plt_grd.format(dir_case))
		# plt.close()
		# #---------------------------------------------------------------------
			
	#------------- Plot error and loss as function of layers size ------
	df_mtr = pd.concat(mtrs)
	df_mtr = pd.melt(df_mtr,
		id_vars=["num_layers","size_layers","seed"], 
		value_vars=['training', 'validation'],
		var_name='Case',
		value_name='value')

	fig, ax = plt.subplots(1, 1, figsize=(16, 8))
	ax = sns.lineplot(data=df_mtr,
						x="size_layers",
						y="value",
						style="Case",
						palette="tab10",
						legend=True,
						zorder=0)
	ax.set_xlabel("Size of layers")
	ax.set_ylabel("Metric {0} [mag]".format(metric))
	ax.set_yscale("log")
	ax.set_ylim(bottom=1e-3,top=1e-1)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig.savefig(file_mtr.format(targets[0],num_layers))
	plt.close()