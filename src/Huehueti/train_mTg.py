import sys
import os
import dill
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from mlp_model import create_custom_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train = True

# ------------------------------ Model properties --------------------------------
features = ["age_Myr","parameter"]
targets  = ["Mini","logTe","logg"]
list_of_num_layers = [4,5,6,7,8,9,10] # Number of hidden layers
size_layers = 256  # Units in each hidden layer
training_epochs = [9999,9999,9999]
learning_rates  = [1e-3,1e-4,1e-5]
activation_layers = "relu" # Activation functions for each hidden layer
output_activation = "linear"  # Activation function for the output layer
batch_fraction = 1.0
#--------------------------------------------------------------------------

#--------------- Directories and files ---------------------------------------------
file_iso  = "/home/jolivares/Repos/Huehueti/data/parametrizations/parametrized_max_label_1_PARSEC_20-200Myr_GDR3+PanSTARRS+2MASS.csv"
dir_mlps  = "/home/jolivares/Repos/Huehueti/mlps/PARSEC/"
base_fld  = "mTg_l{0}_s{1}/"
base_mlp  = "{0}mlp.pkl"
base_lss  = "{0}loss.png"
base_fit  = "{0}fit.png"
base_err  = "{0}error.png"
#------------------------------------------------------------------------------------

#------------- Load data ----------------------------------------------------
df_iso = pd.read_csv(file_iso,usecols=sum([features,targets],[]))
df_iso = df_iso.query("age_Myr == [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200]")
# df_iso = df_iso.query("age_Myr == [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]")
# df_iso = df_iso.query("age_Myr == [20,40,60,80,100,120,140,160,180,200]")
# df_iso = df_iso.query("age_Myr == [20,200]")
print(df_iso.describe())
#----------------------------------------------------------------------------


for num_layers in list_of_num_layers:
	dir_case   = dir_mlps  + base_fld.format(num_layers,size_layers)
	os.makedirs(dir_case,exist_ok=True)

	file_mlp = base_mlp.format(dir_case)

	#------------------ Train ---------------------------
	if train:
		#---------- Numpy arrays ------------------
		raw_features = df_iso[features].to_numpy()
		raw_targets  = df_iso[targets].to_numpy()
		#-------------------------------------------

		#---------------- Transformations ------------------------
		BoxCox = PowerTransformer(
					method="box-cox", 
					standardize=False).fit(raw_features)
		inputs = BoxCox.transform(raw_features)
		MinMax = MinMaxScaler().fit(inputs)
		scalers = [BoxCox, MinMax]
		#---------------------------------------------------------

		#------------ Domains ---------------------------------------------
		phot_min = df_iso[targets].min()
		age_domain = [df_iso["age_Myr"].min(),df_iso["age_Myr"].max()]
		par_domain = [df_iso["parameter"].min(),df_iso["parameter"].max()]
		#-------------------------------------------------------------------------

		batch_size   = int(batch_fraction*df_iso.shape[0])
		transformed_features = scalers[1].transform(scalers[0].transform(raw_features))

		losses = []
		for learning_rate,epochs in zip(learning_rates,training_epochs):

			#--------------- Instantiate model -----------------------
			model = create_custom_model(
			input_shape=len(features), 
			num_layers=num_layers,
			size_layers=size_layers, 
			activation_layers=activation_layers, 
			output_units=len(targets), 
			output_activation=output_activation,
			learning_rate=learning_rate)
			#------------------------------------------------------------

			#------------ Load previous fit --------
			if os.path.isfile(file_mlp):
				with open(file_mlp, "rb") as file:
					mlp = dill.load(file)
					model.set_weights(mlp["weights"])
			#-----------------------------------------

			#------------ Fit -----------
			fit = model.fit(
				transformed_features,
				raw_targets,
				epochs=epochs,
				batch_size=batch_size,
				verbose=1)
			#----------------------------

			#--------- Save --------------------
			mlp = {
				"num_layers":num_layers,
				"size_layers":size_layers,
				"scalers":scalers,
				"weights":model.get_weights(),
				"phot_min":phot_min,
				"targets":targets,
				"age_domain":age_domain,
				"par_domain":par_domain}

			with open(file_mlp, "wb") as file:
				dill.dump(mlp, file)
			#------------------------------------

			#------------- Data for loss plot ----------
			losses.append(fit.history["loss"])
			#---------------------------------------------

		loss = np.concatenate(losses,axis=0)
		df = pd.DataFrame(data={
					"loss":loss,
					"iter":np.arange(len(loss)),
					})

		#------------ Plot Loss --------------------------
		fig, ax = plt.subplots(1, 1, figsize=(16, 8))
		ax = sns.lineplot(data=df,
							x="iter",
							y="loss",
							zorder=0)
		ax.set_xlabel("Iteration")
		ax.set_ylabel("Loss")
		ax.set_yscale('log')
		# plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
		fig.savefig(base_lss.format(dir_case))
		plt.close()


	

	#---------- Instantiate model ---------------------
	model = create_custom_model(
			input_shape=len(features), 
			num_layers=num_layers,
			size_layers=size_layers, 
			activation_layers=activation_layers, 
			output_units=len(targets), 
			output_activation=output_activation)
	#------------------------------------------------

	#------------ Load existing fit --------
	with open(file_mlp, "rb") as file:
		mlp = dill.load(file)
		scalers = mlp["scalers"]
		model.set_weights(mlp["weights"])
	#----------------------------------------

	#-------------- Plot predictions ---------------------
	df_tst = df_iso.copy()
	df_tst.reset_index(drop=True,inplace=True)

	df_features = df_tst[features]
	input_tst = scalers[1].transform(scalers[0].transform(df_features.to_numpy()))
	ouput_tst = model(input_tst,training=False)

	df_tar = pd.DataFrame(columns=targets,data=ouput_tst._numpy())
	df_prd = pd.concat([df_features,df_tar],axis=1,ignore_index=False)

	df = pd.merge(left=df_tst,right=df_tar,left_index=True,right_index=True,suffixes=["_tst","_prd"])
	df.set_index(features,drop=True,inplace=True)

	for trgt in targets:
		df[trgt] = df.apply(lambda x: (x[trgt+"_prd"]-x[trgt+"_tst"])/x[trgt+"_tst"],axis=1)

	df = df.loc[:,targets].stack().reset_index()
	df.columns = ["age_Myr","parameter","target","value"]
	print(df.describe())

	fig, ax = plt.subplots(1, 1, figsize=(16, 8))
	ax = sns.scatterplot(data=df,
						x="parameter",
						y="value",
						hue="age_Myr",
						style="target",
						zorder=0)
	ax.set_xlabel("Parameter")
	ax.set_ylabel("(Predicted-Test)/Test")
	plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
	fig.savefig(base_err.format(dir_case))
	plt.close()

	ax = sns.scatterplot(data=df_tst,
			x="parameter",
			y="Mini",
			hue="age_Myr",
			zorder=0)
	ax = sns.lineplot(data=df_prd,
			x="parameter",
			y="Mini",
			hue="age_Myr",
			legend=False,
			sort=False,
			zorder=1,
			ax=ax)
	ax.set_xlabel("Theta")
	ax.set_ylabel("Mini")
	plt.savefig(base_fit.format(dir_case))
	plt.close()