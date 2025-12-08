import sys
import os
import keras
from keras.layers import Dense
from keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

from typing import List
import dill

from sklearn.preprocessing import MinMaxScaler, PowerTransformer

print(f"Running Tensoflow {tf.__version__}")


# Defining model archiquetures
# ----------------------------
def create_custom_model(
	input_shape: int,
	num_layers: int,
	size_layers: int,
	activation_layers: str,
	output_units: int,
	output_activation: str = "linear",
) -> Sequential:
	"""
	Create a Keras Sequential model with the specified number of layers.

	Parameters:
	- input_shape (int): The number of features in the input data.
	- num_layers (int): The number of hidden layers in the model.
	- units (List[int]): A list containing the number of units for each hidden layer.
	- activations (List[str]): A list containing the activation function for each hidden layer.
	- output_units (int): The number of units in the output layer.
	- output_activation (str): The activation function for the output layer (default is 'linear').

	Returns:
	- model (Sequential): The compiled Keras Sequential model.
	"""

	model = Sequential()

	# Input layer
	model.add(Dense(size_layers, activation=activation_layers, input_shape=(input_shape,)))

	# Hidden layers
	for i in range(1, num_layers):
		model.add(
			Dense(
				size_layers,
				activation=activation_layers,
				kernel_initializer=keras.initializers.RandomNormal(),
				bias_initializer=keras.initializers.RandomNormal(),
			)
		)

	# Output layer
	model.add(
		Dense(
			output_units,
			activation=output_activation,
			kernel_initializer=keras.initializers.RandomNormal(),
			bias_initializer=keras.initializers.RandomNormal()
		)
	)

	# Compile the model
	model.compile(
		# optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
		optimizer=tf.keras.optimizers.Adam(),
		# loss=keras.losses.MeanAbsoluteError(),
		loss=keras.losses.MeanSquaredError(),
		metrics=[keras.metrics.RootMeanSquaredError()],
	)

	return model



dir_base = "/home/jolivares/Repos/SakamII/"
dir_nns  = dir_base + "mlps/"
dir_data = dir_base + "data/"

train = True
fit_transform = False

num_layers = 5 # Number of hidden layers
size_layers = 48  # Units in each hidden layer
training_epochs = 10000
batch_size = 1000

file_iso = dir_data + "parametrizations/parametrized_max_label_1_PARSEC_20-200Myr_GDR3+PanSTARRS+2MASS.csv"

dir_case = dir_nns + "PARSEC_{0}x{1}/".format(num_layers,size_layers)
os.makedirs(dir_case,exist_ok=True)

file_mlp = dir_case  + "mlp.pkl"
file_los = dir_case  + "los.png"
file_rms = dir_case  + "rms.png"
file_err = dir_case  + "err.png"

features = ["age_Myr","parameter"]
targets  = ['Mini','Gmag', 'G_BPmag', 'G_RPmag', 'gP1mag', 'rP1mag', 'iP1mag','zP1mag','yP1mag','Jmag', 'Hmag', 'Ksmag']
photometric_bands = targets[1:]

columns = sum([features,targets],[])

# ------------------------------ Model properties --------------------------------
input_shape = len(features)  # Number of features in the input data
output_units = len(targets)  # Number of units in the output layer
activation_layers = "relu" # Activation functions for each hidden layer
output_activation = "linear"  # Activation function for the output layer
#--------------- Instantiate model -----------------------
model = create_custom_model(
input_shape=input_shape, 
num_layers=num_layers,
size_layers=size_layers, 
activation_layers=activation_layers, 
output_units=output_units, 
output_activation=output_activation
)
#------------------------------------------------------------
#----------------------------------------------------------------------------------


#------------- Load data ------------------------
df_iso = pd.read_csv(file_iso,usecols=columns)
# df_iso = df_iso.query("age_Myr == [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200]")
# df_iso = df_iso.query("age_Myr == [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]")
df_iso = df_iso.query("age_Myr == [20,40,60,80,100,120,140,160,180,200]")
#-------------------------------------------
print(df_iso.describe())



if fit_transform:
	raw_features = df_iso[features].to_numpy()

	#-------- Photometric limits ---------------
	phot_min = df_iso[photometric_bands].min()
	#-------------------------------------------


	#---------------- Transformations ----------------------------------------
	BoxCox = PowerTransformer(
				method="box-cox", 
				standardize=False).fit(raw_features)
	inputs = BoxCox.transform(raw_features)
	MinMax = MinMaxScaler().fit(inputs)
	#-------------------------------------------------------------------------

	#------------ Save transformations ---------------------
	mlp = {
		"scalers":[BoxCox, MinMax],
		"weights": None,
		"phot_min":phot_min,
		"age_domain":[df_iso["age_Myr"].min(),df_iso["age_Myr"].max()],
		"par_domain":[df_iso["parameter"].min(),df_iso["parameter"].max()],
		}
	with open(file_mlp, "wb") as file:
		dill.dump(mlp, file)
	#-------------------------------------------------------

else:
	#-------------- Read transformations ----------------
	with open(file_mlp, "rb") as file:
		mlp = dill.load(file)
		scalers    = mlp["scalers"]
		phot_min   = mlp["phot_min"]
		age_domain = mlp["age_domain"]
		par_domain = mlp["par_domain"]
	#------------------------------------------------

	BoxCox = scalers[0]
	MinMax = scalers[1]


print(80*"-")

if train:
	raw_features = df_iso[features].to_numpy()
	raw_targets  = df_iso[targets].to_numpy()

	transformed_features = MinMax.transform(BoxCox.transform(raw_features))
	
	#------------ Load existing fit --------------------------------
	with open(file_mlp, "rb") as file:
		mlp = dill.load(file)
		weights  = mlp["weights"]
	#-------------------------------------

	#--------------------------------
	if weights is not None:
		model.set_weights(weights)
	#------------------------------

	print("Fitting model ...")
	history = model.fit(
		transformed_features,
		raw_targets,
		epochs=training_epochs,
		batch_size=batch_size,
		verbose=1
		)

	# Plot loss vs. iter
	loss = history.history["loss"]
	iters = range(len(loss))

	fig, axs = plt.subplots(1, 1, figsize=(16, 8))
	fig.suptitle("Training vs. evaluation loss")
	axs.set_title("Total loss")
	axs.set_ylabel("log(msa)")
	axs.plot(iters, loss, label="Total cost", zorder=1)
	axs.set_yscale("log")
	fig.savefig(file_los)
	plt.close()

	# Plot RMSE vs. iter
	pred_rmse = history.history["root_mean_squared_error"]
	iters = range(len(pred_rmse))

	fig, ax = plt.subplots(1, 1, figsize=(16, 8))
	ax.set_title("RMSE")
	ax.plot(iters, pred_rmse, label="Prediction")
	ax.legend()
	ax.set_ylabel("log(rmse)")
	ax.set_yscale("log")
	ax.legend()
	fig.savefig(file_rms)
	plt.close()

	# Get optimal values
	weights = model.get_weights()

	# Save optimal values
	mlp = {
		"scalers":[BoxCox, MinMax],
		"weights":weights,
		"phot_min":phot_min,
		"age_domain":age_domain,
		"par_domain":par_domain}
	with open(file_mlp, "wb") as file:
		dill.dump(mlp, file)

	# Check weights correspondence
	i = 0
	for weight in weights:
		print(f"weights[{i}].shape = {weight.shape}")
		i += 1


#------------ Load existing fit --------------------------------
with open(file_mlp, "rb") as file:
	mlp = dill.load(file)
	weights = mlp["weights"]

model.set_weights(weights)
#----------------------------------------------------------------

#-------------- Plot predictions ---------------------
df_tst = df_iso.copy()
df_tst.reset_index(drop=True,inplace=True)

df_features = df_tst[features]

ouput_tst = model(MinMax.transform(BoxCox.transform(
	df_features.to_numpy())),
	training=False)

df_bands = pd.DataFrame(columns=targets,data=ouput_tst._numpy())
df_prd = pd.concat([df_features,df_bands],axis=1,ignore_index=False)

df = pd.merge(left=df_tst,right=df_bands,left_index=True,right_index=True,suffixes=["_tst","_prd"])
df.set_index(features,drop=True,inplace=True)


for trgt in targets:
	df[trgt] = df.apply(lambda x: x[trgt+"_prd"]-x[trgt+"_tst"],axis=1)

df = df.loc[:,targets].stack().reset_index()
df.columns = ["age_Myr","parameter","target","value"]
print(df.describe())

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax = sns.scatterplot(data=df,
					x="parameter",
					y="value",
					hue="target",
					style="age_Myr",
					zorder=0)
ax.set_xlabel("Parameter")
ax.set_ylabel("Test-Predicted")
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
fig.savefig(file_err)
plt.close()

df_tst.loc[:,"color"] = df_tst.apply(lambda x: x["Gmag"]-x["G_RPmag"],axis=1)
df_prd.loc[:,"color"] = df_prd.apply(lambda x: x["Gmag"]-x["G_RPmag"],axis=1)

ax = sns.scatterplot(data=df_tst,
		x="color",
		y="Gmag",
		hue="age_Myr",
		zorder=0)
ax = sns.lineplot(data=df_prd,
		x="color",
		y="Gmag",
		hue="age_Myr",
		legend=False,
		sort=False,
		zorder=1,
		ax=ax)
ax.set_xlabel("G - RP")
ax.set_ylabel("G [mag]")
ax.invert_yaxis()
plt.show()

df_tst = df_tst.groupby("age_Myr").get_group(120.0)
df_prd = df_prd.groupby("age_Myr").get_group(120.0)

plt.figure(0)
ax = sns.scatterplot(data=df_tst,
		x="color",
		y="Gmag",
		hue="age_Myr",
		zorder=0)
ax = sns.lineplot(data=df_prd,
		x="color",
		y="Gmag",
		hue="age_Myr",
		legend=False,
		sort=False,
		zorder=1,
		ax=ax)
ax.set_xlabel("G -RP [mag]")
ax.set_ylabel("G [mag]")
ax.invert_yaxis()
plt.show()