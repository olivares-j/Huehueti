"""Pre-Trained Neural Network as BT-Settl model interpolator. PyTensor implementation.

This module wraps a pre-trained feed-forward neural network used to interpolate
stellar model outputs (mass and absolute photometry) as a function of age and a
single stellar parameter (theta). The network parameters and scalers are read
from a dill/pickle file and the forward pass is implemented using PyTensor
(pytensor) tensors so the outputs can be used inside PyMC models.
"""
import os
import sys
from pickle import load
import numpy as np
import pytensor
import pytensor.tensor as pt
import pandas as pn

import matplotlib.pyplot as plt
import seaborn as sns

def relu(x): #With alpha set to zero to improve speed
    return 0.5 * (x + abs(x))

def sigmoid(x):
	return 1./(1.+ pt.exp(-x))

def NN_2_layers(X,W,b,mu,sd):
	A1  = sigmoid(pt.dot(X,  W[0]) + b[0])
	A2  = sigmoid(pt.dot(A1, W[1]) + b[1])
	# Final linear output (no activation)
	out = pt.dot(A2,W[2]) + b[2]
	return out

def NN_3_layers(X,W,b,mu,sd):
	A1  = sigmoid(pt.dot(X,  W[0]) + b[0])
	A2  = sigmoid(pt.dot(A1, W[1]) + b[1])
	A3  = sigmoid(pt.dot(A2, W[2]) + b[2])
	# Final linear output (no activation)
	out = pt.dot(A3,W[3]) + b[3]
	return out

def NN_4_layers(X,W,b,mu,sd):
	A1  = sigmoid(pt.dot(X,  W[0]) + b[0])
	A2  = sigmoid(pt.dot(A1, W[1]) + b[1])
	A3  = sigmoid(pt.dot(A2, W[2]) + b[2])
	A4  = sigmoid(pt.dot(A3, W[3]) + b[3])
	# Final linear output (no activation)
	out = pt.dot(A4,W[4]) + b[4]
	return out

def NN_5_layers(X,W,b,mu,sd):
	A1  = sigmoid(pt.dot(X,  W[0]) + b[0])
	A2  = sigmoid(pt.dot(A1, W[1]) + b[1])
	A3  = sigmoid(pt.dot(A2, W[2]) + b[2])
	A4  = sigmoid(pt.dot(A3, W[3]) + b[3])
	A5  = sigmoid(pt.dot(A4, W[4]) + b[4])
	# Final linear output (no activation)
	out = pt.dot(A5,W[5]) + b[5]
	return out

# class MLP_phot:
	"""Wrapper around a pretrained probabilistic photometric MLP.

	The probabilistic ANN has six outputs:
	    [G_BPmag_mean, Gmag_mean, G_RPmag_mean,
	     G_BPmag_sigma, Gmag_sigma, G_RPmag_sigma]

	The first three outputs are the predicted absolute photometry. The last
	three are the ANN approximation uncertainties in magnitudes. The sigma
	head is reconstructed from the raw network output using the same
	softplus transformation and floor used during training.
	"""

	def __init__(self,
		file_mlp: str,
		features = ["logAge","logL"],
		targets = ["G_BPmag","Gmag","G_RPmag"]
		):
		"""Load the trained probabilistic MLP and its scalers."""

		print("Reading targets: {0}".format(targets))
		assert os.path.exists(file_mlp), (
			"The file containing optimal weights and scalers cannot be found. "
			"Please, provide a valid path"
		)

		with open(file_mlp, 'rb') as file:
			tmp = load(file)

		tmp_targets = tmp["targets"]
		tmp_features = tmp["features"]

		assert tmp_features == features, (
			"Features mismatch! Expected: {0}".format(features)
		)
		assert tmp_targets == targets, (
			"Target mismatch! Expected: {0}".format(targets)
		)

		weights = tmp["weights"]
		num_lyrs = tmp["num_layers"]
		phot_min = tmp["phot_min"]
		domain = tmp["domain"]
		mu = tmp["mu_transform"]
		sd = tmp["sd_transform"]
		cvrlwpar = tmp["logL_lower_par"]
		cvruppar = tmp["logL_upper_par"]

		self.bands = targets
		self.domain = domain

		# Residual covariance is retained when present, for compatibility
		# with the original MLP_phot interface.
		self.cov_res = tmp.get("cov_res", None)
		if self.cov_res is not None:
			self.sd_res = np.sqrt(np.diag(self.cov_res))
		else:
			self.sd_res = None

		# Keras stores weights as [kernel, bias, kernel, bias, ...].
		# The final two Dense layers are the mean and raw-sigma heads.
		self.W = weights[::2]
		self.b = weights[1::2]

		self.mu = mu.loc[targets].to_numpy()
		self.sd = sd.loc[targets].to_numpy()
		self.mu_features = mu.loc[features].to_numpy()
		self.sd_features = sd.loc[features].to_numpy()
		self.min = phot_min

		self.logL_limits = {
			"lower": {
				"intercept": cvrlwpar[1],
				"slope": cvrlwpar[0]
			},
			"upper": {
				"intercept": cvruppar[1],
				"slope": cvruppar[0]
			}
		}

		if num_lyrs == 2:
			self.Function = NN_2_layers
		elif num_lyrs == 3:
			self.Function = NN_3_layers
		elif num_lyrs == 4:
			self.Function = NN_4_layers
		elif num_lyrs == 5:
			self.Function = NN_5_layers
		else:
			sys.exit("Unsupported number of layers for Phot")

		self.sigma_floor = 1.0e-6

	def __call__(self, logAge, covariate, n_stars):
		"""Compute photometric means and ANN sigmas.

		Returns
		-------
		photometry : pytensor.tensor.TensorVariable
			Predicted absolute photometry, shape (n_stars, n_bands).

		sigma : pytensor.tensor.TensorVariable
			ANN approximation uncertainty in magnitudes, shape
			(n_stars, n_bands).
		"""

		x = pt.stack(
			[pt.tile(logAge, (n_stars,)), covariate],
			axis=1
		)

		A0 = (x - self.mu_features) / self.sd_features

		# The network returns six quantities:
		# [mean_1, mean_2, mean_3, raw_sigma_1, raw_sigma_2, raw_sigma_3].
		output = self.Function(
			X=A0,
			W=self.W,
			b=self.b,
			mu=self.mu,
			sd=self.sd
		)

		n_bands = len(self.bands)

		phot = output[:, :n_bands]
		raw_sigma = output[:, n_bands:]

		# Must match the Keras training model:
		# sigma = softplus(raw_sigma) + SIGMA_FLOOR.
		sigma = pt.nnet.softplus(raw_sigma) + self.sigma_floor

		return phot, sigma

class MLP_one:
	"""Wrapper around a pretrained multilayer perceptron.
	"""

	def __init__(self,
		file_mlp: str,
		features = ["logAge","logL"],
		target = "Mini"
		):
		"""Load the mlp of the variate given age and covariate
		"""

		print("Reading target: {0}".format(target))
		assert os.path.exists(file_mlp),"The file containing optimal weights and scalers cannot be found. Please, provide a valid path"
		
		with open(file_mlp, 'rb') as file:
			tmp = load(file)
			tmp_targets   = tmp["targets"]
			tmp_features  = tmp["features"]

			assert tmp_features == features, "Features mismatch! Expected: {0}".format(features)
			assert tmp_targets[0] == target, "Target mismatch! Expected {0}".format(target)

			weights   = tmp["weights"]
			num_lyrs  = tmp["num_layers"]
			domain    = tmp["domain"]
			mu        = tmp["mu_transform"]
			sd        = tmp["sd_transform"]
			cvrlwpar  = tmp["covariate_lower_par"]
			cvruppar  = tmp["covariate_upper_par"]

		self.target = target
		self.domain = domain
	
		self.W = weights[::2]
		self.b = weights[1::2]
		
		self.mu = mu.loc[[target]].to_numpy()
		self.sd = sd.loc[[target]].to_numpy()
		self.mu_features = mu.loc[features].to_numpy()
		self.sd_features = sd.loc[features].to_numpy()

		if target == "Mini":
			self.covariate_limits = {
				"lower":{"a":cvrlwpar[0],"b":cvrlwpar[1]},
				"upper":{"a":cvruppar[0],"b":cvruppar[1]},
				}
		elif target == "logL":
			self.covariate_limits = {
				"lower":{"a":cvrlwpar[0],"b":cvrlwpar[1],"c":cvrlwpar[2]},
				"upper":{"a":cvruppar[0],"b":cvruppar[1],"c":cvruppar[2]},
				}
		else:
			sys.exit("Target not supported!")
		
		if num_lyrs == 2:
			self.Function = NN_2_layers
		elif num_lyrs == 3:
			self.Function = NN_3_layers
		elif num_lyrs == 4:
			self.Function = NN_4_layers
		else:
			sys.exit("Unsupported number of layers")

	

	def __call__(self, logAge, covariate, n_stars):
		"""Compute NN predictions for given age and covariate.
		"""
		x = pt.stack([pt.tile(logAge, (n_stars,)), covariate],axis=1)
		# x = pt.stack([logAge, covariate],axis=1)

		A0 = (x - self.mu_features)/self.sd_features

		variate = self.Function(X=A0,
						W=self.W,
						b=self.b,
						mu=self.mu,
						sd=self.sd
						).flatten()

		return variate

# The block below is an example usage / quick visual test when running the file
# directly. It is not required for the library functionality and will only run
# in interactive/script mode.
if __name__ == "__main__":

	rng = np.random.default_rng(42)

	dir_mlps = "/home/jolivares/Models/PARSEC@phanocles/200-600Myr/"
	case = "Optuna_InverseTimeDecay_epochs_5e+02_trials_50_0.5myr"

	file_iso   = dir_mlps + "Gaia_EDR3_1myr.dat"
	file_plt   = dir_mlps + case +"/l4/seed_0_wgt_1/" + "Differences.png"
	files_mlps = {
	"Phot":dir_mlps + case +"/l4/seed_0_wgt_1/mlp.pkl",
	"Mini":dir_mlps + case +"/Mini_l4/seed_0/mlp.pkl",
	"logL":dir_mlps + case +"/logL_l4/seed_0/mlp.pkl"
	}
	bands   = ["G_BPmag","Gmag","G_RPmag"]
	features = ["logAge","logL"]

	alls = sum([["logAge","logL","Mini"],bands],[])

	mlp_phot = MLP_phot(
		features=features,
		targets=bands,
		file_mlp=files_mlps["Phot"])

	mlp_mass = MLP_one(
		features=["logAge","logL"],
		target="Mini",
		file_mlp=files_mlps["Mini"])

	mlp_logl = MLP_one(
		features=["logAge","Mini"],
		target="logL",
		file_mlp=files_mlps["logL"])

	# Example: load an isochrone from a parametrized CSV and overlay predicted photometry
	# logAge = 7.17609 # 15Myr
	# logAge = 7.77815 #60 Myr
	# logAge = 8.0 #100Myr
	# logAge = 8.14613 #140Myr
	logAge = 8.34242 #220Myr
	logAges = [logAge]
	# logAges = [8.25527,8.34242,8.41497]
	# logAges = [7.77815,8.0,8.34242]
	max_label = 1

	df_iso = pn.read_csv(file_iso,
					# skiprows=13,
					delimiter=r"\s+",
					header="infer",
					comment="#")
	df_iso = df_iso.loc[df_iso["label"]<= max_label]
	df_iso = df_iso.loc[:,alls]
	df_iso = pn.concat([df_iso.query('logAge == {0}'.format(logAge)) for logAge in logAges])
	df_iso.reset_index(drop=True,inplace=True)
	print(df_iso)
	# df_iso = df_iso.groupby("logAge").get_group(logAge)
	n_stars = df_iso.shape[0]


	#------------ logL and logTe ------------------------------
	logAge = df_iso["logAge"].to_numpy()
	logL  = df_iso["logL"].to_numpy()
	Mini  = df_iso["Mini"].to_numpy()

	# logTe = df_iso["logTe"].to_numpy()
	# logL = np.linspace(
	# 	start=mlp_phot.domain["logL"][0],
	# 	stop=mlp_phot.domain["logL"][1],
	# 	num=n_stars)
	# logTe = np.linspace(
	# 	start=mlp_phot.domain["logTe"][0],
	# 	stop=mlp_phot.domain["logTe"][1],
	# 	num=n_stars)
	# print(logAge.min(),logAge.max())
	# print(logL.min(),logL.max())
	# print(n_stars)
	# print(Mini.min(),Mini.max())
	# print(logTe.min(),logTe.max())
	#------------------------------------------------


	phot = mlp_phot(
		logAge=logAge,
		covariate=logL,
		n_stars=n_stars)

	mass = mlp_mass(
		logAge=logAge,
		covariate=logL,
		n_stars=n_stars)

	logl = mlp_logl(
		logAge=logAge,
		covariate=Mini,
		n_stars=n_stars)

	df_phot = pn.DataFrame(data=phot.eval(),columns=mlp_phot.bands)
	df_phot["logAge"] = logAge
	df_phot["logL"]   = logL
	
	df_phot.set_index(features,inplace=True)
	df_iso.set_index(features,inplace=True)
	
	df_ph = pn.merge(left=df_iso,right=df_phot,
						left_index=True,
						right_index=True,
						suffixes=["_tst","_prd"])

	for trgt in bands:
		df_ph[trgt] = df_ph.apply(lambda x: (x[trgt+"_prd"]-x[trgt+"_tst"]),axis=1)

	print(np.sqrt(np.square(df_ph.loc[:,bands]).mean(axis=0)))

	df = df_ph.loc[:,bands].stack().reset_index()
	print(df)
	df.columns = sum([features,["target","value"]],[])
	print(df.describe())

	#-------------- Error --------------------------
	fig, ax = plt.subplots(1, 1, figsize=(16, 8))
	ax = sns.scatterplot(data=df,
						x="logL",
						y="value",
						hue="target",
						zorder=0)
	ax.set_xlabel("logL")
	ax.set_ylabel("Diff [mag]")
	plt.legend(bbox_to_anchor=(1.01, 0.5),
						loc="center left")
	plt.savefig(file_plt,dpi=300)
	plt.close()
	#------------------------------------------------