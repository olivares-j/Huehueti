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

def Phot_band(X,W,b,mu,sd):
	A1  = sigmoid(pt.dot(X,  W[0]) + b[0])
	A2  = sigmoid(pt.dot(A1, W[1]) + b[1])
	# A3  = sigmoid(pt.dot(A2, W[2]) + b[2])
	# A4  = sigmoid(pt.dot(A3, W[3]) + b[3])
	# A5  = sigmoid(pt.dot(A4, W[4]) + b[4])
	# A6  = sigmoid(pt.dot(A5, W[5]) + b[5])
	# A7  = sigmoid(pt.dot(A6, W[6]) + b[6])
	# Final linear output (no activation)
	out = pt.dot(A2,W[2]) + b[2]
	# out = pt.dot(A3,W[3]) + b[3]
	# out = pt.dot(A4,W[4]) + b[4]

	return out.flatten()

class MLP_phot:
	"""Wrapper around a pretrained multilayer perceptron.
	"""

	def __init__(self,
		files_mlps: dict,
		features = ["logAge","Mini"],
		targets = ["G_BPmag","Gmag","G_RPmag"],
		):
		"""Load the mlps for each photometric band
		"""
		# Load weights and scalers from provided file path
		assert list(files_mlps.keys()) == targets,"Error in keys! Must be equal to targets!"
		self.features = features
		self.targets = targets
		self.bands = {}
		domains = []
		mus_features = []
		sds_features = []
		for name,file_mlp in files_mlps.items():
			print("Reading band: {0}".format(name))
			assert os.path.exists(file_mlp),"The file containing optimal weights and scalers cannot be found. Please, provide a valid path"
			band = {}
			

			with open(file_mlp, 'rb') as file:
				tmp = load(file)
				tmp_weights   = tmp["weights"]
				tmp_num_lyrs  = tmp["num_layers"]
				tmp_sze_lyrs  = tmp["size_layers"]
				tmp_targets   = tmp["targets"]
				tmp_features  = tmp["features"]
				tmp_phot_min  = tmp["phot_min"]
				tmp_domain    = tmp["domain"]
				tmp_mu        = tmp["mu_transform"]
				tmp_sd        = tmp["sd_transform"]

			assert tmp_features == features, "Features mismatch! Expected: {0}".format(features)
			assert tmp_targets[0] in targets, "Target mismatch! Expected on of {0}".format(targets)
			assert tmp_num_lyrs == 2, "Error: mismatch in number of layers! Expected 4"

			band["W"] = tmp_weights[::2]
			band["b"] = tmp_weights[1::2]
			band["mu"] = tmp_mu.loc[tmp_targets].to_numpy()
			band["sd"] = tmp_sd.loc[tmp_targets].to_numpy()
			band["min"] = tmp_phot_min

			domains.append(tmp_domain)
			mus_features.append(tmp_mu.loc[features].to_numpy())
			sds_features.append(tmp_sd.loc[features].to_numpy())

			self.bands[name] = band

		domain = domains[0]
		for tmp_domain in domains[1:]:
			assert tmp_domain == domain,"Domain mismatch!"

		mu_features = mus_features[0]
		for tmp_mu_features in mus_features[1:]:
			np.testing.assert_equal(tmp_mu_features,mu_features,
				err_msg="mu_features mismatch!")

		sd_features = sds_features[0]
		for tmp_sd_features in sds_features[1:]:
			np.testing.assert_equal(tmp_sd_features,sd_features,
				err_msg="sd_features mismatch!")

		self.domain = domain
		self.mu_features = mu_features
		self.sd_features = sd_features

	def __call__(self, logAge, covariate, n_stars):
		"""Compute NN predictions for given age and theta grid.
		"""

		# x = pt.stack([pt.tile(logAge, (n_stars,)), covariate,logTe],axis=1)
		x = pt.stack([pt.tile(logAge, (n_stars,)), covariate],axis=1)
		# x = pt.stack([logAge, logL, logTe],axis=1)

		A0 = (x - self.mu_features)/self.sd_features


		G_BPmag = Phot_band(X=A0,W=self.bands["G_BPmag"]["W"],b=self.bands["G_BPmag"]["b"],
						mu=self.bands["G_BPmag"]["mu"],sd=self.bands["G_BPmag"]["sd"])

		Gmag = Phot_band(X=A0,W=self.bands["Gmag"]["W"],b=self.bands["Gmag"]["b"],
						mu=self.bands["Gmag"]["mu"],sd=self.bands["Gmag"]["sd"])

		G_RPmag = Phot_band(X=A0,W=self.bands["G_RPmag"]["W"],b=self.bands["G_RPmag"]["b"],
						mu=self.bands["G_RPmag"]["mu"],sd=self.bands["G_RPmag"]["sd"])

		return pt.stack([G_BPmag,Gmag,G_RPmag],axis=1)

# The block below is an example usage / quick visual test when running the file
# directly. It is not required for the library functionality and will only run
# in interactive/script mode.
if __name__ == "__main__":
	import bisect
	rng = np.random.default_rng(42)

	dir_mlps = "/home/jolivares/Models/PARSEC/20-220Myr/"
	case = "Optuna_logAge_logL_epochs_5e+02_0.1myr"

	file_iso      = dir_mlps + "Gaia_EDR3_0.1myr.dat"
	files_mlps = {
	"G_BPmag":dir_mlps + case +"/G_BPmag_l2/seed_0/mlp.pkl",
	"Gmag":   dir_mlps + case +"/Gmag_l2/seed_0/mlp.pkl",
	"G_RPmag":dir_mlps + case +"/G_RPmag_l2/seed_0/mlp.pkl"
	}
	targets = ["G_BPmag","Gmag","G_RPmag"]
	features = ["logAge","logL"]

	mlp_phot = MLP_phot(
		features=features,
		targets=targets,
		files_mlps=files_mlps)

	# Example: load an isochrone from a parametrized CSV and overlay predicted photometry
	
	logAge = 7.77815 #60 Myr
	# logAge = 8.0 #100Myr
	# logAge = 8.14613 #140Myr
	max_label = 1

	df_iso = pn.read_csv(file_iso,
					skiprows=13,
					delimiter=r"\s+",
					header="infer",
					comment="#")
	df_iso = df_iso.loc[df_iso["label"]<= max_label]
	df_iso = df_iso.loc[:,sum([features,targets],[])]
	df_iso = df_iso.groupby("logAge").get_group(logAge)
	n_stars = df_iso.shape[0]

	#------------ logL and logTe ------------------------------
	# logAge = df_iso["logAge"].to_numpy()
	logL  = df_iso["logL"].to_numpy()
	
	# Mini  = df_iso["Mini"].to_numpy()
	# logTe = df_iso["logTe"].to_numpy()
	# logL = np.linspace(
	# 	start=mlp_phot.domain["logL"][0],
	# 	stop=mlp_phot.domain["logL"][1],
	# 	num=n_stars)
	# logTe = np.linspace(
	# 	start=mlp_phot.domain["logTe"][0],
	# 	stop=mlp_phot.domain["logTe"][1],
	# 	num=n_stars)
	print(logL.min(),logL.max())
	# print(Mini.min(),Mini.max())
	# print(logTe.min(),logTe.max())
	#------------------------------------------------

	phot = mlp_phot(
		logAge=logAge,
		covariate=logL,
		# covariate=Mini,
		# logTe=logTe,
		n_stars=n_stars).eval()

	df_prd = pn.DataFrame(data=phot,columns=mlp_phot.targets)
	df_prd["logAge"] = logAge
	df_prd["logL"]   = logL
	# df_prd["Mini"]   = Mini
	# df_prd["logTe"]  = logTe

	df_prd.set_index(features,inplace=True)
	df_iso.set_index(features,inplace=True)

	df = pn.merge(left=df_iso,right=df_prd,
						left_index=True,
						right_index=True,
						suffixes=["_tst","_prd"])

	for trgt in targets:
		df[trgt] = df.apply(lambda x: (x[trgt+"_prd"]-x[trgt+"_tst"]),axis=1)

	print(np.sqrt(np.square(df.loc[:,targets]).mean(axis=0)))

	df = df.loc[:,targets].stack().reset_index()
	df.columns = sum([features,["target","value"]],[])
	print(df.describe())

	#-------------- Error --------------------------
	fig, ax = plt.subplots(1, 1, figsize=(16, 8))
	ax = sns.scatterplot(data=df,
						x=features[1],
						y="value",
						hue="target",
						zorder=0)
	ax.set_xlabel(features[1])
	ax.set_ylabel("Diff [mag]")
	plt.legend(bbox_to_anchor=(1.01, 0.5),
						loc="center left")
	plt.show()
	plt.close()
	sys.exit()
	#------------------------------------------------



	# Simple diagnostic plot to visually compare trained MLP predictions with input isochrone
	for target in mlp_phot.targets:
		plt.figure(0)
		ax = sns.scatterplot(data=df_iso,
							x=features[1],y=target,
							legend=True,
							zorder=0)
		ax = sns.lineplot(data=df_prd,
							x=features[1],y=target,
							legend=True,
							sort=False,
							zorder=1,
							ax=ax)

		ax.set_xlabel(features[1])
		ax.set_ylabel(target)
		plt.show()