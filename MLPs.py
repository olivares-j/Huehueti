"""Pre-Trained Neural Network as BT-Settl model interpolator. PyTensor implementation."""
import os
from pickle import load
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import warnings
import pandas as pn

import matplotlib.pyplot as plt
import seaborn as sns

from Functions import relu

class MLP:
	"""Pre-Trained Neural Network weights and compute the predictions.
	Implementation in PyTensor.
	"""
	def __init__(self, 
		file_mlp : str = 'mlp_iso_BT-Settl.pkl'
		):
		"""Load optimal weights and scalers from trained MLP. 
		
		Parameters
		----------
		file_mlp : str
			Folder containing the scaler and MLP weights.
		"""
		
		# Load weights
		try:
			with open(file_mlp, 'rb') as file:
				mlp = load(file)
				weights = mlp["weights"]
				scalers = mlp["scalers"]
				self.phot_min = mlp["phot_min"]
				self.age_domain  = mlp["age_domain"]
				self.theta_domain = mlp["par_domain"]

		except FileNotFoundError as error:
			raise FileNotFoundError("The file containing optimal weights and scalers cannot be found. Please, provide a valid path") from error
		else:
			self.W1 = weights[0]
			self.b1 = weights[1]
			self.W2 = weights[2]
			self.b2 = weights[3]
			self.W3 = weights[4]
			self.b3 = weights[5]
			self.W4 = weights[6]
			self.b4 = weights[7]
			self.W5 = weights[8]
			self.b5 = weights[9]
			self.W6 = weights[10]
			self.b6 = weights[11]
			self.W7 = weights[12]
			self.b7 = weights[13]
			self.W8 = weights[14]
			self.b8 = weights[15]
			self.W9 = weights[16]
			self.b9 = weights[17]
			self.W10 = weights[18]
			self.b10 = weights[19]
			self.WPho = weights[20]
			self.bPho = weights[21]

			self.BoxCox = scalers[0]
			self.MinMax = scalers[1]

			# BoxCox transformation
			self.lamb = np.array(self.BoxCox.lambdas_).reshape(-1, 1)
			
			# MinMax transformation
			self.min_val = self.MinMax.data_min_.reshape(-1, 1)
			self.max_val = self.MinMax.data_max_.reshape(-1, 1)
			self.dff_val = self.max_val - self.min_val

		print("Age domain: [{0:2.1f},{1:2.1f}]".format(*self.age_domain))
		print("Theta domain: [{0:2.2f},{1:2.3f}]".format(*self.theta_domain))

	def __call__(self, age, theta, n_stars):
		"""Neural Network predictions as interpolation function for BTSettl model.
		PyTensor implementation.

		Parameters
		---------- 
		inputs : Tensor (pytensor)     
			Scaled age and mass as 2-dimensional tensor. 
			- age = inputs[0]
			- theta = inputs[1]

		Returns
		-------
		mass : Tensor (pytensor)
			Mass of each star. Units: [1]
		phot : Tensor (pytensor)
			Photometry (absolute magnitude) for each star. Units: [1]
		"""
			
		amt = pt.stack([pt.tile(age, n_stars),theta])

		x = pt.switch(pt.neq(self.lamb, 0), (amt ** self.lamb - 1) / self.lamb, pt.log(amt))
		inputs = ((x - self.min_val) /self.dff_val).T

		# Forward pass
		Z1 = pt.dot(inputs, self.W1) + self.b1
		A1 = relu(Z1)
		Z2 = pt.dot(A1, self.W2) + self.b2
		A2 = relu(Z2)
		Z3 = pt.dot(A2, self.W3) + self.b3
		A3 = relu(Z3)
		Z4 = pt.dot(A3, self.W4) + self.b4
		A4 = relu(Z4)
		Z5 = pt.dot(A4, self.W5) + self.b5
		A5 = relu(Z5)
		Z6 = pt.dot(A5, self.W6) + self.b6
		A6 = relu(Z6)
		Z7 = pt.dot(A6, self.W7) + self.b7
		A7 = relu(Z7)
		Z8 = pt.dot(A7, self.W8) + self.b8
		A8 = relu(Z8)
		Z9 = pt.dot(A8, self.W9) + self.b9
		A9 = relu(Z9)
		Z10 = pt.dot(A9, self.W10) + self.b10
		A10 = relu(Z10)
		targets = pt.dot(A10, self.WPho) + self.bPho


		mass = targets[:,0]
		photometry = targets[:,1:]

		return mass,photometry


if __name__ == "__main__":

	dir_base = "/home/jolivares/Repos/SakamII/"
	dir_data = dir_base + "data/"

	file_iso = dir_data + "parametrizations/parametrized_PARSEC_GEDR3+2MASS+PanSTARRS.csv"
	file_mlp = dir_base + "mlps/PARSEC_10x96/mlp.pkl"


	phot_names = ['G','G_BP','G_RP','gP1','rP1','iP1','zP1','yP1','J','H','K']
	mlp = MLP(file_mlp=file_mlp)

	# masses = np.array([0.14,0.14])
	# teffs  = np.array([9150.,9150.])
	# photometry = mlp_iso(145.,masses,teffs,132.0,2)
	# print(photometry.eval())
	# sys.exit()

	age = 120.

	df = pn.read_csv(file_iso)
	df_iso = df.groupby("age_Myr").get_group(age).copy()
	df_iso["color"] = df_iso.apply(lambda x: x["G"]-x["G_RP"],axis=1)

	theta  = df_iso["parameter"].to_numpy()
	n_stars = len(theta) 

	mass,absolute_photometry = mlp(age,theta,n_stars)
	df_prd = pn.DataFrame(data=absolute_photometry.eval(),columns=phot_names)
	df_prd["Mass"] = mass.eval()
	df_prd["color"] = df_prd.apply(lambda x: x["G"]-x["G_RP"],axis=1)

	plt.figure(0)
	ax = sns.scatterplot(data=df_iso,
						x="color",y="G",
						legend=True,
						zorder=0)
	ax = sns.lineplot(data=df_prd,
						x="color",y="G",
						legend=True,
						sort=False,
						zorder=1,
						ax=ax)

	ax.set_xlabel("G-RP [mag]")
	ax.set_ylabel("G [mag]")
	ax.invert_yaxis()
	plt.show()
		