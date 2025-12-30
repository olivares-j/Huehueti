"""Pre-Trained Neural Network as BT-Settl model interpolator. PyTensor implementation.

This module wraps a pre-trained feed-forward neural network used to interpolate
stellar model outputs (mass and absolute photometry) as a function of age and a
single stellar parameter (theta). The network parameters and scalers are read
from a dill/pickle file and the forward pass is implemented using PyTensor
(pytensor) tensors so the outputs can be used inside PyMC models.
"""
import os
from pickle import load
import numpy as np
import pytensor.tensor as pt
import pandas as pn

import matplotlib.pyplot as plt
import seaborn as sns

def relu(x, alpha=0):
    """
    Compute the element-wise rectified linear activation function.

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : `scalar or tensor, optional`
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.

    Notes
    -----
    This is numerically equivalent to ``pt.switch(x > 0, x, alpha * x)``
    (or ``pt.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.

    """
    if alpha == 0:
        return 0.5 * (x + abs(x))
    else:
        # We can't use 0.5 and 1 for one and half.  as if alpha is a
        # numpy dtype, they will be considered as float64, so would
        # cause upcast to float64.
        alpha = pt.as_tensor_variable(alpha)
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * abs(x)

class MLP_phot:
	"""Wrapper around a pretrained multilayer perceptron.

	Responsibilities:
	- load NN weights and preprocessing scalers from disk
	- expose a callable that accepts (age, theta, n_stars) and returns
	  PyTensor expressions for mass and photometry suitable for inclusion
	  inside probabilistic models.

	Notes on shapes and conventions used in this code:
	- age: scalar (float) or PyTensor scalar
	- theta: 1-D array-like of length n_stars (PyTensor or numpy)
	- n_stars: integer number of samples in theta
	- The implementation stacks age and theta into a 2xN array and applies
	  the same preprocessing (BoxCox + MinMax) used at training time.
	- The network returns an array "targets" where the first output column
	  corresponds to stellar mass and the remaining columns correspond to
	  absolute magnitudes (photometry) in the trained band order.
	"""

	def __init__(self, 
		file_mlp : str = 'mlp_iso_BT-Settl.pkl'
		):
		"""Load optimal weights and scalers from trained MLP.

		The serialized file is expected to contain a dict-like object with keys:
		  - "weights": list/array of weight and bias matrices in the expected order
		  - "scalers": list of sklearn-like scalers [BoxCox, MinMax]
		  - "phot_min": per-band lower limit used elsewhere for filtering
		  - "age_domain": (min_age, max_age)
		  - "par_domain" (theta domain)
		"""
		# Load weights and scalers from provided file path
		try:
			with open(file_mlp, 'rb') as file:
				mlp = load(file)
				weights = mlp["weights"]
				scalers = mlp["scalers"]
				self.targets = mlp["targets"]
				self.phot_min = mlp["phot_min"]
				self.age_domain  = mlp["age_domain"]
				# theta (parameter) domain used by the MLP
				self.theta_domain = mlp["par_domain"]

		except FileNotFoundError as error:
			# Provide a clear error for the user if file missing
			raise FileNotFoundError("The file containing optimal weights and scalers cannot be found. Please, provide a valid path") from error
		else:
			# Unpack weights/biases according to the trained network layout.
			# The code expects exactly this many items and ordering.
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
			# Final linear layer mapping to output photometry (and mass)
			self.WPho = weights[20]
			self.bPho = weights[21]

			# Scalers used for preprocessing the inputs
			self.BoxCox = scalers[0]
			self.MinMax = scalers[1]

			# Box-Cox lambdas are stored per input dimension and reshaped to column vectors.
			self.lamb = np.array(self.BoxCox.lambdas_).reshape(-1, 1)
			
			# MinMax scaling metadata (data_min_, data_max_) used to scale inputs.
			self.min_val = self.MinMax.data_min_.reshape(-1, 1)
			self.max_val = self.MinMax.data_max_.reshape(-1, 1)
			self.dff_val = self.max_val - self.min_val

		# Print a small summary for user awareness when instantiating the MLP.
		print("Age domain: [{0:2.1f},{1:2.1f}]".format(*self.age_domain))
		print("Theta domain: [{0:2.2f},{1:2.3f}]".format(*self.theta_domain))

	def __call__(self, age, theta, n_stars):
		"""Compute NN predictions for given age and theta grid.

		Returns:
		  - mass: PyTensor vector of length n_stars
		  - photometry: PyTensor matrix shape (n_stars, n_bands)

		Detailed steps:
		1. Stack age and theta into a 2xN tensor so each column corresponds to one star.
	 2. Apply Box-Cox transform (or log for lambda==0) elementwise.
	 3. Apply MinMax scaling using stored min/max values.
	 4. Run forward pass through 10 hidden layers using relu activations.
	 5. Final linear mapping produces targets: [mass, phot_band_1, phot_band_2, ...].
		"""
		# Stack age and tiled age with theta to form input columns:
		# amt has shape (2, n_stars) using pytensor operations.
		amt = pt.stack([pt.tile(age, n_stars),theta])

		# Apply Box-Cox transform: (x^lambda - 1)/lambda, and log if lambda==0.
		# The comparison and switch is done elementwise using pytensor operations.
		x = pt.switch(pt.neq(self.lamb, 0), (amt ** self.lamb - 1) / self.lamb, pt.log(amt))

		# Normalize using MinMax stored values and transpose to shape (n_stars, n_features)
		# so that a forward pass with weight matrices (features x hidden) works naturally.
		inputs = ((x - self.min_val) / self.dff_val).T

		# Forward pass through the network: linear -> relu -> linear -> relu -> ...
		# Each Z* is the pre-activation and A* the post-activation (relu).
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
		# Final linear output (no activation) gives targets for mass + photometry bands
		targets = pt.dot(A10, self.WPho) + self.bPho

		return targets

class MLP_mass:
	"""Wrapper around a pretrained multilayer perceptron.

	Responsibilities:
	- load NN weights and preprocessing scalers from disk
	- expose a callable that accepts (age, theta, n_stars) and returns
	  PyTensor expressions for mass and photometry suitable for inclusion
	  inside probabilistic models.

	Notes on shapes and conventions used in this code:
	- age: scalar (float) or PyTensor scalar
	- theta: 1-D array-like of length n_stars (PyTensor or numpy)
	- n_stars: integer number of samples in theta
	- The implementation stacks age and theta into a 2xN array and applies
	  the same preprocessing (BoxCox + MinMax) used at training time.
	- The network returns an array "targets" where the first output column
	  corresponds to stellar mass and the remaining columns correspond to
	  absolute magnitudes (photometry) in the trained band order.
	"""

	def __init__(self, 
		file_mlp : str = 'mlp_iso_BT-Settl.pkl'
		):
		"""Load optimal weights and scalers from trained MLP.

		The serialized file is expected to contain a dict-like object with keys:
		  - "weights": list/array of weight and bias matrices in the expected order
		  - "scalers": list of sklearn-like scalers [BoxCox, MinMax]
		  - "phot_min": per-band lower limit used elsewhere for filtering
		  - "age_domain": (min_age, max_age)
		  - "par_domain" (theta domain)
		"""
		# Load weights and scalers from provided file path
		try:
			with open(file_mlp, 'rb') as file:
				mlp = load(file)
				weights = mlp["weights"]
				scalers = mlp["scalers"]
				self.targets = mlp["targets"]
				self.age_domain  = mlp["age_domain"]
				# theta (parameter) domain used by the MLP
				self.theta_domain = mlp["par_domain"]

		except FileNotFoundError as error:
			# Provide a clear error for the user if file missing
			raise FileNotFoundError("The file containing optimal weights and scalers cannot be found. Please, provide a valid path") from error
		else:
			# Unpack weights/biases according to the trained network layout.
			# The code expects exactly this many items and ordering.
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
			self.Wout = weights[20]
			self.bout = weights[21]

			# Scalers used for preprocessing the inputs
			self.BoxCox = scalers[0]
			self.MinMax = scalers[1]

			# Box-Cox lambdas are stored per input dimension and reshaped to column vectors.
			self.lamb = np.array(self.BoxCox.lambdas_).reshape(-1, 1)
			
			# MinMax scaling metadata (data_min_, data_max_) used to scale inputs.
			self.min_val = self.MinMax.data_min_.reshape(-1, 1)
			self.max_val = self.MinMax.data_max_.reshape(-1, 1)
			self.dff_val = self.max_val - self.min_val

		# Print a small summary for user awareness when instantiating the MLP.
		print("Age domain: [{0:2.1f},{1:2.1f}]".format(*self.age_domain))
		print("Theta domain: [{0:2.2f},{1:2.3f}]".format(*self.theta_domain))

	def __call__(self, age, theta, n_stars):
		"""Compute NN predictions for given age and theta grid.

		Returns:
		  - mass: PyTensor vector of length n_stars
		  - photometry: PyTensor matrix shape (n_stars, n_bands)

		Detailed steps:
		1. Stack age and theta into a 2xN tensor so each column corresponds to one star.
	 2. Apply Box-Cox transform (or log for lambda==0) elementwise.
	 3. Apply MinMax scaling using stored min/max values.
	 4. Run forward pass through 10 hidden layers using relu activations.
	 5. Final linear mapping produces targets: [mass, phot_band_1, phot_band_2, ...].
		"""
		# Stack age and tiled age with theta to form input columns:
		# amt has shape (2, n_stars) using pytensor operations.
		amt = pt.stack([pt.tile(age, n_stars),theta])

		# Apply Box-Cox transform: (x^lambda - 1)/lambda, and log if lambda==0.
		# The comparison and switch is done elementwise using pytensor operations.
		x = pt.switch(pt.neq(self.lamb, 0), (amt ** self.lamb - 1) / self.lamb, pt.log(amt))

		# Normalize using MinMax stored values and transpose to shape (n_stars, n_features)
		# so that a forward pass with weight matrices (features x hidden) works naturally.
		inputs = ((x - self.min_val) / self.dff_val).T

		# Forward pass through the network: linear -> relu -> linear -> relu -> ...
		# Each Z* is the pre-activation and A* the post-activation (relu).
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
		# Final linear output (no activation) gives targets for mass + photometry bands
		targets = pt.dot(A10, self.Wout) + self.bout

		return targets


# The block below is an example usage / quick visual test when running the file
# directly. It is not required for the library functionality and will only run
# in interactive/script mode.
if __name__ == "__main__":

	dir_base = "/home/jolivares/Repos/Huehueti/"

	file_iso = dir_base + "data/parametrizations/parametrized_max_label_1_PARSEC_20-200Myr_GDR3+PanSTARRS+2MASS.csv"
	file_mlp_phot = dir_base + "mlps/PARSEC_phot_10x20/mlp.pkl"
	file_mlp_mass = dir_base + "mlps/PARSEC_mass_10x100/mlp.pkl"

	mlp_phot = MLP_phot(file_mlp=file_mlp_phot)
	mlp_mass = MLP_mass(file_mlp=file_mlp_mass)

	# Example: load an isochrone from a parametrized CSV and overlay predicted photometry
	age = 50.

	df = pn.read_csv(file_iso)
	df_iso = df.groupby("age_Myr").get_group(age).copy()

	theta  = df_iso["parameter"].to_numpy()
	n_stars = len(theta) 

	absolute_photometry = mlp_phot(age,theta,n_stars)
	mass_logTe_logg = mlp_mass(age,theta,n_stars)
	df_phot = pn.DataFrame(data=absolute_photometry.eval(),columns=mlp_phot.targets)
	df_mass = pn.DataFrame(data=mass_logTe_logg.eval(),columns=mlp_mass.targets)
	df_prd = df_phot.join(df_mass)
	df_prd["parameter"] = theta

	# Simple diagnostic plot to visually compare trained MLP predictions with input isochrone
	for target in sum([mlp_phot.targets,mlp_mass.targets],[]):
		plt.figure(0)
		ax = sns.scatterplot(data=df_iso,
							x="parameter",y=target,
							legend=True,
							zorder=0)
		ax = sns.lineplot(data=df_prd,
							x="parameter",y=target,
							legend=True,
							sort=False,
							zorder=1,
							ax=ax)

		ax.set_xlabel("Theta")
		ax.set_ylabel(target)
		plt.show()