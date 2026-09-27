import sys
import os
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Concatenate, Lambda
from keras.models import Model
from tensorflow.keras import regularizers

SIGMA_FLOOR = 1.0e-6
SIGMA_INIT = 3.0e-3

def _inverse_softplus(x):
	return np.log(np.expm1(x))

def heteroscedastic_gaussian_nll(y_true, y_pred):
	n_targets = tf.shape(y_true)[-1]
	mu = y_pred[:, :n_targets]
	sigma = tf.maximum(y_pred[:, n_targets:], SIGMA_FLOOR)
	residual = y_true - mu
	log_sigma = tf.math.log(sigma)
	nll = 0.5 * (tf.square(residual / sigma) + 2.0 * log_sigma
		+ tf.math.log(tf.constant(2.0 * np.pi, dtype=y_pred.dtype)))
	return tf.reduce_mean(nll, axis=-1)

def photometric_rmse(y_true, y_pred):
	n_targets = tf.shape(y_true)[-1]
	mu = y_pred[:, :n_targets]
	return tf.sqrt(tf.reduce_mean(tf.square(y_true - mu), axis=-1))


SEED = 42
tf.random.set_seed(SEED)

print(f"Running Tensoflow {tf.__version__}")

def evaluate_gradient(model,x):
	x = tf.convert_to_tensor(x)

	with tf.GradientTape() as tape:
		tape.watch(x)
		y = model(x, training=False)
	grads = tape.gradient(y,x)
	return grads

# Defining model archiquetures
# ----------------------------
def create_custom_model(
	input_shape: int,
	output_shape: int,
	num_layers: int,
	size_layers: int,
	activation_layers: str,
	activation_output: str = "linear",
	seed: int = 0,
) -> Model:
	"""Create a heteroscedastic ANN returning [mu, sigma]."""
	if activation_layers == "sigmoid":
		initializer = keras.initializers.GlorotUniform(seed=seed)
	elif activation_layers == "relu":
		initializer = keras.initializers.HeUniform(seed=seed)
	else:
		sys.exit("activation_layers not recognized!")

	inputs = keras.Input(shape=(input_shape,))
	x = inputs
	for _ in range(num_layers):
		x = Dense(size_layers, activation=activation_layers,
			kernel_initializer=initializer, bias_initializer=initializer)(x)

	mu = Dense(output_shape, activation=activation_output,
		kernel_initializer=initializer, bias_initializer=initializer,
		name="photometry_mean")(x)

	sigma_bias = keras.initializers.Constant(_inverse_softplus(SIGMA_INIT))
	raw_sigma = Dense(output_shape, activation=None,
		kernel_initializer=initializer, bias_initializer=sigma_bias,
		name="photometry_sigma_raw")(x)
	sigma = Lambda(lambda z: tf.nn.softplus(z) + SIGMA_FLOOR,
		name="photometry_sigma")(raw_sigma)

	return Model(inputs=inputs,
		outputs=Concatenate(name="photometry_and_sigma")([mu, sigma]))


def compile_model(model,
	lr_schedule,
	beta_1=0.9,
	beta_2=0.999,
	loss="heteroscedastic_gaussian_nll",
	metrics=None,
	clipnorm: float = 1.0,
	use_ema=False,
	):
	"""Compile the heteroscedastic ANN with sample-weight-compatible loss."""
	if metrics is None:
		metrics = ["photometric_rmse"]

	loss_map = {
		"heteroscedastic_gaussian_nll": heteroscedastic_gaussian_nll,
		"gaussian_nll": heteroscedastic_gaussian_nll,
		"mae": "mae",
		"mean_squared_error": "mean_squared_error",
	}
	metric_map = {
		"photometric_rmse": photometric_rmse,
		"root_mean_squared_error": photometric_rmse,
	}
	resolved_loss = loss_map.get(loss, loss)
	resolved_metrics = [metric_map.get(metric, metric) for metric in metrics]

	model.compile(
		optimizer=keras.optimizers.Adam(
			learning_rate=lr_schedule,
			beta_1=beta_1,
			beta_2=beta_2,
			clipnorm=clipnorm,
			use_ema=use_ema
		),
		loss=resolved_loss,
		metrics=resolved_metrics,
	)
	return model


def learning_rate_scheduler(
	lr_decay_function="ExponentialDecay",
	initial_learning_rate: float = 1e-3,
	decay_steps: int = 1000,
	decay_rate = 1.0,
	alpha = 1.0,
	end_learning_rate: float = 1e-3,
	power = 1.0,
	boundaries = [500],
	values = [1e-2,1e-3],
	):
	if lr_decay_function == "ExponentialDecay":
		lr_schedule = keras.optimizers.schedules.ExponentialDecay(
						initial_learning_rate=initial_learning_rate,
						decay_steps=decay_steps,
						decay_rate=decay_rate,
						)
	elif lr_decay_function == "InverseTimeDecay":
		lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
						initial_learning_rate=initial_learning_rate,
						decay_steps=decay_steps,
						decay_rate=decay_rate,
						staircase=False,
						)
	elif lr_decay_function == "CosineDecay":
		lr_schedule = keras.optimizers.schedules.CosineDecay(
						initial_learning_rate=initial_learning_rate,
						decay_steps=decay_steps,
						alpha=alpha
						)
	elif lr_decay_function == "PolynomialDecay":
		lr_schedule = keras.optimizers.schedules.PolynomialDecay(
						initial_learning_rate,
						decay_steps=decay_steps,
						end_learning_rate=end_learning_rate,
						power=power
						)
	
	elif lr_decay_function == "PiecewiseConstantDecay":
		lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
						boundaries=boundaries,
						values=values
						)
	elif lr_decay_function == "NoDecay":
		lr_schedule = final_learning_rate
	else:
		sys.exit("Unrecognized decay function!")

	return lr_schedule



# ---------------- Residual analysis -----------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
def analyze_residuals(model, x_data, y_data, df_original,features,targets, case,
	file_res,
	file_plt_res,
	file_plt_res2d):
	"""
	Evaluate the ANN on a data subset and save residuals and diagnostic plots.

	Residual definition:
		residual = prediction - target

	The plots are made in the original input coordinates (logAge, logL).
	The targets are already in their original units because the current
	forward_transform() only standardizes the input features.
	"""
	n_targets = len(targets)
	x_array = x_data.to_numpy()
	y_array = y_data.to_numpy()

	y_pred_full = model.predict(x_array, verbose=0)
	# Six outputs are [mu, sigma]; residual diagnostics use the mean head.
	y_pred = y_pred_full[:, :n_targets]
	residual = y_pred - y_array

	# Store all relevant quantities in one table.
	df_res = df_original.loc[y_data.index, features + targets].copy()

	for i, target in enumerate(targets):
		df_res[f"pred_{target}"] = y_pred[:, i]
		df_res[f"res_{target}"] = residual[:, i]
		df_res[f"abs_res_{target}"] = np.abs(residual[:, i])

	df_res.to_csv(file_res, index=False)

	#------ Compute the covariance matrix of the residuals ----------
	res = df_res.loc[:,[f"res_{target}" for target in targets]].to_numpy()
	covariance_res = np.cov(res, rowvar=False)
	#---------------------------------------------------

	# Print useful numerical diagnostics.
	print(f"\nResidual diagnostics: {case}")
	for i, target in enumerate(targets):
		rms = np.sqrt(np.mean(residual[:, i] ** 2))
		mae = np.mean(np.abs(residual[:, i]))
		p95 = np.percentile(np.abs(residual[:, i]), 95)
		bias = np.mean(residual[:, i])

		print(
			f"  {target:10s}: "
			f"RMSE={rms:.6g}, "
			f"MAE={mae:.6g}, "
			f"P95={p95:.6g}, "
			f"bias={bias:.6g}"
		)

	# Figure 1: residual distribution and residuals versus each input feature.
	fig, axes = plt.subplots(
		n_targets, 3,
		figsize=(18, 5 * n_targets),
		squeeze=False
	)

	for i, target in enumerate(targets):
		r = residual[:, i]

		# Residual distribution.
		sns.histplot(r, bins=80, kde=True, ax=axes[i, 0])
		axes[i, 0].axvline(0.0, linestyle="--", linewidth=1.5)
		axes[i, 0].set_xlabel(f"{target} residual [prediction - target]")
		axes[i, 0].set_ylabel("Number of objects")

		# Residual versus logAge.
		axes[i, 1].scatter(
			df_original.loc[y_data.index, "logAge"],
			r,
			s=4,
			alpha=0.35,
			rasterized=True
		)
		axes[i, 1].axhline(0.0, linestyle="--", linewidth=1.5)
		axes[i, 1].set_xlabel("logAge")
		axes[i, 1].set_ylabel(f"{target} residual")

		# Residual versus logL.
		axes[i, 2].scatter(
			df_original.loc[y_data.index, "logL"],
			r,
			s=4,
			alpha=0.35,
			rasterized=True
		)
		axes[i, 2].axhline(0.0, linestyle="--", linewidth=1.5)
		axes[i, 2].set_xlabel("logL")
		axes[i, 2].set_ylabel(f"{target} residual")

	fig.suptitle(f"{case} residual diagnostics", fontsize=16)
	fig.tight_layout()
	fig.savefig(file_plt_res, dpi=300)
	plt.close(fig)

	# Figure 2: residual maps in the two-dimensional input space.
	n_targets = len(targets)

	# Global colour scale across all targets
	vmax = np.nanmax(np.abs(residual))
	vmin = -vmax

	fig, axes = plt.subplots(
	    n_targets,
	    1,
	    figsize=(10, 10),
	    sharey=True,
	    squeeze=False
	)
	axes = axes.ravel()

	for i, target in enumerate(targets):
	    r = residual[:, i]
	    ax = axes[i]

	    sc = ax.scatter(
	        df_original.loc[y_data.index, "logAge"],
	        df_original.loc[y_data.index, "logL"],
	        c=r,
	        s=8,
	        alpha=0.6,
	        rasterized=True,
	        cmap="RdBu_r",
	        vmin=vmin,
	        vmax=vmax
	    )

	    ax.set_xlabel("logAge")
	    ax.set_title(f"{case}: {target} residual in input space")

	axes[0].set_ylabel("logL")

	# One common colorbar
	cbar = fig.colorbar(
	    sc,
	    ax=axes,
	    label="Residual [prediction - target]",
	    pad=0.1
	)

	fig.tight_layout()

	fig.savefig(file_plt_res2d,dpi=300
	)

	plt.close(fig)

	return covariance_res
# -----------------------------------------------------------------------------
