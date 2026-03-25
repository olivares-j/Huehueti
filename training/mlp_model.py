import sys
import os
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras import regularizers

SEED = 42
tf.random.set_seed(SEED)

print(f"Running Tensoflow {tf.__version__}")

def compute_gradients(model,x,y):
	grd, vrb, loss_val = evaluate_gradients(
							model=model,
							x=x,
							y=y
							)

	print("-------------- Gradients --------------------")
	for g, v in zip(grd, vrb):
		print(v.name, tf.norm(g).numpy())
	print("---------------------------------------------")

	norms = [tf.norm(g).numpy() if g is not None else 0 for g in grd]
	names = [v.name for v in vrb]

	return names,norms

def evaluate_gradients(model, x, y, loss_fn=None):
    """
    Evaluate gradients of the loss with respect to model trainable variables.

    Parameters
    ----------
    model : tf.keras.Model
        A trained Keras model.
    x : np.array or tf.Tensor
        Input data.
    y : np.array or tf.Tensor
        Target labels.
    loss_fn : callable, optional
        Loss function. If None, model.compiled_loss is used.

    Returns
    -------
    grads : list of tf.Tensor
        Gradients for each trainable variable.
    vars : list of tf.Variable
        Corresponding model variables.
    loss_value : tf.Tensor
        Computed loss for the batch.
    """

    if loss_fn is None:
        loss_fn = model.compiled_loss

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    with tf.GradientTape() as tape:
        preds = model(x, training=False)

        if loss_fn == model.compiled_loss:
            loss_value = loss_fn(y, preds)
        else:
            loss_value = loss_fn(y, preds)

    variables = model.trainable_variables
    grads = tape.gradient(loss_value, variables)

    return grads, variables, loss_value


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
) -> Sequential:
	"""
	Create a Keras Sequential model with the specified number of layers.

	Parameters:
	- input_shape (int): The number of features in the input data.
	- num_layers (int): The number of hidden layers in the model.
	- units (List[int]): A list containing the number of units for each hidden layer.
	- activations (List[str]): A list containing the activation function for each hidden layer.
	- output_units (int): The number of units in the output layer.
	- activation_output (str): The activation function for the output layer (default is 'linear').

	Returns:
	- model (Sequential): The compiled Keras Sequential model.
	"""

	model = Sequential()
	model.add(keras.Input(shape=(input_shape,)))

	# Hidden layers
	for i in range(num_layers):
		model.add(
			Dense(
				size_layers,
				activation=activation_layers,
				kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
				bias_initializer=keras.initializers.GlorotUniform(seed=seed),
				# kernel_regularizer=keras.regularizers.l2(lambda_rgl),
				# bias_regularizer=keras.regularizers.l2(lambda_rgl)
			)
		)

	# Output layer
	model.add(
		Dense(
			output_shape,
			activation=activation_output,
			kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
			bias_initializer=keras.initializers.GlorotUniform(seed=seed),
			# kernel_regularizer=keras.regularizers.l2(lambda_rgl),
			# bias_regularizer=keras.regularizers.l2(lambda_rgl)
		)
	)

	return model

def compile_model(model,
	lr_schedule,
	beta_1=0.9,
	beta_2=0.999,
	loss: str = "mean_squared_error",
	metrics: list = ["mean_squared_error"],
	clipnorm: float = 1.0,
	use_ema=False,
	):
	model.compile(
		optimizer=keras.optimizers.Adam(
						learning_rate=lr_schedule,
						beta_1=beta_1,
						beta_2=beta_2,
						clipnorm=clipnorm,
						use_ema=use_ema
						),
		loss=loss,
		metrics=metrics,
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
