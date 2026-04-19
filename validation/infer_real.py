import sys
import os

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

case = "Pleiades"
model = "extinction"

dir_base = "/home/jolivares/Repos/Huehueti/validation/real/{0}/"
dir_mlps = "/home/jolivares/Models/PARSEC/20-220Myr/Optuna_InverseTimeDecay_logAge_logL_epochs_5e+02_trials_100_0.1myr/"

file_data = dir_base.format(case) + "inputs/Hunt+2024_>10.5g.csv"
dir_out   = dir_base.format(case) + "{0}_l2_0.5Av_10.5G_FullRankADVI/".format(model)

files_mlps = {
	"G_BPmag":dir_mlps + "/G_BPmag_l2/seed_0/mlp.pkl",
	"Gmag":   dir_mlps + "/Gmag_l2/seed_0/mlp.pkl",
	"G_RPmag":dir_mlps + "/G_RPmag_l2/seed_0/mlp.pkl"
	}

absolute_photometry = ['G_BPmag','Gmag','G_RPmag']
observables = {
	"photometry":[ 'BP','G', 'RP'],
	"photometry_error":['BP_error','G_error','RP_error'],
}
parameters = {"age":None}
hyperparameters = {"distance":"distance"}


prior = {
	'age' : {
		'family' : "Uniform",
		},
	'log_lum' : {
		'family' : 'Uniform',
		},
	'distance_mu' : {
		'family' : 'Gaussian',
		'mu' : 50,
		'sigma' : 10.
		},
	"distance_sd":{
		"family": "Exponential",
		"scale" : 5.
		},
	"extinction":{
		"family": "Uniform",
		"lower":0.0,
		"upper":0.5,
		"mu":2.0,
		"sigma" : 0.1,
		},
	"outliers":{
		"family":"StudentT",
		"beta": 1/30.,
		"scale":1.0
		}
}
					
os.makedirs(dir_out,exist_ok=True)

hue = Huehueti(
	dir_out = dir_out, 
	observables=observables,
	absolute_photometry=absolute_photometry,
	hyperparameters = hyperparameters,
	files_mlps=files_mlps
	)
hue.load_data(
	file_data = file_data
	)
hue.setup(
	model=model,
	parameters = parameters, 
	prior = prior
	)
hue.run(
	target_accept=0.85,
	init_method="fullrank_advi",
	init_iters=int(5e5),
	init_tracker = False,
	nuts_sampler="fullrank_advi",
	# nuts_sampler="numpyro",
	tuning_iters=int(2e3),
	sample_iters=int(2e3),
	prior_iters=int(2e3),
	chains=2
	)
hue.load_trace()
hue.convergence()
hue.plot_chains()
hue.plot_posterior()
hue.plot_cpp()
hue.plot_predictions()
hue.plot_cmd(cmd={
		"magnitude":"G",
		"color":["G","RP"]})
hue.save_statistics()
