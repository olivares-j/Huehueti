import sys
import os

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

# case = "BetaPictoris"
# case = "Blanco1"
# case = "Pleiades"
# case = "Hyades"
case = "GroupX"
# case = "Latyshev_2"
# case = "Mecayotl_1"
# authors = "Meingast+2021"
# authors = "Meingast+2021_<4.5G"
# authors = "Miret-Roig+2020"
# authors = "Miret-Roig+2020_<7.0G"
# authors = "Meingast+2019"
# authors = "Meingast+2019_<4.5G"
# authors = "Olivares+2023_<4.5G"
# authors = "Olivares+2023_<5G"
# authors = "Olivares+2023_<6G"
authors = "Olivares+2023_<7G"


# age_range = "15-25Myr"
# age_step = 0.025
# age_range = "20-220Myr"
# age_step  = 0.05
# layers = 2
age_range = "200-600Myr"
age_step = 0.1
layers = 2
# age_range = "600-1000Myr"
# age_step  = 1
# layers = 2

# model = "base"
# model = "dispersion"
model = "linear_dispersion"
# model = "outliers"
# model = "shift"
# model = "base+extinction"
init_method = "FullRank_ADVI"
# init_method = "ADVI"

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
	"dispersion":{
		# "family": "Gamma",
		"family": "Exponential",
		"beta" : 1000.0,
		"lambda":1000.0,
		},
	"linear_dispersion":{
		"family": "Exponential",
		# "family": "SoftPlus",
		"sigma_intercept" : 20.0,
		"sigma_slope":5.0,
		},
	"outliers":{
		"family":"Normal",
		# "family":"Exponential",
		"scale":0.01,
		},
	"shift":{
		"dispersion":{
		# "family": "Gamma",
		"family": "Exponential",
		"beta" : 100.0,
		"lambda":1000.0,
			},
		"outliers":{
		# "family":"Normal",
		"family":"Exponential",
		"scale":0.01,
			},
		},
	"extinction":{
		"family": "Uniform",
		"lower":0.0,
		"upper":1.0,
		"mu":0.1,
		"sigma" : 0.05,
		},
	
}


dir_base = "/home/jolivares/Repos/Huehueti/validation/real/{0}/"
dir_mlps = "/home/jolivares/Models/PARSEC/{0}/Optuna_InverseTimeDecay_logAge_logL_epochs_1e+03_trials_50_{1}myr/".format(age_range,age_step)

file_data = dir_base.format(case) + "inputs/{0}_GDR3.csv".format(authors)
# dir_out   = dir_base.format(case) + "{0}_{1}_l{2}_{3}/".format(authors,model,layers,init_method)
dir_out   = dir_base.format(case) + "{0}_{1}_l{2}_{3}_{4}/".format(authors,model,layers,init_method,prior[model]["family"])



files_mlps = {
	"G_BPmag":dir_mlps + "/G_BPmag_l{0}/seed_0/mlp.pkl".format(layers),
	"Gmag":   dir_mlps + "/Gmag_l{0}/seed_0/mlp.pkl".format(layers),
	"G_RPmag":dir_mlps + "/G_RPmag_l{0}/seed_0/mlp.pkl".format(layers),
	"Mini":dir_mlps + "/Mini_l4/seed_0/mlp.pkl",
	}

absolute_photometry = ['G_BPmag','Gmag','G_RPmag']
observables = {
	"photometry":[ 'bp','g', 'rp'],
	"photometry_error":['bp_error','g_error','rp_error'],
}
parameters = {"age":None}
hyperparameters = {"distance":"distance"}



					
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
	init_method=init_method,
	init_iters=int(5e5),
	init_tracker = False,
	nuts_sampler=init_method,
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
		"magnitude":"g",
		"color":["g","rp"]})
hue.save_statistics()
