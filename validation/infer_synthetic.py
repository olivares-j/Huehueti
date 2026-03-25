import sys
import os
import time
import dill

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

model = "base"

dir_base = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/50-150Myr/{0}/".format(model)
dir_mlps = "/home/jolivares/Models/PARSEC/Gaia_EDR3/50-150Myr/Kroupa/"

list_of_ages      = list(range(60,160,20))
list_of_distances = [50,100,200,400]
list_of_n_stars   = [20]
list_of_seeds     = [0,1,2,3,4]

dir_inputs  = dir_base + "inputs/"
dir_outputs = dir_base + "Optuna_InverseTimeDecay_lrin_None_lrdr_None_bs_10000_epochs_1e+03_l2_17_10_15/"
base_name   = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

files_mlps = {
	"G_BPmag":dir_mlps + "Optuna_InverseTimeDecay_lrin_None_lrdr_None_bs_10000_epochs_1e+03/G_BPmag_l2_s17/seed_0/mlp.pkl",
	"Gmag":   dir_mlps + "Optuna_InverseTimeDecay_lrin_None_lrdr_None_bs_10000_epochs_1e+03/Gmag_l2_s10/seed_0/mlp.pkl",
	"G_RPmag":dir_mlps + "Optuna_InverseTimeDecay_lrin_None_lrdr_None_bs_10000_epochs_1e+03/G_RPmag_l2_s15/seed_0/mlp.pkl"
	}

absolute_photometry = ['G_BPmag','Gmag','G_RPmag']
observables = {
	"photometry":[ 'BP','G', 'RP'],
	"photometry_error":['e_BP','e_G','e_RP'],
}

parameters = {"age":None}
hyperparameters = {"distance":"distance"}

chains ={
	60:[1,2,3],
	80:[1,2,3],
	100:[0,1,2],
	120:[1,2],
	140:[1,2],
}


def set_prior(age,distance):
	priors = {
	'age' : {
		'family' : "Uniform",
		'mu'    : float(age),
		'sigma' : 30.,
		},
	'log_lum' : {
		'family' : 'Uniform',
			},
	'log_tef' : {
		'family' : 'Uniform',
			},
	'distance_mu' : {
		'family' : 'Gaussian',
		'mu' : float(distance),
		'sigma' : 10.
		},
	"distance_sd":{
		"family": "Exponential",
		"scale" : 5.
		},
	"extinction":{
		"family": "Uniform",
		"lower":0.0,
		"upper":5.0,
		"mu":2.0,
		"sigma" : 0.1,
		},
	"outliers":{
		"family":"Exponential",
		"scale":1.0,
		"beta": 1/10.
		}
	}
	return priors

os.makedirs(dir_outputs,exist_ok=True)

for seed in list_of_seeds:
	print(20*"s"+" {0:d} ".format(seed) + 20*"s")
	for age in list_of_ages:
		print(20*"a"+" {0:d} ".format(age) + 20*"a")
		for distance in list_of_distances:
			print(20*"d"+" {0:d} ".format(distance) + 20*"d")
			for n_stars in list_of_n_stars:
				print(20*"n"+" {0:d} ".format(n_stars) + 20*"n")
			
				file_data = dir_inputs  + base_name.format(age,distance,n_stars,seed)+".csv"
				dir_out   = dir_outputs + base_name.format(age,distance,n_stars,seed)+"/"
				file_sts  = dir_out + "Global_statistics.csv"
				file_time = dir_out + "time.pkl"

				if os.path.isfile(file_sts):
					continue
					
				os.makedirs(dir_out,exist_ok=True)

				start_time = time.time()
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
					prior = set_prior(age,distance)
					)
				hue.run(
					target_accept=0.85,
					init_method="advi",
					init_iters=int(3e5),
					nuts_sampler="numpyro",
					tuning_iters=int(2e3),
					sample_iters=int(2e3),
					prior_iters=int(2e3),
					chains=3
					)
				hue.load_trace()#chains=chains[age])
				hue.convergence()
				hue.plot_chains()
				hue.plot_posterior()
				hue.plot_cpp()
				hue.plot_predictions()
				hue.plot_cmd(
					cmd={
						"magnitude":"G",
						"color":["G","RP"]
						})
				hue.save_statistics()
				end_time = time.time()

				#--------- Save time--------------------
				data = {
					"age":age,
					"distance":distance,
					"n_stars":n_stars,
					"seed":seed,
					"time":end_time - start_time
					}

				with open(file_time, "wb") as file:
					dill.dump(data, file)
				#------------------------------------
