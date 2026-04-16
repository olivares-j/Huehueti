import sys
import os
import time
import dill

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

model = "base"
case = "Optuna_logAge_logL_epochs_5e+02_0.1myr"

dir_base = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/20-220Myr/{0}/".format(model)
dir_mlps = "/home/jolivares/Models/PARSEC/20-220Myr/"

list_of_ages      = list(range(20,240,20))
list_of_distances = [50,100,200,400]
list_of_n_stars   = [30]
list_of_seeds     = [0,1,2,3,4]

dir_inputs  = dir_base + "inputs/"
dir_outputs = dir_base + case + "_l2_FullRankADVI/"
base_name   = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

files_mlps = {
	"G_BPmag":dir_mlps + case + "/G_BPmag_l2/seed_0/mlp.pkl",
	"Gmag":   dir_mlps + case + "/Gmag_l2/seed_0/mlp.pkl",
	"G_RPmag":dir_mlps + case + "/G_RPmag_l2/seed_0/mlp.pkl"
	}
features = ["logAge","logL"]

absolute_photometry = ['G_BPmag','Gmag','G_RPmag']
observables = {
	"photometry":[ 'BP','G', 'RP'],
	"photometry_error":['e_BP','e_G','e_RP'],
}

parameters = {"age":None}
hyperparameters = {"distance":"distance"}

chains = {
	# 0:[0,1],
	# 1:[2,0],
	# 2:[0,1],
	# 3:[0,2],
	# 4:[0],
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
		# "lower_par":[-0.72855578,3.0677147 ],
		# "upper_par":[-1.58449405,15.91599175]
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
		"family":"StudentT",
		"beta": 1/30.,
		"scale":1.0
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
				# if not seed in chains.keys():
				# 	continue
					
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
					prior = set_prior(age,distance),
					features=features
					)
				hue.run(
					target_accept=0.85,
					init_method="fullrank_advi",
					init_iters=int(5e5),
					nuts_sampler="fullrank_advi",
					# nuts_sampler="numpyro",
					tuning_iters=int(2e3),
					sample_iters=int(2e3),
					prior_iters=int(2e3),
					chains=2
					)
				hue.load_trace()#chains=chains[seed])
				hue.convergence()
				hue.plot_chains()
				hue.plot_posterior()
				hue.plot_cpp()
				hue.plot_predictions()
				hue.plot_cmd(cmd={
						"magnitude":"G",
						"color":["G","RP"]})
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
