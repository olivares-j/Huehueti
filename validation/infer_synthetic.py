import sys
import os
import time
import dill

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti


# age_range = "15-25Myr"
# age_range = "15-220Myr"
# age_range = "20-220Myr"
age_range = "200-600Myr"
# age_range = "600-1000Myr"

# age_step = 0.025
# age_step = 0.05
# age_step = 0.1    #20-220Myr
age_step = 0.5    #200-600Myr
# age_step = 1

init_iters = int(5e5)


case = "Optuna_InverseTimeDecay_epochs_5e+02_trials_50_{0}myr".format(age_step)
init_method = "fullrank_advi"

dir_base = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC"
dir_mlps = "/home/jolivares/Models/PARSEC/{0}/".format(age_range)


if age_range == "1-21Myr":
	list_of_ages = list(range(1,21,2))
elif age_range == "20-220Myr":
	list_of_ages = list(sum([[25],list(range(40,220,20)),[210]],[]))
elif age_range == "200-600Myr":
	list_of_ages = list(sum([list(range(200,600,50)),[210,590]],[]))
elif age_range == "600-1000Myr":
	list_of_ages = list(range(600,1100,100))
else:
	sys.exit("Undefined age range")


list_of_models = ["base+dispersion"]
list_of_distances = [100]
list_of_n_stars   = [15]
list_of_seeds     = [0,1,2,3,4]

base_inputs  = "{0}/{1}/{2}/inputs/"
base_outputs = "{0}/{1}/{2}/{3}/"
base_name    = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"


match age_range:
	case "20-220Myr":
		files_mlps = {
			"Phot":dir_mlps + case + "/l4/seed_0/mlp.pkl",
			"Mini":dir_mlps + case + "/Mini_l3/seed_0/mlp.pkl",
			"logL":dir_mlps + case + "/logL_l3/seed_0/mlp.pkl",
			}

	case "200-600Myr":
		files_mlps = {
			"Phot":dir_mlps + case + "/l4/seed_0/mlp.pkl",
			"Mini":dir_mlps + case + "/Mini_l4/seed_0/mlp.pkl",
			"logL":dir_mlps + case + "/logL_l4/seed_0/mlp.pkl",
			}

features = ["logAge","logL"]

absolute_photometry = ['G_BPmag','Gmag','G_RPmag']
observables = {
	"photometry":[ 'phot_bp_mean_mag','phot_g_mean_mag', 'phot_rp_mean_mag'],
	"photometry_error":[ 'phot_bp_mean_mag_error','phot_g_mean_mag_error', 'phot_rp_mean_mag_error'],
}
cmd = {
"magnitude":"phot_g_mean_mag",
"color":["phot_g_mean_mag","phot_rp_mean_mag"]}

parameters = {"age":None}
hyperparameters = {"distance":"distance"}

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
		"upper":1.0,
		"mu":0.1,
		"sigma" : 0.05,
		},
	}
	return priors

for model in list_of_models:
	print(20*"m"+" {0} ".format(model) + 20*"m")
	for seed in list_of_seeds:
		print(20*"s"+" {0:d} ".format(seed) + 20*"s")
		for age in list_of_ages:
			print(20*"a"+" {0:d} ".format(age) + 20*"a")
			for distance in list_of_distances:
				print(20*"d"+" {0:d} ".format(distance) + 20*"d")
				for n_stars in list_of_n_stars:
					print(20*"n"+" {0:d} ".format(n_stars) + 20*"n")

					dir_inputs  = base_inputs.format(dir_base,age_range,model)
					dir_outputs = base_outputs.format(dir_base,age_range,model,case)

					os.makedirs(dir_outputs,exist_ok=True)
				
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
						prior = set_prior(age,distance),
						features=features
						)
					hue.run(
						target_accept=0.65,
						init_method=init_method,
						init_iters=init_iters,
						init_tracker=False,
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
					hue.plot_cmd(cmd=cmd)
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
