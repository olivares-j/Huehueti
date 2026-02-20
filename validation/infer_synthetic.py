import sys
import os
import time
import dill

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

case ="l3_s1100"
age_range = "50-150Myr"

dir_base       = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/{0}/".format(age_range)
file_mlp_phot  = "/home/jolivares/Models/PARSEC/Gaia_EDR3/{0}/MLPs/Phot_{1}/mlp.pkl".format(age_range,case)
file_mlp_teff  = None #"/home/jolivares/Models/PARSEC/Gaia_EDR3/15-400Myr/MLPs/Teff_l16_s512/mlp.pkl"

list_of_ages      = [140] #list(range(60,160,20))
list_of_distances = [50]
list_of_n_stars   = [10]
list_of_seeds     = [5]

dir_inputs  = dir_base + "inputs/"
dir_outputs = dir_base + "{0}/".format(case)
base_name   = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

absolute_photometry = ['Gmag', 'G_BPmag', 'G_RPmag']
observables = {
"photometry":['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
"photometry_error":['phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error'],
}

parameters = {"age":None}
hyperparameters = {"distance":"distance"}

chains ={
	60:None,80:[1,2,3],100:[1,2],120:[3],140:[1,2,3],
	160:None,180:None,200:None,
	220:None,240:None,260:None,280:None,300:None,3200:None,340:None,360:None,380:None,400:None
}


def set_prior(age,distance):
	priors = {
	'age' : {
		'family' : "Uniform",
		'mu'    : float(age),
		'sigma' : 30.,
		'lower' : 50,
		'upper' : 150,
		},
	'distance' : {
		'family' : 'Gaussian',
		'mu' : float(distance),
		'sigma' : 10.
		},
	"distance_dispersion":{
		"family": "Exponential",
		"scale" : 5.
		},
	"photometric_dispersion":{
		"family": "Exponential",
		"sigma" : 0.01,
		"beta"  : 100.0
		},
	"astrometric_outliers":{
		"weights" : [90,10],
		"lower"   : 50.0,
		"upper"   : 150.0,
		"beta"    : 1/20.
	},
	"photometric_outliers":{
		"weights" : [90,10],
		"lower"   : 0.0,
		"upper"   : 30.0,
		"beta"    : 1/20.
	},
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

				# if os.path.isfile(file_sts):
				# 	continue
					
				os.makedirs(dir_out,exist_ok=True)

				start_time = time.time()
				hue = Huehueti(
					dir_out = dir_out, 
					file_mlp_phot=file_mlp_phot,
					file_mlp_teff=file_mlp_teff,
					observables=observables,
					absolute_photometry=absolute_photometry,
					hyperparameters = hyperparameters,
				)
				hue.load_data(
					file_data = file_data
				)
				hue.setup(
					parameters = parameters, 
					prior = set_prior(age,distance)
				)
				hue.run(
					target_accept=0.65,
					init_method="advi",
					init_iters=int(5e5),
					nuts_sampler="numpyro",
					tuning_iters=int(4e3),
					sample_iters=int(2e3),
					prior_iters=int(2e3),
					chains=4
					)
				hue.load_trace(chains=chains[age])
				hue.convergence()
				hue.plot_chains()
				hue.plot_posterior()
				hue.plot_cpp()
				hue.plot_predictions()
				hue.plot_cmd(
					cmd={
						"magnitude":"phot_g_mean_mag",
						"color":["phot_g_mean_mag","phot_rp_mean_mag"]
						})
				hue.save_statistics()
				end_time = time.time()

				#--------- Save time--------------------
				data = {
					"case":case,
					"age":age,
					"distance":distance,
					"n_stars":n_stars,
					"seed":seed,
					"time":end_time - start_time
					}

				# with open(file_time, "wb") as file:
				# 	dill.dump(data, file)
				#------------------------------------
