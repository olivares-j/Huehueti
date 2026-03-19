import sys
import os
import time
import dill

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

model = "extinction"
mlp   = "l3_s1100"

dir_base       = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/50-150Myr/{0}/".format(model)
file_mlp_phot  = "/home/jolivares/Models/PARSEC/Gaia_EDR3/50-150Myr/MLPs/Phot_{0}/mlp.pkl".format(mlp)
file_mlp_teff  = None

list_of_ages      = [60] #list(range(60,160,20))
list_of_distances = [50] #[50,100,200,400]
list_of_n_stars   = [20] #[10,20]
list_of_seeds     = [0,1,2,3,4,5]

dir_inputs  = dir_base + "inputs/"
dir_outputs = dir_base + "outputs/"
base_name   = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

absolute_photometry = ['Gmag', 'G_BPmag', 'G_RPmag']
observables = {
"photometry":['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
"photometry_error":['phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error'],
}

parameters = {"age":None}
hyperparameters = {"distance":"distance"}

chains ={
	60:[0,1,2],
	80:[1,2,3],
	100:[0,1,2],
	120:[3],
	140:[2],
	# 160:None,180:None,200:None,
	# 220:None,240:None,260:None,280:None,300:None,3200:None,340:None,360:None,380:None,400:None
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
					model=model,
					parameters = parameters, 
					prior = set_prior(age,distance)
				)
				hue.run(
					target_accept=0.85,
					init_method="advi",
					init_iters=int(1e6),
					nuts_sampler="advi",
					tuning_iters=int(5e5),
					sample_iters=int(2e3),
					prior_iters=int(2e3),
					chains=4
					)
				hue.load_trace()#chains=chains[age])
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
					"mlp":mlp,
					"age":age,
					"distance":distance,
					"n_stars":n_stars,
					"seed":seed,
					"time":end_time - start_time
					}

				with open(file_time, "wb") as file:
					dill.dump(data, file)
				#------------------------------------
