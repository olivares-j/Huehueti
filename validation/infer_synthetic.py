import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

dir_base       = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC_tests/"
file_mlp_phot  = "/home/jolivares/Models/PARSEC/Gaia_EDR3_15-400Myr/MLPs/Phot_l7_s512/mlp.pkl"
file_mlp_teff  = "/home/jolivares/Models/PARSEC/Gaia_EDR3_15-400Myr/MLPs/Teff_l16_s512/mlp.pkl"

list_of_ages = [120] #list(range(20,420,20))
list_of_distances  = [136] #[50]
n_stars   = 10
seed      = 1


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
	80:None,100:None,120:None,140:None,160:None,180:None,200:[1,2],220:None,240:None,260:None,280:None,300:None,3200:None,340:None,360:None,380:None,400:None
}


def set_prior(age,distance):
	priors = {
	'age' : {
		'family' : "Uniform",
		'mu'    : float(age),
		'sigma' : 30.,
		'lower' : 20,
		'upper' : 200,
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

for age in list_of_ages:
	print(20*"="+" {0:d} ".format(age) + 20*"=")
	for distance in list_of_distances:
		print(20*"+"+" {0:d} ".format(distance) + 20*"+")

		file_data = dir_inputs  + base_name.format(age,distance,n_stars,seed)+".csv"
		dir_out   = dir_outputs + base_name.format(age,distance,n_stars,seed)+"/"
		file_sts  = dir_out + "Global_statistics.csv"

		if os.path.isfile(file_sts):
			continue
			
		os.makedirs(dir_out,exist_ok=True)

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
		init_method="advi",
		# init_method="fullrank_advi",
		init_iters=int(5e5),
		# nuts_sampler="advi",
		# nuts_sampler="fullrank_advi",
		nuts_sampler="numpyro",
		# tuning_iters=int(1e4),
		tuning_iters=int(2e3),
		sample_iters=int(2e3),
		prior_iters=int(2e3),
		chains=4)
		hue.load_trace(chains=chains[age])
		hue.convergence()
		hue.plot_chains()
		hue.plot_posterior()
		hue.plot_cpp()
		hue.plot_predictions()
		hue.plot_cmd(cmd={"magnitude":"phot_g_mean_mag","color":["phot_g_mean_mag","phot_rp_mean_mag"]})
		hue.save_statistics()