import sys
import os

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

system = "Pleiades"

dir_base       = "/home/jolivares/Repos/Huehueti/validation/real/"
file_mlp_phot  = "/home/jolivares/Repos/Huehueti/mlps/PARSEC/GP2_l9_s512/mlp.pkl"
file_mlp_mass  = "/home/jolivares/Repos/Huehueti/mlps/PARSEC/mTg_l7_s256/mlp.pkl"


dir_inputs  = dir_base + "inputs/"
dir_outputs = dir_base + "outputs/"
base_name   = "{0}_filtering_value_{1}_error{2}"

list_of_filters = [{"value":10.0,"error":1e-3},{"value":10.0,"error":1e-3}]

observables = {
"photometry":['g', 'bp', 'rp','gmag','rmag','imag','zmag','ymag','Jmag','Hmag','Kmag'],
"photometry_error":['g_error', 'bp_error', 'rp_error','e_gmag','e_rmag','e_imag','e_zmag','e_ymag','e_Jmag','e_Hmag','e_Kmag'],
}


priors = {
	'age' : {
		'family' : "TruncatedNormal",
		'mu'    : 120.0,
		'sigma' : 30.,
		'lower' : 20,
		'upper' : 200,
		},
	'distance' : {
		'family' : 'Gaussian',
		'mu' : 136.0,
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


for flt in list_of_filters:
	file_data = dir_inputs  + "{0}.csv".format(system)
	dir_out   = dir_outputs + base_name.format(system,flt["value"],flt["error"])+"/"
	file_sts  = dir_out + "Global_statistics.csv"

	if os.path.isfile(file_sts):
		continue
		
	os.makedirs(dir_out,exist_ok=True)

	hue = Huehueti(
		dir_out = dir_out, 
		file_mlp_phot=file_mlp_phot,
		file_mlp_mass=file_mlp_mass,
		observables=observables)
	hue.load_data(file_data = file_data)
	hue.setup(prior = set_prior(age,distance))
	# hue.plot_pgm()
	hue.run(
		init_iters=int(8e4),
		init_refine=False,
		nuts_sampler="advi",
		tuning_iters=int(5e4),
		sample_iters=2000,
		prior_iters=2000)
	hue.load_trace()
	hue.convergence()
	hue.plot_chains()
	hue.plot_posterior()
	hue.plot_cpp()
	hue.plot_predictions()
	hue.plot_cmd(cmd={"magnitude":"G","color":["G","RP"]})
	hue.save_statistics()