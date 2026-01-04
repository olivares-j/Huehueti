import sys
import os

sys.path.append("/home/jolivares/Repos/Huehueti/src/Huehueti/")
from Huehueti import Huehueti

dir_base       = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/"
file_mlp_phot  = "/home/jolivares/Repos/Huehueti/mlps/PARSEC/GP2_l9_s512/mlp.pkl"
file_mlp_mass  = "/home/jolivares/Repos/Huehueti/mlps/PARSEC/mTg_l7_s256/mlp.pkl"

list_of_ages = [20,30,40,50,60,70,80,90,100,120,140,160,180,200]
list_of_distances  = [136]
seed      = 0
n_stars   = 50

dir_inputs  = dir_base + "inputs/"
dir_outputs = dir_base + "outputs/"
base_name   = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

observables = {
"photometry":['G', 'BP', 'RP','gP1','rP1','iP1','zP1','yP1','J','H','Ks'],
"photometry_error":['e_G', 'e_BP', 'e_RP','e_gP1','e_rP1','e_iP1','e_zP1','e_yP1','e_J','e_H','e_Ks'],
}

def set_prior(age,distance):
	priors = {
	'age' : {
		'family' : "TruncatedNormal",
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