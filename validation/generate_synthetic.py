import os
import numpy as np
from Amasijo import Amasijo

model      = "PARSEC"
age_range  = "50-150Myr"
dir_inputs = "/home/jolivares/Repos/Huehueti/validation/synthetic/{0}/{1}/inputs/".format(model,age_range)

# Amasijo MLPs can have more layers given that these are only used to generate sources
dir_mlps      = "/home/jolivares/Models/{0}/Gaia_EDR3/15-400Myr/MLPs/".format(model)
file_mlp_phot = dir_mlps + "Phot_l7_s512/mlp.pkl"
file_mlp_teff = dir_mlps + "Teff_l16_s512/mlp.pkl"
file_mlp_logg = dir_mlps + "Logg_l13_s256/mlp.pkl"

base_name = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

list_of_ages      = [140] #list(range(60,160,20))
list_of_distances = [100]
list_of_n_stars   = [20]
list_of_seeds     = [0,1,2,3,4,5]

def mass_limits(age,distance):
	if distance == 50:
		if age < 60:
			return [0.1,3.0]
		elif age < 80:
			return [0.1,3.5]
		elif age < 100:
			return [0.1,3.2]
		elif age < 120:
			return [0.1,3.1]
		elif age < 140:
			return [0.1,3.1]
		elif age < 160:
			return [0.1,3.0]
	elif distance == 100:
		if age == 60:
			return [0.1,5.0]
		elif age == 80:
			return [0.1,4.8]
		elif age == 100:
			return [0.1,4.5]
		elif age == 120:
			return [0.1,4.3]
		elif age == 140:
			return [0.1,4.1]
	else:
		sys.exit("No available distance!")

def phasespace_args(distance):
	args = {
	"position":{"family":"Gaussian",
				"location":np.array([float(distance),0.0,0.0]),
				"covariance":np.diag([9.,9.,9.])},
	"velocity":{"family":"Gaussian",
				"location":np.array([10.0,10.0,10.0]),
				"covariance":np.diag([1.,1.,1.]),
				"kappa":np.ones(3),
				"omega":np.array([[-1,-1,-1],[1,1,1]])
				}}
	return args

def isochrones_args(model,age,distance):
	args = {
		"model":model,
		"age": float(age),
		"MIST_args":{
			"mass_limits":mass_limits(age,distance),
			"metallicity":0.012,
			"Av": 0.0
			},
		"PARSEC_args":{
				"file_mlp_phot":file_mlp_phot,
				"file_mlp_teff":file_mlp_teff,
				"file_mlp_logg":file_mlp_logg,
				"mass_limits":mass_limits(age,distance),
				},          
		"bands":["G","BP","RP"]#,"gP1", "rP1", "iP1", "zP1", "yP1","J", "H", "Ks"],
		}
	return args


os.makedirs(dir_inputs,exist_ok=True)

for age in list_of_ages:
	for distance in list_of_distances:
		for n_stars in list_of_n_stars:
			for seed in list_of_seeds:
				file_data = dir_inputs + base_name.format(age,distance,n_stars,seed) + ".csv"
				file_plot = dir_inputs + base_name.format(age,distance,n_stars,seed) + ".pdf"

				ama = Amasijo(
							phasespace_args=phasespace_args(distance),
							isochrones_args=isochrones_args(model,age,distance),
							seed=seed)
				ama.generate_cluster(file_data,
							n_stars=n_stars,
							angular_correlations=None)
				ama.plot_cluster(
							file_plot=file_plot)