import os
import numpy as np
from Amasijo import Amasijo

model         = "PARSEC"
dir_inputs    = "/home/jolivares/Repos/Huehueti/validation/synthetic/{0}/inputs/".format(model)
dir_mlps      = "/home/jolivares/Models/PARSEC/Gaia_EDR3/15-400Myr/MLPs/"
file_mlp_phot = dir_mlps+"Phot_l7_s512/mlp.pkl"
file_mlp_teff = dir_mlps+"Teff_l16_s512/mlp.pkl"
file_mlp_logg = dir_mlps+"Logg_l13_s256/mlp.pkl"
base_name = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

list_of_ages = list(range(20,420,20))
list_of_distances  = [50]#,100,150,200]
n_stars   = 10
seed      = 0

def mass_limits(age):
	if age < 160:
		return [0.1,4.5]
	elif age < 200:
		return [0.1,4.0]
	elif age < 300:
		return [0.1,3.3]
	elif age <= 400:
		return [0.1,3.0]
	else:
		sys.exit("No available age!")

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

def isochrones_args(model,age):
	args = {
		"model":model,
		"age": float(age),
		"MIST_args":{
			"mass_limits":mass_limits(age),
			"metallicity":0.012,
			"Av": 0.0
			},
		"PARSEC_args":{
				"file_mlp_phot":file_mlp_phot,
				"file_mlp_teff":file_mlp_teff,
				"file_mlp_logg":file_mlp_logg,
				"mass_limits":mass_limits(age),
				},          
		"bands":["G","BP","RP"]#,"gP1", "rP1", "iP1", "zP1", "yP1","J", "H", "Ks"],
		}
	return args


os.makedirs(dir_inputs,exist_ok=True)

for age in list_of_ages:
	for distance in list_of_distances:
		file_data = dir_inputs + base_name.format(age,distance,n_stars,seed) + ".csv"
		file_plot = dir_inputs + base_name.format(age,distance,n_stars,seed) + ".pdf"

		ama = Amasijo(
					phasespace_args=phasespace_args(distance),
					isochrones_args=isochrones_args(model,age),
					seed=seed)
		ama.generate_cluster(file_data,n_stars=n_stars,angular_correlations=None)
		ama.plot_cluster(file_plot=file_plot)