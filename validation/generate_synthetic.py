import os
import numpy as np
from Amasijo import Amasijo



# age_range = "15-25Myr"
# age_range  = "20-220Myr"
age_range  = "200-600Myr"
# age_range  = "600-1000Myr"


# age_step  = "0.025myr"
# age_step   = "0.1myr"
# age_step   = "0.5myr"
age_step   = "1myr"

max_Av     = 0.0
seed_offset = 2322

dir_base = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/{0}".format(age_range)
base_name = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

models = ["binaries"] #,"linear_dispersion"]

if age_range == "1-21Myr":
	list_of_ages = list(range(1,22,1))
elif age_range == "20-220Myr":
	list_of_ages = list(sum([[25],list(range(40,220,20)),[210]],[]))
elif age_range == "200-600Myr":
	list_of_ages = list(sum([[210],list(range(250,600,50)),[590]],[]))
elif age_range == "600-1000Myr":
	list_of_ages = list(range(600,1100,100))
else:
	sys.exit("Undefined age range")

list_of_distances = [100]
list_of_n_stars   = [15,30,50]
list_of_seeds     = [0,1,2,3,4]


def binary_args(model):
	if "binaries" in model:
		enabled = True
	else:
		enabled = False
	
	arguments = {
				"enabled": enabled,
				"q_distribution": "uniform",
				"q_limits": (0.1, 1.0),
				}
	return arguments

def phasespace_args(distance):
	args = {
	"coordinates":["X_gal","Y_gal","Z_gal","U_gal","V_gal","W_gal"],
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

def mass_limits(age_range):
	if age_range == "1-21Myr":
		return [1.1,20.]
	elif age_range == "20-220Myr":
		return [1.1,11.0]
	elif age_range == "200-600Myr":
		return [1.1,3.8]
	else:
		sys.exit("Undefined age range")

def isochrones_args(age,distance):
	args = {
	"model":"PARSEC",
	"age": float(age),
	"Av_limits":[0.0,max_Av],
	"mass_limits":mass_limits(age_range),
	"MIST_args":{
		"metallicity":0.0152,
		},
	"PARSEC_args":{
		"files":[
		"/home/jolivares/Models/PARSEC/{0}/Gaia_EDR3+2MASS_{1}_no-turn.csv".format(age_range,age_step)
		],
		"max_label":1,
		"bands_wavelengths":[6217.6,5109.7,7769.0,12350.,16620.,21590.], # Same order as bands
		"Rv":3.1
		},
	"bands":["G","BP","RP","J","H","Ks"],
	"uncertainties":[0.001,0.02,0.004,0.025,0.030,0.025]
	}
	return args

for model in models:
	dir_inputs = "{0}/{1}/inputs/".format(dir_base,model)
	os.makedirs(dir_inputs,exist_ok=True)

	for age in list_of_ages:
		print(10*"-"+" "+str(age)+" "+"-"*10)
		for distance in list_of_distances:
			for n_stars in list_of_n_stars:
				for seed in list_of_seeds:
					file_data = dir_inputs + base_name.format(age,distance,n_stars,seed) + ".csv"
					file_plot = dir_inputs + base_name.format(age,distance,n_stars,seed) + ".pdf"

					if os.path.isfile(file_data):
						continue

					ama = Amasijo(
								phasespace_args=phasespace_args(distance),
								isochrones_args=isochrones_args(age,distance),
								binary_args= binary_args(model),
								seed=seed+seed_offset)
					ama.generate_cluster(file_data,
								n_stars=n_stars,
								angular_correlations=None)
					# ama.plot_cluster(
					# 			file_plot=file_plot)