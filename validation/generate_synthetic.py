import os
import numpy as np
from Amasijo import Amasijo

model      = "outliers"
age_range  = "15-25Myr"
max_Av     = 0.0

dir_inputs = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/{0}/{1}/inputs/".format(age_range,model)
base_name = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

if age_range == "15-25Myr":
	list_of_ages = list(range(15,27,2))
elif age_range == "20-220Myr":
	list_of_ages = list(range(20,240,20))
elif age_range == "200-600Myr":
	list_of_ages = list(range(200,650,50))
else:
	sys.exit("Undefined age range")

list_of_distances = [50,100,200,400]
list_of_n_stars   = [15,30,60]
list_of_seeds     = [0,1,2,3,4]

def mass_limits(age,distance):
	if distance == 50:
		if age == 15:
			return [0.,3.19]
		elif age == 17:
			return [0.,3.19]
		elif age == 19:
			return [0.,3.59]
		elif age == 21:
			return [0.,3.19]
		elif age == 23:
			return [0.,3.19]
		elif age == 25:
			return [0.,3.0]
		elif age == 20:
			return [0.,3.20]
		elif age == 40:
			return [0.,3.18]
		elif age == 60:
			return [0.,2.98]
		elif age == 80:
			return [0.,3.15]
		elif age == 100:
			return [0.,2.92]
		elif age == 120:
			return [0.,3.00]
		elif age == 140:
			return [0.,2.89]
		elif age == 160:
			return [0.,2.75]
		elif age == 180:
			return [0.,2.75]
		elif age == 200:
			return [0.,2.75]
		elif age == 220:
			return [0.,2.75]
	elif distance == 100:
		if age == 15:
			return [0.,5.8]
		elif age == 17:
			return [0.,5.8]
		elif age == 19:
			return [0.,5.8]
		elif age == 21:
			return [0.,5.4]
		elif age == 23:
			return [0.,5.8]
		elif age == 25:
			return [0.,5.3]
		elif age == 20:
			return [0.,5.7]
		elif age == 40:
			return [0.,5.3]
		elif age == 60:
			return [0.,4.8]
		elif age == 80:
			return [0.,4.6]
		elif age == 100:
			return [0.,4.5]
		elif age == 120:
			return [0.,4.2]
		elif age == 140:
			return [0.,4.0]
		elif age == 160:
			return [0.,3.8]
		elif age == 180:
			return [0.,3.7]
		elif age == 200:
			return [0.,3.6]
		elif age == 220:
			return [0.,3.5]
	elif distance == 200:
		if age == 15:
			return [0.,9.9]
		elif age == 17:
			return [0.,9.3]
		elif age == 19:
			return [0.,8.7]
		elif age == 21:
			return [0.,8.5]
		elif age == 23:
			return [0.,8.5]
		elif age == 25:
			return [0.,8.3]
		elif age == 20:
			return [0.,8.7]
		elif age == 40:
			return [0.,7.1]
		elif age == 60:
			return [0.,6.1]
		elif age == 80:
			return [0.,100.0]
		elif age == 100:
			return [0.,100.0]
		elif age == 120:
			return [0.,100.0]
		elif age == 140:
			return [0.,100.0]
		elif age == 160:
			return [0.,100.0]
		elif age == 180:
			return [0.,100.0]
		elif age == 200:
			return [0.,100.0]
		elif age == 220:
			return [0.,100.0]
	elif distance == 400:
		if age == 15:
			return [0.,12.5]
		elif age == 17:
			return [0.,11.8]
		elif age == 19:
			return [0.,11.2]
		elif age == 21:
			return [0.,10.7]
		elif age == 23:
			return [0.,11.8]
		elif age == 25:
			return [0.,100.0]
		elif age == 20:
			return [0.,10.9]
		elif age == 40:
			return [0.,100.0]
		elif age == 60:
			return [0.,100.0]
		elif age == 80:
			return [0.,100.0]
		elif age == 100:
			return [0.,100.0]
		elif age == 120:
			return [0.,100.0]
		elif age == 140:
			return [0.,100.0]
		elif age == 160:
			return [0.1,100.0]
		elif age == 180:
			return [0.1,100.0]
		elif age == 200:
			return [0.1,100.0]
		elif age == 220:
			return [0.1,100.0]
	else:
		sys.exit("No available distance!")

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

def isochrones_args(age,distance):
	args = {
	"model":"PARSEC",
	"age": float(age),
	"Av_limits":[0.0,max_Av],
	"mass_limits":mass_limits(age,distance),
	"MIST_args":{
		"metallicity":0.012,
		},
	"PARSEC_args":{
		"files":[
		"/home/jolivares/Models/PARSEC/{0}/Gaia_EDR3_0.025myr.dat".format(age_range),
		"/home/jolivares/Models/PARSEC/{0}/2MASS_0.025myr.dat".format(age_range),
		],
		"max_label":1,
		"bands_wavelengths":[6217.6,5109.7,7769.0,12350.,16620.,21590.], # Same order as bands
		"Rv":3.1
		},
	"bands":["G","BP","RP","J","H","Ks"],
	"uncertainties":[0.001,0.02,0.004,0.025,0.030,0.025]
	}
	return args


os.makedirs(dir_inputs,exist_ok=True)

for age in list_of_ages:
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
							seed=seed)
				ama.generate_cluster(file_data,
							n_stars=n_stars,
							angular_correlations=None)
				# ama.plot_cluster(
				# 			file_plot=file_plot)