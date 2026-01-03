import numpy as np
from Amasijo import Amasijo

model     = "PARSEC"
dir_main  = "/home/jolivares/Repos/Huehueti/validation/synthetic/{0}/inputs/".format(model)
file_mlp_phot = "/home/jolivares/Repos/Huehueti/mlps/{0}/GP2_l9_s512/mlp.pkl".format(model)
file_mlp_mass = "/home/jolivares/Repos/Huehueti/mlps/{0}/mTg_l7_s256/mlp.pkl".format(model)
base_name = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"

list_of_ages = [20.0,30.,40.,50.,60.,70.,80.,90.,100.,120.,140.,160.,180.200.]
distance  = 136.0
seed      = 0
n_stars   = 50
theta_limits = [0.05,1.0]

def phasespace_args(distance):
	args = {
	"position":{"family":"Gaussian",
				"location":np.array([distance,0.0,0.0]),
				"covariance":np.diag([9.,9.,9.])},
	"velocity":{"family":"Gaussian",
				"location":np.array([10.0,10.0,10.0]),
				"covariance":np.diag([1.,1.,1.]),
				"kappa":np.ones(3),
				"omega":np.array([[-1,-1,-1],[1,1,1]])
				}}
	return args

def isochrones_args(model,age,theta_limits):
	args = {
		"model":model,
		"age": age,
		"PARSEC_args":{
				"file_mlp_phot":file_mlp_phot,
				"file_mlp_mass":file_mlp_mass,
				"theta_limits":theta_limits
				},          
		"bands":["G","BP","RP","gP1", "rP1", "iP1", "zP1", "yP1","J", "H", "Ks"],
		}
	return args

for age in list_of_ages:
	file_data = dir_main + base_name.format(age,distance,n_stars,seed) + ".csv"
	file_plot = dir_main + base_name.format(age,distance,n_stars,seed) + ".pdf"
	ama = Amasijo(
				phasespace_args=phasespace_args(distance),
				isochrones_args=isochrones_args(model,age,theta_limits),
				seed=seed)
	ama.generate_cluster(file_data,n_stars=n_stars,angular_correlations=None)
	ama.plot_cluster(file_plot=file_plot)