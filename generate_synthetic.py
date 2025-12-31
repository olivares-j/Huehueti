import numpy as np
from Amasijo import Amasijo

seed      = 0
n_stars   = 50
distance  = 136.0
age       = 150.0
min_theta = 0.01
dir_main  = "/home/jolivares/Repos/Huehueti/data/synthetic/"
dir_mlps  = "/home/jolivares/Repos/Huehueti/mlps/PARSEC/"
base_name = "Synthetic_a{0}_n{1}_d{2}_t{3}_s{4}".format(int(age),n_stars,int(distance),min_theta,seed)
file_plot = dir_main + base_name + ".pdf"
file_data = dir_main + base_name + ".csv"

phasespace_args = {
	"position":{"family":"Gaussian",
				"location":np.array([distance,0.0,0.0]),
				"covariance":np.diag([9.,9.,9.])},
	"velocity":{"family":"Gaussian",
				"location":np.array([10.0,10.0,10.0]),
				"covariance":np.diag([1.,1.,1.]),
				"kappa":np.ones(3),
				"omega":np.array([[-1,-1,-1],[1,1,1]])
				}}

isochrones_args = {
"model":"PARSEC",
"age": age,
#"MIST_args":{"metallicity":0.012,"Av": 0.0,"mass_limits":[0.1,4.5],},
"PARSEC_args":{
		"file_mlp_phot":dir_mlps+"GP2_l9_s512/mlp.pkl",
		"file_mlp_mass":dir_mlps+"mTg_l7_s256/mlp.pkl",
		"theta_limits":[min_theta,1.0]
		},          
 
"bands":["G","BP","RP","gP1", "rP1", "iP1", "zP1", "yP1","J", "H", "Ks"],
}

ama = Amasijo(
			phasespace_args=phasespace_args,
			isochrones_args=isochrones_args,
			seed=seed)
ama.generate_cluster(file_data,n_stars=n_stars,angular_correlations=None)
ama.plot_cluster(file_plot=file_plot)