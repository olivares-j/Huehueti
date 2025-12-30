import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.interpolate import CubicSpline,make_splprep


dir_base = "/home/jolivares/Repos/Huehueti/"
dir_data = dir_base + "data/"
dir_para  = dir_data + "parametrizations/"

do_plots = False
max_label = 1
min_par = 1e-6
feature = "Mini"
target  = "logTe"

file_in  = dir_data + "PARSEC_20-200Myr_GDR3+PanSTARRS+2MASS.csv"
file_out = dir_para + "parametrized_max_label_{0}_PARSEC_20-200Myr_GDR3+PanSTARRS+2MASS.csv".format(max_label)


features_input = ["age_Myr","Mini","logTe","logg","label"]
photometric_bands = ['Gmag', 'G_BPmag', 'G_RPmag', 'gP1mag','rP1mag','iP1mag', 'zP1mag', 'yP1mag','Jmag', 'Hmag', 'Ksmag']
columns = sum([features_input,photometric_bands],[])


#------------- Load data ------------------------
df_base = pd.read_csv(file_in,usecols=columns)
df_base = df_base[df_base["label"].le(max_label)]
dfg = df_base.groupby("age_Myr")
#------------------------------------------------
print(dfg.describe())

dfs = []
for name,df_tmp in dfg.__iter__():
	raw_feature = df_tmp[feature].to_numpy()
	raw_target  = df_tmp[target].to_numpy()
	raw_data    = df_tmp.loc[:,[feature,target]].to_numpy()

	spline,par = make_splprep(raw_data.T,k=3,s=1e-8)
	par[0] = min_par
	assert np.min(par) == min_par, "Error: there is a parameter value lower than min_par={0}".format(min_par)


	df_tmp.insert(loc=3,column="parameter",value=par)
	dfs.append(df_tmp)

	if do_plots:
		mass,teff = spline(np.linspace(min_par,1.,10000))
		fig, ax = plt.subplots()
		ax.scatter(raw_feature,raw_target, s = 3)
		ax.plot(mass, teff, label="S")
		plt.show()

df = pd.concat(dfs,ignore_index=True)
df.to_csv(file_out)



