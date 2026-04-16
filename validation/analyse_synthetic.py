'''
Copyright 2023 Javier Olivares Romero

This file is part of Kalkayotl.

    Kalkayotl is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyAspidistra.  If not, see <http://www.gnu.org/licenses/>.
'''
#------------ LOAD LIBRARIES -------------------
import os
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter 
import seaborn as sns

pn.set_option('display.max_rows', None) 

# base_dir = "Optuna_logAge_logL_epochs_5e+02_0.1myr_l3/"
# base_dir = "Optuna_logAge_Mini_epochs_5e+02_0.1myr_l2/"
# base_dir = "Optuna_logAge_Mini_logTe_epochs_5e+02_0.1myr_l2_FullRankADVI/"
# base_dir = "Optuna_logAge_logL_epochs_5e+02_0.1myr_l2_FullRankADVI/"
base_dir = "Optuna_logAge_logL_epochs_5e+02_0.1myr_l2_FullRankADVI/"

dir_base = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/20-220Myr/"
dir_fig  = "/home/jolivares/Dropbox/MisArticulos/BayesianAges/Isochrones/Method/Figures/20-220Myr/"
dir_fig += base_dir
os.makedirs(dir_fig,exist_ok=True)


models = ["base","outliers"]

list_of_ages      = list(range(20,240,20))
list_of_distances = [50,100,200,400]
list_of_n_stars   = [15]
list_of_seeds     = [0,1,2,3,4]

do_process = True
do_plt_grp = True
do_plt_src = True

file_data_all = dir_base + "data_models.h5"
file_plt_grp  = dir_fig + "group-level.pdf"
file_plt_src  = dir_fig + "source-level.pdf"


base_dir_out = "{0}{1}/"+base_dir
base_dir_in  = "{0}{1}/inputs/"
base_obs_grp = "{0}/{1}/Global_statistics.csv"
base_obs_src = "{0}/{1}/Sources_statistics.csv"
base_syn_src = "{0}/{1}.csv"
base_dat     = "{0}{1}/data.h5"
base_name    = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"
#---------------------------------------------------------------------------

coordinates = ["X","Y","Z","U","V","W"]
obs_grp_columns = ["Parameter","mean","sd","hdi_2.5%","hdi_97.5%","r_hat","ess_bulk","ess_tail"]
true_src_columns = ["source_id","logL"]
obs_src_columns = ["source_id","statistic","log_lum"]
mapper_true2obs = {"logL":"log_lum"}

#-----------------------------------------------------------------------------

#------------------------Statistics -----------------------------------
sts_grp = [
		{"key":"err",     "name":"Error [%]",       "ylim":[-20,20]},
		{"key":"unc",     "name":"Uncertainty [%]", "ylim":[0,5]   },
		{"key":"crd",     "name":"Credibility [%]", "ylim":[0,100]   },
		{"key":"r_hat",   "name":"$\\hat{R}$",      "ylim":[0.9,1.5]     },
		{"key":"ess_bulk","name":"ESS bulk",        "ylim":[0,None]     },
		{"key":"ess_tail","name":"ESS tail",        "ylim":[0,None]     },
		]
sts_src = [
		{"key":"err", "name":"Error [%]"      ,"ylim":[-3,10]},
		{"key":"unc", "name":"Uncertainty [%]","ylim":[0,10]   },
		{"key":"crd", "name":"Credibility [%]","ylim":[0,100]   }
		]
#-----------------------------------------------------------------------

#-----------------------------------------------------


if do_process:
	dfss_grp = []
	dfss_src = []
	for model in models:
		print(40*"+" +" " + model + " " +40*"+")
		dir_out   = base_dir_out.format(dir_base,model)
		dir_in    = base_dir_in.format(dir_base,model)
		file_data = base_dat.format(dir_base,model)
		dfs_grp = []
		dfs_src = []
		for age in list_of_ages:
			for distance in list_of_distances:
				for n_stars in list_of_n_stars:
					for seed in list_of_seeds:
						name = base_name.format(age,distance,n_stars,seed)
						print(40*"-" +" " + name + " " +40*"-")

						#------------- Files ---------------------------------
						file_obs_grp   = base_obs_grp.format(dir_out,name)
						file_obs_src   = base_obs_src.format(dir_out,name)
						file_true_src  = base_syn_src.format(dir_in,name)
						#-----------------------------------------------------

						#-------------------- Observed values -------------------------------
						df_obs_src = pn.read_csv(file_obs_src, usecols=obs_src_columns)
						df_obs_src.set_index(["source_id","statistic"],inplace=True)
						df_obs_src = df_obs_src.unstack()
						
						df_obs_grp = pn.read_csv(file_obs_grp,usecols=obs_grp_columns)
						df_obs_grp.set_index("Parameter",inplace=True)
						#------------------------------------------------------------------

						#-------------- True values -----------------------
						df_true_grp = pn.DataFrame.from_dict(
											data={"age":age},
											orient="index",
											columns=["true"])
						df_true_grp.index.name = "Parameter"
						df_true_src = pn.read_csv(file_true_src,
											usecols=true_src_columns)
						df_true_src.set_index("source_id",
											inplace=True)
						df_true_src.rename(columns=mapper_true2obs,
											inplace=True)
						df_true_src.columns = pn.MultiIndex.from_product(
											[df_true_src.columns,["true"]])
						#----------------------------------------------------

						#---------- Join ------------------------
						df_grp = pn.merge(
										left=df_true_grp,
										right=df_obs_grp,
										left_index=True,
										right_index=True)
						df_src = pn.merge(
										left=df_obs_src,
										right=df_true_src,
										left_index=True,
										right_index=True)
						#----------------------------------------

						#-------------- Asses convergence ------------------
						if any(df_grp["r_hat"]>1.05):
							par = df_grp.loc[df_grp["r_hat"]>1.05]
							print("WARNING: Convergence issues at:")
							print(par)
							if any(df_grp["r_hat"]>1.5):
								par = df_grp.loc[df_grp["r_hat"]>1.5]
								print("Error: Convergence issues at:")
								print(par)
								print(300*"<")
								# sys.exit()
						#----------------------------------------------------

						#---------------------------- Error, Uncertainty and Credibility ------------------------
						df_src.columns = df_src.columns.droplevel(0)
						for tmp in [df_src,df_grp]:
							tmp["err"] = tmp.apply(lambda x: 100.*(x["mean"] - x["true"])/x["true"],axis = 1)
							tmp["unc"] = tmp.apply(lambda x: 100.*(x["sd"]/np.abs(x["true"])),  axis = 1)
							tmp["crd"] = tmp.apply(lambda x: 100.*((x["true"] >= x["hdi_2.5%"]) & 
																		 (x["true"] <= x["hdi_97.5%"])),
																		axis = 1)
						#------------------------------------------------------------------------------------------

						#-------------- Select variables -----------------
						df_grp = df_grp.loc[:,["err","unc","crd","r_hat","ess_bulk","ess_tail"]]
						df_src = df_src.loc[:,["err","unc","crd"]]
						#-------------------------------------------------

						#------------ Model -------------
						df_grp["Model"] = model
						df_src["Model"] = model

						df_grp["n_stars"] = n_stars
						df_src["n_stars"] = n_stars

						df_grp["distance"] = distance
						df_src["distance"] = distance

						df_grp["seed"] = seed
						df_src["seed"] = seed

						df_grp["age"] = age
						df_src["age"] = age
						#--------------------------------

						df_grp.reset_index(inplace=True,drop=True)
						df_src.reset_index(inplace=True)

						df_grp.set_index(["Model","age","distance","n_stars","seed"],inplace=True)
						df_src.set_index(["Model","age","distance","n_stars","seed"],inplace=True)

						#----------- Append ----------------
						dfs_grp.append(df_grp)
						dfs_src.append(df_src)
						#------------------------------------

		#------------ Concatenate --------------------
		df_grp = pn.concat(dfs_grp,ignore_index=False)
		df_src = pn.concat(dfs_src,ignore_index=False)
		#---------------------------------------------

		#------------ Save data --------------------------
		df_grp.to_hdf(file_data,key="df_grp")
		df_src.to_hdf(file_data,key="df_src")
		#-------------------------------------------------

		#----------- Append ----------------
		dfss_grp.append(df_grp)
		dfss_src.append(df_src)
		#------------------------------------

	#------------ Concatenate --------------------
	df_grp = pn.concat(dfss_grp,ignore_index=False)
	df_src = pn.concat(dfss_src,ignore_index=False)
	#---------------------------------------------

	#------------ Save data --------------------------
	df_grp.to_hdf(file_data_all,key="df_grp")
	df_src.to_hdf(file_data_all,key="df_src")
	#-------------------------------------------------

#=========================== Plots =======================================
if do_plt_grp:
	print("Plotting group-level parameters")
	#------------ Read data --------------------------------
	df_grp = pn.read_hdf(file_data_all,key="df_grp")
	#-------------------------------------------------------

	df_grp.reset_index(inplace=True)

	pdf = PdfPages(filename=file_plt_grp)
	for st in sts_grp:
		fg = sns.relplot(data=df_grp,
						x="age",
						y=st["key"],
						col="distance",
						style="Model",
						hue="n_stars",
						kind="line",
						palette="tab10",
						facet_kws={"margin_titles":True},
						)
		fg.set_xlabels("Age [Myr]")
		fg.set_ylabels(st["name"])
		fg.set(ylim=st["ylim"])
		fg.add_legend()
		sns.move_legend(fg,
				loc="lower center",
				bbox_to_anchor=(.45, 1),
				ncol=6)
		plt.subplots_adjust(wspace=0.1)
		pdf.savefig(bbox_inches='tight',dpi=300)
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------

if do_plt_src:
	print("Plotting source-level parameters")
	#------------ Read data --------------------------------
	df_src = pn.read_hdf(file_data_all,key="df_src")
	#-------------------------------------------------------

	df_src.reset_index(inplace=True)

	
	pdf = PdfPages(filename=file_plt_src)
	for st in sts_src:
		fg = sns.relplot(data=df_src,
						x="age",
						y=st["key"],
						col="distance",
						style="Model",
						hue="n_stars",
						kind="line",
						palette="tab10",
						facet_kws={"margin_titles":True},
						)
		fg.set_xlabels("Age [Myr]")
		fg.set_ylabels(st["name"])
		fg.set(ylim=st["ylim"])
		fg.add_legend()

		sns.move_legend(fg,
				loc="lower center",
				bbox_to_anchor=(.45, 1),
				ncol=5)
		plt.subplots_adjust(wspace=0.1)
		pdf.savefig(bbox_inches='tight')
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------
