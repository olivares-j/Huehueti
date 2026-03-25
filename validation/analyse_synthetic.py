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


dir_base = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/50-150Myr/"
dir_fig  = "/home/jolivares/Dropbox/MisArticulos/BayesianAges/Isochrones/Method/Figures/"

case = "base"
cases = [case]

list_of_ages      = list(range(60,160,20))
list_of_distances = [50,100,200,400]
list_of_n_stars   = [20]
list_of_seeds     = [0,1,2,3,4]

do_process = True
do_plt_grp = True
do_plt_src = True

file_data_all = dir_base + "data_cases.h5"
file_plt_grp  = dir_fig + "{0}_group-level.pdf"
file_plt_src  = dir_fig + "{0}_source-level.pdf"


base_dir_out = "{0}{1}/Optuna_InverseTimeDecay_lrin_None_lrdr_None_bs_10000_epochs_5e+02_l2/"
base_dir_in  = "{0}{1}/inputs/"
base_obs_grp = "{0}/{1}/Global_statistics.csv"
base_obs_src = "{0}/{1}/Sources_statistics.csv"
base_syn_src = "{0}/{1}.csv"
base_dat     = "{0}{1}/data.h5"
base_name    = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"
#---------------------------------------------------------------------------

coordinates = ["X","Y","Z","U","V","W"]
true_src_columns = ["source_id","teff"]
obs_src_columns = ["source_id","statistic","tef"]
obs_grp_columns = ["Parameter","mean","sd","hdi_2.5%","hdi_97.5%","r_hat","ess_bulk","ess_tail"]
#-----------------------------------------------------------------------------

#------------------------Statistics -----------------------------------
sts_grp = [
		{"key":"err",     "name":"Error [%]",       "ylim":[-100,100]},
		{"key":"unc",     "name":"Uncertainty [%]", "ylim":[0,100]   },
		{"key":"crd",     "name":"Credibility [%]", "ylim":[0,100]   },
		{"key":"r_hat",   "name":"$\\hat{R}$",      "ylim":[0,5]     },
		{"key":"ess_bulk","name":"ESS bulk",        "ylim":[0,5]     },
		{"key":"ess_tail","name":"ESS tail",        "ylim":[0,5]     },
		]
sts_src = [
		{"key":"err", "name":"Error [%]"      ,"ylim":[-100,100]},
		{"key":"unc", "name":"Uncertainty [%]","ylim":[0,100]   },
		{"key":"crd", "name":"Credibility [%]","ylim":[0,100]   }
		]
#-----------------------------------------------------------------------

#-----------------------------------------------------


if do_process:
	dfss_grp = []
	dfss_src = []
	for case in cases:
		print(40*"+" +" " + case + " " +40*"+")
		dir_out   = base_dir_out.format(dir_base,case)
		dir_in    = base_dir_in.format(dir_base,case)
		file_data = base_dat.format(dir_base,case)
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
						df_true_src.set_index("source_id",inplace=True)
						df_true_src.columns = pn.MultiIndex.from_product(
											[["tef"],["true"]])
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

						#------------ Case -------------
						df_grp["Case"] = case
						df_src["Case"] = case

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

						df_grp.set_index(["Case","age","distance","n_stars","seed"],inplace=True)
						df_src.set_index(["Case","age","distance","n_stars","seed"],inplace=True)

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

	for case in cases:
		pdf = PdfPages(filename=file_plt_grp.format(case))
		for st in sts_grp:
			fg = sns.FacetGrid(data=df_grp,
							col="distance",
							# row="n_stars",
							sharey=True,
							sharex=True,
							margin_titles=True,
							# col_wrap=6,
							# hue="seed",
							hue="n_stars"
							)
			# fg.map(sns.scatterplot,"age",st["key"])
			fg.map(sns.lineplot,"age",st["key"])
			fg.set_xlabels("Age [Myr]")
			fg.set_ylabels(st["name"])
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

	for case in cases:
		pdf = PdfPages(filename=file_plt_src.format(case))
		for st in sts_src:
			fg = sns.FacetGrid(data=df_src,
							col="distance",
							sharey=True,
							sharex=True,
							margin_titles=True,
							col_wrap=6,
							hue="n_stars")
			fg.map(sns.lineplot,"age",st["key"])
			fg.set_xlabels("Age [Myr]")
			fg.set_ylabels(st["name"])
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
