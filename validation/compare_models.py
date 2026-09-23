import os
import sys
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import seaborn as sns

dir_base = "/home/jolivares/Models/"

file_plt = "/home/jolivares/Dropbox/MisArticulos/BayesianAges/Isochrones/Method/Figures/Comparison.png"

list_of_models = [
	{"name":"PARSEC",
	"file":dir_base+"PARSEC/GaiaEDR3_10-1000myr.csv",
	"columns":["age_Myr","Mass","Gmag","label"],
	"query":"age_Myr > 10.0 & age_Myr <= 1000.0 & label <=1",
	"mapper":{"Mass":"mass"}
	},
	{"name":"BT-Settl",
	"file":dir_base+"BT-Settl/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv",
	"columns":["age_Myr","M/Ms","G"],
	"query":"age_Myr > 10.0 & age_Myr <= 1000.0",
	"mapper":{"M/Ms":"mass","G":"Gmag"}
	}
	]
masses = np.linspace(0.1,1.4,num=1000)


dfs = []
for model in list_of_models:
	#---------- Load models-------------------
	df_tmp = pn.read_csv(model["file"],
		usecols=model["columns"])
	df_tmp.rename(columns=model["mapper"],
		inplace=True)
	df_tmp = df_tmp.query(model["query"])
	#----------------------------------------

	#--------- Interpolate ----------------
	dfg = df_tmp.groupby("age_Myr")

	dfs_ages = []
	for age,tmp in dfg:
		gmag = np.interp(masses,tmp["mass"],tmp["Gmag"])

		df_age_tmp = pn.DataFrame(data={
			"mass":masses,"Gmag":gmag})
		df_age_tmp["age_Myr"] = age
		df_age_tmp["logAge"] = np.log10(age*1.e6)

		dfs_ages.append(df_age_tmp)

	df_model = pn.concat(dfs_ages,ignore_index=True)
	# df_model["Model"] = model["name"]
	df_model.set_index(["logAge","mass"],inplace=True)
	#------------------------------------------------------
	dfs.append(df_model)

# df = pn.concat(dfs_model,ignore_index=False)

df = pn.merge(dfs[0],dfs[1],how="left",
	left_index=True,right_index=True,
	suffixes=["_PARSEC","_BT-Settl"])

df["Diff_Gmag"] = df.apply(lambda x: x["Gmag_PARSEC"]-x["Gmag_BT-Settl"],axis=1)

#-----------------------------------------


fg = sns.relplot(data=df,
				x="Gmag_PARSEC",
				y="Diff_Gmag",
				# col="age_Myr",
				# col_wrap=5,
				hue="logAge",
				kind="line",
				# kind="scatter",
				# hue="seed",
				palette="viridis",
				# facet_kws={"margin_titles":True},
				# legend="full",
				height=3.0,
				aspect=1.5
				)
fg.set_xlabels("G [mag]")
fg.set_ylabels("$\\Delta$ G [mag]")
fg.set(ylim=[-0.35,0.35],xlim=[3,10])
# sns.move_legend(fg,
# 		loc="lower center",
# 		bbox_to_anchor=(.5, 1),
# 		ncol=2)
# plt.subplots_adjust(wspace=0.1)
fg.savefig(file_plt,bbox_inches='tight',dpi=300)
plt.close()