import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import arviz as az

dir_base = "/home/jolivares/Repos/Huehueti@phanocles/validation/real"
dir_fig = "/home/jolivares/Dropbox/MisArticulos/BayesianAges/Isochrones/Method/Figures/"

SYSTEMS = {
"BPIC":{
	"literature":[
				{"author":"Barrado y Navascués et al. (1999)","age":20. ,"lower":10    ,"upper":10.   ,"method":"Isochrone"},
				{"author":"Zuckerman et al. (2001)"          ,"age":12. ,"lower":4     ,"upper":8.    ,"method":"Isochrone"},
				{"author":"Ortega et al. (2002)"             ,"age":11.5,"lower":10    ,"upper":10.   ,"method":"Traceback"},
				{"author":"Song et al. (2003)"               ,"age":12. ,"lower":np.nan,"upper":np.nan,"method":"Traceback"},
				{"author":"Ortega et al. (2004)"             ,"age":10.8,"lower":0.3   ,"upper":0.3   ,"method":"Traceback"},
				{"author":"Torres et al. (2006)"             ,"age":18. ,"lower":np.nan,"upper":np.nan,"method":"Expansion"},
				{"author":"Makarov (2007)"                   ,"age":31. ,"lower":21    ,"upper":21.   ,"method":"Traceback"},
				{"author":"Mentuch et al. (2008)"            ,"age":21. ,"lower":9.    ,"upper":9.    ,"method":"LDB"},
				{"author":"Macdonald & Mullan (2010)"        ,"age":40. ,"lower":np.nan,"upper":np.nan,"method":"LDB"},
				{"author":"Yee & Jenses (2010)"              ,"age":30. ,"lower":30    ,"upper":0     ,"method":"LDB"},
				{"author":"Binks & Jeffries (2014)"          ,"age":21. ,"lower":4     ,"upper":4.    ,"method":"LDB"},
				{"author":"Malo et al. (2014)"               ,"age":26. ,"lower":3     ,"upper":3.    ,"method":"LDB"},
				{"author":"Malo et al. (2014)"               ,"age":21.5,"lower":6.5   ,"upper":6.5   ,"method":"Isochrone"},
				{"author":"Mamajek & Bell (2014)"            ,"age":22. ,"lower":3     ,"upper":3.    ,"method":"Isochrone"},
				{"author":"Mamajek & Bell (2014)"            ,"age":21. ,"lower":5     ,"upper":10    ,"method":"Expansion"},
				{"author":"Bell et al. (2015)"               ,"age":24. ,"lower":3     ,"upper":3.    ,"method":"Isochrone"},
				{"author":"Herczeg & Hillenbrand (2015)"     ,"age":22. ,"lower":4.    ,"upper":4.    ,"method":"Isochrone"},
				{"author":"Binks & Jeffries (2016)"          ,"age":21. ,"lower":4     ,"upper":4     ,"method":"LDB"},
				{"author":"Binks & Jeffries (2016)"          ,"age":24. ,"lower":4     ,"upper":4     ,"method":"LDB"},
				{"author":"Messina et al. (2016)"            ,"age":25. ,"lower":3     ,"upper":3.    ,"method":"LDB"},
                {"author":"Shkolnik et al. (2017)"           ,"age":22. ,"lower":6     ,"upper":6     ,"method":"LDB"},
				{"author":"Miret-Roig et al. (2018)"         ,"age":13. ,"lower":0     ,"upper":7.    ,"method":"Traceback"},
				{"author":"Crundall et al. (2019)"           ,"age":18.3,"lower":1.2   ,"upper":1.3   ,"method":"Forward-modelling"},
				{"author":"Ujjwal et al. (2020)"             ,"age":19.4,"lower":13.8  ,"upper":35.1  ,"method":"Isochrone"},
				{"author":"Miret-Roig et al. (2020)"         ,"age":18.5,"lower":2.4   ,"upper":2.0   ,"method":"Traceback"},
				{"author":"Galindo-Guil et al. (2022)"       ,"age":24.3,"lower":0.3   ,"upper":0.3   ,"method":"LDB"},
                {"author":"Lee et al. (2022)"                ,"age":9.4 ,"lower":4.9   ,"upper":4.9   ,"method":"Isochrone"},
				{"author":"Couture et al. (2023)"            ,"age":20.4,"lower":2.5   ,"upper":2.5   ,"method":"Traceback"},
                {"author":"Jeffries et al. (2023)"           ,"age":23  ,"lower":1     ,"upper":1     ,"method":"LDB"},
                {"author":"Jeffries et al. (2023)"           ,"age":25  ,"lower":1     ,"upper":1     ,"method":"LDB"},
				{"author":"Lee et al. (2024)"                ,"age":33  ,"lower":9     ,"upper":11    ,"method":"Isochrone"},
				{"author":"Lee et al. (2024)"                ,"age":23  ,"lower":8     ,"upper":8     ,"method":"LDB"},
				{"author":"Luhman (2024)"                    ,"age":24.7,"lower":0.6   ,"upper":0.9   ,"method":"LDB"},
				{"author":"Olivares et al. (2025)"           ,"age":23.4,"lower":4.8   ,"upper":4.8   ,"method":"Expansion"},
				{"author":"This work"                        ,"age":19.4,"lower":1.2   ,"upper":1.2   ,"method":"Isochrone",
				"final":"{0}/BetaPictoris/Miret-Roig+2020_<7.0G_linear_dispersion_l2_FullRank_ADVI_Exponential/".format(dir_base)},
				],
	"figsize":(6,10),
	"loc_legend":{"method":"upper right","samples":"lower right"}
	},
"Hyades":{
	"literature":[
				{"author":"De Gennaro et al. (2009)"         ,"age":648. ,"lower":45.  ,"upper":45.    ,"method":"WD"},
				{"author":"Gaia Collaboration (2018)"        ,"age":794. ,"lower":102. ,"upper":161.   ,"method":"Isochrone"},
				{"author":"Martín et al. (2018)"             ,"age":650. ,"lower":70.  ,"upper":70.    ,"method":"LDB"},
				{"author":"Lodieu et al. (2019)"             ,"age":640. ,"lower":49.  ,"upper":67.    ,"method":"WD"},
				{"author":"Galindo-Guil et al. (2022)"       ,"age":695. ,"lower":67   ,"upper":85     ,"method":"LDB"},
				{"author":"Brandner et al. (2023)"           ,"age":775  ,"lower":25   ,"upper":25     ,"method":"Isochrone"},
				{"author":"Hunt & Reffert (2024) "           ,"age":577  ,"lower":222  ,"upper":413    ,"method":"Isochrone"},
				{"author":"This work"                        ,"age":780  ,"lower":42   ,"upper":42     ,"method":"Isochrone",
				"final":"{0}/Hyades/Meingast+2019_<4.5G_linear_dispersion_l2_FullRank_ADVI_Exponential/".format(dir_base)},
				],
	"figsize":(6,5),
	"loc_legend":{"method":"upper left","samples":"upper right"}
	},
"Pleiades":{
	"literature":[
				{"author":"Naylor (2009)"                    ,"age":115.  ,"lower":11.   ,"upper":3.    ,"method":"Isochrone"},
				{"author":"Bell et al. (2014)"               ,"age":135.  ,"lower":11.    ,"upper":20.  ,"method":"Isochrone"},
				{"author":"Cargile et al. (2014)"            ,"age":134.  ,"lower":10.   ,"upper":9.    ,"method":"Gyrochronology"},
				{"author":"Lodieu et al. (2019)"             ,"age":132.  ,"lower":27.   ,"upper":26.   ,"method":"WD"},
				{"author":"Bossini et al. (2019)"            ,"age":86.7  ,"lower":1.2   ,"upper":1.0   ,"method":"Isochrone"},
				{"author":"Galindo-Guil et al. (2022)"       ,"age":127.4 ,"lower":10.   ,"upper":6.3   ,"method":"LDB"},
				{"author":"Hunt & Reffert (2024) "           ,"age":121   ,"lower":66    ,"upper":75    ,"method":"Isochrone"},
				{"author":"Frasca et al. (2025)"             ,"age":118   ,"lower":6.    ,"upper":6.    ,"method":"LDB"},
				{"author":"Gónzales-Ramírez et al. (2026)"   ,"age":124.5 ,"lower":2.7   ,"upper":3.3   ,"method":"LDB"},
				{"author":"This work"                        ,"age":127.5 ,"lower":9.9   ,"upper":9.9   ,"method":"Isochrone",
				"final":"{0}/Pleiades/Meingast+2021_<4.5G_linear_dispersion_l2_FullRank_ADVI_Exponential_intercept:20_slope:5/".format(dir_base)},
				],
	"figsize":(6,5),
	"loc_legend":{"method":"upper left","samples":"upper right"}
	},
"Blanco1":{
	"literature":[
				{"author":"Bossini et al. (2019)"            ,"age":94    ,"lower":7     ,"upper":5      ,"method":"Isochrone"},
				{"author":"Cargile et al. (2014)"            ,"age":146.  ,"lower":14.   ,"upper":13.    ,"method":"Gyrochronology"},
				{"author":"Juarez et al. (2014)"             ,"age":114.  ,"lower":10.   ,"upper":9.     ,"method":"LDB"},
				{"author":"Galindo-Guil et al. (2022)"       ,"age":137.1 ,"lower":33.   ,"upper":7.     ,"method":"LDB"},
				{"author":"Hunt & Reffert (2024) "           ,"age":173   ,"lower":84    ,"upper":197    ,"method":"Isochrone"},
				{"author":"This work"                        ,"age":125.4 ,"lower":8     ,"upper":8      ,"method":"Isochrone",
				"final":"{0}/Blanco1/Meingast+2021_<4.5G_linear_dispersion_l2_FullRank_ADVI_Exponential/".format(dir_base)},
				],
	"figsize":(6,4),
	"loc_legend":{"method":"upper right","samples":"lower right"}
	},
}

# Step 1: Create a sample dataset
np.random.seed(42)
n_samples = 1000

palette={'Isochrone': 'tab:blue', 
		'Traceback': 'tab:orange',
		'Expansion':'tab:green',
		'LDB': 'tab:red',
		"Forward-modelling":'tab:purple',
		"WD":"tab:brown",
		"Gyrochronology":"tab:cyan",
		"Lithium limit":"tab:pink",
		"Expansion limit":"tab:grey",
		"Spectroscopic limit":"tab:olive"}
markers={'Isochrone': '.', 
		'Traceback': 'X', 
		'Expansion':'*',
		'LDB': 'P',
		"Forward-modelling":'d',
		"WD":"p",
		"Gyrochronology":"s",
		"Lithium limit":"<",
		"Expansion limit":">",
		"Spectroscopic limit":"<"}

for name,values in SYSTEMS.items():
	print(name)
	data = values['literature']

	# Data points for three categories with one data point each and associated errors
	single_data = pd.DataFrame({
		'Age [Myr]': [a["age"] for a in data],
		'Literature studies': [a["author"] for a in data],
		'Method': [a["method"] for a in data],  # Updated Method category
		'errors_low': [a["lower"] for a in data],  # Lower error
		'errors_high': [a["upper"] for a in data]  # Upper error
	})

	# Additional data for 'Category 2' to overlay the violin plot
	additional_data = []
	for wrk in data:
		if "final" in wrk.keys():
			posterior = az.from_netcdf(wrk["final"]+"Chains.nc").posterior
			prior = az.from_netcdf(wrk["final"]+"Prior.nc").prior
			pos_samples = posterior["age"].as_numpy().values.flatten()
			pri_samples = prior["age"].as_numpy().values.flatten()
			additional_data.append(pd.DataFrame({
				'Age [Myr]': np.random.choice(pos_samples,size=n_samples),
				'Literature studies': [wrk["author"]] * n_samples,
				'Samples': ['Posterior'] * n_samples  # Updated Method category
				}))
			additional_data.append(pd.DataFrame({
				'Age [Myr]': np.random.choice(pri_samples,size=n_samples),
				'Literature studies': [wrk["author"]] * n_samples,
				'Samples': ['Prior'] * n_samples  # Updated Method category
				}))
	additional_data = pd.concat(additional_data,ignore_index=False)

	# Combine the datasets
	data = pd.concat([single_data, additional_data], ignore_index=True)

	# Step 2: Create the scatter plot with error bars
	plt.figure(figsize=values['figsize'])  # Set figure size

	# Scatter plot with legend based on "Method" category
	sns.scatterplot(data=single_data, x='Age [Myr]', y='Literature studies', hue='Method', style='Method', 
					palette=palette,markers=markers,s=200, zorder=2)

	# Add error bars for the single data points
	for idx, row in single_data.iterrows():
		plt.errorbar(x=row['Age [Myr]'], y=row['Literature studies'], 
					 xerr=[[row['errors_low']], [row['errors_high']]], 
					 fmt='none', capsize=5, color='silver', zorder=1)

	# Step 3: Add a violin plot for 'Category 2' without including "Method"
	alpha= 1.0
	sns.violinplot(data=additional_data, 
				   x='Age [Myr]', 
				   y='Literature studies',
				   hue="Samples",
				   palette={"Posterior":"red","Prior":"black"},
				   split=True,
				   legend=False,
				   inner=None,
				   edgecolor=None,
				   alpha=alpha,
				   zorder=0)
	
	handles = [Rectangle((0,0),1,1,color="red",alpha=alpha),Rectangle((0,0),1,1,color="black",alpha=alpha)]
	labels= ["Posterior","Prior"]
	age_leg = plt.legend(handles, labels,title="Samples",loc=values["loc_legend"]["samples"])
	plt.gca().add_artist(age_leg)
	plt.legend(title='Method', loc=values["loc_legend"]["method"])  # Set legend with title and position

	plt.title('Age estimates of {0}'.format(name))
	plt.savefig(dir_fig +"{0}_lit.png".format(name),dpi=300,bbox_inches='tight')
	plt.close()




