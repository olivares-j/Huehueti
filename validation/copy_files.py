import os

age_ranges = ["15-25Myr","20-220Myr"]

dir_res = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/{0}/{1}/"
flags = '--exclude="*.nc" --exclude="*.pkl"'
base_copy= "rsync -arv {2} {0} {1}"

for age_range in age_ranges:
	for model in ["base","outliers"]:
		print(30*"-"+" "+model+" "+30*"-")

		#------------- Copy results ----------------------------------------------------
		tmp_org = "phanocles:{0}*".format(dir_res.format(age_range,model))
		tmp_trg = "{0}".format(dir_res.format(age_range,model))
		command = base_copy.format(tmp_org,tmp_trg,flags)
		# print(command)
		os.system(command)
		#--------------------------------------------------------------------------------------
		