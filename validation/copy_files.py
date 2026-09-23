import os

age_ranges = ["15-25Myr","20-220Myr","20-100Myr","100-220Myr","200-600Myr","200-400Myr"]

dir_syn = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/{0}/{1}/"
flags = '--exclude="*.nc" --exclude="*.pkl"'
base_copy= "rsync -arv {2} {0} {1}"

for age_range in age_ranges:
	for model in ["base"]:
		print(30*"-"+" "+model+" "+30*"-")

		#------------- Copy results ----------------------------------------------------
		tmp_org = "phanocles:{0}*".format(dir_syn.format(age_range,model))
		tmp_trg = "{0}".format(dir_syn.format(age_range,model))
		command = base_copy.format(tmp_org,tmp_trg,flags)
		# print(command)
		os.system(command)
		#--------------------------------------------------------------------------------------
		