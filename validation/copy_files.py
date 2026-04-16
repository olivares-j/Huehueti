import os


dir_res = "/home/jolivares/Repos/Huehueti/validation/synthetic/PARSEC/20-220Myr/{0}/"
flags = '--exclude="*.nc" --exclude="*.pkl"'
base_copy= "rsync -arv {2} {0} {1}"


for name in ["base","outliers"]:
	print(30*"-"+" "+name+" "+30*"-")

	#------------- Copy results ----------------------------------------------------
	tmp_org = "phanocles:{0}*".format(dir_res.format(name))
	tmp_trg = "{0}".format(dir_res.format(name))
	command = base_copy.format(tmp_org,tmp_trg,flags)
	# print(command)
	os.system(command)
	#--------------------------------------------------------------------------------------
		