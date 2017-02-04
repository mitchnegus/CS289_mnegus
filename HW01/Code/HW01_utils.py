# HW01_utils.py
#-----------------------------------------
# Python module for CS289A HW01
#-----------------------------------------
#-----------------------------------------


import numpy as np
from scipy import io as spio


def loaddata(shortpath,_DATA_DIR,dictkey):
#Load data
	data_dict = spio.loadmat(_DATA_DIR+"\\"+shortpath)
	data = np.array(data_dict[dictkey])
	return data


def partition(valsetsize,data):
# Separate <valsetsize> items for validation
	trainset = data[valsetsize:]
	valset = data[:valsetsize]
	return trainset,valset
    
    
def separatelabels(inset):
# Separate labels from sets
	setarrays = inset[:,:-1]
	setlabels = inset[:,-1]
	return setarrays,setlabels
			