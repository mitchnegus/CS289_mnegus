#HW03_utils.py
#-----------------------------------------
# Python module for CS289A HW03
#-----------------------------------------
#-----------------------------------------


import math
import numpy as np
from scipy import io as spio


def loaddata(shortpath,_DATA_DIR,dictkey):
#Load data
	data_dict = spio.loadmat(_DATA_DIR+"/"+shortpath)
	data = np.array(data_dict[dictkey])
	return data



