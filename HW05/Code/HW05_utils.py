#HW05_utils.py
#-----------------------------------------
# Python module for CS289A HW05
#-----------------------------------------
#-----------------------------------------


import numpy as np
from scipy import io as spio


def loaddata(datapath,BASE_DIR,dictkey):
#Load data
    data_dict = spio.loadmat(BASE_DIR+"/"+datapath)
    data = data_dict[dictkey]

    return data

