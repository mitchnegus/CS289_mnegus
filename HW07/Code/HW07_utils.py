#HW07_utils.py
#-----------------------------------------
# Python module for CS289A HW07
#-----------------------------------------
#-----------------------------------------


import numpy as np
from scipy import io as spio


def load_data(datapath,BASE_DIR,dictkey):
#Load data
    data_dict = spio.loadmat(BASE_DIR+datapath,mat_dtype=True)
    data = data_dict[dictkey]

    return data


def shuffle_data(data,labels):
    datlbl = np.concatenate((data,labels),axis=1)
    np.random.shuffle(datlbl)
    shuffleddata = datlbl[:,:-1]
    shuffledlabels = datlbl[:,-1]
    
    return shuffleddata,shuffledlabels


def val_partition(data,valfrac):
    # Separate <valsetsize> items for validation
    valsetsize = int(valfrac*len(data))
    valset = data[:valsetsize]
    trainset = data[valsetsize:]
    
    return trainset,valset
   
    
def score_accuracy(predictions,truelabels):
    count,total = 0,0
    for i in range(len(predictions)):
        if predictions[i] == truelabels[i]:
            count += 1
        total += 1
    Acc = count/total
    
    return Acc


