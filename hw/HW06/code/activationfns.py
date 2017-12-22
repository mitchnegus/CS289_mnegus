"""
activationfns.py
=============================================
Python module containing activation functions
=============================================

This module contains a variety of activation functions which can be imported 
into a neural network using the NeuralNet class contained in neuralnet.py. 
Each function takes at least one numpy array as an argument and returns a 
numpy array.
"""

import numpy as np
import scipy.special as spsp

def sigmoid(x):
    """sigmoid function: s = 1/[1+e^(-x)]"""
    s = spsp.expit(x)

    return s


def tanh(x):
    """tanh function: s = [e^(x)-e^(-x)]/[e^(x)+e^(-x)]"""
    y = np.tanh(x)

    return(y)


