"""
gradients.py
=========================================
Python module containing Gradient classes
=========================================

This module contains gradient classes which calculate gradients for neural
net backpropagation quickly. A new, unique gradient class must be constructed
for neural networks with different numbers of layers, and/or different orderings
of activation functions. 
(New gradient classes do not need to be created for neural networks that only differ 
in the number of units per layer.)
"""

import numpy as np
import activationfns as af

    
class tanhsig2layer:
    """
    Gradient class for a two layer neural network. The first layer employs a tanh 
    activation function, the second layer employs a sigmoid activation function, and the
    neural network uses a cross-entropy loss function.
    """
    
    def __init__(self,verbose=False):
        self.V = None
        self.W = None
        self.h = None
        self.z = None
        self.X = None
        self.y = None
        self.grad_WL = None
        self.grad_VL = None
        self.verbose = verbose
    
    
    def prepare(self,data,labels,noutunits):
        self.X = data
        self.y = np.zeros((len(labels),noutunits))
        for l in range(len(labels)):
            self.y[l,int(labels[l,0])-1] += 1
        
        
    def calculate(self,weight_matrices,layeroutputs,labelrange):
        c = 0
        for listset in [weight_matrices,layeroutputs]:
            if len(listset) != 2:
                if len(listset) > 2:
                    estring = 'More'
                elif len(listset) < 2:
                    estring = 'Less'
                if not c:
                    lstring = 'weight matrices'
                else:
                    lstring = 'layer outputs'
                print('ERROR: This gradient is for a two-layer neural network. %s than two %s were provided.' %(estring,lstring))
                return
            c+=1

        self.V = weight_matrices[0]
        self.W = weight_matrices[1]
        self.h = layeroutputs[0]
        self.z = layeroutputs[1]
        
        rangemin,rangemax = labelrange[0],labelrange[1]
        X = self.X[rangemin:rangemax]
        y = self.y[rangemin:rangemax]
        Q = self.z - y.T
        n = len(y)

        self.grad_VL = np.zeros_like(self.V)
        self.grad_WL = np.zeros_like(self.W)
        for i in range(n):
            S = np.array([1/np.square(np.cosh(np.dot(self.V,X[i])))]).T
            X_iTranspose = np.array([X[i]])
            self.grad_VL += np.dot((np.dot(self.W.T,Q)[:-1]*S),X_iTranspose)
            self.grad_WL += np.outer(Q,self.h[:,i])
        
        # Optional output of intermediate steps
        if self.verbose:
            print('V\n',self.V)
            print('W\n',self.W)
            print('h\n',self.h)
            print('z\n',self.z)
            print('Q\n',Q)
            print('S\n',S)
            print('X\n',X)
            print('Grad_VL\n',self.grad_VL)
            print('Grad_WL\n',self.grad_WL)
            
        return self.grad_VL,self.grad_WL
        
        
                   

