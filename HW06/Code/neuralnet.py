"""
neuralnet.py
===============================================
Train a neural network and predict on test data
===============================================

This python module contains the neural network class, which can be trained on 
labeled data (input as numpy arrays: for data, an Nxd matrix with N rows 
corresponding to N sample points and d columns corresponding to d features; for 
labels, an N vector with labels corresponding to each of the N sample points)

**NOTE: Use of this class requires the activationfns.py and gradients.py modules
"""

import numpy as np


class NeuralNet:
    """
    Train and store a neural network, based on supplied training data. 
    Use this network to predict classifications.
    """

    def __init__(self,nlayers=3,unitsperlayer=None,actfns=[af.sigmoid,af.sigmoid],Gradients=None,verbose=False):
        """
        Initialize the neural network
        - nlayers:       the number of layers in the neural network (includes input and output layers)
        - unitsperlayer: a list specifying (in order) the number of units in all sequential layers except input
        - actfns:        a list specifying (in order) the activation function used by all sequential layers except input
        - Gradient:      a class providing optimized gradient calculations for the given sequence of  
                         activation functions
        - verbose:       a boolean for descriptive output
        """
        if unitsperlayer == None:
            unitsperlayer = 3*np.ones(nlayers)
        elif nlayers == len(unitsperlayer)+1:    
            self.nlayers = nlayers-1
            self.unitsperlayer = unitsperlayer 
        elif nlayers > len(unitsperlayer)+1:
            print('ERROR: The number of units per layer were not given for at least one layer.')
        elif nlayers < len(unitsperlayer)+1:
            print('ERROR: More layers were given units than were specified by input "nlayers".')
        if nlayers == len(actfns)+1:
            self.actfns = actfns
        elif nlayers > len(actfns)+1:
            print('ERROR: The activation function was not given for at least one layer.')
        elif nlayers < len(actfns)+1:
            print('ERROR: More activation functions were provided than specified by input "nlayers".')
        if Gradients == None:
            print('ERROR: A gradient generator class must be included.')
        self.gradients = Gradients
        self.weight_matrices = []
        
    
    def initialize_weights(self,shape,mu=0,var=1):
        """
        Initialize weight matrix from normal distribution.
        - shape: tuple specifying desired shape of weight matrix
        - mu:    mean value of normal distribution
        - var:   variance of normal distribution
        """
        weight_matrix = np.random.normal(loc=mu,scale=np.sqrt(var),size=shape)
        
        return weight_matrix
    
    
    def weight_matrix_shape(self,layer_n,nfeatures):
        """
        Create weight matrix with the proper number of rows and columns for this layer
        - n:         the layer which will employ an activation function on the 
                     product of the weight matrix and values
        - nfeatures: an integer specifying the number of features in the dataset
        """
        if layer_n != 0 and layer_n != range(self.nlayers)[-1]:
            WM_nrows = self.unitsperlayer[layer_n]-1
            WM_ncols = self.unitsperlayer[layer_n-1]
        elif layer_n == 0:
            WM_nrows = self.unitsperlayer[layer_n]-1
            WM_ncols = nfeatures
        else:
            WM_nrows = self.unitsperlayer[layer_n]
            WM_ncols = self.unitsperlayer[layer_n-1]
        return WM_nrows,WM_ncols
    
    
    def forward(self,data):
        """
        Perform forward pass through neural network by multiplying data by weights
        and enforcing a nonlinear activation function for each layer.
        - data:           Nxd numpy array with N sample points and d features
        - weightmatrices: ordered list of sequential weight matrices corresponding to layers
        - actfns:         ordered list of sequential activation functions corresponding to layers
                         (functions are defined in activationfuncs.py)
        Returns layeroutputs, a list of the outputs from each layer. The last entry
        is an CxN numpy array with hypotheses for each sample N_i being in class C_j.
        """
        H = data.T
        layeroutputs = []
        for i in range(self.nlayers):
            W = self.weight_matrices[i]
            actfn = self.actfns[i]
            H = actfn(np.dot(W,H))
            # If the layer is not the output layer, add a fictitious unit for bias terms
            if i != self.nlayers-1:
                fictu = np.array([np.ones_like(H[0])])
                H = np.concatenate((H,fictu),axis=0)
            layeroutputs.append(H)
        return layeroutputs
    
    
    def backward(self,layeroutputs,labelrange,gradients=None):
        """
        Perform backward pass through neural network by computing gradients of 
        input weight matrices with respect to the loss function comparing hypotheses 
        to true values. Classes for gradients are provided in gradients.py module 
        (a unique gradient class is required for neural networks with different 
        numbers of layers and/or different activation functions)
        """
        if gradients == None:
            Gradients = self.gradients
        gradients = Gradients.calculate(self.weight_matrices,layeroutputs,labelrange)
        
        return gradients
    
    
    def classify_outputs(self,finaloutputs):
        """
        Convert final outputs into classifications
        -finaloutputs: a CxN numpy array with hypotheses for each sample N_i being in
                       class C_j.
        Returns a 1D, length-N array with values corresponding to point classifications
        """
        if len(finaloutputs) == 1:
            classifications = np.around(finaloutputs[0]).astype(int)
        if len(finaloutputs) > 1:
            # Add one for 1-indexing in classification labels
            classifications = (np.argmax(finaloutputs,axis=0)+np.ones(len(finaloutputs[0]))).astype(int)
        return classifications
    
    
    def train(self,data,labels,epsilon=0.1):
        """
        Train the neural network on input data
        - data:   Nxd numppy array with N sample points and d features
        - labels: 1D, length-N numpy array with labels for the N sample points
        """
        # Ensure labels are integers and that data and labels are the same length
        labels = labels.astype(int)
        if len(data) != len(labels):
            print('ERROR: Data and labels must be the same length.')
        
        # Add fictitious unit for bias terms
        fictu = np.array([np.ones(len(data))]).T
        data = np.concatenate((data,fictu),axis=1)
    
        # Initialize Weights
        nfeatures = len(data[0])
        for layer_n in range(self.nlayers):
            WM_nrows,WM_ncols = self.weight_matrix_shape(layer_n,nfeatures)
            # Variance of weight matrix determined by fan-in (eta), the number of units in the previous layer 
            # (or the number of data features when initializing the first weight matrix)
            eta = WM_ncols
            weight_matrix = self.initialize_weights((WM_nrows,WM_ncols),mu=0,var=(1/eta))
            self.weight_matrices.append(weight_matrix)
                
        # Begin loop
        epochcounter = 0
        while epochcounter < 20:
            # Stochastic gradient descent: Loop over points randomly, one at a time
            # (Execute gradient class overhead before beginning)
            self.gradients.prepare(data,labels,self.unitsperlayer[-1])

            for datapoint_i in range(len(data)): 
                X_i = np.array([data[datapoint_i]])
                layeroutput_i = self.forward(X_i)
                
                gradients = self.backward(layeroutput_i,[datapoint_i,datapoint_i+1])
                
                for n in range(self.nlayers):
                    self.weight_matrices[n]=self.weight_matrices[n]-epsilon*gradients[n]  

            
            epochcounter+=1
            DL = np.concatenate((data,labels),axis=1)
            np.random.shuffle(DL)
            data = DL[:,:-1]
            labels = np.array([DL[:,-1]]).T
            epsilon *=0.75
        
        
    def predict(self,testdata):
        """
        Predict classfications for unlabeled data points using the previously 
        trained neural network.
        - testdata: Nxd numpy array with N sample points and d features
                    *Note, dimension d must match that used for the data array in NeuralNet.train*
        Returns a 1D, length-N numpy array of predictions (one prediction per point)
        """
        
        # Add fictitious unit to input to match dimensions
        fictu = np.array([np.ones(len(testdata))]).T
        testdata = np.concatenate((testdata,fictu),axis=1)
        
        npoints = len(testdata)
        predictions = np.empty(npoints)
        layeroutputs = self.forward(testdata)
        predictions = self.classify_outputs(layeroutputs[-1])
        
    
        return predictions.astype(int)
    