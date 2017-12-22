# # Recurrent Neural Network
# ## Classifier: Traffic as Low, Medium, or High

# Import necessary modules.
import numpy as np
from matplotlib import pyplot as plt


# Load data from text file.
data = np.genfromtxt('/Users/mitch/Dropbox/FinalProject/traffic2.txt')


# Define functions and derivatives of those functions to be used in the RNN.
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(x):
    return np.maximum(x,np.zeros_like(x))

def Leaky_ReLU_deriv(x):
    dRLU = np.empty(len(x))
    for i in range(len(x)):
        if x[i] > 0: 
            dRLU[i] = 1
        else:
            dRLU[i] = 0.1
    return dRLU   

# Rename function for convenience
LRD = Leaky_ReLU_deriv


# Define a recurrent neural network class, which repeats a single layer.
class OneLayerRNN:
    
    def __init__(self,hidden_units=10,output_units=3):
        self.V = None
        self.W = None
        self.h = None
        self.z = None
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.windowlen = None
        self.futuretime = None
    
    def initialize(self,X):
        # Weight matrices start as normal distributions
        self.V = np.random.normal(scale=1.0/np.sqrt(self.hidden_units),size=((self.hidden_units),1))
        self.W = np.random.normal(scale=1.0/np.sqrt(self.hidden_units),size=(self.output_units,self.hidden_units))
        self.h = np.zeros((len(X),self.hidden_units))
        self.z = np.zeros((len(X),self.output_units))

        # Matrices for gradient update
        self.gV = np.zeros_like(self.V)
        self.gW = np.zeros_like(self.W)

        
    def forward(self,X):
        for i in range(len(X)):
            x = X[i]                                                        # traffic at time i
            
            self.h[i] = relu(np.dot(self.V,x)).flatten()        # calculate new layer output
            self.z[i] = sigmoid(np.dot(self.W,self.h[i]))
            
    def backward(self,X,y):
        """Backpropagation through time"""
        for i in range(len(X)):
            x = X[i]
            Q = self.z[i] - y
            self.gV += np.array([np.dot(self.W.T,Q)*LRD(self.h[i]*x)]).T
            self.gW += np.outer(Q,self.h[i])
            
    def update_VW(self,epsilon):
        """Update"""
        self.V = self.V - epsilon*self.gV
        self.W = self.W - epsilon*self.gW

    def one_hot_encode(self,occ):
        if occ < 0.33: return np.array([1,0,0])
        elif occ >= 0.33 and occ < 0.67: return np.array([0,1,0])
        elif occ >= 0.67: return np.array([0,0,1])
        else: print('Error: the given occupancy value was ',occ) 
            
            
    def train(self,timeframe,epsilon,windowlen=24,futuretime=0):
        """
        Train the neural network given input data as X
        X:   - a series of occupancies
        """
        self.windowlen=windowlen
        self.futuretime = futuretime
        
        windowstart = 0
        firstwindow = timeframe[:windowlen]
        self.initialize(firstwindow)
        counter,total = 0,0
        trainingerrors = []
        while windowstart < len(timeframe)-windowlen-futuretime-1:
            timewindow = timeframe[windowstart:windowstart+windowlen]
            timewindowlabel = self.one_hot_encode(timeframe[windowstart+windowlen+futuretime])
            trueforecast = np.argmax(timewindowlabel)
            
            self.forward(timewindow)
            self.backward(timewindow,timewindowlabel)
            self.update_VW(epsilon)
            
            prediction = np.argmax(np.average(self.z,axis=0))

            if prediction == trueforecast:
                counter += 1
            total += 1

            TE = counter/total
            if windowstart%1000==0:
                #print(windowstart,'\t',TE)
                trainingerrors.append(TE)
            windowstart=windowstart+1
        #print(counter/total)
        return trainingerrors
                
        
    def predict(self,timeframe,predicttime,predictgap=0):
        timewindow = timeframe[predicttime-predictgap-self.windowlen:predicttime-predictgap]
        self.forward(timewindow)
        classification = np.argmax(np.average(self.z,axis=0))
        '''if classification==0:
            print('low')
        elif classification==1:
            print('med')
        elif classification==2:
            print('high')'''
        
        return classification


