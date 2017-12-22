
import numpy as np

data = np.genfromtxt('/Users/mitch/Dropbox/FinalProject/traffic2.txt')

sampledata = data[:20,12:36]
samplelabels = data[:20,37]
testdata = data[20:30,12:36]
testlabel=data[20:30,37]


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(x):
    return np.maximum(x,np.zeros_like(x))

def Leaky_ReLU_deriv(x):
    dRLU = np.empty(len(x))
    for i in range(len(x)):
        if x[i] >= 0: 
            dRLU[i] = 1
        else:
            dRLU[i] = 0.001
    return dRLU   

LRD = Leaky_ReLU_deriv


class OneLayerRNN:
    
    def __init__(self,hidden_units=10,output_units=1):
        self.V = None
        self.U = None
        self.W = None
        self.h = None
        self.z = None
        self.hidden_units = hidden_units
        self.output_units = output_units
    
    def initialize(self,timeseries):
        X = timeseries[0]
        # Weight matrices start as normal distributions
        self.V = np.random.normal(scale=1.0/np.sqrt(self.hidden_units),size=(self.hidden_units))
        self.U = np.random.normal(scale=1.0/np.sqrt(self.hidden_units),size=(self.hidden_units,self.hidden_units))
        self.W = np.random.normal(scale=1.0/np.sqrt(self.hidden_units),size=(self.output_units,self.hidden_units))
        self.h = np.zeros((len(X),self.hidden_units))
        # A delta matrix for each unfolded layer
        self.delta_V_h = np.zeros((len(X),self.hidden_units))
        self.delta_U_h = np.zeros((len(X),self.hidden_units,self.hidden_units))
        self.delta_h_h = np.zeros((len(X),self.hidden_units))
        # Matrices for gradient update
        self.gV = np.zeros_like(self.V)
        self.gU = np.zeros_like(self.U)
        self.gW = np.zeros_like(self.W)
        
        self.gV = np.array([self.gV]).T


        
    def forward(self,X):
        
        for i in range(len(X)):
            x = X[i]
            if i != 0:
                h_prev = self.h[i-1]
            else:
                h_prev = np.zeros_like(self.h[0])
            self.h[i] = relu(np.dot(self.U,h_prev)+np.dot(self.V,x))
            self.delta_V_h[i] = x*LRD(self.h[i])
            self.delta_h_h[i] = np.dot(self.U.T,LRD(self.h[i]))
            self.delta_U_h[i] = np.outer(LRD(self.h[i]),h_prev)
        self.z = sigmoid(np.dot(self.W,self.h[-1]))
        
        
    def backward(self,X,y):
        """Backpropagation through time"""
        Q = self.z - y
        
        self.gW = np.outer(Q,self.h[-1])
        layer_grad_V_h = self.W.T*Q
        layer_grad_U_h = self.W.T*Q
        #Accumulate gradients from each layer together
        for i in range(len(X)):
            layer_grad_V_h = layer_grad_V_h*np.array([self.delta_h_h[-i-1]]).T
            full_grad_V_h = layer_grad_V_h*np.array([self.delta_V_h[-i-1]]).T
            self.gV += full_grad_V_h
            
            layer_grad_U_h = layer_grad_U_h*self.delta_h_h[-i-1]
            full_grad_U_h = layer_grad_V_h*self.delta_U_h[-i-1]
            self.gU += full_grad_U_h

            
    def update_grads(self,epsilon):
        """Update gradients """
        self.V = (np.array([self.V]).T - epsilon*self.gV).flatten()
        self.U = self.U - epsilon*self.gU
        self.W = self.W - epsilon*self.gW

        
        
    def train(self,timeseries,labels,epsilon):
        """
        Train the neural network given input data as X
        X:   - a series of occupancies
        """
        self.initialize(timeseries)
        for X,y in zip(timeseries,labels):
            self.forward(X)
            self.backward(X,y)
            self.update_grads(epsilon)
        
        
    def predict(self,X):
        self.forward(X)
        if self.z < 0.33:
            classification='low'
        elif self.z >= 0.33 and self.z < 0.66:
            classification='med'
        elif self.z >= 0.66:
            classification='high'
        print(self.z,classification)
        
        return classification

    


RNN = OneLayerRNN()

timeseries=sampledata
labels=samplelabels

RNN.initialize(timeseries)
RNN.train(timeseries,labels,0.001)

counter,total = 0,0
for i in range(len(testdata)):
    prediction = RNN.predict(tesdata[i])
    if testlabel[i] < 0.33:
        testlabelclass = 'low'
    elif testlabel[i] >= 0.33 and testlabel[i] < 0.66:
        testlabelclass = 'med'
    elif testlabel[i] >= 0.66:
        testlabelclass = 'high'
    
    if prediction==testlabelclass:
        counter += 1
    total += 1
print(counter/total)
        
    


# In[ ]:



