# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:02:37 2018

@author: yanceys

TODO:
- Implement Batch Normalization

"""
import numpy as np
#----------------------------------------------------------------
#Definitions of activation functions and their derivatives below.
#These are used by nnlayer to calculate forward and backward prop
#for a given layer.
#----------------------------------------------------------------
def sigmoid(z,prime=False):
    #Sigmoid activation function used in NN
    if prime:
        #if prime is true, retval is set to derivative of sigmoid
        retval = np.exp(-z)/(1.0+np.exp(-z))**2.0
    else:
        retval = 1.0/(1.0 + np.exp(-z))
    return retval
        
def relu(z,prime=False):
    #Relu activation function used in NN
    if prime:
        #if prime is true, retval is set to derivative of relu
        retval = (z>0).astype('int')
    else:
        retval = (z>0).astype('int')*z
    return retval
    
def tanh(z,prime=False):
    #Tanh activation function used in NN
    if prime:
        #if prime is true, retval is set to derivative of tanh
        retval = (1-np.tanh(z)**2.0)
    else:
        retval = np.tanh(z)
    return retval
    
#----------------------------------------------------------------
#inputlayer class:
#used to create a dummy 0 layer for storing input features 
#into the neural network.
#This class exists so that the forward and backprop calls for
#all layers look the same since these calls take a previous layer
#as an argument.
#----------------------------------------------------------------
class inputlayer:
    
    def __init__(self,layer_size):
        self.activation = 'input'
        self.afunc = None
        self.layer_size = layer_size
        self.W = None
        self.b = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None
    
    def set_activation(self,X):
        assert(X.shape[0] == self.layer_size)
        self.a = X
        
    def forward_prop(self,prelayer):
        return self.a
        
    def backward_prop(self,dA,prelayer):
        return dA
        
    def update_params(self,learning_rate=0.001):
        pass

#----------------------------------------------------------------
#nnlayer class:
#this object contains the weight and bias arrays for a single layer
#and performs the initialization, forward prop, backward prop, and
#parameter update steps for the layer.
#----------------------------------------------------------------
class nnlayer:
    
    def __init__(self,in_size,layer_size,activation='relu',afunc=None,wnorm=None,
                 beta1=None,beta2=None,lambda_reg=None):
        self.activation = activation #Type of activation (e.g. 'relu')
        self.layer_size = layer_size #Number of nodes
        self.lambda_reg = lambda_reg #L2 Regularization factor
        if self.activation == 'sigmoid':
            self.afunc = sigmoid #function hook for activation
            self.W = np.random.randn(layer_size,in_size)*np.sqrt(1.0/in_size) #Weight Array
            self.b = np.zeros((layer_size,1)) #Bias array
        elif self.activation == 'relu':
            self.afunc = relu
            self.W = np.random.randn(layer_size,in_size)*np.sqrt(2.0/in_size)
            self.b = np.zeros((layer_size,1))
        elif self.activation == 'tanh':
            self.afunc = tanh
            self.W = np.random.randn(layer_size,in_size)*np.sqrt(1.0/in_size)
            self.b = np.zeros((layer_size,1))
        elif self.activation == 'user':
            self.afunc = afunc
            self.W = np.random.randn(layer_size,in_size)*wnorm
            self.b = np.zeros((layer_size,1))
        self.z = None #stores linear calculation W*X+B from forward prop for backprop
        self.a = None #stores activiation A(z) from forward prop for backprop
        self.dW = 0 #W update (if beta1 non-zero this is the momentum term)
        self.sW = 0 #b update (if beta1 non-zero this is the momentum term)
        self.db = 0 #RMSprop update term
        self.sb = 0 #RMSprop update term
        self.beta1 = beta1 #momentum exponential averaging factor
        self.beta2 = beta2 #RMSprop exponentail averaging factor

    def set_activation(self,X):
        assert(X.shape[0] == self.layer_size)
        self.a = X        
    
    def forward_prop(self,prelayer):
        self.z = np.dot(self.W,prelayer.a) + self.b
        self.a = self.afunc(self.z)
        return self.a
        
    def backward_prop(self,dA,prelayer):
        m = dA.shape[1]
        dz = dA*self.afunc(self.z,prime=True)
        
        dW_temp = (1.0/m)*np.dot(dz,prelayer.a.T)
        if not(self.lambda_reg is None):
            dW_temp += (self.lambda_reg/m)*self.W
        db_temp = (1.0/m)*np.sum(dz,axis=1,keepdims=True)
        
        if (self.beta1 is None):
            #No Momentum
            self.dW = dW_temp
            self.db = db_temp
        else:
            #Momentum Update
            self.dW = self.dW*self.beta1 + (1-self.beta1)*dW_temp
            self.db = self.db*self.beta1 + (1-self.beta1)*db_temp
        if not(self.beta2 is None):
            #RMSprop Update
            self.sW = self.sW*self.beta2 + (1-self.beta2)*(dW_temp)**2.0
            self.sb = self.sb*self.beta2 + (1-self.beta2)*(db_temp)**2.0
        
        dA_pre = np.dot(self.W.T,dz)
        return dA_pre
        
    def update_params(self,iteration,learning_rate=0.001):
        if (self.beta1 is None) and (self.beta2 is None):
            #Gradient Descent Update
            self.W = self.W - learning_rate*self.dW
            self.b = self.b - learning_rate*self.db
        elif not(self.beta1 is None) and (self.beta2 is None):
            #Gradient Descent w/ Momentum Update
            self.W = self.W - learning_rate*self.dW/(1-self.beta1**iteration)
            self.b = self.b - learning_rate*self.db/(1-self.beta1**iteration)
        elif (self.beta1 is None) and not(self.beta2 is None):
            #RMSprop Update
            self.W = self.W - learning_rate*self.dW/(np.sqrt(self.sW/(1-self.beta2**iteration))+1.0e-8)
            self.b = self.b - learning_rate*self.db/(np.sqrt(self.sb/(1-self.beta2**iteration))+1.0e-8)
        elif not(self.beta1 is None) and not(self.beta2 is None):
            #ADAM Update
            self.W = self.W - learning_rate*(self.dW/(1-self.beta1**iteration))/(np.sqrt(self.sW/(1-self.beta2**iteration))+1.0e-8)
            self.b = self.b - learning_rate*(self.db/(1-self.beta1**iteration))/(np.sqrt(self.sb/(1-self.beta2**iteration))+1.0e-8)
            
#----------------------------------------------------------------
#dnn class:
#This class represents the high-level view of the neural network,
#and is the primary interface to the user.
#It is a container for nnlayers, it propagates the forward,
#backward, and parameter update calls through the entire network.
#It computes the cost function, and performs the training iteration.
#Finally, it does the prediction once trained.
#----------------------------------------------------------------
class dnn:
    
    def __init__(self,layer_dims,layer_types,beta1=None,beta2=None,lambda_reg=None):
        self.layers=[]
        self.lambda_reg = lambda_reg #L2 regularization factor
        for ii in np.arange(len(layer_dims)):
            if layer_types[ii] == 'input':
                #Make the 'dummy' input layer
                layer_ii = inputlayer(layer_dims[ii])
                self.layers.append(layer_ii)
            else:
                #Create a layer (hidden or output)
                layer_ii = nnlayer(layer_dims[ii-1],layer_dims[ii],
                                   activation=layer_types[ii],
                                   beta1=beta1,beta2=beta2,
                                   lambda_reg=lambda_reg)
                self.layers.append(layer_ii)
    
    def forward_step(self,X):
        #Loop over layers to perform forward propagation
        self.layers[0].set_activation(X)
        for ii in np.arange(1,len(self.layers)):
            self.layers[ii].forward_prop(self.layers[ii-1])

    def compute_cost(self,Y):
        #Compute the log-loss cost function
        m = Y.shape[1]
        cost = (-1.0/m)*np.sum(Y*np.log(self.layers[-1].a) + 
                                (1-Y)*np.log(1-self.layers[-1].a))
        cost = np.squeeze(cost)
        
        if not(self.lambda_reg is None):
            for ii in np.arange(1,len(self.layers)):
                cost += (self.lambda_reg/(2.0*m))*np.linalg.norm(self.layers[ii].W,ord='fro')**2.0
        
        return cost        
        
    def backward_step(self,Y):
        #Loop over layers in reverse order to do backward prop
        dA = -Y/self.layers[-1].a + (1-Y)/(1-self.layers[-1].a)
        for ii in np.arange(1,len(self.layers))[::-1]:
            dA = self.layers[ii].backward_prop(dA,self.layers[ii-1])
    
    def update_network(self,iteration,learning_rate=0.001):
        #Loop over layers to perform parameter updates
        for ii in np.arange(1,len(self.layers)):
            self.layers[ii].update_params(iteration,learning_rate=learning_rate)
            
    def train(self,X,Y,iterations=1000,learning_rate=0.001):
        #Perform the iterative training on features X and labels Y
        #Return the trend of the costfunction over iteration
        iters_cost = []
        for ii in np.arange(iterations):
            self.forward_step(X)
            iters_cost.append(self.compute_cost(Y))
            self.backward_step(Y)
            self.update_network(ii,learning_rate=learning_rate)
        
        return iters_cost
        
    def train_minibatch(self,X,Y,epochs=100,learning_rate=0.001,batch_size=128):
        #Perform the iterative training on minibatches
        m = Y.shape[1]
        nbatches = np.int(m/batch_size)        
        
        iters_cost = []
        for ii in np.arange(epochs):
            #Data is reshuffled into new mini-batches every epoch
            idx_shuffle = np.random.rand(m)
            idx_shuffle = np.argsort(idx_shuffle)
            for jj in np.arange(nbatches):
                #Define the mini-batch
                idx_start = jj*batch_size
                idx_end = np.min([m,(jj+1)*batch_size])
                X_batch = X[:,idx_shuffle[idx_start:idx_end]]
                Y_batch = Y[:,idx_shuffle[idx_start:idx_end]]
                
                #Do one iteration of training
                self.forward_step(X_batch)
                iters_cost.append(self.compute_cost(Y_batch))
                self.backward_step(Y_batch)
                self.update_network(ii*jj+jj+1,learning_rate=learning_rate)
        
        return iters_cost    
    
    def predict(self,X):
        #Perform forward prop of X through network to obtain predicted labels
        self.forward_step(X)
        Y = self.layers[-1].a
        return Y