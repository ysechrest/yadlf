# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:02:37 2018

@author: yanceys
"""
import numpy as np

class inputlayer:
    
    def __init__(self,layer_size):
        self.layer_size = layer_size
        self.W = None
        self.b = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None
        
    def forward_prop(self,X):
        assert(X.shape[0] == self.layer_size)
        self.a = X
        return self.a
        
    def backward_prop(self,dA,prelayer):
        return None
        
    def update_params(self,learning_rate=0.001):
        pass

class relulayer:
    
    def __init__(self,in_size,layer_size):
        self.layer_size = layer_size
        self.W = np.random.randn(layer_size,in_size)*np.sqrt(2.0/in_size)
        self.b = np.zeros((layer_size,1))
        self.z = None
        self.a = None
        self.dW = None
        self.db = None
        
    def forward_prop(self,prelayer):
        self.z = np.dot(self.W,prelayer.a) + self.b
        self.a = (self.z>0).astype('int')*self.z
        return self.a
        
    def backward_prop(self,dA,prelayer):
        m = dA.shape[1]
        dz = dA*(self.z>0).astype('int')
        self.dW = (1.0/m)*np.dot(dz,prelayer.a.T)
        self.db = (1.0/m)*np.sum(dz,axis=1,keepdims=True)
        dA_pre = np.dot(self.W.T,dz)
        return dA_pre
        
    def update_params(self,learning_rate=0.001):
        self.W = self.W - learning_rate*self.dW
        self.b = self.b - learning_rate*self.db

class sigmoidlayer:
    
    def __init__(self,in_size,layer_size):     
        self.layer_size = layer_size
        self.W = np.random.randn(layer_size,in_size)*np.sqrt(1.0/in_size)
        self.b = np.zeros((layer_size,1))
        self.z = None
        self.a = None
        self.dW = None
        self.db = None
        
    def forward_prop(self,prelayer):
        self.z = np.dot(self.W,prelayer.a) + self.b
        self.a = 1.0/(1.0 + np.exp(-self.z))
        return self.a
        
    def backward_prop(self,dA,prelayer):
        m = dA.shape[1]
        dz = dA*(np.exp(-self.z)/(1.0+np.exp(-self.z))**2.0)
        self.dW = (1.0/m)*np.dot(dz,prelayer.a.T)
        self.db = (1.0/m)*np.sum(dz,axis=1,keepdims=True)
        dA_pre = np.dot(self.W.T,dz)
        return dA_pre
        
    def update_params(self,learning_rate=0.001):
        self.W = self.W - learning_rate*self.dW
        self.b = self.b - learning_rate*self.db

class dnn:
    
    def __init__(self,layer_dims,layer_types):
        self.layers=[]
        for ii in np.arange(len(layer_dims)):
            if layer_types[ii] == 'input':
                layer_ii = inputlayer(layer_dims[ii])
                self.layers.append(layer_ii)
            elif layer_types[ii] == 'relu':
                layer_ii = relulayer(self.layers[ii-1].layer_size,layer_dims[ii])
                self.layers.append(layer_ii)
            elif layer_types[ii] == 'sigmoid':
                layer_ii = sigmoidlayer(self.layers[ii-1].layer_size,layer_dims[ii])
                self.layers.append(layer_ii)
    
    def forward_step(self,X):
        self.layers[0].forward_prop(X)
        for ii in np.arange(1,len(self.layers)):
            self.layers[ii].forward_prop(self.layers[ii-1])

    def compute_cost(self,Y):
        m = Y.shape[1]
        cost = (-1.0/m)*np.sum(Y*np.log(self.layers[-1].a) + 
                               (1-Y)*np.log(1-self.layers[-1].a))
        cost = np.squeeze(cost)
        return cost        
        
    def backward_step(self,Y):
        dA = -Y/self.layers[-1].a + (1-Y)/(1-self.layers[-1].a)
        for ii in np.arange(1,len(self.layers))[::-1]:
            dA = self.layers[ii].backward_prop(dA,self.layers[ii-1])
    
    def update_network(self,learning_rate=0.001):
        for ii in np.arange(1,len(self.layers)):
            self.layers[ii].update_params(learning_rate)
            
    def train(self,X,Y,iterations=1000,learning_rate=0.001):
        
        iters_cost = []
        for ii in np.arange(iterations):
            self.forward_step(X)
            iters_cost.append(self.compute_cost(Y))
            self.backward_step(Y)
            self.update_network(learning_rate=learning_rate)
        
        return iters_cost
    
    def predict(self,X):
        self.forward_step(X)
        Y = self.layers[-1].a
        return Y