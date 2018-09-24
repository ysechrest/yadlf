# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:45:10 2018

@author: yanceys
"""

from yadlf import neuralnet as nn
import numpy as np
import matplotlib.pyplot as pplot

def get_sin_testdata(ntest):
    np.random.seed(314159)
    x = np.random.rand(2,ntest)
    x_boundary = 0.33*np.sin(2*np.pi*x[0,:])+0.5
    y = (x[1,:] > x_boundary).astype('int')
    y = y.reshape((1,ntest))    
    return x,y
    
def sin_test(ntest=1000):
    
    x,y = get_sin_testdata(ntest)
    pplot.figure('test')    
    pplot.scatter(x[0,:],x[1,:],c=128*y.flatten()+64)
    pplot.xlim((0,1))
    pplot.ylim((0,1))
    pplot.xlabel('X1')
    pplot.ylabel('X2')
    
    layer_dims = [2,8,8,4,4,1]
    layer_types = ['input','relu','relu','relu','relu','sigmoid']
    
    predictor = nn.dnn(layer_dims,layer_types,beta1=0.9,beta2=0.999)
    
    #predictor.forward_step(x)
    #dA = -y/predictor.layers[-1].a + (1-y)/(1-predictor.layers[-1].a)
    #predictor.backward_step(y)

    #print predictor.layers[-1].a
    #print dA.shape
    #print dA    
    
    #print(predictor.layers[1].dW)
    #print(predictor.layers[2].dW)
    
    #assert(len(predictor.layers)==4)
    #assert(predictor.layers[0].layer_size==2)
    #assert(predictor.layers[1].layer_size==4)
    #assert(predictor.layers[2].layer_size==4)
    #assert(predictor.layers[3].layer_size==1)
    
    if True:
        cost = predictor.train_minibatch(x,y,epochs=400,learning_rate=0.003) 
        pplot.figure('cost')
        pplot.plot(cost)  
        Nx = 33
        Ny = 33
        x1 = np.arange(Nx)/np.float(Nx-1)
        x2 = np.arange(Ny)/np.float(Ny-1)
        
        xx2,xx1 = np.meshgrid(x2,x1)
        #pplot.figure()
        #pplot.plot(xx1.flatten(),xx2.flatten(),'.')    
    
        xsamp = np.vstack((xx1.flatten(),xx2.flatten()))
        assert(xsamp.shape == (2,Nx*Ny))    
    
        ypred = predictor.predict(xsamp)
        ypred = ypred.reshape((Nx,Ny))    
    
        pplot.figure()
        extent = [0,1,0,1]
        pplot.imshow(np.transpose(ypred),cmap='Spectral_r',
                     extent=extent,origin='lower')
        pplot.scatter(x[0,:],x[1,:],cmap='RdBu_r',c=64.*y.flatten()+128,edgecolors='face')
        pplot.xlim((0,1))
        pplot.ylim((0,1))
        pplot.xlabel('X1')
        pplot.ylabel('X2')
