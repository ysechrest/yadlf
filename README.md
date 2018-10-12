# yadlf
Deep Learning Playground

The purpose of this project is to create a playground for machine-learning fundamentals and simple deep learning projects. Not particularly novel. Not particularly exciting. The programming equivalent of building your own bicycle.

neuranet.py defines classes that may be used to create a fully-connected network.
The snippet below builds a small network with 2 features and 5 layers of size [8,8,4,4,1]. Activations are defined as relu for hidden and sigmoid for output. The network constructor sets the momentum (beta1), RMSprop (beta2), and L2 regularization parameters (lambda_reg).

```
layer_dims = [2,8,8,4,4,1]
layer_types = ['input','relu','relu','relu','relu','sigmoid']
predictor = nn.dnn(layer_dims,layer_types,beta1=0.9,beta2=0.999,lambda_reg=0.03)
```

examples.py currently contains a simple example of fitting a sin-wave decision boundary. Running sin_test() should produce a plot of the cost vs. iteration, and a plot of the test data plotted over the prediction contours.

