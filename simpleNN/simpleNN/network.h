#pragma once
#include "nnlayer.h"
#include <vector>
#include <string>


class network {
public:
	network(std::vector<int> layer_dims, std::vector<activation> layer_types, int batch_Size,
		    float momentum, float RMSprop, float regularization);
	virtual ~network();
	void forward_step(float *X);
	void backward_step(float *Y, float *X);
	void update_network(int iteration, float learning_rate);
	void train_minibatch(int mtest, float *X, float *Y, int epochs, float learning_rate);
	float *predict(float *X);

private:
	int batch_size;
	float lambdaR;
	std::vector <nnlayer *> layers;
	void pick_minibatch(int j, int in_size, int out_size, std::vector<int> idx, float *X, float *Y, float *Xbatch, float *Ybatch);
};