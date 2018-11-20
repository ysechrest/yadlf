#include "network.h"
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>

network::network(std::vector<int> layer_dims, std::vector<activation> layer_types, int batchSize,
				 float momentum, float RMSprop, float regularization) {
	batch_size = batchSize;
	lambdaR = regularization;

	for (int i = 1; i < layer_dims.size(); i++) {
		int in_size = layer_dims[i - 1];
		int out_size = layer_dims[i];
		nnlayer *layeri = new nnlayer();
		layeri->initialize(batch_size, in_size, out_size, momentum, RMSprop, regularization, layer_types[i]);
		layers.push_back(layeri);
	}

}

void network::forward_step(float *X) {
	float *Aprev = X;
	for (int i = 0; i < layers.size(); i++) {
		Aprev = layers[i]->forward_prop(Aprev);
	}
}

void network::backward_step(float *Y, float *X) {
	std::vector<nnlayer>::reverse_iterator rit = layers.rbegin();
	float *alast = rit->get_activate();
	std::vector<float> hplast = rit->get_hyperparams();
	
	//calc dJ/dA for cost function
	int out_size_last = hplast[2];
	float *dA = new float[batch_size*out_size_last];
	for (int m = 0; m < batch_size; m++) {
		for (int i = 0; i < out_size_last; i++) {
			dA[i*batch_size + m] = Y[i*batch_size + m] / alast[i*batch_size + m];
			dA[i*batch_size + m] += (1.0 - Y[i*batch_size + m]) / (1.0 - alast[i*batch_size + m]);
		}
	}
	std::vector<float>().swap(hplast);

	//propagate backwards through network
	float *dAprev;
	float *Aprev;
	std::vector<float> hps;
	int in_size;
	int out_size;
	for (; rit != layers.rend(); ++rit) {
		hps = rit->get_hyperparams();
		in_size = hps[1];
		out_size = hps[2];

		dAprev = new float[batch_size*in_size];
		if (rit + 1 == layers.rend()) {
			Aprev = X;
		}
		else {
			Aprev = (rit + 1)->get_activate();
		}
		dAprev = rit->backward_prop(dA, Aprev, dAprev);

		delete dA;
		dA = dAprev;
		std::vector<float>().swap(hps);
	}
	delete dAprev;

}

void network::update_network(int iteration, float learning_rate) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i]->update_params(iteration, learning_rate);
	}
}

void network::train_minibatch(int mtest, float *X, float *Y, int epochs, float learning_rate) {
	
	int nbatches = int(floor(mtest / batch_size));
	std::vector<int> idx(mtest);
	for (int i = 0; i < idx.size(); i++) {
		idx[i] = i;
	}

	std::vector<float> hps = layers[0]->get_hyperparams();
	int in_size = hps[1];
	std::vector<float>().swap(hps);
	hps = layers.end->get_hyperparams();
	int out_size = hps[2];

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	float *Xbatch = new float[batch_size*in_size];
	float *Ybatch = new float[batch_size*out_size];
	for (int i = 0; i < epochs; i++) {
		shuffle(idx.begin(),idx.end(),generator);
		for (int j = 0; j < nbatches; j++) {
			this->pick_minibatch(j, in_size, out_size, idx, X, Y, Xbatch, Ybatch);
			this->forward_step(Xbatch);
			this->backward_step(Ybatch, Xbatch);
			this->update_network(i*nbatches + j + 1, learning_rate = learning_rate);
		}
	}
	delete Xbatch;
	delete Ybatch;
}

void network::pick_minibatch(int j, int in_size, int out_size, std::vector<int> idx,
							 float *X, float *Y, float *Xbatch, float *Ybatch) {

	int istart = j * batch_size;
	int iend = (j + 1)*batch_size;
	int m_ctr,m;
	for (int f = 0; f < in_size; f++) {
		m_ctr = 0;
		for (int i = istart; i < iend; i++) {
			m = idx[i];
			Xbatch[f*batch_size + m_ctr] = X[f*batch_size + m];
				m_ctr++;
		}
	}

	for (int f = 0; f < out_size; f++) {
		m_ctr = 0;
		for (int i = istart; i < iend; i++) {
			m = idx[i];
			Ybatch[f*batch_size + m_ctr] = Y[f*batch_size + m];
			m_ctr++;
		}
	}
}

float *network::predict(float *X) {
	this->forward_step(X);
	float *Ypred = layers.back.get_activate();
	return Ypred;
}

network::~network() {
	for (int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}