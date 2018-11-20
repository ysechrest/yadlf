#include "nnlayer.h"
#include <iostream>
#include <random>
#include <math.h>

float relu(float z) {
	if (z > 0) {
		return z;
	}
	else {
		return 0.0;
	}
}

float drelu(float z) {
	if (z > 0) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

float sigmoid(float z) {
	return float(1.0 / (1.0 + exp(-z)));
}

float dsigmoid(float z) {
	return float(exp(-z) / pow(1.0 + exp(-z), 2.0));
}
float linear_activation(float z) {
	return z;
}

float dlinear_activation(float z) {
	return 1.0;
}

nnlayer::nnlayer(void) {

}

void nnlayer::initialize(int batchSize, int inSize, int outSize, float momentum, float rmsprop, float regularization, activation act) {
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<float> randn{0,1};
	
	batch_size = batchSize;
	in_size = inSize;
	out_size = outSize;
	betaM = momentum;
	betaR = rmsprop;
	lambdaR = regularization;
	atype = act;
	z = new float[batch_size*out_size];
	a = new float[batch_size*out_size];
	for (int r = 0; r < out_size; r++) {
		for (int c = 0; c < batch_size; c++) {
			z[r*batch_size + c] = 0.0;
			a[r*batch_size + c] = 0.0;			
		}
	}

	/*
	dAprev = new float[batch_size*in_size];
	for (int r = 0; r < in_size; r++) {
		for (int c = 0; c < batch_size; c++) {
			dAprev[r*batch_size + c] = 0.0;
		}
	}
	*/

	float wnorm = 0.0;
	if (atype == activation::RELU) {
		wnorm = sqrt(2.0 / float(in_size));
		afunc = relu;
		dafunc = drelu;
	}
	else if (atype == activation::SIGMOID) {
		wnorm = sqrt(1.0 / float(in_size));
		afunc = sigmoid;
		dafunc = dsigmoid;
	}
	else {
		afunc = linear_activation;
		dafunc = dlinear_activation;
		wnorm = 0.01;
	}

	dW = new float[out_size*in_size];
	sW = new float[out_size*in_size];
	W = new float[out_size*in_size];
	for (int r = 0; r < out_size; r++) {
		for (int c = 0; c < in_size; c++) {
			W[r*in_size + c] = randn(gen)*wnorm;
			dW[r*in_size + c] = 0.0;
			sW[r*in_size + c] = 0.0;
		}
	}

	b = new float[out_size];
	db = new float[out_size];
	sb = new float[out_size];
	for (int i = 0; i < out_size; i++) {
		b[i] = 0.0;
		db[i] = 0.0;
		sb[i] = 0.0;
	}
}

float *nnlayer::forward_prop(float *X) {
	for (int m = 0; m < batch_size; m++) {
		for (int i = 0; i < out_size; i++) {
			z[i*batch_size + m] = 0.0;
			for (int j = 0; j < in_size; j++) {
				z[i*batch_size + m] += W[i*in_size + j] * X[j*batch_size + m] + b[i];
			}
			a[i*batch_size + m] = afunc(z[i*batch_size + m]);
		}
	}
	return a;
}

float *nnlayer::backward_prop(float *dA, float *X, float *dAprev) {
	//Calculate dJ/dz
	float *dz = new float[out_size*batch_size];
	for (int r = 0; r < out_size; r++) {
		for (int c = 0; c < batch_size; c++) {
			dz[r*batch_size + c] = dA[r*batch_size + c] * dafunc(z[r*batch_size + c]);
		}
	}

	//Calculate dJ/dW
	float dW_temp;
	for (int i = 0; i < out_size; i++) {
		for (int k = 0; k < in_size; k++) {
			dW_temp = 0.0;
			for (int j = 0; j < batch_size; j++) {
				dW_temp += (1.0/float(batch_size))*dz[i*batch_size + j] * X[k*batch_size + j];
			}
			dW_temp += (lambdaR / float(batch_size))*W[i*in_size + k];
			dW[i*in_size + k] = betaM * dW[i*in_size + k] + (1 - betaM)*dW_temp;
			sW[i*in_size + k] = betaR * sW[i*in_size + k] + (1 - betaR)*pow(dW_temp, 2.0);
		}
	}

	//Calculate dJ/db
	float db_temp;
	for (int i = 0; i < out_size; i++) {
		db_temp = 0.0;
		for (int j = 0; j < batch_size; j++) {
			db_temp += dz[i*batch_size + j];
		}
		db_temp = db_temp / float(batch_size);
		db[i] = betaM * db[i] + (1 - betaM)*db_temp;
		sb[i] = betaR * sb[i] + (1 - betaR)*pow(db_temp, 2.0);
	}

	//Calculate dJ/dAprev
	for (int k = 0; k < in_size; k++) {
		for (int j = 0; j < batch_size; j++) {
			dAprev[k*batch_size + j] = 0.0;
			for (int i = 0; i < out_size; i++) {
				dAprev[k*batch_size + j] += W[i*in_size + k] * dz[i*batch_size + j];
			}
		}
	}
	delete[] dz;
	return dAprev;
}

void nnlayer::update_params(int iteration, float learning_rate) {
	float num;
	float den;
	//update W with ADAM
	for (int r = 0; r < out_size; r++) {
		for (int c = 0; c < in_size; c++) {
			num = dW[r*in_size + c] / (1.0 - pow(betaM, iteration));
			den = 1.0e-8 + sqrt(sW[r*in_size + c] / (1.0 - pow(betaR, iteration)));
			W[r*in_size+c] -= learning_rate * (num/den);
		}
	}

	//update b with ADAM
	for (int i = 0; i < out_size; i++) {
		num = db[i] / (1.0 - pow(betaM, iteration));
		den = 1.0e-8 + sqrt(sb[i] / (1.0 - pow(betaR, iteration)));
		b[i] -= learning_rate * (num / den);
	}
	return;
}

std::vector<float> nnlayer::get_hyperparams() {
	std::vector<float> hps(6);
	hps[0] = float(batch_size);
	hps[1] = float(in_size);
	hps[2] = float(out_size);
	hps[3] = betaM;
	hps[4] = betaR;
	hps[5] = lambdaR;
	return hps;
}

activation nnlayer::get_activation_type() {
	return atype;
}

void nnlayer::set_weights(float *argW) {
	for (int r = 0; r < out_size; r++) {
		for (int c = 0; c < in_size; c++) {
			W[r*in_size + c] = argW[r*in_size + c];
		}
	}
	return;
}

float *nnlayer::get_db() {
	return db;
}

float *nnlayer::get_dW() {
	return dW;
}

float *nnlayer::get_activate() {
	return a;
}

nnlayer::~nnlayer() {
	delete[] W;
	delete[] b;
	delete[] z;
	delete[] a;
	delete[] dW;
	delete[] sW;
	delete[] db;
	delete[] sb;
	//delete[] dAprev;
}