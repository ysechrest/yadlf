#include "nnlayer.h"
#include <iostream>
#include <math.h>
#include <assert.h>

void test_initialize() {
	nnlayer layer0 = nnlayer::nnlayer();
	std::vector<float> hps;
	activation atype;
	layer0.initialize(8, 2, 4, 0.9, 0.999, 0.0, activation::SIGMOID);
	hps = layer0.get_hyperparams();
	atype = layer0.get_activation_type();

	int batch_size = int(hps[0]);
	int in_size = int(hps[1]);
	int out_size = int(hps[2]);
	float betaM = float(hps[3]);
	float betaR = float(hps[4]);
	float lambdaR = float(hps[5]);

	std::cout << "Batch Size: " << batch_size << "\n";
	std::cout << "Input Layer Size: " << in_size << "\n";
	std::cout << "Output Layer Size: " << out_size << "\n";
	std::cout << "Momentum Param: " << betaM << "\n";
	std::cout << "RMSprop Param: " << betaR << "\n";
	std::cout << "Regularization Param: " << lambdaR << "\n";
	if (atype == activation::SIGMOID) {
		std::cout << "Activation Type: " << "SIGMOID" << "\n";
	}
	else if (atype == activation::RELU) {
		std::cout << "Activation Type: " << "RELU" << "\n";
	}
	else {
		std::cout << "Activation Type: " << "UNDEF" << "\n";
	}

	assert(batch_size == 8);
	assert(in_size == 2);
	assert(out_size == 4);
	assert(abs(betaM - 0.9) < 1.0e-6 );
	assert(abs(betaR - 0.999) < 1.0e-6);
	assert(abs(lambdaR - 0.0) < 1.0e-6);
	assert(atype == activation::SIGMOID);
}

void test_forward_prop() {
	nnlayer layer0 = nnlayer::nnlayer();
	int in_size = 3;
	int out_size = 3;
	int batch_size = 2;

	layer0.initialize(batch_size, in_size, out_size, 0.9, 0.999, 0.0, activation::RELU);

	float *argW = new float[out_size*in_size];
	float *X = new float[in_size*batch_size];
	float *res;

	argW[0] = 3;
	argW[1] = 2;
	argW[2] = 6;
	argW[3] = 1;
	argW[4] = 6;
	argW[5] = 5;
	argW[6] = 2;
	argW[7] = 6;
	argW[8] = 7;
	std::cout << "W = \n";
	for (int r = 0; r < out_size; r++) {
		std::cout << "   [";
		for (int c = 0; c < in_size; c++) {
			std::cout << argW[r*in_size + c];
			if (c < in_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}

	X[0] = 5;
	X[1] = 1;
	X[2] = 8;
	X[3] = 5;
	X[4] = 4;
	X[5] = 4;
	std::cout << "X = \n";
	for (int r = 0; r < in_size; r++) {
		std::cout << "    [";
		for (int c = 0; c < batch_size; c++) {
			std::cout << X[r*batch_size + c];
			if (c < batch_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}

	layer0.set_weights(argW);
	res = layer0.forward_prop(X);
	std::cout << "A = \n";
	for (int r = 0; r < out_size; r++) {
		std::cout << "    [";
		for (int c = 0; c < batch_size; c++) {
			std::cout << res[r*batch_size + c];
			if (c < batch_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}
}

void test_backward_prop() {
	nnlayer layer0 = nnlayer::nnlayer();
	int in_size = 3;
	int out_size = 4;
	int batch_size = 2;

	layer0.initialize(batch_size, in_size, out_size, 0.0, 0.0, 0.0, activation::LINEAR);

	float *dAprev = new float[in_size*batch_size];//3x2
	float *X = new float[in_size*batch_size];//3x2
	float *dA = new float[out_size*batch_size]; //4x2
	float *dW;
	float *db;
	float *argW = new float[out_size*in_size];

	X[0] = 7;
	X[1] = 8;
	X[2] = 1;
	X[3] = 4;
	X[4] = 9;
	X[5] = 1;
	std::cout << "X = \n";
	for (int r = 0; r < in_size; r++) {
		std::cout << "    [";
		for (int c = 0; c < batch_size; c++) {
			std::cout << X[r*batch_size + c];
			if (c < batch_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}

	dA[0] = 2;
	dA[1] = 1;
	dA[2] = 2;
	dA[3] = 8;
	dA[4] = 3;
	dA[5] = 5;
	dA[6] = 4;
	dA[7] = 3;
	std::cout << "dA = \n";
	for (int r = 0; r < out_size; r++) {
		std::cout << "    [";
		for (int c = 0; c < batch_size; c++) {
			std::cout << dA[r*batch_size + c];
			if (c < batch_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}

	argW[0] = 1;
	argW[1] = 2;
	argW[2] = 3;
	argW[3] = 1;
	argW[4] = 2;
	argW[5] = 3;
	argW[6] = 1;
	argW[7] = 2;
	argW[8] = 3;
	argW[9] = 1;
	argW[10] = 2;
	argW[11] = 3;
	std::cout << "W = \n";
	for (int r = 0; r < out_size; r++) {
		std::cout << "   [";
		for (int c = 0; c < in_size; c++) {
			std::cout << argW[r*in_size + c];
			if (c < in_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}

	layer0.set_weights(argW);
	dAprev = layer0.backward_prop(dA, X, dAprev);
	std::cout << "dAprev = \n";
	for (int r = 0; r < in_size; r++) {
		std::cout << "    [";
		for (int c = 0; c < batch_size; c++) {
			std::cout << dAprev[r*batch_size + c];
			if (c < batch_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}

	dW = layer0.get_dW();
	std::cout << "dW = \n";
	for (int r = 0; r < out_size; r++) {
		std::cout << "    [";
		for (int c = 0; c < in_size; c++) {
			std::cout << dW[r*in_size + c];
			if (c < in_size - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}

	db = layer0.get_db();
	std::cout << "db = ";
	std::cout << "[ ";
	for (int i = 0; i < out_size; i++) {
		std::cout << db[i];
		if (i < out_size - 1) {
			std::cout << ",";
		}
	}
	std::cout << "]\n";

}

int main() {
	test_backward_prop();
}