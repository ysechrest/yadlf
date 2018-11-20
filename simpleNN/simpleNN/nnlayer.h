#ifndef NNLAYER_H
#define NNLAYER_H
#include <vector>

enum class activation {
	SIGMOID,
	RELU,
	LINEAR
};

class nnlayer {
public:
	nnlayer(void);
	virtual ~nnlayer();
	void initialize(int batchSize, int inSize, int outSize, float momentum, float rmsprop, float regularization, activation act);
	float *forward_prop(float *X);
	float *backward_prop(float *dA, float *X, float *dAprev);
	void update_params(int iteration, float learning_rate);
	std::vector<float> get_hyperparams();
	activation get_activation_type();
	void set_weights(float *argW);
	float *get_db();
	float *get_dW();
	float *get_activate();

private:
	int batch_size;
	int in_size;
	int out_size;
	float betaM;
	float betaR;
	float lambdaR;
	activation atype;
	float *W;
	float *b;
	float *z;
	float *a;
	float *dW;
	float *sW;
	float *db;
	float *sb;
	//float *dAprev;
	float (*afunc)(float);
	float (*dafunc)(float);
};

float relu(float z);
float drelu(float z);
float sigmoid(float z);
float dsigmoid(float z);
float linear_activation(float z);
float dlinear_activation(float z);

#endif