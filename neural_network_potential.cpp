#include "neural_network_potential.h"


Layer::Layer(int in, int out) {
    weight = MatrixXd::Random(out, in);
    bias = VectorXd::Random(out);
    set_activation("tanh");
}

Layer::Layer(int in, int out, double *w, double *b, char *act) {
    weight = Map<MatrixXd>(&w[0], out, in);
    bias = Map<VectorXd>(&b[0], out);
    set_activation(act);
}

void Layer::tanh(VectorXd &input) {
    input = input.array().tanh();
}

void Layer::deriv_tanh(VectorXd &input, VectorXd &deriv) {
    input = input.array().tanh();
    deriv = 1.0 - input.array().pow(2);
}

void Layer::sigmoid(VectorXd &input) {
    input = 1.0 / (1.0 + (-input).array().exp());
}

void Layer::deriv_sigmoid(VectorXd &input, VectorXd &deriv) {
    input = 1.0 / (1.0 + (-input).array().exp());
    deriv = input.array() * (1.0 - input.array());
}

void Layer::identity(VectorXd &input) { ; }

void Layer::deriv_identity(VectorXd &input, VectorXd &deriv) {
    deriv = input.setOnes();
}

void Layer::set_activation(char *act) {
    if (strcmp(act, "tanh") == 0) {
        activation = &Layer::tanh;
        activation2 = &Layer::deriv_tanh;
    } else if (strcmp(act, "sigmoid") == 0) {
        activation = &Layer::sigmoid;
        activation2 = &Layer::deriv_sigmoid;
    } else if (strcmp(act, "identity") == 0) {
        activation = &Layer::identity;
        activation2 = &Layer::deriv_identity;
    }
}

void Layer::feedforward(VectorXd &input) {
    input = (weight * input).colwise() + bias;
    (this->*activation)(input);
}

void Layer::feedforward2(VectorXd &input, VectorXd &deriv) {
    input = (weight * input).colwise() + bias;
    (this->*activation2)(input, deriv);
}


NNP::NNP() {
    depth = 3;
    layers = new Layer *[3];
    layers[0] = new Layer(2, 3);
    layers[1] = new Layer(3, 3);
    layers[2] = new Layer(3, 1);
}

NNP::NNP(int n) {
    depth = n;
    layers = new Layer *[depth];
}

void NNP::energy(VectorXd input, double &E) {
    int i;
    for (i = 0; i < depth; i++) layers[i]->feedforward(input);
    E += input(0);
}

void NNP::deriv(VectorXd input, VectorXd &dE_dG) {
    int i;
    VectorXd deriv[depth];

    for (i = 0; i < depth; i++) layers[i]->feedforward2(input, deriv[i]);
    dE_dG = VectorXd::Ones(1);
    for (i = depth - 1; i >= 0; i--) {
        dE_dG = dE_dG.array() * deriv[i].array();
        dE_dG = dE_dG.transpose() * layers[i]->weight;
    }
}
