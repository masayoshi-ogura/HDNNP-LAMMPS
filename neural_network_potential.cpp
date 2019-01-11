#include "neural_network_potential.h"

Layer::Layer(int in, int out, vector<double> &w, vector<double> &b, string act) {
  weight = Map<MatrixXd>(&w[0], out, in);
  bias = Map<VectorXd>(&b[0], out);
  set_activation(act);
}

Layer::~Layer() {}

void Layer::tanh(VectorXd &input, VectorXd &deriv) {
  // return = tanh(x)
  // deriv  = 1 - tanh(x)^2 = 1 - return^2
  input = input.array().tanh();
  deriv = 1.0 - input.array().square();
}

void Layer::elu(VectorXd &input, VectorXd &deriv) {
  // return = exp(x) - 1, when x < 0
  //          x         , when x > 0
  // deriv  = exp(x)    , when x < 0
  //          1         , when x > 0
  // DON'T CHANGE ORDER OF FOLLOWING CALCULATIONS !!!
  deriv = (input.array() < 0).select(input.array().exp(), VectorXd::Ones(input.size()));
  input = (input.array() < 0).select(input.array().exp() - 1.0, input);
}

void Layer::sigmoid(VectorXd &input, VectorXd &deriv) {
  // return = sigmoid(x)
  // deriv  = sigmoid(x) * (1-sigmoid(x)) = return * (1-return)
  input = 1.0 / (1.0 + (-input).array().exp());
  deriv = input.array() * (1.0 - input.array());
}

void Layer::identity(VectorXd &input, VectorXd &deriv) {
  // return = x
  // deriv  = 1
  deriv = VectorXd::Ones(input.size());
}

void Layer::set_activation(string act) {
  if (act == "tanh") {
    activation = &Layer::tanh;
  } else if (act == "elu") {
    activation = &Layer::elu;
  } else if (act == "sigmoid") {
    activation = &Layer::sigmoid;
  } else if (act == "identity") {
    activation = &Layer::identity;
  } else {
    cout << "ERROR!! not implemented ACTIVATION FUNCTION!!" << endl;
  }
}

void Layer::feedforward(VectorXd &input, VectorXd &deriv) {
  input = (weight * input).colwise() + bias;
  (this->*activation)(input, deriv);
}

NNP::NNP(int n) {
  depth = n;
}

NNP::~NNP() {}

void NNP::feedforward(VectorXd input, VectorXd &dE_dG, int eflag,
                      double &evdwl) {
  int i;
  VectorXd deriv[depth];

  for (i = 0; i < depth; i++) layers[i].feedforward(input, deriv[i]);
  dE_dG = VectorXd::Ones(1);
  for (i = depth - 1; i >= 0; i--) {
    dE_dG = dE_dG.array() * deriv[i].array();
    dE_dG = dE_dG.transpose() * layers[i].weight;
  }

  if (eflag) evdwl = input(0);
}
