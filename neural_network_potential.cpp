#include "neural_network_potential.h"

MatrixXd Layer::tanh(const MatrixXd& input){
  return input.array().tanh();
}

MatrixXd Layer::sigmoid(const MatrixXd& input){
  return 1.0 / (1.0 + (-input).array().exp());
}

MatrixXd Layer::identity(const MatrixXd& input){
  return input;
}

Layer::Layer(const int& in, const int& out)
  : in_size(in), out_size(out),
    weight(MatrixXd::Random(in, out)), bias(RowVectorXd(out)){
  set_activation("tanh");
}

Layer::Layer(const int& in, const int& out, vector<double>& w, vector<double>& b, const string& act)
  : in_size(in), out_size(out),
    weight(Map<MatrixXd>(&w[0], in, out)),
    bias(Map<RowVectorXd>(&b[0], out)){
  set_activation(act);
}

void Layer::set_activation(const string& act){
  if (act == "tanh"){
    activation = &Layer::tanh;
  } else if (act == "sigmoid"){
    activation = &Layer::sigmoid;
  } else if (act == "identity"){
    activation = &Layer::identity;
  }
}

MatrixXd Layer::feedforward(MatrixXd& input){
  input *= weight;
  input.rowwise() += bias;
  return (this->*activation)(input);
}


void NNP::parse_xml(const string){}

MatrixXd NNP::energy(const MatrixXd& G){}

MatrixXd NNP::forces(const MatrixXd& dG){}
