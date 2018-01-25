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


NNP::NNP(const int& n, const string& element)
  : nlayer(n), element(element){}

double NNP::energy(MatrixXd m){
  BOOST_FOREACH (Layer& layer, layers){
    m = layer.feedforward(m);
  }
  return m.sum();
}

VectorXd NNP::forces(MatrixXd m){}

template <typename T>
vector<T> split_cast(const string& str){
  istringstream ss(str);
  return vector<T>{istream_iterator<T>(ss), istream_iterator<T>()};
}


vector<NNP> parse_xml(const string& xml_file){
  ptree pt;
  int nelement;
  string element;
  int depth;
  int in_size;
  int out_size;
  vector<double> weight;
  vector<double> bias;
  string activation;

  read_xml(xml_file, pt);
  nelement = pt.get<int>("HDNNP.element.number");
  depth = pt.get<int>("HDNNP.NN.depth");
  vector<NNP> masters;
  
  BOOST_FOREACH (const ptree::value_type& nnp_xml, pt.get_child("HDNNP.NN.items")){
    element = nnp_xml.second.get<string>("symbol");
    NNP nnp = NNP(depth, element);
    BOOST_FOREACH (const ptree::value_type& layer_xml, nnp_xml.second.get_child("items")){
      in_size = layer_xml.second.get<int>("in_size");
      out_size = layer_xml.second.get<int>("out_size");
      weight = split_cast<double>(layer_xml.second.get<string>("weight"));
      bias = split_cast<double>(layer_xml.second.get<string>("bias"));
      activation = layer_xml.second.get<string>("activation");
      Layer layer = Layer(in_size, out_size, weight, bias, activation);
      nnp.layers.push_back(layer);
    }
    masters.push_back(nnp);
  }

  return masters;
}
