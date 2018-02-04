#include "neural_network_potential.h"

// overwrite x to f(x) and return f'(x)
double Layer::tanh(double& x){
  x = tanh(x);
  return 1.0 - tanh(x)**2;
}

double Layer::sigmoid(double& x){
  x = 1.0 / (1.0 + exp(-x));
  return x * (1.0 - x);
}

double Layer::identity(double& x){
  return 1;
}

Layer::Layer(const int& in, const int& out, double** w, double** b, const string& act){
  in_size = in;
  out_size = out;
  weight = w;
  bias = b;
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

void Layer::feedforward(double**& input, double***& deriv_input, double**& output, double***& deriv_output, const int& natom, const int& nfeature){
  int i, j, k, l;
  double deriv;
  output = (double**)malloc(sizeof(double*) * natom);
  double* row = (double*)malloc(sizeof(double) * natom * out_size);
  deriv_output = (double***)malloc(sizeof(double**) * natom);
  double** deriv_matrix = (double**)malloc(sizeof(double*) * natom * nfeature);
  double* deriv_row = (double*)malloc(sizeof(double) * natom * nfeature * out_size);
  for (i=0; i<natom; ++i){
    output[i] = row + i * out_size;
    deriv_output[i] = deriv_matrix + i * nfeature;

    for (j=0; j<out_size; ++j){
      output[i][j] = bias[j];
      for (k=0; k<in_size; ++k){
        output[i][j] += input[i][k] * weight[k][j];
      }
      deriv = (this->*activation)(output[i][j]);

      for (l=0; l<nfeature; ++l){
        deriv_output[i][l] = deriv_row + l * out_size;
        deriv_output[i][l][j] = 0.0;
        for (k=0; k<in_size; ++k){
          deriv_output[i][l][j] += deriv_input[i][l][k] * weight[k][j];
        }
        deriv_output[i][l][j] *= deriv;
      }
    }
  }

  for (i = 0; i < natom; ++i) {
    for (j = 0; j < nfeature; ++j) {
      free(deriv_input[i][j]);
    }
    free(input[i]);
    free(deriv_input[i]);
  }
  free(input);
  free(deriv_input);
}


NNP::NNP(const int& n, const string& e){
  nlayer = n;
  element = e;
}

void feedforward(double** features, double* energy, double** dE_dG){
  int natom = ARRAY_LENGTH(features);
  int nfeature = ARRAY_LENGTH(features[0]);
  double** input = &features[0];
  double*** deriv_input = (double***)malloc(sizeof(double**) * natom);
  double** deriv_matrix = (double**)malloc(sizeof(double*) * natom * nfeature);
  double* deriv_row = (double*)malloc(sizeof(double) * natom * nfeature * nfeature);
  double** hidden;
  double*** deriv_hidden;
  for (int i=0; i<natom; ++i){
    deriv_input[i] = deriv_matrix + i * nfeature;
    for (int j=0; j<nfeature; ++j){
      deriv_input[i][j] = deriv_row + j * nfeature;
      for (int k=0; k<nfeature; ++k){
        deriv_input[i][j][k] = 1.0;
      }
    }
  }

  for (Layer& layer: layers){
    layer.feedforward(input, deriv_input, hidden, deriv_hidden, natom, nfeature);
    input = hidden;
    deriv_input = deriv_hidden;
  }

  // energy = hidden(double**)をdouble*に
  // dE_dG = deriv_hidden(double***)をdouble**に

  free(input);
  free(deriv_input);
  free(deriv_matrix);
  free(deriv_row);
  free(hidden);
  free(deriv_hidden);
}

double NNP::energy(MatrixXd m){
  for (Layer& layer : layers){
    m = layer.feedforward(m);
  }
  return m.sum();
}

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
  
  for (const ptree::value_type& nnp_xml : pt.get_child("HDNNP.NN.ietms")){
    element = nnp_xml.second.get<string>("symbol");
    NNP nnp = NNP(depth, element);
    for (const ptree::value_type& layer_xml : nnp_xml.second.get_child("items")){
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
