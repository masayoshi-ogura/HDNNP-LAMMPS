#include "neural_network_potential_without_Eigen.h"

// overwrite x to f(x) and return f'(x)
double Layer::tanh(double& x) {
  x = std::tanh(x);
  return 1.0 - x * x;
}

double Layer::sigmoid(double& x) {
  x = 1.0 / (1.0 + exp(-x));
  return x * (1.0 - x);
}

double Layer::identity(double& x) { return 1; }

Layer::Layer() {}

Layer::Layer(const int& in, const int& out, double** w, double* b,
             const string& act) {
  in_size = in;
  out_size = out;
  weight = w;
  bias = b;
  set_activation(act);
}

void Layer::set_activation(const string& act) {
  if (act == "tanh") {
    activation = &Layer::tanh;
  } else if (act == "sigmoid") {
    activation = &Layer::sigmoid;
  } else if (act == "identity") {
    activation = &Layer::identity;
  }
}

void Layer::feedforward(double**& input, double***& deriv_input,
                        double**& output, double***& deriv_output,
                        const int& natom, const int& nfeature) {
  int i, j, k, l;
  double deriv;
  output = (double**)malloc(sizeof(double*) * natom);
  double* row = (double*)malloc(sizeof(double) * natom * out_size);
  deriv_output = (double***)malloc(sizeof(double**) * natom);
  double** deriv_matrix = (double**)malloc(sizeof(double*) * natom * nfeature);
  double* deriv_row =
      (double*)malloc(sizeof(double) * natom * nfeature * out_size);
  for (i = 0; i < natom; ++i) {
    output[i] = row + i * out_size;
    deriv_output[i] = deriv_matrix + i * nfeature;

    for (j = 0; j < out_size; ++j) {
      output[i][j] = bias[j];
      for (k = 0; k < in_size; ++k) {
        output[i][j] += input[i][k] * weight[k][j];
      }
      deriv = (this->*activation)(output[i][j]);

      for (l = 0; l < nfeature; ++l) {
        deriv_output[i][l] = deriv_row + l * out_size;
        deriv_output[i][l][j] = 0.0;
        for (k = 0; k < in_size; ++k) {
          deriv_output[i][l][j] += deriv_input[i][l][k] * weight[k][j];
        }
        deriv_output[i][l][j] *= deriv;
      }
    }
  }

  free(input[0]);
  free(input);
  free(deriv_input[0][0]);
  free(deriv_input[0]);
  free(deriv_input);
}

NNP::NNP(const int& depth, const string& e) {
  element = e;
  layers = vector<Layer>(depth);
}

void NNP::feedforward(double**& features, double*& energy, double**& dE_dG,
                      const int& natom, const int& nfeature) {
  int i, j, k;
  double** input = (double**)malloc(sizeof(double*) * natom);
  double* input_row = (double*)malloc(sizeof(double) * natom * nfeature);
  double*** deriv_input = (double***)malloc(sizeof(double**) * natom);
  double** deriv_matrix = (double**)malloc(sizeof(double*) * natom * nfeature);
  double* deriv_row =
      (double*)malloc(sizeof(double) * natom * nfeature * nfeature);
  double** hidden;
  double*** deriv_hidden;
  for (i = 0; i < natom; ++i) {
    input[i] = input_row + i * nfeature;
    deriv_input[i] = deriv_matrix + i * nfeature;
    for (j = 0; j < nfeature; ++j) {
      input[i][j] = features[i][j];
      deriv_input[i][j] = deriv_row + j * nfeature;
      for (k = 0; k < nfeature; ++k) {
        deriv_input[i][j][k] = 1.0;
      }
    }
  }

  for (Layer& layer : layers) {
    layer.feedforward(input, deriv_input, hidden, deriv_hidden, natom,
                      nfeature);
    input = &hidden[0];
    deriv_input = &deriv_hidden[0];
  }

  energy = (double*)malloc(sizeof(double) * natom);
  dE_dG = (double**)malloc(sizeof(double*) * natom);
  double* dE_dG_row = (double*)malloc(sizeof(double) * natom * nfeature);
  for (i = 0; i < natom; ++i) {
    energy[i] = hidden[i][0];
    dE_dG[i] = dE_dG_row + i * nfeature;
    for (j = 0; j < nfeature; ++j) {
      dE_dG[i][j] = deriv_hidden[i][j][0];
    }
  }
  // energy = hidden(double**)をdouble*に
  // dE_dG = deriv_hidden(double***)をdouble**に
}

vector<NNP> parse_xml(const string& xml_file) {
  int i, j;
  ptree root_pt, nnp_pt, layer_pt;
  int nelement;
  string element;
  int depth;
  int in_size;
  int out_size;
  double** weight;
  double* weight_row;
  double* bias;
  // stringstream ss;
  istream_iterator<double> isi;
  string activation;

  read_xml(xml_file, root_pt);
  nelement = root_pt.get<int>("HDNNP.element.number");
  depth = root_pt.get<int>("HDNNP.NN.depth");
  vector<NNP> masters;

  for (const ptree::value_type& nnp_vt : root_pt.get_child("HDNNP.NN.items")) {
    nnp_pt = nnp_vt.second;
    element = nnp_pt.get<string>("symbol");
    NNP nnp = NNP(depth, element);
    for (const ptree::value_type& layer_vt : nnp_pt.get_child("items")) {
      layer_pt = layer_vt.second;
      in_size = layer_pt.get<int>("in_size");
      out_size = layer_pt.get<int>("out_size");

      // set weight param
      weight = (double**)malloc(sizeof(double*) * in_size);
      weight_row = (double*)malloc(sizeof(double) * in_size * out_size);
      stringstream ss_weight(layer_pt.get<string>("weight"));
      isi = istream_iterator<double>(ss_weight);
      for (i = 0; i < in_size; ++i) {
        weight[i] = weight_row + i * out_size;
        for (j = 0; j < out_size; ++j) {
          weight[i][j] = *isi;
          ++isi;
        }
      }

      // set bias param
      bias = (double*)malloc(sizeof(double) * out_size);
      stringstream ss_bias(layer_pt.get<string>("weight"));
      isi = istream_iterator<double>(ss_bias);
      for (i = 0; i < out_size; ++i) {
        bias[i] = *isi;
        ++isi;
      }

      activation = layer_pt.get<string>("activation");
      Layer layer = Layer(in_size, out_size, weight, bias, activation);
      nnp.layers.push_back(layer);
    }
    masters.push_back(nnp);
  }

  return masters;
}

vector<string> split(const string& input, char delimiter) {
  stringstream ss(input);
  string item;
  vector<string> result;
  while (getline(ss, item, delimiter)) {
    item.erase(remove(item.begin(), item.end(), '\n'), item.end());
    item.erase(remove(item.begin(), item.end(), '\r'), item.end());
    result.push_back(item);
  }
  return result;
}

vector<NNP> parse_txt(const string& txt_file) {
  int i, j, k, l, nelement, depth, in_size, out_size;
  double** weight;
  double *weight_row, *bias;
  string row, element, activation;
  ifstream ifs(txt_file);
  vector<string> elements, strs;
  vector<NNP> masters;

  if (ifs.fail()) {
    cerr << "File do not exist." << endl;
    exit(0);
  }

  // #1: comment
  getline(ifs, row);
  // #2: the number of elements
  getline(ifs, row);
  nelement = stoi(row);
  // #3: the symbol of elements
  getline(ifs, row);
  elements = split(row, ' ');
  // #4: the depth of NNP
  getline(ifs, row);
  depth = stoi(row);

  for (i = 0; i < nelement; ++i) {
    NNP nnp = NNP(depth, elements[i]);
    masters.push_back(nnp);
  }

  while (getline(ifs, row)) {  // skip empty line
    // parse each block

    // #1: layer structure and activation function
    getline(ifs, row);
    strs = split(row, ' ');
    element = strs[0];
    j = stoi(strs[1]) - 1;
    in_size = stoi(strs[2]);
    out_size = stoi(strs[3]);
    activation = strs[4];

    // #2: comment
    getline(ifs, row);

    // #3~n+2: weight
    weight = (double**)malloc(sizeof(double*) * in_size);
    weight_row = (double*)malloc(sizeof(double) * in_size * out_size);
    for (k = 0; k < in_size; ++k) {
      weight[k] = weight_row + k * out_size;
      getline(ifs, row);
      strs = split(row, ' ');
      for (l = 0; l < out_size; ++l) {
        weight[k][l] = stod(strs[l]);
      }
    }

    // #n+3: comment
    getline(ifs, row);

    // #n+4: bias
    bias = (double*)malloc(sizeof(double) * out_size);
    getline(ifs, row);
    strs = split(row, ' ');
    for (l = 0; l < out_size; ++l) {
      bias[l] = stod(strs[l]);
    }

    Layer layer = Layer(in_size, out_size, weight, bias, activation);
    for (i = 0; i < nelement; ++i) {
      if (masters[i].element == element) {
        masters[i].layers[j] = layer;
      }
    }
  }

  return masters;
}
