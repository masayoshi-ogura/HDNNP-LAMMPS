#include <chrono>
#include <iostream>
#include <numeric>

#include "neural_network_potential_without_Eigen.h"

using namespace std;

int main() {
  int i, j;
  int natom = 10;
  int nfeature = 2;
  double** features = (double**)malloc(sizeof(double*) * natom);
  double* features_row = (double*)malloc(sizeof(double) * natom * nfeature);
  double* energy;  // size(natom) will be stored
  double** dE_dG;  // size(natom, nfeature) will be stored

  // initialize with some values
  // of course, "natom" can be different for each element
  for (i = 0; i < natom; ++i) {
    features[i] = features_row + i * nfeature;
    for (j = 0; j < nfeature; ++j) {
      features[i][j] = (i + 1.0) / (j + 1.0);
    }
  }

  // vector<NNP> masters = parse_xml("hoge.xml");
  vector<NNP> masters = parse_txt("hdnnp_structure.txt");
  for (NNP& master : masters) {
    master.feedforward(features, energy, dE_dG, natom, nfeature);
  }
}
