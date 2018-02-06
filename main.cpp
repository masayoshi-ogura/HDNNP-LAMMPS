#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
// #include <Eigen/Core>

// #include "neural_network_potential.h"
#include "neural_network_potential_without_Eigen.h"

using namespace std;
// using namespace Eigen;

int main() {
  int i, j;
  int natom = 10;
  int nfeature = 2;
  double** features = (double**)malloc(sizeof(double*) * natom);
  double* features_row = (double*)malloc(sizeof(double) * natom * nfeature);
  double* energy;
  double** dE_dG;

  for (i = 0; i < natom; ++i) {
    features[i] = features_row + i * nfeature;
    for (j = 0; j < nfeature; ++j) {
      features[i][j] = (i + 1.0) / (j + 1.0);
    }
  }

  vector<NNP> masters = parse_xml("hoge.xml");
  for (NNP& master : masters) {
    // features = 初期化
    master.feedforward(features, energy, dE_dG, natom, nfeature);
  }
}

// int main(){
//   vector<NNP> masters = parse_xml("hoge.xml");
//   BOOST_FOREACH (NNP& master, masters){
//     MatrixXd input(10, 2);
//     input << 0.67343895,  0.1912703,
//              0.53043039,  0.74080022,
//              0.6711433 ,  0.86926449,
//              0.25888694,  0.53292936,
//              0.29670314,  0.6873754 ,
//              0.81015958,  0.81623006,
//              0.05966972,  0.16160816,
//              0.80610846,  0.73712088,
//              0.45910406,  0.72128745,
//              0.7669888 ,  0.82978974;
//     cout << "input:" << endl;
//     cout << input << endl;
//     double output = master.energy(input);
//     cout << "output:" << endl;
//     cout << output << endl;
//   }
// }

// int main(){
//   const int nsample = 1000;
//   const int input = 100;
//   const int hidden = 100;
//   const int output = 1;
//
//   // weight, biasをランダム初期化
//   // Layer layer1(input, hidden);
//   // Layer layer2(hidden, output);
//
//   // weight, biasをvectorで与える
//   random_device rnd_device;
//   mt19937 mersenne_engine(rnd_device());
//   uniform_real_distribution<> dist(-1.0, 1.0);
//   auto gen = bind(dist, mersenne_engine);
//
//   vector<double> w1(input * hidden);
//   vector<double> b1(hidden);
//   generate(begin(w1), end(w1), gen);
//   generate(begin(b1), end(b1), gen);
//   Layer layer1(input, hidden, w1, b1, "tanh");
//
//   vector<double> w2(hidden * output);
//   vector<double> b2(output);
//   generate(begin(w2), end(w2), gen);
//   generate(begin(b2), end(b2), gen);
//   Layer layer2(hidden, output, w2, b2, "identity");
//
//   MatrixXd i(nsample, input);
//   i = MatrixXd::Random(nsample, input);
//   // i << 0.1, 0.2,
//   //      0.3, 0.4,
//   //      0.5, 0.6,
//   //      0.7, 0.8;
//
//   // cout << "input:" << endl;
//   // cout << i << endl;
//   // MatrixXd h = layer1.feedforward(i);
//   // cout << "hidden:" << endl;
//   // cout << h << endl;
//   // MatrixXd o = layer2.feedforward(h);
//   // cout << "output:" << endl;
//   // cout << o << endl;
//
//   vector<double> t(1000);
//   for (auto& it : t){
//     const auto start = chrono::system_clock::now();
//     i = layer1.feedforward(i);
//     const auto end = chrono::system_clock::now();
//     it = chrono::duration_cast<chrono::microseconds>(end - start).count();
//   }
//   cout << "Average: " << accumulate(t.begin(), t.end(), 0.0) / t.size() << "
//   ms. " << endl;
// }
