#include <iostream>
#include <numeric>
#include <random>
#include <chrono>
#include <Eigen/Core>

#include "neural_network_potential.h"

using namespace std;
using namespace Eigen;

int main(){
  const int nsample = 1000;
  const int input = 100;
  const int hidden = 100;
  const int output = 1;
  
  // weight, biasをランダム初期化
  // Layer layer1(input, hidden);
  // Layer layer2(hidden, output);

  // weight, biasをvectorで与える
  random_device rnd_device;
  mt19937 mersenne_engine(rnd_device());
  uniform_real_distribution<> dist(-1.0, 1.0);
  auto gen = bind(dist, mersenne_engine);

  vector<double> w1(input * hidden);
  vector<double> b1(hidden);
  generate(begin(w1), end(w1), gen);
  generate(begin(b1), end(b1), gen);
  Layer layer1(input, hidden, w1, b1, "tanh");

  vector<double> w2(hidden * output);
  vector<double> b2(output);
  generate(begin(w2), end(w2), gen);
  generate(begin(b2), end(b2), gen);
  Layer layer2(hidden, output, w2, b2, "identity");

  MatrixXd i(nsample, input);
  i = MatrixXd::Random(nsample, input);
  // i << 0.1, 0.2,
  //      0.3, 0.4,
  //      0.5, 0.6,
  //      0.7, 0.8;

  // cout << "input:" << endl;
  // cout << i << endl;
  // MatrixXd h = layer1.feedforward(i);
  // cout << "hidden:" << endl;
  // cout << h << endl;
  // MatrixXd o = layer2.feedforward(h);
  // cout << "output:" << endl;
  // cout << o << endl;

  vector<double> t(1000);
  for (auto& it : t){
    const auto start = chrono::system_clock::now();
    i = layer1.feedforward(i);
    const auto end = chrono::system_clock::now();
    it = chrono::duration_cast<chrono::microseconds>(end - start).count();
  }
  cout << "Average: " << accumulate(t.begin(), t.end(), 0.0) / t.size() << " ms. " << endl;
}
