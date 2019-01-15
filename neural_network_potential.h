//
// Created by Masayoshi Ogura on 2018/06/~~.
//

#ifndef INCLUDED_NNP_H_
#define INCLUDED_NNP_H_
#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_MPL2_ONLY

#include "Eigen/Core"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace Eigen;

class Layer {
 private:
  void set_activation(string);

  typedef void (Layer::*FuncPtr)(VectorXd &, VectorXd &);

  FuncPtr activation;

  void tanh(VectorXd &, VectorXd &);

  void elu(VectorXd &, VectorXd &);

  void sigmoid(VectorXd &, VectorXd &);

  void identity(VectorXd &, VectorXd &);

 public:
  MatrixXd weight;
  VectorXd bias;

  Layer(int, int, vector<double> &, vector<double> &, string);

  ~Layer();

  void feedforward(VectorXd &, VectorXd &);
};

class NNP {
 public:
  int depth;
  vector<Layer> layers;

  NNP(int);

  ~NNP();

  void feedforward(VectorXd, VectorXd &, int, double &);
};

#endif
