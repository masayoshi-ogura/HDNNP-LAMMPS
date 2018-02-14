// #define EIGEN_USE_MKL_ALL
// #define EIGEN_NO_DEBUG
// #define EIGEN_DONT_PARALLELIZE
// #define EIGEN_MPL2_ONLY

#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
// #include <Eigen/Core>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#ifndef INCLUDED_NNP_H_
#define INCLUDED_NNP_H_

using namespace std;
// using namespace Eigen;
using namespace boost::property_tree;

class Layer {
 private:
  int in_size;
  int out_size;
  double** weight;
  double* bias;
  void set_activation(const string& act);
  // typedef double(FuncPtr)(double& x);
  // FuncPtr activation;
  double (Layer::*activation)(double& x);
  double tanh(double& x);
  double sigmoid(double& x);
  double identity(double& x);

 public:
  Layer();
  Layer(const int& in, const int& out, double** w, double* b,
        const string& act);
  void feedforward(double**& input, double***& deriv_input, double**& output,
                   double***& deriv_output, const int& natom,
                   const int& nfeature);
};

class NNP {
 private:
  int nlayer;

 public:
  string element;
  vector<Layer> layers;
  NNP(const int& depth, const string& e);
  void feedforward(double**& features, double*& energy, double**& dE_dG,
                   const int& natom, const int& nfeature);
};

vector<NNP> parse_xml(const string&);

vector<string> split(const string& input, char delimiter);
vector<NNP> parse_txt(const string&);

#endif
