// #define EIGEN_USE_MKL_ALL
// #define EIGEN_NO_DEBUG
// #define EIGEN_DONT_PARALLELIZE
// #define EIGEN_MPL2_ONLY
#define ARRAY_LENGTH(array) (sizeof(array) / sizeof(array[0]))

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>
// #include <Eigen/Core>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#ifndef INCLUDED_NNP_H_
#define INCLUDED_NNP_H_

using namespace std;
// using namespace Eigen;
using namespace boost::property_tree;

class Layer
{
private:
  int in_size;
  int out_size;
  double **weight;
  double *bias;
  void set_activation(const string& act);
  typedef double (Layer::FuncPtr)(double& x);
  FuncPtr activation;
  double tanh(double& x);
  double sigmoid(double& x);
  double identity(double& x);
public:
  Layer(const int& in, const int& out, double** w, double** b, const string& act);
  void feedforward(double**& input, double***& deriv_input, double**& output, double***& deriv_output, const int& natom, const int& nfeature);
};


class NNP
{
private:
  int nlayer;
  string element;
public:
  vector<Layer> layers;
  NNP(const int& n, const string& e);
  void feedforward(const double** features, double* energy, double** dE_dG);
};

template <typename T>
vector<T> split_cast(const string& str);

vector<NNP> parse_xml(const string&);


#endif
