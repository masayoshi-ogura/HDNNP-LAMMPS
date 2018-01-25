#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/range/combine.hpp>

#ifndef INCLUDED_NNP_H_
#define INCLUDED_NNP_H_

using namespace std;
using namespace Eigen;
using namespace boost::property_tree;

class Layer
{
private:
  const int in_size;
  const int out_size;
  const MatrixXd weight;
  const RowVectorXd bias;
  void set_activation(const string& act);
  typedef MatrixXd (Layer::*FuncPtr)(const MatrixXd&);
  FuncPtr activation;
  MatrixXd tanh(const MatrixXd& input);
  MatrixXd sigmoid(const MatrixXd& input);
  MatrixXd identity(const MatrixXd& input);
public:
  Layer(const int& in, const int& out);
  Layer(const int& in, const int& out, vector<double>& w, vector<double>& b, const string& act);
  MatrixXd feedforward(MatrixXd& input);
};


class NNP
{
private:
  const int nlayer;
  const string element;
public:
  vector<Layer> layers;
  NNP(const int& n, const string& element);
  double energy(MatrixXd m);
  VectorXd forces(MatrixXd m);
};

template <typename T>
vector<T> split_cast(const string& str);

vector<NNP> parse_xml(const string&);


#endif
