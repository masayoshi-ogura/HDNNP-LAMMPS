//
// Created by Masayoshi Ogura on 2018/06/~~.
//

#ifndef INCLUDED_NNP_H_
#define INCLUDED_NNP_H_
#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_MPL2_ONLY

#include <iostream>
#include <string>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

class Layer {
private:
    void set_activation(string);

    typedef void (Layer::*FuncPtr)(VectorXd &);

    typedef void (Layer::*FuncPtr2)(VectorXd &, VectorXd &);

    FuncPtr activation;

    FuncPtr2 activation2;

    void tanh(VectorXd &);

    void deriv_tanh(VectorXd &, VectorXd &);

    void elu(VectorXd &);

    void deriv_elu(VectorXd &, VectorXd &);

    void sigmoid(VectorXd &);

    void deriv_sigmoid(VectorXd &, VectorXd &);

    void identity(VectorXd &);

    void deriv_identity(VectorXd &, VectorXd &);

public:
    MatrixXd weight;
    VectorXd bias;

    Layer(int, int, double *, double *, string);

    ~Layer();

    void feedforward(VectorXd &);

    void feedforward2(VectorXd &, VectorXd &);
};


class NNP {
public:
    int depth;
    Layer **layers;

    NNP(int);

    ~NNP();

    void energy(VectorXd, double &);

    void deriv(VectorXd, VectorXd &);
};


#endif
