#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY

#include <iostream>
#include <string>
#include <Eigen/Core>

#ifndef INCLUDED_NNP_H_
#define INCLUDED_NNP_H_

using namespace std;
using namespace Eigen;

class Layer {
private:
    void set_activation(char *);

    typedef void (Layer::*FuncPtr)(MatrixXd &);

    typedef void (Layer::*FuncPtr2)(MatrixXd &, MatrixXd &);

    FuncPtr activation;

    FuncPtr2 activation2;

    void tanh(MatrixXd &);

    void deriv_tanh(MatrixXd &, MatrixXd &);

    void sigmoid(MatrixXd &);

    void deriv_sigmoid(MatrixXd &, MatrixXd &);

    void identity(MatrixXd &);

    void deriv_identity(MatrixXd &, MatrixXd &);

public:
    MatrixXd weight;
    VectorXd bias;

    Layer(int, int);

    Layer(int, int, double *, double *, char *);

    void feedforward(MatrixXd &);

    void feedforward2(MatrixXd &, MatrixXd &);
};


class NNP {
public:
    int depth;
    Layer **layers;

    NNP();

    NNP(int);

    void energy(int, double *, double &);

    void deriv(int, double *, double *);
};


#endif
