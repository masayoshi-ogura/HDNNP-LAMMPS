//
// Created by Masayoshi Ogura on 2018/06/29.
//

#ifndef HDNNP_LAMMPS_SYMMETRY_FUNCTION_H
#define HDNNP_LAMMPS_SYMMETRY_FUNCTION_H
#define INCLUDED_NNP_H_
#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_MPL2_ONLY

#include <Eigen/Core>
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;

void G1(double *, int, int *, int, VectorXd &, VectorXd *, double *,
        double ***);

void G2(double *, int, int *, int, VectorXd &, VectorXd &, VectorXd *, double *,
        double ***);

void G4(double *, int, int **, int, VectorXd &, VectorXd &, MatrixXd &,
        VectorXd *, MatrixXd *, double *, double ***);

#endif  // HDNNP_LAMMPS_SYMMETRY_FUNCTION_H
