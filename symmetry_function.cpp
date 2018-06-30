//
// Created by Masayoshi Ogura on 2018/06/29.
//

#include "symmetry_function.h"

void G1(double *params, int iparam, int *iG2s, int numneigh, VectorXd &tanh, VectorXd *dR, double *G, double ***dG_dr) {
    int i, iG;
    VectorXd coeff, g, dg[3];
    double Rc = params[0];

    g = tanh.array().cube();
    coeff = -3.0 / Rc * (1.0 - tanh.array().square()) * tanh.array().square();
    dg[0] = coeff.array() * dR[0].array();
    dg[1] = coeff.array() * dR[1].array();
    dg[2] = coeff.array() * dR[2].array();

    for (i = 0; i < numneigh; i++) {
        iG = iparam + iG2s[i];
        G[iG] += g(i);
        dG_dr[0][i][iG] += dg[0](i);
        dG_dr[1][i][iG] += dg[1](i);
        dG_dr[2][i][iG] += dg[2](i);
    }
}

void G2(double *params, int iparam, int *iG2s, int numneigh, VectorXd &R, VectorXd &tanh, VectorXd *dR, double *G,
        double ***dG_dr) {
    int i, iG;
    VectorXd coeff, g, dg[3];
    double Rc = params[0];
    double eta = params[1];
    double Rs = params[2];

    g = (-eta * (R.array() - Rs).square()).exp() * tanh.array().cube();
    coeff = (-eta * (R.array() - Rs).square()).exp() * tanh.array().square() *
            (-2.0 * eta * (R.array() - Rs) * tanh.array() + 3.0 / Rc * (tanh.array().square() - 1.0));
    dg[0] = coeff.array() * dR[0].array();
    dg[1] = coeff.array() * dR[1].array();
    dg[2] = coeff.array() * dR[2].array();

    for (i = 0; i < numneigh; i++) {
        iG = iparam + iG2s[i];
        G[iG] += g(i);
        dG_dr[0][i][iG] += dg[0](i);
        dG_dr[1][i][iG] += dg[1](i);
        dG_dr[2][i][iG] += dg[2](i);
    }
}

void G4(double *params, int iparam, int **iG3s, int numneigh, VectorXd &R, VectorXd &tanh, MatrixXd &cos, VectorXd *dR,
        MatrixXd *dcos, double *G, double ***dG_dr) {
    int i, j, iG;
    double coeffs;
    VectorXd rad1, rad2;
    MatrixXd ang, g, coeff1, coeff2, dg[3];
    double Rc = params[0];
    double eta = params[1];
    double lambda = params[2];
    double zeta = params[3];

    coeffs = pow(2.0, 1 - zeta);
    ang = 1.0 + lambda * cos.array();
    rad1 = (-eta * R.array().square()).exp() * tanh.array().cube();
    rad2 = (-eta * R.array().square()).exp() * tanh.array().square() *
           (-2.0 * eta * R.array() * tanh.array() + 3.0 / Rc * (tanh.array().square() - 1.0));
    g = ((0.5 * coeffs * ang.array().pow(zeta)).colwise() * rad1.array()).rowwise() * rad1.transpose().array();
    coeff1 = ((coeffs * ang.array().pow(zeta)).colwise() * rad2.array()).rowwise() * rad1.transpose().array();
    coeff2 = ((zeta * lambda * coeffs * ang.array().pow(zeta - 1)).colwise() * rad1.array()).rowwise() *
             rad1.transpose().array();
    dg[0] = coeff1.array().colwise() * dR[0].array() + coeff2.array() * dcos[0].array();
    dg[1] = coeff1.array().colwise() * dR[1].array() + coeff2.array() * dcos[1].array();
    dg[2] = coeff1.array().colwise() * dR[2].array() + coeff2.array() * dcos[2].array();

    for (i = 0; i < numneigh; i++) {
        for (j = 0; j < numneigh; j++) {
            if (i == j) continue;
            iG = iparam + iG3s[i][j];
            G[iG] += g(i, j);
            dG_dr[0][i][iG] += dg[0](i, j);
            dG_dr[1][i][iG] += dg[1](i, j);
            dG_dr[2][i][iG] += dg[2](i, j);
        }
    }
}
