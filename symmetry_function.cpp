//
// Created by Masayoshi Ogura on 2018/06/29.
//

#include "symmetry_function.h"

void G1(vector<double> params, int iparam, vector<int> iG2s, int numneigh,
        VectorXd &R, VectorXd *dR, VectorXd &G, MatrixXd &dG_dx, MatrixXd &dG_dy, MatrixXd &dG_dz) {
  int j, iG;
  VectorXd tanh, coeff, g, dg[3];
  double Rc = params[0];

  tanh = (1.0 - R.array() / Rc).tanh();
  g = tanh.array().cube();
  coeff = -3.0 / Rc * (1.0 - tanh.array().square()) * tanh.array().square();
  dg[0] = coeff.array() * dR[0].array();
  dg[1] = coeff.array() * dR[1].array();
  dg[2] = coeff.array() * dR[2].array();

  for (j = 0; j < numneigh; j++) {
    if (R[j] > Rc) continue;
    iG = iparam + iG2s[j];
    G.coeffRef(iG) += g.coeffRef(j);
    dG_dx.coeffRef(iG, j) += dg[0].coeffRef(j);
    dG_dy.coeffRef(iG, j) += dg[1].coeffRef(j);
    dG_dz.coeffRef(iG, j) += dg[2].coeffRef(j);
  }
}

void G2(vector<double> params, int iparam, vector<int> iG2s, int numneigh,
        VectorXd &R, VectorXd *dR, VectorXd &G, MatrixXd &dG_dx, MatrixXd &dG_dy, MatrixXd &dG_dz) {
  int j, iG;
  VectorXd tanh, coeff, g, dg[3];
  double Rc = params[0];
  double eta = params[1];
  double Rs = params[2];
  tanh = (1.0 - R.array() / Rc).tanh();
  g = (-eta * (R.array() - Rs).square()).exp() * tanh.array().cube();
  coeff = (-eta * (R.array() - Rs).square()).exp() * tanh.array().square() *
          (-2.0 * eta * (R.array() - Rs) * tanh.array() +
           3.0 / Rc * (tanh.array().square() - 1.0));
  dg[0] = coeff.array() * dR[0].array();
  dg[1] = coeff.array() * dR[1].array();
  dg[2] = coeff.array() * dR[2].array();

  for (j = 0; j < numneigh; j++) {
    if (R[j] > Rc) continue;
    iG = iparam + iG2s[j];
    G.coeffRef(iG) += g.coeffRef(j);
    dG_dx.coeffRef(iG, j) += dg[0].coeffRef(j);
    dG_dy.coeffRef(iG, j) += dg[1].coeffRef(j);
    dG_dz.coeffRef(iG, j) += dg[2].coeffRef(j);
  }
}

void G4(vector<double> params, int iparam, vector<vector<int> > iG3s, int numneigh,
        VectorXd &R, MatrixXd &cos, VectorXd *dR, MatrixXd *dcos,
        VectorXd &G, MatrixXd &dG_dx, MatrixXd &dG_dy, MatrixXd &dG_dz) {
  int j, k, iG;
  double coeffs;
  VectorXd tanh, rad1, rad2;
  MatrixXd ang, g, coeff1, coeff2, dg[3];
  double Rc = params[0];
  double eta = params[1];
  double lambda = params[2];
  double zeta = params[3];

  tanh = (1.0 - R.array() / Rc).tanh();
  coeffs = pow(2.0, 1 - zeta);
  ang = 1.0 + lambda * cos.array();
  rad1 = (-eta * R.array().square()).exp() * tanh.array().cube();
  rad2 = (-eta * R.array().square()).exp() * tanh.array().square() *
         (-2.0 * eta * R.array() * tanh.array() +
          3.0 / Rc * (tanh.array().square() - 1.0));
  g = ((0.5 * coeffs * ang.array().pow(zeta)).colwise() * rad1.array())
          .rowwise() *
      rad1.transpose().array();
  coeff1 =
      ((coeffs * ang.array().pow(zeta)).colwise() * rad2.array()).rowwise() *
      rad1.transpose().array();
  coeff2 = ((zeta * lambda * coeffs * ang.array().pow(zeta - 1)).colwise() *
            rad1.array())
               .rowwise() *
           rad1.transpose().array();
  dg[0] = coeff1.array().colwise() * dR[0].array() +
          coeff2.array() * dcos[0].array();
  dg[1] = coeff1.array().colwise() * dR[1].array() +
          coeff2.array() * dcos[1].array();
  dg[2] = coeff1.array().colwise() * dR[2].array() +
          coeff2.array() * dcos[2].array();

  for (j = 0; j < numneigh; j++) {
    if (R[j] > Rc) continue;
    for (k = 0; k < numneigh; k++) {
      if (R[k] > Rc) continue;
      if (j == k) continue;
      iG = iparam + iG3s[j][k];
      G.coeffRef(iG) += g.coeffRef(j, k);
      dG_dx.coeffRef(iG, j) += dg[0].coeffRef(j, k);
      dG_dy.coeffRef(iG, j) += dg[1].coeffRef(j, k);
      dG_dz.coeffRef(iG, j) += dg[2].coeffRef(j, k);
    }
  }
}
