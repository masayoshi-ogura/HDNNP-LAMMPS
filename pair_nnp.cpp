/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair_nnp.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairNNP::PairNNP(LAMMPS *lmp) : Pair(lmp) {
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  nelements = 0;
  combinations = NULL;
  elements = NULL;
  masters = NULL;
  nG1params = nG2params = nG4params = 0;
  G1params = G2params = G4params = NULL;
  pca_transform = NULL;
  pca_mean = NULL;
  map = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNP::~PairNNP() {
  int i, j;
  if (copymode) return;

  if (combinations)
    for (i = 0; i < atom->ntypes; i++) delete[] combinations[i];
  delete[] combinations;
  if (elements)
    for (i = 0; i < nelements; i++) delete[] elements[i];
  delete[] elements;
  if (G1params)
    for (i = 0; i < nG1params; i++) delete[] G1params[i];
  delete[] G1params;
  if (G2params)
    for (i = 0; i < nG2params; i++) delete[] G2params[i];
  delete[] G2params;
  if (G4params)
    for (i = 0; i < nG4params; i++) delete[] G4params[i];
  delete[] G4params;
  delete[] preprocesses;
  delete[] pca_transform;
  delete[] pca_mean;
  delete[] scl_max;
  delete[] scl_min;
  delete[] std_mean;
  delete[] std_std;

  if (masters)
    for (i = 0; i < nelements; i++) {
      for (j = 0; j < masters[i]->depth; j++) delete masters[i]->layers[j];
      delete masters[i];
    }
  delete[] masters;

  if (allocated) {
    memory->destroy(cutsq);
    memory->destroy(setflag);
    delete[] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairNNP::compute(int eflag, int vflag) {
  int i, j, k, ii, jj, inum, jnum, p;
  int itype, jtype, iparam;
  double delx, dely, delz, evdwl, fx, fy, fz, fpair;
  int *ilist, *jlist, *numneigh, **firstneigh;
  vector<int> iG2s;
  vector<vector<int> > iG3s;
  VectorXd R, dR[3];
  MatrixXd cos, dcos[3];
  VectorXd G, dE_dG, F[3];
  double *G_raw;
  double ***dG_dr_raw;
  MatrixXd dG_dx, dG_dy, dG_dz;

  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];          // local index of I atom
    itype = map[type[i]];   // element
    jlist = firstneigh[i];  // indices of J neighbors of I atom
    jnum = numneigh[i];     // # of J neighbors of I atom

    geometry(i, jlist, jnum, R, cos, dR, dcos);

    memory->create(G_raw, nfeature, "G");
    memory->create(dG_dr_raw, 3, jnum, nfeature, "dG_dr");
    for (int a = 0; a < nfeature; a++) G_raw[a] = 0.0;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < jnum; b++)
        for (int c = 0; c < nfeature; c++) dG_dr_raw[a][b][c] = 0.0;

    feature_index(jlist, jnum, iG2s, iG3s);
    for (iparam = 0; iparam < nG1params; iparam++)
      G1(G1params[iparam], ntwobody * iparam, iG2s, jnum, R, dR, G_raw,
         dG_dr_raw);
    for (iparam = 0; iparam < nG2params; iparam++)
      G2(G2params[iparam], ntwobody * (nG1params + iparam), iG2s, jnum, R, dR,
         G_raw, dG_dr_raw);
    for (iparam = 0; iparam < nG4params; iparam++)
      G4(G4params[iparam],
         ntwobody * (nG1params + nG2params) + nthreebody * iparam, iG3s, jnum,
         R, cos, dR, dcos, G_raw, dG_dr_raw);

    G = Map<VectorXd>(G_raw, nfeature);
    memory->destroy(G_raw);

    dG_dx = Map<MatrixXd>(&dG_dr_raw[0][0][0], nfeature, jnum);
    dG_dy = Map<MatrixXd>(&dG_dr_raw[1][0][0], nfeature, jnum);
    dG_dz = Map<MatrixXd>(&dG_dr_raw[2][0][0], nfeature, jnum);
    memory->destroy(dG_dr_raw);

    for (p = 0; p < npreprocess; p++) {
      (this->*preprocesses[p])(itype, G, dG_dx, dG_dy, dG_dz);
    }

    masters[itype]->feedforward(G, dE_dG, eflag, evdwl);

    F[0].noalias() = dE_dG.transpose() * dG_dx;
    F[1].noalias() = dE_dG.transpose() * dG_dy;
    F[2].noalias() = dE_dG.transpose() * dG_dz;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      fx = F[0](jj);
      fy = F[1](jj);
      fz = F[2](jj);
      f[j][0] += -fx;
      f[j][1] += -fy;
      f[j][2] += -fz;

      if (evflag) {
        delx = x[i][0] - x[j][0];
        dely = x[i][1] - x[j][1];
        delz = x[i][2] - x[j][2];
        fpair = 0.0;
        k = 0;
        if (delx != 0.0) {
          fpair += fx / delx;
          k++;
        }
        if (dely != 0.0) {
          fpair += fy / dely;
          k++;
        }
        if (delz != 0.0) {
          fpair += fz / delz;
          k++;
        }
        fpair /= k;
        ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely,
                 delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairNNP::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  map = new int[n + 1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNNP::settings(int narg, char **arg) {
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNP::coeff(int narg, char **arg) {
  int i, j, n, idx;
  int ntypes = atom->ntypes;

  if (!allocated) allocate();

  if (narg != 3 + ntypes)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete[] elements[i];
    delete[] elements;
  }
  elements = new char *[ntypes];
  for (i = 0; i < ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i], "NULL") == 0) {
      map[i - 2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i], elements[j]) == 0) break;
    map[i - 2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j], arg[i]);
      nelements++;
    }
  }
  combinations = new int *[ntypes];
  for (i = 0; i < nelements; i++) combinations[i] = new int[ntypes];
  idx = 0;
  for (i = 0; i < nelements; i++)
    for (j = i; j < nelements; j++)
      combinations[i][j] = combinations[j][i] = idx++;
  ntwobody = nelements;
  nthreebody = idx;

  // read potential file and initialize potential parameters

  masters = new NNP *[nelements];
  read_file(arg[2]);
  setup_params();

  cutmax = 0.0;
  for (i = 0; i < nG1params; i++)
    if (G1params[i][0] > cutmax) cutmax = G1params[i][0];
  for (i = 0; i < nG2params; i++)
    if (G2params[i][0] > cutmax) cutmax = G2params[i][0];
  for (i = 0; i < nG4params; i++)
    if (G4params[i][0] > cutmax) cutmax = G4params[i][0];

  for (i = 1; i < ntypes + 1; i++) {
    for (j = 1; j < ntypes + 1; j++) {
      cutsq[i][j] = cutmax * cutmax;
      setflag[i][j] = 1;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNNP::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR,
               "Pair style Neural Network Potential requires newton pair on");
  // need a full neighbor list

  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNNP::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairNNP::get_next_line(ifstream &fin, stringstream &ss, int &nwords) {
  string line;
  int n;

  // remove failbit
  ss.clear();
  // clear stringstream buffer
  ss.str("");

  if (comm->me == 0)
    while (getline(fin, line))
      if (!line.empty() && line[0] != '#') break;

  n = line.size();
  MPI_Bcast(&n, 1, MPI_INT, 0, world);
  line.resize(n);

  MPI_Bcast(&line[0], n + 1, MPI_CHAR, 0, world);
  nwords = atom->count_words(line.c_str());
  ss << line;
}

void PairNNP::read_file(char *file) {
  ifstream fin;
  stringstream ss;
  string sym_func_type, preprocess, element, activation;
  int i, j, k, l, nwords;
  int ntype, depth, depthnum, insize, outsize, size;
  double Rc, eta, Rs, lambda, zeta;
  double *pca_transform_raw, *pca_mean_raw;
  double *scl_max_raw, *scl_min_raw;
  double *std_mean_raw, *std_std_raw;
  double *weight, *bias;

  if (comm->me == 0) {
    fin.open(file);
    if (!fin) {
      char str[128];
      sprintf(str, "Cannot open neural network potential file %s", file);
      error->one(FLERR, str);
    }
  }

  // symmetry function parameters
  nG1params = 0;
  nG2params = 0;
  nG4params = 0;
  get_next_line(fin, ss, nwords);
  ss >> ntype;

  for (i = 0; i < ntype; i++) {
    get_next_line(fin, ss, nwords);
    ss >> sym_func_type >> size;
    if (sym_func_type == "type1") {
      nG1params = size;
      G1params = new double *[nG1params];
      for (j = 0; j < nG1params; j++) {
        get_next_line(fin, ss, nwords);
        ss >> Rc;
        G1params[j] = new double[1]{Rc};
      }
    } else if (sym_func_type == "type2") {
      nG2params = size;
      G2params = new double *[nG2params];
      for (j = 0; j < nG2params; j++) {
        get_next_line(fin, ss, nwords);
        ss >> Rc >> eta >> Rs;
        G2params[j] = new double[3]{Rc, eta, Rs};
      }
    } else if (sym_func_type == "type4") {
      nG4params = size;
      G4params = new double *[nG4params];
      for (j = 0; j < nG4params; j++) {
        get_next_line(fin, ss, nwords);
        ss >> Rc >> eta >> lambda >> zeta;
        G4params[j] = new double[4]{Rc, eta, lambda, zeta};
      }
    }
  }
  nfeature = ntwobody * (nG1params + nG2params) + nthreebody * nG4params;

  // preprocess parameters
  get_next_line(fin, ss, nwords);
  ss >> npreprocess;
  preprocesses = new FuncPtr[npreprocess];

  for (i = 0; i < npreprocess; i++) {
    get_next_line(fin, ss, nwords);
    ss >> preprocess;

    if (preprocess == "pca") {
      preprocesses[i] = &PairNNP::pca;
      pca_transform = new MatrixXd[nelements];
      pca_mean = new VectorXd[nelements];
      for (j = 0; j < nelements; j++) {
        get_next_line(fin, ss, nwords);
        ss >> element >> outsize >> insize;
        pca_transform_raw = new double[insize * outsize];
        pca_mean_raw = new double[insize];

        for (k = 0; k < outsize; k++) {
          get_next_line(fin, ss, nwords);
          for (l = 0; ss >> pca_transform_raw[k * insize + l]; l++)
            ;
        }

        get_next_line(fin, ss, nwords);
        for (k = 0; ss >> pca_mean_raw[k]; k++)
          ;

        for (k = 0; k < nelements; k++)
          if (elements[k] == element) {
            pca_transform[k] =
                Map<MatrixXd>(pca_transform_raw, insize, outsize).transpose();
            pca_mean[k] = Map<VectorXd>(pca_mean_raw, insize);
          }
        delete[] pca_transform_raw;
        delete[] pca_mean_raw;
      }
    } else if (preprocess == "scaling") {
      preprocesses[i] = &PairNNP::scaling;
      scl_max = new VectorXd[nelements];
      scl_min = new VectorXd[nelements];

      get_next_line(fin, ss, nwords);
      ss >> scl_target_max >> scl_target_min;

      for (j = 0; j < nelements; j++) {
        get_next_line(fin, ss, nwords);
        ss >> element >> size;
        scl_max_raw = new double[size];
        scl_min_raw = new double[size];

        get_next_line(fin, ss, nwords);
        for (k = 0; ss >> scl_max_raw[k]; k++)
          ;

        get_next_line(fin, ss, nwords);
        for (k = 0; ss >> scl_min_raw[k]; k++)
          ;

        for (k = 0; k < nelements; k++)
          if (elements[k] == element) {
            scl_max[k] = Map<VectorXd>(scl_max_raw, size);
            scl_min[k] = Map<VectorXd>(scl_min_raw, size);
          }
        delete[] scl_max_raw;
        delete[] scl_min_raw;
      }
    } else if (preprocess == "standardization") {
      preprocesses[i] = &PairNNP::standardization;
      std_mean = new VectorXd[nelements];
      std_std = new VectorXd[nelements];

      for (j = 0; j < nelements; j++) {
        get_next_line(fin, ss, nwords);
        ss >> element >> size;
        std_mean_raw = new double[size];
        std_std_raw = new double[size];

        get_next_line(fin, ss, nwords);
        for (k = 0; ss >> std_mean_raw[k]; k++)
          ;

        get_next_line(fin, ss, nwords);
        for (k = 0; ss >> std_std_raw[k]; k++)
          ;

        for (k = 0; k < nelements; k++)
          if (elements[k] == element) {
            std_mean[k] = Map<VectorXd>(std_mean_raw, size);
            std_std[k] = Map<VectorXd>(std_std_raw, size);
          }
        delete[] std_mean_raw;
        delete[] std_std_raw;
      }
    }
  }

  // neural network parameters
  get_next_line(fin, ss, nwords);
  ss >> depth;
  for (i = 0; i < nelements; i++) masters[i] = new NNP(depth);

  for (i = 0; i < nelements * depth; i++) {
    get_next_line(fin, ss, nwords);
    ss >> element >> depthnum >> insize >> outsize >> activation;
    weight = new double[insize * outsize];
    bias = new double[outsize];

    for (j = 0; j < insize; j++) {
      get_next_line(fin, ss, nwords);
      for (k = 0; ss >> weight[j * outsize + k]; k++)
        ;
    }

    get_next_line(fin, ss, nwords);
    for (j = 0; ss >> bias[j]; j++)
      ;

    for (j = 0; j < nelements; j++)
      if (elements[j] == element)
        masters[j]->layers[depthnum] =
            new Layer(insize, outsize, weight, bias, activation);

    delete[] weight;
    delete[] bias;
  }
  if (comm->me == 0) fin.close();
}

/* ---------------------------------------------------------------------- */

void PairNNP::setup_params() {}

/* ---------------------------------------------------------------------- */

void PairNNP::geometry(int cnt, int *neighlist, int numneigh, VectorXd &R,
                       MatrixXd &cos, VectorXd *dR, MatrixXd *dcos) {
  int i, n;
  double **x = atom->x;
  MatrixXd r, dR_;

  double **r_;
  memory->create(r_, numneigh, 3, "r_");
  for (i = 0; i < numneigh; i++) {
    n = neighlist[i];
    r_[i][0] = x[n][0] - x[cnt][0];
    r_[i][1] = x[n][1] - x[cnt][1];
    r_[i][2] = x[n][2] - x[cnt][2];
  }

  r = Map<MatrixXd>(&r_[0][0], 3, numneigh);
  R = r.colwise().norm();
  dR_ = r.array().rowwise() / R.transpose().array();
  cos.noalias() = dR_.transpose() * dR_;
  for (i = 0; i < 3; i++) {
    dR[i] = dR_.row(i);
    dcos[i] = (R.cwiseInverse() * dR[i].transpose()) -
              (cos.array().colwise() * (dR[i].array() / R.array())).matrix();
  }

  memory->destroy(r_);
}

void PairNNP::feature_index(int *neighlist, int numneigh, std::vector<int> &iG2s,
                            vector< vector<int> > &iG3s) {
  int i, j, itype, jtype;
  int *type = atom->type;
  iG2s = vector<int>(numneigh);
  iG3s = vector<vector<int> >(numneigh, vector<int>(numneigh));
  for (i = 0; i < numneigh; i++) {
    itype = map[type[neighlist[i]]];
    iG2s[i] = itype;

    for (j = 0; j < numneigh; j++) {
      jtype = map[type[neighlist[j]]];
      iG3s[i][j] = combinations[itype][jtype];
    }
  }
}

void PairNNP::pca(int type, VectorXd &G, MatrixXd &dG_dx, MatrixXd &dG_dy,
                  MatrixXd &dG_dz) {
  G = pca_transform[type] * (G - pca_mean[type]);
  dG_dx = pca_transform[type] * dG_dx;
  dG_dy = pca_transform[type] * dG_dy;
  dG_dz = pca_transform[type] * dG_dz;
}

void PairNNP::scaling(int type, VectorXd &G, MatrixXd &dG_dx, MatrixXd &dG_dy,
                      MatrixXd &dG_dz) {
  G = ((G - scl_min[type]).array() *
       (scl_max[type] - scl_min[type]).array().inverse() *
       (scl_target_max - scl_target_min))
          .array() +
      scl_target_min;
  dG_dx = dG_dx.array().colwise() *
          (scl_max[type] - scl_min[type]).array().inverse() *
          (scl_target_max - scl_target_min);
  dG_dy = dG_dy.array().colwise() *
          (scl_max[type] - scl_min[type]).array().inverse() *
          (scl_target_max - scl_target_min);
  dG_dz = dG_dz.array().colwise() *
          (scl_max[type] - scl_min[type]).array().inverse() *
          (scl_target_max - scl_target_min);
}

void PairNNP::standardization(int type, VectorXd &G, MatrixXd &dG_dx,
                              MatrixXd &dG_dy, MatrixXd &dG_dz) {
  G = (G - std_mean[type]).array() * std_std[type].array().inverse();
  dG_dx = dG_dx.array().colwise() * std_std[type].array().inverse();
  dG_dy = dG_dy.array().colwise() * std_std[type].array().inverse();
  dG_dz = dG_dz.array().colwise() * std_std[type].array().inverse();
}
