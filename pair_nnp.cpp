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
#include <iostream>
#include <iomanip>
#include "pair_nnp.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

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
    components = NULL;
    mean = NULL;
    map = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNP::~PairNNP() {
    int i, j;
    if (copymode) return;

    if (combinations) for (i = 0; i < atom->ntypes; i++) delete[] combinations[i];
    delete[] combinations;
    if (elements) for (i = 0; i < nelements; i++) delete[] elements[i];
    delete[] elements;
    if (G1params) for (i = 0; i < nG1params; i++) delete[] G1params[i];
    delete[] G1params;
    if (G2params) for (i = 0; i < nG1params; i++) delete[] G2params[i];
    delete[] G2params;
    if (G4params) for (i = 0; i < nG1params; i++) delete[] G4params[i];
    delete[] G4params;
    delete[] components;
    delete[] mean;

    if (masters) for (i = 0; i < nelements; i++) {
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
    // only for eflag = vflag = 0

    int i, j, ii, jj, inum, jnum;
    int itype, jtype, iparam;
    // double evdwl;
    int *ilist, *jlist, *numneigh, **firstneigh;
    int *iG2s, **iG3s;
    VectorXd R, tanh, dR[3];
    MatrixXd cos, dcos[3];
    VectorXd G, dE_dG, F[3];
    double *G_raw, ***dG_dr_raw;
    MatrixXd dG_dx, dG_dy, dG_dz;

    // evdwl = 0.0;
    // if (eflag || vflag) ev_setup(eflag, vflag);
    // else evflag = vflag_fdotr = 0;

    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];          // local index of I atom
        itype = map[type[i]];   // element
        jlist = firstneigh[i];  // indices of J neighbors of I atom
        jnum = numneigh[i];     // # of J neighbors of I atom


        geometry(i, jlist, jnum, R, tanh, cos, dR, dcos);

        memory->create(G_raw, nfeature, "G");
        memory->create(dG_dr_raw, 3, jnum, nfeature, "dG_dr");
        for (int a = 0; a < nfeature; a++) G_raw[a] = 0.0;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < jnum; b++)
                for (int c = 0; c < nfeature; c++) dG_dr_raw[a][b][c] = 0.0;

        iG2s = new int[jnum];
        iG3s = new int *[jnum];
        for (jj = 0; jj < jnum; jj++) iG3s[jj] = new int[jnum];
        feature_index(jlist, jnum, iG2s, iG3s);
        for (iparam = 0; iparam < nG1params; iparam++)
            G1(G1params[iparam], 2 * iparam, iG2s, jnum, tanh, dR, G_raw, dG_dr_raw);
        for (iparam = 0; iparam < nG2params; iparam++)
            G2(G2params[iparam], 2 * (nG1params + iparam), iG2s, jnum, R, tanh, dR, G_raw, dG_dr_raw);
        for (iparam = 0; iparam < nG4params; iparam++)
            G4(G4params[iparam], 2 * (nG1params + nG2params) + 3 * iparam, iG3s,
               jnum, R, tanh, cos, dR, dcos, G_raw, dG_dr_raw);
        delete[] iG2s;
        for (jj = 0; jj < jnum; jj++) delete[] iG3s[jj];
        delete[] iG3s;

        // if (eflag) masters[itype]->energy(nfeature, Gi, evdwl);

        G = Map<VectorXd>(G_raw, nfeature);
        memory->destroy(G_raw);

        dG_dx = Map<MatrixXd>(&dG_dr_raw[0][0][0], nfeature, jnum);
        dG_dy = Map<MatrixXd>(&dG_dr_raw[1][0][0], nfeature, jnum);
        dG_dz = Map<MatrixXd>(&dG_dr_raw[2][0][0], nfeature, jnum);
        memory->destroy(dG_dr_raw);

        if (preproc_flag) (this->*preproc_func)(itype, G, dG_dx, dG_dy, dG_dz);

        masters[itype]->deriv(G, dE_dG);

        F[0].noalias() = dE_dG.transpose() * dG_dx;
        F[1].noalias() = dE_dG.transpose() * dG_dy;
        F[2].noalias() = dE_dG.transpose() * dG_dz;

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            f[j][0] += -F[0](jj);
            f[j][1] += -F[1](jj);
            f[j][2] += -F[2](jj);
        }

    }
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
    for (i = 0; i < nelements; i++) combinations[i] = new int [ntypes];
    idx = 0;
    for (i = 0; i < nelements; i++)
        for (j = i; j < nelements; j++)
            combinations[i][j] = combinations[j][i] = idx++;

    // read potential file and initialize potential parameters

    masters = new NNP *[nelements];
    read_file(arg[2]);
    setup_params();

    for (int i = 1; i < ntypes + 1; i++) {
        for (int j = 1; j < ntypes + 1; j++) {
            cutsq[i][j] = G1params[nG1params - 1][0] * G1params[nG1params - 1][0];
            setflag[i][j] = 1;
        }
    }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNNP::init_style() {
    if (atom->tag_enable == 0)
        error->all(FLERR, "Pair style neural network requires atom IDs");
    if (force->newton_pair == 0)
        error->all(FLERR, "Pair style neural network requires newton pair on");

    // need a full neighbor list

    int irequest = neighbor->request(this, instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNNP::init_one(int i, int j) {
    return G1params[nG1params - 1][0];
}

/* ---------------------------------------------------------------------- */

void PairNNP::get_next_line(char line[], char *ptr, FILE *fp, int &nwords) {
    int i, j, n;
    int eof = 0;

    if (comm->me == 0) {
        ptr = fgets(line, MAXLINE, fp);
        if (ptr == NULL) {
            eof = 1;
            fclose(fp);
        } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) return;
    MPI_Bcast(&n, 1, MPI_INT, 0, world);
    MPI_Bcast(line, n, MPI_CHAR, 0, world);

    // skip line if blank

    nwords = atom->count_words(line);
    if (nwords == 0) get_next_line(line, ptr, fp, nwords);
}

void PairNNP::read_file(char *file) {
    int i, j, k, l, nwords;
    double *Rc, *eta, *Rs, *lambda, *zeta;
    int nRc, neta, nRs, nlambda, nzeta;
    char line[MAXLINE], *ptr;  // read data and pointer for each iteration
    char *preproc, *element, *activation;
    int depth, depthnum, insize, outsize;
    double *weight, *bias;
    double *components_raw, *mean_raw;

    // open file on proc 0
    FILE *fp;
    if (comm->me == 0) {
        fp = force->open_potential(file);
        if (fp == NULL) {
            char str[128];
            sprintf(str, "Cannot open Neural Network Potential file %s", file);
            error->one(FLERR, str);
        }
    }


    // title
    get_next_line(line, ptr, fp, nwords);


    // symmetry function parameters
    get_next_line(line, ptr, fp, nRc);
    Rc = new double[nRc];
    Rc[0] = atof(strtok(line, " \t\n\r\f"));
    for (i = 1; i < nRc; i++) Rc[i] = atof(strtok(NULL, " \t\n\r\f"));

    get_next_line(line, ptr, fp, neta);
    eta = new double[neta];
    eta[0] = atof(strtok(line, " \t\n\r\f"));
    for (i = 1; i < neta; i++) eta[i] = atof(strtok(NULL, " \t\n\r\f"));

    get_next_line(line, ptr, fp, nRs);
    Rs = new double[nRs];
    Rs[0] = atof(strtok(line, " \t\n\r\f"));
    for (i = 1; i < nRs; i++) Rs[i] = atof(strtok(NULL, " \t\n\r\f"));

    get_next_line(line, ptr, fp, nlambda);
    lambda = new double[nlambda];
    lambda[0] = atof(strtok(line, " \t\n\r\f"));
    for (i = 1; i < nlambda; i++) lambda[i] = atof(strtok(NULL, " \t\n\r\f"));

    get_next_line(line, ptr, fp, nzeta);
    zeta = new double[nzeta];
    zeta[0] = atof(strtok(line, " \t\n\r\f"));
    for (i = 1; i < nzeta; i++) zeta[i] = atof(strtok(NULL, " \t\n\r\f"));

    nG1params = nRc;
    nG2params = nRc * neta * nRs;
    nG4params = nRc * neta * nlambda * nzeta;
    nfeature = 2 * nG1params + 2 * nG2params + 3 * nG4params;
    G1params = new double *[nG1params];
    G2params = new double *[nG2params];
    G4params = new double *[nG4params];
    for (i = 0; i < nRc; i++) {
        G1params[i] = new double[1]{Rc[i]};
        for (j = 0; j < neta; j++) {
            for (k = 0; k < nRs; k++) G2params[(i * neta + j) * nRs + k] = new double[3]{Rc[i], eta[j], Rs[k]};
            for (k = 0; k < nlambda; k++) {
                for (l = 0; l < nzeta; l++) {
                    G4params[((i * neta + j) * nlambda + k) * nzeta + l] = new double[4]{Rc[i], eta[j], lambda[k],
                                                                                         zeta[l]};
                }
            }
        }
    }
    delete[] Rc;
    delete[] eta;
    delete[] Rs;
    delete[] lambda;
    delete[] zeta;


    // preprocess parameters
    get_next_line(line, ptr, fp, nwords);
    if (atoi(line)) {
        preproc_flag = 1;
        get_next_line(line, ptr, fp, nwords);
        preproc = new char[10];
        strcpy(preproc, strtok(line, " \t\n\r\f"));

        if (strcmp(preproc, "pca") == 0) {
            preproc_func = &PairNNP::PCA;
            components = new MatrixXd[nelements];
            mean = new VectorXd[nelements];
            for (i = 0; i < nelements; i++) {
                get_next_line(line, ptr, fp, nwords);
                element = new char[3];
                strcpy(element, strtok(line, " \t\n\r\f"));
                outsize = atoi(strtok(NULL, " \t\n\r\f"));
                insize = atoi(strtok(NULL, " \t\n\r\f"));
                components_raw = new double[insize * outsize];
                mean_raw = new double[insize];

                for (j = 0; j < outsize; j++) {
                    get_next_line(line, ptr, fp, nwords);
                    components_raw[j * insize] = atof(strtok(line, " \t\n\r\f"));
                    for (k = 1; k < insize; k++) components_raw[j * insize + k] = atof(strtok(NULL, " \t\n\r\f"));
                }

                get_next_line(line, ptr, fp, nwords);
                mean_raw[0] = atof(strtok(line, " \t\n\r\f"));
                for (j = 1; j < insize; j++) mean_raw[j] = atof(strtok(NULL, " \t\n\r\f"));

                for (j = 0; j < nelements; j++) {
                    if (strcmp(elements[j], element) == 0) {
                        components[j] = Map<MatrixXd>(components_raw, insize, outsize).transpose();
                        mean[j] = Map<VectorXd>(mean_raw, insize);
                    }
                }
                delete[] element;
                delete[] components_raw;
                delete[] mean_raw;
            }
        }
        delete[] preproc;
    } else {
        preproc_flag = 0;
    }


    // neural network parameters
    get_next_line(line, ptr, fp, nwords);
    depth = atoi(line);
    for (i = 0; i < nelements; i++) masters[i] = new NNP(depth);

    for (i = 0; i < nelements * depth; i++) {
        get_next_line(line, ptr, fp, nwords);
        element = new char[3];
        strcpy(element, strtok(line, " \t\n\r\f"));
        depthnum = atoi(strtok(NULL, " \t\n\r\f")) - 1;
        insize = atoi(strtok(NULL, " \t\n\r\f"));
        outsize = atoi(strtok(NULL, " \t\n\r\f"));
        activation = new char[10];
        strcpy(activation, strtok(NULL, " \t\n\r\f"));
        weight = new double[insize * outsize];
        bias = new double[outsize];

        for (j = 0; j < insize; j++) {
            get_next_line(line, ptr, fp, nwords);
            weight[j * outsize] = atof(strtok(line, " \t\n\r\f"));
            for (k = 1; k < outsize; k++)
                weight[j * outsize + k] = atof(strtok(NULL, " \t\n\r\f"));
        }

        get_next_line(line, ptr, fp, nwords);
        bias[0] = atof(strtok(line, " \t\n\r\f"));
        for (j = 1; j < outsize; j++) bias[j] = atof(strtok(NULL, " \t\n\r\f"));

        for (j = 0; j < nelements; j++)
            if (strcmp(elements[j], element) == 0)
                masters[j]->layers[depthnum] =
                        new Layer(insize, outsize, weight, bias, activation);

        delete[] element;
        delete[] activation;
        delete[] weight;
        delete[] bias;
    }
}

/* ---------------------------------------------------------------------- */

void PairNNP::setup_params() {}

/* ---------------------------------------------------------------------- */

void
PairNNP::geometry(int cnt, int *neighlist, int numneigh, VectorXd &R, VectorXd &tanh, MatrixXd &cos, VectorXd *dR,
                  MatrixXd *dcos) {
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
    for (i = 0; i < nG1params; i++) tanh = (1.0 - R.array() / G1params[i][0]).tanh();
    dR_ = r.array().rowwise() / R.transpose().array();
    cos.noalias() = dR_.transpose() * dR_;
    for (i = 0; i < 3; i++) {
        dR[i] = dR_.row(i);
        dcos[i] =
                (R.cwiseInverse() * dR[i].transpose()) - (cos.array().colwise() * (dR[i].array() / R.array())).matrix();
    }

    memory->destroy(r_);
}

void PairNNP::feature_index(int *neighlist, int numneigh, int *iG2s, int **iG3s) {
    int i, j, itype, jtype;
    int *type = atom->type;
    for (i = 0; i < numneigh; i++) {
        itype = map[type[neighlist[i]]];
        iG2s[i] = itype;

        for (j = 0; j < numneigh; j++) {
            jtype = map[type[neighlist[j]]];
            iG3s[i][j] = combinations[itype][jtype];
        }
    }
}

void PairNNP::PCA(int type, VectorXd &G, MatrixXd &dG_dx, MatrixXd &dG_dy, MatrixXd &dG_dz) {
    G = components[type] * (G - mean[type]);
    dG_dx = components[type] * dG_dx;
    dG_dy = components[type] * dG_dy;
    dG_dz = components[type] * dG_dz;
}
