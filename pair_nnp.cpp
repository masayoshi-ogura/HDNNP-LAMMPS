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
    elements = NULL;
    masters = NULL;
    nparams = nG1params = nG2params = nG4params = 0;
    G1params = G2params = G4params = NULL;
    map = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNP::~PairNNP() {
    int i;
    if (copymode) return;

    if (elements) for (i = 0; i < nelements; i++) delete[] elements[i];
    delete[] elements;
    if (G1params) for (i = 0; i < nG1params; i++) delete[] G1params[i];
    delete[] G1params;
    if (G2params) for (i = 0; i < nG1params; i++) delete[] G2params[i];
    delete[] G2params;
    if (G4params) for (i = 0; i < nG1params; i++) delete[] G4params[i];
    delete[] G4params;

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
    int a, b, c;
    int itype, jtype, iparam, iiparam;
    // double evdwl;
    int *ilist, *jlist, *numneigh, **firstneigh;
    double *Rij, *tanhij, **cosijk, **dRij, ***dcosijk;
    double *Gi, *dEi_dGi, ***dGi_drj;
    VectorXd dE_dG, F;
    MatrixXd dG_dr;

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

        memory->create(Gi, nfeature, "pair:Gi");
        memory->create(dEi_dGi, nfeature, "pair:dEi_dGi");
        memory->create(dGi_drj, nfeature, jnum, 3, "pair:dGi_drj");
        memory->create(Rij, jnum, "pair:Rij");
        memory->create(tanhij, jnum, "pair:tanhij");
        memory->create(cosijk, jnum, jnum, "pair:cosijk");
        memory->create(dRij, jnum, 3, "pair:dRij");
        memory->create(dcosijk, jnum, jnum, 3, "pair:dcosijk");
        for (a = 0; a < nfeature; a++) {
            Gi[a] = 0;
            for (b = 0; b < jnum; b++)
                for (c = 0; c < 3; c++) dGi_drj[a][b][c] = 0;
        }

        geometry(i, jlist, jnum, Rij, tanhij, cosijk, dRij, dcosijk);

        for (iparam = 0; iparam < nparams; iparam++) {
            if (iparam < nG1params) {
                iiparam = iparam;
                G1(i, jlist, jnum, tanhij, dRij, Gi, dGi_drj, iiparam);
            } else if (iparam < nG1params + nG2params) {
                iiparam = iparam - nG1params;
                G2(i, jlist, jnum, Rij, tanhij, dRij, Gi, dGi_drj, iiparam);
            } else if (iparam < nG1params + nG2params + nG4params) {
                iiparam = iparam - nG1params - nG2params;
                G4(i, jlist, jnum, Rij, tanhij, cosijk, dRij, dcosijk, Gi, dGi_drj, iiparam);
            }
        }

        // if (eflag) masters[itype]->energy(nfeature, Gi, evdwl);
        masters[itype]->deriv(nfeature, Gi, dEi_dGi);

        dE_dG = Map<VectorXd>(&dEi_dGi[0], nfeature);
        dG_dr = Map<MatrixXd>(&dGi_drj[0][0][0], jnum * 3, nfeature);
        F.noalias() = dG_dr * dE_dG;

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            f[j][0] += -F(jj * 3 + 0);
            f[j][1] += -F(jj * 3 + 1);
            f[j][2] += -F(jj * 3 + 2);
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
    int i, j, n;
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
    char *element, *activation;
    int depth, depthnum, insize, outsize;
    double *weight, *bias;

    // open file on proc 0
    FILE *fp;
    if (comm->me == 0) {
        fp = force->open_potential(file);
        if (fp == NULL) {
            char str[128];
            sprintf(str, "Cannot open Stillinger-Weber potential file %s", file);
            error->one(FLERR, str);
        }
    }


    // title
    get_next_line(line, ptr, fp, nwords);


    // // general information  =>  set in PairNNP::coeff
    // get_next_line(line, ptr, fp, nwords);
    // nelements = atoi(line);
    // masters = new NNP *[nelements];
    //
    // get_next_line(line, ptr, fp, nwords);
    // elements = new char *[nwords];
    // for (i = 0; i < nwords; i++) {
    //     elements[i] = new char[3];
    //     strcpy(elements[i], strtok(i == 0 ? line : NULL, " \t\n\r\f"));
    // }


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

    nparams = nRc * (1 + neta * (nRs + nlambda * nzeta));
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


    // preconditioning parameters
    get_next_line(line, ptr, fp, nwords);
    if (atoi(line)) {
        // preconditioning layer
    }


    // neural network parameters
    get_next_line(line, ptr, fp, nwords);
    depth = atoi(line);
    for (i = 0; i < nelements; i++) masters[i] = new NNP(depth);

    for (k = 0; k < nelements * depth; k++) {
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

        for (i = 0; i < insize; i++) {
            get_next_line(line, ptr, fp, nwords);
            weight[i * outsize] = atof(strtok(line, " \t\n\r\f"));
            for (j = 1; j < outsize; j++)
                weight[i * outsize + j] = atof(strtok(NULL, " \t\n\r\f"));
        }

        get_next_line(line, ptr, fp, nwords);
        bias[0] = atof(strtok(line, " \t\n\r\f"));
        for (j = 1; j < outsize; j++) bias[j] = atof(strtok(NULL, " \t\n\r\f"));

        for (j = 0; j < nelements; j++) {
            if (strcmp(elements[j], element) == 0) {
                masters[j]->layers[depthnum] =
                        new Layer(insize, outsize, weight, bias, activation);
            }
        }

        delete[] weight;
        delete[] bias;
    }
}

/* ---------------------------------------------------------------------- */

void PairNNP::setup_params() {}

/* ---------------------------------------------------------------------- */

void
PairNNP::geometry(int cnt, int *neighlist, int numneigh, double *R, double *tanh, double **cos, double **dR,
                  double ***dcos) {
    int i, j, n;
    double **x = atom->x;
    double **r;
    memory->create(r, numneigh, 3, "r");

    for (i = 0; i < numneigh; i++) {
        n = neighlist[i];
        r[i][0] = x[n][0] - x[cnt][0];
        r[i][1] = x[n][1] - x[cnt][1];
        r[i][2] = x[n][2] - x[cnt][2];
        R[i] = sqrt(r[i][0] * r[i][0] + r[i][1] * r[i][1] + r[i][2] * r[i][2]);

        for (j = 0; j < nG1params; j++) tanh[i] = std::tanh(1.0 - R[i] / G1params[j][0]);
        // if # of parameter Rc > 1, type of tanh is `double **`
        // and in each symmetry function(G1,dG1...dG4), add line `double* tanh = tanh[iRc]`
        // for (j = 0; j<nG1params; j++) tanh[j][i] = tanh(1.0 - R[i] / nG1params[j][0]);

        dR[i][0] = r[i][0] / R[i];
        dR[i][1] = r[i][1] / R[i];
        dR[i][2] = r[i][2] / R[i];
    }

    for (i = 0; i < numneigh; i++) {
        for (j = 0; j < numneigh; j++) {
            cos[i][j] = (r[i][0] * r[j][0] + r[i][1] * r[j][1] + r[i][2] * r[j][2]) / (R[i] * R[j]);
            dcos[i][j][0] = (-r[i][0] * cos[i][j]) / pow(R[i], 2) + (r[j][0]) / (R[i] * R[j]);
            dcos[i][j][1] = (-r[i][1] * cos[i][j]) / pow(R[i], 2) + (r[j][1]) / (R[i] * R[j]);
            dcos[i][j][2] = (-r[i][2] * cos[i][j]) / pow(R[i], 2) + (r[j][2]) / (R[i] * R[j]);
        }
    }

    memory->destroy(r);
}

void
PairNNP::G1(int cnt, int *neighlist, int numneigh, double *tanh, double **dR, double *G, double ***dG_dr, int iparam) {
    int i, iG;
    double coeff;
    int *type = atom->type;
    int ctype = map[type[cnt]];
    double Rc = G1params[iparam][0];

    for (i = 0; i < numneigh; i++) {
        if (map[type[neighlist[i]]] == ctype) iG = 2 * iparam;
        else iG = 2 * iparam + 1;

        G[iG] += pow(tanh[i], 3);

        coeff = -3.0 / Rc * (1.0 - pow(tanh[i], 2)) * pow(tanh[i], 2);
        dG_dr[iG][i][0] += coeff * dR[i][0];
        dG_dr[iG][i][1] += coeff * dR[i][1];
        dG_dr[iG][i][2] += coeff * dR[i][2];
    }
}

void
PairNNP::G2(int cnt, int *neighlist, int numneigh, double *R, double *tanh, double **dR, double *G, double ***dG_dr,
            int iparam) {
    int i, iG;
    double coeff;
    int *type = atom->type;
    int ctype = map[type[cnt]];
    double Rc = G2params[iparam][0];
    double eta = G2params[iparam][1];
    double Rs = G2params[iparam][2];

    for (i = 0; i < numneigh; i++) {
        if (map[type[neighlist[i]]] == ctype) iG = 2 * (nG1params + iparam);
        else iG = 2 * (nG1params + iparam) + 1;

        G[iG] += exp(-eta * pow((R[i] - Rs), 2)) * pow(tanh[i], 3);

        coeff = exp(-eta * pow((R[i] - Rs), 2)) * pow(tanh[i], 2) *
                (-2.0 * eta * (R[i] - Rs) * tanh[i] + 3.0 / Rc * (pow(tanh[i], 2) - 1.0));
        dG_dr[iG][i][0] += coeff * dR[i][0];
        dG_dr[iG][i][1] += coeff * dR[i][1];
        dG_dr[iG][i][2] += coeff * dR[i][2];
    }
}

void
PairNNP::G4(int cnt, int *neighlist, int numneigh, double *R, double *tanh, double **cos, double **dR, double ***dcos,
            double *G, double ***dG_dr, int iparam) {
    int i, j, itype, jtype, iG;
    double ang, tmp1, tmp2;
    int *type = atom->type;
    int ctype = map[type[cnt]];
    double Rc = G4params[iparam][0];
    double eta = G4params[iparam][1];
    double lambda = G4params[iparam][2];
    double zeta = G4params[iparam][3];

    double rad1[numneigh];
    double rad2[numneigh];
    for (i = 0; i < numneigh; i++) {
        rad1[i] = exp(-eta * pow(R[i], 2)) * pow(tanh[i], 3);
        rad2[i] = exp(-eta * pow(R[i], 2)) * pow(tanh[i], 2) *
                  (-2.0 * eta * R[i] * tanh[i] + 3.0 / Rc * (pow(tanh[i], 2) - 1.0));
    }

    for (i = 0; i < numneigh; i++) {
        itype = map[type[neighlist[i]]];
        for (j = 0; j < i; j++) {
            jtype = map[type[neighlist[j]]];
            if (itype == ctype && jtype == ctype) iG = 2 * (nG1params + nG2params) + 3 * iparam;
            else if (itype != ctype && jtype != ctype) iG = 2 * (nG1params + nG2params) + 3 * iparam + 1;
            else iG = 2 * (nG1params + nG2params) + 3 * iparam + 2;

            G[iG] += pow(2.0, 1 - zeta) * pow(1.0 + lambda * cos[i][j], zeta) * rad1[i] * rad1[j];
        }

        for (j = 0; j < numneigh; j++) {
            if (i == j) continue;
            jtype = map[type[neighlist[j]]];
            if (itype == ctype && jtype == ctype) iG = 2 * (nG1params + nG2params) + 3 * iparam;
            else if (itype != ctype && jtype != ctype) iG = 2 * (nG1params + nG2params) + 3 * iparam + 1;
            else iG = 2 * (nG1params + nG2params) + 3 * iparam + 2;

            ang = 1.0 + lambda * cos[i][j];
            tmp1 = pow(2.0, 1 - zeta) * pow(ang, zeta) * rad2[i] * rad1[j];
            tmp2 = zeta * lambda * pow(2.0, 1 - zeta) * pow(ang, zeta - 1) * rad1[i] * rad1[j];
            dG_dr[iG][i][0] += tmp1 * dR[i][0] + tmp2 * dcos[i][j][0];
            dG_dr[iG][i][1] += tmp1 * dR[i][1] + tmp2 * dcos[i][j][1];
            dG_dr[iG][i][2] += tmp1 * dR[i][2] + tmp2 * dcos[i][j][2];
        }
    }
}

