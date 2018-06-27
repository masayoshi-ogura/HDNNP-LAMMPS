/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nnp,PairNNP)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#include "pair.h"
#include "neural_network_potential.h"

namespace LAMMPS_NS {

    class PairNNP : public Pair {
    public:
        PairNNP(class LAMMPS *);

        virtual ~PairNNP();

        virtual void compute(int, int);

        void settings(int, char **);

        virtual void coeff(int, char **);

        virtual double init_one(int, int);

        virtual void init_style();

    protected:
        int nelements;                     // # of unique elements
        char **elements;                   // names of unique elements
        int *map;                          // mapping from atom types to elements
        NNP **masters;                     // parameter set for an I-J-K interaction
        int nparams, nG1params, nG2params, nG4params;
        double **G1params, **G2params, **G4params;
        int nfeature;


        virtual void allocate();

        void get_next_line(char [], char *, FILE *, int &);

        void read_file(char *);

        virtual void setup_params();

        void geometry(int, int *, int, VectorXd &, VectorXd &, MatrixXd &, VectorXd *, MatrixXd *);

        void feature_index(int, int *, int, int *, int **);

        void G1(int, int, int *, VectorXd &, VectorXd *, double *, double ***);

        void G2(int, int, int *, VectorXd &, VectorXd &, VectorXd *, double *, double ***);

        void G4(int, int, int **, VectorXd &, VectorXd &, MatrixXd &, VectorXd *, MatrixXd *, double *, double ***);
    };

}

#endif
#endif
