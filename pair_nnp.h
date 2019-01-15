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

PairStyle(nnp, PairNNP)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#include "neural_network_potential.h"
#include "pair.h"
#include "symmetry_function.h"

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
  double cutmax;               // max cutoff for all elements
  int nelements;               // # of unique elements
  int ntwobody;                // # of 2-body combinations
  int nthreebody;              // # of 3-body combinations
  vector<vector<int> > combinations;  // index of combination of 2 element
  vector<string> elements;     // names of unique elements
  vector<int> map;             // mapping from atom types to elements
  vector<NNP> masters;         // parameter set for an I-J-K interaction
  int nG1params, nG2params, nG4params, ndirectedG2params;
  vector<vector<double> > G1params, G2params, G4params, directedG2params;
  int nfeature;
  int npreprocess;
  vector<MatrixXd> pca_transform;
  vector<VectorXd> pca_mean;
  vector<VectorXd> scl_max;
  vector<VectorXd> scl_min;
  double scl_target_max;
  double scl_target_min;
  vector<VectorXd> std_mean;
  vector<VectorXd> std_std;

  virtual void allocate();

  void get_next_line(ifstream &, stringstream &, int &);

  void read_file(char *);

  virtual void setup_params();

  void geometry(int, int *, int, VectorXd *, VectorXd &, MatrixXd &, VectorXd *,
                MatrixXd *);

  void feature_index(int *, int, std::vector<int> &, vector< vector<int> > &);

  typedef void (PairNNP::*FuncPtr)(int, VectorXd &, MatrixXd &, MatrixXd &,
                                   MatrixXd &);

  vector<FuncPtr> preprocesses;

  void pca(int, VectorXd &, MatrixXd &, MatrixXd &, MatrixXd &);

  void scaling(int, VectorXd &, MatrixXd &, MatrixXd &, MatrixXd &);

  void standardization(int, VectorXd &, MatrixXd &, MatrixXd &, MatrixXd &);
};

}  // namespace LAMMPS_NS

#endif
#endif
