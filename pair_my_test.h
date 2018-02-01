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

PairStyle(my/test, PairMyTest)

#else

#ifndef LMP_PAIR_MY_TEST
#define LMP_PAIR_MY_TEST

#include "pair.h"

namespace LAMMPS_NS {

class PairMyTest : public Pair {
 public:
  PairMyTest(class LAMMPS *);
  virtual ~PairMyTest();
  virtual void compute(int, int);

  int maxnumneigh();
  double norm(int, double *);
  double dist(int, int);
  double vector_dot(int,double *,double*);
  double fc(double);
  double G2(int, int, int);
  void sf_allocate();
  void prepare_relative_neighlist();
  void calc_riijj();
  void calc_fciijj();
  void calc_costheta_iijjkk();
  void calc_G2();
  void calc_G5();
  void init_features();
  void collect_features();
  std::string to_string(int);

  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  //void compute_inner();
  //void compute_middle();
  //void compute_outer(int, int);

 protected:
  int maxneigh, nG2, nG5, nfeatures;
  int **G2_feature_id, ***G5_feature_id;
  std::string *feature_names;
  double pi;
  double **riijj, **fciijj,**dfciijj_drjj, **G2_hypers,**G5_hypers,**features;
  double ***G2_matrix, ***costheta_iijjkk, ***relative_neigh_list, ***driijj_dxyzjj;
  double ****G5_matrix, ****features_derivs, ****dcostheta_iijjkk_dxyzjj;
  //double ****pG2ii_pxyzjj_matrix;

  double cut_global;
  double **cut;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double *cut_respa;

  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
