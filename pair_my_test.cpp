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
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_my_test.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairMyTest::PairMyTest(LAMMPS *lmp) : Pair(lmp)
{
  // old ones
  respa_enable = 1;
  writedata = 1;

  // SF inits
  maxneigh = 0;
  relative_neigh_list=NULL;                         //(ii, jj, 3)
  riijj = NULL;                                      //(ii, jj)
  driijj_dxyzjj = NULL;                              //(ii, jj, 3)
  fciijj = NULL;
  dfciijj_drjj = NULL; //(ii,jj)
  costheta_iijjkk = NULL;         // (ii,jj,kk)
  dcostheta_iijjkk_dxyzjj = NULL; // (ii,jj,kk,3)
  pi = 3.141592653589793238463;
  nG2 = 0;
  nG5 = 0;
  nfeatures = 0;
  G2_hypers = NULL; //[[yita0, yita1, ...],[Rs0, ... ]]  (2, nG2)
  G5_hypers = NULL; //[[ ,...], [ ,...], [ ,...]]        (3, nG5)
  G2_matrix = NULL; //                               (ii, jj, nG2)
  G5_matrix = NULL; //                               (ii, jj1, jj2, nG2)
  features = NULL;  //                               (ii, nfeatures)
  feature_names = NULL; //                              (nfeatures,)
  G2_feature_id = NULL; //                              (nG2, atom->ntypes)
  G5_feature_id = NULL; //                              (nG5, atom->ntypes, atom->ntypes)
  //pG2ii_pxyzjj_matrix = NULL; //                           (ii, jj, nG2, axis<3)
  features_derivs = NULL;    //                       (ii, nfeatures, jj, axis<3)
}

/* ---------------------------------------------------------------------- */

PairMyTest::~PairMyTest()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    // sf part
    memory->destroy(riijj);
    memory->destroy(fciijj);
    memory->destroy(costheta_iijjkk);
    memory->destroy(G2_hypers);
    memory->destroy(G2_feature_id);
    memory->destroy(G5_hypers);
    memory->destroy(G5_feature_id);
    memory->destroy(G2_matrix);
    memory->destroy(G5_matrix);
    memory->destroy(relative_neigh_list);
    memory->destroy(features);
    memory->destroy(features_derivs);
    memory->destroy(driijj_dxyzjj);
    memory->destroy(dfciijj_drjj);
    //memory->destroy(pG2ii_pxyzjj_matrix);

  }
}
/* ---------------------------------------------------------------------- */
void PairMyTest::sf_allocate(){
  if (riijj) {
    memory->destroy(riijj);
    memory->destroy(fciijj);
    memory->destroy(costheta_iijjkk);
    memory->destroy(G2_matrix);
    memory->destroy(G5_matrix);
    memory->destroy(driijj_dxyzjj);
    memory->destroy(dfciijj_drjj);
    memory->destroy(features_derivs);
    memory->destroy(dcostheta_iijjkk_dxyzjj);
    printf("Old r and fc, etc. have been destroyed\n");
  }
  memory->create(riijj,list->inum,maxneigh,"PairMyTest:riijj");
  memory->create(fciijj,list->inum,maxneigh,"PairMyTest:fciijj");
  memory->create(costheta_iijjkk, list->inum, maxneigh,maxneigh, "PairMyTest:costheta_iijjkk");
  memory->create(dcostheta_iijjkk_dxyzjj, list->inum, maxneigh, maxneigh, 3, "PairMyTest:dcostheta_iijjkk_dxyzj");
  memory->create(G2_matrix, list->inum, maxneigh, nG2, "PairMyTest:G2_matrix");
  memory->create(G5_matrix, list->inum, maxneigh, maxneigh, nG5, "PairMyTest:G5_matrix");
  memory->create(driijj_dxyzjj, list->inum, maxneigh, 3, "PairMyTest:driijj_dxyzjj");
  memory->create(dfciijj_drjj, list->inum, maxneigh,"PairMyTest:dfciijj_drjj");
  memory->create(features_derivs, list->inum, nfeatures, maxneigh, 3, "PairMyTest:features_derivs");


  int ii, feature_id, jj, axis;
  for (ii=0; ii<list->inum; ii++){
    for (feature_id=0; feature_id<nfeatures; feature_id++){
      for (jj=0; jj<maxneigh; jj++){
        for (axis=0; axis<3; axis++){
          features_derivs[ii][feature_id][jj][axis]=0;
        }
      }
    }
  }
  //memory->create(pG2ii_pxyzjj_matrix, list->inum, maxneigh, nG2, 3, "PairMyTest:pG2ii_pxyzjj_matrix");
}

int PairMyTest::maxnumneigh(){
  int i,j,ii,jj,inum,jnum, maxneigh;
  int *ilist,*jlist,*numneigh;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  maxneigh = 0;
  for (ii = 0; ii < inum; ii++){
    i = ilist[ii];
    if (numneigh[i] > maxneigh) maxneigh = numneigh[i];  
  }
  return maxneigh;
}

void PairMyTest::prepare_relative_neighlist(){
  memory->create(relative_neigh_list,list->inum,maxneigh,3,"PairMyTest:relative_neigh_list");
  int i,j,ii,jj,inum,jnum, axis;
  int *ilist,*jlist,*numneigh, **firstneigh;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii = 0; ii < inum; ii++){
    i = ilist[ii];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++){
      j = firstneigh[i][jj];
      j &= NEIGHMASK;
      for (axis=0; axis<3; axis++){
        relative_neigh_list[ii][jj][axis] = atom->x[j][axis] - atom->x[i][axis];
      }
    }
  }
}

double PairMyTest::dist(int i, int j){
  /* to use mkl*/
  int axis;
  double vec[3]={0., 0., 0.};
  double rsq;
  rsq=0.0;
  for (axis=0; axis<3; axis++){
    vec[axis] = atom->x[j][axis] - atom->x[i][axis];
    rsq += vec[axis]*vec[axis];
  }
  return sqrt(rsq);
}

double PairMyTest::norm(int size, double *vec){
  /* to use mkl*/
  int i;
  double rsq=0.0;
  for (i=0; i<size; i++){
    rsq+=vec[i]*vec[i];
  }
  return sqrt(rsq);
}

double PairMyTest::vector_dot(int size, double *a, double *b){
  /* to use mkl*/
  int i;
  double c=0.0;
  for (i=0; i<size; i++){
    c += a[i]*b[i]; 
  }
  return c;
}

void PairMyTest::calc_riijj(){
  int i,j,ii,jj,inum,jnum, axis;
  int *ilist,*jlist,*numneigh, **firstneigh;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii = 0; ii < inum; ii++){
    i = ilist[ii];
    jnum = numneigh[i];
    for (jj = 0; jj < maxneigh; jj++){
      jlist = firstneigh[i];
      j=jlist[jj];
      j &= NEIGHMASK;
      if (jj < jnum){
        riijj[ii][jj] = norm(3, relative_neigh_list[ii][jj]);
        for (axis = 0; axis<3; axis++)
          driijj_dxyzjj[ii][jj][axis] = relative_neigh_list[ii][jj][axis]/riijj[i][jj];
      }
      else {
        riijj[ii][jj] = 100.;
        for (axis = 0; axis<3; axis++)
          driijj_dxyzjj[ii][jj][axis] = 0;
      }
    }
  }
}

void PairMyTest::calc_costheta_iijjkk(){
  int i,j,k,ii,jj,kk,inum,jnum, axis;
  int *ilist,*jlist,*numneigh, **firstneigh;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii = 0; ii < inum; ii++){
    i = ilist[ii];
    jnum = numneigh[i];
    for (jj = 0; jj < maxneigh; jj++){
      costheta_iijjkk[ii][jj][jj] = 0;
      for (kk = 0; kk < maxneigh; kk++){
        costheta_iijjkk[ii][jj][kk] = vector_dot(3, relative_neigh_list[ii][jj], relative_neigh_list[ii][kk]) / riijj[ii][jj] / riijj[ii][kk] *(int)(jj!=kk);
        //printf("costheta_ijjkk i%d jj%d rijj %f xyz %f %f %f || kk%d rikk %f xyz %f %f %f value%.6f\n",i,jj,rijj[i][jj],relative_neigh_list[i][jj][0],relative_neigh_list[i][jj][1],relative_neigh_list[i][jj][2] ,kk,rijj[i][kk],relative_neigh_list[i][kk][0],relative_neigh_list[i][kk][1],relative_neigh_list[i][kk][2],costheta_ijjkk[i][jj][kk]);
        for (axis=0; axis < 3; axis++){
          dcostheta_iijjkk_dxyzjj[ii][jj][kk][axis]= costheta_iijjkk[ii][jj][kk] *(-relative_neigh_list[ii][jj][axis])/ pow(riijj[ii][jj],2) + relative_neigh_list[ii][kk][axis]/riijj[ii][jj]/riijj[ii][kk];
        }
      }
    }
  }
}

double PairMyTest::fc(double r){
    return 0.5*(cos(pi*r/cut_global)+1)*(int)(r < cut_global);
}

void PairMyTest::calc_fciijj(){
  int i,j,ii,jj,inum,jnum;
  int *ilist,*jlist,*numneigh, **firstneigh;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii = 0; ii < inum; ii++){
    i = ilist[ii];
    jnum = numneigh[i];
    for (jj = 0; jj < maxneigh; jj++){
      jlist = firstneigh[i];
      if (jj < jnum){
        fciijj[ii][jj] = fc(riijj[ii][jj]);
        dfciijj_drjj[ii][jj] = -0.5 * pi / cut_global * sin(pi*riijj[ii][jj]/cut_global)*(int)(riijj[ii][jj] < cut_global);
      }
      else {
        fciijj[ii][jj] = 0.;
      }
    }
  }
}

void PairMyTest::calc_G2(){
  int i,ii,jj,j,jjtype, G2_hyper_id, axis, feature_id;
  int *numneigh;
  double exp_rijrs2,pG2ii_pRiijj_part1, pG2ii_pRiijj_part2, pG2ii_pRiijj;
  numneigh = list->numneigh;
  for (ii=0; ii < list->inum; ii++){
    i = list->ilist[ii];
    for (jj=0; jj < numneigh[i]; jj++){
      j = list->firstneigh[i][jj];
      j &= NEIGHMASK;
      jjtype=atom->type[j];

      for (G2_hyper_id=0; G2_hyper_id < nG2; G2_hyper_id++){

        feature_id = G2_feature_id[G2_hyper_id][jjtype-1];

        exp_rijrs2 = exp(-G2_hypers[0][G2_hyper_id]*pow(riijj[ii][jj] - G2_hypers[1][G2_hyper_id],2));
        G2_matrix[ii][jj][G2_hyper_id] = exp_rijrs2*fciijj[ii][jj];

        pG2ii_pRiijj_part1 = -G2_hypers[0][G2_hyper_id]*2*(riijj[ii][jj] - G2_hypers[1][G2_hyper_id]) * G2_matrix[ii][jj][G2_hyper_id];
        pG2ii_pRiijj_part2 = exp_rijrs2*dfciijj_drjj[ii][jj];
        pG2ii_pRiijj = pG2ii_pRiijj_part1 + pG2ii_pRiijj_part2;
        
        for (axis=0; axis<3; axis++){
          //pG2i_pxyzjj_matrix[i][jj][G2_hyper_id][axis] = pG2i_pRijj * drijj_dxyzjj[i][jj][axis];
          // pG2/pxyz_ll = 0 when ll != jj, so no need to store ll dim

          features_derivs[ii][feature_id][jj][axis] = pG2ii_pRiijj * driijj_dxyzjj[ii][jj][axis]; 
          //printf("%d %d %d dfciijj_drjj%f pG2ii_pRiijj%f driijj_dxyzjj%f features_derivs%f\n",ii,jj,axis,dfciijj_drjj[ii][jj],pG2ii_pRiijj,driijj_dxyzjj[ii][jj][axis],features_derivs[ii][feature_id][jj][axis]);
        }
      }
    }
  }
}

double PairMyTest::G2(int ii, int jj, int G2_hyper_id){
  return exp( -G2_hypers[0][G2_hyper_id] * pow(riijj[ii][jj] - G2_hypers[1][G2_hyper_id], 2) ) * fciijj[ii][jj];
}

void PairMyTest::calc_G5(){
  // zita G5_hypers[0][G5_hyper_id]
  // lambda G5_hypers[1][G5_hyper_id]
  // yita G5_hypers[2][G5_hyper_id]
  int i,ii,jj0,j0, jj1,j1, G5_hyper_id, jj0type, jj1type, feature_id, axis, feature_id2;
  double G5_part1, G5_part2, G5_part3, G5_part4, G5_main_part, G5i_cos_part, _fcjk;
  double dG5ijk_dRj, dG5ijk_dcosthetaijk, dG5ijk_dRk; //k == jj1
  int *numneigh;
  numneigh = list->numneigh;

  for (G5_hyper_id = 0; G5_hyper_id< nG5; G5_hyper_id++){
    G5_part1 = pow(2,-G5_hypers[0][G5_hyper_id]+1 ); // 2^(-zita+1)
    for (ii=0; ii < list->inum; ii++){
      i=list->ilist[ii];
      for (jj0=0; jj0 < numneigh[i]; jj0++){
        j0=list->firstneigh[i][jj0];
        j0 &= NEIGHMASK;
        jj0type=atom->type[j0];
        // exclude self
        G5_matrix[ii][jj0][jj0][G5_hyper_id] = 0;

        for (jj1=jj0+1; jj1 <numneigh[i]; jj1++){
          j1=list->firstneigh[i][jj1];
          j1 &= NEIGHMASK;
          jj1type=atom->type[j1];
          feature_id = G5_feature_id[G5_hyper_id][jj0type-1][jj1type-1];


          _fcjk = fciijj[ii][jj0]*fciijj[ii][jj1];
          G5i_cos_part = G5_hypers[1][G5_hyper_id]*costheta_iijjkk[ii][jj0][jj1]+1;
          G5_part2 = pow(G5i_cos_part,G5_hypers[0][G5_hyper_id]);
          G5_part3 = exp(-G5_hypers[2][G5_hyper_id] * (riijj[ii][jj0]*riijj[ii][jj0]+riijj[ii][jj1]*riijj[ii][jj1]));
          G5_main_part = G5_part1*G5_part2*G5_part3;
          G5_matrix[ii][jj0][jj1][G5_hyper_id] = G5_main_part*_fcjk;
          // symmetric matrix
          G5_matrix[ii][jj1][jj0][G5_hyper_id] = G5_matrix[ii][jj0][jj1][G5_hyper_id] ;

          dG5ijk_dRj = -2* riijj[ii][jj0]* G5_hypers[2][G5_hyper_id] * G5_matrix[ii][jj0][jj1][G5_hyper_id] + G5_main_part * fciijj[ii][jj1] * dfciijj_drjj[ii][jj0];
          dG5ijk_dRk = -2* riijj[ii][jj1]* G5_hypers[2][G5_hyper_id] * G5_matrix[ii][jj1][jj0][G5_hyper_id] + G5_main_part * fciijj[ii][jj0] * dfciijj_drjj[ii][jj1];

          dG5ijk_dcosthetaijk = G5_hypers[0][G5_hyper_id] * G5_hypers[1][G5_hyper_id] * pow(G5i_cos_part, G5_hypers[0][G5_hyper_id]-1) * G5_part1 * G5_part3 * _fcjk;

          for (axis=0; axis<3; axis++){
            features_derivs[ii][feature_id][jj0][axis] += (dG5ijk_dRj * driijj_dxyzjj[ii][jj0][axis] + dG5ijk_dcosthetaijk * dcostheta_iijjkk_dxyzjj[ii][jj0][jj1][axis]);
            features_derivs[ii][feature_id][jj1][axis] += dG5ijk_dRk * driijj_dxyzjj[ii][jj1][axis] + dG5ijk_dcosthetaijk * dcostheta_iijjkk_dxyzjj[ii][jj1][jj0][axis];
          }
        }

        for (axis=0; axis<3; axis++)
          features_derivs[ii][G5_feature_id[G5_hyper_id][jj0type-1][jj0type-1]][jj0][axis] *=2;
      }
    }
  }
}

void PairMyTest::init_features(){
  int ii,jj, axis, feature_id;

  if (features == NULL){
    memory->create(features, list->inum, nfeatures, "PairMyTest:features");
  }
  for (ii=0; ii<list->inum; ii++){
    for (feature_id=0; feature_id<nfeatures; feature_id++){
      features[ii][feature_id]=0;
      /*
      for (jj=0; jj<maxneigh; jj++){
        for (axis=0; axis<3; axis++){
          features_derivs[ii][feature_id][jj][axis]=0;
        }
      }*/
    }
  }
}

void PairMyTest::collect_features(){
  /* could be implemented directly in the G2/G5 function and save memory for G2/G5_matrix*/

  // allocate features memory here and set all elements to zero
  // should check the difference between atom->natoms and list->inum


  int i,ii,jj0,j0, jj0type, jj1,jj1type,j1, G5_hyper_id, G2_hyper_id, feature_id;
  int *numneigh, *jlist, **firstneigh;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  init_features();
  printf("Initialized\n");
  for (ii=0; ii < list->inum; ii++){
    i = list->ilist[ii];
    jlist = firstneigh[i];
    for (jj0=0; jj0 < numneigh[i]; jj0++){
      j0 = jlist[jj0];
      j0 &= NEIGHMASK;
      jj0type = atom->type[j0];

      //printf("%d %d %d collecting G2\n", i, jj0, j0);
      // collect G2
      for (G2_hyper_id=0; G2_hyper_id < nG2; G2_hyper_id++){
        feature_id = G2_feature_id[G2_hyper_id][jj0type-1];
        //printf("jj0type:%d feature id:%d \n",jj0type,feature_id);
        features[ii][feature_id] += G2_matrix[ii][jj0][G2_hyper_id];
      }
      //printf("collecting G5\n");
      // collect G5
      for (jj1 = jj0+1; jj1 <numneigh[i]; jj1++){
        j1 = jlist[jj1];
        j1 &= NEIGHMASK;
        jj1type = atom->type[j1];
        //printf("i:%d jj0:%d j0:%d jj1:%d j1:%d collecting G5\n",i, jj0, j0,jj1,j1);
        for (G5_hyper_id = 0; G5_hyper_id< nG5; G5_hyper_id++){
          feature_id = G5_feature_id[G5_hyper_id][jj0type-1][jj1type-1];
          //printf("jj0type:%d jj1type:%d, feature id:%d \n",jj0type,jj1type,feature_id);

          features[ii][feature_id] += (1+(int)(jj0type==jj1type))*G5_matrix[ii][jj0][jj1][G5_hyper_id]; // True = 1, False =0
        }
      }
    }
  }
}

std::string PairMyTest::to_string(int a){
  std::string b;
  char tmp[20];
  sprintf(tmp, "%d", a);
  b = tmp;
  return b;
}

/* ---------------------------------------------------------------------- */

void PairMyTest::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  int feature_id, jj0, jj1;
  maxneigh = maxnumneigh();
  printf("#####################max neighbour num %d   ########################\n",maxneigh);
  //allocate variables every step?
  sf_allocate();

  prepare_relative_neighlist();
  calc_riijj();
  calc_fciijj();
  calc_costheta_iijjkk();
  calc_G2();
  calc_G5();
  collect_features();
  printf("Finish calculating features\n");


  double **x = atom->x;  // coordinates?
  double **f = atom->f;  // atomic forces?
  int *type = atom->type; // elements
  int nlocal = atom->nlocal; //??
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    printf("Atom_id: %i interal_id: %d Proc:%d xyz: %.8f %.8f %.8f type: %i n_neighbour: %i\n", i,ii, comm->me, atom->x[i][0], atom->x[i][1], atom->x[i][2], atom->type[i], list->numneigh[i] );
    /*
    if (ii == 66){
      int G5_hyper_id, G2_hyper_id, jj0type,axis,j0, j1, jj1type;

      printf("G5_matrix\n");
      for (G5_hyper_id = 0; G5_hyper_id<nG5; G5_hyper_id++){
        printf("G5 hyper id %d\n",G5_hyper_id);
        for (jj0 = 0; jj0 < jnum; jj0++) {
          for (jj1 = 0; jj1<jnum; jj1++){
            printf("%.6E\t",G5_matrix[ii][jj0][jj1][G5_hyper_id]);
          }
          printf("\n");
        }
      }


      printf("G2_deriv_matrix\n");
      for (G2_hyper_id=0; G2_hyper_id<nG2; G2_hyper_id++){
        printf("G2 hyper id: %d\n", G2_hyper_id);
        for (jj0=0; jj0 < jnum; jj0++){
          j0 = jlist[jj0];
          jj0type = atom->type[j0];
          feature_id = G2_feature_id[G2_hyper_id][jj0type-1];
          for (axis=0; axis<3; axis++){
            printf("%.6E\t", features_derivs[ii][feature_id][jj0][axis]);
          }
          printf("\n");
        }
        printf("\n");
      }

      printf("G5_deriv_matrix\n");
      for (G5_hyper_id = 0; G5_hyper_id<nG5; G5_hyper_id++){
        printf("G5 hyper id %d\n",G5_hyper_id);
        for (feature_id=nG2*atom->ntypes; feature_id<nfeatures; feature_id++){
          printf("feature_id %d \n",feature_id);
          for (jj0=0; jj0 < jnum; jj0++){
            j0 = jlist[jj0];
            jj0type = atom->type[j0];
            for (axis=0; axis<3; axis++){
              printf("%.6E\t", features_derivs[ii][feature_id][jj0][axis]);
            }
            printf("\n");
          }
          printf("\n");
        }
      }

    }*/
    
    for (feature_id=0; feature_id< nfeatures; feature_id++){
      printf("%s\t", feature_names[feature_id]);
    }
    printf("\n");
    for (feature_id=0; feature_id< nfeatures; feature_id++){
      printf("%.6f\t", features[ii][feature_id]);
    }
    printf("\n");
    //printf("j\tjx\tjy\tjz\tjtype\trij\tfcij\n");
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      
      //printf("%i\t%.8f\t%.8f\t%.8f\t%d\t%.4f\t%.4f\n", j, atom->x[j][0], atom->x[j][1], atom->x[j][2], atom->type[j], riijj[ii][jj],fciijj[ii][jj]);


      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- 

void PairMyTest::compute_inner()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listinner->inum;
  ilist = listinner->ilist;
  numneigh = listinner->numneigh;
  firstneigh = listinner->firstneigh;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];

  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 - rsw*rsw*(3.0 - 2.0*rsw);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- 

void PairMyTest::compute_middle()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listmiddle->inum;
  ilist = listmiddle->ilist;
  numneigh = listmiddle->numneigh;
  firstneigh = listmiddle->firstneigh;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;
        if (rsq < cut_in_on_sq) {
          rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
          fpair *= rsw*rsw*(3.0 - 2.0*rsw);
        }
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 + rsw*rsw*(2.0*rsw - 3.0);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- 

void PairMyTest::compute_outer(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listouter->inum;
  ilist = listouter->ilist;
  numneigh = listouter->numneigh;
  firstneigh = listouter->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          fpair = factor_lj*forcelj*r2inv;
          if (rsq < cut_in_on_sq) {
            rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
            fpair *= rsw*rsw*(3.0 - 2.0*rsw);
          }

          f[i][0] += delx*fpair;
          f[i][1] += dely*fpair;
          f[i][2] += delz*fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx*fpair;
            f[j][1] -= dely*fpair;
            f[j][2] -= delz*fpair;
          }
        }

        if (eflag) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            r2inv = 1.0/rsq;
            r6inv = r2inv*r2inv*r2inv;
            forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
            fpair = factor_lj*forcelj*r2inv;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_lj*forcelj*r2inv;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMyTest::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMyTest::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMyTest::coeff(int narg, char **arg)
{
  int n = atom->ntypes;
  if (!allocated) allocate();

  // SF part coeff read 
  int typeA, typeB0, typeB1, argid, G2_id, G5_id;
  if (narg > 5){
    for (argid = 0; argid < narg; argid++){
      if (strcmp(arg[argid],"G2")==0)
        nG2++;
      if (strcmp(arg[argid],"G5")==0)
        nG5++;
    }
    printf("Reading SF hypers, G2:%d, G5:%d\n",nG2,nG5);

    if (G2_hypers) {
      memory->destroy(G2_hypers);
      memory->destroy(G2_feature_id);
      printf("Warning: Previous G2 settings will be overwritten.");
    }
    if (G5_hypers) {
      memory->destroy(G5_hypers);
      memory->destroy(G5_feature_id);
      printf("Warning: Previous G5 settings will be overwritten.");
    }
    if (nG2 > 0) {
      memory->create(G2_hypers, 2, nG2, "PairMyTest:G2_hypers");
      memory->create(G2_feature_id, nG2, atom->ntypes,"PairMyTest:G2_feature_id");
    }
    if (nG5 > 0) {
      memory->create(G5_hypers, 3, nG5, "PairMyTest:G5_hypers");
      memory->create(G5_feature_id, nG5, atom->ntypes,atom->ntypes,"PairMyTest:G5_feature_id");
    }

    argid = 0;
    G2_id = 0;
    G5_id = 0;
    while (argid < narg){
      if (strcmp(arg[argid],"G2")==0){
        G2_hypers[0][G2_id] = force->numeric(FLERR,arg[argid+1]);
        G2_hypers[1][G2_id] = force->numeric(FLERR,arg[argid+2]);
        argid += 3;
        G2_id++;
        continue;
      }
      if (strcmp(arg[argid],"G5")==0){
        G5_hypers[0][G5_id] = force->numeric(FLERR,arg[argid+1]);
        G5_hypers[1][G5_id] = force->numeric(FLERR,arg[argid+2]);
        G5_hypers[2][G5_id] = force->numeric(FLERR,arg[argid+3]);
        argid += 4;
        G5_id++;
        continue;
      }
    }

    printf("SF hypers read in. The sequence of features is as follows:\n");
    nfeatures = 0;
    for ( typeB0 = 0; typeB0 < atom->ntypes; typeB0++){
      for ( G2_id=0 ; G2_id < nG2; G2_id++){
        printf("%d neighbor atom Type %d G2 %f %f\n", nfeatures, typeB0+1,G2_hypers[0][G2_id],G2_hypers[1][G2_id]);
        G2_feature_id[G2_id][typeB0] = nfeatures;
        nfeatures++;
      }
    }
    

    for ( typeB0 = 0; typeB0 < atom->ntypes; typeB0++){
      for ( typeB1 = typeB0; typeB1 < atom->ntypes; typeB1++){
        for ( G5_id=0 ; G5_id < nG5; G5_id++){
          printf("%d neighbor atom Type1 %d Type2 %d G5 %f %f %f\n", nfeatures,typeB0+1, typeB1+1, G5_hypers[0][G5_id],G5_hypers[1][G5_id], G5_hypers[2][G5_id]);
          G5_feature_id[G5_id][typeB0][typeB1] = nfeatures;
          nfeatures++;
        }
      }
    }
    printf("Finish reading coeff\n");

    std::string G2_name = "G2";
    std::string G5_name = "G5";
    
    feature_names = new std::string[nfeatures];
    //memory->create(feature_names, nfeatures , "PairMyTest::feature_names");

    for ( typeB0 = 0; typeB0 < atom->ntypes; typeB0++){
      for ( G2_id=0 ; G2_id < nG2; G2_id++){
        feature_names[G2_feature_id[G2_id][typeB0]] = G2_name + "_A" + to_string(typeB0) + "_h" + to_string(G2_id);
      }
    }

    for ( typeB0 = 0; typeB0 < atom->ntypes; typeB0++){
      for ( typeB1 = typeB0; typeB1 < atom->ntypes; typeB1++){
        for ( G5_id=0 ; G5_id < nG5; G5_id++){
          feature_names[G5_feature_id[G5_id][typeB0][typeB1]] = G5_name + "_A" + to_string(typeB0) + "_B" + to_string(typeB1) + "_h" + to_string(G5_id);
        }
      }
    }


  }
  // lj part old
  else{
    int ilo,ihi,jlo,jhi;
    force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
    force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);
  
    double epsilon_one = force->numeric(FLERR,arg[2]);
    double sigma_one = force->numeric(FLERR,arg[3]);
  
    double cut_one = cut_global;
    if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);
  
    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
        epsilon[i][j] = epsilon_one;
        sigma[i][j] = sigma_one;
        cut[i][j] = cut_one;
        setflag[i][j] = 1;
        count++;
      }
    }
  
    if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMyTest::init_style()
{
  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this,instance_me);
    else if (respa == 1) {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else irequest = neighbor->request(this,instance_me);

  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairMyTest::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMyTest::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMyTest::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMyTest::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMyTest::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMyTest::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMyTest::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMyTest::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairMyTest::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  double r2inv,r6inv,forcelj,philj;

  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  fforce = factor_lj*forcelj*r2inv;

  philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
    offset[itype][jtype];
  return factor_lj*philj;
}

/* ---------------------------------------------------------------------- */

void *PairMyTest::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return NULL;
}
