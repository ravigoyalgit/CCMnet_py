#include <R.h>
#include <Rmath.h>
#include <math.h>
#include "CCMnet_netprop_mixing.h"

extern double calcCNR(int n, int r);

void calc_stat_mixing(Network *nwp, Model *m, MHproposal *MHp, double *networkstatistics, 
                      int *Cov_types, int *Num_Cov_type, 
                      double *nwp_mixing_matrix, double *MHp_mixing_matrix) {
  int counter, Cov_type;
  ModelTerm *mtp2 = m->termarray;
  mtp2++; 
  
  /* 1. Get Covariate types for the toggled nodes */
  Cov_types[0] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->toggletail) - 1]);
  Cov_types[1] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->togglehead) - 1]);
  
  /* 2. Count global distribution of covariate types */
  Num_Cov_type[0] = 0;
  Num_Cov_type[1] = 0;
  for (counter = 0; counter < nwp->nnodes; counter++) {
    Cov_type = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + counter]);
    Num_Cov_type[Cov_type - 1]++;
  }
  
  /* 3. Setup Mixing Matrices */
  nwp_mixing_matrix[0] = nwp->nedges - networkstatistics[1] - networkstatistics[2];
  nwp_mixing_matrix[1] = networkstatistics[1];
  nwp_mixing_matrix[2] = networkstatistics[2];
  
  int MHp_nedges = nwp->nedges + (int)m->workspace[0];
  MHp_mixing_matrix[0] = MHp_nedges - networkstatistics[1] - networkstatistics[2] - m->workspace[1] - m->workspace[2];
  MHp_mixing_matrix[1] = networkstatistics[1] + m->workspace[1];
  MHp_mixing_matrix[2] = networkstatistics[2] + m->workspace[2];
}

void calc_f_mixing(Network *nwp, int *Cov_types, int *Num_Cov_type, 
                   double *nwp_mixing_matrix, double *MHp_mixing_matrix,
                   double *prob_g_g2, double *prob_g2_g, int MHp_nedges) {
  
  /* 4. Probability g -> g2 */
  if (nwp->nedges < MHp_nedges) { // Add Edge
    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
      *prob_g_g2 = calcCNR(Num_Cov_type[0], 2) - nwp_mixing_matrix[0];
    }
    if (((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1))) {
      *prob_g_g2 = (double)Num_Cov_type[0] * Num_Cov_type[1] - nwp_mixing_matrix[1];
    }
    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
      *prob_g_g2 = calcCNR(Num_Cov_type[1], 2) - nwp_mixing_matrix[2];
    }
  } else { // Remove Edge
    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) *prob_g_g2 = nwp_mixing_matrix[0];
    if (((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1))) *prob_g_g2 = nwp_mixing_matrix[1];
    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) *prob_g_g2 = nwp_mixing_matrix[2];
  }
  
  /* 5. Probability g2 -> g */
  if (nwp->nedges > MHp_nedges) { // Add Edge (Forward was Remove)
    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
      *prob_g2_g = calcCNR(Num_Cov_type[0], 2) - MHp_mixing_matrix[0];
    }
    if (((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1))) {
      *prob_g2_g = (double)Num_Cov_type[0] * Num_Cov_type[1] - MHp_mixing_matrix[1];
    }
    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
      *prob_g2_g = calcCNR(Num_Cov_type[1], 2) - MHp_mixing_matrix[2];
    }
  } else { // Remove Edge (Forward was Add)
    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) *prob_g2_g = MHp_mixing_matrix[0];
    if (((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1))) *prob_g2_g = MHp_mixing_matrix[1];
    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) *prob_g2_g = MHp_mixing_matrix[2];
  }
}

void calc_probs_mixing(int num_mixing_terms, int *Num_Cov_type, 
                       double *nwp_mixing_matrix, double *MHp_mixing_matrix, 
                       double *meanvalues, double *varvalues,
                       double *pdf_gaussian_nwp, double *pdf_gaussian_MHp) {
  
  int counter, counter1, counter2;
  double nwp_mu_diff[3], MHp_mu_diff[3];
  int Num_Cov_type_dem[3] = {Num_Cov_type[0], Num_Cov_type[0], Num_Cov_type[1]};
  
  for (counter = 0; counter < num_mixing_terms; counter++) {
    nwp_mu_diff[counter] = (nwp_mixing_matrix[counter] / (1.0 * Num_Cov_type_dem[counter])) - meanvalues[counter];
    MHp_mu_diff[counter] = (MHp_mixing_matrix[counter] / (1.0 * Num_Cov_type_dem[counter])) - meanvalues[counter];
  }
  
  double nwp_int[3] = {0,0,0}, MHp_int[3] = {0,0,0};
  counter2 = -1;
  for (counter = 0; counter < (num_mixing_terms * num_mixing_terms); counter++) {
    counter1 = counter % num_mixing_terms;
    if (counter1 == 0) counter2++;
    nwp_int[counter2] += nwp_mu_diff[counter1] * varvalues[counter];
    MHp_int[counter2] += MHp_mu_diff[counter1] * varvalues[counter];
  }
  
  *pdf_gaussian_nwp = 0; *pdf_gaussian_MHp = 0;
  for (counter = 0; counter < num_mixing_terms; counter++) {
    *pdf_gaussian_nwp += nwp_int[counter] * nwp_mu_diff[counter];
    *pdf_gaussian_MHp += MHp_int[counter] * MHp_mu_diff[counter];
  }
  *pdf_gaussian_nwp *= -0.5;
  *pdf_gaussian_MHp *= -0.5;
}