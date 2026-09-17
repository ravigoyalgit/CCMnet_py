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
  
  int L = m->n_stats; // Or however your C struct tracks total stats
  int num_params = L - 1; 
  int k = (int)((sqrt(8.0 * num_params + 1.0) - 1.0) / 2.0 + 0.1);
  
  //Rprintf("--- DEBUG in function stat: L=%d, num_params=%d, k=%d ---\n", L, num_params, k);
  
  /* 1. Get Covariate types for the toggled nodes */
  // Accessing the end of inputparams where covPattern was appended
  Cov_types[0] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->toggletail) - 1]);
  Cov_types[1] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->togglehead) - 1]);

  /* 2. Count global distribution of covariate types */
  for (int i = 0; i < k; i++) {
    Num_Cov_type[i] = 0;
  } 
  for (counter = 0; counter < nwp->nnodes; counter++) {
    Cov_type = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + counter]);
    Num_Cov_type[Cov_type - 1]++;
  }

  /* 3. Setup Mixing Matrices */
  for (int i = 0; i < num_params; i++) {
    // Current state
    // i+1 because networkstatistics[0] is total edges
    nwp_mixing_matrix[i] = networkstatistics[i + 1];
    
    // Proposal state
    MHp_mixing_matrix[i] = networkstatistics[i + 1] + m->workspace[i + 1];
  } 

}

void calc_f_mixing(Network *nwp, int *Cov_types, int *Num_Cov_type, 
                   double *nwp_mixing_matrix, double *MHp_mixing_matrix,
                   double *prob_g_g2, double *prob_g2_g, int MHp_nedges,
                   Model *m, MHproposal *MHp, double *networkstatistics) {
  //int counter, Cov_type;
  ModelTerm *mtp2 = m->termarray;
  mtp2++; 
  
  /* 1. Get Covariate types for the toggled nodes */
  Cov_types[0] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->toggletail) - 1]);
  Cov_types[1] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->togglehead) - 1]);
  
  /* 2. Count global distribution of covariate types */
  //int L = m->n_stats; // Or however your C struct tracks total stats
  //int num_params = L - 1; 
  //int k = (int)((sqrt(8.0 * num_params + 1.0) - 1.0) / 2.0 + 0.1);
  
  // Now your DEBUG will finally show:
  //Rprintf("--- DEBUG in function f: L=%d, num_params=%d, k=%d ---\n", L, num_params, k);
  
  /* 3. Mixing Matrix Index Mapping */
  int c1 = Cov_types[0];
  int c2 = Cov_types[1];
  
  // Force j to be the larger level (Row) and i to be the smaller (Col)
  int row = (c1 > c2) ? c1 : c2;
  int col = (c1 > c2) ? c2 : c1;
  
  // Standard lower-triangular index (0-based)
  // (1,1)->0, (2,1)->1, (2,2)->2, (3,1)->3...
  int mixing_matrix_id = (row * (row - 1) / 2) + col - 1; 
  
  // Shift by 1 because networkstatistics[0] is 'edges'
  mixing_matrix_id = mixing_matrix_id + 1;
  //Rprintf("Index: %d \n", mixing_matrix_id);
  
  
  //Rprintf("Level1: %d, Count1: %d\n", Cov_types[0], Num_Cov_type[Cov_types[0]-1]);
  //Rprintf("Level2: %d, Count2: %d\n", Cov_types[1], Num_Cov_type[Cov_types[1]-1]);
  //Rprintf("Level1-2 edges: %f", networkstatistics[mixing_matrix_id]);
  
  /* 4. Probability g -> g2 */
  if (nwp->nedges < MHp_nedges) { // Add Edge
    
    if (Cov_types[0] == Cov_types[1]) {
      //Rprintf("Potential edges - g->g2 same: %f \n", ((Num_Cov_type[Cov_types[0]-1] * (Num_Cov_type[Cov_types[0]-1]-1)) * .5) );
      *prob_g_g2 = ((Num_Cov_type[Cov_types[0]-1] * (Num_Cov_type[Cov_types[0]-1]-1)) * .5)  - networkstatistics[mixing_matrix_id];
    }
    else {
      //Rprintf("Potential edges - g->g2 different: %f \n", Num_Cov_type[Cov_types[0]-1] * Num_Cov_type[Cov_types[1]-1]);
      *prob_g_g2 = Num_Cov_type[Cov_types[0]-1] * Num_Cov_type[Cov_types[1]-1] - networkstatistics[mixing_matrix_id];
    }
  } else { // Remove Edge
    *prob_g_g2 = networkstatistics[mixing_matrix_id];
  }

  /* 5. Probability g2 -> g */
  if (nwp->nedges > MHp_nedges) { // Add Edge (Forward was Remove)
    if (Cov_types[0] == Cov_types[1]) {
      //Rprintf("Potential edges - g2->g same: %f \n", ((Num_Cov_type[Cov_types[0]-1] * (Num_Cov_type[Cov_types[0]-1]-1)) * .5) );
      *prob_g2_g = ((Num_Cov_type[Cov_types[0]-1] * (Num_Cov_type[Cov_types[0]-1]-1)) * .5)  - (networkstatistics[mixing_matrix_id] + m->workspace[mixing_matrix_id]);
    }
    else {
      //Rprintf("Potential edges - g2->g different: %f \n", Num_Cov_type[Cov_types[0]-1] * Num_Cov_type[Cov_types[1]-1]);
      *prob_g2_g = Num_Cov_type[Cov_types[0]-1] * Num_Cov_type[Cov_types[1]-1] - (networkstatistics[mixing_matrix_id] + m->workspace[mixing_matrix_id]);
    }
  } else { // Remove Edge (Forward was Add)
    *prob_g2_g = networkstatistics[mixing_matrix_id] + m->workspace[mixing_matrix_id];
  }
  
  //Rprintf("Mixing Matrix g->g2 and g2->g: %f %f\n", *prob_g_g2, *prob_g2_g);
}
