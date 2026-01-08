#include <R.h>
#include <math.h>
#include "CCMnet_netprop_degmixing.h"

void calc_prob_degmixing(Model *m, double *networkstatistics, double *meanvalues, 
                         double *varvalues, double *pdf_gaussian_nwp, 
                         double *pdf_gaussian_MHp) {
  
  int counter, counter1, counter2;
  int n_stats_minus_one = m->n_stats - 1;
  
  // 1. Calculate the difference from the mean for current and proposed states
  double nwp_mu_diff[n_stats_minus_one];
  double MHp_mu_diff[n_stats_minus_one];
  
  for (counter = 0; counter < n_stats_minus_one; counter++) {
    nwp_mu_diff[counter] = (double)(networkstatistics[counter + 1]) - meanvalues[counter];
    MHp_mu_diff[counter] = (double)(networkstatistics[counter + 1] + m->workspace[counter + 1]) - meanvalues[counter];
  }
  
  // 2. Compute intermediate matrices (Vector-Matrix multiplication: diff' * Precision)
  double nwp_intermediate_mat[n_stats_minus_one];
  double MHp_intermediate_mat[n_stats_minus_one];
  
  counter2 = -1;
  for (counter = 0; counter < (n_stats_minus_one) * (n_stats_minus_one); counter++) {
    counter1 = counter % (n_stats_minus_one);
    
    if (counter1 == 0) {
      counter2 += 1;
      nwp_intermediate_mat[counter2] = 0;
      MHp_intermediate_mat[counter2] = 0;
    }
    
    nwp_intermediate_mat[counter2] += nwp_mu_diff[counter1] * varvalues[counter];
    MHp_intermediate_mat[counter2] += MHp_mu_diff[counter1] * varvalues[counter];
  }
  
  // 3. Finalize Quadratic Form: -.5 * (diff' * Precision * diff)
  *pdf_gaussian_nwp = 0;
  *pdf_gaussian_MHp = 0;
  
  for (counter = 0; counter < n_stats_minus_one; counter++) {
    *pdf_gaussian_nwp += nwp_intermediate_mat[counter] * nwp_mu_diff[counter];
    *pdf_gaussian_MHp += MHp_intermediate_mat[counter] * MHp_mu_diff[counter];
  }
  
  *pdf_gaussian_nwp = -0.5 * (*pdf_gaussian_nwp);
  *pdf_gaussian_MHp = -0.5 * (*pdf_gaussian_MHp);
}