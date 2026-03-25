#include <R.h>
#include <math.h>
#include "CCMnet_netprop_prob_dist.h"

double quadratic_form(double *V, double *mu, double *inv_sigma, int dim) {
  double total = 0.0;
  
  for (int i = 0; i < dim; i++) {
    double diff_i = V[i] - mu[i];
    for (int j = 0; j < dim; j++) {
      total += diff_i * inv_sigma[j * dim + i] * (V[j] - mu[j]);
    }
  }
  return total;
}

void calc_prob_dist(double *v_old, double *v_new, int dim, int *prob_type,
                    double *meanvalues, double *varvalues,
                    double *pdf_old, double *pdf_new) {
  
  int dist_selector = prob_type[5];
  *pdf_old = 0.0;
  *pdf_new = 0.0;
  
  // --- CASE A: Multivariate Normal (The Matrix Case) ---
  if (dist_selector == 1 && dim > 1) { 
    // Here, varvalues is treated as the Precision Matrix (Inverse Covariance)
    // result = (V - mu)^T * Precision * (V - mu)
    *pdf_old = -0.5 * quadratic_form(v_old, meanvalues, varvalues, dim);
    *pdf_new = -0.5 * quadratic_form(v_new, meanvalues, varvalues, dim);
  } 
  
  // --- CASE B: Independent Distributions (The Loop Case) ---
  else {
    for (int i = 0; i < dim; i++) {
      // --- Normal (1) ---
      if (dist_selector == 1) {
        *pdf_old += -0.5 * pow((v_old[i] - meanvalues[i]), 2.0) / varvalues[i];
        *pdf_new += -0.5 * pow((v_new[i] - meanvalues[i]), 2.0) / varvalues[i];
      }
      // --- Log Normal (2) ---
      else if (dist_selector == 2) {
        *pdf_old += -0.5 * pow((log(v_old[i]) - meanvalues[i]), 2.0) / varvalues[i];
        *pdf_new += -0.5 * pow((log(v_new[i]) - meanvalues[i]), 2.0) / varvalues[i];
      }
      // --- Poisson (3) ---
      else if (dist_selector == 3) {
        // Using your specific delta-log logic
        double lambda = meanvalues[i];
        
        // Full Log-Likelihood for each bin
        // We can ignore the '-lambda' and 'log(constant)' because they cancel out in the ratio
        if (v_old[i] > 0) {
          *pdf_old += v_old[i] * log(lambda) - lgammafn(v_old[i] + 1.0);
        } else {
          *pdf_old += 0; // log(1) for v=0 case (since 0! = 1 and lambda^0 = 1)
        }
        
        if (v_new[i] > 0) {
          *pdf_new += v_new[i] * log(lambda) - lgammafn(v_new[i] + 1.0);
        } else {
          *pdf_new += 0;
        }
      }
      // --- Uniform (4) ---
      else if (dist_selector == 4) {
        *pdf_old += 0;
        *pdf_new += 0;
      }
      // --- Beta (5) ---
      else if (dist_selector == 5) {
        double alpha = meanvalues[i];
        double beta  = varvalues[i];

        *pdf_old += (alpha - 1.0) * log(v_old[i]) + (beta - 1.0) * log(1.0 - v_old[i]);
        *pdf_new += (alpha - 1.0) * log(v_new[i]) + (beta - 1.0) * log(1.0 - v_new[i]);
      }
      // --- Dirichlet-Multinomial (6) ---
      else if (dist_selector == 6) {
        // 1. Ensure we pull the alpha/mv parameter for this specific bin i
        int mv = (int)round(meanvalues[i]);
        
        // 2. We must use the same loop boundaries as your original code
        // Original: counter2 = (nwp_Deg_Distr[counter] + 1) to (nwp_Deg_Distr[counter] + mv)
        // New version using j as the offset:
        for (int j = 1; j <= mv; j++) {
          // We use v_old[i] + j to replicate (nwp_Deg_Distr[counter] + 1), etc.
          *pdf_old += log(v_old[i] + (double)j);
          *pdf_new += log(v_new[i] + (double)j);
        }
      }
      // --- Non-parametric (99) ---
      else if (dist_selector == 99) {
        *pdf_old += log(meanvalues[(int)v_old[i]]);
        *pdf_new += log(meanvalues[(int)v_new[i]]);
      }
    }
  }
}

