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
  
  // --- Multivariate Normal (0) ---
  if (dist_selector == 0) { 
    // Here, varvalues is treated as the Precision Matrix (Inverse Covariance)
    // result = (V - mu)^T * Precision * (V - mu)
    *pdf_old = -0.5 * quadratic_form(v_old, meanvalues, varvalues, dim);
    *pdf_new = -0.5 * quadratic_form(v_new, meanvalues, varvalues, dim);
  } 
  
  // --- Normal (1) ---
  else if (dist_selector == 1) {
    for (int i = 0; i < dim; i++) {
      *pdf_old += -0.5 * pow((v_old[i] - meanvalues[i]), 2.0) / varvalues[i];
      *pdf_new += -0.5 * pow((v_new[i] - meanvalues[i]), 2.0) / varvalues[i];
    }
  }
  
  // --- Log Normal (2) ---
  else if (dist_selector == 2) {
    for (int i = 0; i < dim; i++) {
      *pdf_old += -0.5 * pow((log(v_old[i]) - meanvalues[i]), 2.0) / varvalues[i];
      *pdf_new += -0.5 * pow((log(v_new[i]) - meanvalues[i]), 2.0) / varvalues[i];
    }
  }
  
  // --- Poisson (3) ---
  else if (dist_selector == 3) {
    for (int i = 0; i < dim; i++) {
      
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
  }
  
  // --- Uniform (4) ---
  else if (dist_selector == 4) {
    *pdf_old = 0;
    *pdf_new = 0;
  }
  
  // --- Beta (5) ---
  else if (dist_selector == 5) {
    for (int i = 0; i < dim; i++) {
      double alpha = meanvalues[i];
      double beta  = varvalues[i];
      
      *pdf_old += (alpha - 1.0) * log(v_old[i]) + (beta - 1.0) * log(1.0 - v_old[i]);
      *pdf_new += (alpha - 1.0) * log(v_new[i]) + (beta - 1.0) * log(1.0 - v_new[i]);
    }
  }
  
  // --- Dirichlet-Multinomial (6) ---
  else if (dist_selector == 6) {
    for (int i = 0; i < dim; i++) {
      double a = meanvalues[i];
      if (a <= 0) a = 1e-6; 
      
      // We only calculate bin-specific 'rewards'.
      // This is mathematically equivalent to a Multinomial 
      // with an implicit uniform prior on the total sum N.
      *pdf_old += lgammafn(v_old[i] + a) - lgammafn(v_old[i] + 1.0) - lgammafn(a);
      *pdf_new += lgammafn(v_new[i] + a) - lgammafn(v_new[i] + 1.0) - lgammafn(a);
    }
  }
  
  // --- Gamma Distribution (7) ---
  else if (dist_selector == 7) {
    for (int i = 0; i < dim; i++) {
      double alpha = meanvalues[i]; // Shape (alpha)
      double beta = varvalues[i];   // Rate (beta)
      
      // Handling the support [0, inf)
      // If x is 0 and alpha < 1, log(x) is -inf, but (alpha-1) is negative, 
      // leading to +inf (the singularity).
      // If x is 0 and alpha > 1, it leads to -inf (correctly rejecting 0).
      
      if (v_old[i] > 0) {
        *pdf_old += (alpha - 1.0) * log(v_old[i]) - (beta * v_old[i]);
      } else if (v_old[i] == 0) {
        if (alpha < 1.0) *pdf_old += 1e10;  // Approximation of the singularity
        else if (alpha > 1.0) *pdf_old += -1e10; // Probability is 0
        else *pdf_old += 0.0; // Alpha = 1 (Exponential), kernel is e^0 = 1, log(1)=0
      } else {
        *pdf_old += -1e10; // Reject negative values (outside support)
      }
      
      if (v_new[i] > 0) {
        *pdf_new += (alpha - 1.0) * log(v_new[i]) - (beta * v_new[i]);
      } else if (v_new[i] == 0) {
        if (alpha < 1.0) *pdf_new += 1e10;
        else if (alpha > 1.0) *pdf_new += -1e10;
        else *pdf_new += 0.0;
      } else {
        *pdf_new += -1e10;
      }
    }
  }
  
  // --- NP (99) ---
  else if (dist_selector == 99) {
    for (int i = 0; i < dim; i++) {
      *pdf_old += log(meanvalues[(int)v_old[i]]);
      *pdf_new += log(meanvalues[(int)v_new[i]]);
    }
  }
  
}

