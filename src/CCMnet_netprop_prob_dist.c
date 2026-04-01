#include <R.h>
#include <math.h>
#include "CCMnet_netprop_prob_dist.h"

double quadratic_form(double *V, double *mu, double *inv_sigma, int distr_dim) {
  double total = 0.0;
  
  for (int i = 0; i < distr_dim; i++) {
    double diff_i = V[i] - mu[i];
    for (int j = 0; j < distr_dim; j++) {
      total += diff_i * inv_sigma[j * distr_dim + i] * (V[j] - mu[j]);
    }
  }
  return total;
}

void calc_prob_dist(double *g_stats, double *gp_stats, int distr_dim, int *prob_type,
                    double *p1, double *p2,
                    double *g_pdf, double *gp_pdf) {
  
  int dist_selector = prob_type[5];
  *g_pdf = 0.0;
  *gp_pdf = 0.0;
  
  // --- Multivariate Normal (0) ---
  if (dist_selector == 0) { 
    // Here, p2 is treated as the Precision Matrix (Inverse Covariance)
    // result = (V - mu)^T * Precision * (V - mu)
    *g_pdf = -0.5 * quadratic_form(g_stats, p1, p2, distr_dim);
    *gp_pdf = -0.5 * quadratic_form(gp_stats, p1, p2, distr_dim);
  } 
  
  // --- Normal (1) ---
  else if (dist_selector == 1) {
    for (int i = 0; i < distr_dim; i++) {
      *g_pdf += -0.5 * pow((g_stats[i] - p1[i]), 2.0) / p2[i];
      *gp_pdf += -0.5 * pow((gp_stats[i] - p1[i]), 2.0) / p2[i];
    }
  }
  
  // --- Log Normal (2) ---
  else if (dist_selector == 2) {
    for (int i = 0; i < distr_dim; i++) {
      *g_pdf += -0.5 * pow((log(g_stats[i]) - p1[i]), 2.0) / p2[i];
      *gp_pdf += -0.5 * pow((log(gp_stats[i]) - p1[i]), 2.0) / p2[i];
    }
  }
  
  // --- Poisson (3) ---
  else if (dist_selector == 3) {
    for (int i = 0; i < distr_dim; i++) {
      
      // Using your specific delta-log logic
      double lambda = p1[i];
      
      // Full Log-Likelihood for each bin
      // We can ignore the '-lambda' and 'log(constant)' because they cancel out in the ratio
      if (g_stats[i] > 0) {
        *g_pdf += g_stats[i] * log(lambda) - lgammafn(g_stats[i] + 1.0);
      } else {
        *g_pdf += 0; // log(1) for v=0 case (since 0! = 1 and lambda^0 = 1)
      }
      
      if (gp_stats[i] > 0) {
        *gp_pdf += gp_stats[i] * log(lambda) - lgammafn(gp_stats[i] + 1.0);
      } else {
        *gp_pdf += 0;
      }
    }
  }
  
  // --- Uniform (4) ---
  else if (dist_selector == 4) {
    *g_pdf = 0;
    *gp_pdf = 0;
  }
  
  // --- Beta (5) ---
  else if (dist_selector == 5) {
    for (int i = 0; i < distr_dim; i++) {
      double alpha = p1[i];
      double beta  = p2[i];
      
      *g_pdf += (alpha - 1.0) * log(g_stats[i]) + (beta - 1.0) * log(1.0 - g_stats[i]);
      *gp_pdf += (alpha - 1.0) * log(gp_stats[i]) + (beta - 1.0) * log(1.0 - gp_stats[i]);
    }
  }
  
  // --- Dirichlet-Multinomial (6) ---
  else if (dist_selector == 6) {
    for (int i = 0; i < distr_dim; i++) {
      double a = p1[i];
      if (a <= 0) a = 1e-6; 
      
      // We only calculate bin-specific 'rewards'.
      // This is mathematically equivalent to a Multinomial 
      // with an implicit uniform prior on the total sum N.
      *g_pdf += lgammafn(g_stats[i] + a) - lgammafn(g_stats[i] + 1.0) - lgammafn(a);
      *gp_pdf += lgammafn(gp_stats[i] + a) - lgammafn(gp_stats[i] + 1.0) - lgammafn(a);
    }
  }
  
  // --- Gamma Distribution (7) ---
  else if (dist_selector == 7) {
    for (int i = 0; i < distr_dim; i++) {
      double alpha = p1[i]; // Shape (alpha)
      double beta = p2[i];   // Rate (beta)
      
      // Handling the support [0, inf)
      // If x is 0 and alpha < 1, log(x) is -inf, but (alpha-1) is negative, 
      // leading to +inf (the singularity).
      // If x is 0 and alpha > 1, it leads to -inf (correctly rejecting 0).
      
      if (g_stats[i] > 0) {
        *g_pdf += (alpha - 1.0) * log(g_stats[i]) - (beta * g_stats[i]);
      } else if (g_stats[i] == 0) {
        if (alpha < 1.0) *g_pdf += 1e10;  // Approximation of the singularity
        else if (alpha > 1.0) *g_pdf += -1e10; // Probability is 0
        else *g_pdf += 0.0; // Alpha = 1 (Exponential), kernel is e^0 = 1, log(1)=0
      } else {
        *g_pdf += -1e10; // Reject negative values (outside support)
      }
      
      if (gp_stats[i] > 0) {
        *gp_pdf += (alpha - 1.0) * log(gp_stats[i]) - (beta * gp_stats[i]);
      } else if (gp_stats[i] == 0) {
        if (alpha < 1.0) *gp_pdf += 1e10;
        else if (alpha > 1.0) *gp_pdf += -1e10;
        else *gp_pdf += 0.0;
      } else {
        *gp_pdf += -1e10;
      }
    }
  }
  
  // --- NP (99) ---
  else if (dist_selector == 99) {
    for (int i = 0; i < distr_dim; i++) {
      *g_pdf += log(p1[(int)g_stats[i]]);
      *gp_pdf += log(p1[(int)gp_stats[i]]);
    }
  }
  
}

