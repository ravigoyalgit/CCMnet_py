#include "CCMnet_netprop_edges.h"

void calc_f_edges(int nwp_nedges, int MHp_nedges, int total_max_edges,
                  double *networkstatistics, double *prob_g_g2, double *prob_g2_g) {
  if (nwp_nedges > MHp_nedges) {
    *prob_g_g2 = networkstatistics[0];
  } else {
    *prob_g_g2 = total_max_edges - networkstatistics[0];
  }

  if (nwp_nedges < MHp_nedges) {
    *prob_g2_g = (double)MHp_nedges;
  } else {
    *prob_g2_g = (double)(total_max_edges - MHp_nedges);
  }
}

void calc_probs_edges(int MHp_nedges, int total_max_edges, int *prob_type,
                      double *networkstatistics, double *meanvalues, double *varvalues,
                      double *pdf_gaussian_nwp, double *pdf_gaussian_MHp) {

  double nwp_density = networkstatistics[0]; // / (double)total_max_edges;
  double MHp_density = (double)MHp_nedges; // / (double)total_max_edges;

  if (prob_type[0] == 0 && prob_type[1] == 0 && prob_type[2] == 0 && prob_type[3] == 0) {
    if (prob_type[4] == 1) { // Normal
      *pdf_gaussian_nwp = -0.5 * pow((nwp_density - meanvalues[0]), 2.0) / varvalues[0];
      *pdf_gaussian_MHp = -0.5 * pow((MHp_density - meanvalues[0]), 2.0) / varvalues[0];
    }
    else if (prob_type[4] == 2) { // Log Normal
      *pdf_gaussian_nwp = -0.5 * pow((log(nwp_density) - meanvalues[0]), 2.0) / varvalues[0];
      *pdf_gaussian_MHp = -0.5 * pow((log(MHp_density) - meanvalues[0]), 2.0) / varvalues[0];
    }
    else if (prob_type[4] == 3) { // Poisson
      if (nwp_density > MHp_density) {
        *pdf_gaussian_nwp = log(meanvalues[0]) - log(nwp_density);
        *pdf_gaussian_MHp = 0;
      } else {
        *pdf_gaussian_nwp = 0;
        *pdf_gaussian_MHp = log(meanvalues[0]) - log(MHp_density);
      }
    }
    else if (prob_type[4] == 4) { // Uniform
      *pdf_gaussian_nwp = 0;
      *pdf_gaussian_MHp = 0;
    }
    else if (prob_type[4] == 99) { // Non-parametric
      *pdf_gaussian_nwp = log(meanvalues[(int)networkstatistics[0]]);
      *pdf_gaussian_MHp = log(meanvalues[(int)MHp_nedges]);
    }
    else if (prob_type[4] == 11) { // Density - Normal
      
      nwp_density = nwp_density / (double)total_max_edges;
      MHp_density = MHp_density / (double)total_max_edges;
      
      *pdf_gaussian_nwp = -0.5 * pow((nwp_density - meanvalues[0]), 2.0) / varvalues[0];
      *pdf_gaussian_MHp = -0.5 * pow((MHp_density - meanvalues[0]), 2.0) / varvalues[0];
    }
    else if (prob_type[4] == 12) { // Density - Beta
      
      nwp_density = nwp_density / (double)total_max_edges;
      MHp_density = MHp_density / (double)total_max_edges;
      
      *pdf_gaussian_nwp = (meanvalues[0]-1)*log(nwp_density) + (meanvalues[1]-1)*log(1-nwp_density);
      *pdf_gaussian_MHp = (meanvalues[0]-1)*log(MHp_density) + (meanvalues[1]-1)*log(1-MHp_density);
    }
  }
}
