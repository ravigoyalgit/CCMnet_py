#include <R.h>
#include <math.h>
#include <stdlib.h>
#include "CCMnet_netprop_degmixing_clustering.h"

void calc_f_degmixing_clustering(int num_deg_stats, Network *nwp, Model *m,
                                 double *networkstatistics,
                                 int *deg_dist_nwp, int *deg_dist_MHp,
                                 int *g_flat, int *g2_flat,
                                 int *Deg_nwp, int *Deg_MHp, int MHp_nedges,
                                 double *prob_g_g2, double *prob_g2_g) {
  
  int counter, index1, index2;
  int n_dim = num_deg_stats - 1;
  
  // Local 2D arrays to match your g_dmm logic
  int g_dmm[n_dim][n_dim];
  int g2_dmm[n_dim][n_dim];
  for (int i = 0; i < n_dim; i++) {
    for (int j = 0; j < n_dim; j++) {
      g_dmm[i][j] = g_flat[i * n_dim + j];
      g2_dmm[i][j] = g2_flat[i * n_dim + j];
    }
  }
  
  double num_Tri = networkstatistics[m->n_stats - 1];
  double num_Tri_change = fabs(m->workspace[m->n_stats - 1]);
  
  /// g->g2
  if (nwp->nedges < MHp_nedges) { // edge was added
    double pa = 0;
    double pa_num = -3 * num_Tri;
    for (counter = 2; counter < (num_deg_stats); counter++) {
      pa_num += calcCNR(counter, 2) * deg_dist_nwp[counter];
    }
    
    double pa_dem = 0.0;
    for (index1 = 1; index1 < (num_deg_stats); index1++) {
      for (index2 = index1; index2 < (num_deg_stats); index2++) {
        if (index1 == index2) {
          pa_dem += index1 * index2 * (deg_dist_nwp[index1] * (deg_dist_nwp[index2] - 1) * .5 - g_dmm[index1 - 1][index2 - 1]);
        } else {
          pa_dem += index1 * index2 * (deg_dist_nwp[index1] * deg_dist_nwp[index2] - g_dmm[index1 - 1][index2 - 1]);
        }
      }
    }
    
    pa = pa_num / pa_dem;
    
    double p_add_1 = 1.0;
    double frac_k = 1.0;
    double p_add = 0;
    
    for (counter = 0; counter < (num_Tri_change); counter++) {
      frac_k = frac_k * (counter + 1);
      p_add_1 = p_add_1 * (Deg_nwp[0] - counter) * (Deg_nwp[1] - counter);
    }
    p_add = (1.0 / frac_k) * p_add_1 * pow(pa, num_Tri_change) * pow((1 - pa), (Deg_nwp[0] - num_Tri_change) * (Deg_nwp[1] - num_Tri_change));
    
    *prob_g_g2 = *prob_g_g2 * p_add;
    
  } else { // edge was removed
    double pb = 0;
    int pb_num = 3 * num_Tri;
    
    double pb_dem = 0.0;
    for (index1 = 1; index1 < (num_deg_stats); index1++) {
      for (index2 = index1; index2 < (num_deg_stats); index2++) {
        pb_dem += (index1 - 1) * (index2 - 1) * (g_dmm[index1 - 1][index2 - 1]);
      }
    }
    
    pb = pb_num / pb_dem;
    
    double p_remove_1 = 1.0;
    double frac_k = 1.0;
    double p_remove = 0;
    
    for (counter = 0; counter < (num_Tri_change); counter++) {
      frac_k = frac_k * (counter + 1);
      p_remove_1 = p_remove_1 * (Deg_nwp[0] - 1 - counter) * (Deg_nwp[1] - 1 - counter);
    }
    p_remove = frac_k * p_remove_1 * pow(pb, num_Tri_change) * pow((1 - pb), (Deg_nwp[0] - 1 - num_Tri_change) * (Deg_nwp[1] - 1 - num_Tri_change));
    
    *prob_g_g2 = *prob_g_g2 * p_remove;
  }
  
  /// g2->g
  num_Tri = networkstatistics[m->n_stats - 1] + m->workspace[m->n_stats - 1];
  
  if (nwp->nedges > MHp_nedges) { // edge was added from g2 to g
    double pa = 0;
    double pa_num = -3 * num_Tri;
    for (counter = 2; counter < (num_deg_stats); counter++) {
      pa_num += calcCNR(counter, 2) * deg_dist_MHp[counter];
    }
    
    double pa_dem = 0.0;
    for (index1 = 1; index1 < (num_deg_stats); index1++) {
      for (index2 = index1; index2 < (num_deg_stats); index2++) {
        if (index1 == index2) {
          pa_dem += index1 * index2 * (deg_dist_MHp[index1] * (deg_dist_MHp[index2] - 1) * .5 - g2_dmm[index1 - 1][index2 - 1]);
        } else {
          pa_dem += index1 * index2 * (deg_dist_MHp[index1] * deg_dist_MHp[index2] - g2_dmm[index1 - 1][index2 - 1]);
        }
      }
    }
    
    pa = pa_num / pa_dem;
    
    double p_add_1 = 1.0;
    double frac_k = 1.0;
    double p_add = 0;
    
    for (counter = 0; counter < (num_Tri_change); counter++) {
      frac_k = frac_k * (counter + 1);
      p_add_1 = p_add_1 * (Deg_MHp[0] - counter) * (Deg_MHp[1] - counter);
    }
    p_add = frac_k * p_add_1 * pow(pa, num_Tri_change) * pow((1 - pa), (Deg_MHp[0] - num_Tri_change) * (Deg_MHp[1] - num_Tri_change));
    *prob_g2_g = *prob_g2_g * p_add;
    
  } else { // edge was removed
    double pb = 0;
    int pb_num = 3 * num_Tri;
    
    double pb_dem = 0.0;
    for (index1 = 1; index1 < (num_deg_stats); index1++) {
      for (index2 = index1; index2 < (num_deg_stats); index2++) {
        pb_dem += (index1 - 1) * (index2 - 1) * (g2_dmm[index1 - 1][index2 - 1]);
      }
    }
    
    pb = pb_num / pb_dem;
    
    double p_remove_1 = 1.0;
    double frac_k = 1.0;
    double p_remove = 0;
    
    for (counter = 0; counter < (num_Tri_change); counter++) {
      frac_k = frac_k * (counter + 1);
      p_remove_1 = p_remove_1 * (Deg_MHp[0] - 1 - counter) * (Deg_MHp[1] - 1 - counter);
    }
    p_remove = (1.0 / frac_k) * p_remove_1 * pow(pb, num_Tri_change) * pow((1 - pb), (Deg_MHp[0] - 1 - num_Tri_change) * (Deg_MHp[1] - 1 - num_Tri_change));
    
    *prob_g2_g = *prob_g2_g * p_remove;
  }
}