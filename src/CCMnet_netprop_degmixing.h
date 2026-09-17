#ifndef CCMNET_NETPROP_DEGMIXING_H
#define CCMNET_NETPROP_DEGMIXING_H

#include "model.h"
#include "edgetree.h"
#include "MHproposal.h"
#include "MCMC.h"

void get_properties(Model *m, double *stats, int num_deg_stats, 
                    int g_dmm[num_deg_stats-1][num_deg_stats-1], 
                                              int g2_dmm[num_deg_stats-1][num_deg_stats-1]); 
  
void calculate_congruence_ratios(
      Network *nwp,
      MHproposal *MHp,
      int num_deg_stats,
      int g_dmm[][num_deg_stats-1], 
      int g2_dmm[][num_deg_stats-1],
      double *prob_g_g2,
      double *prob_g2_g,
      int MHp_nedges,
      double *pdf_gaussian_nwp,
      double *pdf_gaussian_MHp,
      int *Proposal_prob_zero,
      int deg_dist_nwp[num_deg_stats],
      int deg_dist_MHp[num_deg_stats],
      int Deg_nwp[2],
      int Deg_MHp[2]
);

#endif