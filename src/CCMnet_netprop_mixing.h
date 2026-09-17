#ifndef CCMNET_NETPROP_MIXING_H
#define CCMNET_NETPROP_MIXING_H

#include "model.h"
#include "edgetree.h"
#include "MHproposal.h"

/* Step 1: Identify covariate types and build mixing matrices */
void calc_stat_mixing(Network *nwp, Model *m, MHproposal *MHp, double *networkstatistics, 
                      int *Cov_types, int *Num_Cov_type, 
                      double *nwp_mixing_matrix, double *MHp_mixing_matrix);

/* Step 2: Calculate transition probabilities using the stats from Step 1 */
void calc_f_mixing(Network *nwp, int *Cov_types, int *Num_Cov_type, 
                   double *nwp_mixing_matrix, double *MHp_mixing_matrix,
                   double *prob_g_g2, double *prob_g2_g, int MHp_nedges,
                   Model *m, MHproposal *MHp, double *networkstatistics);

#endif