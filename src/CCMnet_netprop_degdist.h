#ifndef CCMNET_NETPROP_DEGDIST_H
#define CCMNET_NETPROP_DEGDIST_H

#include "model.h"
#include "edgetree.h"

void calc_stat_degdist(Model *m, 
                       int *NetworkForecast, 
                       int *num_deg_stats,
                       int *Deg_Add, 
                       int *Deg_Delete, 
                       int *Proposal_prob_zero);

// Existing declarations (referenced in your snippet)
void calc_f_degdist(int num_deg_stats, Model *m, double *networkstatistics, 
                    int nwp_nedges, int MHp_nedges, int *Deg_Delete, int *Deg_Add, 
                    double *prob_g_g2, double *prob_g2_g, 
                    int *nwp_Deg_Distr, int *MHp_Deg_Distr);


#endif