#ifndef CCMNET_NETPROP_DEGMIXING_CLUSTERING_H
#define CCMNET_NETPROP_DEGMIXING_CLUSTERING_H

#include "model.h"
#include "edgetree.h"
#include "MHproposal.h"
#include "MCMC.h"

void calc_f_degmixing_clustering(int num_deg_stats, Network *nwp, Model *m,
                                 double *networkstatistics,
                                 int *deg_dist_nwp, int *deg_dist_MHp,
                                 int *g_flat, int *g2_flat,
                                 int *Deg_nwp, int *Deg_MHp, int MHp_nedges,
                                 double *prob_g_g2, double *prob_g2_g);

#endif