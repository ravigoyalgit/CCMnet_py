#ifndef CCMNET_NETPROP_EDGES_H
#define CCMNET_NETPROP_EDGES_H

#include <math.h>

/* Function Prototypes */

void calc_f_edges(int nwp_nedges, int MHp_nedges, int total_max_edges,
                  double *networkstatistics, double *prob_g_g2, double *prob_g2_g);

#endif
