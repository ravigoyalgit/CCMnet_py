#ifndef CCMNET_NETPROP_MIXING_DEGDIST_H
#define CCMNET_NETPROP_MIXING_DEGDIST_H

#include "model.h"
#include "edgetree.h"

/* No need for typedefs here if they are in model.h */

void calc_probs_mixing_degdist(int length_deg_dist, Model *m, Network *nwp,
                               ModelTerm *mtp2, int *prob_type,
                               double *networkstatistics, double *meanvalues,
                               double *varvalues, double *nwp_mixing_matrix,
                               double *MHp_mixing_matrix, int MHp_nedges,
                               double *pdf_gaussian_nwp, double *pdf_gaussian_MHp);

#endif
