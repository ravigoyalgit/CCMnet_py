#ifndef CCMNET_NETPROP_MIXING_DEGDIST_H
#define CCMNET_NETPROP_MIXING_DEGDIST_H

#include "model.h"
#include "edgetree.h"
#include "MHproposal.h"
#include "MCMC.h"

/* No need for typedefs here if they are in model.h */

void calculate_congruence_ratio_degmix(
    Model *m, Network *nwp, MHproposal *MHp,
    int length_deg_dist, int MHp_nedges,
    int *Deg_nwp, int *Deg_MHp, int *Cov_types,
    double *networkstatistics,
    int *nwp_Deg_Distr_1, int *nwp_Deg_Distr_2, int *MHp_Deg_Distr_1, int *MHp_Deg_Distr_2,
    double *nwp_mixing_matrix, double *MHp_mixing_matrix,
    double *prob_g_g2, double *prob_g2_g,
    double *pdf_gaussian_nwp, double *pdf_gaussian_MHp, int *Proposal_prob_zero
);

void calc_probs_mixing_degdist(int length_deg_dist, Model *m, Network *nwp,
                               ModelTerm *mtp2, int *prob_type,
                               double *networkstatistics, double *meanvalues,
                               double *varvalues, double *nwp_mixing_matrix,
                               double *MHp_mixing_matrix, int MHp_nedges,
                               double *pdf_gaussian_nwp, double *pdf_gaussian_MHp);


#endif
