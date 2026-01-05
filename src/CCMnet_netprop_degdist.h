#ifndef CCMNET_NETPROP_DEGDIST_H
#define CCMNET_NETPROP_DEGDIST_H

#include <math.h>
#include <R.h> // For Rprintf and log
#include "model.h"

void calc_f_degdist(int num_deg_stats, Model *m, double *networkstatistics,
                    int nwp_nedges, int MHp_nedges, int *Deg_Delete, int *Deg_Add,
                    double *prob_g_g2, double *prob_g2_g,
                    int *nwp_Deg_Distr, int *MHp_Deg_Distr);

void calc_probs_degdist(int num_deg_stats, Network *nwp, int *prob_type,
                        double *meanvalues, double *varvalues,
                        int *nwp_Deg_Distr, int *MHp_Deg_Distr,
                        double *pdf_gaussian_nwp, double *pdf_gaussian_MHp);
#endif
