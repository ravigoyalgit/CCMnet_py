
#ifndef CCMNET_NETPROP_PROB_DIST_H
#define CCMNET_NETPROP_PROB_DIST_H

#include "model.h"
#include "edgetree.h"
#include "MHproposal.h"
#include "MCMC.h"

double quadratic_form(double *V, double *mu, double *inv_sigma, int dim);

void calc_prob_dist(double *g_stats, double *gp_stats, int distr_dim, int *prob_type,
                    double *p1, double *p2,
                    double *g_pdf, double *gp_pdf,
                    int fVerbose);
#endif