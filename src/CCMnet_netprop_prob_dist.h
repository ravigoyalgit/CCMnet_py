
#ifndef CCMNET_NETPROP_PROB_DIST_H
#define CCMNET_NETPROP_PROB_DIST_H

#include "model.h"
#include "edgetree.h"
#include "MHproposal.h"
#include "MCMC.h"

double quadratic_form(double *V, double *mu, double *inv_sigma, int dim);

void calc_prob_dist(double *v_old, double *v_new, int dim, int *prob_type,
                    double *meanvalues, double *varvalues,
                    double *pdf_old, double *pdf_new);

#endif