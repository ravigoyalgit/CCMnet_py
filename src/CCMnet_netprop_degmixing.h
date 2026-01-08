#ifndef CCMNET_NETPROP_DEGMIXING_H
#define CCMNET_NETPROP_DEGMIXING_H

#include "model.h"
#include "edgetree.h"
#include "MHproposal.h"
#include "MCMC.h"

void calc_prob_degmixing(Model *m, double *networkstatistics, double *meanvalues, 
                         double *varvalues, double *pdf_gaussian_nwp, 
                         double *pdf_gaussian_MHp);

#endif