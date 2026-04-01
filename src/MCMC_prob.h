/*
 *  File CCMnet/src/MCMC_prob.h
 *
 *  Sections of this code are derived from the ergm package
 *  All such sections are noted and attributed to the statnet development team.
 */

#include "CCMnet_netprop_edges.h"
#include "CCMnet_netprop_mixing.h"
#include "CCMnet_netprop_degdist.h"
#include "CCMnet_netprop_mixing_degdist.h"
#include "CCMnet_netprop_degmixing.h"
#include "CCMnet_netprop_degmixing_clustering.h"
#include "CCMnet_netprop_prob_dist.h"

MCMCStatus MetropolisHastings(MHproposal *MHp,
                              double *theta, double *statistics,
                              int nsteps, int *staken,
                              int fVerbose,
                              Network *nwp, Model *m,
                              int *prob_type,
                              int *maxdegree,
                              double *meanvalues,
                              double *varvalues,
                              int *BayesInference,
                              Network *TransNW,
                              double *Ia,
                              double *Il,
                              double *R_times,
                              double *beta_a,
                              double *beta_l,
                              int *NetworkForecast,
                              double *evolutionrate,
                              double *evolutionvar);

