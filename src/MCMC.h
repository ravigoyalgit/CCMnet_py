/*
 *  File ergm/src/MCMC.h
 *  Part of the statnet package, http://statnet.org
 *
 *  This software is distributed under the GPL-3 license.  It is free,
 *  open source, and has the attribution requirements (GPL Section 7) in
 *    http://statnet.org/attribution
 *
 *  Copyright 2012 the statnet development team
 */
#ifndef MCMC_H
#define MCMC_H

#include "edgetree.h"
#include "changestat.h"
#include "MHproposal.h"
#include "model.h"

// TODO: This might be worth moving into a common "constants.h".

typedef enum MCMCStatus_enum {
  MCMC_OK = 0,
  MCMC_TOO_MANY_EDGES = 1,
  MCMC_MH_FAILED = 2
} MCMCStatus;

/* MOD ADD Ravi */
 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

/* *** don't forget tail-> head, so this function accepts tails first, not heads  */

void MCMC_wrapper(int *dnumnets, int *dnedges,
		  int *tails, int *heads,
		  int *dn, int *dflag, int *bipartite,
		  int *nterms, char **funnames,
		  char **sonames,
		  char **MHproposaltype, char **MHproposalpackage,
		  double *inputs, double *theta0, int *samplesize,
		  double *sample, int *burnin, int *interval,
		  int *newnetworktails,
		  int *newnetworkheads,
		  int *fVerbose,
		  int *attribs, int *maxout, int *maxin, int *minout,
		  int *minin, int *condAllDegExact, int *attriblength,
		  int *maxedges,
		  int *status,
                  int *prob_type,   /*MOD ADDED RAVI###########*/
                  int *maxdegree,
                  double *meanvalues,
                  double *varvalues,
                  int *BayesInference,
                  int *Trans_nedges,
                  int *Trans_networktails,
                  int *Trans_networkheads,
                  double *Ia,
                  double *Il,
                  double *R_times,
                  double *beta_a,
                  double *beta_l,
                  int *NetworkForecast,
                  double *evolutionrate,
                  double *evolutionvar
        );

MCMCStatus MCMCSample(MHproposal *MHp,
		      double *theta, double *networkstatistics,
		      int samplesize, int burnin,
		      int interval, int fVerbose, int nmax,
		      Network *nwp, Model *m,
                  int *prob_type,   /*MOD ADDED RAVI###########*/
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
                  double *evolutionvar
);


#endif

double sp_gamma(double z);
int calcCNR( int n, int r);
