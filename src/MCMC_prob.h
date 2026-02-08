/*
 *  File CCMnet/src/MCMC_prob.h
 *
 *  Sections of this code are derived from the ergm library
 *  All such sections are noted and attributed to the statnet development team.
 */

#include "CCMnet_netprop_edges.h"
#include "CCMnet_netprop_mixing.h"
#include "CCMnet_netprop_degdist.h"
#include "CCMnet_netprop_mixing_degdist.h"
#include "CCMnet_netprop_degmixing.h"
#include "CCMnet_netprop_degmixing_clustering.h"

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

MCMCStatus MetropolisHastings(MHproposal *MHp,
                              double *theta, double *networkstatistics,
                              int nsteps, int *staken,
                              int fVerbose,
                              Network *nwp,
                              Model *m,
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
                              double *evolutionvar) {
  
  int print_info_MH = 0;
  
  if (print_info_MH == 1) {
    Rprintf("Entered: MH Code \n");
  }
  
  
  /* CODE FROM ERGM Library
   *  File ergm/src/MCMC.c
   *  Part of the statnet package, http://statnet.org
   *
   *  This software is distributed under the GPL-3 license.  It is free,
   *  open source, and has the attribution requirements (GPL Section 7) in
   *    http://statnet.org/attribution
   *
   *  Copyright 2012 the statnet development team
   */
  unsigned int taken=0, unsuccessful=0;
  /*  if (fVerbose)
   Rprintf("Now proposing %d MH steps... ", nsteps); */
  for(unsigned int step=0; step < nsteps; step++) {
    MHp->logratio = 0;
    (*(MHp->func))(MHp, nwp); /* Call MH function to propose toggles */
  if(MHp->toggletail[0]==MH_FAILED){
    if(MHp->togglehead[0]==MH_UNRECOVERABLE)
      error("Something very bad happened during proposal. Memory has not been deallocated, so restart R soon.");
    if(MHp->togglehead[0]==MH_IMPOSSIBLE){
      Rprintf("MH Proposal function encountered a configuration from which no toggle(s) can be proposed.\n");
      return MCMC_MH_FAILED;
    }
    if(MHp->togglehead[0]==MH_UNSUCCESSFUL){
      warning("MH Proposal function failed to find a valid proposal.");
      unsuccessful++;
      if(unsuccessful>taken*MH_QUIT_UNSUCCESSFUL){
        Rprintf("Too many MH Proposal function failures.\n");
        return MCMC_MH_FAILED;
      }
      continue;
    }
  }
  //END of CODE FROM ERGM Library
  
  // if (print_info_MH == 1) {
  //   Rprintf("MH: Before ChangeStats Code \n");
  //   Rprintf("nwp info: %d \n", nwp->nnodes);
  //   Rprintf("WorkSpace: ");
  //   for (int counter_print=0; counter_print < ((m->n_stats)-1); counter_print++){
  //     Rprintf(" %f ",m->workspace[counter_print]);
  //   }
  //   Rprintf("\n");
  //   Rprintf("Toggle Info: %d %d %d \n",MHp->ntoggles, *(MHp->toggletail), *(MHp->togglehead));
  // }
  
  /* Calculate change statistics,
   remembering that tail -> head */
  ChangeStats(MHp->ntoggles, MHp->toggletail, MHp->togglehead, nwp, m);
  
  if (print_info_MH == 1) {
    Rprintf("MH: After ChangeStats Code \n");
    Rprintf("\n");
    Rprintf("Node ID 1 %d Node ID 2 %d\n",*(MHp->toggletail), *(MHp->togglehead));
    Rprintf("\n");
    Rprintf("WorkSpace: ");
    for (int counter_print=0; counter_print < ((m->n_stats)); counter_print++){
      Rprintf(" %f ",m->workspace[counter_print]);
    }
    Rprintf("\n");
    Rprintf("Begin - Network Statistic: ");
    for (unsigned int i = 0; i < m->n_stats; i++){
      Rprintf(" %f ",networkstatistics[i]);
    }
    Rprintf("\n");
  }
  
  /* MOD ADDED */
  
  int MHp_nedges;
  double prob_g2_g = 1;
  double prob_g_g2 = 1;
  int total_max_edges = (nwp->nnodes * (nwp->nnodes-1) * .5) + .5;
  double cutoff=log(0);
  int counter;
  int counter1;
  int counter2;
  
  double nwp_density;
  double MHp_density;
  double pdf_gaussian_nwp = 0;
  double pdf_gaussian_MHp = log(0);
  
  MHp_nedges = networkstatistics[0] + m->workspace[0];
  
  
  //Rprintf("Before ergm vs GUF: MH Code \n");
  if ((theta[0] < -999) && (theta[0] > -1000)) {
    //Rprintf("Entered GUF: MH Code \n");
    
    ///EDGES: BEGIN///
    if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
      calc_f_edges(nwp->nedges, MHp_nedges, total_max_edges, networkstatistics, &prob_g_g2, &prob_g2_g);
      calc_probs_edges(MHp_nedges, total_max_edges, prob_type, networkstatistics, meanvalues, varvalues, &pdf_gaussian_nwp, &pdf_gaussian_MHp);
    }
    ///EDGES: END ///
    
    ///MIXING MATRIX: BEGIN///
    if ((prob_type[0] == 0) && (prob_type[1] >= 1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
      int L = m->n_stats; // Or however your C struct tracks total stats
      int num_params = L - 1; 
      int k = (int)((sqrt(8.0 * num_params + 1.0) - 1.0) / 2.0 + 0.1);
      
      // Now your DEBUG will finally show:
      //Rprintf("--- DEBUG: L=%d, num_params=%d, k=%d ---\n", L, num_params, k);
      
      // 2. Dynamic Allocation
      int *Cov_types = (int *)R_alloc(2, sizeof(int)); 
      int *Num_Cov_type = (int *)R_alloc(k, sizeof(int));
      double *nwp_mix = (double *)R_alloc(num_params, sizeof(double));
      double *MHp_mix = (double *)R_alloc(num_params, sizeof(double));
      
      // 1. Get Stats
      calc_stat_mixing(nwp, m, MHp, networkstatistics, Cov_types, Num_Cov_type, nwp_mix, MHp_mix);
      
      // 2. Get Probabilities
      int MHp_nedges = nwp->nedges + (int)m->workspace[0];
      calc_f_mixing(nwp, Cov_types, Num_Cov_type, nwp_mix, MHp_mix, &prob_g_g2, &prob_g2_g, MHp_nedges, m, MHp, networkstatistics);
      
      // 3. Get Gaussian Math
      calc_probs_mixing(nwp, 3, Cov_types, Num_Cov_type, nwp_mix, MHp_mix, meanvalues, varvalues, &pdf_gaussian_nwp, &pdf_gaussian_MHp, m, MHp, networkstatistics, prob_type);
    }
    ///MIXING MATRIX: END///
    
    /// DEGREE DISTRIBUTION: BEGIN ///
    if ((prob_type[0] >= 1) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)) {
      
      int num_deg_stats;
      int Deg_Add[2];
      int Deg_Delete[2];
      int Proposal_prob_zero;
      
      // Call the new stat function
      calc_stat_degdist(m, NetworkForecast, &num_deg_stats, Deg_Add, Deg_Delete, &Proposal_prob_zero);
      
      // Prepare arrays for distribution
      int MHp_Deg_Distr[num_deg_stats];
      int nwp_Deg_Distr[num_deg_stats];
      
      if (Proposal_prob_zero == 1) {
        prob_g2_g = 1;
        pdf_gaussian_MHp = log(0); // Resulting in -Inf
        prob_g_g2 = 1;
        pdf_gaussian_nwp = 0;
      } else {
        // Step 2: Probability calculation
        calc_f_degdist(num_deg_stats, m, networkstatistics, nwp->nedges, MHp_nedges,
                       Deg_Delete, Deg_Add, &prob_g_g2, &prob_g2_g,
                       nwp_Deg_Distr, MHp_Deg_Distr);
        
        // Step 3: Gaussian calculation
        calc_probs_degdist(num_deg_stats, nwp, prob_type, meanvalues, varvalues,
                           nwp_Deg_Distr, MHp_Deg_Distr, &pdf_gaussian_nwp, &pdf_gaussian_MHp);
      }
    }
    /// DEGREE DISTRIBUTION: END ///
    
    
    /// Two Degree Distributions and Mixing: BEGIN ///
    if ((prob_type[0] >= 1) && (prob_type[1] >= 1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
      
      //int num_deg_stats = m->n_stats-1-(*NetworkForecast) - 3; //The four is for number of mixing - only coded for two node types
      int num_deg_stats = m->n_stats-1 - 3; //The four is for number of mixing - only coded for two node types
      int length_deg_dist = num_deg_stats / 2; //Currently both degree distributions have to be the same size
      int Deg_nwp[2];
      int Deg_MHp[2];
      int Cov_types[2]; //Node types for the endpoints of toggled edge
      //               int Deg_Add_counter = 0;
      //               int Deg_Delete_counter = 0;
      int MHp_Deg_Distr_1[length_deg_dist]; // Degree Distribution and Edges for Node 1 and Node 2
      int MHp_Deg_Distr_2[length_deg_dist]; // Node 1 and Node 2 are the nodes of the toggled edge
      int nwp_Deg_Distr_1[length_deg_dist];
      int nwp_Deg_Distr_2[length_deg_dist];
      double MHp_Deg_Distr_Edges_1[length_deg_dist];
      double MHp_Deg_Distr_Edges_2[length_deg_dist];
      double nwp_Deg_Distr_Edges_1[length_deg_dist];
      double nwp_Deg_Distr_Edges_2[length_deg_dist];
      double nwp_mixing_matrix[3];
      double MHp_mixing_matrix[3];
      double nwp_prob_mixing[2];
      double MHp_prob_mixing[2];
      double nwp_exp_dmm;
      double MHp_exp_dmm;
      int Proposal_prob_zero = 0;
      
      /* Get Degree of tail and head*/
      
      Deg_nwp[0] = OUT_DEG[*(MHp->toggletail)] + IN_DEG[*(MHp->toggletail)];
      Deg_nwp[1] = OUT_DEG[*(MHp->togglehead)] + IN_DEG[*(MHp->togglehead)];
      
      //Rprintf("Node ID 1 %d Node ID 2 %d\n",*(MHp->toggletail), *(MHp->togglehead));
      //Rprintf("Degree 1 %d Degree 2 %d\n",Deg_nwp[0], Deg_nwp[1]);
      
      if (nwp->nedges > MHp_nedges) {
        Deg_MHp[0] = Deg_nwp[0] - 1;
        Deg_MHp[1] = Deg_nwp[1] - 1;
      } else {
        Deg_MHp[0] = Deg_nwp[0] + 1;
        Deg_MHp[1] = Deg_nwp[1] + 1;
      }
      
      if ((Deg_MHp[0] > (length_deg_dist-1)) || (Deg_MHp[1] > (length_deg_dist-1))) {
        Proposal_prob_zero = 1;
      }
      
      if (Proposal_prob_zero == 1) {
        prob_g2_g = 1;
        pdf_gaussian_MHp = log(0);
        prob_g_g2 = 1;
        pdf_gaussian_nwp = 0;
        //Rprintf("Proposal Excesses Max Edges: %f\n", pdf_gaussian_MHp);
      } else {
        /* Construct Degree Distribution and number of edges associated with each degree*/
        ModelTerm *mtp2 = m->termarray;
        mtp2++;
        
        Cov_types[0] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->toggletail) - 1]); //Minus 1 since node ids are from 1 to nnodes
        Cov_types[1] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->togglehead) - 1]);
        
        //nwp_mixing_matrix[0] = nwp->nedges - networkstatistics[2*length_deg_dist + 1] - networkstatistics[2*length_deg_dist + 2];
        //nwp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 1];
        //nwp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 2];
        
        nwp_mixing_matrix[0] = networkstatistics[2*length_deg_dist + 1];
        nwp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 2];
        nwp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 3];
        
        //MHp_mixing_matrix[0] = MHp_nedges - networkstatistics[2*length_deg_dist + 1] - networkstatistics[2*length_deg_dist + 2] - m->workspace[2*length_deg_dist + 1] - m->workspace[2*length_deg_dist + 2];
        //MHp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 1] + m->workspace[2*length_deg_dist + 1];
        //MHp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 2] + m->workspace[2*length_deg_dist + 2];
        
        MHp_mixing_matrix[0] = networkstatistics[2*length_deg_dist + 1] + m->workspace[2*length_deg_dist + 1];
        MHp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 2] + m->workspace[2*length_deg_dist + 2];
        MHp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 3] + m->workspace[2*length_deg_dist + 3];
        
        //Rprintf("nwp_mixing_matrix %f %f %f\n", nwp_mixing_matrix[0], nwp_mixing_matrix[1], nwp_mixing_matrix[2]);
        //Rprintf("MHp_mixing_matrix %f %f %f\n", MHp_mixing_matrix[0], MHp_mixing_matrix[1], MHp_mixing_matrix[2]);
        
        if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
          nwp_prob_mixing[0] = (float)(2*nwp_mixing_matrix[0]) / (float)(2*nwp_mixing_matrix[0] + nwp_mixing_matrix[1]);
          nwp_prob_mixing[1] = nwp_prob_mixing[0];
          MHp_prob_mixing[0] = (float)(2*MHp_mixing_matrix[0]) / (float)(2*MHp_mixing_matrix[0] + MHp_mixing_matrix[1]);
          MHp_prob_mixing[1] = MHp_prob_mixing[0];
        } else if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
          nwp_prob_mixing[0] = (float)(2*nwp_mixing_matrix[2]) / (float)(2*nwp_mixing_matrix[2] + nwp_mixing_matrix[1]);
          nwp_prob_mixing[1] = nwp_prob_mixing[0];
          MHp_prob_mixing[0] = (float)(2*MHp_mixing_matrix[2]) / (float)(2*MHp_mixing_matrix[2] + MHp_mixing_matrix[1]);
          MHp_prob_mixing[1] = MHp_prob_mixing[0];
        } else {
          nwp_prob_mixing[0] = (float)(nwp_mixing_matrix[1]) / (float)(2*nwp_mixing_matrix[0] + nwp_mixing_matrix[1]);
          nwp_prob_mixing[1] = (float)(nwp_mixing_matrix[1]) / (float)(2*nwp_mixing_matrix[2] + nwp_mixing_matrix[1]);
          MHp_prob_mixing[0] = (float)(MHp_mixing_matrix[1]) / (float)(2*MHp_mixing_matrix[0] + MHp_mixing_matrix[1]);
          MHp_prob_mixing[1] = (float)(MHp_mixing_matrix[1]) / (float)(2*MHp_mixing_matrix[2] + MHp_mixing_matrix[1]);
        }
        
        if (Cov_types[0] == 1) {
          for (counter =1; counter < (length_deg_dist+1); counter++) {
            MHp_Deg_Distr_1[counter-1] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
            nwp_Deg_Distr_1[counter-1] = (int)networkstatistics[counter];
            MHp_Deg_Distr_Edges_1[counter-1] = MHp_Deg_Distr_1[counter-1] * (counter-1) * MHp_prob_mixing[0];
            nwp_Deg_Distr_Edges_1[counter-1] = nwp_Deg_Distr_1[counter-1] * (counter-1) * nwp_prob_mixing[0];
          }
        } else {
          for (counter =(length_deg_dist+1); counter < (2*length_deg_dist + 1); counter++) {
            MHp_Deg_Distr_1[counter-(length_deg_dist+1)] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
            nwp_Deg_Distr_1[counter-(length_deg_dist+1)] = (int)networkstatistics[counter];
            MHp_Deg_Distr_Edges_1[counter-(length_deg_dist+1)] = MHp_Deg_Distr_1[counter-(length_deg_dist+1)] * (counter-(length_deg_dist+1)) * MHp_prob_mixing[0];
            nwp_Deg_Distr_Edges_1[counter-(length_deg_dist+1)] = nwp_Deg_Distr_1[counter-(length_deg_dist+1)] * (counter-(length_deg_dist+1)) * nwp_prob_mixing[0];
          }
        }
        
        if (Cov_types[1] == 1) {
          for (counter =1; counter < (length_deg_dist+1); counter++) {
            MHp_Deg_Distr_2[counter-1] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
            nwp_Deg_Distr_2[counter-1] = (int)networkstatistics[counter];
            MHp_Deg_Distr_Edges_2[counter-1] = MHp_Deg_Distr_2[counter-1] * (counter-1) * MHp_prob_mixing[1];
            nwp_Deg_Distr_Edges_2[counter-1] = nwp_Deg_Distr_2[counter-1] * (counter-1) * nwp_prob_mixing[1];
          }
        } else {
          for (counter =(length_deg_dist+1); counter < (2*length_deg_dist + 1); counter++) {
            MHp_Deg_Distr_2[counter-(length_deg_dist+1)] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
            nwp_Deg_Distr_2[counter-(length_deg_dist+1)] = (int)networkstatistics[counter];
            MHp_Deg_Distr_Edges_2[counter-(length_deg_dist+1)] = MHp_Deg_Distr_2[counter-(length_deg_dist+1)] * (counter-(length_deg_dist+1)) * MHp_prob_mixing[1];
            nwp_Deg_Distr_Edges_2[counter-(length_deg_dist+1)] = nwp_Deg_Distr_2[counter-(length_deg_dist+1)] * (counter-(length_deg_dist+1)) * nwp_prob_mixing[1];
          }
        }
        
        //Rprintf("nwp_Deg_Distr_1 %d %d %d %d\n", nwp_Deg_Distr_1[0], nwp_Deg_Distr_1[1], nwp_Deg_Distr_1[2], nwp_Deg_Distr_1[3]);
        //Rprintf("nwp_Deg_Distr_2 %d %d %d %d\n", nwp_Deg_Distr_2[0], nwp_Deg_Distr_2[1], nwp_Deg_Distr_2[2], nwp_Deg_Distr_2[3]);
        //Rprintf("MHp_Deg_Distr_1 %d %d %d %d\n", MHp_Deg_Distr_1[0], MHp_Deg_Distr_1[1], MHp_Deg_Distr_1[2], MHp_Deg_Distr_1[3]);
        //Rprintf("MHp_Deg_Distr_2 %d %d %d %d\n", MHp_Deg_Distr_2[0], MHp_Deg_Distr_2[1], MHp_Deg_Distr_2[2], MHp_Deg_Distr_2[3]);        
        
        /* Now we have the margins for the the expected degree mixing matix*/
        
        /* Construct expected degree distribution */
        double nwp_dmm_norm;
        double MHp_dmm_norm;
        
        if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
          nwp_dmm_norm = (2*nwp_mixing_matrix[0]);
          MHp_dmm_norm = (2*MHp_mixing_matrix[0]);
        } else if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
          nwp_dmm_norm = (2*nwp_mixing_matrix[2]);
          MHp_dmm_norm = (2*MHp_mixing_matrix[2]);
        } else {
          nwp_dmm_norm = (nwp_mixing_matrix[1]);
          MHp_dmm_norm = (MHp_mixing_matrix[1]);
        }
        
        
        nwp_exp_dmm = (nwp_Deg_Distr_Edges_1[Deg_nwp[0]] * nwp_Deg_Distr_Edges_2[Deg_nwp[1]])/ (float)nwp_dmm_norm;
        if ((Deg_nwp[0] == Deg_nwp[1]) && (Cov_types[0] == Cov_types[1])){
          nwp_exp_dmm = nwp_exp_dmm * .5;
        }
        MHp_exp_dmm = (MHp_Deg_Distr_Edges_1[Deg_MHp[0]] * MHp_Deg_Distr_Edges_2[Deg_MHp[1]])/ (float)MHp_dmm_norm;
        if ((Deg_MHp[0] == Deg_MHp[1]) && (Cov_types[0] == Cov_types[1])){
          MHp_exp_dmm = MHp_exp_dmm * .5;
        }
        
        
        if (nwp->nedges > MHp_nedges) {  //Edge Removed nwp -> MHp
          prob_g_g2 = nwp_exp_dmm;
        } else {  //Edge Added nwp -> MHp
          if ((Deg_nwp[0] == Deg_nwp[1])  && (Cov_types[0] == Cov_types[1])) {
            prob_g_g2 = (nwp_Deg_Distr_1[Deg_nwp[0]]* (nwp_Deg_Distr_2[Deg_nwp[1]]-1)*.5) - nwp_exp_dmm; /*nwp_Exp_Deg_Mixing[Deg_Delete[0]][Deg_Delete[1]];*/
          } else {
            prob_g_g2 = nwp_Deg_Distr_1[Deg_nwp[0]]* nwp_Deg_Distr_2[Deg_nwp[1]] - nwp_exp_dmm; /*nwp_Exp_Deg_Mixing[Deg_Delete[0]][Deg_Delete[1]];*/
          }
        }
        
        if (nwp->nedges < MHp_nedges) { //Edge Removed MHp -> nwp
          prob_g2_g = MHp_exp_dmm;
        } else { //Edge Added MHp -> nwp
          if ((Deg_MHp[0] == Deg_MHp[1])  && (Cov_types[0] == Cov_types[1])) {
            prob_g2_g = (MHp_Deg_Distr_1[Deg_MHp[0]]* (MHp_Deg_Distr_2[Deg_MHp[1]]-1)*.5) - MHp_exp_dmm; /*MHp_Exp_Deg_Mixing[Deg_Add[0]][Deg_Add[1]]; */
          } else {
            prob_g2_g = MHp_Deg_Distr_1[Deg_MHp[0]]* MHp_Deg_Distr_2[Deg_MHp[1]] - MHp_exp_dmm; /*MHp_Exp_Deg_Mixing[Deg_Add[0]][Deg_Add[1]]; */
          }
        }
        
        
        /* Calculate Probability for Prob_g and Prob_g2*/
        calc_probs_mixing_degdist(length_deg_dist, m, nwp, mtp2, prob_type,
                                  networkstatistics, meanvalues, varvalues,
                                  nwp_mixing_matrix, MHp_mixing_matrix, MHp_nedges,
                                  &pdf_gaussian_nwp, &pdf_gaussian_MHp);
        
      }
      
    }
    /// Two Degree Distributions and Mixing: END ///
    
    /////DEGREE MIXING MATRIX//////////////////////
    
    // if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] >= 1) && (prob_type[3] >= 0) && (prob_type[4] >= 1)){
    // 
    //   // --- Variables assumed to be in scope from your MH loop ---
    //   // Network *nwp; Model *m; MHproposal *MHp; 
    //   // double *networkstatistics; int *prob_type;
    //   // double *meanvalues; double *varvalues;
    //   
    //   // 1. Local storage variables
    //   int num_deg_stats, Proposal_prob_zero;
    //   int Deg_nwp[2], Deg_MHp[2];
    //   int deg_dist_nwp[1000], deg_dist_MHp[1000]; 
    //   int n_i[1000], n_j[1000];
    //   double prob_g_g2 = 1.0, prob_g2_g = 1.0;
    //   double pdf_gaussian_nwp, pdf_gaussian_MHp;
    //   
    //   // Calculate n_dim to allocate mixing matrices
    //   int n_stats_for_dim = (prob_type[3] > 0) ? (m->n_stats - 2) : (m->n_stats - 1);
    //   int n_dim = (int)((-1.0 + sqrt(1.0 + 8.0 * n_stats_for_dim)) / 2.0);
    //   
    //   int *g_dmm_flat = malloc(n_dim * n_dim * sizeof(int));
    //   int *g2_dmm_flat = malloc(n_dim * n_dim * sizeof(int));
    //   
    //   // 2. Call Statistics and Matrix Reconstruction
    //   calc_stat_degmixing(nwp, m, MHp, networkstatistics, prob_type,
    //                       &num_deg_stats, Deg_nwp, Deg_MHp,
    //                       deg_dist_nwp, deg_dist_MHp,
    //                       g_dmm_flat, g2_dmm_flat, 
    //                       n_i, n_j, &Proposal_prob_zero);
    //   
    //   if (Proposal_prob_zero == 0) {
    //     
    //     // 3. Logic Branch for Clustering vs. Standard Mixing
    //     if (prob_type[3] > 0) {
    //       // Calculate Mixing Base
    //       calc_f_degmixing(num_deg_stats, Deg_nwp, Deg_MHp, 
    //                        deg_dist_nwp, deg_dist_MHp,
    //                        g_dmm_flat, g2_dmm_flat, n_i, n_j, 
    //                        (nwp->nedges + m->workspace[0]), nwp->nedges, 
    //                        &prob_g_g2, &prob_g2_g);
    //       
    //       // Add Clustering Multiplier
    //       calc_f_degmixing_clustering(num_deg_stats, nwp, m, networkstatistics,
    //                                   deg_dist_nwp, deg_dist_MHp,
    //                                   g_dmm_flat, g2_dmm_flat, 
    //                                   Deg_nwp, Deg_MHp, 
    //                                   (nwp->nedges + m->workspace[0]), 
    //                                   &prob_g_g2, &prob_g2_g);
    //       
    //       // Calculate Probabilities (Precision matrix includes Triangles)
    //       calc_prob_degmixing_clustering(m, networkstatistics, meanvalues, 
    //                                      varvalues, &pdf_gaussian_nwp, &pdf_gaussian_MHp);
    //     } else {
    //       // Standard Mixing Only
    //       calc_f_degmixing(num_deg_stats, Deg_nwp, Deg_MHp, 
    //                        deg_dist_nwp, deg_dist_MHp,
    //                        g_dmm_flat, g2_dmm_flat, n_i, n_j, 
    //                        (nwp->nedges + m->workspace[0]), nwp->nedges, 
    //                        &prob_g_g2, &prob_g2_g);
    //       
    //       calc_prob_degmixing(m, networkstatistics, meanvalues, 
    //                           varvalues, &pdf_gaussian_nwp, &pdf_gaussian_MHp);
    //     }
    //     
    //     // 4. Metropolis-Hastings Ratio
    //     double ratio = (pdf_gaussian_MHp - pdf_gaussian_nwp) + log(prob_g2_g / prob_g_g2);
    //     
    //     if (unif_rand() < exp(ratio)) {
    //       // Accept Step...
    //     }
    //   }
    //   
    //   free(g_dmm_flat);
    //   free(g2_dmm_flat);
    // }
    
    if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] >= 1) && (prob_type[3] >= 0) && (prob_type[4] >= 1)){
      
      int num_deg_stats = (int)((-1.0 + sqrt(1.0 + 8.0 * (m->n_stats-1))) / 2.0) + 1; //(int)(sqrt(2*(double)(m->n_stats-1))) + 1;
      if (prob_type[3] > 0) {
        num_deg_stats = (int)((-1.0 + sqrt(1.0 + 8.0 * (m->n_stats-2))) / 2.0) + 1; //(int)(sqrt(2*(double)(m->n_stats-2))) + 1; //minus 2 for edges and clustering
      }
      
      int Proposal_prob_zero = 0;
      
      Edge nextedge=0;
      int nmax = 100000;
      
      int index1;
      int index2;
      
      int g_dmm[num_deg_stats-1][num_deg_stats-1];
      
      counter = 1;
      for (index1 = 0; index1 < (num_deg_stats-1); index1++) { //ignore number of edges
        for (index2 = 0; index2 <= index1; index2++) { //ignore number of edges
          g_dmm[index1][index2] = networkstatistics[counter];
          g_dmm[index2][index1] = networkstatistics[counter];
          counter++;
        }
      }
      
      int deg_dist_nwp[num_deg_stats];
      int sum_g_dmm;
      int num_deg_nodes = 0;
      
      for (index1 = 0; index1 < (num_deg_stats-1); index1++) { //ignore number of edges
        sum_g_dmm = 0;
        for (index2 = 0; index2 < (num_deg_stats-1); index2++) { //ignore number of edges
          sum_g_dmm += g_dmm[index1][index2];
          if (index1 == index2){
            sum_g_dmm += g_dmm[index1][index2];
          }
        }
        deg_dist_nwp[index1+1] = (int)sum_g_dmm/(index1+1);
        num_deg_nodes += deg_dist_nwp[index1+1];
      }
      deg_dist_nwp[0] = nwp->nnodes - num_deg_nodes;
      
      
      //Step 1: begin
      int Deg_nwp[2];
      int Deg_MHp[2];
      
      Deg_nwp[0] = OUT_DEG[*(MHp->toggletail)] + IN_DEG[*(MHp->toggletail)];
      Deg_nwp[1] = OUT_DEG[*(MHp->togglehead)] + IN_DEG[*(MHp->togglehead)];
      
      if (nwp->nedges > MHp_nedges) {
        Deg_MHp[0] = Deg_nwp[0] - 1;
        Deg_MHp[1] = Deg_nwp[1] - 1;
      } else {
        Deg_MHp[0] = Deg_nwp[0] + 1;
        Deg_MHp[1] = Deg_nwp[1] + 1;
      }
      //Step 1: end
      
      
      if ((Deg_MHp[0] > (num_deg_stats-1)) || (Deg_MHp[1] > (num_deg_stats-1))) {
        Proposal_prob_zero = 1;
      }
      
      if (Proposal_prob_zero == 1) {
        prob_g2_g = 1;
        pdf_gaussian_MHp = log(0);
        prob_g_g2 = 1;
        pdf_gaussian_nwp = 0;
      } else {
        
        //Step 2: begin - Find degrees of neighbors
        int n_i[num_deg_stats-1]; //Do not need degree zero
        int n_j[num_deg_stats-1];
        
        for (counter = 0; counter < (num_deg_stats-1); counter++) {
          n_i[counter] = 0;
          n_j[counter] = 0;
        }
        
        for(Vertex e = EdgetreeMinimum(nwp->outedges, *(MHp->toggletail));
            nwp->outedges[e].value != 0 && nextedge < nmax;
            e = EdgetreeSuccessor(nwp->outedges, e)){
          Vertex k = nwp->outedges[e].value;
          n_i[OUT_DEG[k]+IN_DEG[k]-1]++;    //Index starts at zero
        }
        
        for(Vertex e = EdgetreeMinimum(nwp->inedges, *(MHp->toggletail));
            nwp->inedges[e].value != 0 && nextedge < nmax;
            e = EdgetreeSuccessor(nwp->inedges, e)){
          Vertex k = nwp->inedges[e].value;
          n_i[OUT_DEG[k]+IN_DEG[k]-1]++;    //Index starts at zero
        }
        
        for(Vertex e = EdgetreeMinimum(nwp->outedges, *(MHp->togglehead));
            nwp->outedges[e].value != 0 && nextedge < nmax;
            e = EdgetreeSuccessor(nwp->outedges, e)){
          Vertex k = nwp->outedges[e].value;
          n_j[OUT_DEG[k]+IN_DEG[k]-1]++;    //Index starts at zero
        }
        
        for(Vertex e = EdgetreeMinimum(nwp->inedges, *(MHp->togglehead));
            nwp->inedges[e].value != 0 && nextedge < nmax;
            e = EdgetreeSuccessor(nwp->inedges, e)){
          Vertex k = nwp->inedges[e].value;
          n_j[OUT_DEG[k]+IN_DEG[k]-1]++;    //Index starts at zero
        }
        
        int denominator;
        int numerator;
        
        //Step 3a: begin - ADD
        if (nwp->nedges < MHp_nedges) {
          if (Deg_nwp[0] == Deg_nwp[1]) {
            prob_g_g2 = deg_dist_nwp[Deg_nwp[0]] * (deg_dist_nwp[Deg_nwp[0]] - 1) * .5;
          } else {
            prob_g_g2 = deg_dist_nwp[Deg_nwp[0]] * (deg_dist_nwp[Deg_nwp[1]]);
          }
          if (Deg_nwp[0] > 0 && Deg_nwp[1] > 0 ) {
            prob_g_g2 = prob_g_g2 - g_dmm[Deg_nwp[0]-1][Deg_nwp[1]-1];
          }
          
          
          for (index1=0; index1 < ((num_deg_stats)-1); index1++){
            g_dmm[index1][index1] = 2 * g_dmm[index1][index1];
          }
          
          //Step 3b: begin - ADD identical degrees
          
          numerator = 1;
          
          if ((Deg_nwp[0] == Deg_nwp[1]) && (Deg_nwp[0] > 0)) {
            denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0]), (Deg_nwp[0] + Deg_nwp[1]));
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter] + n_j[counter]));
            }
            prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
          } else {
            //Step 3c: begin - ADD degree Node 1
            
            if (Deg_nwp[0] > 0) {
              denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0]), (Deg_nwp[0]));
              numerator = 1;
              for (counter=0; counter < (num_deg_stats-1); counter++) {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter]));
              }
              prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
            }
            //Step 3d: begin - ADD degree Node 2
            if (Deg_nwp[1] > 0) {
              denominator = calcCNR( (deg_dist_nwp[Deg_nwp[1]] * Deg_nwp[1]), (Deg_nwp[1]));
              numerator = 1;
              for (counter=0; counter < (num_deg_stats-1); counter++) {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[1]-1][counter], (n_j[counter]));
              }
              prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
            }
          }
        } else {
          //Step 4a: begin - Remove
          prob_g_g2 = g_dmm[Deg_nwp[0]-1][Deg_nwp[1]-1];
          
          for (index1=0; index1 < ((num_deg_stats)-1); index1++){
            g_dmm[index1][index1] = 2 * g_dmm[index1][index1];
          }
          
          //Step 4b: begin - ADD identical degrees
          numerator = 1;
          
          if (Deg_nwp[0] == Deg_nwp[1]) {
            denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0] -1 ), (Deg_nwp[0] + Deg_nwp[1] - 2));
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              if (counter == (Deg_nwp[0]-1)) {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter]-1, (n_i[counter] + n_j[counter] - 2));
              } else {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter] + n_j[counter]));
              }
            }
            prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
          } else {
            //Step 4c: begin - ADD node 1 degrees
            denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0] - 1), (Deg_nwp[0] - 1));
            numerator = 1;
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              if (counter == (Deg_nwp[1]-1)) {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter]-1, (n_i[counter] - 1));
              } else {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter]));
              }
            }
            prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
            //Step 4d: begin - ADD node 2 degrees
            denominator = calcCNR( (deg_dist_nwp[Deg_nwp[1]] * Deg_nwp[1] - 1), (Deg_nwp[1] - 1));
            numerator = 1;
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              if (counter == (Deg_nwp[0]-1)) {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[1]-1][counter]-1, (n_j[counter]-1));
              } else {
                numerator = numerator * calcCNR(g_dmm[Deg_nwp[1]-1][counter], (n_j[counter]));
              }
            }
            prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
            
          }
        }
        
        ////g2 -> g
        
        int g2_dmm[num_deg_stats-1][num_deg_stats-1];
        
        counter = 1;
        for (index1 = 0; index1 < (num_deg_stats-1); index1++) { //ignore number of edges
          for (index2 = 0; index2 <= index1; index2++) { //ignore number of edges
            g2_dmm[index1][index2] = networkstatistics[counter] + m->workspace[counter];  //Change from g -> g2
            g2_dmm[index2][index1] = networkstatistics[counter] + m->workspace[counter];  //Change from g -> g2
            counter++;
          }
        }
        
        int deg_dist_MHp[num_deg_stats];
        num_deg_nodes = 0;
        
        for (index1 = 0; index1 < (num_deg_stats-1); index1++) { //ignore number of edges
          sum_g_dmm = 0;
          for (index2 = 0; index2 < (num_deg_stats-1); index2++) { //ignore number of edges
            sum_g_dmm += g2_dmm[index1][index2];
            if (index1 == index2){
              sum_g_dmm += g2_dmm[index1][index2];
            }
          }
          deg_dist_MHp[index1+1] = (int)sum_g_dmm/(index1+1);
          num_deg_nodes += deg_dist_MHp[index1+1];
        }
        deg_dist_MHp[0] = nwp->nnodes - num_deg_nodes;
        
        
        if (nwp->nedges < MHp_nedges) { //an edge was added
          n_i[Deg_MHp[1]-1]++;
          n_j[Deg_MHp[0]-1]++;
        } else { //an edge was removed
          n_i[Deg_nwp[1]-1]--;
          n_j[Deg_nwp[0]-1]--;
        }
        
        //Step 3a: begin - ADD
        
        if (nwp->nedges > MHp_nedges) {
          if (Deg_nwp[0] == Deg_nwp[1]) {
            prob_g2_g = deg_dist_MHp[Deg_MHp[0]] * (deg_dist_nwp[Deg_MHp[0]] - 1) * .5;
          } else {
            prob_g2_g = deg_dist_MHp[Deg_MHp[0]] * (deg_dist_nwp[Deg_MHp[1]]);
          }
          if (Deg_MHp[0] > 0 && Deg_MHp[1] > 0 ) {
            prob_g2_g = prob_g2_g - g2_dmm[Deg_MHp[0]-1][Deg_MHp[1]-1];
          }
          
          for (index1=0; index1 < ((num_deg_stats)-1); index1++){
            g2_dmm[index1][index1] = 2 * g2_dmm[index1][index1];
          }
          
          //Step 3b: begin - ADD identical degrees
          
          numerator = 1;
          
          if ((Deg_MHp[0] == Deg_MHp[1]) && (Deg_MHp[0] > 0)) {
            denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0]), (Deg_MHp[0] + Deg_MHp[1]));
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter] + n_j[counter]));
            }
            prob_g2_g = prob_g2_g * ((double)numerator / denominator);
          } else {
            //Step 3c: begin - ADD degree Node 1
            
            if (Deg_MHp[0] > 0) {
              denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0]), (Deg_MHp[0]));
              numerator = 1;
              for (counter=0; counter < (num_deg_stats-1); counter++) {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter]));
              }
              prob_g2_g = prob_g2_g * ((double)numerator / denominator);
            }
            //Step 3d: begin - ADD degree Node 2
            if (Deg_MHp[1] > 0) {
              denominator = calcCNR( (deg_dist_MHp[Deg_MHp[1]] * Deg_MHp[1]), (Deg_MHp[1]));
              numerator = 1;
              for (counter=0; counter < (num_deg_stats-1); counter++) {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[1]-1][counter], (n_j[counter]));
              }
              prob_g2_g = prob_g2_g * ((double)numerator / denominator);
            }
          }
        } else {
          //Step 4a: begin - Remove
          prob_g2_g = g2_dmm[Deg_MHp[0]-1][Deg_MHp[1]-1];
          //Step 4b: begin - ADD identical degrees
          
          for (index1=0; index1 < ((num_deg_stats)-1); index1++){
            g2_dmm[index1][index1] = 2 * g2_dmm[index1][index1];
          }
          
          numerator = 1;
          
          if (Deg_MHp[0] == Deg_MHp[1]) {
            denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0] -1 ), (Deg_MHp[0] + Deg_MHp[1] - 2));
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              if (counter == (Deg_MHp[0]-1)) {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter]-1, (n_i[counter] + n_j[counter] - 2));
              } else {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter] + n_j[counter]));
              }
            }
            prob_g2_g = prob_g2_g * ((double)numerator / denominator);
          } else {
            //Step 4c: begin - ADD node 1 degrees
            denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0] - 1), (Deg_MHp[0] - 1));
            numerator = 1;
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              if (counter == (Deg_MHp[1]-1)) {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter]-1, (n_i[counter] - 1));
              } else {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter]));
              }
            }
            prob_g2_g = prob_g2_g * ((double)numerator / denominator);
            //Step 4d: begin - ADD node 2 degrees
            denominator = calcCNR( (deg_dist_MHp[Deg_MHp[1]] * Deg_MHp[1] - 1), (Deg_MHp[1] - 1));
            numerator = 1;
            for (counter=0; counter < (num_deg_stats-1); counter++) {
              if (counter == (Deg_MHp[0]-1)) {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[1]-1][counter]-1, (n_j[counter]-1));
              } else {
                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[1]-1][counter], (n_j[counter]));
              }
            }
            prob_g2_g = prob_g2_g * ((double)numerator / denominator);
          }
        }
        
        for (index1=0; index1 < ((num_deg_stats)-1); index1++){ //Undo what was done for degree mixing
          g_dmm[index1][index1] = .5 * g_dmm[index1][index1];
          g2_dmm[index1][index1] = .5 * g2_dmm[index1][index1];
        }
        
        calc_prob_degmixing(
          m,                  // Pointer to the ModelStruct
          networkstatistics,  // Current global network statistics array
          meanvalues,         // Target mean values (from your input/model)
          varvalues,          // Precision matrix (inverse of covariance)
          &pdf_gaussian_nwp,  // Address of the current state's log-probability
          &pdf_gaussian_MHp   // Address of the proposed state's log-probability
        );
        
        ///Calculate probs
        
        
        
        //   if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 1) && (prob_type[3] >= 0) && (prob_type[4] == 1)){
        // 
        //     double nwp_mu_diff[m->n_stats-1];
        //     double MHp_mu_diff[m->n_stats-1];
        // 
        //     for (counter = 0; counter < m->n_stats-1; counter++) {
        //         nwp_mu_diff[counter] = (double)(networkstatistics[counter+1]) - meanvalues[counter];
        //         MHp_mu_diff[counter] = (double)(networkstatistics[counter+1] + m->workspace[counter+1]) - meanvalues[counter];
        //     }
        // 
        //     double nwp_intermediate_mat[m->n_stats-1];
        //     double MHp_intermediate_mat[m->n_stats-1];
        // 
        //     counter2 = -1;
        //     for (counter = 0; counter < (m->n_stats-1) * (m->n_stats-1); counter++) {
        //         counter1 = counter % (m->n_stats-1);
        //         if (counter1 == 0) {
        //             counter2 += 1;
        //             nwp_intermediate_mat[counter2] = 0;
        //             MHp_intermediate_mat[counter2] = 0;
        //         }
        //         nwp_intermediate_mat[counter2] += nwp_mu_diff[counter1]*varvalues[counter];
        //         MHp_intermediate_mat[counter2] += MHp_mu_diff[counter1]*varvalues[counter];
        //     }
        // 
        //     pdf_gaussian_nwp = 0;
        //     pdf_gaussian_MHp = 0;
        //     for (counter = 0; counter < (m->n_stats-1); counter++) {
        //         pdf_gaussian_nwp += nwp_intermediate_mat[counter] * nwp_mu_diff[counter];
        //         pdf_gaussian_MHp += MHp_intermediate_mat[counter] * MHp_mu_diff[counter];
        //     }
        //      pdf_gaussian_nwp =  -.5 * pdf_gaussian_nwp;
        //      pdf_gaussian_MHp =  -.5 * pdf_gaussian_MHp;
        // 
        // }
        
        
        /////CLUSTERING//////////////////////
        
        if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] >= 1) && (prob_type[3] >= 1) && (prob_type[4] >= 1)){
          
          
          double num_Tri = networkstatistics[m->n_stats-1]; //should be last statistic
          double num_Tri_change = fabs(m->workspace[m->n_stats-1]); //should be last statistic
          
          int n_dim = num_deg_stats - 1;
          
          // Allocate flat arrays (or use a pre-allocated workspace)
          int g_dmm_flat[n_dim * n_dim];
          int g2_dmm_flat[n_dim * n_dim];
          
          // Flatten and undo the diagonal doubling simultaneously
          for (int i = 0; i < n_dim; i++) {
            for (int j = 0; j < n_dim; j++) {
              int idx = i * n_dim + j;
              if (i == j) {
                g_dmm_flat[idx] = (int)(g_dmm[i][j]);
                g2_dmm_flat[idx] = (int)(g2_dmm[i][j]);
              } else {
                g_dmm_flat[idx] = g_dmm[i][j];
                g2_dmm_flat[idx] = g2_dmm[i][j];
              }
            }
          }
          
          calc_f_degmixing_clustering(
            num_deg_stats,          // The dimension calculated in calc_stat_degmixing
            nwp,                    // Pointer to the current Network
            m,                      // Pointer to the Model (contains workspace)
            networkstatistics,       // The current stats array (used to get num_Tri)
            deg_dist_nwp,           // Degree distribution of the current network
            deg_dist_MHp,           // Degree distribution of the proposed network
            g_dmm_flat,             // Current mixing matrix (flattened)
            g2_dmm_flat,            // Proposed mixing matrix (flattened)
            Deg_nwp,                // Current degrees of the toggle pair
            Deg_MHp,                // Proposed degrees of the toggle pair
            MHp_nedges,             // Proposed edge count (nwp->nedges + m->workspace[0])
            &prob_g_g2,             // Pointer to forward proposal probability
            &prob_g2_g              // Pointer to reverse proposal probability
          );
        }
        /////CLUSTERING//////////////////////
      }
    }
    /////DEGREE MIXING MATRIX//////////////////////
    
    //Rprintf("Probs (before cutoff): %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
    
    
    if (!isfinite(pdf_gaussian_nwp)) {
      prob_g2_g = 1;
      pdf_gaussian_MHp = 0;
      prob_g_g2 = 1;
      pdf_gaussian_nwp = log(0);
      
      if (print_info_MH == 1) {
        Rprintf("NWP INVALID 1: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
      }
    }
    
    if (pdf_gaussian_nwp != pdf_gaussian_nwp) {
      prob_g2_g = 1;
      pdf_gaussian_MHp = 0;
      prob_g_g2 = 1;
      pdf_gaussian_nwp = log(0);
      
      if (print_info_MH == 1) {
        Rprintf("NWP INVALID 2: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
      }
    }
    
    cutoff = (log(prob_g2_g) + pdf_gaussian_MHp) - (log(prob_g_g2) + pdf_gaussian_nwp) + MHp->logratio;
    if (print_info_MH == 1) {
      Rprintf("CUTOFF: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
    }
    //Bayesian: BEGIN//
    
    /* REMOVED*/
    
    //Bayesian: END//
  }
  
  /* CODE FROM ERGM Library
   *  File ergm/src/MCMC.c
   *  Part of the statnet package, http://statnet.org
   *
   *  This software is distributed under the GPL-3 license.  It is free,
   *  open source, and has the attribution requirements (GPL Section 7) in
   *    http://statnet.org/attribution
   *
   *  Copyright 2012 the statnet development team
   */
  /* if we accept the proposed network */
  if (cutoff >= 0.0 || log(unif_rand()) < cutoff) {
    /* Make proposed toggles (updating timestamps--i.e., for real this time) */
    for(unsigned int i=0; i < MHp->ntoggles; i++){
      ToggleEdge(MHp->toggletail[i], MHp->togglehead[i], nwp);
      
      if(MHp->discord)
        for(Network **nwd=MHp->discord; *nwd!=NULL; nwd++){
          ToggleEdge(MHp->toggletail[i],  MHp->togglehead[i], *nwd);
        }
    }
    /* record network statistics for posterity */
    //Rprintf("END - Network Statistic: ");
    for (unsigned int i = 0; i < m->n_stats; i++){
      networkstatistics[i] += m->workspace[i];
      //Rprintf(" %f ",networkstatistics[i]);
    }
    //Rprintf("\n");
    taken++;
  }
  }
  *staken = taken;
  return MCMC_OK;
  //END of CODE FROM ERGM Library
}
