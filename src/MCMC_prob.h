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
  
  double cutoff=log(0);
  
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
  int counter;
  //int counter1;
  //int counter2;
  
  double pdf_gaussian_nwp = 0;
  double pdf_gaussian_MHp = log(0);
  
  MHp_nedges = networkstatistics[0] + m->workspace[0];
  
  
  //Rprintf("Before ergm vs GUF: MH Code \n");
  if ((theta[0] < -999) && (theta[0] > -1000)) {
    //Rprintf("Entered GUF: MH Code \n");
    
    ///EDGES: BEGIN///
    if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)) {
      
      // Stage 1: Keep raw counts for Stage 2
      double v_current_raw[1] = { (double)networkstatistics[0] };
      double v_proposal_raw[1] = { (double)MHp_nedges };
      
      // Create a separate version for Stage 3 if density is required
      double v_current_stat[1] = { v_current_raw[0] };
      double v_proposal_stat[1] = { v_proposal_raw[0] };
      
      // Apply transformation for the Statistical Model if needed
      // (e.g., if prob_type[4] indicates a density-based distribution like Beta)
      if (prob_type[4] == 2) {
        v_current_stat[0] /= (double)total_max_edges;
        v_proposal_stat[0] /= (double)total_max_edges;
      }
      
      // Stage 2: Combinatorial logic ALWAYS uses the discrete counts
      calc_f_edges(v_current_raw[0], v_proposal_raw[0], total_max_edges, networkstatistics, &prob_g_g2, &prob_g2_g);
      
      // Stage 3: Statistical logic uses the transformed values (density or raw)
      calc_prob_dist(v_current_stat, v_proposal_stat, 1, prob_type, 
                     meanvalues, varvalues, 
                     &pdf_gaussian_nwp, &pdf_gaussian_MHp);
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
      //calc_probs_mixing(nwp, 3, Cov_types, Num_Cov_type, nwp_mix, MHp_mix, meanvalues, varvalues, &pdf_gaussian_nwp, &pdf_gaussian_MHp, m, MHp, networkstatistics, prob_type);
      calc_prob_dist(nwp_mix, MHp_mix, num_params, prob_type, 
                     meanvalues, varvalues, 
                     &pdf_gaussian_nwp, &pdf_gaussian_MHp);
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
        
        // 3.1. Allocate temporary double arrays for the universal function
        double *v_current = (double *)R_alloc(num_deg_stats, sizeof(double));
        double *v_proposal = (double *)R_alloc(num_deg_stats, sizeof(double));
        
        // 3.2. Explicitly cast the int distribution counts to doubles
        for (int i = 0; i < num_deg_stats; i++) {
          v_current[i] = (double)nwp_Deg_Distr[i];
          v_proposal[i] = (double)MHp_Deg_Distr[i];
        }
        
        calc_prob_dist(v_current, v_proposal, num_deg_stats, prob_type, 
                       meanvalues, varvalues, 
                       &pdf_gaussian_nwp, &pdf_gaussian_MHp);
      }
    }
    /// DEGREE DISTRIBUTION: END ///
    
    
    /// Two Degree Distributions and Mixing: BEGIN ///
    if ((prob_type[0] >= 1) && (prob_type[1] >= 1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
      
      int num_deg_stats = m->n_stats-1 - 3; //The four is for number of mixing - only coded for two node types
      int length_deg_dist = num_deg_stats / 2; //Currently both degree distributions have to be the same size
      int Deg_nwp[2];
      int Deg_MHp[2];
      int Cov_types[2]; //Node types for the endpoints of toggled edge

      int MHp_Deg_Distr_1[length_deg_dist]; // Degree Distribution and Edges for Node 1 and Node 2
      int MHp_Deg_Distr_2[length_deg_dist]; // Node 1 and Node 2 are the nodes of the toggled edge
      int nwp_Deg_Distr_1[length_deg_dist];
      int nwp_Deg_Distr_2[length_deg_dist];
      double nwp_mixing_matrix[3];
      double MHp_mixing_matrix[3];
      
      ModelTerm *mtp2 = m->termarray;
      
      //Stage 1: Calculate Network Statistics
      for (counter = 1; counter < (length_deg_dist + 1); counter++) {
        MHp_Deg_Distr_1[counter - 1] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
        nwp_Deg_Distr_1[counter - 1] = (int)networkstatistics[counter];
      }
      
      for (counter = (length_deg_dist + 1); counter < (2 * length_deg_dist + 1); counter++) {
        MHp_Deg_Distr_2[counter - (length_deg_dist + 1)] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
        nwp_Deg_Distr_2[counter - (length_deg_dist + 1)] = (int)networkstatistics[counter];
      }
      
      nwp_mixing_matrix[0] = networkstatistics[2*length_deg_dist + 1];
      nwp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 2];
      nwp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 3];
      
      MHp_mixing_matrix[0] = networkstatistics[2*length_deg_dist + 1] + m->workspace[2*length_deg_dist + 1];
      MHp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 2] + m->workspace[2*length_deg_dist + 2];
      MHp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 3] + m->workspace[2*length_deg_dist + 3];
      
      int Proposal_prob_zero = 0;
      
      //Stage 2: Calculate Congruence Class Ratio
      
      Deg_nwp[0] = OUT_DEG[*(MHp->toggletail)] + IN_DEG[*(MHp->toggletail)];
      Deg_nwp[1] = OUT_DEG[*(MHp->togglehead)] + IN_DEG[*(MHp->togglehead)];
      
      if (nwp->nedges > MHp_nedges) { // Removal
        Deg_MHp[0] = Deg_nwp[0] - 1;
        Deg_MHp[1] = Deg_nwp[1] - 1;
      } else { // Addition
        Deg_MHp[0] = Deg_nwp[0] + 1;
        Deg_MHp[1] = Deg_nwp[1] + 1;
      }
      
      calculate_congruence_ratio_degmix(
        m, nwp, MHp, 
        length_deg_dist, MHp_nedges, 
        Deg_nwp, Deg_MHp, Cov_types,
        networkstatistics, 
        nwp_Deg_Distr_1, nwp_Deg_Distr_2, MHp_Deg_Distr_1, MHp_Deg_Distr_2,
        nwp_mixing_matrix, MHp_mixing_matrix,
        &prob_g_g2, &prob_g2_g,           // Pass by address
        &pdf_gaussian_nwp, &pdf_gaussian_MHp, &Proposal_prob_zero
      );
      
        if (Proposal_prob_zero == 0) {
          
          // calc_probs_mixing_degdist(length_deg_dist, m, nwp, mtp2, prob_type,
          //                         networkstatistics, meanvalues, varvalues,
          //                         nwp_mixing_matrix, MHp_mixing_matrix, MHp_nedges,
          //                         &pdf_gaussian_nwp, &pdf_gaussian_MHp);
        
        // --- BLOCK 1: First Degree Distribution ---
        int num_degmix_stats = prob_type[6]; // length of first degree distribution
          int num_degmix_var = prob_type[7];   // length of variance for first
          
          double v_current_stat[num_degmix_stats];
          double v_proposal_stat[num_degmix_stats];
          double meanvalues_TEMP[num_degmix_stats];
          double varvalues_TEMP[num_degmix_var];
          
          counter = 1; 
          for (int i = 0; i < num_degmix_stats; i++) {
            v_current_stat[i] = (double)networkstatistics[counter];
            v_proposal_stat[i] = (double)(networkstatistics[counter] + m->workspace[counter]);
            meanvalues_TEMP[i] = (double)meanvalues[i];
            counter++;
          }
          for (int i = 0; i < num_degmix_var; i++) {
            varvalues_TEMP[i] = (double)varvalues[i];
          }
          
          //double pdf_gaussian_nwp = 0;
          //double pdf_gaussian_MHp = 0;
          
          calc_prob_dist(v_current_stat, v_proposal_stat, num_degmix_stats, prob_type,
                         meanvalues_TEMP, varvalues_TEMP,
                         &pdf_gaussian_nwp, &pdf_gaussian_MHp);
          
          // --- BLOCK 2: Second Degree Distribution ---
          int num_degmix_stats2 = prob_type[9];
          int num_degmix_var2 = prob_type[10];
          
          double v_current_stat2[num_degmix_stats2];
          double v_proposal_stat2[num_degmix_stats2];
          double meanvalues_TEMP2[num_degmix_stats2];
          double varvalues_TEMP2[num_degmix_var2];
          
          // Counter continues from where Block 1 left off
          for (int i = 0; i < num_degmix_stats2; i++) {
            v_current_stat2[i] = (double)networkstatistics[counter];
            v_proposal_stat2[i] = (double)(networkstatistics[counter] + m->workspace[counter]);
            // Meanvalues offset by Block 1 length (prob_type[6])
            meanvalues_TEMP2[i] = (double)meanvalues[prob_type[6] + i];
            counter++;
          }
          for (int i = 0; i < num_degmix_var2; i++) {
            // Varvalues offset by Block 1 variance length (prob_type[7])
            varvalues_TEMP2[i] = (double)varvalues[prob_type[7] + i];
          }
          
          int prob_type_TEMP2[6];
          for (int i = 0; i < 5; i++) prob_type_TEMP2[i] = prob_type[i];
          prob_type_TEMP2[5] = prob_type[8]; // Distribution type for 2nd deg dist
          
          double pdf_gaussian_nwp_TEMP2 = 0;
          double pdf_gaussian_MHp_TEMP2 = 0; // Avoid log(0) unless calc_prob_dist uses +=
          
          calc_prob_dist(v_current_stat2, v_proposal_stat2, num_degmix_stats2, prob_type_TEMP2,
                         meanvalues_TEMP2, varvalues_TEMP2,
                         &pdf_gaussian_nwp_TEMP2, &pdf_gaussian_MHp_TEMP2);
          
          // --- BLOCK 3: Mixing Statistics ---
          // These are the 13th (mean length) and 14th (var length) items in prob_type
          // int num_degmix_stats3 = prob_type[12]; 
          // int num_degmix_var3 = prob_type[13];
          // 
          // double v_current_stat3[num_degmix_stats3];
          // double v_proposal_stat3[num_degmix_stats3];
          // double meanvalues_TEMP3[num_degmix_stats3];
          // double varvalues_TEMP3[num_degmix_var3];
          // 
          // for (int i = 0; i < num_degmix_stats3; i++) {
          //   v_current_stat3[i] = (double)networkstatistics[counter];
          //   v_proposal_stat3[i] = (double)(networkstatistics[counter] + m->workspace[counter]);
          //   // Offset by Block 1 + Block 2
          //   meanvalues_TEMP3[i] = (double)meanvalues[prob_type[6] + prob_type[9] + i];
          //   counter++;
          // }
          // for (int i = 0; i < num_degmix_var3; i++) {
          //   varvalues_TEMP3[i] = (double)varvalues[prob_type[7] + prob_type[10] + i];
          // }
          
          
          double v_current_stat3[1];
          double v_proposal_stat3[1];
          double meanvalues_TEMP3[1];
          double varvalues_TEMP3[1];
          
          counter = prob_type[6] + prob_type[9] + 2; //Need to skip M11
          int counter2 = prob_type[6] + prob_type[9];
          int counter3 = prob_type[7] + prob_type[10];
          
          v_current_stat3[0] = (double)networkstatistics[counter];
          v_proposal_stat3[0] = (double)(networkstatistics[counter] + m->workspace[counter]);
          meanvalues_TEMP3[0] = (double)meanvalues[counter2];
          varvalues_TEMP3[0] = (double)varvalues[counter3];
          
          prob_type_TEMP2[5] = prob_type[11]; // Distribution type for Mixing
          
          double pdf_gaussian_nwp_TEMP3 = 0;
          double pdf_gaussian_MHp_TEMP3 = 0; 
          
          //--- DIAGNOSTIC PRINT STATEMENTS ---
          // Rprintf("\n--- DEBUG BLOCK 3 (Mixing) ---\n");
          // Rprintf("Global Mean Index used: %d | Value: %f\n", prob_type[6] + prob_type[9], meanvalues[prob_type[6] + prob_type[9]]);
          // Rprintf("Global Var Index used: %d  | Value: %f\n", prob_type[7] + prob_type[10], varvalues[prob_type[7] + prob_type[10]]);
          // Rprintf("TEMP3 Mean: %f | TEMP3 Var: %f\n", meanvalues_TEMP3[0], varvalues_TEMP3[0]);
          // Rprintf("Distribution Type (PT_TEMP2[5]): %d\n", prob_type_TEMP2[5]);
          // Rprintf("Full prob_type_TEMP2: [%d, %d, %d, %d, %d, %d]\n",
          //         prob_type_TEMP2[0], prob_type_TEMP2[1], prob_type_TEMP2[2],
          //                                                                prob_type_TEMP2[3], prob_type_TEMP2[4], prob_type_TEMP2[5]);
          // Rprintf("------------------------------\n");
          
          calc_prob_dist(v_current_stat3, v_proposal_stat3, 1, prob_type_TEMP2,
                         meanvalues_TEMP3, varvalues_TEMP3,
                         &pdf_gaussian_nwp_TEMP3, &pdf_gaussian_MHp_TEMP3);
          
          // --- FINAL SUMMATION ---
          pdf_gaussian_nwp += pdf_gaussian_nwp_TEMP2 + pdf_gaussian_nwp_TEMP3;
          pdf_gaussian_MHp += pdf_gaussian_MHp_TEMP2 + pdf_gaussian_MHp_TEMP3;
        }
      }
    /// Two Degree Distributions and Mixing: END ///
    
    /////DEGREE MIXING MATRIX//////////////////////
    
    if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] >= 1) && (prob_type[3] >= 0) && (prob_type[4] >= 1)){
      
      int Proposal_prob_zero = 0;
      
      // 1. Calculate num_deg_stats based on prob_type[3] (Clustering)
      int n_skip = (prob_type[3] > 0) ? 2 : 1;
      int num_deg_stats = (int)((-1.0 + sqrt(1.0 + 8.0 * (m->n_stats - n_skip))) / 2.0) + 1;
      
      // 2. Declare matrices using Variable Length Arrays (VLA)
      int g_dmm[num_deg_stats-1][num_deg_stats-1];
      int g2_dmm[num_deg_stats-1][num_deg_stats-1];
      
      // 3. Populate them in one call
      get_properties(m, networkstatistics, num_deg_stats, g_dmm, g2_dmm);
    
      int deg_dist_nwp[num_deg_stats];
      int deg_dist_MHp[num_deg_stats];
      
      int Deg_nwp[2];
      int Deg_MHp[2];
      
      calculate_congruence_ratios(
        nwp,
        MHp,
        num_deg_stats,
        g_dmm,
        g2_dmm,
        &prob_g_g2,
        &prob_g2_g,
        MHp_nedges,
        &pdf_gaussian_nwp,
        &pdf_gaussian_MHp,
        &Proposal_prob_zero,
        deg_dist_nwp,
        deg_dist_MHp,
        Deg_nwp,
        Deg_MHp
      );
      
      if (Proposal_prob_zero == 1) {
           prob_g2_g = 1;
           pdf_gaussian_MHp = log(0);
           prob_g_g2 = 1;
           pdf_gaussian_nwp = 0;
       } else {
        
      int num_degmix_stats = m->n_stats-1;
      if (prob_type[3] > 0) {
         num_degmix_stats = m->n_stats-2;
      }
      
      double v_current_stat[num_degmix_stats];
      double v_proposal_stat[num_degmix_stats];
      double meanvalues_TEMP[num_degmix_stats];
      
      int num_degmix_var = prob_type[7];
      double varvalues_TEMP[num_degmix_var];
      
      counter = 1;
      for (int index1 = 0; index1 < num_degmix_stats; index1++) { //ignore first edge term
        v_current_stat[index1] = (double)(networkstatistics[counter]);
        v_proposal_stat[index1] = (double)(networkstatistics[counter] + m->workspace[counter]);
        meanvalues_TEMP[index1] = (double)(meanvalues[index1]); //don't ignore
        counter++;
      }
      
      for (int index1 = 0; index1 < num_degmix_var; index1++) {
        varvalues_TEMP[index1] = (double)(varvalues[index1]);
      }
      
      calc_prob_dist(v_current_stat,  v_proposal_stat, num_degmix_stats, prob_type,
                     meanvalues_TEMP, varvalues_TEMP,
                     &pdf_gaussian_nwp, &pdf_gaussian_MHp);
      
      /////CLUSTERING//////////////////////
      
      if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] >= 1) && (prob_type[3] >= 1) && (prob_type[4] >= 1)){
        
        
        //double num_Tri = networkstatistics[m->n_stats-1]; //should be last statistic
        //double num_Tri_change = fabs(m->workspace[m->n_stats-1]); //should be last statistic
        
        int n_dim = num_deg_stats - 1;
        
        // Allocate flat arrays (or use a pre-allocated workspace)
        int g_dmm_flat[n_dim * n_dim];
        int g2_dmm_flat[n_dim * n_dim];
        
        // Flatten
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
        
        double v_current_stat_TEMP2[1];
        v_current_stat_TEMP2[0] = networkstatistics[m->n_stats-1]; //should be last statistic
        
        double v_proposal_stat_TEMP2[1];
        v_proposal_stat_TEMP2[0] = v_current_stat_TEMP2[0] + (m->workspace[m->n_stats-1]); //should be last statistic
        
        double meanvalues_TEMP2[1];
        meanvalues_TEMP2[0] = meanvalues[num_degmix_stats];
        double varvalues_TEMP2[1];
        varvalues_TEMP2[0] = varvalues[num_degmix_var];
        
        double pdf_gaussian_nwp_TEMP2 = 0;
        double pdf_gaussian_MHp_TEMP2 = log(0);
        
        int prob_type_TEMP2[6];
        for (int index1 = 0; index1 < 6; index1++) { //ignore first edge term
          prob_type_TEMP2[index1] = prob_type[index1];
        }
        prob_type_TEMP2[5] = prob_type[8];
        
        //Rprintf("Triangles: %f %f %f %f\n", v_current_stat_TEMP2[0], v_proposal_stat_TEMP2[0], meanvalues_TEMP2[0], varvalues_TEMP2[0]);
        
        calc_prob_dist(v_current_stat_TEMP2,  v_proposal_stat_TEMP2, 1, prob_type_TEMP2,
                       meanvalues_TEMP2, varvalues_TEMP2,
                       &pdf_gaussian_nwp_TEMP2, &pdf_gaussian_MHp_TEMP2);
        
        pdf_gaussian_nwp = pdf_gaussian_nwp_TEMP2 + pdf_gaussian_nwp;
        pdf_gaussian_MHp = pdf_gaussian_MHp_TEMP2 + pdf_gaussian_MHp;
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
