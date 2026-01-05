 /*
 *  File CCMnet/src/MCMC_prob.h
 *
 *  Sections of this code are derived from the ergm library
 *  All such sections are noted and attributed to the statnet development team.
 */

#include "CCMnet_netprop_edges.h"
#include "CCMnet_netprop_degdist.h"
#include "CCMnet_netprop_mixing_degdist.h"

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
                  double *evolutionvar
);

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
        double *evolutionvar
        ) {

    int print_info_MH =0;

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


//int j = 0;

//Rprintf("BayesInference Flag - MH: %d \n", BayesInference);
//Rprintf("BayesInference Flag - MH: %d \n", *BayesInference);
//Rprintf("BayesInference Flag - MH: %d \n", BayesInference[0]);
//Rprintf("BayesInference Flag - MH: %d \n", *BayesInference[0]);

//Rprintf("Epidemic Data\n");
//for (j=0; j < nwp->nnodes; j++){
//   Rprintf("%d : %f %f %f\n",j,Ia[j], Il[j], R_times[j]);
//}
//Rprintf("Beta Acute: %f\n", beta_a_val);
//Rprintf("Beta Long-Term: %f\n", beta_l_val);

//Rprintf("Beta - MH: %f \n", beta_a);
//Rprintf("Beta - MH: %f \n", *beta_a);
//Rprintf("Beta - MH: %f \n", beta_a[0]);
//Rprintf("Beta - MH: %f \n", *beta_a[0]);

if (print_info_MH == 1) {
    Rprintf("MH: Before ChangeStats Code \n");

    Rprintf("nwp info: %d \n", nwp->nnodes);

    Rprintf("WorkSpace: ");
    for (int counter_print=0; counter_print < ((m->n_stats)-1); counter_print++){
        Rprintf(" %f ",m->workspace[counter_print]);
    }
    Rprintf("\n");

    Rprintf("Toggle Info: %d %d %d \n",MHp->ntoggles, *(MHp->toggletail), *(MHp->togglehead));
}

    /* Calculate change statistics,
       remembering that tail -> head */
          ChangeStats(MHp->ntoggles, MHp->toggletail, MHp->togglehead, nwp, m);

    /* Calculate inner product */
/* MOD    double ip=0; */
/* MOD    for (unsigned int i=0; i<m->n_stats; i++){ */
/* MOD      ip += theta[i] * m->workspace[i]; */
/* MOD    } */

if (print_info_MH == 1) {
   Rprintf("MH: After ChangeStats Code \n");

   Rprintf("WorkSpace: ");
   for (int counter_print=0; counter_print < ((m->n_stats)-1); counter_print++){
        Rprintf(" %f ",m->workspace[counter_print]);
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

//for (int counter=0; counter < m->n_stats; counter++){
//        Rprintf("MH Change %d %d %f\n",counter,nwp->nedges,networkstatistics[counter]);
//}

        MHp_nedges = networkstatistics[0] + m->workspace[0];

//Rprintf("MH - theta[0]: %f\n", theta[0]);

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

                double nwp_mixing_matrix[3];
                double MHp_mixing_matrix[3];

                int Cov_types[2];
                int Num_Cov_type[2];

                ModelTerm *mtp2 = m->termarray;
                mtp2++;

                int Cov_type;

                Cov_types[0] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->toggletail) - 1]); //Minus 1 since node ids are from 1 to nnodes
                Cov_types[1] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->togglehead) - 1]);

                Num_Cov_type[0] = 0;
                Num_Cov_type[1] = 0;

                for (counter=0; counter < nwp->nnodes; counter++) {
                    Cov_type = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + counter]);
                    Num_Cov_type[Cov_type - 1] = Num_Cov_type[Cov_type - 1]  + 1;
                }

                nwp_mixing_matrix[0] = nwp->nedges - networkstatistics[1] - networkstatistics[2];
                nwp_mixing_matrix[1] = networkstatistics[1];
                nwp_mixing_matrix[2] = networkstatistics[2];

                MHp_mixing_matrix[0] = MHp_nedges - networkstatistics[1] - networkstatistics[2] - m->workspace[1] - m->workspace[2];
                MHp_mixing_matrix[1] = networkstatistics[1] + m->workspace[1];
                MHp_mixing_matrix[2] = networkstatistics[2] + m->workspace[2];

//g -> g2
                if (nwp->nedges < MHp_nedges) { //Add Edge
                    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
                        prob_g_g2 = calcCNR(Num_Cov_type[0],2) - nwp_mixing_matrix[0];
                    }
                    if ( ((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1)) ) {
                        prob_g_g2 = Num_Cov_type[0]*Num_Cov_type[1]- nwp_mixing_matrix[1];
                    }
                    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
                        prob_g_g2 = calcCNR(Num_Cov_type[1],2) - nwp_mixing_matrix[2];
                    }
                } else { //Remove Edge
                    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
                        prob_g_g2 = nwp_mixing_matrix[0];
                    }
                    if ( ((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1)) ) {
                        prob_g_g2 = nwp_mixing_matrix[1];
                    }
                    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
                        prob_g_g2 = nwp_mixing_matrix[2];
                    }
                }

// g2 -> g

                if (nwp->nedges > MHp_nedges) { //Add Edge
                    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
                        prob_g2_g = calcCNR(Num_Cov_type[0],2) - MHp_mixing_matrix[0];
                    }
                    if ( ((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1)) ) {
                        prob_g2_g = Num_Cov_type[0]*Num_Cov_type[1]- MHp_mixing_matrix[1];
                    }
                    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
                        prob_g2_g = calcCNR(Num_Cov_type[1],2) - MHp_mixing_matrix[2];
                    }
                } else { //Remove Edge
                    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
                        prob_g2_g = MHp_mixing_matrix[0];
                    }
                    if ( ((Cov_types[0] == 1) && (Cov_types[1] == 2)) || ((Cov_types[1] == 2) && (Cov_types[0] == 1)) ) {
                        prob_g2_g = MHp_mixing_matrix[1];
                    }
                    if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
                        prob_g2_g = MHp_mixing_matrix[2];
                    }
                }

//Rprintf("Prob g: %f g2: %f\n",prob_g_g2, prob_g2_g);

//              if ((prob_type[0] == 0) && (prob_type[1] == 1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] == 1)){

                int num_mixing_terms = 3;

                double nwp_mu_diff[num_mixing_terms];
                double MHp_mu_diff[num_mixing_terms];

                int Num_Cov_type_dem[num_mixing_terms];

                Num_Cov_type_dem[0] = Num_Cov_type[0];
                Num_Cov_type_dem[1] = Num_Cov_type[0];
                Num_Cov_type_dem[2] = Num_Cov_type[1];

                for (counter = 0; counter < num_mixing_terms; counter++) {
                    nwp_mu_diff[counter] = (double)(nwp_mixing_matrix[counter])/(1.0*Num_Cov_type_dem[counter]) - meanvalues[counter];
                    MHp_mu_diff[counter] = (double)(MHp_mixing_matrix[counter])/(1.0*Num_Cov_type_dem[counter]) - meanvalues[counter];
                }

                double nwp_intermediate_mat[num_mixing_terms];
                double MHp_intermediate_mat[num_mixing_terms];

                counter2 = -1;
                for (counter = 0; counter < (num_mixing_terms) * (num_mixing_terms); counter++) {
                    counter1 = counter % (num_mixing_terms);
                    if (counter1 == 0) {
                        counter2 += 1;
                        nwp_intermediate_mat[counter2] = 0;
                        MHp_intermediate_mat[counter2] = 0;
                    }
                    nwp_intermediate_mat[counter2] += nwp_mu_diff[counter1]*varvalues[counter];
                    MHp_intermediate_mat[counter2] += MHp_mu_diff[counter1]*varvalues[counter];
                }

                pdf_gaussian_nwp = 0;
                pdf_gaussian_MHp = 0;
                for (counter = 0; counter < num_mixing_terms; counter++) {
                    pdf_gaussian_nwp += nwp_intermediate_mat[counter] * nwp_mu_diff[counter];
                    pdf_gaussian_MHp += MHp_intermediate_mat[counter] * MHp_mu_diff[counter];
                }
                 pdf_gaussian_nwp =  -.5 * pdf_gaussian_nwp;
                 pdf_gaussian_MHp =  -.5 * pdf_gaussian_MHp;

//Rprintf("pdf_nwp %f pdf_MHp %f\n",pdf_gaussian_nwp, pdf_gaussian_MHp);
//            }


//NetworkForecast: BEGIN//

/* REMOVED*/

 //NetworkForecast: END//

            }

///MIXING MATRIX: END///

            if ((prob_type[0] >= 1) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                int num_deg_stats = m->n_stats-1-(*NetworkForecast);
                int Deg_Add[2];
                int Deg_Delete[2];
                int Deg_Add_counter = 0;
                int Deg_Delete_counter = 0;
                int MHp_Deg_Distr[num_deg_stats];
                int nwp_Deg_Distr[num_deg_stats];
                int MHp_Deg_Distr_Edges[num_deg_stats];
                int nwp_Deg_Distr_Edges[num_deg_stats];
                double nwp_exp_dmm;
                double MHp_exp_dmm;
                int Proposal_prob_zero = 0;

                /* Get Degree of tail and head*/
                double changestat_sum = 0;

                 for (counter = 1; counter < (num_deg_stats+1); counter++) {
                     changestat_sum += m->workspace[counter];

                     if (round(m->workspace[counter]) == -1) {
                         Deg_Delete[Deg_Delete_counter] = counter-1;
                         Deg_Delete_counter++;
                     }
                     if (round(m->workspace[counter]) == 1) {
                         Deg_Add[Deg_Add_counter] = counter-1;
                         Deg_Add_counter++;
                     }
                     if (round(m->workspace[counter]) == -2) {
                         Deg_Delete[0] = counter-1;
                         Deg_Delete[1] = counter-1;
                     }
                     if (round(m->workspace[counter]) == 2) {
                         Deg_Add[0] = counter-1;
                         Deg_Add[1] = counter-1;
                     }
                 }
                 if (Deg_Add_counter == 1) {
                     Deg_Delete[1] = (int)(((Deg_Delete[0] + Deg_Add[0]) * .5) + .5);
                     Deg_Add[1] = Deg_Delete[1];
                 }
                if (changestat_sum < -.1)  { /* changesum is negative - some edge was added passed observed degrees */
                    Proposal_prob_zero = 1;
                }

                if (Proposal_prob_zero == 1) {
                    prob_g2_g = 1;
                    pdf_gaussian_MHp = log(0);
                    prob_g_g2 = 1;
                    pdf_gaussian_nwp = 0;
                } else {
                  calc_f_degdist(num_deg_stats, m, networkstatistics, nwp->nedges, MHp_nedges,
                                 Deg_Delete, Deg_Add, &prob_g_g2, &prob_g2_g,
                                 nwp_Deg_Distr, MHp_Deg_Distr);

                  calc_probs_degdist(num_deg_stats, nwp, prob_type, meanvalues, varvalues,
                                     nwp_Deg_Distr, MHp_Deg_Distr, &pdf_gaussian_nwp, &pdf_gaussian_MHp);

                /* BEGIN: NETWORK STABILITY CODE*/
                //NetworkForecast: BEGIN//
                /* REMOVED*/
                //NetworkForecast: END//
                /* END--: NETWORK STABILITY CODE*/
            }
           }

/* Two Degree Distributions and Mixing */
           if ((prob_type[0] >= 1) && (prob_type[1] >= 1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                int num_deg_stats = m->n_stats-1-(*NetworkForecast) - 2; //The four is for number of mixing - only coded for two node types
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

                    nwp_mixing_matrix[0] = nwp->nedges - networkstatistics[2*length_deg_dist + 1] - networkstatistics[2*length_deg_dist + 2];
                    nwp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 1];
                    nwp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 2];

                    MHp_mixing_matrix[0] = MHp_nedges - networkstatistics[2*length_deg_dist + 1] - networkstatistics[2*length_deg_dist + 2] - m->workspace[2*length_deg_dist + 1] - m->workspace[2*length_deg_dist + 2];
                    MHp_mixing_matrix[1] = networkstatistics[2*length_deg_dist + 1] + m->workspace[2*length_deg_dist + 1];
                    MHp_mixing_matrix[2] = networkstatistics[2*length_deg_dist + 2] + m->workspace[2*length_deg_dist + 2];

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

/////DEGREE MIXING MATRIX//////////////////////
          if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] >= 1) && (prob_type[3] >= 0) && (prob_type[4] >= 1)){

              int print_info = 0;
              int num_deg_stats = (int)((-1.0 + sqrt(1.0 + 8.0 * (m->n_stats-1))) / 2.0) + 1; //(int)(sqrt(2*(double)(m->n_stats-1))) + 1;
              if (prob_type[3] > 0) {
                  num_deg_stats = (int)((-1.0 + sqrt(1.0 + 8.0 * (m->n_stats-2))) / 2.0) + 1; //(int)(sqrt(2*(double)(m->n_stats-2))) + 1; //minus 2 for edges and clustering
              }

              int Proposal_prob_zero = 0;

              Edge nextedge=0;
 //             int deg_k;
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

if (print_info == 1) {
    Rprintf("nwp deg mixing:\n");
    for (index1=0; index1 < ((num_deg_stats)-1); index1++){
        for (index2=0; index2 <= index1; index2++){
            Rprintf(" %d ",g_dmm[index1][index2]);
        }
        Rprintf("\n");
    }

    Rprintf("nwp deg dist:\n");
    for (counter=0; counter < num_deg_stats; counter++){
       Rprintf(" %d ",deg_dist_nwp[counter]);
    }
    Rprintf("\n");
}

//Step 1: begin
              int Deg_nwp[2];
              int Deg_MHp[2];

              Deg_nwp[0] = OUT_DEG[*(MHp->toggletail)] + IN_DEG[*(MHp->toggletail)];
              Deg_nwp[1] = OUT_DEG[*(MHp->togglehead)] + IN_DEG[*(MHp->togglehead)];

if (print_info == 1) {
    Rprintf("Node ID 1 %d Node ID 2 %d\n",*(MHp->toggletail), *(MHp->togglehead));
}
              if (nwp->nedges > MHp_nedges) {
                  Deg_MHp[0] = Deg_nwp[0] - 1;
                  Deg_MHp[1] = Deg_nwp[1] - 1;
              } else {
                  Deg_MHp[0] = Deg_nwp[0] + 1;
                  Deg_MHp[1] = Deg_nwp[1] + 1;
              }
//Step 1: end

if (print_info == 1) {
    Rprintf("nwp: Degree 1 %d Degree 2 %d\n",Deg_nwp[0], Deg_nwp[1]);
    Rprintf("MHp: Degree 1 %d Degree 2 %d\n",Deg_MHp[0], Deg_MHp[1]);
}

              if ((Deg_MHp[0] > (num_deg_stats-1)) || (Deg_MHp[1] > (num_deg_stats-1))) {
                  Proposal_prob_zero = 1;
              }

              if (Proposal_prob_zero == 1) {
                  prob_g2_g = 1;
                  pdf_gaussian_MHp = log(0);
                  prob_g_g2 = 1;
                  pdf_gaussian_nwp = 0;
//Rprintf("Proposal Excesses Max Edges: %f\n", pdf_gaussian_MHp);
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
/*
Rprintf("n_i: ");
for (counter=0; counter < (num_deg_stats-1); counter++){
        Rprintf(" %d ",n_i[counter]);
}
Rprintf("\n");

Rprintf("n_j: ");
for (counter=0; counter < (num_deg_stats-1); counter++){
        Rprintf(" %d ",n_j[counter]);
}
Rprintf("\n");
*/
                  int denominator;
                  int numerator;

//Step 3a: begin - ADD
              if (nwp->nedges < MHp_nedges) {
//Rprintf("g to g2: Step 3a \n");
                  if (Deg_nwp[0] == Deg_nwp[1]) {
                      prob_g_g2 = deg_dist_nwp[Deg_nwp[0]] * (deg_dist_nwp[Deg_nwp[0]] - 1) * .5;
                  } else {
                      prob_g_g2 = deg_dist_nwp[Deg_nwp[0]] * (deg_dist_nwp[Deg_nwp[1]]);
                  }
                  if (Deg_nwp[0] > 0 && Deg_nwp[1] > 0 ) {
                      prob_g_g2 = prob_g_g2 - g_dmm[Deg_nwp[0]-1][Deg_nwp[1]-1];
                  }
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);


                for (index1=0; index1 < ((num_deg_stats)-1); index1++){
                        g_dmm[index1][index1] = 2 * g_dmm[index1][index1];
                }

//Step 3b: begin - ADD identical degrees

                  numerator = 1;

                  if ((Deg_nwp[0] == Deg_nwp[1]) && (Deg_nwp[0] > 0)) {
//Rprintf("g to g2: Step 3b \n");
                      denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0]), (Deg_nwp[0] + Deg_nwp[1]));
                      for (counter=0; counter < (num_deg_stats-1); counter++) {
                          numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter] + n_j[counter]));
                      }
                      prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);
                  } else {
//Step 3c: begin - ADD degree Node 1

                      if (Deg_nwp[0] > 0) {
//Rprintf("g to g2: Step 3c \n");
                          denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0]), (Deg_nwp[0]));
                          numerator = 1;
                          for (counter=0; counter < (num_deg_stats-1); counter++) {
                                numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter]));
                          }
                          prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);
                      }
//Step 3d: begin - ADD degree Node 2
                      if (Deg_nwp[1] > 0) {
//Rprintf("g to g2: Step 3d \n");
                          denominator = calcCNR( (deg_dist_nwp[Deg_nwp[1]] * Deg_nwp[1]), (Deg_nwp[1]));
                          numerator = 1;
                          for (counter=0; counter < (num_deg_stats-1); counter++) {
                                numerator = numerator * calcCNR(g_dmm[Deg_nwp[1]-1][counter], (n_j[counter]));
                          }
                          prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);
                      }
                  }
              } else {
//Step 4a: begin - Remove
//Rprintf("g to g2: Step 4a \n");
                  prob_g_g2 = g_dmm[Deg_nwp[0]-1][Deg_nwp[1]-1];
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);

                for (index1=0; index1 < ((num_deg_stats)-1); index1++){
                        g_dmm[index1][index1] = 2 * g_dmm[index1][index1];
                }

//Step 4b: begin - ADD identical degrees
                  numerator = 1;

                  if ((Deg_nwp[0] == Deg_nwp[1])) {
//Rprintf("g to g2: Step 4b \n");
                      denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0] -1 ), (Deg_nwp[0] + Deg_nwp[1] - 2));
                      for (counter=0; counter < (num_deg_stats-1); counter++) {
                          if (counter == (Deg_nwp[0]-1)) {
                              numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter]-1, (n_i[counter] + n_j[counter] - 2));
                          } else {
                              numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter] + n_j[counter]));
                          }
                      }
                      prob_g_g2 = prob_g_g2 * ((double)numerator / denominator);
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);
                  } else {
//Step 4c: begin - ADD node 1 degrees
//Rprintf("g to g2: Step 4c \n");
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
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);
//Step 4d: begin - ADD node 2 degrees
//Rprintf("g to g2: Step 4d \n");
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
//Rprintf("Prob_g_g2: %f\n", prob_g_g2);

                  }
              }

////g2 -> g
if (print_info == 1) {
    Rprintf("Entering: g2 to g \n");
}
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

if (print_info == 1) {
    Rprintf("MHp deg mixing:\n");
    for (index1=0; index1 < ((num_deg_stats)-1); index1++){
        for (index2=0; index2 <= index1; index2++){
            Rprintf(" %d ",g2_dmm[index1][index2]);
        }
        Rprintf("\n");
    }

    Rprintf("MHp deg dist:\n");
    for (counter=0; counter < num_deg_stats; counter++){
        Rprintf(" %d ",deg_dist_MHp[counter]);
    }
    Rprintf("\n");
}

              if (nwp->nedges < MHp_nedges) { //an edge was added
                  n_i[Deg_MHp[1]-1]++;
                  n_j[Deg_MHp[0]-1]++;
              } else { //an edge was removed
                  n_i[Deg_nwp[1]-1]--;
                  n_j[Deg_nwp[0]-1]--;
              }
/*
Rprintf("n_i: ");
for (counter=0; counter < (num_deg_stats-1); counter++){
        Rprintf(" %d ",n_i[counter]);
}
Rprintf("\n");

Rprintf("n_j: ");
for (counter=0; counter < (num_deg_stats-1); counter++){
        Rprintf(" %d ",n_j[counter]);
}
Rprintf("\n");
*/
 //Step 3a: begin - ADD

              if (nwp->nedges > MHp_nedges) {
//Rprintf("g2 to g: Step 3a \n");
                  if (Deg_nwp[0] == Deg_nwp[1]) {
                      prob_g2_g = deg_dist_MHp[Deg_MHp[0]] * (deg_dist_nwp[Deg_MHp[0]] - 1) * .5;
                  } else {
                      prob_g2_g = deg_dist_MHp[Deg_MHp[0]] * (deg_dist_nwp[Deg_MHp[1]]);
                  }
                  if (Deg_MHp[0] > 0 && Deg_MHp[1] > 0 ) {
                      prob_g2_g = prob_g2_g - g2_dmm[Deg_MHp[0]-1][Deg_MHp[1]-1];
                  }
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);

                for (index1=0; index1 < ((num_deg_stats)-1); index1++){
                        g2_dmm[index1][index1] = 2 * g2_dmm[index1][index1];
                }

//Step 3b: begin - ADD identical degrees

                  numerator = 1;

                  if ((Deg_MHp[0] == Deg_MHp[1]) && (Deg_MHp[0] > 0)) {
//Rprintf("g2 to g: Step 3b \n");
                      denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0]), (Deg_MHp[0] + Deg_MHp[1]));
                      for (counter=0; counter < (num_deg_stats-1); counter++) {
                          numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter] + n_j[counter]));
                      }
                      prob_g2_g = prob_g2_g * ((double)numerator / denominator);
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);
                  } else {
//Step 3c: begin - ADD degree Node 1

                      if (Deg_MHp[0] > 0) {
//Rprintf("g2 to g: Step 3c \n");
                          denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0]), (Deg_MHp[0]));
                          numerator = 1;
                          for (counter=0; counter < (num_deg_stats-1); counter++) {
                                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter]));
                          }
                          prob_g2_g = prob_g2_g * ((double)numerator / denominator);
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);
                      }
//Step 3d: begin - ADD degree Node 2
                      if (Deg_MHp[1] > 0) {
//Rprintf("g2 to g: Step 3d \n");
                          denominator = calcCNR( (deg_dist_MHp[Deg_MHp[1]] * Deg_MHp[1]), (Deg_MHp[1]));
                          numerator = 1;
                          for (counter=0; counter < (num_deg_stats-1); counter++) {
                                numerator = numerator * calcCNR(g2_dmm[Deg_MHp[1]-1][counter], (n_j[counter]));
                          }
                          prob_g2_g = prob_g2_g * ((double)numerator / denominator);
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);
                      }
                  }
              } else {
//Step 4a: begin - Remove
//Rprintf("g2 to g: Step 4a \n");
                  prob_g2_g = g2_dmm[Deg_MHp[0]-1][Deg_MHp[1]-1];
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);
//Step 4b: begin - ADD identical degrees

                for (index1=0; index1 < ((num_deg_stats)-1); index1++){
                        g2_dmm[index1][index1] = 2 * g2_dmm[index1][index1];
                }

                  numerator = 1;

                  if ((Deg_MHp[0] == Deg_MHp[1])) {
//Rprintf("g2 to g: Step 4b \n");
                      denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0] -1 ), (Deg_MHp[0] + Deg_MHp[1] - 2));
                      for (counter=0; counter < (num_deg_stats-1); counter++) {
                          if (counter == (Deg_MHp[0]-1)) {
                              numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter]-1, (n_i[counter] + n_j[counter] - 2));
                          } else {
                              numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter] + n_j[counter]));
                          }
                      }
                      prob_g2_g = prob_g2_g * ((double)numerator / denominator);
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);
                  } else {
//Step 4c: begin - ADD node 1 degrees
//Rprintf("g2 to g: Step 4c \n");
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
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);
                          //Step 4d: begin - ADD node 2 degrees
//Rprintf("g2 to g: Step 4d \n");
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
//Rprintf("Prob_g2_g: %f\n", prob_g2_g);
                  }
              }

 ///Calculate probs

//Rprintf("Probs FINAL %f %f\n",prob_g_g2, prob_g2_g);

              if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 1) && (prob_type[3] >= 0) && (prob_type[4] == 1)){

                double nwp_mu_diff[m->n_stats-1];
                double MHp_mu_diff[m->n_stats-1];

                for (counter = 0; counter < m->n_stats-1; counter++) {
                    nwp_mu_diff[counter] = (double)(networkstatistics[counter+1]) - meanvalues[counter];
                    MHp_mu_diff[counter] = (double)(networkstatistics[counter+1] + m->workspace[counter+1]) - meanvalues[counter];
                }
if (print_info == 1) {
    Rprintf("n_stats %d: ", m->n_stats);
  Rprintf("\n");
    Rprintf("Mean Values: ");
    for (counter=0; counter < ((m->n_stats)-1); counter++){
        Rprintf(" %f ",meanvalues[counter]);
    }
    Rprintf("\n");

    Rprintf("networkstatistics NWP: ");
    for (counter=0; counter < ((m->n_stats)-1); counter++){
      Rprintf(" %f ",networkstatistics[counter+1]);
    }
    Rprintf("\n");
    
    Rprintf("networkstatistics MHP: ");
    for (counter=0; counter < ((m->n_stats)-1); counter++){
      Rprintf(" %f ",networkstatistics[counter+1] + m->workspace[counter+1]);
    }
    Rprintf("\n");
    
Rprintf("nwp mu diff: ");
for (counter=0; counter < ((m->n_stats)-1); counter++){
        Rprintf(" %f ",nwp_mu_diff[counter]);
}
Rprintf("\n");

Rprintf("MHp mu diff: ");
for (counter=0; counter < ((m->n_stats)-1); counter++){
        Rprintf(" %f ",MHp_mu_diff[counter]);
}
Rprintf("\n");
}

                double nwp_intermediate_mat[m->n_stats-1];
                double MHp_intermediate_mat[m->n_stats-1];

                counter2 = -1;
                for (counter = 0; counter < (m->n_stats-1) * (m->n_stats-1); counter++) {
                    counter1 = counter % (m->n_stats-1);
                    if (counter1 == 0) {
                        counter2 += 1;
                        nwp_intermediate_mat[counter2] = 0;
                        MHp_intermediate_mat[counter2] = 0;
                    }
                    nwp_intermediate_mat[counter2] += nwp_mu_diff[counter1]*varvalues[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mat[counter2] += MHp_mu_diff[counter1]*varvalues[counter];
                }
/*
Rprintf("Var Values: ");
for (counter=0; counter < (((m->n_stats)-1) * ((m->n_stats)-1)); counter++){
    if ((counter % ((m->n_stats)-1)) == 0) {
        Rprintf("\n");
    }
        Rprintf(" %f ",varvalues[counter]);
}
Rprintf("\n");

Rprintf("nwp Intermediate: ");
for (counter=0; counter < ((m->n_stats)-1); counter++){
        Rprintf(" %f ",nwp_intermediate_mat[counter]);
}
Rprintf("\n");

Rprintf("MHp Intermediate: ");
for (counter=0; counter < ((m->n_stats)-1); counter++){
        Rprintf(" %f ",MHp_intermediate_mat[counter]);
}
Rprintf("\n");
*/
                pdf_gaussian_nwp = 0;
                pdf_gaussian_MHp = 0;
                for (counter = 0; counter < (m->n_stats-1); counter++) {
                    pdf_gaussian_nwp += nwp_intermediate_mat[counter] * nwp_mu_diff[counter];
                    pdf_gaussian_MHp += MHp_intermediate_mat[counter] * MHp_mu_diff[counter];
                }
                 pdf_gaussian_nwp =  -.5 * pdf_gaussian_nwp;
                 pdf_gaussian_MHp =  -.5 * pdf_gaussian_MHp;

//Rprintf("pdf_nwp %f pdf_MHp %f\n",pdf_gaussian_nwp, pdf_gaussian_MHp);
            }


              /////CLUSTERING//////////////////////

              if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] >= 1) && (prob_type[3] >= 1) && (prob_type[4] >= 1)){

                for (index1=0; index1 < ((num_deg_stats)-1); index1++){ //Undo what was done for degree mixing
                        g_dmm[index1][index1] = .5 * g_dmm[index1][index1];
                        g2_dmm[index1][index1] = .5 * g2_dmm[index1][index1];
                }

                double num_Tri = networkstatistics[m->n_stats-1]; //should be last statistic
                double num_Tri_change = abs(m->workspace[m->n_stats-1]); //should be last statistic

if (print_info == 1) {
    Rprintf("Triangles %f Change %f\n",num_Tri, num_Tri_change);
}

                ///  g->g2
                if (nwp->nedges < MHp_nedges) { //edge was added

                  double pa = 0;
                  double pa_num = -3*num_Tri;
                  for (counter = 2; counter < (num_deg_stats); counter++) { //counter is deg of node
                      pa_num += calcCNR(counter, 2)*deg_dist_nwp[counter];
                  }

                  double pa_dem = 0.0;
                  for (index1 = 1; index1 < (num_deg_stats); index1++) { //counter is deg of node
                    for (index2 = index1; index2 < (num_deg_stats); index2++) { //counter is deg of node
                      if (index1 == index2){
                          pa_dem += index1*index2*(deg_dist_nwp[index1]*(deg_dist_nwp[index2]-1)*.5 - g_dmm[index1-1][index2-1]);
                      } else {
                          pa_dem += index1*index2*(deg_dist_nwp[index1]*deg_dist_nwp[index2] - g_dmm[index1-1][index2-1]);
                      }
                    }
                  }

if (print_info == 1) {
    Rprintf("pa_num %f pa_dem %f\n",pa_num, pa_dem);
}
                  pa = pa_num / pa_dem;

                  int p_add_1 = 1;
                  int frac_k = 1;
                  double p_add = 0;

                  for (counter = 0; counter < (num_Tri_change); counter++) { //counter is deg of node
                      frac_k = frac_k * (counter+1);
                      p_add_1 = p_add_1 * (Deg_nwp[0]-counter) * (Deg_nwp[1]-counter);
                  }
                  p_add = (1/frac_k) * p_add_1 * pow(pa, num_Tri_change) * pow((1-pa), (Deg_nwp[0]-num_Tri_change) * (Deg_nwp[1]-num_Tri_change));

if (print_info == 1) {
    Rprintf("pa %f p_add %f\n",pa, p_add);
}
                  prob_g_g2 = prob_g_g2 * p_add;

                } else { //edge was removed

                  double pb = 0;
                  int pb_num = 3*num_Tri;

                  double pb_dem = 0.0;
                  for (index1 = 1; index1 < (num_deg_stats); index1++) { //counter is deg of node - must have degree greater than 1
                    for (index2 = index1; index2 < (num_deg_stats); index2++) { //counter is deg of node - must have degree greater than 1
                       pb_dem += (index1-1)*(index2-1)*(g_dmm[index1-1][index2-1]);
                    }
                  }

if (print_info == 1) {
    Rprintf("pb_num %d pb_dem %f\n",pb_num, pb_dem);
}
                  pb = pb_num / pb_dem;

                  int p_remove_1 = 1;
                  int frac_k = 1;
                  double p_remove = 0;

                  for (counter = 0; counter < (num_Tri_change); counter++) { //counter is deg of node
                      frac_k = frac_k * (counter+1);
                      p_remove_1 = p_remove_1 * (Deg_nwp[0]-1-counter) * (Deg_nwp[1]-1-counter);
                  }
                  p_remove = frac_k * p_remove_1 * pow(pb, num_Tri_change) * pow((1-pb), (Deg_nwp[0]-1-num_Tri_change) * (Deg_nwp[1]-1-num_Tri_change));

if (print_info == 1) {
    Rprintf("pb %f p_remove %f\n",pb, p_remove);
}
                  prob_g_g2 = prob_g_g2 * p_remove;
                }

                ///  g2->g

                num_Tri = networkstatistics[m->n_stats-1] + m->workspace[m->n_stats-1]; //since the deg dist starts at 0

                if (nwp->nedges > MHp_nedges) { //edge was added from g2 to g

                  double pa = 0;
                  double pa_num = -3*num_Tri;
                  for (counter = 2; counter < (num_deg_stats); counter++) { //counter is deg of node
                      pa_num += calcCNR(counter, 2)*deg_dist_MHp[counter];
                  }

                  double pa_dem = 0.0;
                  for (index1 = 1; index1 < (num_deg_stats); index1++) { //counter is deg of node
                    for (index2 = index1; index2 < (num_deg_stats); index2++) { //counter is deg of node
                      if (index1 == index2){
                          pa_dem += index1*index2*(deg_dist_MHp[index1]*(deg_dist_MHp[index2]-1)*.5 - g2_dmm[index1-1][index2-1]);
                      } else {
                          pa_dem += index1*index2*(deg_dist_MHp[index1]*deg_dist_MHp[index2] - g2_dmm[index1-1][index2-1]);
                      }
                    }
                  }

if (print_info == 1) {
  Rprintf("g2->g - pa_num %f pa_dem %f\n",pa_num, pa_dem);
}

                  pa = pa_num / pa_dem;

                  int p_add_1 = 1;
                  int frac_k = 1;
                  double p_add = 0;

                  for (counter = 0; counter < (num_Tri_change); counter++) { //counter is deg of node
                      frac_k = frac_k * (counter+1);
                      p_add_1 = p_add_1 * (Deg_MHp[0]-counter) * (Deg_MHp[1]-counter);
                  }
                  p_add = frac_k * p_add_1 * pow(pa, num_Tri_change) * pow((1-pa), (Deg_MHp[0]-num_Tri_change) * (Deg_MHp[1]-num_Tri_change));
                  prob_g2_g = prob_g2_g * p_add;

if (print_info == 1) {
  Rprintf("g2->g - pa %f p_add %f\n",pa, p_add);
}

                } else { //edge was removed

                  double pb = 0;
                  int pb_num = 3*num_Tri;

                  double pb_dem = 0.0;
                  for (index1 = 1; index1 < (num_deg_stats); index1++) { //counter is deg of node - must have degree greater than 1
                    for (index2 = index1; index2 < (num_deg_stats); index2++) { //counter is deg of node - must have degree greater than 1
                       pb_dem += (index1-1)*(index2-1)*(g2_dmm[index1-1][index2-1]);
                    }
                  }

if (print_info == 1) {
  Rprintf("g2->g - pb_num %d pb_dem %f\n",pb_num, pb_dem);
}

                  pb = pb_num / pb_dem;

                  int p_remove_1 = 1;
                  int frac_k = 1;
                  double p_remove = 0;

                  for (counter = 0; counter < (num_Tri_change); counter++) { //counter is deg of node
                      frac_k = frac_k * (counter+1);
                      p_remove_1 = p_remove_1 * (Deg_MHp[0]-1-counter) * (Deg_MHp[1]-1-counter);
                  }
                  p_remove = (1/frac_k) * p_remove_1 * pow(pb, num_Tri_change) * pow((1-pb), (Deg_MHp[0]-1-num_Tri_change) * (Deg_MHp[1]-1-num_Tri_change));

if (print_info == 1) {
  Rprintf("g2->g - pb %f p_remove %f\n",pb, p_remove);
}

                  prob_g2_g = prob_g2_g * p_remove;
                }


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
      for (unsigned int i = 0; i < m->n_stats; i++){
	networkstatistics[i] += m->workspace[i];
      }
      taken++;
    }
  }
  *staken = taken;
  return MCMC_OK;
 //END of CODE FROM ERGM Library
}
