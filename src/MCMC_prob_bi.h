/* 
 *  File CCMnet/src/MCMC_prob.h
 *
 *  Sections of this code are derived from the ergm library
 *  All such sections are noted and attributed to the statnet development team. 
 */


MCMCStatus MetropolisHastings_bi(MHproposal *MHp,
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

MCMCStatus MetropolisHastings_bi(MHproposal *MHp,
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
    for (int counter_print=0; counter_print <= ((m->n_stats)-1); counter_print++){
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
   for (int counter_print=0; counter_print <= ((m->n_stats)-1); counter_print++){
        Rprintf(" %f ",m->workspace[counter_print]);        
   }
   Rprintf("\n");
}
          
        int MHp_nedges;
        double prob_g2_g = 1;
        double prob_g_g2 = 1;
        int total_max_edges = (nwp->nnodes - nwp->bipartite) * (nwp->bipartite) + .5;
        double cutoff=log(0);
        int counter;
        int counter1;
        int counter2;
        
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

/////////////////////////////


            if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
//Rprintf("Entered Density: MH Code \n"); 
//Rprintf("Density Probs: MH Code %f %f\n",meanvalues[0], varvalues[0]);
                
                double nwp_density;
                double MHp_density;
        
                if (nwp->nedges > MHp_nedges) {
                        prob_g_g2 = networkstatistics[0];
                } else {
                        prob_g_g2 = total_max_edges  - networkstatistics[0];
                }
                if (nwp->nedges < MHp_nedges) {
                        prob_g2_g = (double)MHp_nedges;
                } else {
                        prob_g2_g = (double)(total_max_edges  - MHp_nedges);
                }
                
                nwp_density  = networkstatistics[0]/total_max_edges;
                MHp_density  = (double)((double)MHp_nedges)/total_max_edges;

                if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] == 1)){
                    pdf_gaussian_nwp = -0.5 * pow( (nwp_density-meanvalues[0]), 2.0 )/varvalues[0];
                    pdf_gaussian_MHp = -0.5 * pow( (MHp_density-meanvalues[0]), 2.0 )/varvalues[0];
                }
                if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] == 2)){ //log transform
                    pdf_gaussian_nwp = -0.5 * pow( (log(nwp_density)-meanvalues[0]), 2.0 )/varvalues[0];
                    pdf_gaussian_MHp = -0.5 * pow( (log(MHp_density)-meanvalues[0]), 2.0 )/varvalues[0];
                }
                if ((prob_type[0] == 0) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] == 99)){ //Non-parametric
                    pdf_gaussian_nwp = log(meanvalues[(int)networkstatistics[0]]);
                    pdf_gaussian_MHp = log(meanvalues[(int)MHp_nedges]);


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
                    
if (print_info_MH == 1) {
  Rprintf("Mean Probs %f %f \n",meanvalues[(int)networkstatistics[0]], meanvalues[(int)MHp_nedges]);
  Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                    
                }
                /* BEGIN: NETWORK STABILITY CODE*/

                /* REMOVED */
                
                /* END--: NETWORK STABILITY CODE*/                
                
//Rprintf("Inputs %f %f\n",meanvalues[0], varvalues[0]);
//Rprintf("Probs %f %d %f %f %f %f\n",networkstatistics[0], MHp_nedges, prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
            }              

/////////////////////////////
              
            if ((prob_type[0] >= 1) && (prob_type[1] >= 1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                int Deg_nwp[2];
                int Deg_MHp[2];
                
                Deg_nwp[0] = OUT_DEG[*(MHp->toggletail)] + IN_DEG[*(MHp->toggletail)];
                Deg_nwp[1] = OUT_DEG[*(MHp->togglehead)] + IN_DEG[*(MHp->togglehead)];

if (print_info_MH == 1) {
Rprintf("Node ID 1 %d Node ID 2 %d\n",*(MHp->toggletail), *(MHp->togglehead));                 
Rprintf("NWP Degree 1 %d Degree 2 %d\n",Deg_nwp[0], Deg_nwp[1]);                 
}

                if (nwp->nedges > MHp_nedges) {
                    Deg_MHp[0] = Deg_nwp[0] - 1;
                    Deg_MHp[1] = Deg_nwp[1] - 1;
                } else {
                    Deg_MHp[0] = Deg_nwp[0] + 1;
                    Deg_MHp[1] = Deg_nwp[1] + 1;                    
                }
                
if (print_info_MH == 1) {
    Rprintf("MHP Degree 1 %d Degree 2 %d\n",Deg_MHp[0], Deg_MHp[1]);
}
    
//Rprintf("Max degree: %d \n",*maxdegree);
                
            if ((Deg_nwp[0] > *maxdegree) || (Deg_nwp[1] > *maxdegree) || (Deg_MHp[0] > *maxdegree) || (Deg_MHp[1] > *maxdegree) ) {
               prob_g2_g = 1;
               pdf_gaussian_MHp = log(0);
               prob_g_g2 = 1;
               pdf_gaussian_nwp = 0;
//Rprintf("Proposal Excesses Max Edges: %f\n", pdf_gaussian_MHp);
            } else {
                    
//Figure out number of covariate patterns - Should be easier!

                ModelTerm *mtp2 = m->termarray;                     
                mtp2++;
if (print_info_MH == 1) {
        Rprintf("Term 2: "); 
        for (counter=0; counter < mtp2->ninputparams; counter++){
                Rprintf(" %d ",(int)round(mtp2->inputparams[counter]));        
        }
        Rprintf("\n");  
}
                int *CovPattern = malloc(nwp->nnodes * sizeof(int));
                
                for (counter=0; counter < nwp->bipartite; counter++){
                        CovPattern[counter] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->bipartite + counter]);        
                }

                int max_degree = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->bipartite - 2]);
                
                mtp2++;  

if (print_info_MH == 1) {
Rprintf("Term 3: "); 
for (counter=0; counter < mtp2->ninputparams; counter++){
        Rprintf(" %d ",(int)round(mtp2->inputparams[counter]));        
}
Rprintf("\n");  
}
                
                for (counter=0; counter < (nwp->nnodes - nwp->bipartite); counter++){
                        CovPattern[counter + nwp->bipartite] = (int)round(mtp2->inputparams[mtp2->ninputparams - (nwp->nnodes - nwp->bipartite) + counter]);        
                }

                mtp2++;  

if (print_info_MH == 1) {                
Rprintf("Term 4: "); 
for (counter=0; counter < mtp2->ninputparams; counter++){
        Rprintf(" %d ",(int)round(mtp2->inputparams[counter]));        
}
Rprintf("\n");  

                

Rprintf("Max Degree %d \n",max_degree);
                

Rprintf("CovPattern: "); 
for (counter=0; counter < nwp->nnodes; counter++){
        Rprintf(" %d ", CovPattern[counter]);        
}
Rprintf("\n");  
}
             
                int max_Cov_type = 0;
                int Cov_type;
                for (counter=0; counter < nwp->nnodes; counter++) {
                    Cov_type = CovPattern[counter];
                    if (max_Cov_type < Cov_type) {
                        max_Cov_type = Cov_type;
                    }
                }

                max_Cov_type = 2 * max_Cov_type; //for type 1 nodes + type 2 nodes 
                        
                int *Num_Cov_type;
                Num_Cov_type = malloc(max_Cov_type * sizeof(int));
                //int Num_Cov_type[6]; //max_Cov_type] = 0;
                
                for (counter=0; counter < max_Cov_type; counter++){
                    Num_Cov_type[counter] = 0;        
                }
                
                for (counter=0; counter < nwp->bipartite; counter++) {
                    Cov_type = CovPattern[counter];
                    Num_Cov_type[Cov_type - 1] = Num_Cov_type[Cov_type - 1]  + 1;
                }

                for (counter=nwp->bipartite; counter < nwp->nnodes; counter++) {
                    Cov_type = CovPattern[counter];
                    Num_Cov_type[Cov_type - 1 + (int)round(max_Cov_type/2)] = Num_Cov_type[Cov_type - 1 + (int)round(max_Cov_type/2)]  + 1;
                }
                
if (print_info_MH == 1) {
Rprintf("CovPattern Sizes: "); 
for (counter=0; counter < max_Cov_type; counter++){
        Rprintf(" %d ",Num_Cov_type[counter]);        
}
Rprintf("\n");  
                
                
Rprintf("Stats \n: "); 
for (counter=0; counter < m->n_stats; counter++){
        Rprintf(" %d ", (int)networkstatistics[counter]);        
}
Rprintf("\n");  
}
       
                int cov_pattern[2]; //covariate pattern for each endpoint of toggled edge
                
                cov_pattern[0] = (int)round(CovPattern[*(MHp->toggletail) - 1]); //Minus 1 since node ids are from 1 to nnodes
                cov_pattern[1] = (int)round(CovPattern[*(MHp->togglehead) - 1]); //Minus 1 since node ids are from 1 to nnodes

if (print_info_MH == 1) {
    Rprintf("Num Cov Types: %d %d \n", cov_pattern[0], cov_pattern[1]); 

    Rprintf("WorkSpace (%d): ", m->n_stats); 
    for (int counter_print=0; counter_print < (m->n_stats); counter_print++){
        Rprintf(" %f ",m->workspace[counter_print]);        
    }
    Rprintf("\n");
}
                int **nwp_Deg_Distr_mat_1 = (int **)malloc((int)round(max_Cov_type/2) * sizeof(int*));
                int **nwp_Deg_Distr_mat_2 = (int **)malloc((int)round(max_Cov_type/2) * sizeof(int*));
                int **MHp_Deg_Distr_mat_1 = (int **)malloc((int)round(max_Cov_type/2) * sizeof(int*));
                int **MHp_Deg_Distr_mat_2 = (int **)malloc((int)round(max_Cov_type/2) * sizeof(int*));

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   nwp_Deg_Distr_mat_1[counter] = (int *)malloc((max_degree+1) * sizeof(int)); //+1 for degree 0
                   nwp_Deg_Distr_mat_2[counter] = (int *)malloc((max_degree+1) * sizeof(int));
                   MHp_Deg_Distr_mat_1[counter] = (int *)malloc((max_degree+1) * sizeof(int));
                   MHp_Deg_Distr_mat_2[counter] = (int *)malloc((max_degree+1) * sizeof(int));
                }

                counter = 1; //skip number of edges
                for (counter1 = 0; counter1 < max_Cov_type; counter1++) {
                   for (counter2 = 0; counter2 <= max_degree; counter2++) {
                      m->workspace[counter] = 0;
                      counter++;
                   }
                }
                m->workspace[1 + Deg_nwp[0] + (cov_pattern[0]-1)*(max_degree+1)] = -1;
                m->workspace[1 + Deg_nwp[1] + (cov_pattern[1]-1 + (int)round(max_Cov_type/2))*(max_degree+1)] = -1;
                m->workspace[1 + Deg_MHp[0] + (cov_pattern[0]-1)*(max_degree+1)] = 1;
                m->workspace[1 + Deg_MHp[1] + (cov_pattern[1]-1 + (int)round(max_Cov_type/2))*(max_degree+1)] = 1;

if (print_info_MH == 1) {                
Rprintf("WorkSpace (%d): ", m->n_stats); 
for (int counter_print=0; counter_print < (m->n_stats); counter_print++){
    Rprintf(" %f ",m->workspace[counter_print]);        
}
Rprintf("\n");                
}
                
                counter = 1; //skip number of edges
                for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                   for (counter2 = 0; counter2 <= max_degree; counter2++) {
                       nwp_Deg_Distr_mat_1[counter1][counter2] = (int)networkstatistics[counter];
                       nwp_Deg_Distr_mat_2[counter1][counter2] = (int)networkstatistics[counter + (int)round(max_Cov_type/2)*(max_degree+1)];
                      
                       MHp_Deg_Distr_mat_1[counter1][counter2] = nwp_Deg_Distr_mat_1[counter1][counter2] + (int)(m->workspace[counter]);
                       MHp_Deg_Distr_mat_2[counter1][counter2] = nwp_Deg_Distr_mat_2[counter1][counter2] + (int)(m->workspace[counter + (int)round(max_Cov_type/2)*(max_degree+1)]);

                       counter++;
                   }
                }
                
 //               MHp_Deg_Distr_mat_1[cov_pattern[0]-1][Deg_nwp[0]] = nwp_Deg_Distr_mat_1[cov_pattern[0]-1][Deg_nwp[0]] - 1;
 //               MHp_Deg_Distr_mat_2[cov_pattern[1]-1][Deg_nwp[1]] = nwp_Deg_Distr_mat_2[cov_pattern[1]-1][Deg_nwp[1]] - 1;
 //               MHp_Deg_Distr_mat_1[cov_pattern[0]-1][Deg_MHp[0]] = nwp_Deg_Distr_mat_1[cov_pattern[0]-1][Deg_MHp[0]] + 1;
 //               MHp_Deg_Distr_mat_2[cov_pattern[1]-1][Deg_MHp[1]] = nwp_Deg_Distr_mat_2[cov_pattern[1]-1][Deg_MHp[1]] + 1;

if (print_info_MH == 1) {                
Rprintf("NWP Degree Distribution (Type 1): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %d ", nwp_Deg_Distr_mat_1[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");    


Rprintf("NWP Degree Distribution (Type 2): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %d ", nwp_Deg_Distr_mat_2[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");  


Rprintf("MHP Degree Distribution (Type 1): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %d ", MHp_Deg_Distr_mat_1[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");  


Rprintf("MHP Degree Distribution (Type 2): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %d ", MHp_Deg_Distr_mat_2[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");  
}
                
                int **nwp_mixing_cov = (int **)malloc((int)round(max_Cov_type/2) * sizeof(int*)); //mixing matrix entry i,j is number of type 1 cov i edges to type 2 cov j
                int **MHp_mixing_cov = (int **)malloc((int)round(max_Cov_type/2) * sizeof(int*)); //mixing matrix entry i,j is number of type 1 cov i edges to type 2 cov j

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   nwp_mixing_cov[counter] = (int *)malloc((int)round(max_Cov_type/2) * sizeof(int));
                   MHp_mixing_cov[counter] = (int *)malloc((int)round(max_Cov_type/2) * sizeof(int)); 
                }
                
                counter = 1 + (int)round(max_Cov_type/2)*(max_degree+1)*2; //skip number of edges
                for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                   for (counter2 = 0; counter2 < (int)round(max_Cov_type/2); counter2++) {
                       nwp_mixing_cov[counter2][counter1] = (int)networkstatistics[counter];
                       MHp_mixing_cov[counter2][counter1] = nwp_mixing_cov[counter2][counter1] + (int)(m->workspace[counter]);

                       counter++;
                   }
                }

if (print_info_MH == 1) {                
Rprintf("nwp Mixing: \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    for (counter2 = 0; counter2 < (int)round(max_Cov_type/2); counter2++) {
          Rprintf(" %d ", nwp_mixing_cov[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n"); 

Rprintf("MHp Mixing: \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    for (counter2 = 0; counter2 < (int)round(max_Cov_type/2); counter2++) {
          Rprintf(" %d ", MHp_mixing_cov[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n"); 
}             
                double *nwp_Deg_Distr_1;
                double *nwp_Deg_Distr_2;
                double *MHp_Deg_Distr_1;
                double *MHp_Deg_Distr_2;

                nwp_Deg_Distr_1 = malloc((max_degree+1) * sizeof(double));
                nwp_Deg_Distr_2 = malloc((max_degree+1) * sizeof(double));
                MHp_Deg_Distr_1 = malloc((max_degree+1) * sizeof(double));
                MHp_Deg_Distr_2 = malloc((max_degree+1) * sizeof(double));
                               
                for (counter=0; counter <= max_degree; counter++){
                    nwp_Deg_Distr_1[counter] = nwp_Deg_Distr_mat_1[cov_pattern[0]-1][counter];
                    nwp_Deg_Distr_2[counter] = nwp_Deg_Distr_mat_2[cov_pattern[1]-1][counter];       
                    MHp_Deg_Distr_1[counter] = MHp_Deg_Distr_mat_1[cov_pattern[0]-1][counter];       
                    MHp_Deg_Distr_2[counter] = MHp_Deg_Distr_mat_2[cov_pattern[1]-1][counter];       
                }

if (print_info_MH == 1) {                
Rprintf("NWP Degree Distr 1: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("NWP Degree Distr 2: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_2[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr 1: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr 2: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_2[counter]);        
}
Rprintf("\n");
}      

                // Construct number of edges associated with each degree correcting for mixing patterns

                int nwp_sum_mixing_1=0;
                int nwp_sum_mixing_2=0;
                int MHp_sum_mixing_1=0;
                int MHp_sum_mixing_2=0;

                for(counter=0; counter < (int)round(max_Cov_type/2); counter++) {
                    nwp_sum_mixing_1 = nwp_sum_mixing_1  + nwp_mixing_cov[cov_pattern[0]-1][counter];
                    nwp_sum_mixing_2 = nwp_sum_mixing_2  + nwp_mixing_cov[counter][cov_pattern[1]-1];
                    MHp_sum_mixing_1 = MHp_sum_mixing_1  + MHp_mixing_cov[cov_pattern[0]-1][counter];
                    MHp_sum_mixing_2 = MHp_sum_mixing_2  + MHp_mixing_cov[counter][cov_pattern[1]-1];
                }

if (print_info_MH == 1) {                
Rprintf("NWP SUM Mixing 1 %d 2 %d\n",nwp_sum_mixing_1, nwp_sum_mixing_2);   
Rprintf("MHP SUM Mixing 1 %d 2 %d\n",MHp_sum_mixing_1, MHp_sum_mixing_2); 
}
                
                double nwp_mp_1 = (double)nwp_mixing_cov[cov_pattern[0]-1][cov_pattern[1]-1]/(double)nwp_sum_mixing_1;
                double nwp_mp_2 = (double)nwp_mixing_cov[cov_pattern[0]-1][cov_pattern[1]-1]/(double)nwp_sum_mixing_2;
                double MHp_mp_1 = (double)MHp_mixing_cov[cov_pattern[0]-1][cov_pattern[1]-1]/(double)MHp_sum_mixing_1;
                double MHp_mp_2 = (double)MHp_mixing_cov[cov_pattern[0]-1][cov_pattern[1]-1]/(double)MHp_sum_mixing_2;

if (print_info_MH == 1) {                
Rprintf("NWP Mixing 1 %f 2 %f\n",nwp_mp_1, nwp_mp_2);   
Rprintf("MHP Mixing 1 %f 2 %f\n",MHp_mp_1, MHp_mp_2); 
}
                double nwp_z = 0;
                double MHp_z = 0;
                
                double *nwp_Deg_Distr_Edges_1;
                double *nwp_Deg_Distr_Edges_2;
                double *MHp_Deg_Distr_Edges_1;
                double *MHp_Deg_Distr_Edges_2;

                nwp_Deg_Distr_Edges_1 = malloc((max_degree+1) * sizeof(double));
                nwp_Deg_Distr_Edges_2 = malloc((max_degree+1) * sizeof(double));
                MHp_Deg_Distr_Edges_1 = malloc((max_degree+1) * sizeof(double));
                MHp_Deg_Distr_Edges_2 = malloc((max_degree+1) * sizeof(double));

                for (counter = 0; counter <= max_degree; counter++) {
                    nwp_Deg_Distr_Edges_1[counter] = nwp_Deg_Distr_1[counter] * (counter) * nwp_mp_1;
                    nwp_Deg_Distr_Edges_2[counter] = nwp_Deg_Distr_2[counter] * (counter) * nwp_mp_2;
                    nwp_z = nwp_z + nwp_Deg_Distr_Edges_1[counter];
                    
                    MHp_Deg_Distr_Edges_1[counter] = MHp_Deg_Distr_1[counter] * (counter) * MHp_mp_1;
                    MHp_Deg_Distr_Edges_2[counter] = MHp_Deg_Distr_2[counter] * (counter) * MHp_mp_2;
                    MHp_z = MHp_z + MHp_Deg_Distr_Edges_1[counter];
                }

if (print_info_MH == 1) {                
Rprintf("NWP Degree Distr Edges 1: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_Edges_1[counter]);        
}
Rprintf("\n");

Rprintf("NWP Degree Distr Edges 2: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_Edges_2[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr Edges 1: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_Edges_1[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr Edges 2: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_Edges_2[counter]);        
}
Rprintf("\n");                
}                
                // Construct expected degree distribution 

                double nwp_exp_dmm;
                double MHp_exp_dmm;

                nwp_exp_dmm = (nwp_Deg_Distr_Edges_1[Deg_nwp[0]] * nwp_Deg_Distr_Edges_2[Deg_nwp[1]])/ nwp_z;                      
                MHp_exp_dmm = (MHp_Deg_Distr_Edges_1[Deg_MHp[0]] * MHp_Deg_Distr_Edges_2[Deg_MHp[1]])/ MHp_z;

if (print_info_MH == 1) {                
Rprintf("NWP DMM %f MHP DMM %f\n",nwp_exp_dmm, MHp_exp_dmm);
} 

                if (nwp->nedges > MHp_nedges) {  //Edge is removed from g to g2
                        prob_g_g2 = nwp_exp_dmm;
                } else { //Edge is added from g to g2
                        prob_g_g2 = (nwp_Deg_Distr_1[Deg_nwp[0]] * nwp_Deg_Distr_2[Deg_nwp[1]])  - nwp_exp_dmm;
                }
                if (nwp->nedges < MHp_nedges) {  //Edge is removed from g2 to g
                        prob_g2_g = MHp_exp_dmm;
                } else {  //Edge is added from g2 to g
                        prob_g2_g = (double)((MHp_Deg_Distr_1[Deg_MHp[0]] * MHp_Deg_Distr_2[Deg_MHp[1]])  - MHp_exp_dmm);
                }

if (print_info_MH == 1) {
Rprintf("Moves g to g2: %f Moves g2 to g %f\n",prob_g_g2, prob_g2_g);
}
//////////////////////////////
                
           if ((prob_type[0] == 1) && (prob_type[1] == 1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
      
                double *nwp_mu_diff_1;
                double *nwp_mu_diff_2;
                double *MHp_mu_diff_1;
                double *MHp_mu_diff_2;

                nwp_mu_diff_1 = malloc((max_degree+1) * sizeof(double));
                nwp_mu_diff_2 = malloc((max_degree+1) * sizeof(double));
                MHp_mu_diff_1 = malloc((max_degree+1) * sizeof(double));
                MHp_mu_diff_2 = malloc((max_degree+1) * sizeof(double));
    
                double *nwp_mu_diff_mixing;
                double *MHp_mu_diff_mixing;

                nwp_mu_diff_mixing = malloc((int)round(max_Cov_type/2 * max_Cov_type/2) * sizeof(double));
                MHp_mu_diff_mixing = malloc((int)round(max_Cov_type/2 * max_Cov_type/2) * sizeof(double));
                
                double **mean_Deg_Distr_mat_1 = (double **)malloc((int)round(max_Cov_type/2) * sizeof(double*));
                double **mean_Deg_Distr_mat_2 = (double **)malloc((int)round(max_Cov_type/2) * sizeof(double*));

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   mean_Deg_Distr_mat_1[counter] = (double *)malloc((max_degree+1) * sizeof(double)); //+1 for degree 0
                   mean_Deg_Distr_mat_2[counter] = (double *)malloc((max_degree+1) * sizeof(double));
                }

                int length_meanvalues = (max_degree+1)*(max_Cov_type) + (round(max_Cov_type/2)*round(max_Cov_type/2));

if (print_info_MH == 1) {                
Rprintf("Mean Values (%d): ", length_meanvalues); 
for (int counter_print=0; counter_print < length_meanvalues; counter_print++){
   Rprintf(" %f ",meanvalues[counter_print]);       
}
Rprintf("\n");                
}
                
                counter = 0; //skip number of edges
                for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                   for (counter2 = 0; counter2 <= max_degree; counter2++) {
                       mean_Deg_Distr_mat_1[counter1][counter2] = meanvalues[counter];
                       mean_Deg_Distr_mat_2[counter1][counter2] = meanvalues[counter + (int)round(max_Cov_type/2)*(max_degree+1)];
                
                       counter++;
                   }
                }

                double *mean_mixing;

                mean_mixing = malloc((int)(round((max_Cov_type/2) * (max_Cov_type/2))) * sizeof(double));
                
                for (counter1 = 0; counter1 < (int)(round((max_Cov_type/2) * (max_Cov_type/2))); counter1++) {
                    mean_mixing[counter1] = meanvalues[counter + (int)round(max_Cov_type/2)*(max_degree+1)];
                
                    counter++;
                }
                
if (print_info_MH == 1) {
Rprintf("Mean Degree Distribution (Type 1): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %f ", mean_Deg_Distr_mat_1[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");    


Rprintf("Mean Degree Distribution (Type 2): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %f ", mean_Deg_Distr_mat_2[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");  
                

Rprintf("Mean Mixing (%d): ", length_meanvalues); 
for (int counter_print=0; counter_print < (int)(round((max_Cov_type/2) * (max_Cov_type/2))); counter_print++){
   Rprintf(" %f ",mean_mixing[counter_print]);       
}
Rprintf("\n");   
}
                double *mean_Deg_Distr_1;
                double *mean_Deg_Distr_2;

                mean_Deg_Distr_1 = malloc((max_degree+1) * sizeof(double));
                mean_Deg_Distr_2 = malloc((max_degree+1) * sizeof(double));
                               
                for (counter=0; counter <= max_degree; counter++){
                    mean_Deg_Distr_1[counter] = mean_Deg_Distr_mat_1[cov_pattern[0]-1][counter];
                    mean_Deg_Distr_2[counter] = mean_Deg_Distr_mat_2[cov_pattern[1]-1][counter];        
                }

if (print_info_MH == 1) {                
Rprintf("Mean Degree Distr 1: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",mean_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("Mean Degree Distr 2: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",mean_Deg_Distr_2[counter]);        
}
Rprintf("\n");
}
//Variance 

                double **var_Deg_Distr_mat_1 = (double **)malloc((int)round(max_Cov_type/2) * sizeof(double*));
                double **var_Deg_Distr_mat_2 = (double **)malloc((int)round(max_Cov_type/2) * sizeof(double*));

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   var_Deg_Distr_mat_1[counter] = (double *)malloc((max_degree+1) * (max_degree+1) * sizeof(double)); //+1 for degree 0
                   var_Deg_Distr_mat_2[counter] = (double *)malloc((max_degree+1) * (max_degree+1) * sizeof(double));
                }

//int length_varvalues = (max_degree+1) * (max_degree+1) *(max_Cov_type) + (int)(pow((int)(max_Cov_type/2), 4));
//Rprintf("Var Values (%d): ", length_varvalues); 
//for (int counter_print=0; counter_print < length_varvalues; counter_print++){
//   Rprintf(" %f ",varvalues[counter_print]);       
//}
//Rprintf("\n");                
                
                counter = 0; //skip number of edges
                for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                   for (counter2 = 0; counter2 < ((max_degree+1)*(max_degree+1)); counter2++) {
                       var_Deg_Distr_mat_1[counter1][counter2] = varvalues[counter];
                       var_Deg_Distr_mat_2[counter1][counter2] = varvalues[counter + (int)round(max_Cov_type/2)*(max_degree+1)*(max_degree+1)];
                
                       counter++;
                   }
                }

                double *var_mixing;

                var_mixing = malloc((int)(pow((int)round(max_Cov_type/2), 4)) * sizeof(double));
                
                for (counter1 = 0; counter1 < (int)(pow((int)round(max_Cov_type/2), 4)); counter1++) {
                    var_mixing[counter1] = varvalues[counter + (int)round(max_Cov_type/2)*(max_degree+1)*(max_degree+1)];
                
                    counter++;
                }
if (print_info_MH == 1) {                
Rprintf("Var Degree Distribution (Type 1): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 < (max_degree+1)*(max_degree+1); counter2++) {
          Rprintf(" %f ", var_Deg_Distr_mat_1[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");    


Rprintf("Var Degree Distribution (Type 2): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 < (max_degree+1)*(max_degree+1); counter2++) {
          Rprintf(" %f ", var_Deg_Distr_mat_2[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");                 
 

Rprintf("Var Mixing: \n");                
for (counter1 = 0; counter1 <(int)(pow((int)round(max_Cov_type/2), 4)); counter1++) {
    Rprintf(" %f ", var_mixing[counter1]);
}
Rprintf("\n");
}
                double *varvalues_1;
                double *varvalues_2;

                varvalues_1 = malloc((max_degree+1)*(max_degree+1) * sizeof(double));
                varvalues_2 = malloc((max_degree+1)*(max_degree+1) * sizeof(double));
                               
                for (counter=0; counter < (max_degree+1)*(max_degree+1); counter++){
                    varvalues_1[counter] = var_Deg_Distr_mat_1[cov_pattern[0]-1][counter];
                    varvalues_2[counter] = var_Deg_Distr_mat_2[cov_pattern[1]-1][counter];        
                }
                
//end of variance
                
                int nnodes_type1 = 0;
                int nnodes_type2 = 0;
                                                           
                for (counter = 0; counter <= max_degree; counter++) {
                   nnodes_type1 += nwp_Deg_Distr_1[counter];
                   nnodes_type2 += nwp_Deg_Distr_2[counter];
                }
if (print_info_MH == 1) {
Rprintf("Number of Nodes %d %d \n",nnodes_type1, nnodes_type2); 
}

                    for (counter = 0; counter <= max_degree; counter++) {
                        nwp_mu_diff_1[counter] = (double)(nwp_Deg_Distr_1[counter])/(1.0*nnodes_type1) - mean_Deg_Distr_1[counter];
                        MHp_mu_diff_1[counter] = (double)(MHp_Deg_Distr_1[counter])/(1.0*nnodes_type1) - mean_Deg_Distr_1[counter];
                        
                        nwp_mu_diff_2[counter] = (double)(nwp_Deg_Distr_2[counter])/(1.0*nnodes_type2) - mean_Deg_Distr_2[counter];
                        MHp_mu_diff_2[counter] = (double)(MHp_Deg_Distr_2[counter])/(1.0*nnodes_type2) - mean_Deg_Distr_2[counter];
                    }

                    int *nwp_CovEdges_1 = malloc((int)round(max_Cov_type/2) * sizeof(int));
                    int *MHp_CovEdges_1 = malloc((int)round(max_Cov_type/2) * sizeof(int));

                    for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                        nwp_CovEdges_1[counter1] = 0;
                        MHp_CovEdges_1[counter1] = 0;
                        for (counter2 = 0; counter2 < (int)round(max_Cov_type/2); counter2++) {
                            nwp_CovEdges_1[counter1] = nwp_CovEdges_1[counter1] + nwp_mixing_cov[counter1][counter2];
                            MHp_CovEdges_1[counter1] = MHp_CovEdges_1[counter1] + MHp_mixing_cov[counter1][counter2];
                        }
                    }                

if (print_info_MH == 1) {
Rprintf("nwp CovEdges 1: "); 
for (counter=0; counter < (int)round(max_Cov_type/2); counter++){
        Rprintf(" %d ",nwp_CovEdges_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp CovEdges 1: "); 
for (counter=0; counter < (int)round(max_Cov_type/2); counter++){
        Rprintf(" %d ",MHp_CovEdges_1[counter]);        
}
Rprintf("\n");                    
}      
                    counter = 0;
                    for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                        for (counter2 = 0; counter2 < (int)round(max_Cov_type/2); counter2++) {
                            nwp_mu_diff_mixing[counter] = (double)(nwp_mixing_cov[counter1][counter2])/nwp_CovEdges_1[counter1] - mean_mixing[counter];
                            MHp_mu_diff_mixing[counter] = (double)(MHp_mixing_cov[counter1][counter2])/MHp_CovEdges_1[counter1] - mean_mixing[counter];
                            counter++;
                        }
                    }
if (print_info_MH == 1) {
Rprintf("nwp mu diff 1: "); 
for (counter=0; counter <= max_degree; counter++){
        Rprintf(" %f ",nwp_mu_diff_1[counter]);        
}
Rprintf("\n");

Rprintf("nwp mu diff 2: "); 
for (counter=0; counter <= max_degree; counter++){
        Rprintf(" %f ",nwp_mu_diff_2[counter]);        
}
Rprintf("\n");

Rprintf("MHp mu diff 1: "); 
for (counter=0; counter <= max_degree; counter++){
        Rprintf(" %f ",MHp_mu_diff_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp mu diff 2: "); 
for (counter=0; counter <= max_degree; counter++){
        Rprintf(" %f ",MHp_mu_diff_2[counter]);        
}
Rprintf("\n");

Rprintf("nwp mu diff Mixing: "); 
for (counter=0; counter < (int)pow((int)round(max_Cov_type/2),2); counter++){
        Rprintf(" %f ",nwp_mu_diff_mixing[counter]);        
}
Rprintf("\n");

Rprintf("MHp mu diff Mixing: "); 
for (counter=0; counter < (int)pow((int)round(max_Cov_type/2),2); counter++){
        Rprintf(" %f ",MHp_mu_diff_mixing[counter]);        
}
Rprintf("\n");
}
                double *nwp_intermediate_mat_1;
                double *nwp_intermediate_mat_2;
                double *MHp_intermediate_mat_1;
                double *MHp_intermediate_mat_2;

                nwp_intermediate_mat_1 = malloc((max_degree+1) * sizeof(double));
                nwp_intermediate_mat_2 = malloc((max_degree+1) * sizeof(double));
                MHp_intermediate_mat_1 = malloc((max_degree+1) * sizeof(double));
                MHp_intermediate_mat_2 = malloc((max_degree+1) * sizeof(double));
                              
                counter2 = -1;
                for (counter = 0; counter < ((max_degree+1) * (max_degree+1)); counter++) {
                    counter1 = counter % ((max_degree+1));
                    if (counter1 == 0) {
                        counter2 += 1;
                        nwp_intermediate_mat_1[counter2] = 0;
                        MHp_intermediate_mat_1[counter2] = 0;
                        
                        nwp_intermediate_mat_2[counter2] = 0;
                        MHp_intermediate_mat_2[counter2] = 0;
                    }
                    nwp_intermediate_mat_1[counter2] += nwp_mu_diff_1[counter1]*varvalues_1[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mat_1[counter2] += MHp_mu_diff_1[counter1]*varvalues_1[counter];
                    nwp_intermediate_mat_2[counter2] += nwp_mu_diff_2[counter1]*varvalues_2[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mat_2[counter2] += MHp_mu_diff_2[counter1]*varvalues_2[counter];                      
                }
if (print_info_MH == 1) {                
Rprintf("nwp Intermediate (Type 1): "); 
for (counter=0; counter < (max_degree+1); counter++){
        Rprintf(" %f ",nwp_intermediate_mat_1[counter]);        
}
Rprintf("\n");

Rprintf("nwp Intermediate (Type 2): "); 
for (counter=0; counter < (max_degree+1); counter++){
        Rprintf(" %f ",nwp_intermediate_mat_2[counter]);        
}
Rprintf("\n");

Rprintf("MHp Intermediate (Type 1): "); 
for (counter=0; counter < (max_degree+1); counter++){
        Rprintf(" %f ",MHp_intermediate_mat_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp Intermediate (Type 2): "); 
for (counter=0; counter < (max_degree+1); counter++){
        Rprintf(" %f ",MHp_intermediate_mat_2[counter]);        
}
Rprintf("\n");
}
                double *nwp_intermediate_mixing;
                double *MHp_intermediate_mixing;

                nwp_intermediate_mixing = malloc((int)pow((int)round(max_Cov_type/2),2) * sizeof(double));
                MHp_intermediate_mixing = malloc((int)pow((int)round(max_Cov_type/2),2) * sizeof(double));
                              
                counter2 = -1;
                for (counter = 0; counter < (int)pow((int)round(max_Cov_type/2),4); counter++) {
                    counter1 = counter % ((int)pow((int)round(max_Cov_type/2),2));
                    if (counter1 == 0) {
                        counter2 += 1;
                        nwp_intermediate_mixing[counter2] = 0;
                        MHp_intermediate_mixing[counter2] = 0;
                        
                        nwp_intermediate_mixing[counter2] = 0;
                        MHp_intermediate_mixing[counter2] = 0;
                    }
                    nwp_intermediate_mixing[counter2] += nwp_mu_diff_mixing[counter1]*var_mixing[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mixing[counter2] += MHp_mu_diff_mixing[counter1]*var_mixing[counter];
                    
                }
if (print_info_MH == 1) {                
Rprintf("nwp Intermediate mixing: "); 
for (counter=0; counter < ((int)pow((int)round(max_Cov_type/2),2)); counter++){
        Rprintf(" %f ",nwp_intermediate_mixing[counter]);        
}
Rprintf("\n");

Rprintf("MHp Intermediate mixing: "); 
for (counter=0; counter < ((int)pow((int)round(max_Cov_type/2),2)); counter++){
        Rprintf(" %f ",MHp_intermediate_mixing[counter]);        
}
Rprintf("\n");
}

                double pdf_gaussian_nwp_1 = 0;
                double pdf_gaussian_MHp_1 = 0;
                
                double pdf_gaussian_nwp_2 = 0;
                double pdf_gaussian_MHp_2 = 0;

                double pdf_gaussian_nwp_3 = 0;
                double pdf_gaussian_MHp_3 = 0;
                
                for (counter = 0; counter <= max_degree; counter++) {
                    pdf_gaussian_nwp_1 += nwp_intermediate_mat_1[counter] * nwp_mu_diff_1[counter];
                    pdf_gaussian_MHp_1 += MHp_intermediate_mat_1[counter] * MHp_mu_diff_1[counter];
                    
                    pdf_gaussian_nwp_2 += nwp_intermediate_mat_2[counter] * nwp_mu_diff_2[counter];
                    pdf_gaussian_MHp_2 += MHp_intermediate_mat_2[counter] * MHp_mu_diff_2[counter];
                }

                for (counter = 0; counter <= ((int)pow((int)round(max_Cov_type/2),2)); counter++) {
                    pdf_gaussian_nwp_3 += nwp_intermediate_mixing[counter] * nwp_mu_diff_mixing[counter];
                    pdf_gaussian_MHp_3 += MHp_intermediate_mixing[counter] * MHp_mu_diff_mixing[counter];
                }
                
                pdf_gaussian_nwp =  (-.5 * pdf_gaussian_nwp_1) + (-.5 * pdf_gaussian_nwp_2) + (-.5 * pdf_gaussian_nwp_3);
                pdf_gaussian_MHp =  (-.5 * pdf_gaussian_MHp_1) + (-.5 * pdf_gaussian_MHp_2) +  (-.5 * pdf_gaussian_MHp_3);
                
if (print_info_MH == 1) {                
Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                
                
                free(nwp_mu_diff_1);
                free(nwp_mu_diff_2);
                free(MHp_mu_diff_1);
                free(MHp_mu_diff_2);

                free(nwp_mu_diff_mixing);
                free(MHp_mu_diff_mixing);

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   free(mean_Deg_Distr_mat_1[counter]);
                   free(mean_Deg_Distr_mat_2[counter]);
                }
                
                free(mean_Deg_Distr_mat_1);
                free(mean_Deg_Distr_mat_2); 

                free(mean_mixing);

                free(mean_Deg_Distr_1);
                free(mean_Deg_Distr_2);

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   free(var_Deg_Distr_mat_1[counter]); 
                   free(var_Deg_Distr_mat_2[counter]);
                }
                
                free(var_Deg_Distr_mat_1);
                free(var_Deg_Distr_mat_2);

                free(var_mixing);
                
                free(varvalues_1);
                free(varvalues_2);

                free(nwp_intermediate_mat_1);
                free(nwp_intermediate_mat_2);
                free(MHp_intermediate_mat_1);
                free(MHp_intermediate_mat_2);

                free(nwp_intermediate_mixing);
                free(MHp_intermediate_mixing);
                
                }
////////////////
                
           if ((prob_type[0] == 2) && (prob_type[1] == 2) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
      
                double **mean_Deg_Distr_mat_1 = (double **)malloc((int)round(max_Cov_type/2) * sizeof(double*));
                double **mean_Deg_Distr_mat_2 = (double **)malloc((int)round(max_Cov_type/2) * sizeof(double*));

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   mean_Deg_Distr_mat_1[counter] = (double *)malloc((max_degree+1) * sizeof(double)); //+1 for degree 0
                   mean_Deg_Distr_mat_2[counter] = (double *)malloc((max_degree+1) * sizeof(double));
                }

                int length_meanvalues = (max_degree+1)*(max_Cov_type) + (round(max_Cov_type/2)*round(max_Cov_type/2));

if (print_info_MH == 1) {                
Rprintf("Mean Values (%d): ", length_meanvalues); 
for (int counter_print=0; counter_print < length_meanvalues; counter_print++){
   Rprintf(" %f ",meanvalues[counter_print]);       
}
Rprintf("\n");                
}
                
                counter = 0; //skip number of edges
                for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                   for (counter2 = 0; counter2 <= max_degree; counter2++) {
                       mean_Deg_Distr_mat_1[counter1][counter2] = meanvalues[counter];
                       mean_Deg_Distr_mat_2[counter1][counter2] = meanvalues[counter + (int)round(max_Cov_type/2)*(max_degree+1)];
                
                       counter++;
                   }
                }

                double *mean_mixing;

                mean_mixing = malloc((int)(round((max_Cov_type/2) * (max_Cov_type/2))) * sizeof(double));
                
                for (counter1 = 0; counter1 < (int)(round((max_Cov_type/2) * (max_Cov_type/2))); counter1++) {
                    mean_mixing[counter1] = meanvalues[counter + (int)round(max_Cov_type/2)*(max_degree+1)];
                
                    counter++;
                }
                
if (print_info_MH == 1) {
Rprintf("Mean Degree Distribution (Type 1): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %f ", mean_Deg_Distr_mat_1[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");    


Rprintf("Mean Degree Distribution (Type 2): \n");                
for (counter1 = 0; counter1 <(int)round(max_Cov_type/2); counter1++) {
    Rprintf("Type %d: \n", counter1);
    for (counter2 = 0; counter2 <= max_degree; counter2++) {
          Rprintf(" %f ", mean_Deg_Distr_mat_2[counter1][counter2]);
    }
    Rprintf("\n");
}
Rprintf("\n");  
                

Rprintf("Mean Mixing (%d): ", length_meanvalues); 
for (int counter_print=0; counter_print < (int)(round((max_Cov_type/2) * (max_Cov_type/2))); counter_print++){
   Rprintf(" %f ",mean_mixing[counter_print]);       
}
Rprintf("\n");   
}
                double *mean_Deg_Distr_1;
                double *mean_Deg_Distr_2;

                mean_Deg_Distr_1 = malloc((max_degree+1) * sizeof(double));
                mean_Deg_Distr_2 = malloc((max_degree+1) * sizeof(double));
                               
                for (counter=0; counter <= max_degree; counter++){
                    mean_Deg_Distr_1[counter] = mean_Deg_Distr_mat_1[cov_pattern[0]-1][counter];
                    mean_Deg_Distr_2[counter] = mean_Deg_Distr_mat_2[cov_pattern[1]-1][counter];        
                }

if (print_info_MH == 1) {                
Rprintf("Mean Degree Distr 1: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",mean_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("Mean Degree Distr 2: "); 
for (counter=0; counter <= max_degree; counter++){
    Rprintf(" %f ",mean_Deg_Distr_2[counter]);        
}
Rprintf("\n");
}
                
                int nnodes_type1 = 0;
                int nnodes_type2 = 0;
                                                           
                for (counter = 0; counter <= max_degree; counter++) {
                   nnodes_type1 += nwp_Deg_Distr_1[counter];
                   nnodes_type2 += nwp_Deg_Distr_2[counter];
                }
if (print_info_MH == 1) {
Rprintf("Number of Nodes %d %d \n",nnodes_type1, nnodes_type2); 
}

                
                double pdf_gaussian_nwp_1 = 0;
                double pdf_gaussian_MHp_1 = 0;
                
                double pdf_gaussian_nwp_2 = 0;
                double pdf_gaussian_MHp_2 = 0;

                double pdf_gaussian_nwp_3 = 0;
                double pdf_gaussian_MHp_3 = 0;
                
                
                for (counter = 0; counter <= *maxdegree; counter++) {
                        pdf_gaussian_nwp_1 += (mean_Deg_Distr_1[counter]-1) * log((double)(nwp_Deg_Distr_1[counter])/(1.0*nnodes_type1));
                        pdf_gaussian_MHp_1 += (mean_Deg_Distr_1[counter]-1) * log((double)(MHp_Deg_Distr_1[counter])/(1.0*nnodes_type1));
                        
                        pdf_gaussian_nwp_2 += (mean_Deg_Distr_2[counter]-1) * log((double)(nwp_Deg_Distr_2[counter])/(1.0*nnodes_type2));
                        pdf_gaussian_MHp_2 += (mean_Deg_Distr_2[counter]-1) * log((double)(MHp_Deg_Distr_2[counter])/(1.0*nnodes_type2));
                }
                
                    int *nwp_CovEdges_1 = malloc((int)round(max_Cov_type/2) * sizeof(int));
                    int *MHp_CovEdges_1 = malloc((int)round(max_Cov_type/2) * sizeof(int));

                    for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                        nwp_CovEdges_1[counter1] = 0;
                        MHp_CovEdges_1[counter1] = 0;
                        for (counter2 = 0; counter2 < (int)round(max_Cov_type/2); counter2++) {
                            nwp_CovEdges_1[counter1] = nwp_CovEdges_1[counter1] + nwp_mixing_cov[counter1][counter2];
                            MHp_CovEdges_1[counter1] = MHp_CovEdges_1[counter1] + MHp_mixing_cov[counter1][counter2];
                        }
                    }                

if (print_info_MH == 1) {
Rprintf("nwp CovEdges 1: "); 
for (counter=0; counter < (int)round(max_Cov_type/2); counter++){
        Rprintf(" %d ",nwp_CovEdges_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp CovEdges 1: "); 
for (counter=0; counter < (int)round(max_Cov_type/2); counter++){
        Rprintf(" %d ",MHp_CovEdges_1[counter]);        
}
Rprintf("\n");                    
}
                    
                    counter = 0;
                    for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                        for (counter2 = 0; counter2 < (int)round(max_Cov_type/2); counter2++) {
                            
                            pdf_gaussian_nwp_3 += (mean_mixing[counter]-1) * log((double)(nwp_mixing_cov[counter1][counter2])/nwp_CovEdges_1[counter1]);
                            pdf_gaussian_MHp_3 += (mean_mixing[counter]-1) * log((double)(MHp_mixing_cov[counter1][counter2])/MHp_CovEdges_1[counter1]);
                            
                            counter++;
                        }
                    }

                pdf_gaussian_nwp =  (pdf_gaussian_nwp_1) + (pdf_gaussian_nwp_2) + (pdf_gaussian_nwp_3);
                pdf_gaussian_MHp =  (pdf_gaussian_MHp_1) + (pdf_gaussian_MHp_2) +  (pdf_gaussian_MHp_3);
                
if (print_info_MH == 1) {
        Rprintf("PDF NWP: %f %f %f\n", pdf_gaussian_nwp_1, pdf_gaussian_nwp_2, pdf_gaussian_nwp_3);
        Rprintf("PDF MHP: %f %f %f\n", pdf_gaussian_MHp_1, pdf_gaussian_MHp_2, pdf_gaussian_MHp_3);
        Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                
                
                if (!isfinite(pdf_gaussian_nwp)) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 1: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }
                
                if (pdf_gaussian_nwp != pdf_gaussian_nwp) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 2: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }
                      
        
                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   free(mean_Deg_Distr_mat_1[counter]);
                   free(mean_Deg_Distr_mat_2[counter]);
                }
                
                free(mean_Deg_Distr_mat_1);
                free(mean_Deg_Distr_mat_2); 

                free(mean_mixing);

                free(mean_Deg_Distr_1);
                free(mean_Deg_Distr_2);

                
                }

/////////////////////////////
                
                free(CovPattern);
                free(Num_Cov_type); 

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   free(nwp_Deg_Distr_mat_1[counter]);
                   free(nwp_Deg_Distr_mat_2[counter]);
                   free(MHp_Deg_Distr_mat_1[counter]);
                   free(MHp_Deg_Distr_mat_2[counter]);
                }
                
                free(nwp_Deg_Distr_mat_1);
                free(nwp_Deg_Distr_mat_2);
                free(MHp_Deg_Distr_mat_1);
                free(MHp_Deg_Distr_mat_2);

                for (counter = 0; counter < (int)round(max_Cov_type/2); counter++) {
                   free(nwp_mixing_cov[counter]);
                   free(MHp_mixing_cov[counter]); 
                }
                
                free(nwp_mixing_cov);
                free(MHp_mixing_cov);

                free(nwp_Deg_Distr_1);
                free(nwp_Deg_Distr_2);
                free(MHp_Deg_Distr_1);
                free(MHp_Deg_Distr_2);

                free(nwp_Deg_Distr_Edges_1);
                free(nwp_Deg_Distr_Edges_2);
                free(MHp_Deg_Distr_Edges_1);
                free(MHp_Deg_Distr_Edges_2);

                
                
                /* BEGIN: NETWORK STABILITY CODE*/

                /* REMOVED */
                
                /* END--: NETWORK STABILITY CODE*/                
              }  
//Rprintf("Probs %f %d %f %f %f %f\n",networkstatistics[0], MHp_nedges, prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
            }
            
/////////////////////////////
//Begin: Degree Dist Only/////
//////////////////////////// 
            
/////////////////////////////
//Begin: Degree Dist Only/////
//////////////////////////// 
            
            if ((prob_type[0] >= 1) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                int Deg_nwp[2];
                int Deg_MHp[2];
                
                Deg_nwp[0] = OUT_DEG[*(MHp->toggletail)] + IN_DEG[*(MHp->toggletail)];
                Deg_nwp[1] = OUT_DEG[*(MHp->togglehead)] + IN_DEG[*(MHp->togglehead)];

if (print_info_MH == 1) {
Rprintf("Node ID 1 %d Node ID 2 %d\n",*(MHp->toggletail), *(MHp->togglehead));                 
Rprintf("NWP Degree 1 %d Degree 2 %d\n",Deg_nwp[0], Deg_nwp[1]);                 
}

                if (nwp->nedges > MHp_nedges) {
                    Deg_MHp[0] = Deg_nwp[0] - 1;
                    Deg_MHp[1] = Deg_nwp[1] - 1;
                } else {
                    Deg_MHp[0] = Deg_nwp[0] + 1;
                    Deg_MHp[1] = Deg_nwp[1] + 1;                    
                }
                
if (print_info_MH == 1) {
    Rprintf("MHP Degree 1 %d Degree 2 %d\n",Deg_MHp[0], Deg_MHp[1]);
}
    
//Rprintf("Max degree: %d \n",*maxdegree);
                
            if ((Deg_nwp[0] > *maxdegree) || (Deg_nwp[1] > *maxdegree) || (Deg_MHp[0] > *maxdegree) || (Deg_MHp[1] > *maxdegree) ) {
               prob_g2_g = 1;
               pdf_gaussian_MHp = log(0);
               prob_g_g2 = 1;
               pdf_gaussian_nwp = 0;
//Rprintf("Proposal Excesses Max Edges: %f\n", pdf_gaussian_MHp);
            } else {
                       
if (print_info_MH == 1) {                
Rprintf("WorkSpace (%d): ", m->n_stats); 
for (int counter_print=0; counter_print < (m->n_stats); counter_print++){
    Rprintf(" %f ",m->workspace[counter_print]);        
}
Rprintf("\n");                
}

                double *nwp_Deg_Distr_1;
                double *nwp_Deg_Distr_2;
                double *MHp_Deg_Distr_1;
                double *MHp_Deg_Distr_2;

                nwp_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));

                double *nwp_Deg_Distr_Edges_1;
                double *nwp_Deg_Distr_Edges_2;
                double *MHp_Deg_Distr_Edges_1;
                double *MHp_Deg_Distr_Edges_2;

                nwp_Deg_Distr_Edges_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_Deg_Distr_Edges_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_Edges_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_Edges_2 = malloc((*maxdegree+1) * sizeof(double));

                double nwp_z = 0;
                double MHp_z = 0;
                
                counter = 1; //skip number of edges
                for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
                       nwp_Deg_Distr_1[counter2] = (int)networkstatistics[counter];
                       nwp_Deg_Distr_2[counter2] = (int)networkstatistics[counter + (*maxdegree+1)];
                      
                       MHp_Deg_Distr_1[counter2] = nwp_Deg_Distr_1[counter2] + (int)(m->workspace[counter]);
                       MHp_Deg_Distr_2[counter2] = nwp_Deg_Distr_2[counter2] + (int)(m->workspace[counter + (*maxdegree+1)]);
                       
                       nwp_Deg_Distr_Edges_1[counter2] = nwp_Deg_Distr_1[counter2] * (counter2);
                       nwp_Deg_Distr_Edges_2[counter2] = nwp_Deg_Distr_2[counter2] * (counter2);
                       
                       MHp_Deg_Distr_Edges_1[counter2] = MHp_Deg_Distr_1[counter2] * (counter2);
                       MHp_Deg_Distr_Edges_2[counter2] = MHp_Deg_Distr_2[counter2] * (counter2);
                       
                      nwp_z = nwp_z + nwp_Deg_Distr_Edges_1[counter2];
                      MHp_z = MHp_z + MHp_Deg_Distr_Edges_1[counter2];
                    
                       counter++;
                }
           
if (print_info_MH == 1) {                
Rprintf("NWP Degree Distr 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("NWP Degree Distr 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_2[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_2[counter]);        
}
Rprintf("\n");
}                      
                
if (print_info_MH == 1) {                
Rprintf("NWP Degree Distr Edges 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_Edges_1[counter]);        
}
Rprintf("\n");

Rprintf("NWP Degree Distr Edges 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_Edges_2[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr Edges 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_Edges_1[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr Edges 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_Edges_2[counter]);        
}
Rprintf("\n");                
}                              

                // Construct expected degree distribution 

                double nwp_exp_dmm;
                double MHp_exp_dmm;

                nwp_exp_dmm = (nwp_Deg_Distr_Edges_1[Deg_nwp[0]] * nwp_Deg_Distr_Edges_2[Deg_nwp[1]])/ nwp_z;                      
                MHp_exp_dmm = (MHp_Deg_Distr_Edges_1[Deg_MHp[0]] * MHp_Deg_Distr_Edges_2[Deg_MHp[1]])/ MHp_z;

if (print_info_MH == 1) {                
Rprintf("NWP DMM %f MHP DMM %f\n",nwp_exp_dmm, MHp_exp_dmm);
} 

                if (nwp->nedges > MHp_nedges) {  //Edge is removed from g to g2
                        prob_g_g2 = nwp_exp_dmm;
                } else { //Edge is added from g to g2
                        prob_g_g2 = (nwp_Deg_Distr_1[Deg_nwp[0]] * nwp_Deg_Distr_2[Deg_nwp[1]])  - nwp_exp_dmm;
                }
                if (nwp->nedges < MHp_nedges) {  //Edge is removed from g2 to g
                        prob_g2_g = MHp_exp_dmm;
                } else {  //Edge is added from g2 to g
                        prob_g2_g = (double)((MHp_Deg_Distr_1[Deg_MHp[0]] * MHp_Deg_Distr_2[Deg_MHp[1]])  - MHp_exp_dmm);
                }

if (print_info_MH == 1) {
Rprintf("Moves g to g2: %f Moves g2 to g %f\n",prob_g_g2, prob_g2_g);
}
                
//////////////////////////                
                
           if ((prob_type[0] == 1) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                double *mean_Deg_Distr_1;
                double *mean_Deg_Distr_2;

                mean_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                mean_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));
                
                double *nwp_mu_diff_1;
                double *nwp_mu_diff_2;
                double *MHp_mu_diff_1;
                double *MHp_mu_diff_2;

                nwp_mu_diff_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_mu_diff_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_mu_diff_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_mu_diff_2 = malloc((*maxdegree+1) * sizeof(double));
    
if (print_info_MH == 1) {                
Rprintf("Mean Values (%d): ", (*maxdegree+1)*2); 
for (int counter_print=0; counter_print < ((*maxdegree+1)*2); counter_print++){
   Rprintf(" %f ",meanvalues[counter_print]);       
}
Rprintf("\n");                
}
                
                counter = 0; //skip number of edges
                for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
                       mean_Deg_Distr_1[counter2] = meanvalues[counter];
                       mean_Deg_Distr_2[counter2] = meanvalues[counter + (*maxdegree+1)];
                
                       counter++;
                }

                
if (print_info_MH == 1) {
Rprintf("Mean Degree Distribution (Type 1): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_1[counter2]);
}
Rprintf("\n");


Rprintf("Mean Degree Distribution (Type 2): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_2[counter2]);
}
   
}


//Variance 

//int length_varvalues = (max_degree+1) * (max_degree+1) *(max_Cov_type) + (int)(pow((int)(max_Cov_type/2), 4));
//Rprintf("Var Values (%d): ", length_varvalues); 
//for (int counter_print=0; counter_print < length_varvalues; counter_print++){
//   Rprintf(" %f ",varvalues[counter_print]);       
//}
//Rprintf("\n");                
                
                double *varvalues_1;
                double *varvalues_2;

                varvalues_1 = malloc((*maxdegree+1)*(*maxdegree+1) * sizeof(double));
                varvalues_2 = malloc((*maxdegree+1)*(*maxdegree+1) * sizeof(double));
                
                counter = 0; //skip number of edges
                for (counter2 = 0; counter2 < ((*maxdegree+1)*(*maxdegree+1)); counter2++) {
                       varvalues_1[counter2] = varvalues[counter];
                       varvalues_2[counter2] = varvalues[counter + (*maxdegree+1)*(*maxdegree+1)];
                
                       counter++;
                }

if (print_info_MH == 1) {                
Rprintf("Var Degree Distribution (Type 1): \n");                
for (counter2 = 0; counter2 < (*maxdegree+1)*(*maxdegree+1); counter2++) {
    Rprintf(" %f ", varvalues_1[counter2]);
}
Rprintf("\n");

Rprintf("Var Degree Distribution (Type 2): \n");                
for (counter2 = 0; counter2 < (*maxdegree+1)*(*maxdegree+1); counter2++) {
    Rprintf(" %f ", varvalues_2[counter2]);
}
Rprintf("\n");                 
}

                
//end of variance
                
                int nnodes_type1 = 0;
                int nnodes_type2 = 0;
                                                           
                for (counter = 0; counter <= *maxdegree; counter++) {
                   nnodes_type1 += nwp_Deg_Distr_1[counter];
                   nnodes_type2 += nwp_Deg_Distr_2[counter];
                }
if (print_info_MH == 1) {
Rprintf("Number of Nodes %d %d \n",nnodes_type1, nnodes_type2); 
}

                    for (counter = 0; counter <= *maxdegree; counter++) {
                        nwp_mu_diff_1[counter] = (double)(nwp_Deg_Distr_1[counter])/(1.0*nnodes_type1) - mean_Deg_Distr_1[counter];
                        MHp_mu_diff_1[counter] = (double)(MHp_Deg_Distr_1[counter])/(1.0*nnodes_type1) - mean_Deg_Distr_1[counter];
                        
                        nwp_mu_diff_2[counter] = (double)(nwp_Deg_Distr_2[counter])/(1.0*nnodes_type2) - mean_Deg_Distr_2[counter];
                        MHp_mu_diff_2[counter] = (double)(MHp_Deg_Distr_2[counter])/(1.0*nnodes_type2) - mean_Deg_Distr_2[counter];
                    }

if (print_info_MH == 1) {
Rprintf("nwp mu diff 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",nwp_mu_diff_1[counter]);        
}
Rprintf("\n");

Rprintf("nwp mu diff 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",nwp_mu_diff_2[counter]);        
}
Rprintf("\n");

Rprintf("MHp mu diff 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",MHp_mu_diff_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp mu diff 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",MHp_mu_diff_2[counter]);        
}
Rprintf("\n");

}
                double *nwp_intermediate_mat_1;
                double *nwp_intermediate_mat_2;
                double *MHp_intermediate_mat_1;
                double *MHp_intermediate_mat_2;

                nwp_intermediate_mat_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_intermediate_mat_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_intermediate_mat_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_intermediate_mat_2 = malloc((*maxdegree+1) * sizeof(double));
                              
                counter2 = -1;
                for (counter = 0; counter < ((*maxdegree+1) * (*maxdegree+1)); counter++) {
                    counter1 = counter % ((*maxdegree+1));
                    if (counter1 == 0) {
                        counter2 += 1;
                        nwp_intermediate_mat_1[counter2] = 0;
                        MHp_intermediate_mat_1[counter2] = 0;
                        
                        nwp_intermediate_mat_2[counter2] = 0;
                        MHp_intermediate_mat_2[counter2] = 0;
                    }
                    nwp_intermediate_mat_1[counter2] += nwp_mu_diff_1[counter1]*varvalues_1[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mat_1[counter2] += MHp_mu_diff_1[counter1]*varvalues_1[counter];
                    nwp_intermediate_mat_2[counter2] += nwp_mu_diff_2[counter1]*varvalues_2[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mat_2[counter2] += MHp_mu_diff_2[counter1]*varvalues_2[counter];                      
                }
if (print_info_MH == 1) {                
Rprintf("nwp Intermediate (Type 1): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",nwp_intermediate_mat_1[counter]);        
}
Rprintf("\n");

Rprintf("nwp Intermediate (Type 2): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",nwp_intermediate_mat_2[counter]);        
}
Rprintf("\n");

Rprintf("MHp Intermediate (Type 1): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",MHp_intermediate_mat_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp Intermediate (Type 2): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",MHp_intermediate_mat_2[counter]);        
}
Rprintf("\n");
}

                double pdf_gaussian_nwp_1 = 0;
                double pdf_gaussian_MHp_1 = 0;
                
                double pdf_gaussian_nwp_2 = 0;
                double pdf_gaussian_MHp_2 = 0;
                
                for (counter = 0; counter <= *maxdegree; counter++) {
                    pdf_gaussian_nwp_1 += nwp_intermediate_mat_1[counter] * nwp_mu_diff_1[counter];
                    pdf_gaussian_MHp_1 += MHp_intermediate_mat_1[counter] * MHp_mu_diff_1[counter];
                    
                    pdf_gaussian_nwp_2 += nwp_intermediate_mat_2[counter] * nwp_mu_diff_2[counter];
                    pdf_gaussian_MHp_2 += MHp_intermediate_mat_2[counter] * MHp_mu_diff_2[counter];
                }
                
                pdf_gaussian_nwp =  (-.5 * pdf_gaussian_nwp_1) + (-.5 * pdf_gaussian_nwp_2);
                pdf_gaussian_MHp =  (-.5 * pdf_gaussian_MHp_1) + (-.5 * pdf_gaussian_MHp_2);
                
if (print_info_MH == 1) {                
Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                
                
                free(nwp_mu_diff_1);
                free(nwp_mu_diff_2);
                free(MHp_mu_diff_1);
                free(MHp_mu_diff_2);

                free(mean_Deg_Distr_1);
                free(mean_Deg_Distr_2);
               
                free(varvalues_1);
                free(varvalues_2);

                free(nwp_intermediate_mat_1);
                free(nwp_intermediate_mat_2);
                free(MHp_intermediate_mat_1);
                free(MHp_intermediate_mat_2);
                
                }                
                
 //////////////               
     
                
           if ((prob_type[0] == 2) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                double *mean_Deg_Distr_1;
                double *mean_Deg_Distr_2;

                mean_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                mean_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));
                   
if (print_info_MH == 1) {                
Rprintf("Mean Values (%d): ", (*maxdegree+1)*2); 
for (int counter_print=0; counter_print < ((*maxdegree+1)*2); counter_print++){
   Rprintf(" %f ",meanvalues[counter_print]);       
}
Rprintf("\n");                
}
                
                counter = 0; //skip number of edges
                for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
                       mean_Deg_Distr_1[counter2] = meanvalues[counter];
                       mean_Deg_Distr_2[counter2] = meanvalues[counter + (*maxdegree+1)];
                
                       counter++;
                }

                
if (print_info_MH == 1) {
Rprintf("Mean Degree Distribution (Type 1): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_1[counter2]);
}
Rprintf("\n");


Rprintf("Mean Degree Distribution (Type 2): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_2[counter2]);
}
   
}

                int nnodes_type1 = 0;
                int nnodes_type2 = 0;
                                                           
                for (counter = 0; counter <= *maxdegree; counter++) {
                   nnodes_type1 += nwp_Deg_Distr_1[counter];
                   nnodes_type2 += nwp_Deg_Distr_2[counter];
                }
if (print_info_MH == 1) {
Rprintf("Number of Nodes %d %d \n",nnodes_type1, nnodes_type2); 
}

                
                double pdf_gaussian_nwp_1 = 0;
                double pdf_gaussian_MHp_1 = 0;
                
                double pdf_gaussian_nwp_2 = 0;
                double pdf_gaussian_MHp_2 = 0;
                
                for (counter = 0; counter <= *maxdegree; counter++) {
                        pdf_gaussian_nwp_1 += (mean_Deg_Distr_1[counter]-1) * log((double)(nwp_Deg_Distr_1[counter])/(1.0*nnodes_type1));
                        pdf_gaussian_MHp_1 += (mean_Deg_Distr_1[counter]-1) * log((double)(MHp_Deg_Distr_1[counter])/(1.0*nnodes_type1));
                        
                        pdf_gaussian_nwp_2 += (mean_Deg_Distr_2[counter]-1) * log((double)(nwp_Deg_Distr_2[counter])/(1.0*nnodes_type2));
                        pdf_gaussian_MHp_2 += (mean_Deg_Distr_2[counter]-1) * log((double)(MHp_Deg_Distr_2[counter])/(1.0*nnodes_type2));
                }
                
                pdf_gaussian_nwp =  (pdf_gaussian_nwp_1) + (pdf_gaussian_nwp_2);
                pdf_gaussian_MHp =  (pdf_gaussian_MHp_1) + (pdf_gaussian_MHp_2);
                
if (print_info_MH == 1) {                
Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                                
                if (!isfinite(pdf_gaussian_nwp)) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 1: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }
                
                if (pdf_gaussian_nwp != pdf_gaussian_nwp) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 2: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }
                
                free(mean_Deg_Distr_1);
                free(mean_Deg_Distr_2);
                
                }  

           if ((prob_type[0] == 99) && (prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){
                                
       
                pdf_gaussian_nwp = log(meanvalues[(int)nwp_Deg_Distr_1[0]]);
                pdf_gaussian_MHp = log(meanvalues[(int)MHp_Deg_Distr_1[0]]);

                
if (print_info_MH == 1) {
  Rprintf("Mean Probs %f %f \n", meanvalues[(int)nwp_Deg_Distr_1[0]], meanvalues[(int)MHp_Deg_Distr_1[0]]);
  Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                                
                if (!isfinite(pdf_gaussian_nwp)) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 1: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }
                
                if (pdf_gaussian_nwp != pdf_gaussian_nwp) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 2: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }

                }
                
                
//////////////////////////                
                
                free(nwp_Deg_Distr_1);
                free(nwp_Deg_Distr_2);
                free(MHp_Deg_Distr_1);
                free(MHp_Deg_Distr_2);
                
                free(nwp_Deg_Distr_Edges_1);
                free(nwp_Deg_Distr_Edges_2);
                free(MHp_Deg_Distr_Edges_1);
                free(MHp_Deg_Distr_Edges_2);

                
            }
          }
////////////////////////
//End: Degree Dist Only//
////////////////////////                
        
////////////////////////
//Begin: Mixing Nested in Degree Dist //
////////////////////////  
            
            if ((prob_type[0] >= 1) && (prob_type[1] == -1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                int Deg_nwp[2];
                int Deg_MHp[2];
                
                Deg_nwp[0] = OUT_DEG[*(MHp->toggletail)] + IN_DEG[*(MHp->toggletail)];
                Deg_nwp[1] = OUT_DEG[*(MHp->togglehead)] + IN_DEG[*(MHp->togglehead)];

if (print_info_MH == 1) {
Rprintf("Node ID 1 %d Node ID 2 %d\n",*(MHp->toggletail), *(MHp->togglehead));                 
Rprintf("NWP Degree 1 %d Degree 2 %d\n",Deg_nwp[0], Deg_nwp[1]);                 
}

                if (nwp->nedges > MHp_nedges) {
                    Deg_MHp[0] = Deg_nwp[0] - 1;
                    Deg_MHp[1] = Deg_nwp[1] - 1;
                } else {
                    Deg_MHp[0] = Deg_nwp[0] + 1;
                    Deg_MHp[1] = Deg_nwp[1] + 1;                    
                }
                
if (print_info_MH == 1) {
    Rprintf("MHP Degree 1 %d Degree 2 %d\n",Deg_MHp[0], Deg_MHp[1]);
}
    
//Rprintf("Max degree: %d \n",*maxdegree);
                
            if ((Deg_nwp[0] > *maxdegree) || (Deg_nwp[1] > *maxdegree) || (Deg_MHp[0] > *maxdegree) || (Deg_MHp[1] > *maxdegree) ) {
               prob_g2_g = 1;
               pdf_gaussian_MHp = log(0);
               prob_g_g2 = 1;
               pdf_gaussian_nwp = 0;
//Rprintf("Proposal Excesses Max Edges: %f\n", pdf_gaussian_MHp);
            } else {
                    
//Figure out number of covariate patterns - Should be easier!

                ModelTerm *mtp2 = m->termarray;                     
                mtp2++;
if (print_info_MH == 1) {
        Rprintf("Term 2: "); 
        for (counter=0; counter < mtp2->ninputparams; counter++){
                Rprintf(" %d ",(int)round(mtp2->inputparams[counter]));        
        }
        Rprintf("\n");  
}
                int *CovPattern = malloc(nwp->nnodes * sizeof(int));
                
                for (counter=0; counter < nwp->bipartite; counter++){
                        CovPattern[counter] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->bipartite + counter]);        
                }

                int max_degree = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->bipartite - 2]);
                
                mtp2++;  

if (print_info_MH == 1) {
Rprintf("Term 3: "); 
for (counter=0; counter < mtp2->ninputparams; counter++){
        Rprintf(" %d ",(int)round(mtp2->inputparams[counter]));        
}
Rprintf("\n");  
}
                
                for (counter=0; counter < (nwp->nnodes - nwp->bipartite); counter++){
                        CovPattern[counter + nwp->bipartite] = (int)round(mtp2->inputparams[mtp2->ninputparams - (nwp->nnodes - nwp->bipartite) + counter]);        
                }

                mtp2++;  

if (print_info_MH == 1) {                
Rprintf("Term 4: "); 
for (counter=0; counter < mtp2->ninputparams; counter++){
        Rprintf(" %d ",(int)round(mtp2->inputparams[counter]));        
}
Rprintf("\n");  

                

Rprintf("Max Degree %d \n",max_degree);
                

Rprintf("CovPattern: "); 
for (counter=0; counter < nwp->nnodes; counter++){
        Rprintf(" %d ", CovPattern[counter]);        
}
Rprintf("\n");  
}
             
                int max_Cov_type = 0;
                int Cov_type;
                for (counter=0; counter < nwp->nnodes; counter++) {
                    Cov_type = CovPattern[counter];
                    if (max_Cov_type < Cov_type) {
                        max_Cov_type = Cov_type;
                    }
                }

                max_Cov_type = 2 * max_Cov_type; //for type 1 nodes + type 2 nodes 
                        
if (print_info_MH == 1) {        
Rprintf("Stats \n: "); 
for (counter=0; counter < m->n_stats; counter++){
        Rprintf(" %d ", (int)networkstatistics[counter]);        
}
Rprintf("\n");  
}
       
                int cov_pattern[2]; //covariate pattern for each endpoint of toggled edge
                
                cov_pattern[0] = (int)round(CovPattern[*(MHp->toggletail) - 1]); //Minus 1 since node ids are from 1 to nnodes
                cov_pattern[1] = (int)round(CovPattern[*(MHp->togglehead) - 1]); //Minus 1 since node ids are from 1 to nnodes

if (print_info_MH == 1) {
    Rprintf("Num Cov Types: %d %d \n", cov_pattern[0], cov_pattern[1]); 

    Rprintf("WorkSpace (%d): ", m->n_stats); 
    for (int counter_print=0; counter_print < (m->n_stats); counter_print++){
        Rprintf(" %f ",m->workspace[counter_print]);        
    }
    Rprintf("\n");
}

                double *nwp_Deg_Distr_1;
                double *nwp_Deg_Distr_2;
                double *MHp_Deg_Distr_1;
                double *MHp_Deg_Distr_2;

                nwp_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));

                double *nwp_Deg_Distr_Edges_1;
                double *nwp_Deg_Distr_Edges_2;
                double *MHp_Deg_Distr_Edges_1;
                double *MHp_Deg_Distr_Edges_2;

                nwp_Deg_Distr_Edges_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_Deg_Distr_Edges_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_Edges_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_Deg_Distr_Edges_2 = malloc((*maxdegree+1) * sizeof(double));

                double nwp_z = 0;
                double MHp_z = 0;

                for (counter = 0; counter <= max_degree; counter++) {
                   nwp_Deg_Distr_1[counter] = 0; //+1 for degree 0
                   nwp_Deg_Distr_2[counter] = 0;
                   MHp_Deg_Distr_1[counter] = 0;
                   MHp_Deg_Distr_2[counter] = 0;
                   
                   nwp_Deg_Distr_Edges_1[counter] = 0; //+1 for degree 0
                   nwp_Deg_Distr_Edges_2[counter] = 0;
                   MHp_Deg_Distr_Edges_1[counter] = 0;
                   MHp_Deg_Distr_Edges_2[counter] = 0;
                }

                counter = 1; //skip number of edges
                for (counter1 = 0; counter1 < max_Cov_type; counter1++) {
                   for (counter2 = 0; counter2 <= max_degree; counter2++) {
                      m->workspace[counter] = 0;
                      counter++;
                   }
                }
                m->workspace[1 + Deg_nwp[0] + (cov_pattern[0]-1)*(max_degree+1)] = -1;
                m->workspace[1 + Deg_nwp[1] + (cov_pattern[1]-1 + (int)round(max_Cov_type/2))*(max_degree+1)] = -1;
                m->workspace[1 + Deg_MHp[0] + (cov_pattern[0]-1)*(max_degree+1)] = 1;
                m->workspace[1 + Deg_MHp[1] + (cov_pattern[1]-1 + (int)round(max_Cov_type/2))*(max_degree+1)] = 1;

if (print_info_MH == 1) {                
Rprintf("WorkSpace (%d): ", m->n_stats); 
for (int counter_print=0; counter_print < (m->n_stats); counter_print++){
    Rprintf(" %f ",m->workspace[counter_print]);        
}
Rprintf("\n");                
}
                
                counter = 1; //skip number of edges
                for (counter1 = 0; counter1 < (int)round(max_Cov_type/2); counter1++) {
                   for (counter2 = 0; counter2 <= max_degree; counter2++) {
                       nwp_Deg_Distr_1[counter2] = nwp_Deg_Distr_1[counter2] + (int)networkstatistics[counter];
                       nwp_Deg_Distr_2[counter2] = nwp_Deg_Distr_2[counter2] + (int)networkstatistics[counter + (int)round(max_Cov_type/2)*(max_degree+1)];
                      
                       MHp_Deg_Distr_1[counter2] = MHp_Deg_Distr_1[counter2] + (int)networkstatistics[counter] + (int)(m->workspace[counter]);
                       MHp_Deg_Distr_2[counter2] = MHp_Deg_Distr_2[counter2] + (int)networkstatistics[counter + (int)round(max_Cov_type/2)*(max_degree+1)] + (int)(m->workspace[counter + (int)round(max_Cov_type/2)*(max_degree+1)]);

                       nwp_Deg_Distr_Edges_1[counter2] = nwp_Deg_Distr_Edges_1[counter2] + counter2*(int)networkstatistics[counter];
                       nwp_Deg_Distr_Edges_2[counter2] = nwp_Deg_Distr_Edges_2[counter2] + counter2*(int)networkstatistics[counter + (int)round(max_Cov_type/2)*(max_degree+1)];
                      
                       MHp_Deg_Distr_Edges_1[counter2] = MHp_Deg_Distr_Edges_1[counter2] + counter2*(int)networkstatistics[counter] + (int)(m->workspace[counter]);
                       MHp_Deg_Distr_Edges_2[counter2] = MHp_Deg_Distr_Edges_2[counter2] + counter2*(int)networkstatistics[counter + (int)round(max_Cov_type/2)*(max_degree+1)] + (int)(m->workspace[counter + (int)round(max_Cov_type/2)*(max_degree+1)]);

                       nwp_z = nwp_z + counter2*(int)networkstatistics[counter];
                       MHp_z = MHp_z + counter2*((int)networkstatistics[counter] + (int)(m->workspace[counter]));

                       counter++;
                   }
                }

if (print_info_MH == 1) {                
Rprintf("NWP Degree Distr 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("NWP Degree Distr 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_2[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_1[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_2[counter]);        
}
Rprintf("\n");
}                      
                
if (print_info_MH == 1) {                
Rprintf("NWP Degree Distr Edges 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_Edges_1[counter]);        
}
Rprintf("\n");

Rprintf("NWP Degree Distr Edges 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",nwp_Deg_Distr_Edges_2[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr Edges 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_Edges_1[counter]);        
}
Rprintf("\n");

Rprintf("MHP Degree Distr Edges 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
    Rprintf(" %f ",MHp_Deg_Distr_Edges_2[counter]);        
}
Rprintf("\n");  

Rprintf("NWP Z %f MHP Z %f\n",nwp_z, MHp_z);

}                              

                // Construct expected degree distribution 

                double nwp_exp_dmm;
                double MHp_exp_dmm;

                nwp_exp_dmm = (nwp_Deg_Distr_Edges_1[Deg_nwp[0]] * nwp_Deg_Distr_Edges_2[Deg_nwp[1]])/ nwp_z;                      
                MHp_exp_dmm = (MHp_Deg_Distr_Edges_1[Deg_MHp[0]] * MHp_Deg_Distr_Edges_2[Deg_MHp[1]])/ MHp_z;

if (print_info_MH == 1) {                
Rprintf("NWP DMM %f MHP DMM %f\n",nwp_exp_dmm, MHp_exp_dmm);
} 

                if (nwp->nedges > MHp_nedges) {  //Edge is removed from g to g2
                        prob_g_g2 = nwp_exp_dmm;
                } else { //Edge is added from g to g2
                        prob_g_g2 = (nwp_Deg_Distr_1[Deg_nwp[0]] * nwp_Deg_Distr_2[Deg_nwp[1]])  - nwp_exp_dmm;
                }
                if (nwp->nedges < MHp_nedges) {  //Edge is removed from g2 to g
                        prob_g2_g = MHp_exp_dmm;
                } else {  //Edge is added from g2 to g
                        prob_g2_g = (double)((MHp_Deg_Distr_1[Deg_MHp[0]] * MHp_Deg_Distr_2[Deg_MHp[1]])  - MHp_exp_dmm);
                }

if (print_info_MH == 1) {
Rprintf("Moves g to g2: %f Moves g2 to g %f\n",prob_g_g2, prob_g2_g);
}
                
//////////////////////////                
                
           if ((prob_type[0] == 1) && (prob_type[1] == -1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                double *mean_Deg_Distr_1;
                double *mean_Deg_Distr_2;

                mean_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                mean_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));
                
                double *nwp_mu_diff_1;
                double *nwp_mu_diff_2;
                double *MHp_mu_diff_1;
                double *MHp_mu_diff_2;

                nwp_mu_diff_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_mu_diff_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_mu_diff_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_mu_diff_2 = malloc((*maxdegree+1) * sizeof(double));
    
if (print_info_MH == 1) {                
Rprintf("Mean Values (%d): ", (*maxdegree+1)*2); 
for (int counter_print=0; counter_print < ((*maxdegree+1)*2); counter_print++){
   Rprintf(" %f ",meanvalues[counter_print]);       
}
Rprintf("\n");                
}
                
                counter = 0; //skip number of edges
                for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
                       mean_Deg_Distr_1[counter2] = meanvalues[counter];
                       mean_Deg_Distr_2[counter2] = meanvalues[counter + (*maxdegree+1)];
                
                       counter++;
                }

                
if (print_info_MH == 1) {
Rprintf("Mean Degree Distribution (Type 1): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_1[counter2]);
}
Rprintf("\n");


Rprintf("Mean Degree Distribution (Type 2): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_2[counter2]);
}
   
}


//Variance 

//int length_varvalues = (max_degree+1) * (max_degree+1) *(max_Cov_type) + (int)(pow((int)(max_Cov_type/2), 4));
//Rprintf("Var Values (%d): ", length_varvalues); 
//for (int counter_print=0; counter_print < length_varvalues; counter_print++){
//   Rprintf(" %f ",varvalues[counter_print]);       
//}
//Rprintf("\n");                
                
                double *varvalues_1;
                double *varvalues_2;

                varvalues_1 = malloc((*maxdegree+1)*(*maxdegree+1) * sizeof(double));
                varvalues_2 = malloc((*maxdegree+1)*(*maxdegree+1) * sizeof(double));
                
                counter = 0; //skip number of edges
                for (counter2 = 0; counter2 < ((*maxdegree+1)*(*maxdegree+1)); counter2++) {
                       varvalues_1[counter2] = varvalues[counter];
                       varvalues_2[counter2] = varvalues[counter + (*maxdegree+1)*(*maxdegree+1)];
                
                       counter++;
                }

if (print_info_MH == 1) {                
Rprintf("Var Degree Distribution (Type 1): \n");                
for (counter2 = 0; counter2 < (*maxdegree+1)*(*maxdegree+1); counter2++) {
    Rprintf(" %f ", varvalues_1[counter2]);
}
Rprintf("\n");

Rprintf("Var Degree Distribution (Type 2): \n");                
for (counter2 = 0; counter2 < (*maxdegree+1)*(*maxdegree+1); counter2++) {
    Rprintf(" %f ", varvalues_2[counter2]);
}
Rprintf("\n");                 
}

                
//end of variance
                
                int nnodes_type1 = 0;
                int nnodes_type2 = 0;
                                                           
                for (counter = 0; counter <= *maxdegree; counter++) {
                   nnodes_type1 += nwp_Deg_Distr_1[counter];
                   nnodes_type2 += nwp_Deg_Distr_2[counter];
                }
if (print_info_MH == 1) {
Rprintf("Number of Nodes %d %d \n",nnodes_type1, nnodes_type2); 
}

                    for (counter = 0; counter <= *maxdegree; counter++) {
                        nwp_mu_diff_1[counter] = (double)(nwp_Deg_Distr_1[counter])/(1.0*nnodes_type1) - mean_Deg_Distr_1[counter];
                        MHp_mu_diff_1[counter] = (double)(MHp_Deg_Distr_1[counter])/(1.0*nnodes_type1) - mean_Deg_Distr_1[counter];
                        
                        nwp_mu_diff_2[counter] = (double)(nwp_Deg_Distr_2[counter])/(1.0*nnodes_type2) - mean_Deg_Distr_2[counter];
                        MHp_mu_diff_2[counter] = (double)(MHp_Deg_Distr_2[counter])/(1.0*nnodes_type2) - mean_Deg_Distr_2[counter];
                    }

if (print_info_MH == 1) {
Rprintf("nwp mu diff 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",nwp_mu_diff_1[counter]);        
}
Rprintf("\n");

Rprintf("nwp mu diff 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",nwp_mu_diff_2[counter]);        
}
Rprintf("\n");

Rprintf("MHp mu diff 1: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",MHp_mu_diff_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp mu diff 2: "); 
for (counter=0; counter <= *maxdegree; counter++){
        Rprintf(" %f ",MHp_mu_diff_2[counter]);        
}
Rprintf("\n");

}
                double *nwp_intermediate_mat_1;
                double *nwp_intermediate_mat_2;
                double *MHp_intermediate_mat_1;
                double *MHp_intermediate_mat_2;

                nwp_intermediate_mat_1 = malloc((*maxdegree+1) * sizeof(double));
                nwp_intermediate_mat_2 = malloc((*maxdegree+1) * sizeof(double));
                MHp_intermediate_mat_1 = malloc((*maxdegree+1) * sizeof(double));
                MHp_intermediate_mat_2 = malloc((*maxdegree+1) * sizeof(double));
                              
                counter2 = -1;
                for (counter = 0; counter < ((*maxdegree+1) * (*maxdegree+1)); counter++) {
                    counter1 = counter % ((*maxdegree+1));
                    if (counter1 == 0) {
                        counter2 += 1;
                        nwp_intermediate_mat_1[counter2] = 0;
                        MHp_intermediate_mat_1[counter2] = 0;
                        
                        nwp_intermediate_mat_2[counter2] = 0;
                        MHp_intermediate_mat_2[counter2] = 0;
                    }
                    nwp_intermediate_mat_1[counter2] += nwp_mu_diff_1[counter1]*varvalues_1[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mat_1[counter2] += MHp_mu_diff_1[counter1]*varvalues_1[counter];
                    nwp_intermediate_mat_2[counter2] += nwp_mu_diff_2[counter1]*varvalues_2[counter];
//Rprintf("Check MHp: %d %d %d %f\n",counter, counter1, counter2, (MHp_mu_diff[counter1]*varvalues[counter]));
                    MHp_intermediate_mat_2[counter2] += MHp_mu_diff_2[counter1]*varvalues_2[counter];                      
                }
if (print_info_MH == 1) {                
Rprintf("nwp Intermediate (Type 1): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",nwp_intermediate_mat_1[counter]);        
}
Rprintf("\n");

Rprintf("nwp Intermediate (Type 2): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",nwp_intermediate_mat_2[counter]);        
}
Rprintf("\n");

Rprintf("MHp Intermediate (Type 1): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",MHp_intermediate_mat_1[counter]);        
}
Rprintf("\n");

Rprintf("MHp Intermediate (Type 2): "); 
for (counter=0; counter < (*maxdegree+1); counter++){
        Rprintf(" %f ",MHp_intermediate_mat_2[counter]);        
}
Rprintf("\n");
}

                double pdf_gaussian_nwp_1 = 0;
                double pdf_gaussian_MHp_1 = 0;
                
                double pdf_gaussian_nwp_2 = 0;
                double pdf_gaussian_MHp_2 = 0;
                
                for (counter = 0; counter <= *maxdegree; counter++) {
                    pdf_gaussian_nwp_1 += nwp_intermediate_mat_1[counter] * nwp_mu_diff_1[counter];
                    pdf_gaussian_MHp_1 += MHp_intermediate_mat_1[counter] * MHp_mu_diff_1[counter];
                    
                    pdf_gaussian_nwp_2 += nwp_intermediate_mat_2[counter] * nwp_mu_diff_2[counter];
                    pdf_gaussian_MHp_2 += MHp_intermediate_mat_2[counter] * MHp_mu_diff_2[counter];
                }
                
                pdf_gaussian_nwp =  (-.5 * pdf_gaussian_nwp_1) + (-.5 * pdf_gaussian_nwp_2);
                pdf_gaussian_MHp =  (-.5 * pdf_gaussian_MHp_1) + (-.5 * pdf_gaussian_MHp_2);
                
if (print_info_MH == 1) {                
Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                
                
                free(nwp_mu_diff_1);
                free(nwp_mu_diff_2);
                free(MHp_mu_diff_1);
                free(MHp_mu_diff_2);

                free(mean_Deg_Distr_1);
                free(mean_Deg_Distr_2);
               
                free(varvalues_1);
                free(varvalues_2);

                free(nwp_intermediate_mat_1);
                free(nwp_intermediate_mat_2);
                free(MHp_intermediate_mat_1);
                free(MHp_intermediate_mat_2);
                
                }                
 //////////////               
     
                
           if ((prob_type[0] == 2) && (prob_type[1] == -1) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] >= 1)){

                double *mean_Deg_Distr_1;
                double *mean_Deg_Distr_2;

                mean_Deg_Distr_1 = malloc((*maxdegree+1) * sizeof(double));
                mean_Deg_Distr_2 = malloc((*maxdegree+1) * sizeof(double));
                   
if (print_info_MH == 1) {                
Rprintf("Mean Values (%d): ", (*maxdegree+1)*2); 
for (int counter_print=0; counter_print < ((*maxdegree+1)*2); counter_print++){
   Rprintf(" %f ",meanvalues[counter_print]);       
}
Rprintf("\n");                
}
                
                counter = 0; //skip number of edges
                for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
                       mean_Deg_Distr_1[counter2] = meanvalues[counter];
                       mean_Deg_Distr_2[counter2] = meanvalues[counter + (*maxdegree+1)];
                
                       counter++;
                }

                
if (print_info_MH == 1) {
Rprintf("Mean Degree Distribution (Type 1): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_1[counter2]);
}
Rprintf("\n");


Rprintf("Mean Degree Distribution (Type 2): \n");                
for (counter2 = 0; counter2 <= *maxdegree; counter2++) {
    Rprintf(" %f ", mean_Deg_Distr_2[counter2]);
}
   
}

                int nnodes_type1 = 0;
                int nnodes_type2 = 0;
                                                           
                for (counter = 0; counter <= *maxdegree; counter++) {
                   nnodes_type1 += nwp_Deg_Distr_1[counter];
                   nnodes_type2 += nwp_Deg_Distr_2[counter];
                }
if (print_info_MH == 1) {
Rprintf("Number of Nodes %d %d \n",nnodes_type1, nnodes_type2); 
}

                
                double pdf_gaussian_nwp_1 = 0;
                double pdf_gaussian_MHp_1 = 0;
                
                double pdf_gaussian_nwp_2 = 0;
                double pdf_gaussian_MHp_2 = 0;
                
                for (counter = 0; counter <= *maxdegree; counter++) {
                        pdf_gaussian_nwp_1 += (mean_Deg_Distr_1[counter]-1) * log((double)(nwp_Deg_Distr_1[counter])/(1.0*nnodes_type1));
                        pdf_gaussian_MHp_1 += (mean_Deg_Distr_1[counter]-1) * log((double)(MHp_Deg_Distr_1[counter])/(1.0*nnodes_type1));
                        
                        pdf_gaussian_nwp_2 += (mean_Deg_Distr_2[counter]-1) * log((double)(nwp_Deg_Distr_2[counter])/(1.0*nnodes_type2));
                        pdf_gaussian_MHp_2 += (mean_Deg_Distr_2[counter]-1) * log((double)(MHp_Deg_Distr_2[counter])/(1.0*nnodes_type2));
                }
                
                pdf_gaussian_nwp =  (pdf_gaussian_nwp_1) + (pdf_gaussian_nwp_2);
                pdf_gaussian_MHp =  (pdf_gaussian_MHp_1) + (pdf_gaussian_MHp_2);
                
if (print_info_MH == 1) {                
Rprintf("Final Probs %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
}
                                
                if (!isfinite(pdf_gaussian_nwp)) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 1: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }
                
                if (pdf_gaussian_nwp != pdf_gaussian_nwp) {
                        prob_g2_g = 1;
                        pdf_gaussian_MHp = 0;
                        prob_g_g2 = 1;
                        pdf_gaussian_nwp = log(0);
                        
//if (print_info_MH == 1) {                
Rprintf("NWP INVALID 2: %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
//}                        
                }

                free(mean_Deg_Distr_1);
                free(mean_Deg_Distr_2);
                
                }                
                     
                
//////////////////////////                
                free(CovPattern);

                free(nwp_Deg_Distr_1);
                free(nwp_Deg_Distr_2);
                free(MHp_Deg_Distr_1);
                free(MHp_Deg_Distr_2);
                
                free(nwp_Deg_Distr_Edges_1);
                free(nwp_Deg_Distr_Edges_2);
                free(MHp_Deg_Distr_Edges_1);
                free(MHp_Deg_Distr_Edges_2);   
                
            }
        }

////////////////////////
//End: Mixing Nested in Degree Dist //
////////////////////////  
    
            
        //Rprintf("Probs (before cutoff): %f %f %f %f\n", prob_g_g2, prob_g2_g, pdf_gaussian_nwp, pdf_gaussian_MHp);
            
            cutoff = (log(prob_g2_g) + pdf_gaussian_MHp) - (log(prob_g_g2) + pdf_gaussian_nwp) + MHp->logratio;
            
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
