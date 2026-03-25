#include <R.h>
#include <math.h>
#include "CCMnet_netprop_degmixing.h"

void get_properties(Model *m, double *stats, int num_deg_stats, 
                    int g_dmm[num_deg_stats-1][num_deg_stats-1], 
                                              int g2_dmm[num_deg_stats-1][num_deg_stats-1]) {
  
  int counter = 1; // Start at 1 to skip the 'Edges' statistic
  
  for (int i = 0; i < (num_deg_stats - 1); i++) {
    for (int j = 0; j <= i; j++) {
      // Populate current matrix (g)
      g_dmm[i][j] = (int)stats[counter];
      g_dmm[j][i] = (int)stats[counter];
      
      // Populate proposed matrix (g2)
      int val_g2 = (int)(stats[counter] + m->workspace[counter]);
      g2_dmm[i][j] = val_g2;
      g2_dmm[j][i] = val_g2;
      
      counter++;
    }
  }
}


void calculate_congruence_ratios(
    Network *nwp,
    MHproposal *MHp,
    int num_deg_stats,
    int g_dmm[][num_deg_stats-1],
    int g2_dmm[][num_deg_stats-1],
    double *prob_g_g2,
    double *prob_g2_g,
    int MHp_nedges,
    double *pdf_gaussian_nwp,
    double *pdf_gaussian_MHp,
    int *Proposal_prob_zero,
    int deg_dist_nwp[num_deg_stats],
    int deg_dist_MHp[num_deg_stats],
    int Deg_nwp[2],
    int Deg_MHp[2]
) {

  Edge nextedge=0;
  int nmax = 100000;
  int index1;
  int index2;

  //calc_f: BEGIN
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
    *Proposal_prob_zero = 1;
  }

  if (*Proposal_prob_zero == 1) {
    *prob_g2_g = 1.0;
    *pdf_gaussian_MHp = log(0);
    *prob_g_g2 = 1.0;
    *pdf_gaussian_nwp = 0;
  } else {

    //Step 2: begin - Find degrees of neighbors
    int n_i[num_deg_stats-1]; //Do not need degree zero
    int n_j[num_deg_stats-1];

    for (int counter = 0; counter < (num_deg_stats-1); counter++) {
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
        *prob_g_g2 = deg_dist_nwp[Deg_nwp[0]] * (deg_dist_nwp[Deg_nwp[0]] - 1) * .5;
      } else {
        *prob_g_g2 = deg_dist_nwp[Deg_nwp[0]] * (deg_dist_nwp[Deg_nwp[1]]);
      }
      if (Deg_nwp[0] > 0 && Deg_nwp[1] > 0 ) {
        *prob_g_g2 = *prob_g_g2 - g_dmm[Deg_nwp[0]-1][Deg_nwp[1]-1];
      }


      for (index1=0; index1 < ((num_deg_stats)-1); index1++){
        g_dmm[index1][index1] = 2 * g_dmm[index1][index1];
      }

      //Step 3b: begin - ADD identical degrees

      numerator = 1;

      if ((Deg_nwp[0] == Deg_nwp[1]) && (Deg_nwp[0] > 0)) {
        denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0]), (Deg_nwp[0] + Deg_nwp[1]));
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter] + n_j[counter]));
        }
        *prob_g_g2 = *prob_g_g2 * ((double)numerator / denominator);
      } else {
        //Step 3c: begin - ADD degree Node 1

        if (Deg_nwp[0] > 0) {
          denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0]), (Deg_nwp[0]));
          numerator = 1;
          for (int counter=0; counter < (num_deg_stats-1); counter++) {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter]));
          }
          *prob_g_g2 = *prob_g_g2 * ((double)numerator / denominator);
        }
        //Step 3d: begin - ADD degree Node 2
        if (Deg_nwp[1] > 0) {
          denominator = calcCNR( (deg_dist_nwp[Deg_nwp[1]] * Deg_nwp[1]), (Deg_nwp[1]));
          numerator = 1;
          for (int counter=0; counter < (num_deg_stats-1); counter++) {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[1]-1][counter], (n_j[counter]));
          }
          *prob_g_g2 = *prob_g_g2 * ((double)numerator / denominator);
        }
      }
    } else {
      //Step 4a: begin - Remove
      *prob_g_g2 = g_dmm[Deg_nwp[0]-1][Deg_nwp[1]-1];

      for (index1=0; index1 < ((num_deg_stats)-1); index1++){
        g_dmm[index1][index1] = 2 * g_dmm[index1][index1];
      }

      //Step 4b: begin - ADD identical degrees
      numerator = 1;

      if (Deg_nwp[0] == Deg_nwp[1]) {
        denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0] -1 ), (Deg_nwp[0] + Deg_nwp[1] - 2));
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          if (counter == (Deg_nwp[0]-1)) {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter]-1, (n_i[counter] + n_j[counter] - 2));
          } else {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter] + n_j[counter]));
          }
        }
        *prob_g_g2 = *prob_g_g2 * ((double)numerator / denominator);
      } else {
        //Step 4c: begin - ADD node 1 degrees
        denominator = calcCNR( (deg_dist_nwp[Deg_nwp[0]] * Deg_nwp[0] - 1), (Deg_nwp[0] - 1));
        numerator = 1;
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          if (counter == (Deg_nwp[1]-1)) {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter]-1, (n_i[counter] - 1));
          } else {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[0]-1][counter], (n_i[counter]));
          }
        }
        *prob_g_g2 = *prob_g_g2 * ((double)numerator / denominator);
        //Step 4d: begin - ADD node 2 degrees
        denominator = calcCNR( (deg_dist_nwp[Deg_nwp[1]] * Deg_nwp[1] - 1), (Deg_nwp[1] - 1));
        numerator = 1;
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          if (counter == (Deg_nwp[0]-1)) {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[1]-1][counter]-1, (n_j[counter]-1));
          } else {
            numerator = numerator * calcCNR(g_dmm[Deg_nwp[1]-1][counter], (n_j[counter]));
          }
        }
        *prob_g_g2 = *prob_g_g2 * ((double)numerator / denominator);

      }
    }

    ////g2 -> g
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
        *prob_g2_g = deg_dist_MHp[Deg_MHp[0]] * (deg_dist_nwp[Deg_MHp[0]] - 1) * .5;
      } else {
        *prob_g2_g = deg_dist_MHp[Deg_MHp[0]] * (deg_dist_nwp[Deg_MHp[1]]);
      }
      if (Deg_MHp[0] > 0 && Deg_MHp[1] > 0 ) {
        *prob_g2_g = *prob_g2_g - g2_dmm[Deg_MHp[0]-1][Deg_MHp[1]-1];
      }

      for (index1=0; index1 < ((num_deg_stats)-1); index1++){
        g2_dmm[index1][index1] = 2 * g2_dmm[index1][index1];
      }

      //Step 3b: begin - ADD identical degrees

      numerator = 1;

      if ((Deg_MHp[0] == Deg_MHp[1]) && (Deg_MHp[0] > 0)) {
        denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0]), (Deg_MHp[0] + Deg_MHp[1]));
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter] + n_j[counter]));
        }
        *prob_g2_g = *prob_g2_g * ((double)numerator / denominator);
      } else {
        //Step 3c: begin - ADD degree Node 1

        if (Deg_MHp[0] > 0) {
          denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0]), (Deg_MHp[0]));
          numerator = 1;
          for (int counter=0; counter < (num_deg_stats-1); counter++) {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter]));
          }
          *prob_g2_g = *prob_g2_g * ((double)numerator / denominator);
        }
        //Step 3d: begin - ADD degree Node 2
        if (Deg_MHp[1] > 0) {
          denominator = calcCNR( (deg_dist_MHp[Deg_MHp[1]] * Deg_MHp[1]), (Deg_MHp[1]));
          numerator = 1;
          for (int counter=0; counter < (num_deg_stats-1); counter++) {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[1]-1][counter], (n_j[counter]));
          }
          *prob_g2_g = *prob_g2_g * ((double)numerator / denominator);
        }
      }
    } else {
      //Step 4a: begin - Remove
      *prob_g2_g = g2_dmm[Deg_MHp[0]-1][Deg_MHp[1]-1];
      //Step 4b: begin - ADD identical degrees

      for (index1=0; index1 < ((num_deg_stats)-1); index1++){
        g2_dmm[index1][index1] = 2 * g2_dmm[index1][index1];
      }

      numerator = 1;

      if (Deg_MHp[0] == Deg_MHp[1]) {
        denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0] -1 ), (Deg_MHp[0] + Deg_MHp[1] - 2));
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          if (counter == (Deg_MHp[0]-1)) {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter]-1, (n_i[counter] + n_j[counter] - 2));
          } else {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter] + n_j[counter]));
          }
        }
        *prob_g2_g = *prob_g2_g * ((double)numerator / denominator);
      } else {
        //Step 4c: begin - ADD node 1 degrees
        denominator = calcCNR( (deg_dist_MHp[Deg_MHp[0]] * Deg_MHp[0] - 1), (Deg_MHp[0] - 1));
        numerator = 1;
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          if (counter == (Deg_MHp[1]-1)) {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter]-1, (n_i[counter] - 1));
          } else {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[0]-1][counter], (n_i[counter]));
          }
        }
        *prob_g2_g = *prob_g2_g * ((double)numerator / denominator);
        //Step 4d: begin - ADD node 2 degrees
        denominator = calcCNR( (deg_dist_MHp[Deg_MHp[1]] * Deg_MHp[1] - 1), (Deg_MHp[1] - 1));
        numerator = 1;
        for (int counter=0; counter < (num_deg_stats-1); counter++) {
          if (counter == (Deg_MHp[0]-1)) {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[1]-1][counter]-1, (n_j[counter]-1));
          } else {
            numerator = numerator * calcCNR(g2_dmm[Deg_MHp[1]-1][counter], (n_j[counter]));
          }
        }
        *prob_g2_g = *prob_g2_g * ((double)numerator / denominator);
      }
    }

    for (index1=0; index1 < ((num_deg_stats)-1); index1++){ //Undo what was done for degree mixing
      g_dmm[index1][index1] = .5 * g_dmm[index1][index1];
      g2_dmm[index1][index1] = .5 * g2_dmm[index1][index1];
    }
    //calc_f: END

}
}


