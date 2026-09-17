#include <R.h>
#include <math.h>
#include "CCMnet_netprop_degdist.h"

void calc_stat_degdist(Model *m, 
                       int *NetworkForecast, 
                       int *num_deg_stats,
                       int *Deg_Add, 
                       int *Deg_Delete, 
                       int *Proposal_prob_zero) {
  
  int counter;
  int Deg_Add_counter = 0;
  int Deg_Delete_counter = 0;
  double changestat_sum = 0;
  
  /* Calculate the number of degree statistics */
  *num_deg_stats = m->n_stats - 1 - (*NetworkForecast);
  *Proposal_prob_zero = 0;
  
  /* Get Degree changes of tail and head from workspace */
  for (counter = 1; counter < (*num_deg_stats + 1); counter++) {
    changestat_sum += m->workspace[counter];
    
    // Edge toggle results in degree decrease for a node
    if (round(m->workspace[counter]) == -1) {
      Deg_Delete[Deg_Delete_counter] = counter - 1;
      Deg_Delete_counter++;
    }
    // Edge toggle results in degree increase for a node
    if (round(m->workspace[counter]) == 1) {
      Deg_Add[Deg_Add_counter] = counter - 1;
      Deg_Add_counter++;
    }
    // Special case: Both toggle nodes had same degree, both decreased
    if (round(m->workspace[counter]) == -2) {
      Deg_Delete[0] = counter - 1;
      Deg_Delete[1] = counter - 1;
    }
    // Special case: Both toggle nodes had same degree, both increased
    if (round(m->workspace[counter]) == 2) {
      Deg_Add[0] = counter - 1;
      Deg_Add[1] = counter - 1;
    }
  }
  
  /* Handling cases with single node changes */
  if (Deg_Add_counter == 1) {
    Deg_Delete[1] = (int)(((Deg_Delete[0] + Deg_Add[0]) * .5) + .5);
    Deg_Add[1] = Deg_Delete[1];
  }
  
  /* Check for proposal validity based on observed degrees */
  if (fabs(changestat_sum) > 0.1) { 
    *Proposal_prob_zero = 1;
  }
}

void calc_f_degdist(int num_deg_stats, Model *m, double *networkstatistics,
                    int nwp_nedges, int MHp_nedges, int *Deg_Delete, int *Deg_Add,
                    double *prob_g_g2, double *prob_g2_g,
                    int *nwp_Deg_Distr, int *MHp_Deg_Distr) {

  int counter;
  double nwp_exp_dmm, MHp_exp_dmm;
  int MHp_Deg_Distr_Edges[num_deg_stats];
  int nwp_Deg_Distr_Edges[num_deg_stats];

  /* Construct Degree Distribution and number of edges associated with each degree */
  for (counter = 1; counter < (num_deg_stats + 1); counter++) {
    MHp_Deg_Distr[counter - 1] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
    nwp_Deg_Distr[counter - 1] = (int)networkstatistics[counter];
    MHp_Deg_Distr_Edges[counter - 1] = MHp_Deg_Distr[counter - 1] * (counter - 1);
    nwp_Deg_Distr_Edges[counter - 1] = nwp_Deg_Distr[counter - 1] * (counter - 1);
  }

  /* Construct expected degree distribution */
  if (networkstatistics[0] > 0) {
    nwp_exp_dmm = (nwp_Deg_Distr_Edges[Deg_Delete[0]] * (double)nwp_Deg_Distr_Edges[Deg_Delete[1]]) / (2.0 * networkstatistics[0]);
    if (Deg_Delete[0] == Deg_Delete[1]) {
      nwp_exp_dmm = nwp_exp_dmm * 0.5;
    }
  } else {
    nwp_exp_dmm = 0.0; // At zero edges, expected mixing is zero
  }

  if (MHp_nedges > 0) {
    MHp_exp_dmm = (MHp_Deg_Distr_Edges[Deg_Add[0]] * (double)MHp_Deg_Distr_Edges[Deg_Add[1]]) / (2.0 * (double)MHp_nedges);
    if (Deg_Add[0] == Deg_Add[1]) {
      MHp_exp_dmm = MHp_exp_dmm * 0.5;
    }
  } else {
    MHp_exp_dmm = 0.0;
  }

  /* Assign prob_g_g2 */
  if (nwp_nedges > MHp_nedges) { // Edge Removed nwp -> MHp
    *prob_g_g2 = nwp_exp_dmm;
  } else { // Edge Added nwp -> MHp
    if (Deg_Add[0] == Deg_Add[1]) {
      *prob_g_g2 = (nwp_Deg_Distr[Deg_Delete[0]] * (nwp_Deg_Distr[Deg_Delete[1]] - 1) * 0.5) - nwp_exp_dmm;
    } else {
      *prob_g_g2 = (double)nwp_Deg_Distr[Deg_Delete[0]] * nwp_Deg_Distr[Deg_Delete[1]] - nwp_exp_dmm;
    }
  }

  /* Assign prob_g2_g */
  if (nwp_nedges < MHp_nedges) { // Edge Removed MHp -> nwp
    *prob_g2_g = MHp_exp_dmm;
  } else { // Edge Added MHp -> nwp
    if (Deg_Add[0] == Deg_Add[1]) {
      *prob_g2_g = (MHp_Deg_Distr[Deg_Add[0]] * (MHp_Deg_Distr[Deg_Add[1]] - 1) * 0.5) - MHp_exp_dmm;
    } else {
      *prob_g2_g = (double)MHp_Deg_Distr[Deg_Add[0]] * MHp_Deg_Distr[Deg_Add[1]] - MHp_exp_dmm;
    }
  }
}

