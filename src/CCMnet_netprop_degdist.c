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
  if (changestat_sum < -.1) { 
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
  nwp_exp_dmm = (nwp_Deg_Distr_Edges[Deg_Delete[0]] * (double)nwp_Deg_Distr_Edges[Deg_Delete[1]]) / (2.0 * networkstatistics[0]);
  if (Deg_Delete[0] == Deg_Delete[1]) {
    nwp_exp_dmm = nwp_exp_dmm * 0.5;
  }

  MHp_exp_dmm = (MHp_Deg_Distr_Edges[Deg_Add[0]] * (double)MHp_Deg_Distr_Edges[Deg_Add[1]]) / (2.0 * (double)MHp_nedges);
  if (Deg_Add[0] == Deg_Add[1]) {
    MHp_exp_dmm = MHp_exp_dmm * 0.5;
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

void calc_probs_degdist(int num_deg_stats, Network *nwp, int *prob_type,
                        double *meanvalues, double *varvalues,
                        int *nwp_Deg_Distr, int *MHp_Deg_Distr,
                        double *pdf_gaussian_nwp, double *pdf_gaussian_MHp) {

  int counter, counter1, counter2;

  // Type 1, 4, and 5 use similar Multivariate logic
  if ((prob_type[1] == 0) && (prob_type[2] == 0) && (prob_type[3] == 0) && (prob_type[4] == 1)) {

    // Multivariate Gaussian (1), Student-t (4), or Logit-Gaussian (5)
    if (prob_type[0] == 1 || prob_type[0] == 4 || prob_type[0] == 5) {
      double nwp_mu_diff[num_deg_stats];
      double MHp_mu_diff[num_deg_stats];
      double nwp_intermediate_mat[num_deg_stats];
      double MHp_intermediate_mat[num_deg_stats];

      for (counter = 0; counter < num_deg_stats; counter++) {
        if (prob_type[0] == 5) { // Logit transform
          double nwp_dens = ((double)nwp_Deg_Distr[counter] / (1.0 * nwp->nnodes));
          double MHp_dens = ((double)MHp_Deg_Distr[counter] / (1.0 * nwp->nnodes));
          nwp_mu_diff[counter] = log(nwp_dens / (1.0 - nwp_dens) + .00001) - meanvalues[counter];
          MHp_mu_diff[counter] = log(MHp_dens / (1.0 - MHp_dens) + .00001) - meanvalues[counter];
        } else { // Standard density
          nwp_mu_diff[counter] = (double)nwp_Deg_Distr[counter] / (1.0 * nwp->nnodes) - meanvalues[counter];
          MHp_mu_diff[counter] = (double)MHp_Deg_Distr[counter] / (1.0 * nwp->nnodes) - meanvalues[counter];
        }
      }

      // Matrix Multiplication (x' * Sigma_inv)
      counter2 = -1;
      for (counter = 0; counter < (num_deg_stats * num_deg_stats); counter++) {
        counter1 = counter % num_deg_stats;
        if (counter1 == 0) {
          counter2 += 1;
          nwp_intermediate_mat[counter2] = 0;
          MHp_intermediate_mat[counter2] = 0;
        }
        nwp_intermediate_mat[counter2] += nwp_mu_diff[counter1] * varvalues[counter];
        MHp_intermediate_mat[counter2] += MHp_mu_diff[counter1] * varvalues[counter];
      }

      double sum_nwp = 0, sum_MHp = 0;
      for (counter = 0; counter < num_deg_stats; counter++) {
        sum_nwp += nwp_intermediate_mat[counter] * nwp_mu_diff[counter];
        sum_MHp += MHp_intermediate_mat[counter] * MHp_mu_diff[counter];
      }

      if (prob_type[0] == 4) { // Student-t
        *pdf_gaussian_nwp = -(meanvalues[num_deg_stats] + num_deg_stats) * .5 * log(1 + (1/meanvalues[num_deg_stats]) * sum_nwp);
        *pdf_gaussian_MHp = -(meanvalues[num_deg_stats] + num_deg_stats) * .5 * log(1 + (1/meanvalues[num_deg_stats]) * sum_MHp);
      } else { // Gaussian or Logit-Gaussian
        *pdf_gaussian_nwp = -0.5 * sum_nwp;
        *pdf_gaussian_MHp = -0.5 * sum_MHp;
      }
    }

    // Negative Binomial (Type 2)
    else if (prob_type[0] == 2) {
      int r_value = (int)round(meanvalues[0]);
      double p_value = meanvalues[1];
      double sum_log_i = 0;
      *pdf_gaussian_nwp = 0; *pdf_gaussian_MHp = 0;

      for (counter = 1; counter <= (r_value - 1); counter++) sum_log_i += log(counter);

      *pdf_gaussian_nwp += nwp_Deg_Distr[0] * sum_log_i;
      *pdf_gaussian_MHp += MHp_Deg_Distr[0] * sum_log_i;

      for (counter = 1; counter < num_deg_stats; counter++) {
        sum_log_i += log(counter + r_value - 1) - log(counter);
        *pdf_gaussian_nwp += nwp_Deg_Distr[counter] * (sum_log_i + counter * log(p_value));
        *pdf_gaussian_MHp += MHp_Deg_Distr[counter] * (sum_log_i + counter * log(p_value));
      }
    }

    // Poisson-like? Dirichlet-Multinomial
    else if (prob_type[0] == 3) {
      *pdf_gaussian_nwp = 0; *pdf_gaussian_MHp = 0;
      for (counter = 0; counter < num_deg_stats; counter++) {
        double g_nwp = 0, g_MHp = 0;
        int mv = (int)round(meanvalues[counter]);
        for (counter2 = (nwp_Deg_Distr[counter] + 1); counter2 <= (nwp_Deg_Distr[counter] + mv); counter2++)
          g_nwp += log(counter2);
        for (counter2 = (MHp_Deg_Distr[counter] + 1); counter2 <= (MHp_Deg_Distr[counter] + mv); counter2++)
          g_MHp += log(counter2);
        *pdf_gaussian_nwp += g_nwp;
        *pdf_gaussian_MHp += g_MHp;
      }
    }
  }
}
