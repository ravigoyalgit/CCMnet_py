#include "CCMnet_netprop_mixing_degdist.h"
#include <math.h>


void calculate_congruence_ratio_degmix(
    Model *m, Network *nwp, MHproposal *MHp,
    int length_deg_dist, int MHp_nedges,
    int *Deg_nwp, int *Deg_MHp, int *Cov_types,
    double *networkstatistics,
    int *nwp_Deg_Distr_1, int *nwp_Deg_Distr_2, int *MHp_Deg_Distr_1, int *MHp_Deg_Distr_2,
    double *nwp_mixing_matrix, double *MHp_mixing_matrix,
    double *prob_g_g2, double *prob_g2_g,
    double *pdf_gaussian_nwp, double *pdf_gaussian_MHp, int *Proposal_prob_zero
) {
  int counter;
  
  // Arrays for local distribution calculations
  double MHp_Deg_Distr_Edges_1[length_deg_dist];
  double MHp_Deg_Distr_Edges_2[length_deg_dist];
  double nwp_Deg_Distr_Edges_1[length_deg_dist];
  double nwp_Deg_Distr_Edges_2[length_deg_dist];
  double nwp_prob_mixing[2];
  double MHp_prob_mixing[2];
  double nwp_exp_dmm, MHp_exp_dmm;
  
  // 1. Bound Check
  if ((Deg_MHp[0] > (length_deg_dist - 1)) || (Deg_MHp[1] > (length_deg_dist - 1))) {
    *Proposal_prob_zero = 1;
  }
  
  if (*Proposal_prob_zero == 1) {
    *prob_g2_g = 1;
    *pdf_gaussian_MHp = log(0);
    *prob_g_g2 = 1;
    *pdf_gaussian_nwp = 0;
  } else {
    // 2. Map Covariate Types
    ModelTerm *mtp2 = m->termarray;
    mtp2++;
    Cov_types[0] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->toggletail) - 1]);
    Cov_types[1] = (int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + *(MHp->togglehead) - 1]);
    
    // 3. Mixing Probabilities Logic
    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
      nwp_prob_mixing[0] = (float)(2 * nwp_mixing_matrix[0]) / (float)(2 * nwp_mixing_matrix[0] + nwp_mixing_matrix[1]);
      nwp_prob_mixing[1] = nwp_prob_mixing[0];
      MHp_prob_mixing[0] = (float)(2 * MHp_mixing_matrix[0]) / (float)(2 * MHp_mixing_matrix[0] + MHp_mixing_matrix[1]);
      MHp_prob_mixing[1] = MHp_prob_mixing[0];
    } else if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
      nwp_prob_mixing[0] = (float)(2 * nwp_mixing_matrix[2]) / (float)(2 * nwp_mixing_matrix[2] + nwp_mixing_matrix[1]);
      nwp_prob_mixing[1] = nwp_prob_mixing[0];
      MHp_prob_mixing[0] = (float)(2 * MHp_mixing_matrix[2]) / (float)(2 * MHp_mixing_matrix[2] + MHp_mixing_matrix[1]);
      MHp_prob_mixing[1] = MHp_prob_mixing[0];
    } else {
      nwp_prob_mixing[0] = (float)(nwp_mixing_matrix[1]) / (float)(2 * nwp_mixing_matrix[0] + nwp_mixing_matrix[1]);
      nwp_prob_mixing[1] = (float)(nwp_mixing_matrix[1]) / (float)(2 * nwp_mixing_matrix[2] + nwp_mixing_matrix[1]);
      MHp_prob_mixing[0] = (float)(MHp_mixing_matrix[1]) / (float)(2 * MHp_mixing_matrix[0] + MHp_mixing_matrix[1]);
      MHp_prob_mixing[1] = (float)(MHp_mixing_matrix[1]) / (float)(2 * MHp_mixing_matrix[2] + MHp_mixing_matrix[1]);
    }
    
    // 4. Construct Degree Margins (Node 1) - Use arrays passed from Stage 1
    for (counter = 0; counter < length_deg_dist; counter++) {
      MHp_Deg_Distr_Edges_1[counter] = (double)MHp_Deg_Distr_1[counter] * (double)counter * MHp_prob_mixing[0];
      nwp_Deg_Distr_Edges_1[counter] = (double)nwp_Deg_Distr_1[counter] * (double)counter * nwp_prob_mixing[0];
    }
    
    // 5. Construct Degree Margins (Node 2) - Use arrays passed from Stage 1
    for (counter = 0; counter < length_deg_dist; counter++) {
      MHp_Deg_Distr_Edges_2[counter] = (double)MHp_Deg_Distr_2[counter] * (double)counter * MHp_prob_mixing[1];
      nwp_Deg_Distr_Edges_2[counter] = (double)nwp_Deg_Distr_2[counter] * (double)counter * nwp_prob_mixing[1];
    }
    
    // 6. Normalization and Expected Mixing
    double nwp_dmm_norm, MHp_dmm_norm;
    if ((Cov_types[0] == 1) && (Cov_types[1] == 1)) {
      nwp_dmm_norm = (2 * nwp_mixing_matrix[0]);
      MHp_dmm_norm = (2 * MHp_mixing_matrix[0]);
    } else if ((Cov_types[0] == 2) && (Cov_types[1] == 2)) {
      nwp_dmm_norm = (2 * nwp_mixing_matrix[2]);
      MHp_dmm_norm = (2 * MHp_mixing_matrix[2]);
    } else {
      nwp_dmm_norm = (nwp_mixing_matrix[1]);
      MHp_dmm_norm = (MHp_mixing_matrix[1]);
    }
    
    nwp_exp_dmm = (nwp_Deg_Distr_Edges_1[Deg_nwp[0]] * nwp_Deg_Distr_Edges_2[Deg_nwp[1]]) / (float)nwp_dmm_norm;
    if ((Deg_nwp[0] == Deg_nwp[1]) && (Cov_types[0] == Cov_types[1])) nwp_exp_dmm *= 0.5;
    
    MHp_exp_dmm = (MHp_Deg_Distr_Edges_1[Deg_MHp[0]] * MHp_Deg_Distr_Edges_2[Deg_MHp[1]]) / (float)MHp_dmm_norm;
    if ((Deg_MHp[0] == Deg_MHp[1]) && (Cov_types[0] == Cov_types[1])) MHp_exp_dmm *= 0.5;
    
    // 7. Final Proposal Probabilities
    if (nwp->nedges > MHp_nedges) {
      *prob_g_g2 = nwp_exp_dmm;
    } else {
      if ((Deg_nwp[0] == Deg_nwp[1]) && (Cov_types[0] == Cov_types[1])) {
        *prob_g_g2 = (nwp_Deg_Distr_1[Deg_nwp[0]] * (nwp_Deg_Distr_2[Deg_nwp[1]] - 1) * 0.5) - nwp_exp_dmm;
      } else {
        *prob_g_g2 = nwp_Deg_Distr_1[Deg_nwp[0]] * nwp_Deg_Distr_2[Deg_nwp[1]] - nwp_exp_dmm;
      }
    }
    
    if (nwp->nedges < MHp_nedges) {
      *prob_g2_g = MHp_exp_dmm;
    } else {
      if ((Deg_MHp[0] == Deg_MHp[1]) && (Cov_types[0] == Cov_types[1])) {
        *prob_g2_g = (MHp_Deg_Distr_1[Deg_MHp[0]] * (MHp_Deg_Distr_2[Deg_MHp[1]] - 1) * 0.5) - MHp_exp_dmm;
      } else {
        *prob_g2_g = MHp_Deg_Distr_1[Deg_MHp[0]] * MHp_Deg_Distr_2[Deg_MHp[1]] - MHp_exp_dmm;
      }
    }
  }
}

void calc_probs_mixing_degdist(int length_deg_dist, Model *m, Network *nwp,
                               ModelTerm *mtp2, int *prob_type,
                               double *networkstatistics, double *meanvalues,
                               double *varvalues, double *nwp_mixing_matrix,
                               double *MHp_mixing_matrix, int MHp_nedges,
                               double *pdf_gaussian_nwp, double *pdf_gaussian_MHp) {

  int counter, counter1, counter2;
  int nwp_Deg_Distr_1[length_deg_dist], MHp_Deg_Distr_1[length_deg_dist];
  int nwp_Deg_Distr_2[length_deg_dist], MHp_Deg_Distr_2[length_deg_dist];

  // Case 1 & 2: Independent Multivariate (Gaussian or Student-t)
  if ((prob_type[1] >= 1 && prob_type[1] <= 2) && prob_type[2] == 0 && prob_type[3] == 0 && prob_type[4] >= 1) {

    for (counter = 1; counter < (length_deg_dist + 1); counter++) {
      MHp_Deg_Distr_1[counter - 1] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
      nwp_Deg_Distr_1[counter - 1] = (int)networkstatistics[counter];
    }
    for (counter = (length_deg_dist + 1); counter < (2 * length_deg_dist + 1); counter++) {
      MHp_Deg_Distr_2[counter - (length_deg_dist + 1)] = (int)networkstatistics[counter] + (int)(m->workspace[counter]);
      nwp_Deg_Distr_2[counter - (length_deg_dist + 1)] = (int)networkstatistics[counter];
    }

    double nwp_mu_diff_1[length_deg_dist], MHp_mu_diff_1[length_deg_dist];
    double nwp_mu_diff_2[length_deg_dist], MHp_mu_diff_2[length_deg_dist];
    double nwp_int_1[length_deg_dist], MHp_int_1[length_deg_dist];
    double nwp_int_2[length_deg_dist], MHp_int_2[length_deg_dist];

    for (counter = 0; counter < length_deg_dist; counter++) {
      nwp_mu_diff_1[counter] = (double)nwp_Deg_Distr_1[counter]  - meanvalues[counter];
      MHp_mu_diff_1[counter] = (double)MHp_Deg_Distr_1[counter]  - meanvalues[counter];
      nwp_mu_diff_2[counter] = (double)nwp_Deg_Distr_2[counter]  - meanvalues[counter + length_deg_dist];
      MHp_mu_diff_2[counter] = (double)MHp_Deg_Distr_2[counter]  - meanvalues[counter + length_deg_dist];
    }
    
    counter2 = -1;
    int size_sq = length_deg_dist * length_deg_dist;
    for (counter = 0; counter < size_sq; counter++) {
      counter1 = counter % length_deg_dist;
      if (counter1 == 0) {
        counter2++;
        nwp_int_1[counter2] = 0; MHp_int_1[counter2] = 0;
        nwp_int_2[counter2] = 0; MHp_int_2[counter2] = 0;
      }
      nwp_int_1[counter2] += nwp_mu_diff_1[counter1] * varvalues[counter];
      MHp_int_1[counter2] += MHp_mu_diff_1[counter1] * varvalues[counter];
      nwp_int_2[counter2] += nwp_mu_diff_2[counter1] * varvalues[counter + size_sq];
      MHp_int_2[counter2] += MHp_mu_diff_2[counter1] * varvalues[counter + size_sq];
    }

    double nwp_q1 = 0, MHp_q1 = 0, nwp_q2 = 0, MHp_q2 = 0;
    for (counter = 0; counter < length_deg_dist; counter++) {
      nwp_q1 += nwp_int_1[counter] * nwp_mu_diff_1[counter];
      MHp_q1 += MHp_int_1[counter] * MHp_mu_diff_1[counter];
      nwp_q2 += nwp_int_2[counter] * nwp_mu_diff_2[counter];
      MHp_q2 += MHp_int_2[counter] * MHp_mu_diff_2[counter];
    }

    double nwp_q3 = pow(((nwp_mixing_matrix[1] ) - meanvalues[2 * length_deg_dist]), 2.0) / varvalues[2 * size_sq];
    double MHp_q3 = pow(((MHp_mixing_matrix[1] ) - meanvalues[2 * length_deg_dist]), 2.0) / varvalues[2 * size_sq];
    
    *pdf_gaussian_nwp = (-0.5 * nwp_q1) + (-0.5 * nwp_q2) + (-0.5 * nwp_q3);
    *pdf_gaussian_MHp = (-0.5 * MHp_q1) + (-0.5 * MHp_q2) + (-0.5 * MHp_q3);
  }
}
