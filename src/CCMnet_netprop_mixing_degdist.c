#include "CCMnet_netprop_mixing_degdist.h"
#include <math.h>


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

    // int nnodes_type1 = 0, nnodes_type2 = 0;
    // for (counter = 0; counter < nwp->nnodes; counter++) {
    //   if ((int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + counter]) == 1) nnodes_type1++;
    //   else nnodes_type2++;
    // }

    double nwp_mu_diff_1[length_deg_dist], MHp_mu_diff_1[length_deg_dist];
    double nwp_mu_diff_2[length_deg_dist], MHp_mu_diff_2[length_deg_dist];
    double nwp_int_1[length_deg_dist], MHp_int_1[length_deg_dist];
    double nwp_int_2[length_deg_dist], MHp_int_2[length_deg_dist];

    // for (counter = 0; counter < length_deg_dist; counter++) {
    //   nwp_mu_diff_1[counter] = (double)nwp_Deg_Distr_1[counter] / nnodes_type1 - meanvalues[counter];
    //   MHp_mu_diff_1[counter] = (double)MHp_Deg_Distr_1[counter] / nnodes_type1 - meanvalues[counter];
    //   nwp_mu_diff_2[counter] = (double)nwp_Deg_Distr_2[counter] / nnodes_type2 - meanvalues[counter + length_deg_dist];
    //   MHp_mu_diff_2[counter] = (double)MHp_Deg_Distr_2[counter] / nnodes_type2 - meanvalues[counter + length_deg_dist];
    // }

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

    // double nwp_q3 = pow(((nwp_mixing_matrix[1] / nwp->nedges) - meanvalues[2 * length_deg_dist]), 2.0) / varvalues[2 * size_sq];
    // double MHp_q3 = pow(((MHp_mixing_matrix[1] / MHp_nedges) - meanvalues[2 * length_deg_dist]), 2.0) / varvalues[2 * size_sq];

    double nwp_q3 = pow(((nwp_mixing_matrix[1] ) - meanvalues[2 * length_deg_dist]), 2.0) / varvalues[2 * size_sq];
    double MHp_q3 = pow(((MHp_mixing_matrix[1] ) - meanvalues[2 * length_deg_dist]), 2.0) / varvalues[2 * size_sq];
    
    //Rprintf("Mixing value, mean, and variance: %f %f %f\n",MHp_mixing_matrix[1], meanvalues[2 * length_deg_dist], varvalues[2 * size_sq]);
    
    if (prob_type[0] == 1) { // Gaussian
      *pdf_gaussian_nwp = (-0.5 * nwp_q1) + (-0.5 * nwp_q2) + (-0.5 * nwp_q3);
      *pdf_gaussian_MHp = (-0.5 * MHp_q1) + (-0.5 * MHp_q2) + (-0.5 * MHp_q3);
    } else { // Student-t
      double t1_n = -(meanvalues[2 * length_deg_dist + 1] + length_deg_dist) * 0.5 * log(1 + (1 / meanvalues[2 * length_deg_dist + 1]) * nwp_q1);
      double t1_m = -(meanvalues[2 * length_deg_dist + 1] + length_deg_dist) * 0.5 * log(1 + (1 / meanvalues[2 * length_deg_dist + 1]) * MHp_q1);
      double t2_n = -(meanvalues[2 * length_deg_dist + 2] + length_deg_dist) * 0.5 * log(1 + (1 / meanvalues[2 * length_deg_dist + 2]) * nwp_q2);
      double t2_m = -(meanvalues[2 * length_deg_dist + 2] + length_deg_dist) * 0.5 * log(1 + (1 / meanvalues[2 * length_deg_dist + 2]) * MHp_q2);
      double t3_n = -(meanvalues[2 * length_deg_dist + 3] + 1) * 0.5 * log(1 + (1 / meanvalues[2 * length_deg_dist + 3]) * nwp_q3);
      double t3_m = -(meanvalues[2 * length_deg_dist + 3] + 1) * 0.5 * log(1 + (1 / meanvalues[2 * length_deg_dist + 3]) * MHp_q3);
      *pdf_gaussian_nwp = t1_n + t2_n + t3_n;
      *pdf_gaussian_MHp = t1_m + t2_m + t3_m;
    }
  }

  // Case 3: Combined Student-t
  if (prob_type[0] == 3 && prob_type[1] == 3 && prob_type[2] == 0 && prob_type[3] == 0 && prob_type[4] >= 1) {
    int combo_len = 2 * length_deg_dist + 1;
    int nwp_comb[combo_len], MHp_comb[combo_len];
    int nnodes_type1 = 0, nnodes_type2 = 0;

    for (counter = 0; counter < nwp->nnodes; counter++) {
      if ((int)round(mtp2->inputparams[mtp2->ninputparams - nwp->nnodes + counter]) == 1) nnodes_type1++;
      else nnodes_type2++;
    }

    for (counter = 1; counter < (length_deg_dist + 1); counter++) {
      nwp_comb[counter - 1] = (int)networkstatistics[counter];
      MHp_comb[counter - 1] = nwp_comb[counter - 1] + (int)(m->workspace[counter]);
    }
    for (counter = (length_deg_dist + 1); counter < (2 * length_deg_dist + 1); counter++) {
      nwp_comb[counter - 1] = (int)networkstatistics[counter];
      MHp_comb[counter - 1] = nwp_comb[counter - 1] + (int)(m->workspace[counter]);
    }
    nwp_comb[2 * length_deg_dist] = nwp_mixing_matrix[1];
    MHp_comb[2 * length_deg_dist] = MHp_mixing_matrix[1];

    double nwp_mu[combo_len], MHp_mu[combo_len], nwp_int[combo_len], MHp_int[combo_len];
    for (counter = 0; counter < length_deg_dist; counter++) {
      nwp_mu[counter] = (double)nwp_comb[counter] / nnodes_type1 - meanvalues[counter];
      MHp_mu[counter] = (double)MHp_comb[counter] / nnodes_type1 - meanvalues[counter];
      nwp_mu[counter + length_deg_dist] = (double)nwp_comb[counter + length_deg_dist] / nnodes_type2 - meanvalues[counter + length_deg_dist];
      MHp_mu[counter + length_deg_dist] = (double)MHp_comb[counter + length_deg_dist] / nnodes_type2 - meanvalues[counter + length_deg_dist];
    }
    nwp_mu[2 * length_deg_dist] = (double)nwp_comb[2 * length_deg_dist] / nwp->nedges - meanvalues[2 * length_deg_dist];
    MHp_mu[2 * length_deg_dist] = (double)MHp_comb[2 * length_deg_dist] / MHp_nedges - meanvalues[2 * length_deg_dist];

    counter2 = -1;
    for (counter = 0; counter < (combo_len * combo_len); counter++) {
      counter1 = counter % combo_len;
      if (counter1 == 0) {
        counter2++; nwp_int[counter2] = 0; MHp_int[counter2] = 0;
      }
      nwp_int[counter2] += nwp_mu[counter1] * varvalues[counter];
      MHp_int[counter2] += MHp_mu[counter1] * varvalues[counter];
    }

    double q_n = 0, q_m = 0;
    for (counter = 0; counter < combo_len; counter++) {
      q_n += nwp_int[counter] * nwp_mu[counter];
      q_m += MHp_int[counter] * MHp_mu[counter];
    }

    double df = meanvalues[2 * length_deg_dist + 1];
    *pdf_gaussian_nwp = -(df + combo_len) * 0.5 * log(1 + (1 / df) * q_n);
    *pdf_gaussian_MHp = -(df + combo_len) * 0.5 * log(1 + (1 / df) * q_m);
  }
}
