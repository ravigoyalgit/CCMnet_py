#include "CCMnet_netprop_edges.h"

void calc_f_edges(int nwp_nedges, int MHp_nedges, int total_max_edges,
                  double *networkstatistics, double *prob_g_g2, double *prob_g2_g) {
  if (nwp_nedges > MHp_nedges) {
    *prob_g_g2 = networkstatistics[0];
  } else {
    *prob_g_g2 = total_max_edges - networkstatistics[0];
  }

  if (nwp_nedges < MHp_nedges) {
    *prob_g2_g = (double)MHp_nedges;
  } else {
    *prob_g2_g = (double)(total_max_edges - MHp_nedges);
  }
}
