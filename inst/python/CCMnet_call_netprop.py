
from CCMnet_netprop_edge import *
from CCMnet_netprop_degree import *
from CCMnet_netprop_mixing import *
from CCMnet_netprop_degmix import *
from CCMnet_netprop_degmixcluster import *

#################################
###Calculate initial statistic###
#################################

def calc_network_stat(g, Network_stats):

  if Network_stats[0].strip().lower() == "edge" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_edge(g)
    
  if Network_stats[0].strip().lower() == "mixing" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_mixing(g)

  if Network_stats[0].strip().lower() == "degree" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_degree(g)

  if Network_stats[0].strip().lower() == "degmix" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_degmix(g)

  if Network_stats[0].strip().lower() == "degmix_clustering" and len(Network_stats) == 1:
    g_net_stat = calc_network_stat_degmix_clustering(g)
    
  return g_net_stat

#################################
########Update statistic#########
#################################

def calc_network_stat_2(Network_stats, proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g, print_calculations):

  if Network_stats[0].strip().lower() == "edge" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_edge_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern)

  if Network_stats[0].strip().lower() == "mixing" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_mixing_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, print_calculations)

  if Network_stats[0].strip().lower() == "degree" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_degree_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g)

  if Network_stats[0].strip().lower() == "degmix" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_degmix_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g, print_calculations)

  if Network_stats[0].strip().lower() == "degmix_clustering" and len(Network_stats) == 1:
    g2_net_stat = calc_network_stat_degmix_clustering_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g, print_calculations)

  return g2_net_stat

##########################################
###Calculate Congruence Class Statistic###
##########################################

def calc_f(Network_stats,g_net_stat, g2_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations):
  
  if Network_stats[0].strip().lower() == "edge" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_edge(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g)

  if Network_stats[0].strip().lower() == "mixing" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_mixing(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat)

  if Network_stats[0].strip().lower() == "degree" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_degree(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool)

  if Network_stats[0].strip().lower() == "degmix" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_degmix(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

  if Network_stats[0].strip().lower() == "degmix_clustering" and len(Network_stats) == 1:
    prob_g_g2 = calc_f_degmix_clustering(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

  return prob_g_g2

#####################################
###Calculate statistic probability###
#####################################

def calc_probs(g_net_stat, g2_net_stat, proposal_edge, covPattern, Network_stats, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge):

  if Network_stats[0].strip().lower() == "edge" and len(Network_stats) == 1:
    prob_g, prob_g2 = calc_probs_edge(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g_proposal_edge)

  if Network_stats[0].strip().lower() == "mixing" and len(Network_stats) == 1:
    prob_g, prob_g2 = calc_probs_mixing(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params)

  if Network_stats[0].strip().lower() == "degree" and len(Network_stats) == 1:
    prob_g, prob_g2 = calc_probs_degree(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge)

  if Network_stats[0].strip().lower() == "degmix" and len(Network_stats) == 1:
    prob_g, prob_g2 = calc_probs_degmix(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge)

  if Network_stats[0].strip().lower() == "degmix_clustering" and len(Network_stats) == 1:
    prob_g, prob_g2 = calc_probs_degmix_clustering(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge)

  return prob_g, prob_g2
