
from CCMnet_utils import *

#################################
###Calculate initial statistic###
#################################

def calc_network_stat_edge(g):
  
  g_net_stat = g.number_of_edges()
  return g_net_stat

#################################
########Update statistic#########
#################################

def calc_network_stat_edge_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern):

  if g_proposal_edge:
    g2_net_stat = g2_net_stat - 1
  else:
    g2_net_stat = g2_net_stat + 1
  
  return g2_net_stat

##########################################
###Calculate Congruence Class Statistic###
##########################################

def calc_f_edge(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g):
  
  population = g.number_of_nodes()
  
  if g_proposal_edge:
    #g->g2 remove edge
    prob_g_g2 = g_net_stat
  else:
    #g->g2 add edge
    prob_g_g2 = ncr(population,2) - g_net_stat

  return prob_g_g2

#####################################
###Calculate statistic probability###
#####################################

def calc_probs_edge(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g_proposal_edge):

  if (Prob_Distr[0] == "NP"):
    prob_g = math.log(Prob_Distr_Params[0][g_net_stat])
    prob_g2 = math.log(Prob_Distr_Params[0][g2_net_stat])
  elif (Prob_Distr[0] == "Poisson"):
    if g_proposal_edge:
      #removed edge
      prob_g2 = 0
      prob_g = math.log(Prob_Distr_Params[0]/g_net_stat)
    else:
      #add edge
      prob_g2 = math.log(Prob_Distr_Params[0]/g2_net_stat)
      prob_g = 0
  elif (Prob_Distr[0] == "uniform"):
    prob_g = 0
    prob_g2 = 0
  else:
    print("ERROR")

  return prob_g, prob_g2
