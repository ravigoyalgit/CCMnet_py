
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import math

#################################
###Calculate initial statistic###
#################################

def calc_network_stat_degree(g):

  g_net_stat = nx.degree_histogram(g)
  g_net_stat.extend([0] * (g.number_of_nodes() - len(g_net_stat)))

  return  g_net_stat

#################################
########Update statistic#########
#################################

def calc_network_stat_degree_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g):

  deg0 = g.degree[proposal_edge[0]] 
  deg1 = g.degree[proposal_edge[1]] 

  if g_proposal_edge:
    g2_net_stat[deg0] = g2_net_stat[deg0] - 1
    g2_net_stat[deg1] = g2_net_stat[deg1] - 1
    g2_net_stat[deg0-1] = g2_net_stat[deg0-1] + 1
    g2_net_stat[deg1-1] = g2_net_stat[deg1-1] + 1
  else:
    g2_net_stat[deg0] = g2_net_stat[deg0] - 1
    g2_net_stat[deg1] = g2_net_stat[deg1] - 1
    g2_net_stat[deg0+1] = g2_net_stat[deg0+1] + 1
    g2_net_stat[deg1+1] = g2_net_stat[deg1+1] + 1

  return g2_net_stat 

##########################################
###Calculate Congruence Class Statistic###
##########################################

def calc_f_degree(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool):

  if f_g_g2_bool:
    g_degree_0 = g.degree[proposal_edge[0]] 
    g_degree_1 = g.degree[proposal_edge[1]] 
  else:
    if g_proposal_edge:
      g_degree_0 = g.degree[proposal_edge[0]] + 1 
      g_degree_1 = g.degree[proposal_edge[1]] + 1 
    else:
      g_degree_0 = g.degree[proposal_edge[0]] - 1 
      g_degree_1 = g.degree[proposal_edge[1]] - 1 


  g_Deg_Distr_Edges = np.multiply(g_net_stat,range(len(g_net_stat)))

  g_ecount = sum(g_Deg_Distr_Edges)/2

  if g_ecount != 0:
      g_exp_dmm = (g_Deg_Distr_Edges[g_degree_0] * g_Deg_Distr_Edges[g_degree_1])/ (2.0*g_ecount)
      if g_degree_0 == g_degree_1:
          g_exp_dmm = g_exp_dmm * .5
  else:
      g_exp_dmm = 0
  
  if g_proposal_edge:
    #g->g2 remove edge
    prob_g_g2 = g_exp_dmm
  else:
    #g->g2 add edge
    if g_degree_0 == g_degree_1:
      prob_g_g2 = (g_net_stat[g_degree_0]* (g_net_stat[g_degree_0]-1)*.5) - g_exp_dmm
    else:
      prob_g_g2 = g_net_stat[g_degree_0]* g_net_stat[g_degree_1] - g_exp_dmm
      
  return prob_g_g2  

#####################################
###Calculate statistic probability###
#####################################

def calc_probs_degree(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge):

  deg0 = g.degree[proposal_edge[0]] 
  deg1 = g.degree[proposal_edge[1]]

  if g_proposal_edge:
    #removed edge
    if (deg0 == deg1):
      log_prob_g2 = -math.log(g2_net_stat[deg0-1]) - math.log(g2_net_stat[deg0-1]-1) + math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg0]-1) + 2*math.log(Prob_Distr_Params[1][deg0-1]) - 2*math.log(Prob_Distr_Params[1][deg0])
    elif (deg0 == deg1 + 1):
      log_prob_g2 = -math.log(g2_net_stat[deg1-1]) + math.log(g_net_stat[deg0]) + math.log(Prob_Distr_Params[1][deg1-1]) - math.log(Prob_Distr_Params[1][deg0])
    elif (deg0 == deg1 - 1):
      log_prob_g2 = -math.log(g2_net_stat[deg0-1]) + math.log(g_net_stat[deg1]) + math.log(Prob_Distr_Params[1][deg0-1]) - math.log(Prob_Distr_Params[1][deg1])
    else:
      log_prob_g2 = -math.log(g2_net_stat[deg0-1]) - math.log(g2_net_stat[deg1-1]) + math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg1]) + math.log(Prob_Distr_Params[1][deg0-1]) + math.log(Prob_Distr_Params[1][deg1-1]) - math.log(Prob_Distr_Params[1][deg0]) - math.log(Prob_Distr_Params[1][deg1])
  else:
    #add edge
    if (deg0 == deg1):
      log_prob_g2 = math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg0]-1) - math.log(g2_net_stat[deg0+1]) - math.log(g2_net_stat[deg0+1]-1) - 2*math.log(Prob_Distr_Params[1][deg0]) + 2*math.log(Prob_Distr_Params[1][deg0+1])
    elif (deg0 == deg1 + 1):
      log_prob_g2 = math.log(g_net_stat[deg1]) - math.log(g2_net_stat[deg0+1]) - math.log(Prob_Distr_Params[1][deg1]) + math.log(Prob_Distr_Params[1][deg0+1])
    elif (deg0 == deg1 - 1):
      log_prob_g2 = math.log(g_net_stat[deg0]) - math.log(g2_net_stat[deg1+1]) - math.log(Prob_Distr_Params[1][deg0]) + math.log(Prob_Distr_Params[1][deg1+1])
    else:
      log_prob_g2 = math.log(g_net_stat[deg0]) + math.log(g_net_stat[deg1]) - math.log(g2_net_stat[deg0+1]) - math.log(g2_net_stat[deg1+1]) - math.log(Prob_Distr_Params[1][deg0]) - math.log(Prob_Distr_Params[1][deg1]) + math.log(Prob_Distr_Params[1][deg0+1]) + math.log(Prob_Distr_Params[1][deg1+1])

  prob_g = 0
  prob_g2 = log_prob_g2

  return prob_g, prob_g2
