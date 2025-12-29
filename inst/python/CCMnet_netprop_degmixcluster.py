import networkx as nx
import pandas as pd
import numpy as np
import random, json
import math

from CCMnet_netprop_degmix import *

#################################
###Calculate initial statistic###
#################################

def calc_network_stat_degmix_clustering(g):
  """
  Initializes the statistics bundle for the starting graph.
  Returns a dictionary containing the DMM and the total triangle count.
  """
  # Calculate DMM using existing degree mixing library
  dmm = calc_network_stat_degmix(g)
    
  # Calculate total triangles (NetworkX returns 3 * triangles)
  triangles = sum(nx.triangles(g).values()) // 3
  
  return {'dmm': dmm, 'triangles': triangles}

#################################
########Update statistic#########
#################################

def calc_network_stat_degmix_clustering_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g, print_calculations):
  """
  Calculates the statistics for the proposed state g2.
  Uses incremental updates for efficiency.
  """
  u, v = proposal_edge
    
  # 1. Update DMM using existing logic
  g2_dmm = calc_network_stat_degmix_2(
    proposal_edge=proposal_edge, 
    g_net_stat=g_net_stat['dmm'], 
    g2_net_stat=g2_net_stat['dmm'], 
    g_proposal_edge=g_proposal_edge, 
    covPattern=covPattern, 
    g=g, 
    print_calculations=print_calculations
  )
    
  # 2. Update Triangles incrementally
  num_tri_change = len(list(nx.common_neighbors(g, u, v)))
    
  if g_proposal_edge: # Edge exists in g, so we are removing it for g2
    g2_triangles = g_net_stat['triangles'] - num_tri_change
  else:               # Edge does not exist in g, so we are adding it for g2
    g2_triangles = g_net_stat['triangles'] + num_tri_change
        
  return {'dmm': g2_dmm, 'triangles': g2_triangles}

##########################################
###Calculate Congruence Class Statistic###
##########################################

def calc_f_degmix_clustering(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations):
  u, v = proposal_edge
  current_dmm = g_net_stat['dmm']
  current_triangles = g_net_stat['triangles']
    
  prob_f = calc_f_degmix(current_dmm, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

  if f_g_g2_bool:
    d_u, d_v = g.degree[u], g.degree[v]
  else:
    edge_exists_in_g = g.has_edge(u, v)
    if g_proposal_edge != edge_exists_in_g:
      d_u = g.degree[u] + 1 if g_proposal_edge else g.degree[u] - 1
      d_v = g.degree[v] + 1 if g_proposal_edge else g.degree[v] - 1
    else:
      d_u, d_v = g.degree[u], g.degree[v]

  k_val = len(list(nx.common_neighbors(g, u, v)))
  is_adding = not g_proposal_edge 
  num_deg_stats = current_dmm.shape[0]

  deg_dist = np.zeros(num_deg_stats + 1)
  for i in range(num_deg_stats):
    row_sum = np.sum(current_dmm[i, :]) + current_dmm[i, i]
    deg_dist[i+1] = int(round(row_sum / (i + 1)))

  if is_adding:
    pa_num = -3.0 * current_triangles
    for k in range(2, num_deg_stats + 1):
      pa_num += (k * (k - 1) / 2.0) * deg_dist[k]
    pa_dem = 0.0
    for i in range(1, num_deg_stats + 1):
      for j in range(i, num_deg_stats + 1):
        slots = (deg_dist[i] * (deg_dist[j] - 1) * 0.5) if i == j else (deg_dist[i] * deg_dist[j])
        pa_dem += i * j * (slots - current_dmm[i-1, j-1])
        
    pa = max(1e-15, min(1 - 1e-15, pa_num / pa_dem)) if pa_dem != 0 else 1e-15
        
    # LOG-STABLE MATH TO PREVENT OVERFLOW
    log_perms = sum(math.log(max(1, d_u - c)) + math.log(max(1, d_v - c)) for c in range(k_val))
    log_p_cluster = (-math.lgamma(k_val + 1) + log_perms + (k_val * math.log(pa)) + ((d_u - k_val) * (d_v - k_val) * math.log(1 - pa)))
    prob_f *= math.exp(log_p_cluster)

  else:
    pb_num = 3.0 * current_triangles
    pb_dem = 0.0
    for i in range(1, num_deg_stats + 1):
      for j in range(i, num_deg_stats + 1):
        pb_dem += (i - 1) * (j - 1) * current_dmm[i-1, j-1]
        
    pb = max(1e-15, min(1 - 1e-15, pb_num / pb_dem)) if pb_dem != 0 else 1e-15
        
    # LOG-STABLE MATH TO PREVENT OVERFLOW
    log_perms = sum(math.log(max(1, d_u - 1 - c)) + math.log(max(1, d_v - 1 - c)) for c in range(k_val))
    log_p_cluster = (math.lgamma(k_val + 1) + log_perms + (k_val * math.log(pb)) + ((d_u - 1 - k_val) * (d_v - 1 - k_val) * math.log(1 - pb)))
    prob_f *= math.exp(log_p_cluster)

  return prob_f

#####################################
###Calculate statistic probability###
#####################################

def calc_probs_degmix_clustering(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge):
  """
  Returns (log_p_g, log_p_g2).
  """
  # 1. Degree Mixing log-probabilities
  log_p_g_dmm, log_p_g2_dmm = calc_probs_degmix(
    g_net_stat['dmm'], 
    g2_net_stat['dmm'], 
    proposal_edge, 
    covPattern, 
    Prob_Distr[0], 
    Prob_Distr_Params[0], 
    g, 
    g_proposal_edge
  )
    
  # 2. Clustering log-probabilities
  t1 = g_net_stat['triangles']
  t2 = g2_net_stat['triangles']
  dist_type_tri = Prob_Distr[1]
  params_tri = Prob_Distr_Params[1]

  if dist_type_tri == "Normal":
    mu, sigma = params_tri[0], params_tri[1]
    log_p_g_tri = -0.5 * ((t1 - mu) / sigma) ** 2
    log_p_g2_tri = -0.5 * ((t2 - mu) / sigma) ** 2
  else:
    log_p_g_tri = 0.0
    log_p_g2_tri = 0.0

  # 3. Combine in log-space and set baseline to 0
  log_ratio = (log_p_g2_dmm - log_p_g_dmm) + (log_p_g2_tri - log_p_g_tri)
    
  return 0, log_ratio
