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

# def calc_f_degmix_clustering(g_net_stat, proposal_edge, g_proposal_edge, covPattern,bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations):
#   u, v = proposal_edge
#   current_dmm = g_net_stat['dmm']
#   num_deg_stats = current_dmm.shape[0]
#   is_adding = not g_proposal_edge 
#     
#   # 1. Get Base Degree Mixing f (Handles degree 0 correctly)
#   prob_f = calc_f_degmix(current_dmm, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)
# 
#   # 2. Reconstruct Degree Distribution (Include Degree 0)
#   deg_dist = np.zeros(num_deg_stats + 1)
#   for i in range(num_deg_stats):
#     row_sum = np.sum(current_dmm[i, :]) + current_dmm[i, i]
#     deg_dist[i+1] = int(round(row_sum / (i + 1)))
#   deg_dist[0] = max(0, g.number_of_nodes() - sum(deg_dist))
# 
#   # 3. Determine Source Degrees and Shared Neighbors
#   if f_g_g2_bool:
#     d_u, d_v = g.degree[u], g.degree[v]
#     k_val = len(list(nx.common_neighbors(g, u, v)))
#   else:
#     delta = 1 if not g_proposal_edge else -1
#     d_u, d_v = g.degree[u] + delta, g.degree[v] + delta
#     # Shared neighbors in g2 (source)
#     k_val = len(list(nx.common_neighbors(g, u, v)))
# 
#     # 4. Calculate p_a and p_b (The "Goyal" probabilities)
#     # p_u[d] is the probability that a node of degree d is connected to u
#   p_u = np.zeros(num_deg_stats + 1)
#   p_v = np.zeros(num_deg_stats + 1)
#     
#   for d in range(1, num_deg_stats + 1):
#     if deg_dist[d] > 0:
#       # Prob = (Edges between class d_u and d) / (Total stubs in class d)
#       # This is the local probability a node of degree d connects to u's class
#       if d_u > 0: p_u[d] = current_dmm[d_u-1, d-1] / (deg_dist[d] * d)
#       if d_v > 0: p_v[d] = current_dmm[d_v-1, d-1] / (deg_dist[d] * d)
# 
#   # 5. Transition Measure Correction
#   # Instead of a global pa, we iterate over the nodes that COULD be shared neighbors
#   log_q = 0.0
#   for node_w in g.nodes():
#     if node_w == u or node_w == v: continue
#         
#     dw = g.degree[node_w]
#     if dw == 0: continue
#         
#     # Prob(w is connected to both u and v)
#     p_shared = p_u[dw] * p_v[dw]
#     p_shared = max(1e-15, min(1 - 1e-15, p_shared))
# 
#     if g.has_edge(u, node_w) and g.has_edge(v, node_w):
#       log_q += np.log(p_shared)
#     else:
#       log_q += np.log(1 - p_shared)
# 
#   # In Goyal CCM, the clustering factor is 1/exp(log_q) for addition
#   if is_adding:
#     prob_f *= np.exp(-log_q)
#   else:
#     prob_f *= np.exp(log_q)
# 
#   return prob_f

def calc_f_degmix_clustering(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations):
  u, v = proposal_edge
  current_dmm = g_net_stat['dmm']
  current_triangles = g_net_stat['triangles']
  
  population = g.number_of_nodes()

  prob_f = calc_f_degmix(current_dmm, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

  if f_g_g2_bool:
    d_u, d_v = g.degree[u], g.degree[v]
  else:
    d_u = g.degree[u] + 1 if g_proposal_edge else g.degree[u] - 1
    d_v = g.degree[v] + 1 if g_proposal_edge else g.degree[v] - 1

  is_adding = not g_proposal_edge

  deg_dist = np.zeros(population)
  total_nodes_found = 0
  for i in range(population-1):
    row_sum = np.sum(current_dmm[i, :]) + current_dmm[i, i]
    count = int(round(row_sum / (i + 1)))
    deg_dist[i+1] = count
    total_nodes_found += count
  deg_dist[0] = max(0, population - total_nodes_found)

  k = len(list(nx.common_neighbors(g, u, v)))

  if is_adding:
    pa_num = -3.0 * current_triangles
    for deg in range(2, population):
      pa_num += (deg * (deg - 1) / 2.0) * deg_dist[deg]
      
    pa_dem = 0.0
    # i and j represent the degrees (1 to population-1)
    for i in range(1, population):
      for j in range(i, population):
        # 1. Total possible pairs of nodes between degree classes i and j
        if i == j:
          # n_i * (n_i - 1) / 2
          slots = (deg_dist[i] * (deg_dist[i] - 1)) / 2.0
        else:
          # n_i * n_j
          slots = deg_dist[i] * deg_dist[j]
            
        # 2. Number of non-edges (available spots for a new edge)
        # Recall current_dmm index is degree - 1
        non_edges = max(0, slots - current_dmm[i-1, j-1])
            
        # 3. Weight by (i * j)
        # This accounts for how many potential paths of length 2 
        # would be closed by an edge connecting a degree i and degree j node.
        pa_dem += (i * j) * non_edges

    pa = max(1e-15, min(1 - 1e-15, pa_num / pa_dem)) if pa_dem != 0 else 1e-15

    # LOG-STABLE MATH TO PREVENT OVERFLOW

    # 2. Log of Term 1: ln[ (du choose k) * (dv choose k) * k! ]
    # Formula: ln(du!) - ln(k!) - ln(du-k)!) + ln(dv!) - ln(k!) - ln(dv-k)!) + ln(k!)
    # Simplified: ln(du!) + ln(dv!) - ln(k!) - ln(du-k)!) - ln(dv-k)!)
    log_term1 = (gammaln(d_u + 1) + gammaln(d_v + 1) - gammaln(k + 1) - gammaln(d_u - k + 1) - gammaln(d_v - k + 1))

    # 3. Log of Term 2: k * ln(pa)
    log_term2 = k * np.log(pa)

    # 4. Log of Term 3: (du - k) * (dv - k) * ln(1 - pa)
    log_term3 = (d_u - k) * (d_v - k) * np.log(1.0 - pa)

    # Summing for the clustering component of the log probability
    log_p_add = log_term1 + log_term2 + log_term3
    
    # Update total prob_f
    prob_f += log_p_add

  else:
    pb_dem = 0.0
    # s and r represent degrees (1 to population-1)
    # current_dmm index is degree - 1
    for r in range(1, population):
      for s in range(1, r+1): # s goes from 1 to r
        # Number of edges between degree r and degree s
        m_rs = current_dmm[r-1, s-1]
            
        # Weighted contribution to the denominator
        pb_dem += (r - 1) * (s - 1) * m_rs

    # Equation 21: pb = (3 * T) / pb_dem
    pb = (3.0 * current_triangles) / pb_dem if pb_dem != 0 else 1e-15
    pb = max(1e-15, min(1 - 1e-15, pb))

    # 2. Log of Term 1: ln[ (du-1 choose k) * (dv-1 choose k) * k! ]
    # Formula: ln((du-1)!) + ln((dv-1)!) - ln(k!) - ln((du-1-k)!) - ln((dv-1-k)!)
    log_term1 = (gammaln(d_u) + gammaln(d_v) - gammaln(k + 1) - gammaln(d_u - k) - gammaln(d_v - k))

    # 3. Log of Term 2: k * ln(pb)
    log_term2 = k * np.log(pb)

    # 4. Log of Term 3: (du-1-k) * (dv-1-k) * ln(1 - pb)
    log_term3 = (d_u - 1 - k) * (d_v - 1 - k) * np.log(1.0 - pb)

    # Summing for the clustering component
    log_p_sub = log_term1 + log_term2 + log_term3
    
    # Update total prob_f (which already contains the deg-mix component)
    prob_f += log_p_sub
  return prob_f

#####################################
###Calculate statistic probability###
#####################################

def calc_probs_degmix_clustering(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge, print_calculations):
  """
  Returns (log_p_g, log_p_g2).
  """
  if print_calculations:
    print("DEBUG Clustering: Degmix Params Type: ", type(Prob_Distr_Params[0]))
    print("DEBUG Clustering: Degmix Params Length: ", len(Prob_Distr_Params[0]))
    print("DEBUG Clustering: Degmix Prob Dist: ", Prob_Distr[0])
  
  # 1. Degree Mixing log-probabilities
  log_p_g_dmm, log_p_g2_dmm = calc_probs_degmix(
    g_net_stat['dmm'], 
    g2_net_stat['dmm'], 
    proposal_edge, 
    covPattern, 
    [Prob_Distr[0]], 
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
  
  if print_calculations:
    print(f"DEBUG Clustering: DMM probs: g={log_p_g_dmm}, g2={log_p_g2_dmm}")
    print(f"DEBUG Clustering: Clustering probs: g={log_p_g_tri}, g2={log_p_g2_tri}")
    
  return 0, log_ratio
  #return log_p_g_dmm, log_p_g2_dmm
