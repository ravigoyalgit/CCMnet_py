import networkx as nx
import pandas as pd
import numpy as np
import random, json
import math
from scipy.special import gammaln

#################################
### Supporting Functions      ###
#################################

import networkx as nx
import numpy as np
import math
from scipy.special import gammaln

def log_comb(n, k):
	if k < 0 or k > n:
		return -np.inf
	return gammaln(n + 1) - (gammaln(k + 1) + gammaln(n - k + 1))

def safe_diff(a, b):
	if np.isneginf(a) or np.isneginf(b):
		return -np.inf
	return a - b

#################################
###Calculate initial statistic###
#################################

def calc_network_stat_degmix(g):
	# n is Population - 1
	n = g.number_of_nodes() - 1
	deg_dict = dict(g.degree())
	# Matrix size: (N-1)x(N-1). Row 0 = Degree 1, Row N-2 = Degree N-1.
	g_net_stat = np.zeros((n, n), dtype=int) 
	for u, v in g.edges():
		du = deg_dict[u] - 1
		dv = deg_dict[v] - 1
		if du >= 0 and dv >= 0:
			g_net_stat[du, dv] += 1
			if du != dv:
				g_net_stat[dv, du] += 1
	return g_net_stat

#################################
########Update statistic#########
#################################

def calc_network_stat_degmix_2(proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g, print_calculations):
	u, v = proposal_edge
	num_stats = g2_net_stat.shape[0]
	
	# Current state in g
	d_u_old = g.degree[u]
	d_v_old = g.degree[v]
	idx_u_old, idx_v_old = d_u_old - 1, d_v_old - 1
	
	# Target state in g2
	delta = -1 if g_proposal_edge else 1
	d_u_new = d_u_old + delta
	d_v_new = d_v_old + delta
	idx_u_new, idx_v_new = d_u_new - 1, d_v_new - 1

	# 1. Update neighbors of u (excluding v)
	for nb in g.neighbors(u):
		if nb == v: continue
		idx_nb = g.degree[nb] - 1
		# Remove old edge position
		if 0 <= idx_u_old < num_stats and 0 <= idx_nb < num_stats:
			g2_net_stat[idx_u_old, idx_nb] -= 1
			if idx_u_old != idx_nb:
				g2_net_stat[idx_nb, idx_u_old] -= 1
		# Add new edge position
		if 0 <= idx_u_new < num_stats and 0 <= idx_nb < num_stats:
			g2_net_stat[idx_u_new, idx_nb] += 1
			if idx_u_new != idx_nb:
				g2_net_stat[idx_nb, idx_u_new] += 1

	# 2. Update neighbors of v (excluding u)
	for nb in g.neighbors(v):
		if nb == u: continue
		idx_nb = g.degree[nb] - 1
		if 0 <= idx_v_old < num_stats and 0 <= idx_nb < num_stats:
			g2_net_stat[idx_v_old, idx_nb] -= 1
			if idx_v_old != idx_nb:
				g2_net_stat[idx_nb, idx_v_old] -= 1
		if 0 <= idx_v_new < num_stats and 0 <= idx_nb < num_stats:
			g2_net_stat[idx_v_new, idx_nb] += 1
			if idx_v_new != idx_nb:
				g2_net_stat[idx_nb, idx_v_new] += 1

	# 3. Handle the proposal edge (u, v)
	if g_proposal_edge:
		# Removal: existed at old degrees
		if 0 <= idx_u_old < num_stats and 0 <= idx_v_old < num_stats:
			g2_net_stat[idx_u_old, idx_v_old] -= 1
			if idx_u_old != idx_v_old:
				g2_net_stat[idx_v_old, idx_u_old] -= 1
	else:
		# Addition: now exists at new degrees
		if 0 <= idx_u_new < num_stats and 0 <= idx_v_new < num_stats:
			g2_net_stat[idx_u_new, idx_v_new] += 1
			if idx_u_new != idx_v_new:
				g2_net_stat[idx_v_new, idx_u_new] += 1
				
	return g2_net_stat

##########################################
###Calculate Congruence Class Statistic###
##########################################

def calc_f_degmix(g_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations):
  u, v = proposal_edge
  num_deg_stats = g_net_stat.shape[0]
  is_removing = g_proposal_edge
  population = g.number_of_nodes()
	
  # Source Degrees
  if f_g_g2_bool:
    d_u, d_v = g.degree[u], g.degree[v]
  else:
    delta = 1 if g_proposal_edge else -1
    d_u, d_v = g.degree[u] + delta, g.degree[v] + delta

  idx_u, idx_v = d_u - 1, d_v - 1

  # Source Neighbors
  n_i = np.zeros(num_deg_stats, dtype=int)
  n_j = np.zeros(num_deg_stats, dtype=int)
  for neighbor in g.neighbors(u):
    n_i[g.degree[neighbor] - 1] += 1
  for neighbor in g.neighbors(v):
    n_j[g.degree[neighbor] - 1] += 1

  if not f_g_g2_bool:
    if g_proposal_edge: 
      if d_v > 0: n_i[d_v - 1] += 1
      if d_u > 0: n_j[d_u - 1] += 1
    else: 
      if g.degree[v] > 0: n_i[g.degree[v] - 1] -= 1
      if g.degree[u] > 0: n_j[g.degree[u] - 1] -= 1

	# Reconstruction
  deg_dist = np.zeros(num_deg_stats + 1)
  total_nodes_found = 0
	
  for i in range(num_deg_stats):
    # row_sum + diag element (to compensate for symmetry subtraction in calc_network_stat_degmix)
    row_sum = np.sum(g_net_stat[i, :]) + g_net_stat[i, i]
    count = int(round(row_sum / (i + 1)))
    deg_dist[i+1] = count
    total_nodes_found += count
		
  deg_dist[0] = max(0, population - total_nodes_found)

  working_dmm = g_net_stat.copy().astype(float)
  for i in range(num_deg_stats):
    working_dmm[i, i] *= 2

  log_prob = 0.0
  if not is_removing:
    # ADDITION
    if d_u == d_v:
      term = deg_dist[d_u] * (deg_dist[d_u] - 1) * 0.5
    else:
      term = deg_dist[d_u] * deg_dist[d_v]
		
    existing = g_net_stat[idx_u, idx_v] if (d_u > 0 and d_v > 0) else 0
    log_prob = np.log(max(1e-15, term - existing))

    if d_u == d_v:
      log_denom = log_comb(deg_dist[d_u] * d_u, d_u + d_v)
      log_num = sum(log_comb(working_dmm[idx_u, k], n_i[k] + n_j[k]) for k in range(num_deg_stats)) if idx_u >= 0 else 0
      log_prob += safe_diff(log_num, log_denom)
    else:
      log_num_u = sum(log_comb(working_dmm[idx_u, k], n_i[k]) for k in range(num_deg_stats)) if idx_u >= 0 else 0
      log_num_v = sum(log_comb(working_dmm[idx_v, k], n_j[k]) for k in range(num_deg_stats)) if idx_v >= 0 else 0
      log_prob += safe_diff(log_num_u, log_comb(deg_dist[d_u] * d_u, d_u))
      log_prob += safe_diff(log_num_v, log_comb(deg_dist[d_v] * d_v, d_v))
  else:
    # REMOVAL
    if d_u <= 0 or d_v <= 0: return 0.0
    log_prob = np.log(max(1e-15, g_net_stat[idx_u, idx_v]))
    if d_u == d_v:
      log_denom = log_comb(deg_dist[d_u] * d_u - 1, d_u + d_v - 2)
      log_num = 0.0
      for k in range(num_deg_stats):
        indicator = 1 if k == idx_u else 0
        log_num += log_comb(working_dmm[idx_u, k] - indicator, n_i[k] + n_j[k] - (2 * indicator))
      log_prob += safe_diff(log_num, log_denom)
    else:
      log_num_u = sum(log_comb(working_dmm[idx_u, k] - (1 if k == idx_v else 0), n_i[k] - (1 if k == idx_v else 0)) for k in range(num_deg_stats))
      log_num_v = sum(log_comb(working_dmm[idx_v, k] - (1 if k == idx_u else 0), n_j[k] - (1 if k == idx_u else 0)) for k in range(num_deg_stats))
      log_prob += safe_diff(log_num_u, log_comb(deg_dist[d_u] * d_u - 1, d_u - 1))
      log_prob += safe_diff(log_num_v, log_comb(deg_dist[d_v] * d_v - 1, d_v - 1))

  return np.exp(log_prob)

#####################################
### Calculate statistic probability###
#####################################

def calc_probs_degmix(g_net_stat, g2_net_stat, proposal_edge, covPattern, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge):
	
  #print("DEBUG - Degmix: Degmix Prob Dist: ", Prob_Distr[0])
  
  prob_g = 0
  prob_g2 = 0
  if (Prob_Distr[0] == "Multinomial_Poisson"):
    x_g = sum(g_net_stat[np.triu_indices(g_net_stat.shape[0])])
    x_g2 = sum(g2_net_stat[np.triu_indices(g2_net_stat.shape[0])])
    deg0 = g.degree[proposal_edge[0]] 
    deg1 = g.degree[proposal_edge[1]]
    entry_id = sum(range(deg1+1)) + deg0
    if x_g > x_g2:
      log_prob_x_g2 = -math.log(Prob_Distr_Params[0]) + math.log(x_g)
      log_prob_x2_g2 = -math.log(x_g) + math.log(g_net_stat[deg0-1][deg1-1]) - math.log(Prob_Distr_Params[1][entry_id])
    else:
      log_prob_x_g2 = math.log(Prob_Distr_Params[0]) - math.log(x_g2)
      log_prob_x2_g2 = math.log(x_g2) - math.log(g2_net_stat[deg0-1][deg1-1]) + math.log(Prob_Distr_Params[1][entry_id])
    prob_g2 = log_prob_x_g2 + log_prob_x2_g2

  if (Prob_Distr[0] == "Multivariate_normal"):
    mu = Prob_Distr_Params[0]
    inv_sigma = Prob_Distr_Params[1]
    diff1 = g2_net_stat[np.triu_indices(g2_net_stat.shape[0])] - mu
    diff2 = g_net_stat[np.triu_indices(g_net_stat.shape[0])] - mu
    quad1 = diff1.T @ inv_sigma @ diff1
    quad2 = diff2.T @ inv_sigma @ diff2
    prob_g2 = -0.5 * (quad1 - quad2)
    #print(f"DEBUG DMM - Degmix 1: g={prob_g}, g2={prob_g2}")

  #print(f"DEBUG DMM - Degmix 2: g={prob_g}, g2={prob_g2}")
  return prob_g, prob_g2

