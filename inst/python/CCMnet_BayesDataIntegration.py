
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import operator as op
from functools import reduce
from itertools import cycle, islice
import math
import cProfile

def bayes_inf_MH_prob_calc(MH_prob, g, proposal_edge, Pnet, Ia, Il, R, epi_params):

  if Pnet.has_edge(proposal_edge[0], proposal_edge[1]): 
    #Reject the toggle
    MH_prob = -math.inf
  else:
    if Ia[proposal_edge[0]] < 999999 or Ia[proposal_edge[1]] < 999999:  

      beta_a_val = epi_params[0]
      beta_l_val = epi_params[1]

      Il_i = Il[proposal_edge[0]]
      Ia_i = Ia[proposal_edge[0]]
      R_i = R[proposal_edge[0]]
      Il_j = Il[proposal_edge[1]]
      Ia_j = Ia[proposal_edge[1]]
      R_j = R[proposal_edge[1]]
      
      if (Ia_j < Ia_i):
        time_a = min(Il_j,Ia_i)-Ia_j
        time_l = max(min(R_j,Ia_i),Il_j) - Il_j
        log_muij = (-beta_a_val*time_a) + (-beta_l_val*time_l)
      else:
        time_a = min(Il_i,Ia_j)-Ia_i
        time_l = max(min(R_i,Ia_j),Il_i) - Il_i
        log_muij = (-beta_a_val*time_a) + (-beta_l_val*time_l)

      if g.has_edge(proposal_edge[0], proposal_edge[1]):
        MH_prob = MH_prob - log_muij
      else:
        MH_prob = log_muij + MH_prob   

  return MH_prob

def bayes_inf_MH_prob_calc_2(MH_prob, g, proposal_edge, Pnet, Ia, Il, R, epi_params):

#  print("Bayes MH_prob: ", MH_prob)
  beta_a_val = epi_params[0]
  beta_l_val = epi_params[1]

  if g.has_edge(proposal_edge[0], proposal_edge[1]):
    if MH_prob > 500:
      p_edge1 = .000001
    else:  
      p_edge1 = 1 / (1 + math.exp(MH_prob))
  else:
    if MH_prob > 500:
      p_edge1 = .999999
    else:
      p_edge1 = math.exp(MH_prob) / (1 + math.exp(MH_prob));

#  print("Bayes p_edge1: ", p_edge1)

  if Pnet.has_edge(proposal_edge[0], proposal_edge[1]): 
    #Reject the toggle
    MH_prob = -math.inf
  else:
    if Ia[proposal_edge[0]] < 999999 or Ia[proposal_edge[1]] < 999999:  

      Il_i = Il[proposal_edge[0]];
      Ia_i = Ia[proposal_edge[0]];
      R_i = R[proposal_edge[0]];
      Il_j = Il[proposal_edge[1]];
      Ia_j = Ia[proposal_edge[1]];
      R_j = R[proposal_edge[1]];
      
      if (Ia_j < Ia_i):
        time_a = min(Il_j,Ia_i)-Ia_j;
        time_l = max(min(R_j,Ia_i),Il_j) - Il_j;
#        muij = math.exp(-beta_a_val*time_a) * math.exp(-beta_l_val*time_l);
        muij = math.exp(-beta_l_val*time_l);
        
      else:
        time_a = min(Il_i,Ia_j)-Ia_i;
        time_l = max(min(R_i,Ia_j),Il_i) - Il_i;
#        muij = math.exp(-beta_a_val*time_a) * math.exp(-beta_l_val*time_l);
        muij = math.exp(-beta_l_val*time_l);
    
      p_noinfect = (muij*p_edge1)/((1-p_edge1) + muij*p_edge1);
      
#      print("Bayes p_noinfect: ", p_noinfect)

      if g.has_edge(proposal_edge[0], proposal_edge[1]):
        if p_noinfect > .999999:
          MH_prob = -math.inf
        elif p_noinfect < .000001:
          MH_prob = math.inf
        else:
          MH_prob = math.log((1-p_noinfect)/p_noinfect)
      else:
        if p_noinfect > .999999:
          MH_prob = math.inf
        elif p_noinfect < .000001:
          MH_prob = -math.inf
        else:
          MH_prob = math.log(p_noinfect/(1 - p_noinfect))

#      print("Bayes MH_prob END: ", MH_prob)

  return MH_prob 
