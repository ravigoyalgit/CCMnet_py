
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import operator as op
from functools import reduce
from itertools import cycle, islice
import math
import cProfile
import copy

from CCMnet_utils import *
from CCMnet_call_netprop import *
from CCMnet_save_stats import *
from CCMnet_BayesDataIntegration import *
from CCMnet_SocialNetRecruit import *
from CCMnet_network_gen import *
from CCMnet_proposal import *

def CCMnet_constr_py(Network_stats=["Degree"],
                          Prob_Distr=["MultinomialPoisson"],
                          Prob_Distr_Params=[0,[]],
                          samplesize=1,
                          burnin=1000, 
                          interval=1,
                          statsonly=True, 
                          G=None,
                          P=None,
                          population=0, 
                          covPattern=[],
                          bayesian_inference=0,
                          Ia=[], 
                          Il=[], 
                          R=[], 
                          epi_params=[],
                          print_calculations=False,
                          use_G=0,
                          outfile="favites",
                          partial_network=0,
                          obs_nodes=[],
                          MH_proposal_type="random",
                          Obs_stats=None,
                          config_file=None,
                          degree_distribution=None,
                          small_prob=None):

  if config_file:
    Network_stats,Prob_Distr,Prob_Distr_Params, samplesize,burnin, interval,statsonly, G,P,population, covPattern,bayesian_inference,Ia, Il, R, epi_params,print_calculations,use_G,outfile = readconfig(config_file)
  if degree_distribution:
    deg_dist = pad_deg_dist(degree_distribution,small_prob,population)
    Prob_Distr_Params = [Prob_Distr_Params[0], np.array(deg_dist)]
  if use_G == 1:
    if isinstance(G,str):
      G = pd.read_csv(G)
    G_list = [tuple(r) for r in G.to_numpy().tolist()]
    g = generate_net(population, G_list, covPattern)
  else:
    g = generate_initial_g(population, covPattern)

  g_net_stat = calc_network_stat(g, Network_stats)
  g2_net_stat = copy.deepcopy(g_net_stat)

  if print_calculations:
    print("####Graph information: Begin####")
    print(g)
    print("####Graph information: End####")
    print("####Initial g statistics: Begin####")
    print(g_net_stat)
    print("####Initial g statistics: End####")

  if bayesian_inference == 1:
    if isinstance(P,str):
      P = pd.read_csv(P)
    P_list = [tuple(r) for r in P.to_numpy().tolist()]
    Pnet = generate_net(population, P_list, covPattern)
    P_net_stat = calc_network_stat(Pnet, Network_stats)
    if print_calculations:
      print("P info:", nx.info(Pnet))
      print("Pnet statistics:", P_net_stat)
  else:
    P_net_stat = 0

  results = [[] for _ in range(samplesize)]
  counter = 0

  for i in range(burnin+samplesize*interval):
    
    if print_calculations:
      print("####Graph information: Begin####")
      print(g)
      print("####Graph information: End####")
      print("####g statistics: Begin####")
      print(g_net_stat)
      print("####g statistics: End####")
    
    proposal_edge = proposal_edge_func(g, MH_proposal_type)
    g_proposal_edge = g.has_edge(proposal_edge[0], proposal_edge[1])
    g2_proposal_edge = not g_proposal_edge

    g2_net_stat = calc_network_stat_2(Network_stats, proposal_edge, g_net_stat, g2_net_stat, g_proposal_edge, covPattern, g, print_calculations)

    if print_calculations:
      print("####Proposal information: Begin####")
      print("Proposal Edge IDs: ", proposal_edge)
      print("Proposal Edge degrees: ", g.degree[proposal_edge[0]], g.degree[proposal_edge[1]])
      print("Proposal Edge in g: ", g_proposal_edge)
      print("Proposal Edge in g2: ", g2_proposal_edge)
      print("####Proposal information: End####")
      print("####g2 statistics: Begin####")
      print(g2_net_stat)
      print("####g2 statistcs: End####")
      
    f_g_g2_bool = True
    f_g_g2 = calc_f(Network_stats,g_net_stat, g2_net_stat, proposal_edge, g_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

    f_g_g2_bool = False
    f_g2_g = calc_f(Network_stats,g2_net_stat, g_net_stat, proposal_edge, g2_proposal_edge, covPattern, bayesian_inference, P_net_stat, g, f_g_g2_bool, print_calculations)

    if print_calculations:
      print("####CCM calculations: Begin####")
      print("g->g2: ", f_g_g2)
      print("g2->g: ", f_g2_g)
      print("####CCM calculations: End####")
      
    prob_g, prob_g2 = calc_probs(g_net_stat, g2_net_stat, proposal_edge, covPattern, Network_stats, Prob_Distr, Prob_Distr_Params, g, g_proposal_edge, print_calculations)

    if print_calculations:
      print("####Prob calculations: Begin####")
      print("g: ", prob_g)
      print("g2: ", prob_g2)
      print("####Prob calculations: End####")

    if math.isnan(prob_g):
      MH_prob = math.inf
    elif math.isnan(prob_g2) or f_g_g2<=0 or f_g2_g<=0:
      MH_prob = -math.inf
    else: 
      MH_prob = math.log(f_g2_g) + prob_g2 - (math.log(f_g_g2) + prob_g)
    
    if MH_proposal_type == "TNT":
      nedges = g.number_of_edges()
      num_nodes = g.number_of_nodes()
      ndyads = num_nodes * (num_nodes-1) * 0.5
      if g_proposal_edge:
        if nedges == 1:
          MH_prob_TNT = math.log(1.0/(ndyads + 0.5))
        else:
          MH_prob_TNT = math.log(nedges / (ndyads + nedges))
      else:
        if nedges == 0:
          MH_prob_TNT = math.log(ndyads + 0.5)
        else:
          MH_prob_TNT = math.log(1.0 + (ndyads/(nedges+1))) ##Updated
      MH_prob = MH_prob + MH_prob_TNT

    if print_calculations:
      print(MH_prob)

    if bayesian_inference == 1 and MH_prob < math.inf:
      MH_prob = bayes_inf_MH_prob_calc(MH_prob, g, proposal_edge, Pnet, Ia, Il, R, epi_params)

    if partial_network == 1:
        if ((proposal_edge[0] in obs_nodes) and (proposal_edge[1] in obs_nodes)): 
          #Reject the toggle
          MH_prob = -math.inf
      
    if MH_prob >= 0 or math.log(np.random.uniform(0,1)) < MH_prob:
      #Accept proposal
      if print_calculations:
        print("###################Proposal: Accept####")
      g = proposal_g2(g, proposal_edge)
      g_net_stat = copy.deepcopy(g2_net_stat)
    else:   
      #Reject proposal
      if print_calculations:
        print("###################Proposal: Reject####")
      g2_net_stat = copy.deepcopy(g_net_stat)

    if (i+1) % interval == 0 and (i+1) > burnin:
      if statsonly:
        save_stats(g_net_stat, results, counter, Network_stats, g, Obs_stats)
      else:
        save_network(results, counter)
      counter = counter + 1

  g_df = nx.to_pandas_edgelist(g)
  results = pd.DataFrame(np.vstack(results))

  if outfile == "favites":
    return g
  else:
    return g_df, results
