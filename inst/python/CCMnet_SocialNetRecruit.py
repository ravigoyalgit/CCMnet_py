
import networkx as nx
import pandas as pd
import numpy as np
import random, json
import operator as op
from functools import reduce
from itertools import cycle, islice
import math
import cProfile

def pad_deg_dist(deg_dict,small_prob, num_nodes):
    '''
    Given the nonzero values of a degree distribution, pad all remaining degree sizes with a small probability. 
    '''
    deg_dict = { int(k) : v for k,v in deg_dict.items() }
    deg_dist = []
    for k in range(num_nodes):
        if k not in deg_dict or (k in deg_dict and deg_dict[k] == 0):
            # need to avoid zero values in deg dist for computational reasons
            deg_dist.append(small_prob)
        else:
            deg_dist.append(deg_dict[k]+small_prob)
    return deg_dist


def readconfig(CCMconfig):
  ccmc = json.load(open(CCMconfig))
  deg_dist = pad_deg_dist(ccmc["degree_distribution"],ccmc["small_prob"],ccmc["population"])
  Prob_Distr_Params = [ccmc["Prob_Distr_Params"][0], np.array(deg_dist)]
  return ccmc["Network_stats"],ccmc["Prob_Distr"],Prob_Distr_Params, ccmc["samplesize"],ccmc["burnin"], ccmc["interval"],ccmc["statsonly"], ccmc["G"],ccmc["P"],ccmc["population"], ccmc["covPattern"],ccmc["bayesian_inference"],ccmc["Ia"], ccmc["Il"], ccmc["R"], ccmc["epi_params"],ccmc["print_calculations"],ccmc["use_G"],ccmc["outfile"]
