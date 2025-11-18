# 
# 

#reticulate::conda_create("ccmnet_env", packages = c("python=3.11", "numpy=2.1.3"))
#reticulate::use_condaenv("ccmnet_env", required = TRUE)
#reticulate::py_config()
#reticulate::py_install("networkx")
#reticulate::py_install("pandas")
#
#reticulate::source_python(paste(find.package("CCMnetpy"), "/python/CCMnet_constr_py.py", sep = ""), envir = globalenv())
#
#reticulate::use_python("/Users/ravigoyal/anaconda3/bin/python", required = TRUE)
#reticulate::py_config()
#
#library(reticulate)
#library(CCMnetpy)
#library(tidyverse)
#library(igraph)
#
#CCMnet_python_setup()
# 
# #Initialize
# 
# deg_dist="nbinom"
# size=1.017340
# mu=6.192894
# 
# alpha =  1
# MCMC_wgt = 1
# 
# sample_fraction = 0.1
# n_mcmc_trials = 100 #2000 ########CHANGE
# 
# #Population parameters
# population = 1000
# covPattern = c(rep(0,population/2), rep(1,population/2)) #only use for Mixing
# 
# prior_mult = 10000
# #Network model parameters
# Network_stats = list(c("Degree"))
# Prob_Distr = list(c("Multinomial_Poisson"))
# Prob_Distr_Params_prior = vector("list", 3)
# Prob_Distr_Params_prior[[1]] = 'Dirichlet_Gamma'
# Prob_Distr_Params_prior[[2]][1] = 1 #gamma k (shape)
# Prob_Distr_Params_prior[[2]][2] = population #gamma theta (scale)
# Prob_Distr_Params_prior[[3]] = rep(1/(population*prior_mult), population) #dirichlet alpha
# 
# Prob_Distr_Params = vector("list", 2)
# Prob_Distr_Params[[1]] = c(population) #Number or edges in mixing matrix [1,1], [1,2], and [2,2]
# Prob_Distr_Params[[2]] =  dnbinom(c(0:(population-1)),size=1.017340,mu=6.192894,log=FALSE)
# 
# #Population parameters
# population = 100
# covPattern = c(rep(0,population)) #only use for Mixing
# 
# prior_mult = 10000
# #Network model parameters
# Network_stats = list(c("Edge"))
# Prob_Distr = list(c("NP"))
# Prob_Distr_Params = vector("list", 2)
# Prob_Distr_Params[[1]] =  dnbinom(c(0:choose(population,2)),size=1.017340,mu=6.192894,log=FALSE)
# 
# #dbinom(c(0:choose(population,2)), size = choose(population,2), prob = 0.5, log = FALSE) + 1/prior_mult
# 
# Network_stats=Network_stats
# Prob_Distr=Prob_Distr
# Prob_Distr_Params=Prob_Distr_Params
# samplesize = as.integer(1000)
# burnin=as.integer(200000)
# interval=as.integer(100)
# statsonly=TRUE
# G=NULL
# P=NULL
# population=as.integer(population)
# covPattern = as.integer(covPattern)
# bayesian_inference = FALSE
# Ia = NULL
# Il = NULL
# R = NULL
# epi_params = NULL
# print_calculations = FALSE
# use_G = FALSE
# outfile = "none"
# partial_network = as.integer(0)
# obs_nodes = NULL
# MH_proposal_type = "random"
# # 
# CCMnet_Result = CCMnetpy::CCMnet_constr(Network_stats=Network_stats,
#                                         Prob_Distr=Prob_Distr,
#                                         Prob_Distr_Params=Prob_Distr_Params,
#                                         samplesize = as.integer(10000),
#                                         burnin=as.integer(100000), ##CHANGE
#                                         interval=as.integer(1000),
#                                         statsonly=TRUE,
#                                         G=NULL,
#                                         P=NULL,
#                                         population=as.integer(population),
#                                         covPattern = as.integer(covPattern),
#                                         bayesian_inference = FALSE,
#                                         Ia = NULL,
#                                         Il = NULL,
#                                         R = NULL,
#                                         epi_params = NULL,
#                                         print_calculations = FALSE,
#                                         use_G = FALSE,
#                                         outfile = "none",
#                                         partial_network = as.integer(0),
#                                         obs_nodes = NULL,
#                                         MH_proposal_type = "random")
# 
# CCM_stats = CCMnet_Result[[2]]
# apply(CCM_stats, 2, mean)[c(1:10)]
# (Prob_Distr_Params[[1]] *population)[c(1:10)]
# 
# plot(CCM_stats[,2])
# plot(CCM_stats[,1])
# 
# mean(rnbinom(10000,size=1.017340,mu=6.192894))

