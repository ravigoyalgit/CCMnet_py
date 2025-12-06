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

######################
######################
######################
# 
# reticulate::use_condaenv("ccmnet_env", required = TRUE)
# 
# library(reticulate)
# library(CCMnetpy)
# library(tidyverse)
# library(igraph)
# 
# # Initialize CCMnet Python code
# CCMnet_python_setup()
# 
# population <- 100
# covPattern <- rep(0L, population)
# 
# # Target statistic: edge count
# Network_stats <- list("Edge")
# Prob_Distr    <- list("NP")
# 
# # Negative binomial prior on number of edges
# Prob_Distr_Params <- vector("list", 2)
# Prob_Distr_Params[[1]] <- dnbinom(
#   0:choose(population, 2),
#   size = 1.017340,
#   mu   = 6.192894
# )
# 
# # Run a small MCMC chain (for demonstration)
# result <- CCMnetpy::CCMnet_constr(
#   Network_stats      = Network_stats,
#   Prob_Distr         = Prob_Distr,
#   Prob_Distr_Params  = Prob_Distr_Params,
#   G=NULL,
#   P=NULL,
#   samplesize         = 1000L,
#   burnin             = 50000L,
#   interval           = 10000L,
#   statsonly          = TRUE,
#   population         = population,
#   covPattern         = covPattern,
#   bayesian_inference = FALSE,
#   Ia = NULL,
#   Il = NULL,
#   R = NULL,
#   epi_params = NULL,
#   print_calculations = FALSE,
#   use_G = FALSE,
#   outfile = "none",
#   partial_network = as.integer(0),
#   obs_nodes = NULL,
#   MH_proposal_type = "TNT"
# )
# 
# # Extract results
# g     <- result[[1]]
# stats <- result[[2]]
# 
# library(ggplot2)
# library(tidyverse)
# 
# # MCMC sampled edge counts
# df_sampled <- data.frame(
#   value = stats[,population + 1],
#   type  = "MCMC Sample"
# )
# 
# # Negative binomial simulated prior
# set.seed(123)
# df_prior <- data.frame(
#   value = rnbinom(10000, size = 1.017340, mu = 6.192894),
#   type  = "Negative Binomial"
# )
# 
# # Combine
# df_all <- bind_rows(df_sampled, df_prior)
# 
# # Density overlay
# ggplot(df_all, aes(x = value, color = type, fill = type)) +
#   geom_density(alpha = 0.3, linewidth = 1.2) +
#   labs(
#     title = "MCMC Sample vs. Negative Binomial",
#     x = "Number of Edges",
#     y = "Density"
#   ) +
#   theme_minimal(base_size = 14) +
#   scale_color_manual(values = c("MCMC Sample" = "red", "Negative Binomial" = "blue")) +
#   scale_fill_manual(values = c("MCMC Sample" = "red", "Negative Binomial" = "blue"))
# 
# # ######################
# # ######################
# # ######################
# 
# reticulate::use_condaenv("ccmnet_env", required = TRUE)
# 
# library(reticulate)
# library(CCMnetpy)
# library(tidyverse)
# library(igraph)
# 
# # Initialize CCMnet Python code
# CCMnet_python_setup()
# 
# #Population parameters
# population = 100
# covPattern = c(rep(0,population)) #only use for Mixing
# 
# #Network model parameters
# Network_stats = list(c("Degree"))
# Prob_Distr = list(c("Multinomial_Poisson"))
# 
# Prob_Distr_Params = vector("list", 2)
# Prob_Distr_Params[[1]] = c(population) #Number or edges in mixing matrix [1,1], [1,2], and [2,2]
# Prob_Distr_Params[[2]] =  dnbinom(c(0:(population-1)),size=1.017340,mu=6.192894,log=FALSE)
# 
# Network_stats=Network_stats
# Prob_Distr=Prob_Distr
# Prob_Distr_Params=Prob_Distr_Params
# samplesize = as.integer(1000)
# burnin=as.integer(200000)
# interval=as.integer(1000)
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
# MH_proposal_type = "TNT"
# #
# 
# CCMnet_Result = CCMnetpy::CCMnet_constr(Network_stats=Network_stats,
#                                         Prob_Distr=Prob_Distr,
#                                         Prob_Distr_Params=Prob_Distr_Params,
#                                         samplesize = samplesize,
#                                         burnin=burnin,
#                                         interval=interval,
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
#                                         MH_proposal_type = "TNT")
# 
# degrees_to_check <- 0:9
# population <- population                   # already defined
# theoretical_pmf <- Prob_Distr_Params[[2]]  # negative binomial PMF
# n_sim <- 1000  # number of multinomial draws
# 
# ### --- EMPIRICAL (MCMC) DISTRIBUTION ---
# df_empirical <- CCM_stats %>%
#   as.data.frame() %>%
#   pivot_longer(
#     cols = everything(),      # all columns are node degrees
#     names_to = "degree",
#     values_to = "count"
#   ) %>%
#   filter(degree %in% degrees_to_check) %>%
#   group_by(degree) %>%
#   mutate(source = "MCMC")
# 
# ### --- THEORETICAL DISTRIBUTION ---
# set.seed(123)
# multi_pois <- t(rmultinom(1000, population, prob =  theoretical_pmf))
# df_theory <- multi_pois %>%
#   as.data.frame()
# 
# colnames(df_theory) = c(0:(population-1))
# 
# df_theory = df_theory %>%
#   pivot_longer(
#     cols = everything(),      # all columns are node degrees
#     names_to = "degree",
#     values_to = "count"
#   ) %>%
#   filter(degree %in% degrees_to_check) %>%
#   group_by(degree) %>%
#   mutate(source = "Theoretical")
# 
# ### --- COMBINE AND PLOT ---
# df_plot <- bind_rows(df_empirical, df_theory)
# 
# ggplot(df_plot, aes(x = count, color = source, fill = source)) +
#   geom_density(alpha = 0.25) +
#   facet_wrap(~degree, scales = "free", ncol = 5) +
#   theme_bw() +
#   labs(
#     title = "Degree Distribution: Empirical MCMC vs Theoretical Multinomial-Poisson (Degrees 0–9)",
#     x = "Count per Degree",
#     y = "Density"
#   )
# 
# 
# # ######################
# # ######################
# # ######################
# 
# reticulate::use_condaenv("ccmnet_env", required = TRUE)
# 
# library(reticulate)
# library(CCMnetpy)
# library(tidyverse)
# library(igraph)
# 
# # Initialize CCMnet Python code
# CCMnet_python_setup()
# 
# population <- 100
# covPattern <- c(rep(0L, population/2),rep(1L, population/2))
# 
# Network_stats = list(c("Mixing"))
# Prob_Distr = list(c("Multinomial_Poisson"))
# 
# Prob_Distr_Params = vector("list", 2)
# Prob_Distr_Params[[1]] = c(20) #Number or edges in mixing matrix [1,1], [1,2], and [2,2]
# Prob_Distr_Params[[2]] =  c(.3, .6, .1)
# 
# Network_stats=Network_stats
# Prob_Distr=Prob_Distr
# Prob_Distr_Params=Prob_Distr_Params
# samplesize = as.integer(1000)
# burnin=as.integer(200000)
# interval=as.integer(1000)
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
# MH_proposal_type = "TNT"
# #
# 
# CCMnet_Result = CCMnetpy::CCMnet_constr(Network_stats=Network_stats,
#                                         Prob_Distr=Prob_Distr,
#                                         Prob_Distr_Params=Prob_Distr_Params,
#                                         samplesize = samplesize,
#                                         burnin=burnin,
#                                         interval=interval,
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
#                                         MH_proposal_type = "TNT")
# 
# stats = CCMnet_Result[[2]]
# colnames(stats) <- c("M11", "M12", "M22")
# 
# multi_pois = c()
# for (i in c(1:1000)) {
#   multi_pois = bind_cols(multi_pois, rmultinom(1, rpois(1,20), prob =  c(.3, .6, .1)))
# }
# multi_pois = t(multi_pois)
# colnames(multi_pois) <- c("M11", "M12", "M22")
# 
# # Put into long format
# df_stats <- stats %>%
#   as.data.frame() %>%
#   mutate(source = "MCMC") %>%
#   pivot_longer(cols = c(M11, M12, M22), names_to = "metric", values_to = "count")
# 
# df_pois <- multi_pois %>%
#   as.data.frame() %>%
#   mutate(source = "Poisson") %>%
#   pivot_longer(cols = c(M11, M12, M22), names_to = "metric", values_to = "count")
# 
# df_all <- bind_rows(df_stats, df_pois)
# 
# # Plot
# ggplot(df_all, aes(x = count, color = source, fill = source)) +
#   geom_density(alpha = 0.3) +
#   facet_wrap(~metric, scales = "free") +
#   theme_bw() +
#   labs(
#     title = "Distribution of Edge Counts: MCMC vs Poisson Multinomial",
#     x = "Count",
#     y = "Density"
#   )
# 
