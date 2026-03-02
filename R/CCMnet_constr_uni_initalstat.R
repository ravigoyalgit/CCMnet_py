#' Calculate initial network statistics
#'
#' @noRd

CCMnet_constr_uni_initalstat <- function(Network_stats, Prob_Distr, Prob_Distr_Params,
                                         nedges, g, max_degree,
                                         population, covPattern, remove_var_last_entry,
                                         CCM_constr_info) {
  
  if ((length(Network_stats) == 1) && (Network_stats == "edges" || Network_stats == "density")) {
    
    CCM_constr_info = CCMnet_constr_uni_initalstat_edges(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                         nedges, g, max_degree,
                                                         population, covPattern, remove_var_last_entry,
                                                         CCM_constr_info)
    
  } else if ((length(Network_stats) == 1) && (Network_stats == "mixing")) {
    
    CCM_constr_info = CCMnet_constr_uni_initalstat_mixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                          nedges, g, max_degree,
                                                          population, covPattern, remove_var_last_entry,
                                                          CCM_constr_info)
    
  } else if ((length(Network_stats) == 1) && (Network_stats == "degreedist")) {
    
    CCM_constr_info = CCMnet_constr_uni_initalstat_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                           nedges, g, max_degree,
                                                           population, covPattern, remove_var_last_entry,
                                                           CCM_constr_info)
    
  } else if (((length(Network_stats) == 2) && (Network_stats[1] == "mixing") && (Network_stats[2] == "degreedist")) ||
             ((length(Network_stats) == 2) && (Network_stats[1] == "degreedist") && (Network_stats[2] == "mixing"))) {
    
    CCM_constr_info = CCMnet_constr_uni_initalstat_mixing_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  nedges, g, max_degree,
                                                                  population, covPattern, remove_var_last_entry,
                                                                  CCM_constr_info)
    
  } else if ((length(Network_stats) == 1) && (Network_stats == "degmixing"))  {
    
    CCM_constr_info = CCMnet_constr_uni_initalstat_degmixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                             nedges, g, max_degree,
                                                             population, covPattern, remove_var_last_entry,
                                                             CCM_constr_info)
    
  } else if  (((length(Network_stats) == 2) && (Network_stats[1] == c("degmixing")) && (Network_stats[2] == c("triangles"))) ||
              ((length(Network_stats) == 2) && (Network_stats[1] == "triangles") && (Network_stats[2] == "degmixingg"))) {
    
    CCM_constr_info = CCMnet_constr_uni_initalstat_degmixing_clustering(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                        nedges, g, max_degree,
                                                                        population, covPattern, remove_var_last_entry,
                                                                        CCM_constr_info)
    
  } else {
    stop("No such NETWORK STATS currently implemented.")
    CCM_constr_info <- list(
      error = 1,
      prob_type = NULL,
      mean_vector = NULL,
      var_vector = NULL,
      Clist_nterms = NULL,
      Clist_fnamestring = NULL,
      Clist_snamestring = NULL,
      inputs =  NULL,
      eta0 = NULL,
      stats = NULL,
      MHproposal_name = NULL,
      MHproposal_package = NULL
    )
  }
  return(CCM_constr_info)
}
