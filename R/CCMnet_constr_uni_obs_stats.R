#' Dispatcher for Observed Statistics Constraints in CCMnet
#'
#' This function identifies which specific constraint initialization function to 
#' call based on the provided \code{Obs_stats}. It aggregates the results and 
#' updates the \code{CCM_constr_info} object to be passed to the C-level sampler.
#'
#' @param CCM_constr_info A list containing existing constraint information.
#' @param Network_stats A list or vector of current network statistics.
#' @param Prob_Distr A character vector specifying the distributions to be used.
#' @param Prob_Distr_Params A nested list of parameters (means, variances, etc.).
#' @param samplesize Integer. Number of MCMC samples to collect.
#' @param burnin Integer. Number of initial iterations to discard.
#' @param interval Integer. Thinning interval for MCMC sampling.
#' @param statsonly Logical. If \code{TRUE}, only statistics are returned.
#' @param G An \code{igraph} or network object.
#' @param population Integer. The number of nodes in the network.
#' @param covPattern A vector representing nodal attributes.
#' @param remove_var_last_entry Logical. If \code{TRUE}, removes linear 
#'   dependencies in degree distributions during variance inversion.
#' @param Obs_stats A character vector specifying the types of statistics to 
#'   constrain. Supported values: "Edges", "Mixing", "DegreeDist", 
#'   "DegMixing", "Triangles". Can be a combination (e.g., \code{c("DegMixing", "Triangles")}).
#'
#' @details 
#' The function uses conditional logic to match \code{Obs_stats} to the 
#' appropriate helper function (e.g., \code{\link{CCMnet_constr_uni_degmixing_clustering}}). 
#' 
#' \strong{Special Handling for DegreeDist:} 
#' If \code{Obs_stats} is "DegreeDist", the function overrides \code{Prob_Distr} 
#' to "DirMult" and sets a default parameter list of ones.
#' 
#' 
#'
#' @return An updated \code{CCM_constr_info} list with the following updated fields:
#' \itemize{
#'   \item \code{Clist_nterms}
#'   \item \code{Clist_fnamestring}
#'   \item \code{Clist_snamestring}
#'   \item \code{inputs}
#'   \item \code{eta0}
#'   \item \code{stats}
#' }
#' 
#' @noRd

CCMnet_constr_uni_obs_stats <-function(CCM_constr_info, Network_stats, Prob_Distr, Prob_Distr_Params,
                            samplesize, burnin, interval,
                            statsonly, nedges, g, max_degree,
                            population, covPattern, remove_var_last_entry,
                            Obs_stats) {
  
  if ((length(Obs_stats) == 1) && (Obs_stats == "Edges")) {
    
    CCM_constr_info_verify = CCMnet_constr_uni_verifyinput_edges(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                          population, covPattern, remove_var_last_entry)
    
    CCM_constr_info_initial = CCMnet_constr_uni_initalstat_edges(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                         nedges, g, max_degree,
                                                         population, covPattern, remove_var_last_entry,
                                                         CCM_constr_info)
    
  } else if ((length(Obs_stats) == 1) && (Obs_stats == "Mixing")) {
    
    CCM_constr_info_verify = CCMnet_constr_uni_verifyinput_mixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                 population, covPattern, remove_var_last_entry)
    
    CCM_constr_info_initial = CCMnet_constr_uni_initalstat_mixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                 nedges, g, max_degree,
                                                                 population, covPattern, remove_var_last_entry,
                                                                 CCM_constr_info)
    
  } else if ((length(Obs_stats) == 1) && (Obs_stats == "DegreeDist")) {
    Prob_Distr='DirMult'
    Prob_Distr_Params=list(list(rep(1,population)))
    
    CCM_constr_info_verify = CCMnet_constr_uni_verifyinput_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  population, covPattern, remove_var_last_entry)
    
    CCM_constr_info_initial = CCMnet_constr_uni_initalstat_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  nedges, g, max_degree,
                                                                  population, covPattern, remove_var_last_entry,
                                                                  CCM_constr_info)
    
  } else if (((length(Obs_stats) == 2) && (Obs_stats[1] == "Mixing") && (Obs_stats[2] == "DegreeDist")) ||
             ((length(Obs_stats) == 2) && (Obs_stats[1] == "DegreeDist") && (Obs_stats[2] == "Mixing"))) {
    
    CCM_constr_info_verify = CCMnet_constr_uni_verifyinput_mixing_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  population, covPattern, remove_var_last_entry)
    
    CCM_constr_info_initial = CCMnet_constr_uni_initalstat_mixing_degdist(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  nedges, g, max_degree,
                                                                  population, covPattern, remove_var_last_entry,
                                                                  CCM_constr_info)
    
  } else if ((length(Obs_stats) == 1) && (Obs_stats == "DegMixing"))  {
    
    CCM_constr_info_verify = CCMnet_constr_uni_verifyinput_degmixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  population, covPattern, remove_var_last_entry)
    
    CCM_constr_info_initial = CCMnet_constr_uni_initalstat_degmixing(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  nedges, g, max_degree,
                                                                  population, covPattern, remove_var_last_entry,
                                                                  CCM_constr_info)
    
  } else if  (((length(Obs_stats) == 2) && (Obs_stats[1] == c("DegMixing")) && (Obs_stats[2] == c("Triangles"))) ||
              ((length(Obs_stats) == 2) && (Obs_stats[1] == "Triangles") && (Obs_stats[2] == "DegMixingg"))) {
    
    CCM_constr_info_verify = CCMnet_constr_uni_verifyinput_degmixing_clustering(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  population, covPattern, remove_var_last_entry)
    
    CCM_constr_info_initial = CCMnet_constr_uni_initalstat_degmixing_clustering(Network_stats, Prob_Distr, Prob_Distr_Params,
                                                                  nedges, g, max_degree,
                                                                  population, covPattern, remove_var_last_entry,
                                                                  CCM_constr_info)
  } else {
    stop("No such NETWORK PROPERTY of observation currently implemented.")
    return(CCM_constr_info)
  }
  
  ####NEED TO CHANGE BY ADDING#######
  ####THIS REQUIRES CHANGING THE C CODE TO MAKE SURE CHANGE STATISTICS ARE ALIGNED######
  
  #CCM_constr_info[["error"]]
  #CCM_constr_info[["prob_type"]]
  #CCM_constr_info[["mean_vector"]]
  #CCM_constr_info[["var_vector"]]
  CCM_constr_info[["Clist_nterms"]] = CCM_constr_info_verify[["Clist_nterms"]]
  CCM_constr_info[["Clist_fnamestring"]] = CCM_constr_info_verify[["Clist_fnamestring"]]
  CCM_constr_info[["Clist_snamestring"]] = CCM_constr_info_verify[["Clist_snamestring"]]
  CCM_constr_info[["inputs"]] = CCM_constr_info_verify[["inputs"]]
  CCM_constr_info[["eta0"]] = CCM_constr_info_verify[["eta0"]]
  CCM_constr_info[["stats"]] = CCM_constr_info_initial[["stats"]]
  #CCM_constr_info[["MHproposal_name"]]
  #CM_constr_info[["MHproposal_package"]]
  
  return(CCM_constr_info)
}
