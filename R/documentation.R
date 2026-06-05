#' Network Properties in CCMnet
#'
#' @name ccm_properties
#' @description \pkg{CCMnet} implements a systematic hierarchy of network properties 
#' motivated by the dk-series framework (Mahadevan et al., 2006; Orsini et al., 2015). 
#' These properties can be targeted using soft constraints governed by 
#' user-specified probability distributions.
#'
#' @section Topological-based Properties (dk-series):
#' The following properties follow the dk-series framework, which defines 
#' increasingly constrained random graph ensembles:
#' \describe{
#'   \item{\code{"edges"} (0k)}{The total number of edges within the graph. This 
#'   is the simplest topological constraint.}
#'   \item{\code{"degreedist"} (1k)}{A vector where the j-th entry represents 
#'   the number of nodes having degree j.}
#'   \item{\code{"degmixing"} (2k)}{The joint degree distribution, represented 
#'   as a degree mixing matrix. The (i,j) entry represents the number of edges 
#'   between nodes with degree i and degree j, capturing degree-degree 
#'   correlations (degree assortativity).}
#'   \item{\code{"degmixing" + "triangles"} (2.1k)}{Extends the 2k distribution by including 
#'   the total number of closed triads (triangles) in the network, allowing for 
#'   clustering constraints.}
#' }
#'
#' @section Covariate-based Properties:
#' \pkg{CCMnet} incorporates node-level attributes by defining congruence classes 
#' based on categorical or discretized covariates (e.g., homophily or social 
#' stratification). 
#' \describe{
#'   \item{\code{"mixing"}}{The attribute mixing matrix. The (i,j) entry 
#'   represents the number of edges between nodes belonging to covariate groups 
#'   $i$ and $j$. Requires \code{cov_pattern}.}
#'   \item{\code{"degreedist+degreedist+mixing"}}{A combination property that includes the 
#'   attribute mixing matrix alongside separate degree distributions calculated 
#'   for each distinct covariate group. Currently on a binary covariate is implemented. 
#'   Therefore, input is characteristics for two degree distributions and one scalar mixing value.}
#' }
#'
#' @references
#' Mahadevan, P., Krioukov, D., Fomenkov, K., Huffaker, B., & Vahdat, A. (2006). 
#' Systematic topology analysis and generation using degree correlations. 
#' \emph{ACM SIGCOMM Computer Communication Review}.
#'
#' Orsini, C., et al. (2015). Quantifying randomness in real networks. 
#' \emph{Nature Communications}.
#'
#' @family ccm_core
#' @seealso \code{\link{ccm_distributions}}, \code{\link{sample_ccm}}
NULL

#' Probability Distributions in CCMnet
#'
#' @name ccm_distributions
#' @description Details on the probability distributions implemented in the 
#' \pkg{CCMnet}. These distributions define the target distribution 
#' placed on network properties during the MCMC sampling process.
#' 
#' @section Supported Distributions:
#' \describe{
#'   \item{\code{"poisson"}}{Requires \code{list(lambda)}. Typically used for 
#'   count-based statistics like \code{"edges"} or \code{"triangles"}.}
#'   \item{\code{"gamma"}}{Requires \code{list(shape, rate)}. Useful for 
#'   continuous or skewed properties. The implementation uses the kernel 
#'   \eqn{x^{\alpha-1}e^{-\beta x}}.}
#'   \item{\code{"dirmult"}}{Requires \code{list(alphas)}. A Dirichlet-Multinomial 
#'   implementation optimized for proportions. In \pkg{CCMnet}, the global 
#'   normalizing constant is omitted to facilitate sampling in systems where 
#'   the total count (e.g., total edges) is variable.}
#'   \item{\code{"normal"}}{Requires \code{list(mean, sd)}. Standard Gaussian 
#'   constraint.}
#'   \item{\code{"lognormal"}}{Requires \code{list(log mean, log sd)}. Log-scale 
#'   of standard Gaussian constraint.}
#'   \item{\code{"beta"}}{Requires \code{list(shape1, shape2)}. Restricted to 
#'   properties bounded in the interval [0,1], such as \code{"density"}.}
#'   \item{\code{"uniform"}}{A flat distribution where the MCMC explores the 
#'   congruence class without preference for specific statistic values.}
#' }
#' 
#' @details 
#' By decoupling the probability distribution from the network property, 
#' \pkg{CCMnet} allows researchers to represent structural uncertainty. For 
#' example, one might target a specific degree distribution via \code{"dirmult"} 
#' while allowing the total edge count to follow a wide \code{"gamma"} distribution.
#' 
#' @family ccm_core
#' @seealso \code{\link{ccm_properties}}, \code{\link{sample_ccm}}
NULL
