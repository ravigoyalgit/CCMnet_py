#' Bayesian Inference for a Partially Observed Network
#'
#' Two-block Gibbs sampler for a partially observed network under a Congruence
#' Class Model (CCM). The network properties and their probability
#' distributions are specified exactly as in \code{\link{sample_ccm}}
#' (\code{network_stats}, \code{prob_distr}, \code{prob_distr_params}); in
#' addition, a prior is placed on the parameters of each distribution
#' (\code{prior_distr}, \code{prior_params}). Each iteration
#' \enumerate{
#'   \item draws the network given the current parameters with
#'     \code{sample_ccm(..., partial_network = TRUE, partial_g = partial_g)},
#'     so only the missing dyads (those that are not edges of
#'     \code{partial_g}) change; then
#'   \item draws each property's parameters from their full conditional given
#'     the network's statistics (a conjugate update; see below).
#' }
#'
#' @param g_obs \code{igraph} on all \code{population} nodes. Its edges on
#'   fixed dyads are the observed edges; any other edges are starting values.
#' @param partial_g \code{igraph} on all \code{population} nodes whose edges
#'   are the fixed (known) dyads. For node sampling use
#'   \code{make_partial_g(population, obs_nodes)}.
#' @param network_stats,prob_distr As in \code{\link{sample_ccm}}.
#' @param prob_distr_params As in \code{\link{sample_ccm}}. These are the
#'   starting values of the parameters; components that the prior does not
#'   update (e.g. the variance of a \code{"normal"} distribution, or the shape
#'   of a \code{"gamma"} distribution) stay fixed at these values.
#' @param prior_distr Character vector, one prior per element of
#'   \code{network_stats}; see the table below. Use \code{"fixed"} to keep a
#'   property's parameters fixed. An element may also be a function
#'   \code{function(params, h, prior_params)} returning updated
#'   \code{params}, for priors not built in.
#' @param prior_params List, one element per property: a list of the prior's
#'   parameters (see the table). Ignored for \code{"fixed"}. Scalars are
#'   recycled to the length of the statistic.
#' @param population Number of nodes.
#' @param cov_pattern Node group labels, as in \code{\link{sample_ccm}}
#'   (needed for \code{"mixing"}; groups must be labelled \code{1..k}).
#' @param n_mcmc_trials Number of retained Gibbs iterations.
#' @param burnin_trials Number of initial Gibbs iterations discarded.
#' @param ccm_steps Number of CCM toggles in each network update.
#' @param verbose Logical; report progress.
#'
#' @section Built-in conjugate updates:
#' \tabular{lll}{
#'   \code{prob_distr} \tab \code{prior_distr} and \code{prior_params} \tab Parameter updated \cr
#'   \code{"poisson"} \tab \code{"gamma"}: \code{list(shape, rate)} \tab
#'     \eqn{\lambda_k \sim \mathrm{Gamma}(shape_k + h_k,\ rate_k + 1)} \cr
#'   \code{"poisson"} (\code{degreedist} only) \tab \code{"dirichlet"}: \code{list(alpha)} \tab
#'     \eqn{\theta \sim \mathrm{Dirichlet}(\alpha + h)}; with \eqn{\sum h = n} fixed
#'     this is the multinomial model \cr
#'   \code{"normal"} \tab \code{"normal"}: \code{list(mean, var)} \tab
#'     mean, with the variance (parameter 2) known \cr
#'   \code{"lognormal"} \tab \code{"normal"}: \code{list(mean, var)} \tab
#'     meanlog, with varlog (parameter 2) known; uses \eqn{\log h} \cr
#'   \code{"mvn"} \tab \code{"normal"}: \code{list(mean, cov)} \tab
#'     mean vector, with the covariance (parameter 2) known \cr
#'   \code{"gamma"} \tab \code{"gamma"}: \code{list(shape, rate)} \tab
#'     rate (parameter 2), with the shape (parameter 1) known \cr
#'   any \tab \code{"fixed"} \tab none \cr
#' }
#' Here \eqn{h} is the property's statistic in the current network (one row of
#' \code{mcmc_stats} from \code{sample_ccm}), treated as a single observation
#' from \code{prob_distr}. \code{"dirmult"} already integrates out its
#' probabilities, so use \code{"fixed"} with it; \code{"beta"},
#' \code{"uniform"} and \code{"np"} have no built-in conjugate prior.
#'
#' @return A list with \code{stats} (data frame of the network statistics per
#'   retained iteration, columns named as in \code{sample_ccm}), \code{params}
#'   (list, one matrix per property of the updated parameter per retained
#'   iteration; \code{NULL} for fixed properties), \code{g_final}, and the
#'   settings.
#'
#' @examples
#' \dontrun{
#' # Degree distribution, multinomial model with a Dirichlet prior
#' K <- 20
#' fit <- CCM_MissingInference(g_obs, partial_g,
#'   network_stats = "degreedist", prob_distr = "poisson",
#'   prob_distr_params = list(list(rep(1 / (K + 1), K + 1))),
#'   prior_distr = "dirichlet", prior_params = list(list(alpha = 0.5)))
#'
#' # Mixing between two groups, Poisson cell counts with Gamma priors
#' fit <- CCM_MissingInference(g_obs, partial_g,
#'   network_stats = "mixing", prob_distr = "poisson",
#'   prob_distr_params = list(list(c(50, 20, 50))),
#'   prior_distr = "gamma", prior_params = list(list(shape = 1, rate = 0.01)),
#'   cov_pattern = groups)
#' }
#' @export

CCM_MissingInference <- function(g_obs,
                                 partial_g,
                                 network_stats,
                                 prob_distr,
                                 prob_distr_params,
                                 prior_distr,
                                 prior_params = NULL,
                                 population = igraph::vcount(g_obs),
                                 cov_pattern = NULL,
                                 n_mcmc_trials = 1000,
                                 burnin_trials = 100,
                                 ccm_steps = 10000,
                                 verbose = FALSE) {
  
  .validate_ccm_missing_inputs(g_obs, partial_g, network_stats, prob_distr, prob_distr_params,
                               prior_distr, prior_params, population, cov_pattern,
                               n_mcmc_trials, burnin_trials, ccm_steps)
  n_props <- length(network_stats)
  if (!is.list(prior_distr)) prior_distr <- as.list(prior_distr)
  if (is.null(prior_params)) prior_params <- vector("list", n_props)
  
  # Width of each property's block of columns in mcmc_stats
  widths <- vapply(seq_len(n_props), function(i) {
    if (prob_distr[i] %in% c("np", "uniform")) 1L else length(prob_distr_params[[i]][[1]])
  }, integer(1))
  
  # Resolve the update rule for each property and check its prior
  updaters <- lapply(seq_len(n_props), function(i) {
    .ccm_get_updater(network_stats[i], prob_distr[i], prior_distr[[i]],
                     prior_params[[i]], widths[i])
  })
  
  params <- prob_distr_params
  g <- g_obs
  n_total <- burnin_trials + n_mcmc_trials
  stats_mat <- NULL
  param_draws <- lapply(seq_len(n_props), function(i) {
    if (is.null(updaters[[i]]$fn)) NULL else
      matrix(NA_real_, n_mcmc_trials, widths[i])
  })
  
  for (it in seq_len(n_total)) {
    
    # 1. Network | parameters: only dyads that are not edges of partial_g change
    fit <- sample_ccm(network_stats = network_stats,
                      prob_distr = prob_distr,
                      prob_distr_params = params,
                      population = population,
                      cov_pattern = cov_pattern,
                      sample_size = 2,          # sample_ccm requires >= 2; keep the last
                      burnin = ccm_steps,
                      interval = 1,
                      initial_g = g,
                      use_initial_g = TRUE,
                      stats_only = TRUE,
                      partial_network = TRUE,
                      partial_g = partial_g)
    g <- fit$g[[length(fit$g)]]
    h_all <- as.numeric(fit$mcmc_stats[nrow(fit$mcmc_stats), ])
    if (is.null(stats_mat)) {
      if (length(h_all) != sum(widths)) {
        stop(sprintf("sample_ccm returned %d statistics but the parameters imply %d.",
                     length(h_all), sum(widths)))
      }
      stats_mat <- matrix(NA_real_, n_mcmc_trials, length(h_all),
                          dimnames = list(NULL, colnames(fit$mcmc_stats)))
    }
    
    # 2. Parameters | network, one property at a time
    ends_ <- cumsum(widths)
    for (i in seq_len(n_props)) {
      if (is.null(updaters[[i]]$fn)) next
      h <- h_all[(ends_[i] - widths[i] + 1):ends_[i]]
      params[[i]] <- updaters[[i]]$fn(params[[i]], h, updaters[[i]]$prior)
    }
    
    if (it > burnin_trials) {
      j <- it - burnin_trials
      stats_mat[j, ] <- h_all
      for (i in seq_len(n_props)) {
        if (!is.null(param_draws[[i]])) {
          param_draws[[i]][j, ] <- params[[i]][[updaters[[i]]$which]]
        }
      }
    }
    if (verbose && it %% max(1, n_total %/% 10) == 0) {
      message("Gibbs iteration ", it, " / ", n_total)
    }
  }
  
  names(param_draws) <- network_stats
  obj <- list(mcmc_stats = as.data.frame(stats_mat),   # same name as ccm_sample
              stats = as.data.frame(stats_mat),
              params = param_draws,
              params_final = params,
              g_final = g,
              network_stats = network_stats,
              prob_distr = prob_distr,
              prior_distr = prior_distr,
              prior_params = lapply(updaters, `[[`, "prior"),
              population = population,
              n_mcmc_trials = n_mcmc_trials,
              burnin_trials = burnin_trials,
              ccm_steps = ccm_steps,
              prob_distr_params = prob_distr_params,   # starting values
              cov_pattern = cov_pattern,
              target_distr = NULL)
  class(obj) <- c("ccm_missing", "ccm_sample")
  obj
}


#' Fixed-dyad graph for node sampling
#'
#' All dyads between two observed nodes are known.
#' @export
make_partial_g <- function(population, obs_nodes) {
  pg <- igraph::make_empty_graph(population, directed = FALSE)
  if (length(obs_nodes) >= 2) {
    pg <- igraph::add_edges(pg, as.vector(utils::combn(sort(obs_nodes), 2)))
  }
  pg
}


# ---- conjugate updates -------------------------------------------------------

# Parameters passed to the C sampler are logged, so keep them strictly positive.
.CCM_POS_FLOOR <- 1e-300

#' Look up and validate the parameter update for one property
#'
#' Returns list(fn, prior, which): fn(params, h, prior) -> params (NULL for
#' "fixed"), the prior parameters recycled to the statistic's length, and the
#' index of the component of params that is updated (for recording draws).
#' @noRd
.ccm_get_updater <- function(network_stat, prob_distr, prior_distr, prior_params, width) {
  
  if (is.function(prior_distr)) {
    return(list(fn = prior_distr, prior = prior_params, which = 1L))
  }
  if (identical(prior_distr, "fixed")) {
    return(list(fn = NULL, prior = NULL, which = NA_integer_))
  }
  
  key <- paste(prob_distr, prior_distr, sep = ":")
  rule <- .CCM_CONJUGATE[[key]]
  if (is.null(rule)) {
    stop(sprintf(paste0(
      "No built-in conjugate prior '%s' for prob_distr '%s'. Built-in pairs: %s. ",
      "Use prior_distr = \"fixed\" or supply an update function."),
      prior_distr, prob_distr, paste(names(.CCM_CONJUGATE), collapse = ", ")))
  }
  if (!is.null(rule$network_stats) && !(network_stat %in% rule$network_stats)) {
    stop(sprintf("prior '%s' with prob_distr '%s' is only valid for network_stats %s.",
                 prior_distr, prob_distr, paste(rule$network_stats, collapse = ", ")))
  }
  if (!is.list(prior_params) || length(prior_params) != length(rule$args)) {
    stop(sprintf("prior_params for '%s' must be a list of %d element(s): %s.",
                 prior_distr, length(rule$args), paste(rule$args, collapse = ", ")))
  }
  names(prior_params) <- rule$args
  prior_params <- rule$check(prior_params, width)
  list(fn = rule$update, prior = prior_params, which = rule$which)
}

#' @noRd
.recycle_pos <- function(x, width, name) {
  if (length(x) == 1) x <- rep(x, width)
  if (length(x) != width || any(!is.finite(x)) || any(x <= 0)) {
    stop(sprintf("prior parameter '%s' must be positive, of length 1 or %d.", name, width))
  }
  x
}

#' @noRd
.recycle_num <- function(x, width, name) {
  if (length(x) == 1) x <- rep(x, width)
  if (length(x) != width || any(!is.finite(x))) {
    stop(sprintf("prior parameter '%s' must be finite, of length 1 or %d.", name, width))
  }
  x
}

#' @noRd
.normal_mean_update <- function(m0, v0, y, v) {
  post_var <- 1 / (1 / v0 + 1 / v)
  stats::rnorm(length(y), post_var * (m0 / v0 + y / v), sqrt(post_var))
}

.CCM_CONJUGATE <- list(
  
  "poisson:gamma" = list(
    args = c("shape", "rate"), which = 1L, network_stats = NULL,
    check = function(p, w) list(shape = .recycle_pos(p$shape, w, "shape"),
                                rate = .recycle_pos(p$rate, w, "rate")),
    update = function(params, h, prior) {
      params[[1]] <- pmax(stats::rgamma(length(h), shape = prior$shape + h, rate = prior$rate + 1),
                          .CCM_POS_FLOOR)
      params
    }),
  
  "poisson:dirichlet" = list(
    args = "alpha", which = 1L, network_stats = "degreedist",
    check = function(p, w) list(alpha = .recycle_pos(p$alpha, w, "alpha")),
    update = function(params, h, prior) {
      theta <- pmax(as.numeric(gtools::rdirichlet(1, prior$alpha + h)), .CCM_POS_FLOOR)
      params[[1]] <- theta / sum(theta)
      params
    }),
  
  "normal:normal" = list(
    args = c("mean", "var"), which = 1L, network_stats = NULL,
    check = function(p, w) list(mean = .recycle_num(p$mean, w, "mean"),
                                var = .recycle_pos(p$var, w, "var")),
    update = function(params, h, prior) {
      params[[1]] <- .normal_mean_update(prior$mean, prior$var, h, params[[2]])
      params
    }),
  
  "lognormal:normal" = list(
    args = c("mean", "var"), which = 1L, network_stats = NULL,
    check = function(p, w) list(mean = .recycle_num(p$mean, w, "mean"),
                                var = .recycle_pos(p$var, w, "var")),
    update = function(params, h, prior) {
      if (any(h <= 0)) stop("lognormal: the statistic must be positive to update meanlog.")
      params[[1]] <- .normal_mean_update(prior$mean, prior$var, log(h), params[[2]])
      params
    }),
  
  "mvn:normal" = list(
    args = c("mean", "cov"), which = 1L, network_stats = NULL,
    check = function(p, w) {
      m <- .recycle_num(p$mean, w, "mean")
      S <- p$cov
      if (length(S) == 1) S <- diag(S, w)
      if (!is.matrix(S) || any(dim(S) != w)) stop(sprintf("prior 'cov' must be a %d x %d matrix.", w, w))
      list(mean = m, cov = S, prec = solve(S))
    },
    update = function(params, h, prior) {
      lik_prec <- solve(params[[2]])
      post_cov <- solve(prior$prec + lik_prec)
      post_mean <- post_cov %*% (prior$prec %*% prior$mean + lik_prec %*% h)
      params[[1]] <- as.numeric(mvtnorm::rmvnorm(1, as.numeric(post_mean), post_cov))
      params
    }),
  
  "gamma:gamma" = list(
    args = c("shape", "rate"), which = 2L, network_stats = NULL,
    check = function(p, w) list(shape = .recycle_pos(p$shape, w, "shape"),
                                rate = .recycle_pos(p$rate, w, "rate")),
    update = function(params, h, prior) {
      params[[2]] <- pmax(stats::rgamma(length(h), shape = prior$shape + params[[1]],
                                        rate = prior$rate + h), .CCM_POS_FLOOR)
      params
    })
)

#' Validate CCM_MissingInference Inputs
#'
#' @noRd

.validate_ccm_missing_inputs <- function(g_obs, partial_g,
                                         network_stats, prob_distr, prob_distr_params,
                                         prior_distr, prior_params,
                                         population, cov_pattern,
                                         n_mcmc_trials, burnin_trials, ccm_steps) {
  
  # 1. Likelihood inputs: same checks as sample_ccm, with the arguments the
  #    Gibbs loop will pass (sample_size = 2, interval = 1, burnin = ccm_steps)
  .validate_sample_ccm_inputs(network_stats, prob_distr, prob_distr_params,
                              population, sample_size = 2, burnin = ccm_steps,
                              interval = 1, cov_pattern = cov_pattern,
                              initial_g = g_obs, use_initial_g = TRUE)
  
  # 2. Iteration counts
  if (length(n_mcmc_trials) != 1 || !is.numeric(n_mcmc_trials) ||
      n_mcmc_trials %% 1 != 0 || n_mcmc_trials < 1) {
    stop("`n_mcmc_trials` must be a single integer >= 1")
  }
  if (length(burnin_trials) != 1 || !is.numeric(burnin_trials) ||
      burnin_trials %% 1 != 0 || burnin_trials < 0) {
    stop("`burnin_trials` must be a single integer >= 0")
  }
  
  # 3. Graphs: undirected, simple, on the full population
  for (nm in c("g_obs", "partial_g")) {
    g <- get(nm)
    if (is.null(g) || !inherits(g, "igraph")) stop(sprintf("`%s` must be an igraph object.", nm))
    if (igraph::is_directed(g)) stop(sprintf("`%s` must be undirected.", nm))
    if (!igraph::is_simple(g)) stop(sprintf("`%s` must be simple (no loops or multiple edges).", nm))
    if (igraph::vcount(g) != population) {
      stop(sprintf("`%s` vertex count (%d) must match population (%d).",
                   nm, igraph::vcount(g), population))
    }
  }
  if (igraph::gsize(partial_g) == choose(population, 2)) {
    stop("All dyads are fixed in `partial_g`; there is nothing to impute.")
  }
  
  # 4. Priors: one per property
  if (!is.list(prior_distr)) prior_distr <- as.list(prior_distr)
  if (length(prior_distr) != length(network_stats)) {
    stop(paste0("Mismatched input: 'network_stats' has length ", length(network_stats),
                ", but 'prior_distr' has length ", length(prior_distr), "."))
  }
  if (!is.null(prior_params) &&
      (!is.list(prior_params) || length(prior_params) != length(network_stats))) {
    stop("`prior_params` must be a list with one element per element of `network_stats`.")
  }
  for (i in seq_along(prior_distr)) {
    p <- prior_distr[[i]]
    if (!is.function(p) && !(is.character(p) && length(p) == 1)) {
      stop(sprintf("`prior_distr[[%d]]` must be a prior name or an update function.", i))
    }
  }
  
  return(invisible(TRUE))
}
