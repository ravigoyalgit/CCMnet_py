#' Probability Distribution Config
#'
#' @importFrom stats rlnorm
#' @noRd

.get_distr_settings <- function(distr_name) {
  # Define the core settings for each distribution
  configs <- list(
    "mvn"     = list(sub_code = 0, 
                     mean_bool = TRUE, 
                     var_bool = TRUE, 
                     use_solve_var = TRUE,
                     rules = list(p1 = c("is_numeric"), 
                                  p2 = c("is_numeric", "is_square_mat", "match_matrix_dim")),
                     sampler = function(p, n, ...) {
                       # p[[1]] = mean_vector, p[[2]] = sigma_matrix
                       # Returning as a matrix (n_sim x length(mean_vector))
                       mvtnorm::rmvnorm(n, mean = p[[1]], sigma = p[[2]])
                     },
                     valid_network_prop = c("degreedist", "degmixing", "mixing")
    ),
    "normal"     = list(sub_code = 1, 
                        mean_bool = TRUE, 
                        var_bool = TRUE, 
                        use_solve_var = FALSE,
                        rules = list(p1 = c("is_numeric"), 
                                     p2 = c("is_numeric", "all_positive", "match_length")),
                        sampler = function(p, n, ...) {
                          mu <- p[[1]]; sigma <- p[[2]]
                          matrix(rnorm(n * length(mu), mean = mu, sd = sqrt(sigma)), nrow = n, byrow = TRUE)
                        },
                        valid_network_prop = c("edges", "density", "degreedist", "degmixing", "triangles", "mixing")
    ),
    "lognormal" = list(sub_code = 2, 
                       mean_bool = TRUE, 
                       var_bool = TRUE, 
                       use_solve_var = FALSE,
                       rules = list(p1 = c("is_numeric"), 
                                    p2 = c("is_numeric", "all_positive", "match_length")),
                       sampler = function(p, n, ...) {
                         # p[[1]] is meanlog (mu), p[[2]] is varlog (sigma^2)
                         mu <- p[[1]]
                         sigma <- p[[2]]
                         matrix(
                           rlnorm(n * length(mu), meanlog = mu, sdlog = sqrt(sigma)), 
                           nrow = n, 
                           byrow = TRUE
                         )
                       },
                       valid_network_prop = c("edges", "density", "mixing")
    ),
    "poisson" = list(sub_code = 3, 
                     mean_bool = TRUE, 
                     var_bool = FALSE, 
                     use_solve_var = FALSE,
                     rules = list(p1 = c("is_numeric", "all_positive")),
                     sampler = function(p, n, ...) {
                       lambda <- p[[1]]
                       matrix(rpois(n * length(lambda), lambda), nrow = n, byrow = TRUE)
                     },
                     valid_network_prop = c("edges", "degreedist", "degmixing", "triangles", "mixing")
    ),
    "uniform" = list(sub_code = 4, 
                     mean_bool = FALSE, 
                     var_bool = FALSE, 
                     use_solve_var = FALSE,
                     rules = list(),
                     sampler = function(p, n, max_val, ...) {
                       matrix(sample(0:max_val, size = n, replace = TRUE), ncol = 1)
                     },
                     valid_network_prop = c("edges")
    ),
    "beta"     = list(sub_code = 5, 
                      mean_bool = TRUE, 
                      var_bool = TRUE, 
                      use_solve_var = FALSE,
                      rules = list(p1 = c("is_numeric", "all_positive"), 
                                   p2 = c("is_numeric", "all_positive", "match_length")),
                      sampler = function(p, n, ...) {
                        # p[[1]] is shape1 (alpha), p[[2]] is shape2 (beta)
                        # Returning as a matrix with 1 column for consistency
                        matrix(rbeta(n, shape1 = p[[1]], shape2 = p[[2]]), ncol = 1)
                      },
                      valid_network_prop = c("density")
    ),
    "dirmult" = list(sub_code = 6, 
                     mean_bool = TRUE, 
                     var_bool = FALSE, 
                     use_solve_var = FALSE,
                     rules = list(p1 = c("is_numeric", "all_positive")),
                     sampler = function(p, n, population, ...) {
                       alpha <- p[[1]]
                       # 1. Generate n samples from Dirichlet
                       dir_draws <- gtools::rdirichlet(n, alpha = alpha)
                       
                       # 2. Generate Multinomial draws for each Dirichlet sample
                       # Using a vectorized apply or a row-wise rmultinom
                       t(apply(dir_draws, 1, function(probs) rmultinom(1, size = population, prob = probs)))
                     },
                     valid_network_prop = c("degreedist")
    ),
    "gamma" = list(sub_code = 7, 
                   mean_bool = TRUE, 
                   var_bool = TRUE, 
                   use_solve_var = FALSE,
                   rules = list(p1 = c("is_numeric", "all_positive"), 
                                p2 = c("is_numeric", "all_positive", "match_length")),
                   sampler = function(p, n, population, ...) {
                     alpha <- p[[1]] # Shape
                     beta  <- p[[2]] # Rate (matches your C code beta * v_old)
                     
                     # Using 'rate' instead of 'scale' to match Case 7
                     matrix(rgamma(n * length(alpha), shape = alpha, rate = beta), 
                            nrow = n, byrow = TRUE)
                   },
                   valid_network_prop = c("edges", "density", "degreedist", "degmixing", "triangles", "mixing")
    ),
    "np" = list(sub_code = 99, 
                mean_bool = TRUE, 
                var_bool = FALSE, 
                use_solve_var = FALSE,
                rules = list(p1 = c("is_numeric", "non_negative", "sums_to_one")),
                sampler = function(p, n, max_val, ...) {
                  matrix(sample(0:max_val, size = n, replace = TRUE, prob = p[[1]]), ncol = 1)
                },
                valid_network_prop = c("edges")
    )
    
  )
  
  if (!(distr_name %in% names(configs))) {
    stop(paste("Unsupported distribution:", distr_name))
  }
  
  return(configs[[distr_name]])
}

.VAL_RULES <- list(
  is_numeric    = function(x, name) if (!is.numeric(x)) stop(sprintf("%s must be numeric.", name)),
  all_positive  = function(x, name) if (any(x <= 0 | !is.finite(x))) stop(sprintf("All %s values must be positive.", name)),
  non_negative  = function(x, name) if (any(x < 0 | !is.finite(x))) stop(sprintf("All %s values must be non-negative.", name)),
  sums_to_one   = function(x, name) if (abs(sum(x) - 1) > 1e-8) stop(sprintf("%s must sum to 1.", name)),
  is_square_mat = function(x, name) if (!is.matrix(x) || nrow(x) != ncol(x)) stop(sprintf("%s must be a square matrix.", name)),
  
  #Checks if length of p2 matches p1
  match_length = function(x, name, target_val) {
    if (length(x) != length(target_val)) stop(sprintf("Length of %s must match parameter 1.", name))
  },
  
  #Checks if MVN matrix rows match mean vector length
  match_matrix_dim = function(x, name, target_val) {
    if (nrow(x) != length(target_val)) stop(sprintf("Dimensions of %s must match length of mean vector.", name))
  }
)

.VAL_NETPROPS <- c("edges", "density", "degreedist", "degmixing", "triangles", "mixing")

.VAL_NETPROPS_COMB <- c("edges", "density", "degreedist", "degmixing", 
                        "degmixing_triangles", "mixing", "degreedist_degreedist_mixing")