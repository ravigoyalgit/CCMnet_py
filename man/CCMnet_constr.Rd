\name{CCMnet_constr}
\alias{CCMnet_constr}
\title{MCMC Construction of Networks Under Constrained Network Statistics}
\description{
\code{CCMnet_constr} generates networks through a Metropolis–Hastings MCMC sampler
that targets specific network statistics such as edge count, mixing patterns,
or degree distribution. This function is an R wrapper for the Python
implementation \code{CCMnet_constr_py}, which performs the underlying MCMC updates.
}
\usage{
CCMnet_constr(Network_stats,
              Prob_Distr,
              Prob_Distr_Params,
              samplesize,
              burnin,
              interval,
              statsonly,
              G,
              P,
              population,
              covPattern,
              bayesian_inference,
              Ia,
              Il,
              R,
              epi_params,
              print_calculations,
              use_G = FALSE,
              outfile = "none",
              partial_network = 0,
              obs_nodes,
              MH_proposal_type = "random")
}
\arguments{
 \item{Network_stats}{Character vector indicating which network statistics are used to
   evaluate MCMC proposals. Supported values in the Python implementation include
   \code{"Edge"}, \code{"Mixing"}, and \code{"Degree"}.}
 \item{Prob_Distr}{Character vector specifying the probability model associated with the
   network statistic. Examples include \code{"NP"} (nonparametric), 
   \code{"Multinomial_Poisson"}, or others defined by the user.}
 \item{Prob_Distr_Params}{List of parameters for \code{Prob_Distr}. The required structure
   depends on the type of statistic being modeled (e.g., lookup tables for NP,
   degree probabilities for degree models).}
 \item{samplesize}{Number of MCMC samples to retain after burn-in and thinning.}
 \item{burnin}{Number of initial MCMC iterations to discard.}
 \item{interval}{Thinning interval; number of iterations between saved samples.}
 \item{statsonly}{Logical; if TRUE, only network statistics are saved; if FALSE,
   full edge lists may be returned.}
 \item{G}{Initial graph or edge list used to initialize the sampler when
   \code{use_G = TRUE}.}
 \item{P}{Observed partial network used in Bayesian inference mode (optional).}
 \item{population}{Total number of nodes in the network. Nodes are assumed to be
   1-indexed.}
 \item{covPattern}{Integer vector of length \code{population} denoting covariate groups
   used for mixing matrices.}
 \item{bayesian_inference}{Logical or integer; when TRUE, includes additional
   likelihood terms involving \code{Ia}, \code{Il}, \code{R} and \code{epi_params}.}
 \item{Ia}{Vector of infection (or analogous) times for each node.}
 \item{Il}{Vector of latent (or analogous) times for each node.}
 \item{R}{Vector of recovery/removal times.}
 \item{epi_params}{Numeric vector giving epidemiological parameters (e.g., \code{beta_a},
   \code{beta_l}). Required only when \code{bayesian_inference = TRUE}.}
 \item{print_calculations}{Logical; if TRUE, prints diagnostic information
   while running the Python implementation.}
 \item{use_G}{Logical; if TRUE, \code{G} is used as the initial graph. Otherwise an empty
   graph is created.}
 \item{outfile}{Character string passed to the Python code; \code{"none"} and
   \code{"favites"} are supported.}
 \item{partial_network}{Integer indicator (0/1) specifying whether a partial network is
   in use (affects proposal rejection rules).}
 \item{obs_nodes}{Vector of observed node IDs used when \code{partial_network = 1}.}
 \item{MH_proposal_type}{Type of MCMC proposal: \code{"random"} (uniform dyad toggle)
   or \code{"TNT"} (toggle–near–toggle).}
}
\details{
This function calls a Python implementation using \pkg{reticulate}. The Python code
constructs a \pkg{networkx} graph, computes statistics, performs MH updates, and returns
the resulting network and sampled statistics. The R wrapper converts the Python edge list
to an \pkg{igraph} object.

Supported statistics include:
\itemize{
  \item \strong{Edge count}: number of edges in the network.
  \item \strong{Mixing matrix}: edge counts across covariate groups.
  \item \strong{Degree distribution}: degree counts for each node.
}

The function supports simple Bayesian inference for epidemic models by including
likelihood terms based on node-specific event times (\code{Ia}, \code{Il}, \code{R}).
}
\value{
A list with two elements:
\describe{
  \item{\code{[[1]]}}{An \pkg{igraph} graph object built from the Python edge list.}
  \item{\code{[[2]]}}{A matrix or data frame of sampled network statistics, with one
    row per saved MCMC sample.}
}
\section{Python Requirements}{
This function requires Python (\code{>=3.10}) with the following packages:
\itemize{
  \item \pkg{networkx}
  \item \pkg{numpy}
  \item \pkg{pandas}
}
For convenience, the package provides \code{CCMnet_python_setup()} to automatically
create a working conda environment.
}
\examples{
## Not run:
## --- Python environment setup (first time only) ---
# library(CCMnetpy)
# CCMnet_python_setup()

## Load Python implementation if needed:
# reticulate::source_python(
#   file.path(find.package("CCMnetpy"), "python", "CCMnet_constr_py.py"),
#   envir = globalenv()
# )

## --- Demonstration example ---
population <- 100
covPattern <- rep(0L, population)

## Target statistic: edge count
Network_stats <- list("Edge")
Prob_Distr    <- list("NP")

## Negative binomial prior on number of edges (example)
Prob_Distr_Params <- vector("list", 2)
Prob_Distr_Params[[1]] <- dnbinom(
    0:choose(population, 2),
    size = 1.017340,
    mu   = 6.192894
)

## Run a small MCMC chain (for demonstration)
result <- CCMnet_constr(
    Network_stats      = Network_stats,
    Prob_Distr         = Prob_Distr,
    Prob_Distr_Params  = Prob_Distr_Params,
    samplesize         = 100L,
    burnin             = 5000L,
    interval           = 100L,
    statsonly          = TRUE,
    G                  = NULL,
    P                  = NULL,
    population         = population,
    covPattern         = covPattern,
    bayesian_inference = FALSE,
    Ia                 = NULL,
    Il                 = NULL,
    R                  = NULL,
    epi_params         = NULL,
    print_calculations = FALSE,
    use_G              = FALSE,
    outfile            = "none",
    partial_network    = 0L,
    obs_nodes          = NULL,
    MH_proposal_type   = "random"
)

## Extract results
g     <- result[[1]]
stats <- result[[2]]

print(g)
plot(stats[,1], type = "l", main = "Sampled Edge Count")

## Compare empirical mean to prior expectation:
mean(stats[,1])
mean(rnbinom(10000, size = 1.017340, mu = 6.192894))

## End(Not run)
}
\author{CCMnet developers; documentation adapted by ChatGPT}
\keyword{network}
\keyword{MCMC}

