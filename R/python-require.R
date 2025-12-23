#' Declare CCMnet Python requirements
#'
#' @export
py_require_ccmnet <- function() {
  reticulate::py_require(c(
    "numpy>=1.24",
    "scipy",
    "networkx"
  ))
}
