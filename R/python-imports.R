py_networkx <- function() {
  reticulate::import("networkx", delay_load = TRUE)
}

py_pandas <- function() {
  reticulate::import("pandas", delay_load = TRUE)
}

py_numpy <- function() {
  reticulate::import("numpy", delay_load = TRUE)
}