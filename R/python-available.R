ccmnet_python_available <- function() {
  reticulate::py_available(initialize = FALSE)
}
