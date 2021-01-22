
CCMnet_python_setup <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE) & (reticulate::py_config()$available)) {
    stop("Package \"reticulate\" and Python are needed for this function to work. Please install it.",
         call. = FALSE)
  }
  reticulate::source_python(paste(find.package("CCMnetpy"), "/python/CCMnet.py", sep = ""), envir = globalenv())
}

