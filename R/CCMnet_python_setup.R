#' Setup CCMnet Python environment
#'
#' This function configures the Python environment for CCMnet.
#'
#' @export

CCMnet_python_setup <- function() {
  python_file <- system.file("python", "CCMnet_constr_py.py", package = "CCMnetpy")
  
  if (python_file == "") {
    stop("Unable to locate CCMnet_constr_py.py inside the installed package.")
  }
  
  reticulate::source_python(python_file, envir = globalenv())
  message("Python CCMnet module loaded successfully.")
}