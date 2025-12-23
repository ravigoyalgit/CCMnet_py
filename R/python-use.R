#' Activate CCMnet Python environment
#'
#' @export
use_ccmnet <- function(envname = "r-ccmnet") {
  
  if (!reticulate::py_available(initialize = FALSE)) {
    
    if (envname %in% reticulate::conda_list()$name ||
        reticulate::virtualenv_exists(envname)) {
      
      reticulate::use_python(
        reticulate::py_exe(envname),
        required = FALSE
      )
      return(TRUE)
      
    } else {
      message(
        "Python environment '", envname, "' not found.\n",
        "Run install_ccmnet() to enable Python acceleration."
      )
      return(FALSE)
    }
  }
  
  TRUE
}
