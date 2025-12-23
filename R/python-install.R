#' Install Python dependencies for CCMnet
#'
#' @description
#' Installs Python and required Python packages into a persistent
#' virtual environment for optional CCMnet acceleration.
#'
#' Python is not required for core CCMnet functionality. This function
#' is only needed to enable Python-backed graph operations.
#'
#' @param method Installation method. One of "auto", "virtualenv", or "conda".
#' @param envname Name of the Python environment to create.
#' @param python_version Python version to use when creating the environment.
#' @param extra_packages Optional additional Python packages to install.
#' @param new_env Logical; if TRUE, remove any existing environment first.
#' @param restart_session Logical; restart R session after installation (RStudio only).
#' @param ... Additional arguments passed to reticulate::py_install().
#'
#' @export

install_ccmnet <- function(
    method = c("auto", "virtualenv", "conda"),
    envname = "r-ccmnet",
    python_version = NULL,
    extra_packages = NULL,
    new_env = identical(envname, "r-ccmnet"),
    restart_session = TRUE,
    ...
) {
  method <- match.arg(method)
  
  if (reticulate::py_available(initialize = FALSE)) {
    stop(
      "install_ccmnet() must be called before Python is initialized.\n",
      "Restart R and run install_ccmnet() first."
    )
  }
  
  if (new_env) {
    if (method %in% c("auto", "virtualenv") &&
        reticulate::virtualenv_exists(envname)) {
      reticulate::virtualenv_remove(envname, confirm = FALSE)
    }
    
    if (method %in% c("auto", "conda") &&
        envname %in% reticulate::conda_list(conda = "auto")$name) {
      reticulate::conda_remove(envname, conda = "auto")
    }
  }
  
  packages <- c(
    "numpy>=1.24",
    "pandas>=2.0",
    "networkx>=3.1",
    "scipy>=1.11",
    extra_packages
  )
  
  reticulate::py_install(
    packages       = packages,
    envname        = envname,
    method         = method,
    python_version = python_version,
    pip            = TRUE,
    ...
  )
  
  message("\nCCMnet Python environment '", envname,
          "' installation complete with packages: ",
          paste(packages, collapse = ", "), ".\n")
  
  if (restart_session &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::hasFun("restartSession")) {
    rstudioapi::restartSession()
  }
  
  invisible(NULL)
}
