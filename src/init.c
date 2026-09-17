#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Declare ONLY the main entry point(s) called by R
extern void MCMC_wrapper(
    int*, int*, int*, int*, int*, int*, int*, int*,       // 1-8
    char**, char**, char**, char**,                       // 9-12
    double*, double*,                                     // 13-14
    int*,                                                 // 15
    double*,                                              // 16
    int*, int*, int*, int*, int*,                         // 17-21
    int*, int*, int*, int*, int*, int*, int*, int*, int*, // 22-30
    int*, int*,                                           // 31-32
    double*, double*,                                     // 33-34
    int*, int*, int*, int*,                               // 35-38
    double*, double*, double*, double*, double*,          // 39-43
    int*,                                                 // 44
    double*, double*                                      // 45-46
);
  
static const R_CMethodDef CEntries[] = {
  {"MCMC_wrapper", (DL_FUNC) &MCMC_wrapper, 46},
  {NULL, NULL, 0}
};

void R_init_CCMnet(DllInfo *dll)
{
  // Register the main wrapper
  R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
  
  // Set this to TRUE so MCMC_wrapper can find d_edges, d_mixing, etc.
  // even if they aren't explicitly listed in CEntries.
  R_useDynamicSymbols(dll, TRUE); 
}
