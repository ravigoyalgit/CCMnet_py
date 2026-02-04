#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Declare ONLY the main entry point(s) called by R
extern void MCMC_wrapper(int*, int*, int*, int*, int*, int*, int*, int*, char**, 
                         /* ... all the other args ... */ char**);
                         
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
