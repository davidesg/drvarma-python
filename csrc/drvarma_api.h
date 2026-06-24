/*****************************************************************************/
/*  drvarma_api.h -- C library API for the drvarma estimator (CFFI bridge).  */
/*  GPL v2 or later (see COPYING).                                            */
/*****************************************************************************/
#ifndef DRVARMA_API_H
#define DRVARMA_API_H

typedef struct {
    int     m;
    int     nobs;
    double *w;            /* nobs*m, row-major: w[t*m + j]                    */
    int     p, q;
    int     include_mean;
    int     diag_ar, diag_ma, diag_cov;
    int     method;       /* 1 = exact, 2 = approximate                      */
    int     twostep;
    int     maxits;       /* 0 -> default 500                                */
    double  grtol, sptol; /* 0 -> defaults                                   */
} DrvarmaModelSpec;

typedef struct {
    int     ifault;
    int     npar;
    int     m;
    int     nresiduals;
    double *params;
    double *std_errors;
    double *cov_matrix;
    double *residuals;
    double  sigma2;
    double  logelf;
} DrvarmaResult;

void           drvarma_defaults(DrvarmaModelSpec *spec);
DrvarmaResult *drvarma_estimate(const DrvarmaModelSpec *spec);
void           drvarma_result_free(DrvarmaResult *r);
const char    *drvarma_strerror(int ifault);

#endif
