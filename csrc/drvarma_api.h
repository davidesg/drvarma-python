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
    int     p, q;
    int     nresiduals;
    double *params;
    double *std_errors;
    double *cov_matrix;
    double *residuals;     /* nobs*m (row-major)                              */
    double *mu;            /* m                                               */
    double *phi;           /* p*m*m (lag 1..p, row-major per lag)             */
    double *theta;         /* q*m*m                                           */
    double *sigma;         /* m*m  (= sigma2 * Q)                             */
    double  sigma2;
    double  logelf;
} DrvarmaResult;

void           drvarma_defaults(DrvarmaModelSpec *spec);
DrvarmaResult *drvarma_estimate(const DrvarmaModelSpec *spec);
void           drvarma_result_free(DrvarmaResult *r);
const char    *drvarma_strerror(int ifault);

/* ------------------------------------------------------------------------- */
/* drvarma_elf -- the exact likelihood, evaluated at a GIVEN structure.       */
/*                                                                           */
/* drvarma_estimate fits a FREE VARMA(p,q) from the data.  This is the other  */
/* direction: score a Phi/Theta/Sigma that the caller built.  It is what a    */
/* restricted model needs -- a transfer function, a network, anything whose   */
/* structure comes from a cast rather than from (p,q) -- and it could not be  */
/* asked for through the estimate entry point.                                */
/*                                                                           */
/* Arrays are FLAT and 0-based, row-major; this routine does the 1-based      */
/* marshalling that `elf` expects:                                            */
/*                                                                           */
/*   mu     [m]           means                                              */
/*   phi    [p*m*m]       phi[k*m*m + i*m + j] = Phi_(k+1)[i][j]              */
/*   theta  [q*m*m]       likewise                                           */
/*   qq     [m*m]         innovation covariance (or its ratios; see sigma2)   */
/*   w      [n*m]         the stationary series, w[t*m + i]                   */
/*   a_out  [n*m]         residuals, filled only when atf != 0 (may be NULL)  */
/*                                                                           */
/* Returns ifault (0 = ok), and writes f1, f2 and logelf.  Passing sigma2=1   */
/* with qq normalised gives the CONCENTRATED likelihood, which is what the    */
/* estimators use.                                                           */
/* ------------------------------------------------------------------------- */
int drvarma_elf(int m, int n, int p, int q,
                const double *mu, const double *phi, const double *theta,
                const double *qq, const double *w,
                double sigma2, double delta, int atf,
                double *a_out, double *f1, double *f2, double *logelf);

#endif
