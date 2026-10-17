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
    int     termcode;      /* raxopt: 1 gradient, 2 step, 3-5 no convergence  */
    int     nit;           /* raxopt iterations                               */
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

/* drvarma_marma -- Shea's exact likelihood (AS 242) at a GIVEN structure,    */
/* the independent benchmark beside elf. Same arrays as drvarma_elf. Always  */
/* EXACT (no steady-state shortcut); the MA frontier is elf's (chekma: a      */
/* point elf would refuse returns ifault 4). No residuals: marma's are the   */
/* one-step innovations, and the forecasts need elf's exact residuals.       */
int drvarma_marma(int m, int n, int p, int q,
                  const double *mu, const double *phi, const double *theta,
                  const double *qq, const double *w, double sigma2,
                  double *f1, double *f2, double *logelf);

/* ------------------------------------------------------------------------- */
/* GSL's eigensolvers, exported for callers that must reproduce a C program   */
/* that calls them: drvec's Johansen start (canonical_b2) and its reports of  */
/* eigenvalues. The engine links GSL on every wheel platform, so the caller   */
/* gets GSL's own routine without looking for a system libgsl.               */
/*                                                                           */
/* Arrays are FLAT, 0-based, row-major. GSL's error handler is switched off  */
/* for the call and restored after it: a failure is the return code, never  */
/* abort().                                                                  */
/* ------------------------------------------------------------------------- */

/* drvarma_gensymmv -- A v = lambda B v, A symmetric, B symmetric positive    */
/* definite (gsl_eigen_gensymmv), then gsl_eigen_gensymmv_sort with `sort`   */
/* (a gsl_eigen_sort_t: 0 VAL_ASC, 1 VAL_DESC, 2 ABS_ASC, 3 ABS_DESC; -1 does */
/* not sort). A and B are n*n and are not modified; eval [n]; evec [n*n],    */
/* evec[i*n + j] = component i of eigenvector j (the vectors are COLUMNS, as */
/* in GSL). Returns GSL's status (0 = ok; GSL_EDOM when B is not positive    */
/* definite), and then eval/evec are undefined.                              */
int drvarma_gensymmv(int n, const double *A, const double *B,
                     double *eval, double *evec, int sort);

/* drvarma_eigen_nonsymm -- the eigenvalues of a real general n*n matrix A    */
/* (gsl_eigen_nonsymm, Schur form not computed, balanced when `balance`),    */
/* in GSL's order: wr[k] + i wi[k]. A is not modified. Returns GSL's status   */
/* (0 = ok; GSL_EMAXITER when the QR iteration did not converge, and then    */
/* only some eigenvalues are valid, as GSL documents).                       */
int drvarma_eigen_nonsymm(int n, const double *A, int balance,
                          double *wr, double *wi);

#endif
