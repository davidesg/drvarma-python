/*****************************************************************************/
/*  drvmlest.c -- part of drvarma (multivariate VARMA modelling).
 *
 *  Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.  Distributed WITHOUT ANY WARRANTY; see the
 *  GNU General Public License (file COPYING) for details.
 *****************************************************************************/

/*****************************************************************************/
/*  DRVMLEST.C                                                               */
/*  Driver module for exact maximum likelihood estimation.                   */
/*  Copyright (C) Jos‚ Alberto Mauricio, 1995.                               */
/*****************************************************************************/

#include "main.h"
#include <math.h>              /* Header file (prototype declarations)        */
extern real macheps;          /* Machine epsilon (global: declared in DRV.C) */
extern FILE *outputv;         /* Output file (global: declared in DRV.C)     */

/*****************************************************************************/

real pi10x, pi20x, xitolx;
int  est_ma_boundary = 0;     /* MA inverse roots on the wall at the stop (lik.h) */
int  est_ma_nroots   = 0;
int  est_fdhess = 0;          /* 1: standard errors from fdhess at the optimum */
int  est_fixed  = 0;          /* >0: parameter held while fdhess runs (flat dir.) */
int  est_se_how = EST_SE_BFGS;/* <- which Hessian gave the standard errors      */
extern int   opt_iters;       /* qnewtopt.c: iterations of the last raxopt  */
static int   objc_rejects;    /* points objcfunc refused (boundary sentinel)     */
static real *fix_full;        /* full vector behind the reduced objective        */
static int   fix_n;
struct Tvarma varmax;
void (*castx)( real *, struct Tvarma *,int *, int, int );

/*****************************************************************************/

void fdhess( real (*func)(real *), int n, real *x, real f, real eta, real **H );
real objcfunc( real *x );

/* The Hessian behind the standard errors, as the reports write it.         */
const char *est_se_label( int how )
{
   switch ( how )
      {
      case EST_SE_FDHESS:   return "fdhess";
      case EST_SE_BOUNDARY: return "bfgs (fdhess: the optimum is on the boundary "
                                   "of the admissible region)";
      case EST_SE_NOTPD:    return "bfgs (fdhess: the Hessian is not positive definite)";
      case EST_SE_NONE_BOUNDARY:
         return "none (fdhess: the optimum is on the boundary of the admissible "
                "region; the search did not move, so it built no BFGS Hessian)";
      case EST_SE_NONE_NOTPD:
         return "none (fdhess: the Hessian is not positive definite; the search "
                "did not move, so it built no BFGS Hessian)";
      default:              return "bfgs";
      }
}

/* The objective with parameter est_fixed held at fix_full[est_fixed].       */
static real objcfix( real *z )
{
   int i, k = 1;
   for ( i = 1; i <= fix_n; i++ )
       if ( i != est_fixed ) fix_full[i] = z[k++];
   return objcfunc( fix_full );
}

/* 1 if H (k x k, symmetric) is positive definite by a plain Cholesky.       */
static int strict_pd( real **H, int k )
{
   int i, j, l, ok = 1;
   real s, **L = matrix( 1, k, 1, k );
   for ( j = 1; j <= k && ok; j++ )
       {
       s = H[j][j];
       for ( l = 1; l < j; l++ ) s -= L[j][l] * L[j][l];
       if ( !(s > 0.0) || !isfinite( s ) ) { ok = 0; break; }
       L[j][j] = sqrt( s );
       for ( i = j + 1; i <= k; i++ )
           {
           s = H[i][j];
           for ( l = 1; l < j; l++ ) s -= L[i][l] * L[j][l];
           L[i][j] = s / L[j][j];
           }
       }
   free_matrix( L, 1, k, 1, k );
   return ok;
}

/* Covariance 2 F H^-1 / n from fdhess at par. Sets est_se_how; fills cov    */
/* and dev only when it is EST_SE_FDHESS (otherwise the caller uses BFGS).   */
static void fdhess_cov( int npar, real *par, real f, real **cov, real *dev )
{
   int  i, j, ii, jj, k = npar - ( est_fixed > 0 ), pfault;
   real d1, d2, *z, *v, **H;

   z = vector( 1, k );  v = vector( 1, k );  H = matrix( 1, k, 1, k );
   for ( i = 1, ii = 1; i <= npar; i++ ) if ( i != est_fixed ) z[ii++] = par[i];
   fix_full = vector( 1, npar );  fix_n = npar;
   for ( i = 1; i <= npar; i++ ) fix_full[i] = par[i];

   objc_rejects = 0;
   fdhess( est_fixed > 0 ? objcfix : objcfunc, k, z, f, macheps, H );

   if ( objc_rejects > 0 )
      est_se_how = EST_SE_BOUNDARY;
   else if ( !strict_pd( H, k ) )
      est_se_how = EST_SE_NOTPD;
   else
      {
      pfault = 0;
      choldcp( H, k, &d1, &d2, &pfault );
      if ( pfault ) est_se_how = EST_SE_NOTPD;
      else
         {
         est_se_how = EST_SE_FDHESS;
         for ( i = 1; i <= npar; i++ )
             { dev[i] = NAN; for ( j = 1; j <= npar; j++ ) cov[i][j] = NAN; }
         for ( i = 1, ii = 1; i <= npar; i++ )
             {
             if ( i == est_fixed ) continue;
             for ( j = 1; j <= k; j++ ) v[j] = 0.0;
             v[ii] = 1.0;
             cholsol( H, k, v );
             for ( j = 1, jj = 1; j <= npar; j++ )
                 {
                 if ( j == est_fixed ) continue;
                 cov[j][i] = ( 2.0 * f * v[jj] ) / varmax.n;
                 jj++;
                 }
             dev[i] = sqrt( cov[i][i] );
             ii++;
             }
         }
      }
   /* raxopt starts b at the identity: with no iteration there is no BFGS    */
   /* Hessian to fall back on (a .pre that already is the optimum), and the  */
   /* identity would give sqrt(2F/n) for every parameter. Say so instead.    */
   if ( ( est_se_how == EST_SE_BOUNDARY || est_se_how == EST_SE_NOTPD ) &&
        opt_iters == 0 )
      {
      est_se_how = ( est_se_how == EST_SE_BOUNDARY ) ? EST_SE_NONE_BOUNDARY
                                                     : EST_SE_NONE_NOTPD;
      for ( i = 1; i <= npar; i++ )
          { dev[i] = NAN; for ( j = 1; j <= npar; j++ ) cov[i][j] = NAN; }
      }
   free_vector( fix_full, 1, npar );
   free_matrix( H, 1, k, 1, k );  free_vector( v, 1, k );  free_vector( z, 1, k );
}

/*****************************************************************************/
/*****************************************************************************/

void est( void (*cast)( real *, struct Tvarma *, int *, int, int ),
          int npar, real *par, real *dev, real **cov, int maxits, int nrits,
          real grtol, real sptol, real xitol, real **a, real *sigma2,
          real *logelf, int *ifault )

/*****************************************************************************/
/*                                                                           */
/* cast ->   : casting routine [drv.c].                                      */
/* npar ->   : n§ of parameters to estimate.                                 */
/* par <->   : -> initial estimate (guess); <- final estimate (solution).    */
/* dev <-    : estimated standard deviations at final estimate.              */
/* cov <-    : estimated covariance matrix at final estimate.                */
/* maxits -> : maximum number of iterations.                                 */
/* nrits ->  : report optimizer progress every [nrits] iterations.           */
/* grtol ->  : stopping tolerance 1: relative gradient value.                */
/* sptol ->  : stopping tolerance 2: relative parameter change.              */
/* xitol ->  : stopping tolerance for computing recursive xi [elfvarma.c].   */
/* a <-      : residual vector at final estimate.                            */
/* sigma2 <- : final estimate of "concentrated" residual variance.           */
/* logelf <- : exact log-likelihood at final estimate.                       */
/* ifault <- : [0]: successful return;                                       */
/*             [1]: bad initial estimate: Q is not positive definite;        */
/*             [2]: bad initial estimate: AR has at least one unit root;     */
/*             [3]: bad initial estimate: AR is strictly non-stationary;     */
/*             [4]: bad initial estimate: MA is strictly non-invertible;     */
/*             [5]: bad initial estimate: unknown numerical problems (rare); */
/*             [6]: bad initial estimate: see cast().                        */
/*                                                                           */
/*****************************************************************************/

{
   const real LOG2PI = 1.837877066;

/* Declaration of local variables:                                           */

   int  i, j;
   real pi1, pi2, pi3, **mtmp, *vtmp;

/* Declaration of local functions:                                           */

   void raxopt( real (*func)( real [] ), real *fk, int n, real *xk, real **b,
                int maxits, int nrits, real gradtol, real steptol );
   real objcfunc( real *x );


/* [1]: First computation of the log-likelihood: initialize pi10x & pi20x:   */

   *ifault = 0;                               /* Initialize fault indicator. */
   est_se_how = EST_SE_BFGS;                  /* until the Hessian is taken  */

   varmax.xitol = xitol;                      /* Estimation method.          */

   (*cast)( par, &varmax, ifault, 1, 0 );     /* Allocate VARMA structure.   */

   if ( *ifault > 0 ) return;                 /* Bad initial estimates (1).  */

   varma_lik( varmax.m, varmax.n, varmax.p, varmax.q, varmax.mu, varmax.phi,
        varmax.theta, varmax.qq, varmax.w, 1.0, varmax.xitol,
        TRUE, varmax.a, &pi10x, &pi20x, &pi3, ifault );

   if ( *ifault > 0 ) return;                 /* Bad initial estimates (2):  */
                                              /* ifault = 1-2-3-4-5.         */

/* [2]: Maximization of the specified objective function:                    */

   mtmp  = matrix( 1, npar, 1, npar );
   vtmp  = vector( 1, npar );
   castx = cast;                              /* Assign casting routine.     */

   opt_report_deferred = 1;             /* report_stop, below, once the MA   */
   raxopt( objcfunc, &pi1, npar, par, mtmp, maxits, nrits, grtol, sptol );
   opt_report_deferred = 0;             /* roots at the stop are known       */
   est_ma_boundary = 0;
   est_ma_nroots = 0;
   {
   int  cf = 0, ir;
   (*cast)( par, &varmax, &cf, 0, 0 );
   if ( cf == 0 && varmax.q > 0 )
      {
      int  nr = varmax.m * varmax.q;
      real *wr = vector( 1, nr ), *wi = vector( 1, nr ), *wmod = vector( 1, nr );
      chekma( varmax.m, varmax.q, varmax.theta, wr, wi, wmod, &cf );
      est_ma_nroots = nr;
      for ( ir = 1; ir <= nr; ir++ ) if ( wmod[ir] >= 1.0 - MA_WALL_TOL ) est_ma_boundary++;
      free_vector( wmod, 1, nr ); free_vector( wi, 1, nr ); free_vector( wr, 1, nr );
      }
   }
   report_stop( est_ma_boundary, est_ma_nroots );

/* This is an alternative way of computing the second derivative matrix:     */
/*                                                                           */
/*   raxopt leaves in mtmp the Hessian ACCUMULATED by BFGS along the path:   */
/*   good to steer the search, but not the curvature at the optimum -- it    */
/*   depends on the path and degrades in the flattest directions, which are  */
/*   those with the largest standard errors. With est_fdhess the Hessian is  */
/*   recomputed by finite differences AT the optimum, as drtran does. The    */
/*   ladder (.pre input) sets it; the .inp path keeps the 0.4.1 behaviour.   */

/*                                                                           */
/*   Two guards (docs/STUDY-standard-errors.md in drvarma-python):           */
/*   - a neighbour objcfunc refuses (Q not PD, unit root, non-invertible)    */
/*     means the optimum is on the boundary: no unrestricted Hessian;        */
/*   - choldcp is a MODIFIED Cholesky that patches small pivots, so the      */
/*     Hessian is first checked with a plain one.                            */
/*   Either way the BFGS factor is kept and est_se_how says why. With        */
/*   est_fixed = k, parameter k (qq[1,1] in the .inp path: Q -> cQ leaves    */
/*   the likelihood unchanged) is held; its dev and cov are NAN.             */

   est_se_how = EST_SE_BFGS;
   if ( est_fdhess )
      fdhess_cov( npar, par, pi1, cov, dev );

/* [3]: Sample estimation of the variance-covariance matrix:                 */

   if ( est_se_how == EST_SE_BFGS || est_se_how == EST_SE_BOUNDARY ||
        est_se_how == EST_SE_NOTPD )
   for ( i = 1; i <= npar; i++ )
       {
       for ( j = 1; j <= npar; j++ ) vtmp[j] = 0.0;
       vtmp[i] = 1.0;
       cholsol( mtmp, npar, vtmp );
       for ( j = 1; j <= npar; j++ )
           cov[j][i] = ( 2.0 * pi1 * vtmp[j] ) / varmax.n;
       dev[i] = sqrt( cov[i][i] );
       }

   free_vector( vtmp, 1, npar );
   free_matrix( mtmp, 1, npar, 1, npar );


/* [4]: Last computation of the log-likelihood: residuals and sigma2:        */

   (*cast)( par, &varmax, ifault, 0, 0 );

   varma_lik( varmax.m, varmax.n, varmax.p, varmax.q, varmax.mu, varmax.phi,
        varmax.theta, varmax.qq, varmax.w, 1.0, varmax.xitol,
        TRUE, a, &pi1, &pi2, &pi3, ifault );

   *logelf = -0.5 * varmax.m * varmax.n * ( LOG2PI - log( varmax.m )
             - log( varmax.n ) + 1.0 )
             - 0.5 * varmax.n * ( varmax.m * log( pi1 ) + log( pi2 ) );
   *sigma2 = pi1 / (varmax.n * varmax.m);

   (*cast)( par, &varmax, ifault, 0, 1 );     /* Deallocate VARMA structure. */
}

/*****************************************************************************/

real objcfunc( real *x )

{
   int  ifault;
   real pi1, pi2, pi3;

/* [1]: Put parameter vector into VARMA structure [since (*castx) receives   */
/*      ifault as a formal parameter, it is possible to check for model      */
/*      adequacy within (*castx), prior to the standard tests carried out    */
/*      within subroutine elf]:                                              */

   ifault = 0;
   (*castx)( x, &varmax, &ifault, 0, 0 );
   if ( ifault > 0 )                            /* ifault = 6-7-8- ...       */
      { objc_rejects++; return( 1.0 ); }

/* [2]: Compute objective function and return:                               */

   varma_lik( varmax.m, varmax.n, varmax.p, varmax.q, varmax.mu, varmax.phi,
        varmax.theta, varmax.qq, varmax.w, 1.0, varmax.xitol,
        FALSE, varmax.a, &pi1, &pi2, &pi3, &ifault );

   if ( ifault > 0 )                            /* ifault = 1-2-3-4-5.       */
      { objc_rejects++; return( 1.0 ); }

/* A point whose objective is not finite is inadmissible, exactly like an     */
/* elf ifault: return 1.0 (the article's strategy, sec. 3). Without this the  */
/* line search received the NaN and never returned (BUGS.md). As in drtran.   */
   {
   real f = pow( (pi1 / pi10x), varmax.m ) * (pi2 / pi20x);
   if ( !isfinite( f ) ) { objc_rejects++; return( 1.0 ); }
   return( f );
   }
}

/*****************************************************************************/