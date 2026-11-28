/*****************************************************************************/
/*  lik.c -- part of drvarma (multivariate VARMA modelling).
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
/*  LIK.C -- which exact likelihood the program evaluates.                  */
/*                                                                           */
/*  drvarma computes the exact likelihood of a VARMA model with Mauricio's   */
/*  algorithm (elf, AS 311). Its source also carried, compiled and never     */
/*  called since 1996, Shea's (marma, AS 242, 1989): the other of the two    */
/*  efficient exact methods, and the benchmark Mauricio's papers compare     */
/*  against. -lik shea makes it the objective:                               */
/*                                                                           */
/*  - two independent algorithms must give the same likelihood, so each is   */
/*    an oracle for the other (and for the Python port);                     */
/*  - elf truncates the xi sequence when its norm falls below xitol (-m 1,   */
/*    the default); with -m 2 it does not. Shea is always run EXACT (its own  */
/*    shortcut is a different approximation), so with -m 2 the two must      */
/*    agree to rounding -- measured, 1e-9 to 1e-13 in logL at every point the */
/*    optimizer visits -- and with -m 1 -lik both measures elf's truncation.  */
/*                                                                           */
/*  What Shea does NOT replace:                                              */
/*  - the RESIDUALS. marma returns the one-step innovations; drvarma's       */
/*    forecasts and diagnostics use elf's exact residuals (atf = TRUE), and  */
/*    confusing the two is drtran's BUG-55. With atf = TRUE the residuals    */
/*    come from elf and the likelihood from marma;                          */
/*  - the admissible region. marma does not test the MA operator; the same   */
/*    check elf applies (chekma, modulus >= 1.00005 -> ifault 4) is applied  */
/*    first, so both objectives refuse the same points.                      */
/*****************************************************************************/

#include "likhost.h"      /* the engine: real, elf, chekma, allocators */

int est_lik = LIK_ELF;

/* -lik both: elf is the objective and Shea is evaluated at every point too.  */
static long   chk_n = 0, chk_only_one = 0;
static double chk_max = 0.0, chk_last = 0.0;

const char *lik_label( void )
{
   return est_lik == LIK_SHEA ? "exact, Shea (1989), AS 242"
        : est_lik == LIK_BOTH ? "exact, Mauricio (1997), AS 311; checked "
                                "against Shea (1989), AS 242"
                              : "exact, Mauricio (1997), AS 311";
}

/* The concentrated log-likelihood from (f1, f2), as est() writes it.        */
static double loglik_of( int m, int n, real f1, real f2 )
{
   const real LOG2PI = 1.837877066;
   return -0.5 * m * n * ( LOG2PI - log( m ) - log( n ) + 1.0 )
          - 0.5 * n * ( m * log( f1 ) + log( f2 ) );
}

static void shea_only( int m, int n, int p, int q, real *mu, real ***phi,
                       real ***theta, real **qq, real **w, real sigma2,
                       real xitol, real *f1, real *f2, real *logelf,
                       int *ifault );

void lik_check_report( FILE *f )
{
   if ( est_lik != LIK_BOTH ) return;
   fprintf( f, "Shea check       : %ld points; max |dlogL| = %.3e, at the "
               "optimum %.3e", chk_n, chk_max, chk_last );
   if ( chk_only_one )
      fprintf( f, "; %ld points admissible for one algorithm only", chk_only_one );
   fprintf( f, "\n" );
}

void varma_lik( int m, int n, int p, int q, real *mu, real ***phi,
                real ***theta, real **qq, real **w, real sigma2, real xitol,
                int atf, real **a, real *f1, real *f2, real *logelf,
                int *ifault )
{
   int  i, t;
   real r1, r2, rlogl, **wt, **v;

   if ( est_lik == LIK_BOTH )
      {
      real g1, g2, glog;
      int  gf = 0;
      elf( m, n, p, q, mu, phi, theta, qq, w, sigma2, xitol, atf, a,
           f1, f2, logelf, ifault );
      shea_only( m, n, p, q, mu, phi, theta, qq, w, sigma2, xitol,
                 &g1, &g2, &glog, &gf );
      if ( ( *ifault > 0 ) != ( gf > 0 ) ) chk_only_one++;
      else if ( *ifault == 0 && *f1 > 0.0 && *f2 > 0.0 && g1 > 0.0 && g2 > 0.0 )
         {
         double d = fabs( loglik_of( m, n, *f1, *f2 ) - loglik_of( m, n, g1, g2 ) );
         chk_n++;
         if ( d > chk_max ) chk_max = d;
         if ( atf ) chk_last = d;          /* the final evaluation is atf    */
         }
      return;
      }
   if ( est_lik != LIK_SHEA )
      {
      elf( m, n, p, q, mu, phi, theta, qq, w, sigma2, xitol, atf, a,
           f1, f2, logelf, ifault );
      return;
      }

   *ifault = 0;
   if ( atf )                            /* the residuals are elf's          */
      {
      elf( m, n, p, q, mu, phi, theta, qq, w, sigma2, xitol, TRUE, a,
           f1, f2, logelf, ifault );
      if ( *ifault > 0 ) return;
      }
   shea_only( m, n, p, q, mu, phi, theta, qq, w, sigma2, xitol,
              f1, f2, logelf, ifault );
}

/* Shea's likelihood alone: elf's MA frontier (chekma), then marma.          */
static void shea_only( int m, int n, int p, int q, real *mu, real ***phi,
                       real ***theta, real **qq, real **w, real sigma2,
                       real xitol, real *f1, real *f2, real *logelf,
                       int *ifault )
{
   int  i, t;
   real r1, r2, rlogl, **wt, **v;

   *ifault = 0;
   if ( q > 0 )                          /* the same MA frontier as elf      */
      {
      real *wr = vector( 1, m * q ), *wi = vector( 1, m * q ),
           *wmod = vector( 1, m * q );
      chekma( m, q, theta, wr, wi, wmod, ifault );
      free_vector( wmod, 1, m * q );
      free_vector( wi, 1, m * q );
      free_vector( wr, 1, m * q );
      if ( *ifault > 0 ) { *ifault = 4; return; }
      }

   /* marma takes the series as w[series][time]; elf as w[time][series].    */
   wt = matrix( 1, m, 1, n );
   v  = matrix( 1, m, 1, n );
   for ( t = 1; t <= n; t++ )
       for ( i = 1; i <= m; i++ ) wt[i][t] = w[t][i];

   /* Shea is the REFERENCE, so it never truncates: its own shortcut (the   */
   /* steady-state recursion once F(t) is within xtol of Q) is a different   */
   /* approximation from elf's xi truncation, and with -m 1 it made the m6    */
   /* gate fail by 0.118. Exact, -lik both -m 1 measures elf's truncation.   */
   (void) xitol;
   marma( m, n, p, q, mu, phi, theta, qq, wt, sigma2, -1.0, TRUE, FALSE,
          v, &r1, &r2, &rlogl, ifault );

   free_matrix( v, 1, m, 1, n );
   free_matrix( wt, 1, m, 1, n );
   if ( *ifault > 0 ) return;

   *f1 = r1;
   *f2 = r2;
   *logelf = rlogl;
}
