/*****************************************************************************/
/*  multshea.c -- part of drvarma (multivariate VARMA modelling).
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
/*  MULTSHEA.C                                                               */
/*  Computation of the log-likelihood function of a vector ARMA(p,q) model.  */
/*  Source: Shea, B.L. (1989) ALGORITHM AS 242, 161-184.                     */
/*  Copyright (C) Jos‚ Alberto Mauricio, 1996.                               */
/*****************************************************************************/

#include "main.h"              /* Header file (prototype declarations)        */

/*****************************************************************************/
/*****************************************************************************/

void marma( int k, int n, int p, int q, real *mu, real ***phi, real ***theta,
            real **qq, real **w, real sigma2, real xtol, int chkma, int atf,
            real **v, real *r1, real *r2, real *rlogl, int *ifault )

{
   const real LOG2PI = 1.837877066;

/* Declaration of variables and functions:                                   */

   int  i, j, j7, k1, k2, kr, l, l8, m, r, t, annot, delta, pearl;
   real sum, detp, ssq, sm, sig, tsig;
   real ***gamma, ***mat, ***temp, ***tempk;
   real **a, **f, **invf, **mt, **templ, **tempm, **wa, *b, *z;

   void chol( real **a, int k, real **l, int *ifault );
   void bksb( real **l, int k, int m, int upper, real **b, int *ifault );
   void covars( int k, int p, int q, int r, real ***phi, real ***theta,
                real **qq, real ***gamwa, real *gamma, int *ifault );

/* Initialization and workspace allocation:                                  */

   r      = ( p > q ) ? p : q;
   kr     = k * r;
   annot  = ( p > q ) ? TRUE : FALSE;
   *r1    = 0.0;
   *r2    = 0.0;
   *rlogl = 0.0;

   gamma = tensor( 0, r, 1, k, 1, k );
   mat   = tensor( 1, r, 1, k, 1, k );
   temp  = tensor( 1, r, 1, k, 1, k );
   tempk = tensor( 1, r, 1, k, 1, k );
   a     = matrix( 1, k, 1, k );
   f     = matrix( 1, k, 1, k );
   invf  = matrix( 1, k, 1, k );
   mt    = matrix( 1, k, 1, k );
   templ = matrix( 1, k, 1, k );
   tempm = matrix( 1, k, 1, k );
   wa    = matrix( 1, k, 1, k );
   b     = vector( 1, k );
   z     = vector( 1, k * k * (r+1) );

/* Set upper triangle of qq to lower triangle of qq:                         */

   for ( i = 2; i <= k; i++ )
       for ( j = 1; j <= i-1; j++ ) qq[j][i] = qq[i][j];

/* Check wether qq is positive-definite and evaluate its determinant         */
/* (storing it in tsig):                                                     */

   chol( qq, k, a, ifault );

   if ( *ifault > 0 )
      {
      *ifault = 1;                           /* qq is NOT positive-definite: */
      goto r0;                               /* set flag and return.         */
      }
   else
      {
      tsig = a[1][1];
      for ( i = 2; i <= k; i++ ) tsig *= a[i][i];
      tsig *= tsig;
      }

/* Call "covars" to calculate the autocovariances of W(t) and the cross      */
/* covariances between W(t) and E(t):                                        */

   covars( k, p, q, r, phi, theta, qq, gamma, z, ifault );

   if ( *ifault > 0 )               /* Autoregressive unit root(s) detected: */
      {                             /* set flag and return.                  */
      *ifault = 2;
      goto r0;
      }

/* Calculate the r (kxk) matrix components of P(1/0)h and store as tempk:    */

   for ( k1 = 1; k1 <= r; k1++ )
       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               sum = 0.0;
               for ( k2 = 1; k2 <= k; k2++ )
                   for ( m = k1; m <= p; m++ )
                       sum += phi[m][i][k2] * z[(m-k1+1)*k*k+(k2-1)*k+j];
               for ( m = k1; m <= q; m++ )
                   for ( k2 = 1; k2 <= k; k2++ )
                       sum -= theta[m][i][k2] * gamma[m-k1+1][j][k2];
               tempk[k1][i][j] = sum;
               }

/* Initialize A(1/0), V(1), F(1), ssq and detp:                              */

   for ( i = 1; i <= kr; i++ ) z[i] = 0.0;

   for ( i = 1; i <= k; i++ )
       {
       for ( j = 1; j <= k; j++ )
           f[i][j] = tempk[1][i][j] + qq[i][j];
       v[i][1] = w[i][1] - mu[i];
       b[i] = v[i][1];
       }

/* Factorize F(1) as LL' and store L as matrix a:                            */

   chol( f, k, a, ifault );

   if ( *ifault > 0 )
      {
      *ifault = 3;                      /* Strict non-stationarity detected: */
      goto r0;                          /* set flag and return.              */
      }

/* Calculate determinant (detp) of F(1):                                     */

   detp = a[1][1];
   for ( i = 2; i <= k; i++ ) detp *= a[i][i];
   detp *= detp;

/* Set mt = F(1) inverse:                                                    */

   for ( i = 1; i <= k; i++ )
       {
       for ( j = 1; j <= k; j++ ) mt[i][j] = 0.0;
       mt[i][i] = 1.0;
       }
   bksb( a, k, k, FALSE, mt, ifault );

/* Set upper triangle of invf to a transpose:                                */

   for ( j = 1; j <= k; j++ )
       for ( i = 1; i <= j; i++ ) invf[i][j] = a[j][i];
   bksb( invf, k, k, TRUE, mt, ifault );
   for ( i = 1; i <= k; i++ )
       {
       sum = b[i];
       for ( j = 1; j <= i-1; j++ ) sum -= a[i][j] * b[j];
       b[i] = sum / a[i][i];
       }

   ssq = 0.0;
   for ( i = 1; i <= k; i++ ) ssq += b[i] * b[i];
   detp = logl( detp );

/* Calculate L(1) and K(1):                                                  */

   for ( l = 1; l <= r; l++ )
       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               mat[l][i][j]   = 0.0;
               gamma[l][i][j] = 0.0;
               }

   for ( l = 1; l <= r; l++ )
       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               sum = 0.0;
               if ( l <= p )
                  for ( k2 = 1; k2 <= k; k2++ )
                      sum += phi[l][i][k2] * tempk[1][k2][j];
               if ( l < r )
                  sum += tempk[l+1][i][j];
               for ( k2 = 1; k2 <= k; k2++ )
                   {
                   sm = 0.0;
                   if ( l <= p ) sm = phi[l][i][k2];
                   if ( l <= q ) sm -= theta[l][i][k2];
                   sum += sm * qq[k2][j];
                   }
               mat[l][i][j] = sum;
               }

   if ( annot )
      for ( l = 1; l <= r; l++ )
          for ( i = 1; i <= k; i++ )
              for ( j = 1; j <= k; j++ )
                  {
                  sum = 0.0;
                  if ( l <= q )
                     for ( k2 = 1; k2 <= k; k2++ )
                         sum += theta[l][i][k2] * tempk[1][k2][j];
                  if ( l < r )
                     sum += tempk[l+1][i][j];
                  gamma[l][i][j] = sum;
                  }
   else
      for ( l = 1; l <= r; l++ )
          for ( i = 1; i <= k; i++ )
              for ( j = 1; j <= k; j++ ) gamma[l][i][j] = mat[l][i][j];

/* Start the recursions (main computations):                                 */

   delta = FALSE;
   pearl = FALSE;
   j7    = n;

   for ( t = 2; t <= n; t++ )
       {
       if ( (annot) && (t > p-q) ) pearl = TRUE;
       if ( delta == FALSE )
          {
          for ( j = 1; j <= k; j++ )
              {
              for ( i = 1; i <= k; i++ )
                  {
                  tempk[1][i][j] = a[j][i];
                  invf[i][j] = 0.0;
                  }
              invf[j][j] = 1.0;
              }
          bksb( tempk[1], k, k, TRUE, invf, ifault );

          for ( l = 1; l <= r; l++ )
              for ( i = 1; i <= k; i++ )
                  for ( j = 1; j <= k; j++ )
                      {
                      sum = 0.0;
                      for ( k2 = 1; k2 <= k; k2++ )
                          sum += gamma[l][i][k2] * invf[k2][j];
                      tempk[l][i][j] = sum;
                      }
          }

       for ( l = 1; l <= r; l++ )
           for ( i = 1; i <= k; i++ )
               for ( j = 1; j <= k; j++ ) temp[l][i][j] = 0.0;

       if ( annot ) goto r1;
       if ( t == 2 ) goto r2;

       for ( l = 1; l <= r; l++ )
           for ( i = 1; i <= k; i++ )
               {
               sum = 0.0;
               if ( l <= p )
                  for ( k2 = 1; k2 <= k; k2++ )
                      sum += phi[l][i][k2] * z[k2];
               if ( l < r )
                  sum += z[l*k+i];
               temp[l][1][i] = sum;
               }

       goto r2;

r1:    if ( t > 2 )
          for ( l = 1; l <= r; l++ )
              for ( i = 1; i <= k; i++ )
                  {
                  sum = 0.0;
                  if ( l <= q )
                     for ( k2 = 1; k2 <= k; k2++ )
                         sum += theta[l][i][k2] * z[k2];
                  if ( l < r )
                     sum += z[l*k+i];
                  temp[l][1][i] = sum;
                  }
       for ( l = 1; l <= r; l++ )
           for ( i = 1; i <= k; i++ )
               {
               sum = 0.0;
               for ( k2 = 1; k2 <= k; k2++ )
                   {
                   sm = 0.0;
                   if ( l <= p ) sm = phi[l][i][k2];
                   if ( l <= q ) sm -= theta[l][i][k2];
                   sum += sm * (w[k2][t-1] - mu[k2]);
                   }
               temp[l][1][i] += sum;
               }

r2:    for ( l = 1; l <= r; l++ )
           for ( l8 = 1; l8 <= k; l8++ )
               {
               sum = temp[l][1][l8];
               if ( (delta == FALSE) || (annot == FALSE) )
                  for ( k2 = 1; k2 <= k; k2++ )
                      sum += tempk[l][l8][k2] * b[k2];
               z[(l-1)*k+l8] = sum;
               }

       if ( delta ) goto r4;

       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ ) templ[i][j] = mat[1][i][j];

       if ( annot == FALSE )
          for ( l = 1; l <= r; l++ )
              for ( i = 1; i <= k; i++ )
                  for ( j = 1; j <= k; j++ )
                      {
                      sum = 0.0;
                      if ( l <= p )
                         for ( k2 = 1; k2 <= k; k2++ )
                             sum += phi[l][i][k2] * mat[1][k2][j];
                      if ( l < r )
                         sum += mat[l+1][i][j];
                      temp[l][i][j] = sum;
                      }
       else
          for ( l = 1; l <= r; l++ )
              for ( i = 1; i <= k; i++ )
                  for ( j = 1; j <= k; j++ )
                      {
                      sum = 0.0;
                      if ( l <= q )
                         for ( k2 = 1; k2 <= k; k2++ )
                             sum += theta[l][i][k2] * mat[1][k2][j];
                      if ( l < r )
                         sum += mat[l+1][i][j];
                      temp[l][i][j] = sum;
                      }

       chol( mt, k, invf, ifault );

       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               sum = 0.0;
               for ( k2 = j; k2 <= k; k2++ ) sum += templ[i][k2] * invf[k2][j];
               wa[i][j] = sum;
               }
       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               sum = 0.0;
               for ( k2 = 1; k2 <= i; k2++ ) sum += invf[i][k2] * wa[j][k2];
               tempm[i][j] = sum;
               }
       for ( l = 1; l <= r; l++ )
           for ( i = 1; i <= k; i++ )
               for ( j = 1; j <= k; j++ )
                   {
                   sum = 0.0;
                   if ( (pearl) && (l >= q+1) )
                      gamma[l][i][j] = sum;
                   else
                      for ( k2 = 1; k2 <= k; k2++ )
                          sum -= temp[l][i][k2] * tempm[k2][j];
                   gamma[l][i][j] += sum;
                   }
       for ( i = 1; i <= k; i++ )
           for ( j = i; j <= k; j++ )
               {
               sum = f[i][j];
               for ( k2 = 1; k2 <= k; k2++ ) sum -= wa[i][k2] * wa[j][k2];
               f[i][j] = sum;
               f[j][i] = sum;
               }

       sum = 0.0;
       for ( i = 1; i <= k; i++ )
           if ( qq[i][i] > 0.0 )
              sum = ( sum > fabsl( f[i][i] - qq[i][i] ) / qq[i][i] ) ? sum
                    : fabsl( f[i][i] - qq[i][i] ) / qq[i][i];
           else
              sum = ( sum > fabsl( f[i][i] - qq[i][i] ) ) ? sum
                    : fabsl( f[i][i] - qq[i][i] );
       if ( sum < xtol ) delta = TRUE;
       if ( delta == FALSE ) goto r3;

       j7 = t;

       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               for ( l = 1; l <= r; l++ )
                   {
                   sum = 0.0;
                   if ( l <= p ) sum = phi[l][i][j];
                   if ( l <= q ) sum -= theta[l][i][j];
                   tempk[l][i][j] = sum;
                   }
       for ( l = 1; l <= r; l++ )
           {
           for ( i = 1; i <= k; i++ )
               for ( j = 1; j <= k; j++ )
                      {
                      sum = 0.0;
                      for ( k2 = j; k2 <= k; k2++ )
                          sum += tempk[l][i][k2] * a[k2][j] ;
                      wa[i][j] = sum;
                      }
           for ( i = 1; i <= k; i++ )
               for ( j = 1; j <= k; j++ ) tempk[l][i][j] = wa[i][j];
           }

r3:    for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ ) wa[i][j] = a[i][j];
       chol( f, k, a, ifault );
       if ( *ifault > 0 )
          {
          *ifault = 3;                  /* Strict non-stationarity detected: */
          goto r0;
          }
       sig = a[1][1];
       for ( i = 2; i <= k; i++ ) sig *= a[i][i];
       sig *= sig;

       if ( delta ) goto r4;

       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= i; j++ )
               {
               sum = tempm[i][j];
               tempm[i][j] = tempm[j][i];
               tempm[j][i] = sum;
               }

       bksb( a, k, k, FALSE, tempm, ifault );

       for ( i = 1; i <= k; i++ )
           for ( j = i; j <= k; j++ )
               {
               sum = mt[i][j];
               for ( k2 = 1; k2 <= k; k2++ )
                   sum += tempm[k2][i] * tempm[k2][j];
               mt[i][j] = sum;
               mt[j][i] = sum;
               }

       bksb( wa, k, k, FALSE, templ, ifault );

       for ( l = 1; l <= r; l++ )
           for ( i = 1; i <= k; i++ )
               for ( j = 1; j <= k; j++ )
                   {
                   sum = 0.0;
                   if ( (pearl) && ( l >= q+1) )
                      mat[l][i][j] = sum;
                   else
                      for ( k2 = 1; k2 <= k; k2++ )
                          sum += tempk[l][i][k2] * templ[k2][j];
                   mat[l][i][j] = temp[l][i][j] - sum;
                   }

r4:    for ( i = 1; i <= k; i++ )
           {
           v[i][t] = w[i][t] - z[i] - mu[i];
           b[i] = v[i][t];
           }
       for ( i = 1; i <= k; i++ )
           {
           sum = b[i];
           for ( j = 1; j <= i-1; j++ ) sum -= a[i][j] * b[j];
           b[i] = sum / a[i][i];
           }
       for ( i = 1; i <= k; i++ ) ssq += b[i] * b[i];
       if ( (delta == FALSE) || (t <= j7) ) detp += logl( sig );
       }

/* Compute the determinant = exp( detp ) * pow( tsig, n - j7 ) (note that r2 */
/* is returned as the appropriate factor in the objective function istead):  */

   *r1    = ssq;

   *r2    = expl( detp ) * powl( tsig, n - j7 );
   *r2    = logl( *r2 ) / n;
   *r2    = expl( *r2 );

   *rlogl = -0.5 * (n * k * (LOG2PI + logl( sigma2 )) + (n - j7) * logl( tsig ) +
            detp + ssq / sigma2 );

/* Deallocate workspace and return:                                          */

r0:free_vector( z, 1, k * k * (r+1) );
   free_vector( b, 1, k );
   free_matrix( wa, 1, k, 1, k );
   free_matrix( tempm, 1, k, 1, k );
   free_matrix( templ, 1, k, 1, k );
   free_matrix( mt, 1, k, 1, k );
   free_matrix( invf, 1, k, 1, k );
   free_matrix( f, 1, k, 1, k );
   free_matrix( a, 1, k, 1, k );
   free_tensor( tempk, 1, r, 1, k, 1, k );
   free_tensor( temp, 1, r, 1, k, 1, k );
   free_tensor( mat, 1, r, 1, k, 1, k );
   free_tensor( gamma, 0, r, 1, k, 1, k );
}

/*****************************************************************************/
/*****************************************************************************/

void covars( int k, int p, int q, int r, real ***phi, real ***theta,
             real **qq, real ***gamwa, real *gamma, int *ifault )

{
   int  i, i2, j, j2, k2, kw, l, l4, m, *iwork;
   real sum, **mat;

   *ifault = 0;

/* [1]: Compute the q cross-covariance matrices and return as gamwa:         */

   if ( q > 0 )
      for ( i = 1; i <= k; i++ )
          for ( j = 1; j <= k; j++ ) gamwa[0][i][j] = qq[i][j];

   for ( m = 1; m <= q; m++ )
       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               sum = 0.0;
               for ( i2 = 1; i2 <= k; i2++ )
                   sum -= theta[m][i][i2] * qq[i2][j];
               for ( k2 = 1; k2 <= p; k2++ )
                   for ( j2 = 1; j2 <= k; j2++ )
                       if ( m >= k2 )
                          sum += phi[k2][i][j2] * gamwa[m-k2][j2][j];
               gamwa[m][i][j] = sum;
               }

/* [2]: Set up right-hand side vector (gamma) and coefficient matrix (mat):  */

   kw  = k * k * (p + 1);
   mat = matrix( 1, kw, 1, kw );

   for ( j = 1; j <= kw; j++ )
       {
       for ( i = 1; i <= kw; i++ ) mat[i][j] = 0.0;
       gamma[j] = 0.0;
       }

   for ( m = 0; m <= p; m++ )
       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               l = m * k * k + (i-1) * k + j;
               if ( m == 0 ) gamma[l] = qq[i][j];
               if ( (m > 0) && (m <= q) )
                  for ( i2 = 1; i2 <= k; i2++ )
                      gamma[l] -= qq[i][i2] * theta[m][j][i2];
               for ( l4 = m + 1; l4 <= q; l4++ )
                   for ( i2 = 1; i2 <= k; i2++ )
                       gamma[l] -= gamwa[l4-m][i][i2] * theta[l4][j][i2];
               mat[l][l] = 1.0;
               for ( i2 = 1; i2 <= p; i2++ )
                   for ( k2 = 1; k2 <= k; k2++ )
                       {
                       if ( m >= i2 )
                          l4 = (m-i2) * k * k + (i-1) * k + k2;
                       else
                          l4 = (i2-m) * k * k + (k2-1) * k + i;
                       mat[l][l4] -= phi[i2][j][k2];
                       }
               }

/* [3]: Calculate the first p autocovariances (overwrite gamma):             */

   if ( p > 0 )
      {
      iwork = ivector( 1, kw );
      ludcp( mat, kw, iwork );
      if ( iwork[kw] == 0 )
         {
         *ifault = 1;
         free_ivector( iwork, 1, kw );
         free_matrix( mat, 1, kw, 1, kw );
         return;
         }
      else
         lusol( mat, gamma, kw, iwork );
      free_ivector( iwork, 1, kw );
      }
      free_matrix( mat, 1, kw, 1, kw );

/* [4]: If needed, calculate the autocovariances from p+1 up to lag r:       */

   for ( m = p + 1; m <= r; m++ )
       for ( i = 1; i <= k; i++ )
           for ( j = 1; j <= k; j++ )
               {
               sum = 0.0;
               for ( l4 = 1; l4 <= p; l4++ )
                   for ( i2 = 1; i2 <= k; i2++ )
                       sum += gamma[(m-l4)*k*k+(i-1)*k+i2] * phi[l4][j][i2];
               if ( m <= q )
                  {
                  for ( i2 = 1; i2 <= k; i2++ )
                      sum -= qq[i][i2] * theta[m][j][i2];
                  for ( i2 = m + 1; i2 <= q; i2++ )
                      for ( l4 = 1; l4 <= k; l4++ )
                          sum -= gamwa[i2-m][i][l4] * theta[i2][j][l4];
                  }
               gamma[m*k*k+(i-1)*k+j] = sum;
               }
}

/*****************************************************************************/

void chol( real **a, int k, real **l, int *ifault )

{
   real sum;
   int  i, j, k2;

   *ifault = 0;

   for ( j = 1; j <= k; j++ )
       {
       sum = a[j][j];
       for ( k2 = 1; k2 <= j-1; k2++ ) sum -= l[j][k2] * l[j][k2];

       if ( sum <= 0.0 )
          {
          *ifault = 1;
          return;
          }
       else
          l[j][j] = sqrtl( sum );

       for ( i = j+1; i <= k; i++ )
           {
           sum = a[i][j];
           for ( k2 = 1; k2 <= j-1; k2++ ) sum -= l[i][k2] * l[j][k2];
           l[i][j] = sum / l[j][j];
           l[j][i] = 0.0;
           }
       }
}

/*****************************************************************************/

void bksb( real **l, int k, int m, int upper, real **b, int *ifault )

{
   real sum;
   int  i, j, j2;

   *ifault = 0;

   if ( upper )
      {
      for ( j2 = 1; j2 <= m; j2++ ) for ( i = k; i >= 1; i-- )
          {
          sum = b[i][j2];
          for ( j = i+1; j <= k; j++ ) sum -= l[i][j] * b[j][j2];
          if ( l[i][i] != 0.0 )
             b[i][j2] = sum / l[i][i];
          else
             {
             *ifault = 1;
             return;
             }
          }
      }
   else
      {
      for ( j2 = 1; j2 <= m; j2++ ) for ( i = 1; i <= k; i++ )
          {
          sum = b[i][j2];
          for ( j = 1; j <= i-1; j++ ) sum -= l[i][j] * b[j][j2];
          if ( l[i][i] != 0.0 )
             b[i][j2] = sum / l[i][i];
          else
             {
             *ifault = 1;
             return;
             }
          }
      }
}

/*****************************************************************************/
