/* lik.h -- lib/lik: which exact likelihood the engine evaluates.
 *
 * Mauricio's elf (AS 311, the engine's own elfvarma.c) or Shea's marma
 * (AS 242, multshea.c here), or elf checked against Shea at every point.
 * Shared by drvarma and drtran. Include it from the host's main.h, after
 * `real` is defined: the host provides elf(), chekma() and the allocators
 * (likhost.h). */
#ifndef LIK_H
#define LIK_H
#include <stdio.h>

/* Which exact likelihood (lib/lik/lik.c): Mauricio's elf (default) or Shea's marma. */
#define LIK_ELF   0
#define LIK_SHEA  1
#define LIK_BOTH  2       /* elf is the objective; Shea checked at every point */

/* The MA invertibility wall, one tolerance for both sides (2026-09-28).     */
/* chekma refuses a point with an MA inverse root at modulus >= 1 + 5e-5;    */
/* a stop with a root at modulus >= 1 - 5e-5 is reported as on the wall. A   */
/* root at 1.000048 and one at 0.99999999 are the same fact: the unit circle */
/* to chekma's own precision. Before, only >= 1 counted, so an optimizer     */
/* that reached the wall from inside said "converged".                       */
#define MA_WALL_TOL 5e-5
extern int est_lik;
const char *lik_label( void );
void lik_check_report( FILE *f );
void varma_lik( int m, int n, int p, int q, real *mu, real ***phi,
                real ***theta, real **qq, real **w, real sigma2, real xitol,
                int atf, real **a, real *f1, real *f2, real *logelf,
                int *ifault );
void marma( int k, int n, int p, int q, real *mu, real ***phi, real ***theta,
            real **qq, real **w, real sigma2, real xtol, int chkma, int atf,
            real **v, real *r1, real *r2, real *rlogl, int *ifault );

#endif
