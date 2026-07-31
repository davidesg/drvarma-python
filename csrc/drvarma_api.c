/*****************************************************************************/
/*  drvarma_api.c -- C library API over the drvarma estimation core.         */
/*  GPL v2 or later (see COPYING).                                            */
/*****************************************************************************/

#include "main.h"
#include "drvarma_api.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

real macheps;
FILE *outputv;
int  quiet_mode = 1;

real **datamat;
int  nser, nobs;
int  data_freq = 1, data_start_sub = 1, data_start_year = 1;
char **series_names = NULL;

real trans_lambda = 1.0;
real trans_scale  = 1.0;
int  trans_d = 0, trans_D = 0;
real **bc_series = NULL;
int  nobs_raw = 0;
int  do_deseason = 0, deseason_mode = 0;
real **seasonal_dummies = NULL;

int  global_p, global_q;
int  global_include_mean = 0;
int  global_diag_ar = 0, global_diag_ma = 0, global_diag_cov = 0;
int  met = 1;
int  global_twostep = 0;
int  g_estwin = 0, g_in_est = 0;

/* ---- estimation machinery extracted from drvarma.c ---- */
static void shootx(real *x, struct Tvarma *armax, int *ifaultx, int firstx, int lastx)
{
    int i, j, k, idx = 1;
    int m = nser;        /* system dimension */
    int p = global_p;
    int q = global_q;

    *ifaultx = 0;

    /* [1] Set dimensions.  During estimation (g_in_est) the likelihood is
       restricted to the first g_estwin observations; buffers stay full size. */
    armax->m = m;
    armax->n = (g_in_est && g_estwin > 0) ? g_estwin : nobs;
    armax->p = p;
    armax->q = q;

    /* [2] Allocate memory on first call */
    if (firstx) {
        armax->mu    = vector(1, m);
        armax->phi   = tensor(0, p, 1, m, 1, m);
        armax->theta = tensor(0, q, 1, m, 1, m);
        armax->qq    = matrix(1, m, 1, m);
        armax->w     = matrix(1, nobs, 1, m);
        armax->a     = matrix(1, nobs, 1, m);

        /* Initialize to zero */
        for (i = 1; i <= m; i++) {
            armax->mu[i] = 0.0;
            for (j = 1; j <= m; j++) {
                for (k = 0; k <= p; k++) armax->phi[k][i][j] = 0.0;
                for (k = 0; k <= q; k++) armax->theta[k][i][j] = 0.0;
                armax->qq[i][j] = 0.0;
            }
            for (j = 1; j <= nobs; j++) {
                armax->w[j][i] = 0.0;
                armax->a[j][i] = 0.0;
            }
        }
        /* phi[0] and theta[0] = identity */
        for (i = 1; i <= m; i++) {
            armax->phi[0][i][i] = 1.0;
            armax->theta[0][i][i] = 1.0;
        }
    }

    /* [3] Build unnormalized model (phi1, theta1, qq1, mu1) */
    /* Allocate temporary memory */
    real *mu1    = vector(1, m);
    real ***phi1   = tensor(0, p, 1, m, 1, m);
    real ***theta1 = tensor(0, q, 1, m, 1, m);
    real **qq1    = matrix(1, m, 1, m);

    /* Initialize to zero */
    for (i = 1; i <= m; i++) {
        mu1[i] = 0.0;
        for (j = 1; j <= m; j++) {
            for (k = 0; k <= p; k++) phi1[k][i][j] = 0.0;
            for (k = 0; k <= q; k++) theta1[k][i][j] = 0.0;
            qq1[i][j] = 0.0;
        }
        phi1[0][i][i] = 1.0;
        theta1[0][i][i] = 1.0;
    }

    /* [3a] Means (if included) */
    if (global_include_mean) {
        for (i = 1; i <= m; i++)
            mu1[i] = x[idx++];
    }

    /* [3b] AR parameters */
    for (k = 1; k <= p; k++) {
        if (global_diag_ar) {
            for (i = 1; i <= m; i++) {
                phi1[k][i][i] = x[idx++];
            }
        } else {
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= m; j++) {
                    phi1[k][i][j] = x[idx++];
                }
            }
        }
    }

    /* [3c] MA parameters */
    for (k = 1; k <= q; k++) {
        if (global_diag_ma) {
            for (i = 1; i <= m; i++) {
                theta1[k][i][i] = x[idx++];
            }
        } else {
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= m; j++) {
                    theta1[k][i][j] = x[idx++];
                }
            }
        }
    }

    /* [3d] Covariance matrix Q (lower triangular) */
    if (global_diag_cov) {
        for (i = 1; i <= m; i++) {
            qq1[i][i] = x[idx++];
        }
    } else {
        for (i = 1; i <= m; i++) {
            for (j = 1; j <= i; j++) {
                qq1[i][j] = x[idx++];
                qq1[j][i] = qq1[i][j];   /* symmetry */
            }
        }
    }

    /* [4] Normalization (as in original, using phi1[0] and theta1[0]) */
    /* Assumes functions ludcp, lusol, etc. are available */
    real **mtmp1 = matrix(1, m, 1, m);
    real **mtmp2 = matrix(1, m, 1, m);
    real **mtmp0 = matrix(1, m, 1, m);
    real *vtmp0  = vector(1, m);
    int *index   = ivector(1, m);

    /* mtmp0 = phi1[0] */
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++)
            mtmp0[i][j] = phi1[0][i][j];
    ludcp(mtmp0, m, index);
    for (j = 1; j <= m; j++) {
        for (i = 1; i <= m; i++) vtmp0[i] = 0.0;
        vtmp0[j] = 1.0;
        lusol(mtmp0, vtmp0, m, index);
        for (i = 1; i <= m; i++) mtmp1[i][j] = vtmp0[i];
    }

    /* mtmp0 = theta1[0] */
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++)
            mtmp0[i][j] = theta1[0][i][j];
    ludcp(mtmp0, m, index);
    for (j = 1; j <= m; j++) {
        for (i = 1; i <= m; i++) vtmp0[i] = 0.0;
        vtmp0[j] = 1.0;
        lusol(mtmp0, vtmp0, m, index);
        for (i = 1; i <= m; i++) mtmp2[i][j] = vtmp0[i];
    }

    free_ivector(index, 1, m);
    free_vector(vtmp0, 1, m);
    free_matrix(mtmp0, 1, m, 1, m);

    /* Normalized AR */
    for (k = 1; k <= p; k++)
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++) {
                armax->phi[k][i][j] = 0.0;
                for (int k1 = 1; k1 <= m; k1++)
                    armax->phi[k][i][j] += mtmp1[i][k1] * phi1[k][k1][j];
            }

    real **mtmp3 = matrix(1, m, 1, m);
    real **mtmp4 = matrix(1, m, 1, m);

    /* mtmp3 = theta1[0]^{-1} * phi1[0] */
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) {
            mtmp3[i][j] = 0.0;
            for (int k1 = 1; k1 <= m; k1++)
                mtmp3[i][j] += mtmp2[i][k1] * phi1[0][k1][j];
        }

    /* Normalized MA */
    for (k = 1; k <= q; k++) {
        /* mtmp4 = phi1[0]^{-1} * theta1[k] */
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++) {
                mtmp4[i][j] = 0.0;
                for (int k1 = 1; k1 <= m; k1++)
                    mtmp4[i][j] += mtmp1[i][k1] * theta1[k][k1][j];
            }
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++) {
                armax->theta[k][i][j] = 0.0;
                for (int k1 = 1; k1 <= m; k1++)
                    armax->theta[k][i][j] += mtmp4[i][k1] * mtmp3[k1][j];
            }
    }

    /* mtmp3 = phi1[0]^{-1} * theta1[0] */
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) {
            mtmp3[i][j] = 0.0;
            for (int k1 = 1; k1 <= m; k1++)
                mtmp3[i][j] += mtmp1[i][k1] * theta1[0][k1][j];
        }

    /* Make qq1 symmetric (just in case) */
    for (i = 1; i <= m; i++)
        for (j = i+1; j <= m; j++)
            qq1[i][j] = qq1[j][i];

    /* mtmp4 = phi1[0]^{-1} * theta1[0] * qq1 */
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) {
            mtmp4[i][j] = 0.0;
            for (int k1 = 1; k1 <= m; k1++)
                mtmp4[i][j] += mtmp3[i][k1] * qq1[k1][j];
        }

    /* Normalized covariance: armax->qq = mtmp4 * mtmp3' */
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) {
            armax->qq[i][j] = 0.0;
            for (int k1 = 1; k1 <= m; k1++)
                armax->qq[i][j] += mtmp4[i][k1] * mtmp3[j][k1];
        }

    /* Mean */
    for (i = 1; i <= m; i++)
        armax->mu[i] = mu1[i];

    /* Free temporary memory */
    free_matrix(mtmp4, 1, m, 1, m);
    free_matrix(mtmp3, 1, m, 1, m);
    free_matrix(mtmp2, 1, m, 1, m);
    free_matrix(mtmp1, 1, m, 1, m);
    free_matrix(qq1, 1, m, 1, m);
    free_tensor(theta1, 0, q, 1, m, 1, m);
    free_tensor(phi1, 0, p, 1, m, 1, m);
    free_vector(mu1, 1, m);

    /* [5] Data: copy datamat to armax->w */
    for (i = 1; i <= nobs; i++)
        for (j = 1; j <= m; j++)
            armax->w[i][j] = datamat[i][j];

    /* [6] Deallocate model memory if lastx is active */
    if (lastx) {
        free_matrix(armax->a, 1, armax->n, 1, armax->m);
        free_matrix(armax->w, 1, armax->n, 1, armax->m);
        free_matrix(armax->qq, 1, armax->m, 1, armax->m);
        free_tensor(armax->theta, 0, armax->q, 1, armax->m, 1, armax->m);
        free_tensor(armax->phi, 0, armax->p, 1, armax->m, 1, armax->m);
        free_vector(armax->mu, 1, armax->m);
    }
}

static int calc_nparametrs(void)
{
    int npar = 0;
    int m = nser;

    if (global_include_mean)
        npar += m;

    int n_ar = global_diag_ar ? m : m * m;
    npar += n_ar * global_p;

    int n_ma = global_diag_ma ? m : m * m;
    npar += n_ma * global_q;

    int n_cov = global_diag_cov ? m : m * (m + 1) / 2;
    npar += n_cov;

    return npar;
}

static void init_varma(real *x, int npar)
{
    int m = nser;
    int i, j, k, t, idx = 1;
    real *mean_est = NULL;

    /* 1. Medias muestrales (si se incluyen) */
    if (global_include_mean) {
        mean_est = vector(1, m);
        for (j = 1; j <= m; j++) {
            real sum = 0.0;
            for (t = 1; t <= nobs; t++) sum += datamat[t][j];
            mean_est[j] = sum / nobs;
            x[idx++] = mean_est[j];
        }
    }

    /* Datos centrados */
    real **datac = matrix(1, nobs, 1, m);
    for (t = 1; t <= nobs; t++)
        for (j = 1; j <= m; j++)
            datac[t][j] = datamat[t][j] - (global_include_mean ? mean_est[j] : 0.0);

    /* Matrices para covarianza y correlaciÃÂ³n de residuos */
    real **resid_cov = matrix(1, m, 1, m);
    real **resid_corr = matrix(1, m, 1, m);
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++)
            resid_cov[i][j] = 0.0;

    /* 2. ParÃÂ¡metros AR y cÃÂ¡lculo de residuos */
    if (global_p > 0) {
        int T = nobs - global_p;
        real **Z = matrix(1, T, 1, m * global_p);
        real **y = matrix(1, T, 1, m);

        /* Construir matriz de regresores y vector respuesta */
        for (t = global_p + 1; t <= nobs; t++) {
            int row = t - global_p;
            int col = 1;
            for (k = 1; k <= global_p; k++) {
                for (j = 1; j <= m; j++) {
                    Z[row][col++] = datac[t - k][j];
                }
            }
            for (j = 1; j <= m; j++)
                y[row][j] = datac[t][j];
        }

        /* Z'Z y factorizaciÃÂ³n LU */
        real **ZtZ = matrix(1, m * global_p, 1, m * global_p);
        for (i = 1; i <= m * global_p; i++) {
            for (j = 1; j <= m * global_p; j++) {
                real sum = 0.0;
                for (t = 1; t <= T; t++) sum += Z[t][i] * Z[t][j];
                ZtZ[i][j] = sum;
            }
        }
        int *indx = ivector(1, m * global_p);
        ludcp(ZtZ, m * global_p, indx);

        /* Coeficientes por ecuaciÃÂ³n */
        real **coef = matrix(1, m, 1, m * global_p);
        for (int eq = 1; eq <= m; eq++) {
            real *Zty = vector(1, m * global_p);
            for (i = 1; i <= m * global_p; i++) {
                Zty[i] = 0.0;
                for (t = 1; t <= T; t++)
                    Zty[i] += Z[t][i] * y[t][eq];
            }
            lusol(ZtZ, Zty, m * global_p, indx);
            for (i = 1; i <= m * global_p; i++)
                coef[eq][i] = Zty[i];
            free_vector(Zty, 1, m * global_p);
        }

        /* Almacenar coeficientes AR en x */
        for (k = 1; k <= global_p; k++) {
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= m; j++) {
                    if (global_diag_ar && i != j) continue;
                    /* Posicion dentro del bloque AR de x, segun el layout que
                       espera shootx: diagonal -> m valores por retardo (i=1..m);
                       completo -> m*m valores por retardo (fila i, columna j). */
                    int pos = global_diag_ar
                              ? (k - 1) * m + i
                              : (k - 1) * m * m + (i - 1) * m + j;
                    int coef_idx = (k - 1) * m + j;
                    x[idx + pos - 1] = coef[i][coef_idx];
                }
            }
        }
        idx += (global_diag_ar ? m : m * m) * global_p;

        /* Calcular residuos del AR y su matriz de covarianza */
        real **resid = matrix(1, nobs, 1, m);
        for (t = 1; t <= nobs; t++) {
            for (i = 1; i <= m; i++) {
                real pred = 0.0;
                for (k = 1; k <= global_p; k++) {
                    if (t > k) {
                        for (j = 1; j <= m; j++) {
                            pred += coef[i][(k-1)*m + j] * datac[t - k][j];
                        }
                    }
                }
                resid[t][i] = datac[t][i] - pred;
            }
        }
        for (i = 1; i <= m; i++) {
            for (j = 1; j <= m; j++) {
                real sum = 0.0;
                for (t = 1; t <= nobs; t++)
                    sum += resid[t][i] * resid[t][j];
                resid_cov[i][j] = sum / nobs;
            }
        }
        free_matrix(resid, 1, nobs, 1, m);
        free_matrix(Z, 1, T, 1, m * global_p);
        free_matrix(y, 1, T, 1, m);
        free_matrix(ZtZ, 1, m * global_p, 1, m * global_p);
        free_ivector(indx, 1, m * global_p);
        free_matrix(coef, 1, m, 1, m * global_p);
    }
    else {
        /* p == 0: usar covarianza muestral de los datos centrados */
        for (i = 1; i <= m; i++) {
            for (j = 1; j <= m; j++) {
                real sum = 0.0;
                for (t = 1; t <= nobs; t++)
                    sum += datac[t][i] * datac[t][j];
                resid_cov[i][j] = sum / nobs;
            }
        }
    }

    /* Convertir a matriz de correlaciÃÂ³n */
    for (i = 1; i <= m; i++) {
        real sd_i = sqrt(resid_cov[i][i]);
        if (sd_i < 1e-12) sd_i = 1.0;
        for (j = 1; j <= m; j++) {
            real sd_j = sqrt(resid_cov[j][j]);
            if (sd_j < 1e-12) sd_j = 1.0;
            resid_corr[i][j] = resid_cov[i][j] / (sd_i * sd_j);
        }
    }
    /* RegularizaciÃÂ³n: diagonal = 1 y garantÃÂ­a de definida positiva */
    for (i = 1; i <= m; i++) {
        resid_corr[i][i] = 1.0;
        /* AÃÂ±adir una pequeÃÂ±a constante si la matriz es casi singular */
        for (j = 1; j <= m; j++) {
            if (i == j) continue;
            if (fabs(resid_corr[i][j]) > 0.9999)
                resid_corr[i][j] *= 0.9999;
        }
    }

    /* 3. ParÃÂ¡metros MA (cero) */
    if (global_q > 0) {
        int n_ma_est = (global_diag_ma ? m : m * m) * global_q;
        for (i = 0; i < n_ma_est; i++)
            x[idx++] = 0.0;
    }

    /* 4. ParÃÂ¡metros de covarianza (basados en la matriz de correlaciÃÂ³n) */
    if (global_diag_cov) {
        for (i = 1; i <= m; i++)
            x[idx++] = 1.0;          /* varianza inicial = 1 */
    } else {
        for (i = 1; i <= m; i++) {
            for (j = 1; j <= i; j++) {
                x[idx++] = resid_corr[i][j];
            }
        }
    }

    /* Liberar memoria */
    free_matrix(resid_corr, 1, m, 1, m);
    free_matrix(resid_cov, 1, m, 1, m);
    free_matrix(datac, 1, nobs, 1, m);
    if (global_include_mean) free_vector(mean_est, 1, m);
}

static void init_diag_varma(real *x, int npar)
{
    int m = nser;
    int p = global_p;
    int q = global_q;
    int i, j, k, t, idx = 1;
    real *mean_est = NULL;

    /* Sample means (if included) */
    if (global_include_mean) {
        mean_est = vector(1, m);
        for (j = 1; j <= m; j++) {
            real sum = 0.0;
            for (t = 1; t <= nobs; t++) sum += datamat[t][j];
            mean_est[j] = sum / nobs;
            x[idx++] = mean_est[j];
        }
    }

    /* Center data */
    real **datac = matrix(1, nobs, 1, m);
    for (t = 1; t <= nobs; t++)
        for (j = 1; j <= m; j++)
            datac[t][j] = datamat[t][j] - (global_include_mean ? mean_est[j] : 0.0);

    /* Arrays for diagonal AR coefficients and residuals */
    real **phi_diag = matrix(1, p, 1, m);  /* phi_diag[k][i] */
    real **resid = matrix(1, nobs, 1, m);

    for (i = 1; i <= m; i++) {
        int T = nobs - p;
        if (T <= 0) {
            for (k = 1; k <= p; k++) phi_diag[k][i] = 0.0;
            for (t = 1; t <= nobs; t++) resid[t][i] = datac[t][i];
            continue;
        }
        /* Build regressor matrix Z (p lags) and vector y */
        real **Z = matrix(1, T, 1, p);
        real *y = vector(1, T);
        for (t = p+1; t <= nobs; t++) {
            int row = t - p;
            for (k = 1; k <= p; k++)
                Z[row][k] = datac[t - k][i];
            y[row] = datac[t][i];
        }

        /* Z'Z */
        real **ZtZ = matrix(1, p, 1, p);
        for (k = 1; k <= p; k++) {
            for (j = 1; j <= p; j++) {
                real sum = 0.0;
                for (t = 1; t <= T; t++) sum += Z[t][k] * Z[t][j];
                ZtZ[k][j] = sum;
            }
        }
        int *indx = ivector(1, p);
        ludcp(ZtZ, p, indx);
        real *Zty = vector(1, p);
        for (k = 1; k <= p; k++) {
            Zty[k] = 0.0;
            for (t = 1; t <= T; t++) Zty[k] += Z[t][k] * y[t];
        }
        lusol(ZtZ, Zty, p, indx);

        for (k = 1; k <= p; k++) phi_diag[k][i] = Zty[k];

        /* Compute residuals for all observations */
        for (t = 1; t <= nobs; t++) {
            if (t <= p) {
                resid[t][i] = datac[t][i];
            } else {
                real pred = 0.0;
                for (k = 1; k <= p; k++) pred += phi_diag[k][i] * datac[t - k][i];
                resid[t][i] = datac[t][i] - pred;
            }
        }

        free_matrix(Z, 1, T, 1, p);
        free_vector(y, 1, T);
        free_matrix(ZtZ, 1, p, 1, p);
        free_ivector(indx, 1, p);
        free_vector(Zty, 1, p);
    }

    /* Store AR coefficients */
    for (k = 1; k <= p; k++)
        for (i = 1; i <= m; i++)
            x[idx++] = phi_diag[k][i];

    /* MA parameters (all zero) */
    for (k = 1; k <= q; k++)
        for (i = 1; i <= m; i++)
            x[idx++] = 0.0;

    /* Compute residual variances and scale them to have average 1 */
    real *var_resid = vector(1, m);
    for (i = 1; i <= m; i++) {
        real sum = 0.0;
        for (t = 1; t <= nobs; t++) sum += resid[t][i] * resid[t][i];
        var_resid[i] = sum / nobs;
    }
    real scale_factor = 0.0;
    for (i = 1; i <= m; i++) scale_factor += var_resid[i];
    scale_factor /= m;
    if (scale_factor < 1e-12) scale_factor = 1.0;
    for (i = 1; i <= m; i++) var_resid[i] /= scale_factor;
    for (i = 1; i <= m; i++) x[idx++] = var_resid[i];
    free_vector(var_resid, 1, m);

    /* Free memory */
    free_matrix(phi_diag, 1, p, 1, m);
    free_matrix(resid, 1, nobs, 1, m);
    free_matrix(datac, 1, nobs, 1, m);
    if (global_include_mean) free_vector(mean_est, 1, m);
}

static void combine_vectors(real *x_full, real *x_diag, int npar_full, int npar_diag)
{
    int m = nser;
    int p = global_p;
    int q = global_q;
    int idx_full = 1, idx_diag = 1;

    /* Means */
    if (global_include_mean) {
        for (int i = 1; i <= m; i++)
            x_full[idx_full++] = x_diag[idx_diag++];
    }

    /* AR */
    for (int k = 1; k <= p; k++) {
        if (global_diag_ar) {
            for (int i = 1; i <= m; i++)
                x_full[idx_full++] = x_diag[idx_diag++];
        } else {
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= m; j++) {
                    if (i == j)
                        x_full[idx_full] = x_diag[idx_diag++];
                    /* else keep original value (already in x_full) */
                    idx_full++;
                }
            }
        }
    }

    /* MA */
    for (int k = 1; k <= q; k++) {
        if (global_diag_ma) {
            for (int i = 1; i <= m; i++)
                x_full[idx_full++] = x_diag[idx_diag++];
        } else {
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= m; j++) {
                    if (i == j)
                        x_full[idx_full] = x_diag[idx_diag++];
                    /* else keep original value (zero in init_varma) */
                    idx_full++;
                }
            }
        }
    }

    /* Covariance */
    if (global_diag_cov) {
        for (int i = 1; i <= m; i++)
            x_full[idx_full++] = x_diag[idx_diag++];
    } else {
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= i; j++) {
                if (i == j)
                    x_full[idx_full] = x_diag[idx_diag++];
                /* else keep original covariance (already scaled) */
                idx_full++;
            }
        }
    }

    /* Consistency check */
    if (idx_full - 1 != npar_full || idx_diag - 1 != npar_diag) {
        printf("Error in vector combination: inconsistent indices (full:%d vs %d, diag:%d vs %d).\n",
               idx_full-1, npar_full, idx_diag-1, npar_diag);
        exit(1);
    }
}

static void hannan_rissanen_diag(real *x, int npar)
{
    int m = nser;
    int p = global_p;
    int q = global_q;
    int i, j, k, t, idx = 1;
    real *mean_est = NULL;

    /* Sample means (if included) */
    if (global_include_mean) {
        mean_est = vector(1, m);
        for (j = 1; j <= m; j++) {
            real sum = 0.0;
            for (t = 1; t <= nobs; t++) sum += datamat[t][j];
            mean_est[j] = sum / nobs;
            x[idx++] = mean_est[j];
        }
    }

    /* Center data */
    real **datac = matrix(1, nobs, 1, m);
    for (t = 1; t <= nobs; t++)
        for (j = 1; j <= m; j++)
            datac[t][j] = datamat[t][j] - (global_include_mean ? mean_est[j] : 0.0);

    /* Storage for diagonal coefficients */
    real **phi_diag   = matrix(1, p, 1, m);   /* phi_diag[k][i] */
    real **theta_diag = matrix(1, q, 1, m);   /* theta_diag[k][i] */

    for (i = 1; i <= m; i++) {
        if (q == 0) {
            /* Pure AR: use OLS as before */
            int T = nobs - p;
            if (T <= 0) {
                for (k = 1; k <= p; k++) phi_diag[k][i] = 0.0;
                continue;
            }
            real **Z = matrix(1, T, 1, p);
            real  *y = vector(1, T);
            for (t = p+1; t <= nobs; t++) {
                int row = t - p;
                for (k = 1; k <= p; k++)
                    Z[row][k] = datac[t - k][i];
                y[row] = datac[t][i];
            }
            real **ZtZ = matrix(1, p, 1, p);
            for (k = 1; k <= p; k++) {
                for (j = 1; j <= p; j++) {
                    real sum = 0.0;
                    for (t = 1; t <= T; t++) sum += Z[t][k] * Z[t][j];
                    ZtZ[k][j] = sum;
                }
            }
            int *indx = ivector(1, p);
            ludcp(ZtZ, p, indx);
            real *Zty = vector(1, p);
            for (k = 1; k <= p; k++) {
                Zty[k] = 0.0;
                for (t = 1; t <= T; t++) Zty[k] += Z[t][k] * y[t];
            }
            lusol(ZtZ, Zty, p, indx);
            for (k = 1; k <= p; k++) phi_diag[k][i] = Zty[k];
            for (k = 1; k <= q; k++) theta_diag[k][i] = 0.0;

            free_matrix(Z, 1, T, 1, p);
            free_vector(y, 1, T);
            free_matrix(ZtZ, 1, p, 1, p);
            free_ivector(indx, 1, p);
            free_vector(Zty, 1, p);
        }
        else {
            /* q > 0: Hannan-Rissanen */
            int maxpq = (p > q ? p : q);
            int L = (int) floor(sqrt((double) nobs));
            if (L < p+q) L = p+q;
            if (L + maxpq >= nobs) L = nobs - maxpq - 1;
            if (L < 1) L = 1;

            /* Step 1: fit AR(L) by OLS to obtain residuals e_t */
            int T_ar = nobs - L;
            if (T_ar <= L) {   /* Not enough data ? fallback to zeros */
                for (k = 1; k <= p; k++) phi_diag[k][i] = 0.0;
                for (k = 1; k <= q; k++) theta_diag[k][i] = 0.0;
                continue;
            }
            real **Z_ar = matrix(1, T_ar, 1, L);
            real  *y_ar = vector(1, T_ar);
            for (t = L+1; t <= nobs; t++) {
                int row = t - L;
                for (k = 1; k <= L; k++)
                    Z_ar[row][k] = datac[t - k][i];
                y_ar[row] = datac[t][i];
            }
            real **ZtZ_ar = matrix(1, L, 1, L);
            for (k = 1; k <= L; k++) {
                for (j = 1; j <= L; j++) {
                    real sum = 0.0;
                    for (t = 1; t <= T_ar; t++) sum += Z_ar[t][k] * Z_ar[t][j];
                    ZtZ_ar[k][j] = sum;
                }
            }
            int *indx_ar = ivector(1, L);
            ludcp(ZtZ_ar, L, indx_ar);
            real *Zty_ar = vector(1, L);
            for (k = 1; k <= L; k++) {
                Zty_ar[k] = 0.0;
                for (t = 1; t <= T_ar; t++) Zty_ar[k] += Z_ar[t][k] * y_ar[t];
            }
            lusol(ZtZ_ar, Zty_ar, L, indx_ar);
            real *ar_coef = vector(1, L);
            for (k = 1; k <= L; k++) ar_coef[k] = Zty_ar[k];

            /* Compute residuals e_hat[t] for t = L+1 .. nobs */
            real *e_hat = vector(1, nobs);
            for (t = 1; t <= L; t++) e_hat[t] = 0.0;  /* pre-sample zeros */
            for (t = L+1; t <= nobs; t++) {
                real pred = 0.0;
                for (k = 1; k <= L; k++) pred += ar_coef[k] * datac[t - k][i];
                e_hat[t] = datac[t][i] - pred;
            }

            /* Step 2: regress y_t on y_{t-1}..y_{t-p} and e_{t-1}..e_{t-q} */
            int start = maxpq + 1;
            if (start < L+2) start = L+2;   /* need e_hat[t-1] available */
            if (start > nobs) {
                /* Not enough data ? fallback */
                for (k = 1; k <= p; k++) phi_diag[k][i] = 0.0;
                for (k = 1; k <= q; k++) theta_diag[k][i] = 0.0;
                free_vector(e_hat, 1, nobs);
                free_vector(ar_coef, 1, L);
                free_matrix(Z_ar, 1, T_ar, 1, L);
                free_vector(y_ar, 1, T_ar);
                free_matrix(ZtZ_ar, 1, L, 1, L);
                free_ivector(indx_ar, 1, L);
                free_vector(Zty_ar, 1, L);
                continue;
            }
            int T_reg = nobs - start + 1;
            real **X = matrix(1, T_reg, 1, p+q);
            real  *y_reg = vector(1, T_reg);
            for (t = start; t <= nobs; t++) {
                int row = t - start + 1;
                /* AR lags */
                for (k = 1; k <= p; k++) X[row][k] = datac[t - k][i];
                /* MA lags using e_hat */
                for (k = 1; k <= q; k++) X[row][p + k] = e_hat[t - k];
                y_reg[row] = datac[t][i];
            }
            /* X'X and X'y */
            real **XtX = matrix(1, p+q, 1, p+q);
            for (k = 1; k <= p+q; k++) {
                for (j = 1; j <= p+q; j++) {
                    real sum = 0.0;
                    for (t = 1; t <= T_reg; t++) sum += X[t][k] * X[t][j];
                    XtX[k][j] = sum;
                }
            }
            real *Xty = vector(1, p+q);
            for (k = 1; k <= p+q; k++) {
                Xty[k] = 0.0;
                for (t = 1; t <= T_reg; t++) Xty[k] += X[t][k] * y_reg[t];
            }
            int *indx_reg = ivector(1, p+q);
            ludcp(XtX, p+q, indx_reg);
            lusol(XtX, Xty, p+q, indx_reg);
            for (k = 1; k <= p; k++) phi_diag[k][i] = Xty[k];
            for (k = 1; k <= q; k++) theta_diag[k][i] = -Xty[p + k];

            /* Free memory for this series */
            free_ivector(indx_reg, 1, p+q);
            free_vector(Xty, 1, p+q);
            free_matrix(XtX, 1, p+q, 1, p+q);
            free_matrix(X, 1, T_reg, 1, p+q);
            free_vector(y_reg, 1, T_reg);
            free_vector(e_hat, 1, nobs);
            free_vector(ar_coef, 1, L);
            free_matrix(Z_ar, 1, T_ar, 1, L);
            free_vector(y_ar, 1, T_ar);
            free_matrix(ZtZ_ar, 1, L, 1, L);
            free_ivector(indx_ar, 1, L);
            free_vector(Zty_ar, 1, L);
        }
    } /* end loop over series */

    /* Compute residuals from the ARMA model for each series to estimate variance */
    real **resid = matrix(1, nobs, 1, m);
    for (i = 1; i <= m; i++) {
        int maxpq = (p > q ? p : q);
        /* Initialize pre-sample residuals to zero */
        for (t = 1; t <= maxpq; t++) resid[t][i] = 0.0;
        for (t = maxpq+1; t <= nobs; t++) {
            real pred = 0.0;
            for (k = 1; k <= p; k++) pred += phi_diag[k][i] * datac[t - k][i];
            for (k = 1; k <= q; k++) pred += theta_diag[k][i] * resid[t - k][i];
            resid[t][i] = datac[t][i] - pred;
        }
    }

    /* Compute variances from t = max(p,q)+1 .. nobs */
    real *var_resid = vector(1, m);
    for (i = 1; i <= m; i++) {
        int maxpq = (p > q ? p : q);
        real sum = 0.0;
        int cnt = 0;
        for (t = maxpq+1; t <= nobs; t++) {
            sum += resid[t][i] * resid[t][i];
            cnt++;
        }
        var_resid[i] = (cnt > 0) ? sum / cnt : 1.0;
    }

    /* Scale variances to have average 1 (as in original) */
    real scale_factor = 0.0;
    for (i = 1; i <= m; i++) scale_factor += var_resid[i];
    scale_factor /= m;
    if (scale_factor < 1e-12) scale_factor = 1.0;
    for (i = 1; i <= m; i++) var_resid[i] /= scale_factor;

    /* Store parameters in x[] in the correct order */
    /* AR */
    for (k = 1; k <= p; k++)
        for (i = 1; i <= m; i++)
            x[idx++] = phi_diag[k][i];
    /* MA */
    for (k = 1; k <= q; k++)
        for (i = 1; i <= m; i++)
            x[idx++] = theta_diag[k][i];
    /* Covariance (diagonal) */
    for (i = 1; i <= m; i++)
        x[idx++] = var_resid[i];

    /* Free remaining memory */
    free_vector(var_resid, 1, m);
    free_matrix(resid, 1, nobs, 1, m);
    free_matrix(phi_diag, 1, p, 1, m);
    free_matrix(theta_diag, 1, q, 1, m);
    free_matrix(datac, 1, nobs, 1, m);
    if (global_include_mean) free_vector(mean_est, 1, m);
}

/* ---- public API ---- */

void drvarma_defaults(DrvarmaModelSpec *spec)
{
    memset(spec, 0, sizeof(*spec));
    spec->method = 1;
    spec->maxits = 500;
    spec->grtol  = 1.0e-7;
    spec->sptol  = 1.0e-7;
}

const char *drvarma_strerror(int ifault)
{
    switch (ifault) {
        case 0: return "ok";
        case 1: return "Q matrix not positive definite";
        case 2: return "unit root in AR";
        case 3: return "AR nonstationary";
        case 4: return "MA noninvertible";
        case 5: return "numerical problem";
        case 6: return "error in shootx";
        default: return "unknown";
    }
}

void drvarma_result_free(DrvarmaResult *r)
{
    if (!r) return;
    free(r->params); free(r->std_errors); free(r->cov_matrix);
    free(r->residuals); free(r->mu); free(r->phi); free(r->theta);
    free(r->sigma); free(r);
}

DrvarmaResult *drvarma_estimate(const DrvarmaModelSpec *spec)
{
    int i, j, t, npar, nrits = 10, ifault = 0, maxits, est_fault;
    real grtol, sptol, xitol;
    struct Tvarma varma1;
    real *x, *dev, **cov;
    DrvarmaResult *r;

    nser = spec->m;  nobs = spec->nobs;
    global_p = spec->p;  global_q = spec->q;
    global_include_mean = spec->include_mean;
    global_diag_ar = spec->diag_ar;
    global_diag_ma = spec->diag_ma;
    global_diag_cov = spec->diag_cov;
    met = spec->method ? spec->method : 1;
    global_twostep = spec->twostep;
    g_estwin = 0;  g_in_est = 0;
    macheps = cmacheps();
    if (!outputv) outputv = fopen("/dev/null", "w");

    datamat = matrix(1, nobs, 1, nser);
    for (t = 1; t <= nobs; t++)
        for (j = 1; j <= nser; j++)
            datamat[t][j] = spec->w[(t - 1) * nser + (j - 1)];

    npar = calc_nparametrs();
    x   = vector(1, npar);
    dev = vector(1, npar);
    cov = matrix(1, npar, 1, npar);

    init_varma(x, npar);
    if (global_twostep && global_q > 0 &&
        (global_diag_ar == 0 || global_diag_ma == 0 || global_diag_cov == 0)) {
        int sa = global_diag_ar, sm = global_diag_ma, sc = global_diag_cov;
        global_diag_ar = global_diag_ma = global_diag_cov = 1;
        int npd = calc_nparametrs();
        real *xd = vector(1, npd);
        hannan_rissanen_diag(xd, npd);
        global_diag_ar = sa; global_diag_ma = sm; global_diag_cov = sc;
        combine_vectors(x, xd, npar, npd);
        free_vector(xd, 1, npd);
    }

    maxits = spec->maxits > 0 ? spec->maxits : 500;
    grtol  = spec->grtol  > 0 ? spec->grtol  : 1.0e-7;
    sptol  = spec->sptol  > 0 ? spec->sptol  : 1.0e-7;
    xitol  = (met == 2) ? -1.0e-3 : 1.0e-3;
    varma1.xitol = xitol;

    shootx(x, &varma1, &ifault, 1, 0);
    est(&shootx, npar, x, dev, cov, maxits, nrits, grtol, sptol,
        xitol, varma1.a, &varma1.sigma2, &varma1.logelf, &ifault);
    est_fault = ifault;
    shootx(x, &varma1, &ifault, 0, 0);

    r = (DrvarmaResult *) calloc(1, sizeof(DrvarmaResult));
    r->ifault = est_fault; r->npar = npar; r->m = nser; r->nresiduals = nobs;
    r->p = global_p; r->q = global_q;
    r->sigma2 = varma1.sigma2; r->logelf = varma1.logelf;
    r->params     = (double *) malloc((size_t)npar * sizeof(double));
    r->std_errors = (double *) malloc((size_t)npar * sizeof(double));
    r->cov_matrix = (double *) malloc((size_t)npar * npar * sizeof(double));
    r->residuals  = (double *) malloc((size_t)nobs * nser * sizeof(double));
    r->mu    = (double *) calloc((size_t)nser, sizeof(double));
    r->phi   = (double *) calloc((size_t)(global_p > 0 ? global_p : 1) * nser * nser, sizeof(double));
    r->theta = (double *) calloc((size_t)(global_q > 0 ? global_q : 1) * nser * nser, sizeof(double));
    r->sigma = (double *) calloc((size_t)nser * nser, sizeof(double));
    for (i = 1; i <= npar; i++) { r->params[i-1] = x[i]; r->std_errors[i-1] = dev[i]; }
    for (i = 1; i <= npar; i++)
        for (j = 1; j <= npar; j++)
            r->cov_matrix[(i-1)*npar + (j-1)] = cov[i][j];
    for (t = 1; t <= nobs; t++)
        for (j = 1; j <= nser; j++)
            r->residuals[(t-1)*nser + (j-1)] = varma1.a[t][j];
    for (i = 1; i <= nser; i++) r->mu[i-1] = varma1.mu[i];
    for (int k = 1; k <= global_p; k++)
        for (i = 1; i <= nser; i++)
            for (j = 1; j <= nser; j++)
                r->phi[((k-1)*nser + (i-1))*nser + (j-1)] = varma1.phi[k][i][j];
    for (int k = 1; k <= global_q; k++)
        for (i = 1; i <= nser; i++)
            for (j = 1; j <= nser; j++)
                r->theta[((k-1)*nser + (i-1))*nser + (j-1)] = varma1.theta[k][i][j];
    for (i = 1; i <= nser; i++)
        for (j = 1; j <= nser; j++)
            r->sigma[(i-1)*nser + (j-1)] = varma1.sigma2 * varma1.qq[i][j];

    shootx(x, &varma1, &ifault, 0, 1);
    free_matrix(cov, 1, npar, 1, npar);
    free_vector(dev, 1, npar);
    free_vector(x, 1, npar);
    free_matrix(datamat, 1, nobs, 1, nser);
    return r;
}

/* ========================================================================= */
/* drvarma_elf -- the exact likelihood at a GIVEN structure.                 */
/*                                                                           */
/* `elf` wants 1-based vectors/matrices/tensors (Numerical-Recipes style);    */
/* callers across the FFI have flat 0-based buffers.  All this does is the    */
/* marshalling, so that a restricted model -- a transfer function, a network  */
/* -- can be scored by the same reference likelihood the estimators use,      */
/* instead of being re-implemented on the other side of the binding.         */
/* ========================================================================= */

int drvarma_elf(int m, int n, int p, int q,
                const double *mu, const double *phi, const double *theta,
                const double *qq, const double *w,
                double sigma2, double delta, int atf,
                double *a_out, double *f1, double *f2, double *logelf)
{
    real  *Mu, **Qq, **W, **A = NULL;
    real ***Phi = NULL, ***Theta = NULL;
    real   lf1 = 0.0, lf2 = 0.0, llog = 0.0;
    int    i, j, k, t, ifault = 0;

    if (m < 1 || n < 1 || mu == NULL || qq == NULL || w == NULL) return 5;

    /* p = q = 0 no lo sobrevive el motor: g = max(p,q) = 0 y por dentro se
       reserva matrix(1, m*g, 1, m*g), degenerada, que acaba en segfault.  Es
       una limitacion PREEXISTENTE -- drvarma_estimate(p=0,q=0) vuelca core
       igual -- y no algo de esta entrada, pero esta puerta no va a entregarle
       un caso que no aguanta.  El puerto de Python (_as311.elf) SI lo resuelve.
       Un VARMA sin AR ni MA es ruido blanco: su verosimilitud es inmediata y no
       necesita este motor.                                                    */
    if (p == 0 && q == 0) return 6;

    /* macheps es un GLOBAL que fija el main de cada programa (y, aqui,
       drvarma_estimate).  Entrando por esta puerta nadie lo habia fijado, asi
       que valia 0 y las tolerancias que dependen de el dejaban de cortar: elf
       no volvia.  Se fija aqui, que es lo que hace todo llamante de elf.      */
    macheps = cmacheps();
    Mu = vector(1, m);
    Qq = matrix(1, m, 1, m);
    W  = matrix(1, n, 1, m);
    A  = matrix(1, n, 1, m);            /* elf writes here when atf != 0 */

    /* Lower bound 0, NOT 1: elf touches lag zero, so the allocation has to
       include it -- exactly as this file already does for armax->phi/theta
       above.  Starting at 1 corrupts the heap ("double free or corruption")
       instead of failing cleanly, which is how this was found.               */
    Phi   = tensor(0, p, 1, m, 1, m);
    Theta = tensor(0, q, 1, m, 1, m);

    for (i = 1; i <= m; i++) Mu[i] = mu[i - 1];
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) Qq[i][j] = qq[(i - 1) * m + (j - 1)];
    for (t = 1; t <= n; t++)
        for (i = 1; i <= m; i++) W[t][i] = w[(t - 1) * m + (i - 1)];

    /* Lag ZERO is Phi_0 = Theta_0 = IDENTITY -- the convention elf expects, and
       what this file already does for armax->phi/theta.  Zero-filling it hands
       elf a singular Phi_0 and the run ends in heap corruption, not in a clean
       ifault.  The caller passes only lags 1..p / 1..q, which is the natural
       thing to have on the other side of the binding.                        */
    for (k = 0; k <= p; k++)
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++)
                Phi[k][i][j] = (k == 0) ? (i == j ? 1.0 : 0.0)
                             : (phi != NULL
                                ? phi[((k - 1) * m + (i - 1)) * m + (j - 1)] : 0.0);
    for (k = 0; k <= q; k++)
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++)
                Theta[k][i][j] = (k == 0) ? (i == j ? 1.0 : 0.0)
                               : (theta != NULL
                                  ? theta[((k - 1) * m + (i - 1)) * m + (j - 1)] : 0.0);

    for (t = 1; t <= n; t++)
        for (i = 1; i <= m; i++) A[t][i] = 0.0;

    elf(m, n, p, q, Mu, Phi, Theta, Qq, W, sigma2, delta, atf, A,
        &lf1, &lf2, &llog, &ifault);

    if (f1)     *f1     = lf1;
    if (f2)     *f2     = lf2;
    if (logelf) *logelf = llog;

    if (atf && a_out != NULL)
        for (t = 1; t <= n; t++)
            for (i = 1; i <= m; i++) a_out[(t - 1) * m + (i - 1)] = A[t][i];

    free_tensor(Theta, 0, q, 1, m, 1, m);
    free_tensor(Phi,   0, p, 1, m, 1, m);
    free_matrix(A,  1, n, 1, m);
    free_matrix(W,  1, n, 1, m);
    free_matrix(Qq, 1, m, 1, m);
    free_vector(Mu, 1, m);

    return ifault;
}
