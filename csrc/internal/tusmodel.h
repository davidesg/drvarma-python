/*
 * tusmodel.h -- el modelo univariante de un .pre, tal como lo deja el lector.
 *
 * Vivia en engines/drtran/include/main.h. Sale aqui porque ya no es de
 * drtran: lo llenan read_fue_pre y lo consumen drtran, su GUI y drvarma, y
 * dos copias de una estructura que un lector llena campo a campo son dos
 * lectores esperando a divergir.
 *
 * El anfitrion lo incluye desde su main.h, DESPUES de definir `real`.
 */

#ifndef ATSW_TUSMODEL_H
#define ATSW_TUSMODEL_H

struct Tusmodel
{
    int NdetVar;                 /* numero de variables deterministas */
    char **detspec;              /* especificacion de cada detvar tal como viene
                                    en el .pre ("cos 2", "step 6 2008", ...).
                                    Se guarda para poder REGENERARLAS en fechas
                                    futuras al prever: son funciones del tiempo. */
    int *Nomega, *Ndelta;        /* ordenes de omega y delta para cada detvar */
    real **Omega, **Delta;       /* coeficientes omega y delta */
    int **Imega, **Ielta;        /* indicadores: 1=estimado, 0=fijo */

    int NumAr1, NumAr2;          /* numero de factores AR regulares y anuales */
    int *p1, *p2;                /* ordenes de cada factor AR */
    real **Ar1, **Ar2;           /* coeficientes AR */
    int **Ia1, **Ia2;            /* flags estimacion */

    int NumMa1, NumMa2;          /* numero de factores MA regulares y anuales */
    int *q1, *q2;                /* ordenes de cada factor MA */
    real **Ma1, **Ma2;           /* coeficientes MA */
    int **Im1, **Im2;            /* flags estimacion */

    int NumAr1f, NumMa1f;        /* numero de factores AR/MA de frecuencia fija */
    int *pfre1, *qfre1;          /* frecuencias de los factores fijos */
    real **Ar1f, **Ma1f;         /* coeficientes AR/MA de frecuencia fija */
    int *Ia1f, *Im1f;            /* flags estimacion */

    int Imu;                     /* flag para la media */
    real mu;                     /* valor de la media */

    real boxlam;                 /* parametro lambda de Box-Cox */
    int nrdiff;                  /* diferencias regulares */
    int nadiff;                  /* diferencias anuales completas */
    int *ifadf;                  /* factores irreducibles de la dif. anual */
    int sper;                    /* periodo estacional (freq) */
    int ornsop;                  /* orden del operador no estacionario */
    real *rnsop;                 /* coeficientes del operador no estacionario */
    char *residuals;             /* nombre de la serie de residuos */
    real cbands;                 /* bandas de confianza para ACF */
};

#endif /* ATSW_TUSMODEL_H */
