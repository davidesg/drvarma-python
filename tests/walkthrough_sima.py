"""Recorrido de sima de punta a punta, por la superficie MCP y sin atajos.

Es la condicion para publicar: que un analista pueda hacer el ciclo entero
-cargar, caracterizar, identificar, estimar, diagnosticar, IRF/FEVD, prever y
exportar- sin salirse de las tools. Cada paso se ejecuta y se comprueba algo
concreto de su salida; no basta con que no lance excepcion.
"""
import json
import sys

import numpy as np

from drvarma import mcp_server as S

OK, FAIL = [], []


def step(label, fn, check=None):
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        FAIL.append((label, "EXCEPCION: %s" % str(e)[:120]))
        return None
    if check:
        try:
            why = check(out)
        except Exception as e:  # noqa: BLE001
            why = "el check reventó: %s" % str(e)[:80]
        if why:
            FAIL.append((label, why))
            return out
    OK.append(label)
    return out


def build_data(n=240, s=12, seed=17):
    """VAR(1) bivariante con estacionalidad, en NIVELES positivos.

    Empieza en el subperiodo 2 a proposito: es la fase que rompia la
    desestacionalizacion, asi que el recorrido pasa por el caso dificil.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.0], [0.5, 0.8]]).T
    w = np.zeros((n, 2))
    P = np.array([[0.55, 0.30], [-0.10, 0.45]])
    for k in range(1, n):
        w[k] = P @ w[k - 1] + e[k]
    seas = np.column_stack([0.8 * np.sin(2 * np.pi * t / s),
                            0.3 * np.cos(2 * np.pi * t / s)])
    return 100.0 * np.exp(np.cumsum(0.004 * w, 0) + 0.02 * seas)


if __name__ == "__main__":
    data = build_data()

    step("load_data",
         lambda: S.load_data("PT", values_json=repr(data.tolist()), freq=12,
                             start_year=2005, start_period=2,
                             series_names="IPC,WTI"),
         lambda o: None if "240" in o else "no confirma las 240 obs")

    step("series_info", lambda: S.series_info("PT"),
         lambda o: None if "IPC" in o else "no nombra las series")

    step("characterize_series", lambda: S.characterize_series("PT"),
         lambda o: None if ("Consenso" in o and "Comprobación de la desestacionalización" in o
                            and "|" in o) else "falta el consenso o la post-condición")

    step("characterize: la comprobación NO marca fallo",
         lambda: S.characterize_series("PT"),
         lambda o: "la desestacionalización se reporta rota" if "🛑" in o else None)

    step("cross_correlation_matrices", lambda: S.cross_correlation_matrices("PT"),
         lambda o: None if "lag 1" in o else "sin tabla de lags")

    step("partial_autoregression_matrices",
         lambda: S.partial_autoregression_matrices("PT"),
         lambda o: None if "lag" in o.lower() else "sin salida Tiao-Box")

    step("identify_varma_order (alcanza VAR)",
         lambda: S.identify_varma_order("PT"),
         lambda o: None if any(f"({p},0)" in o or f"({p}, 0)" in o
                               for p in (1, 2)) else
                   "el techo sigue sin alcanzar un VAR puro")

    step("confirm_and_estimate", lambda: S.confirm_and_estimate("PT", p=1, q=0),
         lambda o: None if "log" in o.lower() else "sin verosimilitud")

    step("estimación: reporta la convergencia",
         lambda: S.confirm_and_estimate("PT", p=1, q=0),
         lambda o: None if ("termcode" in o.lower() or "converg" in o.lower())
                   else "no dice por que paro el optimizador")

    step("diagnose", lambda: S.diagnose("PT"),
         lambda o: None if "Hosking" in o else "sin Hosking")

    step("impulse_response CON bandas",
         lambda: S.impulse_response("PT", horizon=8),
         lambda o: None if "[" in o and "Bandas 95" in o else "sin bandas")

    step("variance_decomposition CON bandas",
         lambda: S.variance_decomposition("PT", horizon=12),
         lambda o: None if "[" in o and "Bandas 95" in o else "sin bandas")

    step("generate_forecast", lambda: S.generate_forecast("PT", horizon=6),
         lambda o: None if "IC 95" in o else "sin intervalo")

    def _check_export(o):
        d = json.loads(o)
        if "residuals" not in d or "params" not in d:
            return "faltan residuos o parametros"
        if np.asarray(d["residuals"]).ndim != 2:
            return "los residuos no son una matriz"
        if d["termcode"] < 1:
            return "termcode=%s: la diagnosis sigue inerte" % d["termcode"]
        if d["params"]["std_errors"] is None:
            return "sin errores estandar"
        return None

    step("export_fit (residuos + params + termcode)",
         lambda: S.export_fit("PT"), _check_export)

    # el aviso de cointegracion, sobre datos que SI la tienen
    tr = np.cumsum(np.random.default_rng(5).normal(0, 1, 240))
    coint = np.column_stack([100 + tr + np.random.default_rng(6).normal(0, .4, 240),
                             50 + 0.6 * tr + np.random.default_rng(7).normal(0, .4, 240)])
    step("aviso de cointegración (datos cointegrados)",
         lambda: (S.load_data("CO", values_json=repr(coint.tolist()), freq=12,
                              start_year=2000, start_period=1, series_names="A,B"),
                  S.characterize_series("CO"))[1],
         lambda o: None if "COINTEGRACIÓN" in o else "no avisa")

    print("\n%d pasos OK" % len(OK))
    for lab in OK:
        print("   ok   %s" % lab)
    if FAIL:
        print("\n%d FALLOS:" % len(FAIL))
        for lab, why in FAIL:
            print("   FALLO  %-44s %s" % (lab, why))
    sys.exit(1 if FAIL else 0)
