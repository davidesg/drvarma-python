"""BUG-0015 — the real size of Hosking's Q, with df = m²s and with m²(s-p-q).

Simulates a bivariate VAR(1) and VARMA(1,1), fits the true model with the C
engine and counts how often Q rejects at the nominal 5%.

    python tools/hosking_size.py
"""
import numpy as np, warnings
from scipy.stats import chi2
from drvarma._engine import estimate_w
from drvarma.diagnostics import hosking_q
rng = np.random.default_rng(7)
def sim(n, phi, theta):
    m = phi.shape[0]; a = rng.standard_normal((n + 100, m)); w = np.zeros_like(a)
    for t in range(1, len(a)):
        w[t] = phi @ w[t-1] + a[t] - theta @ a[t-1]
    return w[100:]
cases = {"VAR(1)": (np.array([[.5,.2],[.1,.4]]), np.zeros((2,2)), 1, 0),
         "VARMA(1,1)": (np.array([[.5,.2],[.1,.4]]), np.array([[.4,0],[.2,.3]]), 1, 1)}
R = 400
for name,(phi,th,p,q) in cases.items():
    for n, s in [(150, 6), (300, 12)]:
        Qs = []
        for _ in range(R):
            w = sim(n, phi, th)
            r = estimate_w(w, p, q)
            Q, df, _ = hosking_q(r["residuals"], s); Qs.append(Q)
        Qs = np.array(Qs); m = 2
        d0, d1 = m*m*s, m*m*(s-p-q)
        print(f"{name:10} n={n} s={s}: mean Q {Qs.mean():6.2f} | df m²s={d0}: size {np.mean(Qs>chi2.ppf(.95,d0)):.3f} | df m²(s-p-q)={d1}: size {np.mean(Qs>chi2.ppf(.95,d1)):.3f}")
