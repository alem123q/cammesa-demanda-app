"""Exploratory: does the real utilities price (regional CPI housing & utilities / CPI general) explain demand?

Year-over-year log change of demand vs. weather (degree-day differences), EMAE and the real price change
lagged 0-12 months. Run from the repository root after analysis/ext_01_ridge_levels.py.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

p = pd.read_csv("data/external/prices_monthly_regions.csv", dtype={"indice_tiempo": str})
P = p.set_index(["region", "indice_tiempo"])["relative_price"]
D = pd.read_csv("analysis/tabla_ext.csv")
D["target"] = np.log(D.y / D.lag12)
D["d_hdd"], D["d_cdd"] = D.hdd - D.hdd12, D.cdd - D.cdd12
sh = lambda f, k: str(pd.Period(f, "M") - k)
rp = lambda r, f, k: P.get((r, sh(f, k)), np.nan)
for L in (0, 3, 6, 9, 12):
    D[f"rp_l{L}"] = [np.log(rp(r, f, L) / rp(r, f, L + 12)) for r, f in zip(D.region, D.fecha)]
base = ["d_hdd", "d_cdd", "emae_yoy"]
for L in (0, 3, 6, 9, 12):
    c = f"rp_l{L}"
    E = D.dropna(subset=[c])
    m, m0 = LinearRegression().fit(E[base + [c]], E.target), LinearRegression().fit(E[base], E.target)
    print(c, len(E), E.fecha.min(), "coef %.3f" % m.coef_[-1],
          "R2 %.3f -> %.3f" % (m0.score(E[base], E.target), m.score(E[base + [c]], E.target)))
c = "rp_l6"
E, cols = D.dropna(subset=[c]), base + [c]
m = LinearRegression().fit(E[cols], E.target)
print("lag-6 model", dict(zip(cols, m.coef_.round(4))))
for name, a, b in [("drop Sep18-Aug19", "2018-09", "2019-08"), ("recovery Sep19-Feb20", "2019-09", "2020-02")]:
    Q = E[(E.fecha >= a) & (E.fecha <= b)]
    w = Q.lag12 / Q.lag12.sum()
    real = (w * Q.target).sum()
    contrib = {k: round(100 * (w * Q[k] * m.coef_[i]).sum(), 1) for i, k in enumerate(cols)}
    print(name, "actual %.1f%%" % (100 * real), contrib,
          "rest %.1f" % (100 * real - sum(contrib.values()) - 100 * m.intercept_))
