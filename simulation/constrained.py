"""Constrained policy: maximize predicted incremental value subject to a borrower-protection limit.

The article's operating model says: maximize incremental risk-adjusted value *subject to* explicit
customer constraints. This script tests that directly in the base case of sim2.py.

From the same randomized pilot used by the incremental-value policy, the lender also learns how the
increase changes each customer's probability of default (a T-learner on observed defaults, which in
practice would take months to observe). It then:
  - drops every account whose predicted increase in default probability exceeds a cap, and
  - gives the increase to up to 200,000 remaining accounts with the highest predicted value
    (never to an account whose predicted value is negative).

Reports lender value and true additional defaults for each cap, alongside the unconstrained policies.
Usage: python constrained.py [seed ...]   (default seeds 7 11 23 42) -> constrained_results.json
"""
import json, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sim2 import (BASE, N, K, PILOT_N, HIST_N, X, population, truth,
                  HistGradientBoostingClassifier, HistGradientBoostingRegressor)

CAPS = [None, 0.03, 0.02, 0.01, 0.005]   # max predicted increase in default probability (None = no cap)


def run(seed):
    p = dict(BASE); rng = np.random.default_rng(seed)
    pop = population(rng, N, p, 0.0)
    pd0, pd1, v0, v1, _ = truth(rng, pop, p)
    ite = (v1 - v0).values; dpd = (pd1 - pd0).values

    # risk model and expected-profit rule, exactly as in sim2.py
    hist = population(rng, HIST_N, p, 0.0)
    h_pd0, _, _, _, h_take = truth(rng, hist, p)
    hist["y"] = (rng.random(HIST_N) < h_pd0).astype(int)
    risk = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(hist[X], hist["y"])
    p_def = risk.predict_proba(pop[X])[:, 1]
    obs_take = h_take + rng.normal(0, 0.05, HIST_N)
    hr = risk.predict_proba(hist[X])[:, 1]; safest = hr <= np.quantile(hr, 0.2)
    _ = rng.random(HIST_N)  # same draw sim2.py uses for its 'generous' benchmark, keeps random streams aligned
    tm = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(hist.loc[safest, X], obs_take[safest])
    t_hat = np.clip(tm.predict(pop[X]), 0, 1); C = p["CLI"]
    ep = (C * t_hat * p["NET_YIELD"] * p["LIFE"] * (1 - p_def)
          - p_def * p["LGD"] * 1.15 * (C * t_hat + C * 0.6 * (1 - t_hat)) - p["COST"])

    # randomized pilot: value outcome (as in sim2.py) plus observed default
    pilot = population(rng, PILOT_N, p, 0.0)
    ppd0, ppd1, pv0, pv1, _ = truth(rng, pilot, p)
    t = (rng.random(PILOT_N) < 0.5).astype(int)
    pilot["y"] = np.where(t == 1, pv1, pv0) + rng.normal(0, 25, PILOT_N)
    pilot["d"] = (rng.random(PILOT_N) < np.where(t == 1, ppd1, ppd0)).astype(int)
    kw = dict(random_state=seed)
    m1 = HistGradientBoostingRegressor(max_iter=300, **kw).fit(pilot.loc[t == 1, X], pilot.loc[t == 1, "y"])
    m0 = HistGradientBoostingRegressor(max_iter=300, **kw).fit(pilot.loc[t == 0, X], pilot.loc[t == 0, "y"])
    upl = m1.predict(pop[X]) - m0.predict(pop[X])
    d1 = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=200, **kw).fit(pilot.loc[t == 1, X], pilot.loc[t == 1, "d"])
    d0 = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=200, **kw).fit(pilot.loc[t == 0, X], pilot.loc[t == 0, "d"])
    dpd_hat = d1.predict_proba(pop[X])[:, 1] - d0.predict_proba(pop[X])[:, 1]

    def score(idx, name):
        return dict(policy=name, treated=int(len(idx)), value_m=float(ite[idx].sum() / 1e6),
                    extra_defaults=float(dpd[idx].sum()))

    out = [score(np.argsort(p_def)[:K], "Lowest predicted risk"),
           score(np.argsort(-ep)[:K], "Expected-profit rule"),
           score(np.argsort(-ite)[:K], "Perfect hindsight")]
    for cap in CAPS:
        ok = (upl > 0) if cap is None else (upl > 0) & (dpd_hat <= cap)
        cand = np.where(ok)[0]
        idx = cand[np.argsort(-upl[cand])][:K]
        name = "Incremental value, no cap (positive predicted value only)" if cap is None else \
               f"Incremental value, predicted default increase <= {cap * 100:.1f} pts"
        out.append(score(idx, name))
    # the unconstrained top-200,000 policy reported in the article
    out.append(score(np.argsort(-upl)[:K], "Incremental value, top 200,000 (article)"))
    for r in out:
        r["seed"] = seed
    return out


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [7, 11, 23, 42]
    with ProcessPoolExecutor() as ex:
        res = [r for rs in ex.map(run, seeds) for r in rs]
    json.dump(res, open("constrained_results.json", "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(res)
    print(df.groupby("policy", sort=False)[["treated", "value_m", "extra_defaults"]].mean().round(2).to_string())
