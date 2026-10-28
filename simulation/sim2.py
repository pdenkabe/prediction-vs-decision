"""Revised simulation for the HBR resubmission.

Extends sim.py (same data-generating process and default parameters) with:
  1. A stronger, non-causal benchmark: expected profit of the increase computed from a
     default model and a usage model, the way a sophisticated issuer would do it without
     a randomized test. Two versions:
       - "observational": usage learned from past increases that went to the safest 20%
         (what an issuer's history actually contains)
       - "generous": usage learned from a random sample of past increases (an upper bound
         on what a non-causal approach can do)
  2. Gross and net-of-pilot value, with the pilot's capacity counted against cycle 1.
  3. Sensitivity to funding cost / yield, loss severity, utilization, hidden fragility,
     and a macro downturn (including one where models are trained in good times and
     deployed in a downturn).

Usage: python sim2.py  -> writes results_v2.json
"""
import json, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

N, K, PILOT_N, HIST_N = 1_000_000, 200_000, 60_000, 200_000
X = ["score", "util", "revolver", "tenure", "limit"]

BASE = dict(CLI=500.0, NET_YIELD=0.20, LIFE=1.8, LGD=0.90, COST=4.0,
            util_shift=0.0,       # added to utilization (customers more/less credit-hungry)
            frag_pd=0.45,         # how much hidden fragility raises baseline default
            frag_cli=0.9,         # how much fragility raises default when given more line
            macro=0.0,            # downturn: shifts everyone's hidden stress up
            macro_deploy=None)    # if set, models trained at `macro`, deployed at this level


def population(rng, n, p, macro):
    score = np.clip(rng.normal(640, 40, n), 540, 760)
    util = np.clip(rng.beta(2.0, 2.4, n) - (score - 640) / 260 + p["util_shift"], 0.01, 0.99)
    revolver = rng.random(n) < (0.25 + 0.6 * util)
    tenure = rng.integers(6, 60, n)
    limit = np.round(rng.normal(900, 250, n).clip(300, 2000), -1)
    stress = rng.normal(macro, 1, n)
    return pd.DataFrame(dict(score=score, util=util, revolver=revolver.astype(int),
                             tenure=tenure, limit=limit, stress=stress))


def truth(rng, d, p):
    z0 = -3.1 - 0.026 * (d.score - 640) + 1.1 * d.util + p["frag_pd"] * d.stress - 0.01 * (d.tenure - 30)
    pd0 = 1 / (1 + np.exp(-z0))
    dz = p["frag_cli"] * np.clip(d.stress - 0.3, 0, None) * d.util + 0.04
    pd1 = 1 / (1 + np.exp(-(z0 + dz)))
    bal0 = d.limit * d.util
    take = np.clip(0.03 + 0.8 * d.revolver * d.util ** 1.3 + 0.1 * rng.normal(0, 1, len(d)) * d.util, 0, 0.95)
    bal1 = bal0 + p["CLI"] * take
    y, L = p["NET_YIELD"], p["LIFE"]
    v0 = bal0 * y * L * (1 - pd0) - pd0 * p["LGD"] * bal0 * 1.15
    v1 = bal1 * y * L * (1 - pd1) - pd1 * p["LGD"] * (bal1 + p["CLI"] * 0.6 * (1 - take)) * 1.15 - p["COST"]
    return pd0, pd1, v0, v1, take


def run(args):
    over, seed = args
    p = {**BASE, **over}
    rng = np.random.default_rng(seed)
    m_train = p["macro"]
    m_dep = p["macro_deploy"] if p["macro_deploy"] is not None else p["macro"]

    # portfolio the decision is made on (deployment conditions)
    pop = population(rng, N, p, m_dep)
    pd0, pd1, v0, v1, _ = truth(rng, pop, p)
    ite = (v1 - v0).values

    # history: no increases, used for the default model
    hist = population(rng, HIST_N, p, m_train)
    h_pd0, _, _, _, h_take = truth(rng, hist, p)
    hist["y"] = (rng.random(HIST_N) < h_pd0).astype(int)
    risk = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(hist[X], hist["y"])
    p_def = risk.predict_proba(pop[X])[:, 1]

    # usage model used by the older "highest predicted usage" policy (as in sim.py)
    use_m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(hist[X], hist.util * hist.revolver)
    p_use = use_m.predict(pop[X])

    # --- stronger non-causal benchmark: expected profit of the increase -------------------
    # Issuer estimates take-up of new line from past increases, and assumes default risk
    # is unchanged by the increase (the standard non-causal assumption).
    obs_take = h_take + rng.normal(0, 0.05, HIST_N)  # observed take-up, measured with noise
    def ep_policy(train_mask):
        tm = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(
            hist.loc[train_mask, X], obs_take[train_mask])
        t_hat = np.clip(tm.predict(pop[X]), 0, 1)
        y, L, C = p["NET_YIELD"], p["LIFE"], p["CLI"]
        rev = C * t_hat * y * L * (1 - p_def)
        loss = p_def * p["LGD"] * 1.15 * (C * t_hat + C * 0.6 * (1 - t_hat))
        return rev - loss - p["COST"]
    hist_risk = risk.predict_proba(hist[X])[:, 1]
    safest = hist_risk <= np.quantile(hist_risk, 0.2)            # past increases went to safest 20%
    rand20 = rng.random(HIST_N) < 0.2                              # generous: random past increases
    ep_obs = ep_policy(safest)
    ep_gen = ep_policy(rand20)

    # --- randomized pilot + T-learner ----------------------------------------------------
    pilot = population(rng, PILOT_N, p, m_train)
    _, _, pv0, pv1, _ = truth(rng, pilot, p)
    t = (rng.random(PILOT_N) < 0.5).astype(int)
    pilot["y"] = np.where(t == 1, pv1, pv0) + rng.normal(0, 25, PILOT_N)
    pilot_cost = float((pv1 - pv0)[t == 1].sum())   # value lost/gained on the 30k randomly treated
    m1 = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(pilot.loc[t == 1, X], pilot.loc[t == 1, "y"])
    m0 = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(pilot.loc[t == 0, X], pilot.loc[t == 0, "y"])
    upl = m1.predict(pop[X]) - m0.predict(pop[X])

    def top(score, k=K):
        return float(ite[np.argsort(-score)[:k]].sum())

    n_treated_pilot = int(t.sum())
    out = dict(
        seed=seed, scenario=over,
        random=float(ite[rng.permutation(N)[:K]].sum()),
        lowest_risk=top(-p_def),
        highest_usage=top(p_use),
        expected_profit_observational=top(ep_obs),
        expected_profit_generous=top(ep_gen),
        incremental_value=top(upl),
        # cycle-1 capacity reconciliation: pilot uses 30k of the 200k slots
        incremental_value_cycle1_capacity=top(upl, K - n_treated_pilot),
        ceiling=top(ite),
        pilot_cost=pilot_cost,
        pilot_treated=n_treated_pilot,
        mean_pd=float(pd0.mean()),
        mean_ite=float(ite.mean()),
    )
    return out


SCENARIOS = {
    "Base case": {},
    "Lower margin (net yield 15%, higher funding cost)": dict(NET_YIELD=0.15),
    "Higher margin (net yield 25%)": dict(NET_YIELD=0.25),
    "Lower loss severity (LGD 75%)": dict(LGD=0.75),
    "Higher loss severity (LGD 100%)": dict(LGD=1.00),
    "Less credit-hungry customers (util -0.10)": dict(util_shift=-0.10),
    "More credit-hungry customers (util +0.10)": dict(util_shift=0.10),
    "Weaker hidden fragility (half)": dict(frag_pd=0.225, frag_cli=0.45),
    "Stronger hidden fragility (1.5x)": dict(frag_pd=0.675, frag_cli=1.35),
    "Downturn (stress +0.5)": dict(macro=0.5),
    "Models built in good times, deployed in downturn": dict(macro=0.0, macro_deploy=0.5),
}

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [7, 11, 23, 42]
    jobs = [(over, s) for over in SCENARIOS.values() for s in seeds]
    names = [n for n in SCENARIOS for _ in seeds]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs))
    for n, r in zip(names, res):
        r["name"] = n
    json.dump(res, open("results_v2.json", "w"), indent=1)
    df = pd.DataFrame(res)
    cols = ["random", "lowest_risk", "highest_usage", "expected_profit_observational",
            "expected_profit_generous", "incremental_value", "incremental_value_cycle1_capacity",
            "ceiling", "pilot_cost"]
    summ = (df.groupby("name", sort=False)[cols].agg(["mean", "min", "max"]) / 1e6).round(2)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(summ.xs("mean", axis=1, level=1))
    print()
    print(summ.xs("min", axis=1, level=1)[["expected_profit_generous", "incremental_value"]])
    print(summ.xs("max", axis=1, level=1)[["expected_profit_generous", "incremental_value"]])
