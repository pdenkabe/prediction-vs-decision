"""Simulated card portfolio: who should get a $500 credit line increase?
Ground truth potential outcomes are known (simulation), models are learned from a
randomized pilot, and targeting policies are evaluated against the truth."""
import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(7)
N = 1_000_000
CLI = 500.0
NET_YIELD = 0.20      # APR revenue net of cost of funds, per $ of balance per year
LIFE = 1.8            # discounted years of balance life
LGD = 0.90
COST = 4.0            # servicing/notice cost per treated account

def population(n):
    score = np.clip(rng.normal(640, 40, n), 540, 760)
    util = np.clip(rng.beta(2.0, 2.4, n) - (score - 640) / 260, 0.01, 0.99)
    revolver = rng.random(n) < (0.25 + 0.6 * util)
    tenure = rng.integers(6, 60, n)
    limit = np.round(rng.normal(900, 250, n).clip(300, 2000), -1)
    stress = rng.normal(0, 1, n)   # hidden: financial fragility, partly reflected in score
    return pd.DataFrame(dict(score=score, util=util, revolver=revolver.astype(int),
                             tenure=tenure, limit=limit, stress=stress))

def truth(d):
    z0 = -3.1 - 0.026 * (d.score - 640) + 1.1 * d.util + 0.45 * d.stress - 0.01 * (d.tenure - 30)
    pd0 = 1 / (1 + np.exp(-z0))
    # extra default risk from more open-to-buy: high for fragile, high-utilization customers
    dz = 0.9 * np.clip(d.stress - 0.3, 0, None) * d.util + 0.04
    pd1 = 1 / (1 + np.exp(-(z0 + dz)))
    bal0 = d.limit * d.util
    # share of new line actually used: credit-constrained revolvers use it; transactors don't
    take = np.clip(0.03 + 0.8 * d.revolver * d.util ** 1.3 + 0.1 * rng.normal(0, 1, len(d)) * d.util, 0, 0.95)
    bal1 = bal0 + CLI * take
    rev0 = bal0 * NET_YIELD * LIFE * (1 - pd0)
    rev1 = bal1 * NET_YIELD * LIFE * (1 - pd1)
    loss0 = pd0 * LGD * bal0 * 1.15
    loss1 = pd1 * LGD * (bal1 + CLI * 0.6 * (1 - take)) * 1.15  # drawdown before default
    v0 = rev0 - loss0
    v1 = rev1 - loss1 - COST
    return pd0, pd1, v0, v1

pop = population(N)
pd0, pd1, v0, v1 = truth(pop)
pop["pd0"], pop["ite"] = pd0, v1 - v0
pop["default0"] = (rng.random(N) < pd0).astype(int)
pop["take_obs"] = 0

X = ["score", "util", "revolver", "tenure", "limit"]  # stress is not observed

# historical data for the risk model (no treatment)
hist = population(200_000); h_pd0, _, _, _ = truth(hist)
hist["y"] = (rng.random(len(hist)) < h_pd0).astype(int)
risk = HistGradientBoostingClassifier(max_iter=200).fit(hist[X], hist["y"])
pop["p_default"] = risk.predict_proba(pop[X])[:, 1]
auc = roc_auc_score(pop["default0"], pop["p_default"])

# propensity-to-use model (who will use more credit) - a typical "response" model
pilot = population(60_000)
p_pd0, p_pd1, p_v0, p_v1 = truth(pilot)
t = (rng.random(len(pilot)) < 0.5).astype(int)
noise = rng.normal(0, 25, len(pilot))   # realized value is noisy
pilot["y"] = np.where(t == 1, p_v1, p_v0) + noise
pilot["t"] = t
# response: observed balance growth among treated
resp = HistGradientBoostingRegressor(max_iter=200).fit(pilot[X], pilot["util"] * pilot["revolver"])
pop["p_use"] = resp.predict(pop[X])

# T-learner on randomized pilot: value with and without the action
m1 = HistGradientBoostingRegressor(max_iter=300).fit(pilot.loc[t == 1, X], pilot.loc[t == 1, "y"])
m0 = HistGradientBoostingRegressor(max_iter=300).fit(pilot.loc[t == 0, X], pilot.loc[t == 0, "y"])
pop["uplift_hat"] = m1.predict(pop[X]) - m0.predict(pop[X])

K = 200_000
def evaluate(order, name):
    idx = order[:K]
    sel = pop.iloc[idx]
    return dict(policy=name, treated=K, incr_value=float(sel.ite.sum()),
                per_acct=float(sel.ite.mean()), share_negative=float((sel.ite < 0).mean()),
                avg_pd=float(sel.pd0.mean()))

res = [
    evaluate(rng.permutation(N), "Random 200,000"),
    evaluate(np.argsort(pop.p_default.values), "Lowest predicted risk"),
    evaluate(np.argsort(-pop.p_use.values), "Highest predicted usage"),
    evaluate(np.argsort(-pop.uplift_hat.values), "Highest predicted incremental value"),
    evaluate(np.argsort(-pop.ite.values), "Perfect hindsight (ceiling)"),
]
# value policy without a fixed quota: treat everyone with positive predicted uplift
pos = pop.uplift_hat > 0
res.append(dict(policy="Value policy, no quota (all with predicted gain > 0)", treated=int(pos.sum()),
                incr_value=float(pop.ite[pos].sum()), per_acct=float(pop.ite[pos].mean()),
                share_negative=float((pop.ite[pos] < 0).mean()), avg_pd=float(pop.pd0[pos].mean())))
everyone = dict(policy="Everyone", treated=N, incr_value=float(pop.ite.sum()), per_acct=float(pop.ite.mean()),
                share_negative=float((pop.ite < 0).mean()), avg_pd=float(pop.pd0.mean()))
res.append(everyone)

# value by risk decile (1 = safest)
pop["risk_decile"] = pd.qcut(pop.p_default.rank(method="first"), 10, labels=range(1, 11)).astype(int)
dec = pop.groupby("risk_decile").agg(avg_ite=("ite", "mean"), share_pos=("ite", lambda s: (s > 0).mean()),
                                     pd=("pd0", "mean")).reset_index()

# four customer types
used = (pop.ite > 0)
safe = pop.p_default < pop.p_default.median()
seg = pd.DataFrame({
    "type": np.select([safe & ~used, safe & used, ~safe & used, ~safe & ~used],
                      ["Safe but indifferent", "Safe and responsive", "Riskier but responsive", "Riskier and harmed"], default="")})
seg["ite"] = pop.ite
segt = seg.groupby("type").agg(n=("ite", "size"), avg=("ite", "mean"), total=("ite", "sum")).reset_index()

# rank correlation between risk score and value
rho = pd.Series(pop.p_default).corr(pop.ite, method="spearman")
rho_use = pd.Series(pop.p_use).corr(pop.ite, method="spearman")
rho_hat = pd.Series(pop.uplift_hat).corr(pop.ite, method="spearman")

# overlap: how many of the lowest-risk 200k are in the top-value 200k
top_val = set(np.argsort(-pop.ite.values)[:K]); low_risk = set(np.argsort(pop.p_default.values)[:K])
overlap = len(top_val & low_risk) / K

out = dict(auc=auc, rho_risk_value=rho, rho_use_value=rho_use, rho_hat_value=rho_hat, overlap=overlap,
           results=res, deciles=dec.to_dict("records"), segments=segt.to_dict("records"),
           pop_mean_pd=float(pop.pd0.mean()), pilot_n=len(pilot))
json.dump(out, open("results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
