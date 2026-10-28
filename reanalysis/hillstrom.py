"""Rerun the 'predict who' vs 'value what happens if' comparison on the Hillstrom
randomized e-mail test. Usage: python hillstrom.py path/to/hillstrom.csv [arm]
arm: 'Womens E-Mail' (default) or 'Mens E-Mail'; compared against 'No E-Mail'.

Customers were randomized, so within ANY group the treated-minus-control difference
is an unbiased estimate of what the e-mail caused for that group."""
import sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

path = sys.argv[1]
arm = sys.argv[2] if len(sys.argv) > 2 else "Womens E-Mail"
df = pd.read_csv(path)
df = df[df.segment.isin([arm, "No E-Mail"])].copy()
df["t"] = (df.segment == arm).astype(int)
X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                   columns=["zip_code", "channel"], dtype=float)
FRAC, SPLITS = 0.25, int(__import__("os").environ.get("SPLITS", 50))

rows = []
dec_resp, dec_upl = np.zeros(10), np.zeros(10)
for seed in range(SPLITS):
    rng = np.random.default_rng(seed)
    test = rng.random(len(df)) < 0.5
    tr, te = ~test, test
    Xtr, Xte = X[tr], X[te]
    t_tr = df.t.values[tr]
    for outcome in ["visit", "spend"]:
        y = df[outcome].values.astype(float)
        Model = HistGradientBoostingClassifier if outcome == "visit" else HistGradientBoostingRegressor
        kw = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, random_state=seed)
        # "predict who": among people who got the e-mail, who will visit / spend?
        # (the response model is the same as the treated-arm model of the T-learner)
        m1 = Model(**kw).fit(Xtr[t_tr == 1], y[tr][t_tr == 1]); resp = m1
        # "value what happens if": T-learner, with minus without
        m0 = Model(**kw).fit(Xtr[t_tr == 0], y[tr][t_tr == 0])
        pred = (lambda m, Z: m.predict_proba(Z)[:, 1]) if outcome == "visit" else (lambda m, Z: m.predict(Z))
        s_resp = pred(resp, Xte)
        s_upl = pred(m1, Xte) - pred(m0, Xte)
        yte, tte = y[te], df.t.values[te]
        n = te.sum(); k = int(FRAC * n)
        def evaluate(score):
            top = np.zeros(n, bool); top[np.argsort(-score)[:k]] = True
            inc = yte[top & (tte == 1)].mean() - yte[top & (tte == 0)].mean()
            base = yte[top & (tte == 0)].mean()
            return inc, base
        r_inc, r_base = evaluate(s_resp)
        u_inc, u_base = evaluate(s_upl)
        all_inc = yte[tte == 1].mean() - yte[tte == 0].mean()
        rand_inc = all_inc
        rows.append(dict(seed=seed, outcome=outcome, resp_inc=r_inc, upl_inc=u_inc, rand_inc=rand_inc,
                         resp_base=r_base, upl_base=u_base, resp_treated_rate=yte[np.isin(np.arange(n), np.argsort(-s_resp)[:k]) & (tte == 1)].mean()))
        if outcome == "visit":
            for name, sc, acc in [("resp", s_resp, dec_resp), ("upl", s_upl, dec_upl)]:
                q = pd.qcut(pd.Series(sc).rank(method="first"), 10, labels=False).values  # 0 = lowest score
                for d in range(10):
                    s = q == 9 - d   # decile 1 = highest score
                    acc[d] += (yte[s & (tte == 1)].mean() - yte[s & (tte == 0)].mean()) / SPLITS

R = pd.DataFrame(rows)
summary = {}
for outcome in ["visit", "spend"]:
    g = R[R.outcome == outcome]
    summary[outcome] = {c: dict(mean=float(g[c].mean()), p10=float(g[c].quantile(.1)), p90=float(g[c].quantile(.9)))
                        for c in ["resp_inc", "upl_inc", "rand_inc", "resp_base", "upl_base", "resp_treated_rate"]}
    summary[outcome]["upl_beats_resp_share"] = float((g.upl_inc > g.resp_inc).mean())
out = dict(arm=arm, n=int(len(df)), n_treated=int(df.t.sum()), target_frac=FRAC, splits=SPLITS,
           overall=dict(visit_t=float(df[df.t == 1].visit.mean()), visit_c=float(df[df.t == 0].visit.mean()),
                        spend_t=float(df[df.t == 1].spend.mean()), spend_c=float(df[df.t == 0].spend.mean())),
           summary=summary, deciles_visit=dict(resp=dec_resp.tolist(), upl=dec_upl.tolist()))
json.dump(out, open("hillstrom_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
