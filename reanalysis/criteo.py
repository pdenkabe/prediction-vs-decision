"""Same comparison as hillstrom.py on a 10% random sample of the Criteo uplift dataset
(randomized ad-incrementality tests). Usage: python criteo.py path.csv.gz
Policies, each targeting the top FRAC of a held-out half:
  response = highest predicted outcome if treated ("who will visit")
  uplift   = T-learner: predicted outcome if treated minus if not
Evaluation uses the randomized treatment/control split inside the targeted group."""
import sys, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv(sys.argv[1])
F = [f"f{i}" for i in range(12)]
SPLITS = int(os.environ.get("SPLITS", 10))
FRACS = [0.10, 0.25]
X = df[F].values.astype(np.float32); T = df.treatment.values

res = []
dec = {o: {"resp": np.zeros(10), "upl": np.zeros(10)} for o in ["visit", "conversion"]}
for seed in range(SPLITS):
    rng = np.random.default_rng(100 + seed)
    te = rng.random(len(df)) < 0.5; tr = ~te
    for outcome in ["visit", "conversion"]:
        y = df[outcome].values
        kw = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, random_state=seed)
        m1 = HistGradientBoostingClassifier(**kw).fit(X[tr & (T == 1)], y[tr & (T == 1)])
        m0 = HistGradientBoostingClassifier(**kw).fit(X[tr & (T == 0)], y[tr & (T == 0)])
        p1 = m1.predict_proba(X[te])[:, 1]; p0 = m0.predict_proba(X[te])[:, 1]
        s_resp, s_upl = p1, p1 - p0
        yt, tt = y[te], T[te]; n = te.sum()
        overall = yt[tt == 1].mean() - yt[tt == 0].mean()
        for frac in FRACS:
            k = int(frac * n)
            for name, sc in [("resp", s_resp), ("upl", s_upl)]:
                top = np.zeros(n, bool); top[np.argsort(-sc)[:k]] = True
                rt, rc = yt[top & (tt == 1)].mean(), yt[top & (tt == 0)].mean()
                res.append(dict(seed=seed, outcome=outcome, frac=frac, policy=name, treated_rate=rt,
                                control_rate=rc, inc=rt - rc, overall=overall,
                                captured=(rt - rc) * k / (overall * n)))  # share of all incremental outcomes
        for name, sc in [("resp", s_resp), ("upl", s_upl)]:
            q = pd.qcut(pd.Series(sc).rank(method="first"), 10, labels=False).values
            for d in range(10):
                s = q == 9 - d
                dec[outcome][name][d] += (yt[s & (tt == 1)].mean() - yt[s & (tt == 0)].mean()) / SPLITS

R = pd.DataFrame(res)
summ = R.groupby(["outcome", "frac", "policy"]).agg(
    treated_rate=("treated_rate", "mean"), control_rate=("control_rate", "mean"),
    inc=("inc", "mean"), inc_p10=("inc", lambda s: s.quantile(.1)), inc_p90=("inc", lambda s: s.quantile(.9)),
    captured=("captured", "mean"), overall=("overall", "mean")).reset_index()
wins = (R.pivot_table(index=["outcome", "frac", "seed"], columns="policy", values="inc")
        .assign(w=lambda d: d.upl > d.resp).groupby(["outcome", "frac"]).w.mean().reset_index())
out = dict(n=int(len(df)), treated_share=float(T.mean()), splits=SPLITS,
           rates=dict(visit_t=float(df.visit[T == 1].mean()), visit_c=float(df.visit[T == 0].mean()),
                      conv_t=float(df.conversion[T == 1].mean()), conv_c=float(df.conversion[T == 0].mean())),
           summary=summ.to_dict("records"), upl_wins=wins.to_dict("records"),
           deciles={o: {k: v.tolist() for k, v in d.items()} for o, d in dec.items()})
json.dump(out, open("criteo_results.json", "w"), indent=1)
print(summ.to_string()); print(wins.to_string())
for o in dec: print(o, "resp", np.round(dec[o]["resp"], 4)); print(o, "upl ", np.round(dec[o]["upl"], 4))
