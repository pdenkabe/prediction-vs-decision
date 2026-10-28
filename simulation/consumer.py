"""Consumer-side check (base case): how many additional defaults does each policy cause?
Lender value and borrower outcomes are different measures; this reports the second one."""
import numpy as np, json, sim2
from sim2 import *
import sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 7
p=dict(BASE); rng=np.random.default_rng(seed)
pop=population(rng,N,p,0.0); pd0,pd1,v0,v1,_=truth(rng,pop,p); ite=(v1-v0).values; dpd=(pd1-pd0).values
hist=population(rng,HIST_N,p,0.0); h_pd0,_,_,_,h_take=truth(rng,hist,p)
hist["y"]=(rng.random(HIST_N)<h_pd0).astype(int)
risk=HistGradientBoostingClassifier(max_iter=200,random_state=seed).fit(hist[X],hist["y"]); p_def=risk.predict_proba(pop[X])[:,1]
use_m=HistGradientBoostingRegressor(max_iter=200,random_state=seed).fit(hist[X],hist.util*hist.revolver); p_use=use_m.predict(pop[X])
obs_take=h_take+rng.normal(0,0.05,HIST_N)
hr=risk.predict_proba(hist[X])[:,1]; safest=hr<=np.quantile(hr,0.2)
tm=HistGradientBoostingRegressor(max_iter=200,random_state=seed).fit(hist.loc[safest,X],obs_take[safest]); t=np.clip(tm.predict(pop[X]),0,1)
C=p["CLI"]; ep=C*t*p["NET_YIELD"]*p["LIFE"]*(1-p_def)-p_def*p["LGD"]*1.15*(C*t+C*0.6*(1-t))-p["COST"]
pilot=population(rng,PILOT_N,p,0.0); _,_,pv0,pv1,_=truth(rng,pilot,p); tt=(rng.random(PILOT_N)<0.5).astype(int)
pilot["y"]=np.where(tt==1,pv1,pv0)+rng.normal(0,25,PILOT_N)
m1=HistGradientBoostingRegressor(max_iter=300,random_state=seed).fit(pilot.loc[tt==1,X],pilot.loc[tt==1,"y"])
m0=HistGradientBoostingRegressor(max_iter=300,random_state=seed).fit(pilot.loc[tt==0,X],pilot.loc[tt==0,"y"])
upl=m1.predict(pop[X])-m0.predict(pop[X])
out={}
for name,s in [("lowest_risk",-p_def),("highest_usage",p_use),("expected_profit",ep),("incremental_value",upl),("ceiling",ite)]:
    idx=np.argsort(-s)[:K]
    out[name]=dict(value_m=ite[idx].sum()/1e6, extra_defaults=float(dpd[idx].sum()),
                   share_negative=float((ite[idx]<0).mean()))
# overlap of chosen sets
top=lambda s:set(np.argsort(-s)[:K])
out["overlap_value_vs_ep"]=len(top(upl)&top(ep))/K
print(json.dumps(out,indent=1))
