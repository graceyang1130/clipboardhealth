"""Phase 2: worker-side drivers"""
import sys, pandas as pd, numpy as np
from scipy import stats
S=sys.argv[1]; pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
rng=np.random.default_rng(0)
cl=pd.read_pickle(f"{S}/claims.pkl"); cl=cl[cl.claim_lead>=0].copy()
cl['ncns']=cl.outcome=='ncns'; cl['lc']=cl.band.isin(['2_<4h','3_4-24h'])
w=cl.groupby('Worker ID').agg(n=('late_fail','size'),lf=('late_fail','sum'),ncns=('ncns','sum'),lc=('lc','sum'))
print("workers in claim cohort:",len(w)," with >=1 late failure:",(w.lf>0).sum(),f"({(w.lf>0).mean():.1%})", " with >=1 NCNS:",(w.ncns>0).sum(),f"({(w.ncns>0).mean():.1%})")
p=cl.late_fail.mean()
# concentration observed vs homogeneous-rate null
def top_share(counts, frac):
    c=np.sort(counts)[::-1]; k=max(1,int(len(c)*frac)); return c[:k].sum()/c.sum()
sim=[rng.binomial(w.n.values,p) for _ in range(50)]
for frac in [.01,.05,.10,.20]:
    print(f"top {frac:.0%} workers' share of late failures: observed {top_share(w.lf.values,frac):.3f} | null(same rate for all) {np.mean([top_share(x,frac) for x in sim]):.3f} | their share of claims {w.sort_values('lf',ascending=False).n.iloc[:int(len(w)*frac)].sum()/w.n.sum():.3f}")
for col,lab in [('ncns','NCNS'),('lc','late cancel')]:
    pp=cl[col].mean(); simc=[rng.binomial(w.n.values,pp) for _ in range(50)]
    print(f"  {lab}: top 5% share observed {top_share(w[col].values,.05):.3f} vs null {np.mean([top_share(x,.05) for x in simc]):.3f}; top 20% observed {top_share(w[col].values,.20):.3f} vs null {np.mean([top_share(x,.20) for x in simc]):.3f}")
# overdispersion among workers with n>=10
w10=w[w.n>=10]; r=w10.lf/w10.n
print("workers n>=10:",len(w10)," var(rate) observed",round(r.var(),4)," binomial expected",round((p*(1-p)/w10.n).mean(),4), " ratio",round(r.var()/(p*(1-p)/w10.n).mean(),2))
# rate distribution by volume tier
w['tier']=pd.cut(w.n,[0,2,5,10,20,40,1000],labels=['1-2','3-5','6-10','11-20','21-40','41+'])
print("== late-fail rate by worker volume tier (claims in Oct-Nov)")
print(w.groupby('tier',observed=True).agg(workers=('n','size'),claims=('n','sum'),lf=('lf','sum'),ncns=('ncns','sum')).assign(rate=lambda d:d.lf/d.claims,ncns_rate=lambda d:d.ncns/d.claims,share_claims=lambda d:d.claims/d.claims.sum(),share_lf=lambda d:d.lf/d.lf.sum()).round(3).to_string())
# persistence: Oct vs Nov starts
cl['per']=np.where(cl.start<'2021-11-01','A','B')
pw=cl.groupby(['Worker ID','per']).agg(n=('late_fail','size'),lf=('late_fail','sum'),ncns=('ncns','sum')).unstack()
pw=pw.dropna(); pw=pw[(pw[('n','A')]>=5)&(pw[('n','B')]>=5)]
rA=pw[('lf','A')]/pw[('n','A')]; rB=pw[('lf','B')]/pw[('n','B')]
print("== persistence (workers with >=5 claims in Oct AND Nov):",len(pw)," spearman(rate Oct, rate Nov)=",round(stats.spearmanr(rA,rB).statistic,3))
q=pd.cut(rA,[-.01,0,.1,.2,.35,1.01],labels=['0','0-10%','10-20%','20-35%','>35%'])
print(pd.DataFrame({'rA':rA,'rB':rB,'q':q,'nB':pw[('n','B')],'lfB':pw[('lf','B')]}).groupby('q',observed=True).agg(workers=('rB','size'),octrate=('rA','mean'),nov_rate=('rB','mean'),nov_lf=('lfB','sum'),nov_claims=('nB','sum')).assign(share_nov_lf=lambda d:d.nov_lf/d.nov_lf.sum(),share_nov_claims=lambda d:d.nov_claims/d.nov_claims.sum()).round(3).to_string())
nA=pw[('ncns','A')]; nB=pw[('ncns','B')]/pw[('n','B')]
print("Nov NCNS rate by Oct NCNS count:", pd.DataFrame({'a':nA.clip(upper=3),'b':nB,'n':pw[('n','B')],'k':pw[('ncns','B')]}).groupby('a').agg(workers=('b','size'),nov_ncns_rate=('k','sum')).assign(rate=lambda d: d.nov_ncns_rate/ pd.DataFrame({'a':nA.clip(upper=3),'n':pw[('n','B')]}).groupby('a').n.sum()).round(4).to_dict('index'))
# tenure
cl['ten']=pd.cut(cl.tenure_d,[-1,14,30,90,180,365,5000],labels=['<2wk','2wk-1mo','1-3mo','3-6mo','6-12mo','1y+'])
print("== by worker tenure at claim"); print(cl.groupby('ten',observed=True).agg(claims=('late_fail','size'),late_fail=('late_fail','mean'),ncns=('ncns','mean'),late_cancel=('lc','mean'),any_cancel=('outcome',lambda o:(o!='kept').mean())).assign(share=lambda d:d.claims/d.claims.sum()).round(4).to_string())
# first claim at facility (eval: starts in Nov, lookback = any earlier claim in cohort)
cl=cl.sort_values('at'); cl['pair_idx']=cl.groupby(['Worker ID','Facility ID']).cumcount()
cl['w_idx']=cl.groupby('Worker ID').cumcount()
ev=cl[cl.start>='2021-11-01']
ev2=ev.assign(newfac=np.where(ev.pair_idx==0,'first at facility','repeat at facility'))
print("== first vs repeat at facility (Nov starts; lookback from Aug)"); print(ev2.groupby('newfac').agg(claims=('late_fail','size'),late_fail=('late_fail','mean'),ncns=('ncns','mean'),lc=('lc','mean')).round(4).to_string())
ev3=ev2[ev2.tenure_d>180]
print("  restricted to workers with tenure>180d:"); print(ev3.groupby('newfac').agg(claims=('late_fail','size'),late_fail=('late_fail','mean'),ncns=('ncns','mean')).round(4).to_string())
print("  by facility visit index:"); print(ev2.assign(k=ev2.pair_idx.clip(upper=5)).groupby('k').agg(claims=('late_fail','size'),late_fail=('late_fail','mean'),ncns=('ncns','mean')).round(4).to_string())
cl.to_pickle(f"{S}/claims2.pkl")
