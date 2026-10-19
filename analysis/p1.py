"""Phase 1: size & severity"""
import sys, pandas as pd, numpy as np
S=sys.argv[1]; pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
cl=pd.read_pickle(f"{S}/claims.pkl"); cx=pd.read_pickle(f"{S}/cx2.pkl"); bk=pd.read_pickle(f"{S}/bk2.pkl"); s=pd.read_pickle(f"{S}/cle_shifts.pkl")
cl=cl[cl.claim_lead>=0]  # exclude retroactive claims
print("claims (lead>=0):",len(cl))
print("== Outcome band share per claim (all markets / Cleveland)")
print(pd.concat([cl.band.value_counts(normalize=True).sort_index(), cl[cl.cleveland].band.value_counts(normalize=True).sort_index()],axis=1,keys=['all','cle']).round(4).to_string())
# conditional rates with survival denominators
a24=cl[(cl.claim_lead>=24)&~((cl.outcome=='cancel')&(cl.cancel_lead>=24))]
print("active at T-24h:",len(a24)," -> late cancel(<24h):",round((a24.outcome=='cancel').mean(),4)," NCNS:",round((a24.outcome=='ncns').mean(),4)," any late fail:",round(a24.late_fail.mean(),4), " kept:", round((a24.outcome=='kept').mean(),4))
a0=cl[~(cl.outcome=='cancel')]
print("active at start (not worker-canceled):",len(a0)," NCNS rate:",round((a0.outcome=='ncns').mean(),4))
late_book=cl[cl.claim_lead<24]
print("claims made <24h before start:",len(late_book),f"({len(late_book)/len(cl):.1%})"," late-fail rate:",round(late_book.late_fail.mean(),4)," NCNS:",round((late_book.outcome=='ncns').mean(),4))
print("share of all late failures coming from claims made <24h out:", round(late_book.late_fail.sum()/cl.late_fail.sum(),4))
# composition of late failures
lf=cl[cl.late_fail]; print("late failures composition:", lf.band.value_counts(normalize=True).sort_index().round(3).to_dict())
# worker cancel lead distribution (event-level, all cancels)
wc=cx[cx.Action=='WORKER_CANCEL']
print("== Worker-cancel lead CDF (all events Oct-Jan):")
for h in [1,2,4,8,12,24,48,72,168]: print(f"  <{h}h: {(wc.lead<h).mean():.3f}")
# ---------- refill curve (national, Oct-Nov starts) ----------
wcc=wc[(wc.Start>='2021-10-01')&(wc.Start<'2021-12-01')].copy()
b=bk[['Shift ID','at','Worker ID']].rename(columns={'at':'rc_at','Worker ID':'rc_w'})
m=wcc[['Action ID','Shift ID','at','Start','lead','band','Worker ID']].merge(b,on='Shift ID',how='left')
m=m[(m.rc_at>m['at'])&(m.rc_at<=m.Start)]
first=m.sort_values('rc_at').groupby('Action ID').first()
wcc['refilled']=wcc['Action ID'].isin(first.index)
wcc=wcc.merge(first[['rc_at']],left_on='Action ID',right_index=True,how='left')
wcc['ttr_h']=(wcc.rc_at-wcc['at']).dt.total_seconds()/3600
bins=[0,1,2,4,8,12,24,48,72,168,10000]
wcc['lb']=pd.cut(wcc.lead,bins,right=False)
print("== Refill before start by cancel lead (national, Oct-Nov starts)")
print(wcc.groupby('lb',observed=True).agg(n=('refilled','size'),refill=('refilled','mean'),med_ttr_h=('ttr_h','median')).round(3).to_string())
# ---------- Cleveland final outcome by last failure band ----------
f=s[s.n_fail>0]
print("== Cleveland shifts w/ HCP failure: final state by LAST failure band")
print(pd.crosstab(f.last_band,f.state,normalize='index').round(3).assign(n=f.last_band.value_counts()).to_string())
f2=f.copy(); f2['lb']=pd.cut(f2.last_lead,bins,right=False)
print(pd.crosstab(f2[f2.last_band!='1_NCNS'].lb,f2[f2.last_band!='1_NCNS'].worked,normalize='index').round(3).to_string())
print("Cleveland: booked shifts:",s.ever_booked.sum()," with >=1 late failure:",(s.n_late>0).sum(), f"({(s.n_late>0).sum()/s.ever_booked.sum():.1%})", " with NCNS:",(s.n_ncns>0).sum())
wcc.to_pickle(f"{S}/refill.pkl")
