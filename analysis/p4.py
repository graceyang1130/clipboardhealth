"""Hazard by hour-before-start; supply cost of reliability segments; segment sizing; baselines/MDE"""
import sys, pandas as pd, numpy as np
S=sys.argv[1]; pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
cl=pd.read_pickle(f"{S}/claims3.pkl"); s=pd.read_pickle(f"{S}/cle_shifts2.pkl"); cx=pd.read_pickle(f"{S}/cx2.pkl")
# hazard: worker-cancel per active booking-hour
c=cl[cl.claim_lead>=0]
cancel_lead=c.cancel_lead.where(c.outcome=='cancel')
rows=[]
for h in list(range(0,24))+[24,36,48,60,72,96,120,144,167]:
    active=((c.claim_lead>h+1)&~(cancel_lead>=h+1)).sum()
    ev=((cancel_lead>=h)&(cancel_lead<h+1)).sum()
    rows.append((h,active,ev,ev/active))
hz=pd.DataFrame(rows,columns=['h','active','cancels','hazard_per_hr'])
print("== hourly cancel hazard (per active booking) by hours before start"); print(hz.assign(hazard_per_hr=lambda d:(d.hazard_per_hr*100).round(3)).to_string(index=False))
print("cumulative: P(cancel in last 24h | active at T-24h)",round(1-np.prod(1-hz[hz.h<24].hazard_per_hr),4))
# segment sizing of late failures (claim cohort)
c=c.copy(); c['first_fac']=c.pair_idx==0; c['new_w']=c.tenure_d<30; c['short_book']=c.claim_lead<72
c['ncns3']=False
tot=c.late_fail.sum()
for col in ['first_fac','new_w','short_book']:
    print(f"{col}: share of claims {c[col].mean():.3f} | share of late failures {c.loc[c[col],'late_fail'].sum()/tot:.3f} | rate {c.loc[c[col],'late_fail'].mean():.3f} vs {c.loc[~c[col],'late_fail'].mean():.3f}")
anyr=c.first_fac|c.new_w|c.short_book
print(f"any of the three: share of claims {anyr.mean():.3f} | share of late failures {c.loc[anyr,'late_fail'].sum()/tot:.3f} | NCNS share {c.loc[anyr,'ncns'].sum()/c.ncns.sum():.3f}")
# supply: Cleveland worked hours by worker reliability (4-month events / approx bookings)
e=cx[cx['Facility ID'].isin(set(s['Facility ID']))].dropna(subset=['Worker ID'])
wf=e.groupby('Worker ID').agg(lf=('late_fail','sum'),ncns=('band',lambda b:(b=='1_NCNS').sum()))
ww=s[s.has_worker].groupby('Worker ID').agg(assigned=('Shift ID','size'),worked=('worked','sum'),hrs=('hrs',lambda x: x.sum()))
ww['hrs_worked']=s[s.worked].groupby('Worker ID').hrs.sum()
W=ww.join(wf,how='outer').fillna(0); W['bookings']=W.assigned+e.groupby('Worker ID').size().reindex(W.index).fillna(0)
W['rate']=W.lf/W.bookings
W['seg']=np.select([W.ncns>=3, W.rate>=0.3, W.rate>=0.15, W.lf>0],['3+ NCNS','rate>=30%','rate 15-30%','some late fail'],'no late fail')
print("== Cleveland workers by reliability segment (Oct-Jan)")
print(W.groupby('seg').agg(workers=('lf','size'),hrs_worked=('hrs_worked','sum'),late_fail=('lf','sum')).assign(share_hrs=lambda d:(d.hrs_worked/d.hrs_worked.sum()).round(3),share_lf=lambda d:(d.late_fail/d.late_fail.sum()).round(3)).to_string())
# weekly Cleveland baselines for metrics + MDE
b=s[s.ever_booked&s.Start_loc.notna()].copy(); b['wk']=b.Start_loc.dt.to_period('W')
wkly=b.groupby('wk').agg(booked=('Shift ID','size'),late=('n_late',lambda x:(x>0).mean()),ncns=('n_ncns',lambda x:(x>0).mean()))
wkly=wkly[wkly.booked>=300]
print("== Cleveland weekly (booked shifts):", len(wkly),"weeks; booked/wk mean",round(wkly.booked.mean()),"; late-fail rate mean",round(wkly.late.mean(),4),"sd",round(wkly.late.std(),4),"; NCNS mean",round(wkly.ncns.mean(),4),"sd",round(wkly.ncns.std(),4))
n=wkly.booked.mean()*4; p=wkly.late.mean()
mde=2.8*np.sqrt(2*p*(1-p)/n)
print(f"MDE (80% power, a=.05, 2-arm, 4 weeks per arm at ~{int(n)} booked shifts/arm): {mde:.4f} abs = {mde/p:.1%} relative")
n8=n*2; print(f"  8 weeks/arm: {2.8*np.sqrt(2*p*(1-p)/n8):.4f} abs = {2.8*np.sqrt(2*p*(1-p)/n8)/p:.1%} rel")
print("Note: worker-level randomization -> clustered; facility-level randomization needs more time")
