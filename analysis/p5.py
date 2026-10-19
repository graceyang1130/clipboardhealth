import sys, pandas as pd, numpy as np
S=sys.argv[1]; pd.set_option("display.width",220)
s=pd.read_pickle(f"{S}/cle_shifts2.pkl"); cx=pd.read_pickle(f"{S}/cx2.pkl"); cl=pd.read_pickle(f"{S}/claims3.pkl")
e=cx[cx['Facility ID'].isin(set(s['Facility ID']))].dropna(subset=['Worker ID'])
wf=e.groupby('Worker ID').agg(lf=('late_fail','sum'),ncns=('band',lambda b:(b=='1_NCNS').sum()),ev=('band','size'))
ww=s[s.has_worker].groupby('Worker ID').agg(assigned=('Shift ID','size'))
ww['hrs_worked']=s[s.worked].groupby('Worker ID').hrs.sum()
W=ww.join(wf,how='outer').fillna(0); W['bookings']=W.assigned+W.ev; W['rate']=W.lf/W.bookings; W['ncns_rate']=W.ncns/W.bookings
W['seg']=np.select([W.ncns>=3, W.rate>=0.3, W.rate>=0.15, W.lf>0],['3+ NCNS','rate>=30%','rate 15-30%','some late fail'],'no late fail')
print(W.groupby('seg').agg(workers=('lf','size'),med_bookings=('bookings','median'),bookings=('bookings','sum'),lf=('lf','sum'),ncns=('ncns','sum'),hrs=('hrs_worked','sum')).assign(rate=lambda d:(d.lf/d.bookings).round(3),ncns_rate=lambda d:(d.ncns/d.bookings).round(3)).to_string())
# rate vs count: 3+NCNS split by ncns rate
x=W[W.ncns>=3]; print("3+ NCNS workers: ncns rate quartiles", x.ncns_rate.quantile([.25,.5,.75]).round(3).to_dict(), " hrs share of those with ncns_rate<5%:", round(x[x.ncns_rate<.05].hrs_worked.sum()/W.hrs_worked.sum(),3), " n:",(x.ncns_rate<.05).sum())
# refill determinants among Cleveland late-canceled (non-NCNS) shifts
k=['Facility ID','Agent Req','Shift Type']; s['rel']=s.charge/s.groupby(k).charge.transform('median')
f=s[(s.n_fail>0)&s.last_band.isin(['2_<4h','3_4-24h'])].copy()
f['relq']=pd.cut(f.rel,[0,.95,1.0001,1.1,10],labels=['below','at','+0-10%','>+10%'])
print("worked after late cancel, by relative charge:", f.groupby('relq',observed=True).worked.agg(['size','mean']).round(3).to_dict('index'))
print("worked after late cancel, by shift type:", f.groupby('Shift Type').worked.agg(['size','mean']).round(3).to_dict('index'))
print("worked after late cancel, by role:", f.groupby('Agent Req').worked.agg(['size','mean']).round(3).to_dict('index'))
# facility-side cancels of booked shifts
b=s[s.ever_booked]; print("booked Cleveland shifts deleted by facility while a worker was assigned:", round((b.state=='deleted_after_booked').mean(),3))
# NCNS signature vs late cancel signature
c=cl[cl.claim_lead>=0]
for lab,m in [('NCNS',c.outcome=='ncns'),('late cancel',c.band.isin(['2_<4h','3_4-24h'])),('kept',c.outcome=='kept')]:
    z=c[m]; print(f"{lab}: first-at-fac {(z.pair_idx==0).mean():.3f}, tenure<30d {(z.tenure_d<30).mean():.3f}, booked<72h {(z.claim_lead<72).mean():.3f}, median booking lead {z.claim_lead.median():.0f}h, median worker claims {z.groupby('Worker ID').size().median():.0f}")
