"""Phase 2: booking behavior"""
import sys, pandas as pd, numpy as np
S=sys.argv[1]; pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
cl=pd.read_pickle(f"{S}/claims2.pkl"); cl['ncns']=cl.outcome=='ncns'; cl['lc']=cl.band.isin(['2_<4h','3_4-24h'])
H=lambda x: pd.Timedelta(hours=x)
# booking lead
cl['blb']=pd.cut(cl.claim_lead,[0,4,24,72,168,336,720,5000],right=False,labels=['<4h','4-24h','1-3d','3-7d','1-2wk','2-4wk','4wk+'])
a24=(cl.claim_lead>=24)&~((cl.outcome=='cancel')&(cl.cancel_lead>=24))
cl['a24']=a24
print("== by booking lead time")
print(cl.groupby('blb',observed=True).agg(claims=('late_fail','size'),any_cancel=('outcome',lambda o:(o!='kept').mean()),late_fail=('late_fail','mean'),ncns=('ncns','mean'),
   cancel_gt72=('band',lambda b:(b=='5_>72h').mean())).assign(share=lambda d:d.claims/d.claims.sum()).round(4).to_string())
x=cl[cl.a24]; print("  P(late fail | active at T-24h) by booking lead:", x.groupby('blb',observed=True).late_fail.mean().round(4).to_dict())
# concurrency: active holds at claim time and overlapping holds
cl['end_act']=cl.cancel_at.where(cl.outcome=='cancel',cl.start)
cl['end_act']=pd.to_datetime(cl.end_act)
holds=np.zeros(len(cl),dtype=int); conflict=np.zeros(len(cl),dtype=bool); conflict_kept_other=np.zeros(len(cl),dtype=bool)
prior_kept=np.zeros(len(cl),dtype=bool)
cl=cl.reset_index(drop=True)
for wkr,idx in cl.groupby('Worker ID').indices.items():
    at=cl['at'].values[idx]; ea=cl.end_act.values[idx]; st=cl.start.values[idx]; kept=(cl.outcome.values[idx]=='kept')
    for j_pos,j in enumerate(idx):
        act=(at<at[j_pos])&(ea>at[j_pos]); holds[j]=act.sum()
        dst=np.abs((st-st[j_pos]).astype('timedelta64[m]').astype(float))/60
        other=np.arange(len(idx))!=j_pos
        # time-slot conflict: another claim by same worker for a shift starting within 4h, held at the same time at some point
        ov=other&(dst<4)&(at<ea[j_pos])&(ea>at[j_pos])
        conflict[j]=ov.any(); conflict_kept_other[j]=(ov&kept).any()
        gap=(st[j_pos]-st).astype('timedelta64[m]').astype(float)/60
        prior_kept[j]=(other&kept&(gap>=7)&(gap<=16)).any()   # worked shift ending ~0-8h before this one starts (assume 8h shifts)
cl['holds']=holds; cl['conflict']=conflict; cl['conflict_kept_other']=conflict_kept_other; cl['prior_kept']=prior_kept
print("== by # other active holds at claim time")
print(cl.assign(h=cl.holds.clip(upper=15)).assign(h=lambda d:pd.cut(d.h,[-1,0,2,5,10,15],labels=['0','1-2','3-5','6-10','11+'])).groupby('h',observed=True).agg(claims=('late_fail','size'),any_cancel=('outcome',lambda o:(o!='kept').mean()),late_fail=('late_fail','mean'),ncns=('ncns','mean')).round(4).to_string())
print("== time-slot conflicts (worker held 2+ shifts starting within 4h of each other)")
print("claims in conflict:",cl.conflict.sum(),f"({cl.conflict.mean():.1%})")
print(cl.groupby('conflict').agg(claims=('late_fail','size'),any_cancel=('outcome',lambda o:(o!='kept').mean()),late_fail=('late_fail','mean'),ncns=('ncns','mean'),lc=('lc','mean')).round(4).to_string())
lf=cl[cl.late_fail]
print("share of late failures where worker held a conflicting shift:",round(lf.conflict.mean(),3)," ...and kept the other one:",round(lf.conflict_kept_other.mean(),3))
for b in ['1_NCNS','2_<4h','3_4-24h','4_24-72h','5_>72h']:
    z=cl[cl.band==b]; print(f"   {b}: n={len(z)} conflict={z.conflict.mean():.3f} kept_other={z.conflict_kept_other.mean():.3f}")
# switching: canceled, and the same worker made a new claim on a conflicting slot within 24h before -> 2h after the cancel
cc=cl[cl.outcome=='cancel'][['Worker ID','Shift ID','cancel_at','start','band','Facility ID']]
oth=cl[['Worker ID','Shift ID','at','start','outcome','Facility ID']].rename(columns={'Shift ID':'s2','at':'at2','start':'start2','outcome':'out2','Facility ID':'f2'})
m=cc.merge(oth,on='Worker ID'); m=m[m['Shift ID']!=m.s2]
m=m[((m.start2-m.start).abs()<H(4))&(m.at2>=m.cancel_at-H(24))&(m.at2<=m.cancel_at+H(2))]
sw=m.groupby(['Worker ID','Shift ID']).agg(kept2=('out2',lambda o:(o=='kept').any()),samefac=('f2',lambda f: False)).reset_index()
cc=cc.merge(sw,on=['Worker ID','Shift ID'],how='left'); cc['switch']=cc.kept2.notna(); cc['switch_kept']=cc.kept2.fillna(False).astype(bool)
print("== switching: cancel + new claim on same time slot within [-24h,+2h] of cancel")
print(cc.groupby('band').agg(n=('switch','size'),switch=('switch','mean'),switch_and_worked_other=('switch_kept','mean')).round(3).to_string())
print("== prior shift within ~0-8h before (back-to-back/short rest)")
print(cl.groupby('prior_kept').agg(claims=('late_fail','size'),late_fail=('late_fail','mean'),ncns=('ncns','mean'),lc=('lc','mean')).round(4).to_string())
cl.to_pickle(f"{S}/claims3.pkl")
# within-worker check for booking lead (demeaned by worker)
z=cl[cl.groupby('Worker ID').late_fail.transform('size')>=10].copy()
z['wmean']=z.groupby('Worker ID').late_fail.transform('mean')
print("== within-worker: late-fail rate minus worker's own mean, by booking lead"); print(z.assign(d=z.late_fail-z.wmean).groupby('blb',observed=True).d.mean().round(4).to_dict())
z['wm_fac']=z.groupby('Worker ID').late_fail.transform('mean')
