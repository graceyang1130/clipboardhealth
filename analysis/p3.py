"""Phase 3: policy lens - bunching at thresholds, strike reconstruction, enforcement proxy"""
import sys, pandas as pd, numpy as np
S=sys.argv[1]; pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
cx=pd.read_pickle(f"{S}/cx2.pkl"); bk=pd.read_pickle(f"{S}/bk2.pkl"); cl=pd.read_pickle(f"{S}/claims3.pkl")
wc=cx[cx.Action=='WORKER_CANCEL']
# hourly histogram 0-80h
h=np.floor(wc.lead[(wc.lead>=0)&(wc.lead<80)]).astype(int).value_counts().sort_index()
print("worker cancels per hour-before-start bin (0-80h):"); print(h.to_string())
# bunching: 15-min bins around thresholds
for T in [4,24,72]:
    z=wc[(wc.lead>=T-1.5)&(wc.lead<T+1.5)]
    bins=np.floor((z.lead-(T-1.5))/0.25).astype(int)
    cnt=bins.value_counts().sort_index()
    print(f"-- around {T}h (15-min bins from {T-1.5}h):", cnt.tolist())
    below=((z.lead>=T-0.5)&(z.lead<T)).sum(); above=((z.lead>=T)&(z.lead<T+0.5)).sum()
    print(f"   30min just below {T}h: {below} | 30min just above: {above} | ratio above/below {above/below:.2f}")
# share of volume by policy band (late-failure world)
print("event share by band:", cx.band.value_counts(normalize=True).sort_index().round(3).to_dict())
# ---- strike reconstruction (events Oct-Jan, chronological) ----
e=cx.dropna(subset=['Worker ID']).sort_values('at').copy()
e['is_ncns']=e.band=='1_NCNS'; e['is_co']=e.band=='2_<4h'; e['is_std']=e.band.isin(['3_4-24h','4_24-72h'])
g=e.groupby('Worker ID')
e['k_ncns']=g.is_ncns.cumsum(); e['k_co']=g.is_co.cumsum(); e['k_std']=g.is_std.cumsum()
deact=e[((e.is_ncns)&(e.k_ncns==3))|((e.is_co)&(e.k_co==4))]
first_deact=deact.groupby('Worker ID')['at'].min().rename('deact_at')
print("workers reaching a deactivation threshold (3 NCNS or 4 call-offs) within Oct-Jan window:",len(first_deact)," of",e['Worker ID'].nunique(),"workers with events")
# activity after deactivation threshold: any later claim (bk created) or later cancel/NCNS on a shift that STARTS >7d after threshold
later_c=e.merge(first_deact,on='Worker ID'); later_c=later_c[later_c.Start>later_c.deact_at+pd.Timedelta(days=7)]
bk2=bk.dropna(subset=['Worker ID']).merge(first_deact,on='Worker ID'); bk2=bk2[bk2['at']>bk2.deact_at+pd.Timedelta(days=7)]
active_after=set(later_c['Worker ID'])|set(bk2['Worker ID'])
# need opportunity window: threshold reached before Jan 15
elig=first_deact[first_deact<'2022-01-15']
print("  reached threshold before Jan 15:",len(elig)," still booking shifts starting >7d later:",len(set(elig.index)&active_after),f"({len(set(elig.index)&active_after)/len(elig):.1%})")
print("  among those, events logged after threshold:", later_c[later_c['Worker ID'].isin(elig.index)].groupby('Worker ID').size().describe().round(1).to_dict())
print("  late failures (NCNS/<24h) on shifts after threshold+7d, by these workers:", later_c[later_c['Worker ID'].isin(elig.index)].late_fail.sum(), " share of all late-fail events:", round(later_c[later_c['Worker ID'].isin(elig.index)].late_fail.sum()/cx.late_fail.sum(),3))
# suspension check: after 1st NCNS, claims created within next 7 days
n1=e[e.is_ncns&(e.k_ncns==1)][['Worker ID','at']].rename(columns={'at':'n1_at'})
n1=n1[(n1.n1_at>='2021-10-01')&(n1.n1_at<'2021-11-20')]
bb=bk.dropna(subset=['Worker ID']).merge(n1,on='Worker ID')
inwin=bb[(bb['at']>bb.n1_at)&(bb['at']<=bb.n1_at+pd.Timedelta(days=7))].groupby('Worker ID').size()
print("1st NCNS logged Oct1-Nov19:",len(n1)," workers who claimed a new shift within 7 days after:",len(inwin),f"({len(inwin)/len(n1):.1%})")
# call-off double: 2nd call-off -> suspension
c2=e[e.is_co&(e.k_co==2)][['Worker ID','at']].rename(columns={'at':'c2_at'}); c2=c2[(c2.c2_at>='2021-10-01')&(c2.c2_at<'2021-11-20')]
bb=bk.dropna(subset=['Worker ID']).merge(c2,on='Worker ID'); inwin=bb[(bb['at']>bb.c2_at)&(bb['at']<=bb.c2_at+pd.Timedelta(days=7))].groupby('Worker ID').size()
print("2nd call-off Oct1-Nov19:",len(c2)," claimed within 7 days after:",len(inwin),f"({len(inwin)/len(c2):.1%})")
# what the policy does NOT count: 4-24h cancels are 'standard' (6 -> suspension only); >72h free
lf=cx[cx.late_fail]; print("late-failure events by policy class: NCNS",round((lf.band=='1_NCNS').mean(),3)," call-off(<4h)",round((lf.band=='2_<4h').mean(),3)," 'standard' 4-24h",round((lf.band=='3_4-24h').mean(),3))
# worker-level event counts distribution over 4 months
cnt=e.groupby('Worker ID').agg(ncns=('is_ncns','sum'),co=('is_co','sum'),std=('is_std','sum'))
print("workers with >=3 NCNS:",(cnt.ncns>=3).sum()," >=4 call-offs:",(cnt.co>=4).sum()," >=6 standard:",(cnt['std']>=6).sum())
# post-NCNS behavior: does a first NCNS change the next-claims failure rate? (event-study within claim cohort)
