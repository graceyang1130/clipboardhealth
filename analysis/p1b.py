"""Phase 1b: $ sizing, trend, facility churn proxy, holidays"""
import sys, pandas as pd, numpy as np
from scipy import stats
S=sys.argv[1]; pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
s=pd.read_pickle(f"{S}/cle_shifts.pkl"); s=s[s.Start.notna()]; cl=pd.read_pickle(f"{S}/claims.pkl")
# charge imputation
med=s.groupby('Agent Req').charge.median()
s['charge_i']=s.charge.fillna(s['Agent Req'].map(med)).fillna(med['LVN'])
s['hrs']=s.Time.clip(lower=0)
s['gmv']=s.charge_i*s.hrs
print("Cleveland worked GMV (4 mo):", round(s.loc[s.worked,'gmv'].sum()), " revenue @22%:", round(0.22*s.loc[s.worked,'gmv'].sum()))
print("Cleveland unfilled-state shifts:",(s.state=='unfilled').sum()," of which had an HCP failure:",((s.state=='unfilled')&(s.n_fail>0)).sum())
lost=s[(s.n_fail>0)&(s.state.isin(['unfilled']))]
lostl=lost[lost.last_band.isin(['1_NCNS','2_<4h','3_4-24h'])]
print("Lost (unfilled after HCP failure) shifts:",len(lost)," GMV:",round(lost.gmv.sum())," rev:",round(.22*lost.gmv.sum()))
print("  of which last failure was late (<24h or NCNS):",len(lostl)," GMV:",round(lostl.gmv.sum())," rev:",round(.22*lostl.gmv.sum()))
print("  lost GMV by role:", lost.groupby("Agent Req").gmv.agg(["size","sum"]).round(0).to_dict("index"))
print("  lost by last band:", lost.groupby('last_band').agg(n=('gmv','size'),gmv=('gmv','sum')).round(0).to_dict('index'))
print("lost GMV as % of worked GMV:", round(lost.gmv.sum()/s.loc[s.worked,'gmv'].sum(),4), " late-only:", round(lostl.gmv.sum()/s.loc[s.worked,'gmv'].sum(),4))
print("fill rate of non-deleted posted shifts:", round(s.loc[~s.deleted].worked.mean(),4))
# trend by month (Cleveland, booked shifts)
s['mon']=s.Start_loc.dt.to_period('M')
b=s[s.ever_booked]
t=b.groupby('mon').agg(booked=('Shift ID','size'),late_any=('n_late',lambda x:(x>0).mean()),ncns_any=('n_ncns',lambda x:(x>0).mean()),
    fail_any=('n_fail',lambda x:(x>0).mean()),worked=('worked','mean'))
t['posted']=s.groupby('mon').size(); t['fill_nondeleted']=s[~s.deleted].groupby('mon').worked.mean()
print("== Cleveland monthly"); print(t.round(3).to_string())
# national claim-cohort by start week
cl=cl[cl.claim_lead>=0]; cl['wk']=cl.start.dt.to_period('W')
print("== national claim cohort weekly late-fail rate"); print(cl.groupby('wk').agg(n=('late_fail','size'),late=('late_fail','mean'),ncns=('outcome',lambda o:(o=='ncns').mean())).round(4).to_string())
# holidays (Cleveland, local date of shift start)
s['d']=s.Start_loc.dt.date
dd=s[s.ever_booked].groupby('d').agg(n=('Shift ID','size'),late=('n_late',lambda x:(x>0).mean()))
dd=dd[dd.n>=50]
print("Cleveland daily late-fail rate: mean",round(dd.late.mean(),3)," sd",round(dd.late.std(),3))
print("top 10 days:\n", dd.sort_values('late',ascending=False).head(10).round(3).to_string())
# facility churn proxy
fm=s.groupby(['Facility ID','mon']).agg(posted=('Shift ID','size')).unstack(fill_value=0)
fm.columns=fm.columns.droplevel(0)
fr=b.groupby(['Facility ID','mon']).agg(n=('Shift ID','size'),late=('n_late',lambda x:(x>0).mean()),ncns=('n_ncns',lambda x:(x>0).mean())).reset_index()
rows=[]
mons=[pd.Period(x,'M') for x in ['2021-10','2021-11','2021-12','2022-01']]
for i in range(len(mons)-1):
    m0,m1=mons[i],mons[i+1]
    x=fr[fr.mon==m0].set_index('Facility ID')
    x=x[x.n>=20]
    g=np.log((fm.loc[x.index,m1]+1)/(fm.loc[x.index,m0]+1))
    rows.append(pd.DataFrame({'late':x.late,'ncns':x.ncns,'growth':g,'m':str(m0)}))
R=pd.concat(rows)
# demean growth by month to remove seasonality
R['g_adj']=R.growth-R.groupby('m').growth.transform('mean')
print("facility-month pairs:",len(R)," spearman(late fail rate, next-month posting growth, month-adjusted):",stats.spearmanr(R.late,R.g_adj))
print("  spearman(NCNS rate, growth adj):",stats.spearmanr(R.ncns,R.g_adj))
R['q']=pd.qcut(R.late,3,labels=['low','mid','high'])
print(R.groupby('q',observed=True).agg(n=('g_adj','size'),late=('late','mean'),g_adj=('g_adj','mean'),zero_next=('growth',lambda g:(g<np.log(1/21)).mean())).round(3).to_string())
# facilities that stopped posting
last=s.groupby('Facility ID').Start_loc.max(); first=s.groupby('Facility ID').Start_loc.min()
print("facilities with last shift before Jan 1:",(last<'2022-01-01').sum()," first shift after Nov 1:",(first>'2021-11-01').sum())
s.to_pickle(f"{S}/cle_shifts2.pkl")
