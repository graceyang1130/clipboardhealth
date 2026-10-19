"""Phase 2: shift/facility attributes and timing (Cleveland local time)"""
import sys, pandas as pd, numpy as np
from scipy import stats
S=sys.argv[1]; pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
s=pd.read_pickle(f"{S}/cle_shifts2.pkl"); cx=pd.read_pickle(f"{S}/cx2.pkl")
b=s[s.ever_booked].copy(); b['lf']=b.n_late>0; b['nc']=b.n_ncns>0
def tab(col, d=b, minn=100):
    t=d.groupby(col,observed=True).agg(n=('lf','size'),late_fail=('lf','mean'),ncns=('nc','mean'),worked=('worked','mean'))
    return t[t.n>=minn].round(3)
print("== booked Cleveland shifts overall late_fail",round(b.lf.mean(),3)," ncns",round(b.nc.mean(),3))
print(tab('Shift Type').to_string()); print(tab('Agent Req').to_string())
b['dow']=b.Start_loc.dt.day_name().str[:3]; print(tab('dow').to_string())
b['durb']=pd.cut(b.dur,[0,6,8.01,10,12.01,30],labels=['<6h','6-8h','8-10h','10-12h','12h+']); print(tab('durb').to_string())
b['plb']=pd.cut(b.post_lead_h,[-1e9,24,72,168,336,720,1e9],labels=['<1d','1-3d','3-7d','1-2wk','2-4wk','4wk+']); print("posting lead:\n",tab('plb').to_string())
# relative charge: vs facility x role x shift type median
k=['Facility ID','Agent Req','Shift Type']
b['rel']=b.charge/b.groupby(k).charge.transform('median')
b['relq']=pd.cut(b.rel,[0,.95,1.0001,1.1,10],labels=['below median','at median','+0-10%','>+10%']); print("relative charge:\n",tab('relq').to_string())
# facility heterogeneity
f=b.groupby('Facility ID').agg(n=('lf','size'),lf=('lf','sum'),nc=('nc','sum'))
f=f[f.n>=50]; p=b.lf.mean()
f['rate']=f.lf/f.n; f['z']=(f.rate-p)/np.sqrt(p*(1-p)/f.n)
print("facilities n>=50:",len(f)," rate range",round(f.rate.min(),3),"-",round(f.rate.max(),3)," |z|>3:",(f.z.abs()>3).sum()," above:",(f.z>3).sum()," below:",(f.z<-3).sum())
print("  share of late failures at facilities z>3:",round(f[f.z>3].lf.sum()/f.lf.sum(),3)," their share of booked shifts:",round(f[f.z>3].n.sum()/f.n.sum(),3))
print("  overdispersion ratio:", round(f.rate.var()/(p*(1-p)/f.n).mean(),2))
# timing of late cancels (Cleveland, local clock)
cc=cx[cx['Shift ID'].isin(set(s['Shift ID']))&(cx.Action=='WORKER_CANCEL')].merge(s[['Shift ID','Shift Type','Start_loc']],on='Shift ID')
cc['at_loc']=cc['at'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
late=cc[cc.lead<24]
print("== clock hour of late (<24h) worker cancels, by shift type (local)")
print(pd.crosstab(late.at_loc.dt.hour,late['Shift Type'],normalize='columns').round(3).to_string())
print("share of late cancels made between 10pm-6am local:", round(late.at_loc.dt.hour.isin([22,23,0,1,2,3,4,5]).mean(),3))
print("share of late cancels for AM shifts made same calendar day as shift or night before after 8pm:", round(((late['Shift Type']=='am')&((late.at_loc.dt.date==late.Start_loc.dt.date)|(late.at_loc.dt.hour>=20))).sum()/(late['Shift Type']=='am').sum(),3))
# shock day check
d=s[s.ever_booked&(s.Start_loc.dt.date.astype(str).isin(['2022-01-16','2022-01-17']))]
print("Jan16-17 booked shifts:",len(d)," late fail:",round((d.n_late>0).mean(),3)," facilities affected:",d[d.n_late>0]['Facility ID'].nunique(),"of",d['Facility ID'].nunique())
cj=cc[cc.Start_loc.dt.date.astype(str).isin(['2022-01-16','2022-01-17'])&(cc.lead<24)]
print("  late cancels Jan16-17:",len(cj)," distinct workers:",cj['Worker ID'].nunique())
