"""Baseline values for every success metric, on clean data (with raw data alongside to show what cleaning moved).

Usage: python baselines.py <clean_dir> <raw_dir> [out_dir]     (both dirs are build.py outputs)
"""
import sys, numpy as np, pandas as pd
CLEAN, RAW = sys.argv[1], sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else None
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
TAKE, ANNUAL = 0.22, 3
LATE = ['1_NCNS', '2_<4h', '3_4-24h']


def metrics(d):
    s = pd.read_pickle(f"{d}/cle_shifts.pkl"); cl = pd.read_pickle(f"{d}/claims.pkl")
    cx = pd.read_pickle(f"{d}/cx2.pkl"); bk = pd.read_pickle(f"{d}/bk2.pkl")
    M = {}
    # ---------------- Cleveland, shift level ----------------
    med = s.groupby('Agent Req').charge.median()
    s['gmv'] = s.charge.fillna(s['Agent Req'].map(med)).fillna(med.get('LVN', np.nan)) * s.Time.clip(lower=0)
    b = s[s.ever_booked]
    M['Cleveland: posted shifts'] = len(s)
    M['Cleveland: booked shifts'] = len(b)
    M['Cleveland: late-failure rate (booked shifts with >=1 NCNS or <24h cancel)'] = (b.n_late > 0).mean()
    M['Cleveland: NCNS rate (booked shifts with >=1 NCNS)'] = (b.n_ncns > 0).mean()
    w = b[b.Start_loc.notna()].groupby(b.Start_loc.dt.to_period('W')).agg(n=('Shift ID', 'size'), lf=('n_late', lambda x: (x > 0).mean()), nc=('n_ncns', lambda x: (x > 0).mean()))
    w = w[w.n >= 300]
    M['Cleveland: weekly booked shifts (mean)'] = w.n.mean()
    M['Cleveland: weekly late-failure rate, mean'] = w.lf.mean(); M['Cleveland: weekly late-failure rate, SD'] = w.lf.std()
    M['Cleveland: weekly NCNS rate, mean'] = w.nc.mean(); M['Cleveland: weekly NCNS rate, SD'] = w.nc.std()
    M['Cleveland: fill rate (worked / posted, not deleted)'] = s.loc[~s.deleted, 'worked'].mean()
    lc = s[(s.n_fail > 0) & s.last_band.isin(['2_<4h', '3_4-24h'])]
    M['Cleveland: worked after a <24h cancel (refill)'] = lc.worked.mean()
    M['Cleveland: worked after a <4h cancel'] = s[(s.n_fail > 0) & (s.last_band == '2_<4h')].worked.mean()
    M['Cleveland: worked after NCNS'] = s[(s.n_fail > 0) & (s.last_band == '1_NCNS')].worked.mean()
    lost = s[(s.state == 'unfilled') & s.last_band.isin(LATE)]
    M['Cleveland: shifts left unfilled after a late failure (4 mo)'] = len(lost)
    M['Cleveland: revenue lost to late failures, per year'] = lost.gmv.sum() * TAKE * ANNUAL
    M['Cleveland: lost billings as % of worked billings'] = lost.gmv.sum() / s.loc[s.worked, 'gmv'].sum()
    n = w.n.mean() * 4; p = w.lf.mean()
    M['Cleveland: smallest detectable change, 4 wks/arm (relative)'] = 2.8 * np.sqrt(2 * p * (1 - p) / n) / p
    M['Cleveland: smallest detectable change, 8 wks/arm (relative)'] = 2.8 * np.sqrt(2 * p * (1 - p) / (2 * n)) / p
    # ---------------- all markets, booking level (Oct-Nov starts) ----------------
    c = cl[cl.claim_lead >= 0]
    a24 = c[(c.claim_lead >= 24) & ~((c.outcome == 'cancel') & (c.cancel_lead >= 24))]
    M['All markets: bookings'] = len(c)
    M['All markets: late-failure rate, bookings active at T-24h'] = a24.late_fail.mean()
    M['All markets: late-cancel rate (<24h), bookings active at T-24h'] = (a24.outcome == 'cancel').mean()
    M['All markets: NCNS rate, bookings active at T-24h'] = (a24.outcome == 'ncns').mean()
    for bd in ['1_NCNS', '2_<4h', '3_4-24h', '4_24-72h', '5_>72h']:
        M[f'All markets: share of bookings ending {bd}'] = (c.band == bd).mean()
    wc = cx[(cx.Action == 'WORKER_CANCEL') & (cx.Start >= '2021-10-01') & (cx.Start < '2021-12-01')]
    m = wc[['Action ID', 'Shift ID', 'at', 'Start', 'lead']].merge(bk[['Shift ID', 'at']].rename(columns={'at': 'rc'}), on='Shift ID')
    re = set(m[(m.rc > m['at']) & (m.rc <= m.Start)]['Action ID'])
    wc = wc.assign(refilled=wc['Action ID'].isin(re))
    M['All markets: re-booked before start after a <4h cancel'] = wc[wc.lead < 4].refilled.mean()
    M['All markets: re-booked before start after a 4-24h cancel'] = wc[(wc.lead >= 4) & (wc.lead < 24)].refilled.mean()
    M['All markets: re-booked before start after a >72h cancel'] = wc[wc.lead > 72].refilled.mean()
    return pd.Series(M)


T = pd.DataFrame({'raw': metrics(RAW), 'clean': metrics(CLEAN)})
T['change'] = T.clean - T.raw
fmt = lambda v, k: (f"${v:,.0f}" if '$' in k or 'revenue' in k else f"{v:,.0f}" if abs(v) >= 100 else f"{v:.4f}")
print("BASELINES FOR TARGETS  (clean = the numbers to set targets against)")
print(pd.DataFrame({k: [fmt(r.raw, k), fmt(r.clean, k), fmt(r.change, k)] for k, r in T.iterrows()}, index=['raw', 'clean', 'change']).T.to_string())
if OUT: T.to_csv(f"{OUT}/baselines.csv")
try:
    print("\nEXCLUSION LOG"); print(pd.read_csv(f"{CLEAN}/exclusions.csv").to_string(index=False))
except FileNotFoundError:
    pass
