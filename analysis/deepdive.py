"""Deep-dive on the two problems -- (1) fewer late drops, (2) refill late drops fast -- for <24h, <4h and NCNS.

Usage: python deepdive.py <clean_dir> [out_dir]        (clean build.py outputs)

  0  Prize matrix: what each failure type costs and what each kind of lever is worth (Cleveland, per year)
  PREVENT
  P1 Sick streaks: after a worker drops one shift late, are their next shifts at risk? Do drops come in batches?
  P2 First-visit risk: concentrated at a few facilities, or everywhere? Does experience elsewhere remove it?
  P3 Warning signs: what share of NCNS / <4h / 4-24h drops came from a booking that showed a sign at booking time?
  P4 NCNS by shift start time
  RECOVER
  R1 Golden hour: how fast late drops get re-booked, if they are
  R2 Who picks up late drops, and do they show up?
  R3 Clock time of the drop: are night drops harder to refill?
  R4 Bench depth and market pressure: what makes a refill likely?
  R5 What one extra hour of notice is worth
  R6 NCNS recovery ceiling: what if we knew at start, or 2h before?
"""
import sys, os, numpy as np, pandas as pd
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import booking_features
S = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
TAKE, ANNUAL = 0.22, 3
LATE = ['1_NCNS', '2_<4h', '3_4-24h']
H = lambda x: pd.Timedelta(hours=x)
TZ = 'America/New_York'
to_loc = lambda t: t.dt.tz_localize('UTC').dt.tz_convert(TZ).dt.tz_localize(None)

cl = pd.read_pickle(f"{S}/claims.pkl"); cx = pd.read_pickle(f"{S}/cx2.pkl"); bk = pd.read_pickle(f"{S}/bk2.pkl"); s = pd.read_pickle(f"{S}/cle_shifts.pkl")
c = booking_features(cl, cx, with_holds=False)
med = s.groupby('Agent Req').charge.median()
s['gmv'] = s.charge.fillna(s['Agent Req'].map(med)) * s.Time.clip(lower=0)
GMV = s.loc[(s.state == 'unfilled') & s.last_band.isin(LATE), 'gmv'].mean()
VAL = GMV * TAKE                                            # revenue value of one recovered shift
hdr = lambda t: print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def fe_effect(df, y, x):
    """same-worker effect of binary x on y (pp), worker fixed effects, clustered SE"""
    d = df[df.groupby('Worker ID')['Worker ID'].transform('size') >= 2]
    g = d['Worker ID']; Y = d[y].astype(float); X = d[x].astype(float)
    Yd = Y - Y.groupby(g).transform('mean'); Xd = X - X.groupby(g).transform('mean')
    m = sm.OLS(Yd.values, Xd.values.reshape(-1, 1)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(g)[0]})
    lo, hi = m.conf_int()[0]
    return f"{m.params[0] * 100:+.1f} pts [{lo * 100:+.1f}, {hi * 100:+.1f}]"


# ================================================================================================ 0 prize matrix
hdr("0. PRIZE MATRIX (Cleveland clean data, per year; a shift's outcome is attributed to its last failure)")
f = s[s.n_fail > 0]
P_UNF = f.groupby('last_band').apply(lambda g: (g.state == 'unfilled').mean(), include_groups=False)
nxt = {'1_NCNS': '2_<4h', '2_<4h': '3_4-24h', '3_4-24h': '4_24-72h'}
rows = []
for b in LATE:
    x = f[f.last_band == b]
    rows.append({'failure': b, 'events_per_yr': len(x) * ANNUAL, 'end_unfilled': P_UNF[b], 'worked': x.worked.mean(),
                 'lost_rev_per_yr': x.loc[x.state == 'unfilled', 'gmv'].sum() * TAKE * ANNUAL,
                 'prevent 10% of them': 0.10 * x.loc[x.state == 'unfilled', 'gmv'].sum() * TAKE * ANNUAL,
                 'refill +10 pts': 0.10 * len(x) * ANNUAL * VAL,
                 'move 10% one band earlier': 0.10 * len(x) * ANNUAL * (P_UNF[b] - P_UNF[nxt[b]]) * VAL})
P0 = pd.DataFrame(rows).set_index('failure')
print(P0.round(3).to_string()); print(f"(value of one recovered shift = ${VAL:,.0f} revenue on ${GMV:,.0f} billings)")

# ================================================================================================ P1 sick streaks
hdr("P1. SICK STREAKS: is a worker's next shift at risk right after they drop one late?")
e = c[c.start >= '2021-11-01'].copy()
A = e[e.active24].copy()
lc_ev = cx[(cx.Action == 'WORKER_CANCEL') & (cx.lead < 24)]
times = {w: np.sort(g['at'].values) for w, g in lc_ev.groupby('Worker ID')}
t24 = (A.start - H(24)).values
for lab, lo_h, hi_h in [('late drop in the 24h before', 0, 24), ('late drop 1-3 days before', 24, 72), ('late drop 3-7 days before', 72, 168)]:
    flag = np.zeros(len(A), dtype=bool)
    for w, idx in A.groupby('Worker ID').indices.items():
        t = times.get(w)
        if t is None: continue
        a = np.searchsorted(t, t24[idx] - np.timedelta64(hi_h, 'h')); b = np.searchsorted(t, t24[idx] - np.timedelta64(lo_h, 'h'))
        flag[idx] = (b - a) > 0
    A[lab] = flag
    x, y = A[flag], A[~flag]
    print(f"{lab:<30} (measured at this booking's 24h mark): {flag.mean():.3%} of bookings | late-fail {x.late_fail.mean():.3f} vs {y.late_fail.mean():.3f} "
          f"({x.late_fail.mean() / y.late_fail.mean():.1f}x) | <4h {x.band.eq('2_<4h').mean():.3f} vs {y.band.eq('2_<4h').mean():.3f} | NCNS {x.ncns.mean():.3f} vs {y.ncns.mean():.3f} "
          f"| same worker: {fe_effect(A, 'late_fail', lab)}")
# cleaner streak test: the SAME workers' bookings in the week before vs the week after a late drop
dr = lc_ev[lc_ev['Worker ID'].isin(set(A['Worker ID']))][['Worker ID', 'Shift ID', 'at']].rename(columns={'Shift ID': 'drop_shift', 'at': 't0'})
pr = A[['Worker ID', 'Shift ID', 'start', 'late_fail', 'band']].merge(dr, on='Worker ID')
pr = pr[pr['Shift ID'] != pr.drop_shift]
pr['dt_days'] = ((pr.start - H(24)) - pr.t0).dt.total_seconds() / 86400
pr = pr[pr.dt_days.abs() <= 7]
pr['window'] = pd.cut(pr.dt_days, [-7, -3, -1, 0, 1, 3, 7], labels=['3-7d before', '1-3d before', '<1d before', '<1d after', '1-3d after', '3-7d after'])
print("late-fail rate of the same workers' other bookings, by when their 24h mark falls relative to a late drop:")
print(pr.groupby('window', observed=True).agg(pairs=('late_fail', 'size'), late_fail=('late_fail', 'mean'), lt4=('band', lambda b: (b == '2_<4h').mean()),
      ncns=('band', lambda b: (b == '1_NCNS').mean())).round(3).to_string())
# follow-on drops: share of late failures on a shift starting within 48h after the same worker's earlier late drop
fo = A[['Worker ID', 'Shift ID', 'start', 'late_fail', 'band', 'cancel_at']].merge(dr, on='Worker ID')
fo = fo[(fo['Shift ID'] != fo.drop_shift) & (fo.t0 < fo.start) & (fo.start - fo.t0 <= H(48))
        & (fo.cancel_at.isna() | (fo.t0 < fo.cancel_at))]
A['follow_on'] = A.set_index(['Worker ID', 'Shift ID']).index.isin(fo.set_index(['Worker ID', 'Shift ID']).index)
print(f"bookings starting within 48h after the same worker dropped another shift late: {A.follow_on.mean():.1%} of bookings, "
      f"late-fail {A[A.follow_on].late_fail.mean():.3f} vs {A[~A.follow_on].late_fail.mean():.3f}; "
      f"they hold {A[A.follow_on].late_fail.sum() / A.late_fail.sum():.1%} of late failures "
      f"({A[A.follow_on].band.eq('2_<4h').sum() / A.band.eq('2_<4h').sum():.1%} of <4h, {A[A.follow_on].ncns.sum() / A.ncns.sum():.1%} of NCNS)")
# batches: several shifts dropped in one sitting
lc2 = lc_ev.sort_values(['Worker ID', 'at']).copy()
gap_prev = lc2.groupby('Worker ID')['at'].diff().dt.total_seconds() / 60
gap_next = -lc2.groupby('Worker ID')['at'].diff(-1).dt.total_seconds() / 60
lc2['batch'] = (gap_prev <= 60) | (gap_next <= 60)
print(f"late (<24h) cancels made within 60 min of another cancel by the same worker: {lc2.batch.mean():.1%}")
allc = cx[cx.Action == 'WORKER_CANCEL'].sort_values(['Worker ID', 'at']).copy()
gp = allc.groupby('Worker ID')['at'].diff().dt.total_seconds() / 60; gn = -allc.groupby('Worker ID')['at'].diff(-1).dt.total_seconds() / 60
allc['batch'] = (gp <= 60) | (gn <= 60)
print("share of cancels made in a batch, by notice band:", allc.groupby('band').batch.mean().round(3).to_dict())

# ================================================================================================ P2 first-visit risk
hdr("P2. FIRST-VISIT RISK: concentrated at a few facilities, or everywhere?")
fv = A.groupby(['Facility ID', 'first_fac']).late_fail.agg(['size', 'mean']).unstack()
fv = fv[(fv[('size', True)] >= 20) & (fv[('size', False)] >= 20)]
rr = fv[('mean', True)] / fv[('mean', False)].replace(0, np.nan)
print(f"facilities with >=20 first-visit and >=20 repeat bookings: {len(fv)}; first visits riskier at {(rr > 1).mean():.0%} of them; "
      f"facility-level ratio quartiles {rr.quantile([.25, .5, .75]).round(2).to_dict()}")
for lab, m in [('worker with <20 past bookings', A.prior_n < 20), ('worker with 20+ past bookings', A.prior_n >= 20)]:
    x = A[m]; print(f"{lab:<32} first visit {x[x.first_fac].late_fail.mean():.3f} vs repeat {x[~x.first_fac].late_fail.mean():.3f} "
                    f"({x[x.first_fac].late_fail.mean() / x[~x.first_fac].late_fail.mean():.1f}x)")
for lab, m in [('booked 1-3 days ahead', A.claim_lead < 72), ('booked 3+ days ahead', A.claim_lead >= 72)]:
    x = A[m]; print(f"{lab:<32} first visit {x[x.first_fac].late_fail.mean():.3f} vs repeat {x[~x.first_fac].late_fail.mean():.3f}")
print("first-visit risk by band: ", {b: f"{A[A.first_fac].band.eq(b).mean():.3f} vs {A[~A.first_fac].band.eq(b).mean():.3f}" for b in LATE})

# ================================================================================================ P3 warning signs
hdr("P3. WARNING SIGNS: what share of each drop type came from a booking that showed a sign at booking time?")
recent = np.zeros(len(e), dtype=bool); at = e['at'].values
for w, idx in e.groupby('Worker ID').indices.items():
    t = times.get(w)
    if t is None: continue
    recent[idx] = (np.searchsorted(t, at[idx]) - np.searchsorted(t, at[idx] - np.timedelta64(72, 'h'))) > 0
SIGNS = {'past late-fail rate >=10% (5+ bookings)': (e.prior_n >= 5) & (e.prior_rate >= 0.10),
         'fewer than 5 past bookings': e.prior_n < 5,
         '2+ past call-offs': e.prior_calloff >= 2, '2+ past NCNS': e.prior_ncns >= 2,
         'first visit to this facility': e.first_fac, 'booked 1-3 days ahead': (e.claim_lead >= 24) & (e.claim_lead < 72),
         'dropped another shift late in last 72h': pd.Series(recent, index=e.index)}
W = pd.DataFrame(SIGNS); anyw = W.any(axis=1)
tab = {'all bookings': W.mean()}
for b, lab in [('1_NCNS', 'NCNS'), ('2_<4h', '<4h cancels'), ('3_4-24h', '4-24h cancels')]:
    tab[lab] = W[e.band == b].mean()
T3 = pd.DataFrame(tab); T3.loc['ANY sign'] = [anyw.mean()] + [anyw[e.band == b].mean() for b in LATE]
T3.loc['NO sign: failure rate'] = [np.nan] + [e[~anyw].band.eq(b).mean() for b in LATE]
T3.loc['ANY sign: failure rate'] = [np.nan] + [e[anyw].band.eq(b).mean() for b in LATE]
print(T3.round(3).to_string())

# ================================================================================================ P4 NCNS by start time
hdr("P4. NCNS AND <4h DROPS BY SHIFT START TIME (Cleveland, booked shifts)")
b_ = s[s.ever_booked].copy(); b_['start_hr'] = b_.Start_loc.dt.hour
b_['slot'] = pd.cut(b_.start_hr, [-1, 4, 9, 13, 17, 21, 24], labels=['0-4am', '5-9am', '10am-1pm', '2-5pm', '6-9pm', '10pm-12am'])
print(b_.groupby('slot', observed=True).agg(booked=('Shift ID', 'size'), ncns=('n_ncns', lambda v: (v > 0).mean()),
      lt4=('last_band', lambda v: (v == '2_<4h').mean()), any_late=('n_late', lambda v: (v > 0).mean())).round(3).to_string())

# ================================================================================================ R1 golden hour
hdr("R1. GOLDEN HOUR: how fast are late drops re-booked (all markets, Oct-Nov starts)?")
wc = cx[(cx.Action == 'WORKER_CANCEL') & (cx.Start >= '2021-10-01') & (cx.Start < '2021-12-01') & (cx.lead < 24)].copy()
m = wc[['Action ID', 'Shift ID', 'at', 'Start', 'Worker ID']].merge(
    bk[['Shift ID', 'at', 'Action ID', 'Worker ID']].rename(columns={'at': 'rc', 'Action ID': 'rc_id', 'Worker ID': 'rc_w'}), on='Shift ID')
m = m[(m.rc > m['at']) & (m.rc <= m.Start)].sort_values('rc').groupby('Action ID').first()
wc = wc.merge(m[['rc', 'rc_id', 'rc_w']], left_on='Action ID', right_index=True, how='left')
wc['ttr_min'] = (wc.rc - wc['at']).dt.total_seconds() / 60
wc['band2'] = np.where(wc.lead < 4, '<4h', '4-24h')
for bnd, g in wc.groupby('band2'):
    r = g[g.ttr_min.notna()]
    print(f"{bnd}: {len(g):,} drops, re-booked before start {g.ttr_min.notna().mean():.1%}; of the re-bookings: "
          + ", ".join(f"<= {k} min {(r.ttr_min <= k).mean():.0%}" for k in [15, 30, 60, 120]))
g = wc[wc.lead >= 4]
print(f"4-24h drops: P(re-booked within 1h) {(g.ttr_min <= 60).mean():.1%}; P(re-booked later) {((g.ttr_min > 60)).mean():.1%}  "
      f"-> share of all eventual re-bookings that happen in the first hour: {(g.ttr_min <= 60).sum() / g.ttr_min.notna().sum():.0%}")

# ================================================================================================ R2 who refills
hdr("R2. WHO PICKS UP LATE DROPS, AND DO THEY SHOW UP?")
cf = c.set_index('Action ID')
rf = cf.loc[cf.index.intersection(wc.rc_id.dropna())].copy()
other_short = cf[(cf.claim_lead < 24) & ~cf.index.isin(rf.index)]
# already booked at the same facility within +-16h (staying over / coming in early)
kept = c[c.outcome == 'kept'][['Worker ID', 'Facility ID', 'start', 'Shift ID']]
def same_site_same_day(df):
    mm = df.reset_index()[['Action ID', 'Worker ID', 'Facility ID', 'start', 'Shift ID']].merge(kept, on=['Worker ID', 'Facility ID'], suffixes=('', '_o'))
    mm = mm[(mm['Shift ID'] != mm['Shift ID_o']) & ((mm.start - mm.start_o).abs() <= H(16))]
    return df.index.isin(set(mm['Action ID']))
rows = []
for lab, d in [('re-bookings of late drops', rf), ('other bookings made <24h out', other_short), ('all bookings', cf)]:
    rows.append({'group': lab, 'n': len(d), 'regular at facility (not 1st visit)': (~d.first_fac).mean(),
                 'also booked there same day': same_site_same_day(d).mean(), 'median past bookings': d.prior_n.median(),
                 'past late-fail rate (5+ bkgs)': d.loc[d.prior_n >= 5, 'prior_rate'].mean(),
                 'then fails late themselves': d.late_fail.mean(), 'NCNS': d.ncns.mean()})
print(pd.DataFrame(rows).set_index('group').round(3).to_string())

# ================================================================================================ R3 clock time
hdr("R3. CLOCK TIME OF THE DROP vs CHANCE THE SHIFT IS STILL WORKED (Cleveland, local time)")
lcs = s[(s.n_fail > 0) & s.last_band.isin(['2_<4h', '3_4-24h'])].copy()
lcs['hr'] = to_loc(lcs.last_at).dt.hour
lcs['when'] = pd.cut(lcs.hr, [-1, 4, 8, 16, 21, 24], labels=['midnight-5am', '5-9am', '9am-5pm', '5-10pm', '10pm-midnight'])
print(lcs.groupby(['last_band', 'when'], observed=True).agg(drops=('worked', 'size'), worked=('worked', 'mean')).round(3).to_string())

# ================================================================================================ R4 bench + pressure
hdr("R4. BENCH DEPTH AND MARKET PRESSURE vs REFILL (Cleveland late drops)")
wk = s[s.worked][['Facility ID', 'Worker ID', 'Start']]
bench = np.zeros(len(lcs), dtype=int)
for i, (fac, st) in enumerate(zip(lcs['Facility ID'], lcs.Start)):
    x = wk[(wk['Facility ID'] == fac) & (wk.Start < st) & (wk.Start >= st - pd.Timedelta(days=30))]
    bench[i] = x['Worker ID'].nunique()
lcs['bench'] = bench
lcs['bench_q'] = pd.cut(lcs.bench, [-1, 10, 30, 1000], labels=['<=10 workers', '11-30', '31+'])
day = s[~s.deleted].groupby(s.Start_loc.dt.date).size()
lcs['demand'] = lcs.Start_loc.dt.date.map(day)
lcs['demand_q'] = pd.qcut(lcs.demand, 3, labels=['quiet day', 'normal day', 'busy day'])
print(lcs.groupby('bench_q', observed=True).agg(drops=('worked', 'size'), worked=('worked', 'mean')).round(3).to_string())
print(lcs.groupby('demand_q', observed=True).agg(drops=('worked', 'size'), worked=('worked', 'mean'), shifts_that_day=('demand', 'mean')).round(3).to_string())

# ================================================================================================ R5 value of an hour
hdr("R5. WHAT ONE EXTRA HOUR OF NOTICE IS WORTH (all markets re-booking curve)")
wca = cx[(cx.Action == 'WORKER_CANCEL') & (cx.Start >= '2021-10-01') & (cx.Start < '2021-12-01') & (cx.lead < 72)].copy()
m = wca[['Action ID', 'Shift ID', 'at', 'Start']].merge(bk[['Shift ID', 'at']].rename(columns={'at': 'rc'}), on='Shift ID')
wca['refilled'] = wca['Action ID'].isin(set(m[(m.rc > m['at']) & (m.rc <= m.Start)]['Action ID']))
edges = [0, 1, 2, 3, 4, 6, 8, 12, 24, 48, 72]
wca['lb'] = pd.cut(wca.lead, edges, right=False)
cur = wca.groupby('lb', observed=True).refilled.agg(['size', 'mean'])
mid = np.array([(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])])
cur['pts_per_extra_hour'] = np.r_[np.nan, np.diff(cur['mean'].values) / np.diff(mid) * 100]
cur['$_per_extra_hour_per_drop'] = cur.pts_per_extra_hour / 100 * VAL
print(cur.round(3).to_string())

# is the 6-8h dip the overnight window for morning shifts? (Cleveland, worked after a <24h drop)
cle = s[(s.n_fail > 0) & s.last_band.isin(['2_<4h', '3_4-24h'])].copy()
cle['lb'] = pd.cut(cle.last_lead, [0, 2, 4, 6, 8, 12, 24], right=False)
cle['type'] = np.where(cle['Shift Type'] == 'am', 'AM shift', 'PM/NOC/custom')
print("Cleveland: worked after a <24h drop, by notice and shift type")
print(cle.pivot_table(index='lb', columns='type', values='worked', aggfunc=['size', 'mean'], observed=True).round(3).to_string())

# ================================================================================================ R6 NCNS ceiling
hdr("R6. NCNS RECOVERY CEILING (Cleveland): what if an NCNS became known at start, or 2h before?")
fl = s[s.n_fail > 0].copy(); fl['lead_last'] = fl.last_lead
n_ncns = (fl.last_band == '1_NCNS').sum()
w_ncns = fl[fl.last_band == '1_NCNS'].worked.mean()
w_0_1 = fl[(fl.last_band == '2_<4h') & (fl.lead_last < 1)].worked.mean()
w_1_3 = fl[(fl.last_band == '2_<4h') & (fl.lead_last >= 1) & (fl.lead_last < 3)].worked.mean()
for lab, w in [('known at shift start (like a <1h cancel)', w_0_1), ('known ~2h before (like a 1-3h cancel)', w_1_3)]:
    print(f"{lab:<42}: worked {w_ncns:.3f} -> {w:.3f}; if ALL NCNS converted: +{(w - w_ncns) * n_ncns * ANNUAL:,.0f} shifts/yr = ${(w - w_ncns) * n_ncns * ANNUAL * VAL:,.0f}/yr; "
          f"vs preventing all NCNS outright: ${fl[(fl.last_band == '1_NCNS') & (fl.state == 'unfilled')].gmv.sum() * TAKE * ANNUAL:,.0f}/yr")

if OUT:
    P0.to_csv(f"{OUT}/deepdive_prize_matrix.csv"); T3.to_csv(f"{OUT}/deepdive_warning_signs.csv"); cur.to_csv(f"{OUT}/deepdive_value_of_notice.csv")
