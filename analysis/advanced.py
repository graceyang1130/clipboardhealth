"""Advanced analyses on clean data.

Usage: python advanced.py <clean_dir> [out_dir]

  A. Do penalties change behaviour?  Regression discontinuity at the 2021 policy's own cut-offs:
     a cancel at 3h55m counted as a call-off (heavier strike) and one at 4h05m as a standard cancel; a cancel at 71h55m
     earned a standard strike and one at 72h05m none. Cancels do not bunch at either cut-off (p3.py), so which side a
     worker lands on is as good as random -- comparing the two sides isolates what the heavier strike did to later behaviour.
  B. Better reliability estimates for thin histories (empirical Bayes): every worker starts at a fitted prior risk and
     moves toward their own record as bookings accumulate; prior and update speed are fitted on Nov 1-15, scored on Nov 16-30.
  C. Does higher pay cause more refills? Late-cancelled Cleveland shifts compared within the same facility x role x shift
     type, controlling for notice, clock time of the drop, day, month and bench depth.
  D. How much late-failure risk sits with the worker vs the facility vs the day? Crossed random-effects variance
     decomposition (EM / backfitting; validated on simulated data with known answers first).
"""
import sys, os, warnings, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import booking_features, at_decision
warnings.filterwarnings('ignore')
S = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
H = lambda x: pd.Timedelta(hours=x); D_ = lambda x: pd.Timedelta(days=x)
TZ = 'America/New_York'
hdr = lambda t: print("\n" + "=" * 120 + f"\n{t}\n" + "=" * 120)
cl = pd.read_pickle(f"{S}/claims.pkl"); cx = pd.read_pickle(f"{S}/cx2.pkl"); bk = pd.read_pickle(f"{S}/bk2.pkl"); s = pd.read_pickle(f"{S}/cle_shifts.pkl")
c = booking_features(cl, cx, with_holds=False)
saved = {}

# =============================================================================================== A. deterrence (RD)
hdr("A. DO PENALTIES CHANGE BEHAVIOUR?  regression discontinuity at the policy cut-offs")


def times_by_worker(df):
    return {w: np.sort(g['at'].values) for w, g in df.groupby('Worker ID')}


def count_in(ev_times, workers, t, lo, hi):
    """count of each worker's events with time in (t+lo, t+hi]"""
    out = np.zeros(len(t), dtype=int)
    for w, idx in pd.Series(np.arange(len(t))).groupby(workers).indices.items():
        a = ev_times.get(w)
        if a is None: continue
        out[idx] = np.searchsorted(a, t[idx] + hi, side='right') - np.searchsorted(a, t[idx] + lo, side='right')
    return out


ev = cx[(cx.Action == 'WORKER_CANCEL')].dropna(subset=['Worker ID']).copy()
T_fail = times_by_worker(cx[cx.late_fail]); T_ncns = times_by_worker(cx[cx.band == '1_NCNS']); T_book = times_by_worker(bk.dropna(subset=['Worker ID']))
t = ev['at'].values; w = ev['Worker ID'].values
ev['late_fails_next30d'] = count_in(T_fail, w, t, np.timedelta64(0, 's'), np.timedelta64(30, 'D'))
ev['ncns_next30d'] = count_in(T_ncns, w, t, np.timedelta64(0, 's'), np.timedelta64(30, 'D'))
ev['any_late_next30d'] = (ev.late_fails_next30d > 0).astype(float)
ev['bookings_next14d'] = count_in(T_book, w, t, np.timedelta64(0, 's'), np.timedelta64(14, 'D'))
ev['late_fails_prior30d'] = count_in(T_fail, w, t, -np.timedelta64(30, 'D'), -np.timedelta64(1, 's'))
ev['bookings_prior14d'] = count_in(T_book, w, t, -np.timedelta64(14, 'D'), -np.timedelta64(1, 's'))


def rd(df, y, bw):
    d = df[df.r.abs() <= bw].copy()
    d['D'] = (d.r < 0).astype(float)
    wt = (1 - d.r.abs() / bw).clip(lower=1e-6)
    X = pd.DataFrame({'const': 1.0, 'D': d.D, 'r': d.r, 'Dr': d.D * d.r})
    m = sm.WLS(d[y].astype(float).values, X, weights=wt).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['Worker ID'])[0]})
    lo, hi = m.conf_int().loc['D']
    return dict(n=len(d), control_mean=d.loc[d.D == 0, y].mean(), effect=m.params['D'], lo=lo, hi=hi, p=m.pvalues['D'])


rows = []
for T, lab, bws in [(4, 'call-off (<4h) vs standard cancel (4h+)', [1, 2]), (72, 'standard strike (<=72h) vs no strike (>72h)', [4, 8])]:
    for bw in bws:
        win_fail = ev[(ev['at'] >= '2021-10-01') & (ev['at'] < '2022-01-01')].assign(r=lambda d: d.lead - T)
        win_book = ev[(ev['at'] >= '2021-10-01') & (ev['at'] < '2021-11-06')].assign(r=lambda d: d.lead - T)
        for y, df in [('late_fails_next30d', win_fail), ('any_late_next30d', win_fail), ('ncns_next30d', win_fail), ('bookings_next14d', win_book),
                      ('late_fails_prior30d', win_fail), ('bookings_prior14d', win_book)]:
            rows.append({'cut-off': lab, 'bandwidth_h': bw, 'outcome': y + ('  [balance check: should be ~0]' if 'prior' in y else ''), **rd(df, y, bw)})
A = pd.DataFrame(rows)
print("effect = value just below the cut-off (heavier treatment) minus just above; 95% CI clustered by worker")
print(A.round(3).to_string(index=False)); saved['A_deterrence_rd'] = A

# =============================================================================================== B. empirical Bayes reliability
hdr("B. RELIABILITY ESTIMATES FOR THIN HISTORIES (empirical Bayes) vs the current cut-off rules  -- 24h before start")
d = at_decision(c, cx, 24); d = d[d.start >= '2021-11-01'].copy()
n = d.prior_n_t.values.astype(float)
k = {'late_fail': np.rint(np.nan_to_num(d.rate_lf.values) * n), 'ncns': np.rint(np.nan_to_num(d.rate_ncns.values) * n)}
train = (d.start < '2021-11-16').values; test = ~train
bal = ((d.prior_n_t < 5) | ((d.prior_n_t >= 5) & (d.rate_lf >= 0.10))).values


def eb_fit(y, kk):
    base = y[train].mean(); best = None
    for m0 in np.linspace(0.5 * base, 3.0 * base, 26):
        for s0 in [0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50]:
            p = np.clip((m0 * s0 + kk) / (s0 + n), 1e-4, 1 - 1e-4)
            ll = np.sum(y[train] * np.log(p[train]) + (1 - y[train]) * np.log(1 - p[train]))
            if best is None or ll > best[0]: best = (ll, m0, s0)
    _, m0, s0 = best
    return m0, s0, (m0 * s0 + kk) / (s0 + n)


def at_workload(score_, y, q):
    thr = np.quantile(score_[test], 1 - q); f = score_[test] >= thr
    return f.mean(), (f & y[test]).sum() / y[test].sum(), (f & y[test]).sum() / f.sum()


rows = []
for ycol in ['late_fail', 'ncns']:
    y = d[ycol].values.astype(bool)
    m0, s0, post = eb_fit(y.astype(float), k[ycol])
    raw = np.where(n >= 5, k[ycol] / np.maximum(n, 1), -1.0)            # current approach: raw rate, only trusted with 5+ bookings
    print(f"{ycol}: fitted prior risk for a worker with no history = {m0:.3f} (Nov 1-15 base rate {y[train].mean():.3f}); "
          f"prior weight = {s0:g} bookings (a worker's own record outweighs the prior after ~{s0:g} bookings)")
    bf, br, bp = bal[test].mean(), (bal[test] & y[test]).sum() / y[test].sum(), (bal[test] & y[test]).sum() / bal[test].sum()
    for q in [0.05, 0.10, 0.20, round(bf, 3)]:
        ef, er, ep = at_workload(post, y, q)
        rf, rr_, rp = at_workload(raw + 1e-9 * np.random.default_rng(0).random(len(raw)), y, q)
        rows.append({'outcome': ycol, 'workload': q, 'EB: caught': er, 'EB: precision': ep, 'raw rate (5+ bkgs): caught': rr_, 'raw rate: precision': rp})
    rows.append({'outcome': ycol, 'workload': f"balanced rule ({bf:.3f})", 'EB: caught': np.nan, 'EB: precision': np.nan, 'raw rate (5+ bkgs): caught': br, 'raw rate: precision': bp})
    if ycol == 'late_fail': post_lf = post
B = pd.DataFrame(rows)
print("\nTest half (Nov 16-30): share of failures caught and precision at the same workload"); print(B.round(3).to_string(index=False))
saved['B_empirical_bayes'] = B
# fairness: who gets flagged at the balanced rule's workload
y = d.late_fail.values.astype(bool); q = bal[test].mean()
ebf = post_lf >= np.quantile(post_lf[test], 1 - q)
grp = np.select([n == 0, (n >= 1) & (n <= 4) & (k['late_fail'] == 0), (n >= 1) & (n <= 4), (n >= 20) & (k['late_fail'] == 0), (n >= 5) & (k['late_fail'] / np.maximum(n, 1) >= 0.10)],
                ['no history', '1-4 bookings, clean record', '1-4 bookings, has failed', '20+ bookings, clean record', '5+ bookings, late-fail >=10%'], 'other established')
G = pd.DataFrame({'group': grp[test], 'actual late-fail rate': y[test], 'flagged by balanced rule': bal[test], 'flagged by EB (same workload)': ebf[test]})
G = G.groupby('group').agg(bookings=('actual late-fail rate', 'size'), actual_late_fail_rate=('actual late-fail rate', 'mean'),
                           flagged_by_balanced_rule=('flagged by balanced rule', 'mean'), flagged_by_EB=('flagged by EB (same workload)', 'mean'))
print("\nWho gets flagged (test half, same workload):"); print(G.round(3).to_string()); saved['B_who_is_flagged'] = G

# =============================================================================================== C. pay -> refill
hdr("C. DOES HIGHER PAY CAUSE MORE REFILLS?  (Cleveland shifts dropped <24h before start)")
k3 = ['Facility ID', 'Agent Req', 'Shift Type']
s['rel'] = s.charge / s.groupby(k3).charge.transform('median')
L = s[(s.n_fail > 0) & s.last_band.isin(['2_<4h', '3_4-24h']) & s.rel.notna()].copy()
L['rel_bin'] = pd.cut(L.rel, [0, 0.95, 1.05, 1.15, 10], labels=['below', 'at', 'above', 'well_above']).astype(str)
L['relpct10'] = (L.rel - 1) * 10                                                   # per +10% above the cell norm
L['lead_bin'] = pd.cut(L.last_lead, [0, 1, 2, 4, 8, 12, 24], right=False).astype(str)
hr = L.last_at.dt.tz_localize('UTC').dt.tz_convert(TZ).dt.hour
L['clock'] = pd.cut(hr, [-1, 4, 8, 16, 21, 24], labels=['0-5', '5-9', '9-17', '17-22', '22-24']).astype(str)
L['dow'] = L.Start_loc.dt.dayofweek.astype(str); L['month'] = L.Start_loc.dt.month.astype(str)
L['cell'] = L['Facility ID'] + '|' + L['Agent Req'] + '|' + L['Shift Type']
wk = s[s.worked][['Facility ID', 'Worker ID', 'Start']]
L['bench'] = [wk[(wk['Facility ID'] == f_) & (wk.Start < st) & (wk.Start >= st - D_(30))]['Worker ID'].nunique() for f_, st in zip(L['Facility ID'], L.Start)]
L['lbench'] = np.log1p(L.bench); L['worked_f'] = L.worked.astype(float)
ctrl = "C(lead_bin) + C(clock) + C(dow) + C(month)"
specs = [('1 raw', "worked_f ~ C(rel_bin, Treatment('at'))"),
         ('2 + notice, clock, day, month, type, role', f"worked_f ~ C(rel_bin, Treatment('at')) + {ctrl} + C(Q('Shift Type')) + C(Q('Agent Req'))"),
         ('3 + facility fixed effects + bench', f"worked_f ~ C(rel_bin, Treatment('at')) + {ctrl} + C(Q('Shift Type')) + C(Q('Agent Req')) + C(Q('Facility ID')) + lbench"),
         ('4 same facility x role x type (cell FE) + bench', f"worked_f ~ C(rel_bin, Treatment('at')) + {ctrl} + C(cell) + lbench")]
rows = []
for lab, f_ in specs:
    m = smf.ols(f_, L).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(L['Facility ID'])[0]})
    ci = m.conf_int()
    row = {'model': lab, 'n': int(m.nobs)}
    for b in ['below', 'above', 'well_above']:
        kname = f"C(rel_bin, Treatment('at'))[T.{b}]"
        row[f'{b} vs at-norm (pts)'] = f"{m.params[kname] * 100:+.1f} [{ci.loc[kname, 0] * 100:+.1f},{ci.loc[kname, 1] * 100:+.1f}]"
    rows.append(row)
m = smf.ols(f"worked_f ~ relpct10 + {ctrl} + C(cell) + lbench", L).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(L['Facility ID'])[0]})
C_ = pd.DataFrame(rows)
print(f"share worked by pay bin (raw): {L.groupby('rel_bin').worked.agg(['size', 'mean']).round(3).to_dict('index')}")
print(C_.to_string(index=False))
lo, hi = m.conf_int().loc['relpct10']
print(f"dose-response (model 4 controls): each +10% pay above the cell norm -> {m.params['relpct10'] * 100:+.1f} pts worked [{lo * 100:+.1f}, {hi * 100:+.1f}], p={m.pvalues['relpct10']:.3f}")
print(f"bench depth (model 4): each doubling of recent workers at the facility -> {m.params['lbench'] * np.log(2) * 100:+.1f} pts worked")
saved['C_pay_refill'] = C_

# =============================================================================================== D. variance decomposition
hdr("D. HOW MUCH LATE-FAILURE RISK SITS WITH THE WORKER vs THE FACILITY vs THE DAY?")


def crossed_vc(y, groups, iters=500, tol=1e-10):
    """EM / backfitting for y = mu + a_g1 + b_g2 + ... + e with independent crossed random intercepts"""
    N = len(y); mu = y.mean(); codes = [pd.factorize(g)[0] for g in groups]; K = len(codes)
    ng = [np.bincount(cd) for cd in codes]; eff = [np.zeros(len(x)) for x in ng]
    s2 = [y.var() / (K + 1)] * K; se2 = y.var() / (K + 1); v = [None] * K
    for it in range(iters):
        old = np.array(s2 + [se2])
        for j in range(K):
            other = sum(eff[i][codes[i]] for i in range(K) if i != j)
            sg = np.bincount(codes[j], weights=y - mu - other, minlength=len(ng[j]))
            lam = se2 / s2[j]; eff[j] = sg / (ng[j] + lam); v[j] = se2 / (ng[j] + lam)
            s2[j] = np.mean(eff[j] ** 2 + v[j])
        fit = sum(eff[i][codes[i]] for i in range(K)); mu = np.mean(y - fit)
        se2 = (np.sum((y - mu - fit) ** 2) + sum(np.sum(ng[i] * v[i]) for i in range(K))) / N
        if np.max(np.abs(np.array(s2 + [se2]) - old)) < tol: break
    return mu, s2, se2


# self-test on simulated data shaped like ours (binary outcome, known worker/facility/day SDs in probability points)
rng = np.random.default_rng(1); N = len(c)
gw, gf, gd = rng.integers(0, 6800, N), rng.integers(0, 1100, N), rng.integers(0, 61, N)
true = (0.08, 0.03, 0.01)
p = np.clip(0.10 + rng.normal(0, true[0], 6800)[gw] + rng.normal(0, true[1], 1100)[gf] + rng.normal(0, true[2], 61)[gd], 0.001, 0.999)
_, s2_, _ = crossed_vc((rng.random(N) < p).astype(float), [gw, gf, gd])
print(f"self-test: true SDs worker/facility/day = {true} -> recovered {tuple(np.round(np.sqrt(s2_), 3))}")
rows = []
c['day'] = c.start.dt.date
for lab, ycol, sub in [('late failure, all bookings', 'late_fail', c), ('NCNS, all bookings', 'ncns', c), ('late failure, Cleveland only', 'late_fail', c[c.cleveland])]:
    mu, s2, se2 = crossed_vc(sub[ycol].astype(float).values, [sub['Worker ID'].values, sub['Facility ID'].values, sub['day'].values])
    tot = sum(s2) + se2; sysv = sum(s2)
    rows.append({'outcome': lab, 'mean rate': mu, 'SD worker (pts)': np.sqrt(s2[0]) * 100, 'SD facility (pts)': np.sqrt(s2[1]) * 100, 'SD day (pts)': np.sqrt(s2[2]) * 100,
                 'worker share of systematic risk': s2[0] / sysv, 'facility share': s2[1] / sysv, 'day share': s2[2] / sysv,
                 'systematic share of all variation': sysv / tot})
Dv = pd.DataFrame(rows).set_index('outcome')
print("SD = spread of true underlying rates across workers / facilities / days, in percentage points")
print(Dv.round(3).to_string()); saved['D_variance'] = Dv

if OUT:
    for kname, df in saved.items(): df.to_csv(f"{OUT}/advanced_{kname}.csv")
