"""Which booking-time signals can we actually attribute as higher risk?  (attribution tests -- deliberately NOT a risk score)

Usage: python indicators.py <work_dir> [out_dir]      (needs clean build.py outputs)

Main sample  all-markets bookings for shifts starting Nov 1-30 (>= 1 month of history before them) that were still on the
             books 24h before start, so every booking had the same 24h window in which a late failure could happen.
Outcome      late failure in the final 24h (NCNS or worker cancel <24h). NCNS alone is reported as a secondary outcome.

Every indicator is a contrast against a named reference group and goes through four tests:
  T1 raw         failure rate in the group vs the reference (relative risk, 95% CI)
  T2 adjusted    still higher after accounting for every other indicator at once (logistic regression, SEs clustered by worker)
  T3 same worker for booking-context signals only: does the same worker fail more on these bookings than on their other
                 bookings? (linear probability model with worker fixed effects, SEs clustered by worker)
  T4 stable      same direction in Nov 1-15 and Nov 16-30 starts, and in Cleveland and in other markets
The verdict comes from verdict() below; nothing is hand-labelled.

Shift attributes (role, shift type, day, length, pay, posting notice) exist only for Cleveland. They get T1/T2 on Cleveland
bookings (Oct-Nov starts, worker-history signals as controls) and T4 = same direction in Oct-Nov and Dec-Jan shift-level data.
Bookings made <24h before start are tested separately (they never pass through the 24h mark).
"""
import sys, os, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import booking_features
warnings.filterwarnings('ignore')
S = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)

cl = pd.read_pickle(f"{S}/claims.pkl"); cx = pd.read_pickle(f"{S}/cx2.pkl"); s = pd.read_pickle(f"{S}/cle_shifts.pkl")
c = booking_features(cl, cx, with_holds=True)

# ---------------------------------------------------------------- categorical versions of each signal
c['cat_calloff'] = np.select([c.prior_calloff == 0, c.prior_calloff == 1], ['0', '1'], '2+')
c['cat_ncns'] = np.select([c.prior_ncns == 0, c.prior_ncns == 1], ['0', '1'], '2+')
c['cat_late424'] = np.where(c.prior_late_cancel >= 1, '1+', '0')
c['cat_early'] = np.select([c.prior_early_cancel == 0, c.prior_early_cancel == 1], ['0', '1'], '2+')
c['cat_volume'] = pd.cut(c.prior_n, [-1, 0, 4, 19, 1e9], labels=['0', '1-4', '5-19', '20+']).astype(str)
c['cat_heavy'] = np.where(c.prior_n >= 20, '20+', '<20')
c['cat_rate'] = np.select([c.prior_n < 5, c.prior_rate < 0.10, c.prior_rate < 0.25], ['n<5', '<10%', '10-25%'], '>=25%')
c['cat_tenure'] = pd.cut(c.tenure_d, [-1e9, 30, 180, 1e9], labels=['<30d', '30-180d', '180d+']).astype(str)
c['cat_firstfac'] = np.where(c.first_fac, 'first', 'repeat')
c['cat_lead'] = pd.cut(c.claim_lead, [0, 24, 72, 168, 336, 1e9], right=False, labels=['<24h', '24-72h', '3-7d', '1-2wk', '2wk+']).astype(str)
c['cat_holds'] = pd.cut(c.holds, [-1, 0, 5, 1e9], labels=['0', '1-5', '6+']).astype(str)
c['half'] = np.where(c.start < '2021-11-16', 'Nov 1-15', 'Nov 16-30')

# (column, reference, label for the signal family, in which model it is tested)
SIGNALS = [
    ('cat_calloff', '0', 'past call-offs (<4h cancels)', 'main'),
    ('cat_ncns', '0', 'past NCNS', 'main'),
    ('cat_late424', '0', 'past 4-24h cancels', 'main'),
    ('cat_early', '0', 'past early cancels (>72h)', 'main'),
    ('cat_rate', '<10%', 'past late-failure rate (>=5 bookings)', 'alt'),
    ('cat_volume', '5-19', 'past resolved bookings (volume)', 'main'),
    ('cat_tenure', '180d+', 'account age', 'main'),
    ('cat_firstfac', 'repeat', 'first booking at this facility', 'main'),
    ('cat_lead', '1-2wk', 'how far ahead it was booked', 'main'),
    ('cat_holds', '1-5', 'other shifts held when booking', 'main'),
]
CONTEXT = ['cat_firstfac', 'cat_lead', 'cat_holds']       # vary within a worker -> get the same-worker test
term = lambda col, ref: f"C({col}, Treatment('{ref}'))"
MAIN = [x for x in SIGNALS if x[3] == 'main']
ALT = [('cat_rate', '<10%'), ('cat_heavy', '<20'), ('cat_early', '0'), ('cat_tenure', '180d+'), ('cat_firstfac', 'repeat'), ('cat_lead', '1-2wk'), ('cat_holds', '1-5')]


def rr_ci(x1, n1, x0, n0):
    p1, p0 = x1 / n1, x0 / n0
    rr = p1 / p0; se = np.sqrt(max(1 / max(x1, 1) - 1 / n1 + 1 / max(x0, 1) - 1 / n0, 0))
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se), p1, p0


def raw(df, col, lvl, ref, y='late_fail'):
    a, b = df[df[col] == lvl], df[df[col] == ref]
    if len(a) < 30 or len(b) < 30: return (np.nan,) * 5
    return rr_ci(a[y].sum(), len(a), b[y].sum(), len(b))


def logit(df, y, terms):
    f = f"{y} ~ " + " + ".join(term(c_, r_) for c_, r_ in terms)
    d = df.copy(); d[y] = d[y].astype(int)
    return smf.logit(f, d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d['Worker ID'])[0]})


def fe_lpm(df, y, cats):
    d = df[df.groupby('Worker ID')['Worker ID'].transform('size') >= 2].copy()
    X = pd.get_dummies(d[[c_ for c_, _ in cats]].astype(str), prefix_sep='=')
    X = X.drop(columns=[f"{c_}={r_}" for c_, r_ in cats]).astype(float)
    Y = d[y].astype(float)
    g = d['Worker ID']
    Xd = X - X.groupby(g).transform('mean'); Yd = Y - Y.groupby(g).transform('mean')
    m = sm.OLS(Yd.values, Xd.values).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(g)[0]})
    ci = m.conf_int()
    return {name: (m.params[i], ci[i][0], ci[i][1]) for i, name in enumerate(X.columns)}, d['Worker ID'].nunique()


def verdict(rr, lo, hi, a_lo, a_hi, splits, fe=None):
    if not np.isfinite(rr): return 'too few bookings to judge'
    if lo > 1: d = 1
    elif hi < 1: d = -1
    else: return 'no clear difference'
    word = 'HIGHER RISK' if d == 1 else 'LOWER RISK'
    checks = [('explained away by other signals', (a_lo > 1) if d == 1 else (a_hi < 1)),
              ('direction not stable across periods/markets', all(np.isfinite(x) and ((x > 1) if d == 1 else (x < 1)) for x in splits)),
              ('same worker does not differ', True if fe is None else ((fe[1] > 0) if d == 1 else (fe[2] < 0))),
              ('effect smaller than 1.3x', rr >= 1.3 if d == 1 else rr <= 1 / 1.3)]
    failed = [msg for msg, ok in checks if not ok]
    if not failed: return f'ATTRIBUTABLE: {word}'
    if 'explained away by other signals' in failed: return f'not independent ({word.lower()} raw, but explained away by other signals)'
    return f'{word.lower()}, weak: ' + '; '.join(failed)


# ================================================================ main sample
e = c[c.start >= '2021-11-01']
A = e[e.active24].copy()
print("=" * 120)
print(f"MAIN SAMPLE: {len(A):,} bookings (Nov starts, still active 24h before start), {A['Worker ID'].nunique():,} workers; "
      f"late-failure rate {A.late_fail.mean():.3f}, NCNS rate {A.ncns.mean():.3f}")
m_main = logit(A, 'late_fail', [(c_, r_) for c_, r_, _, _ in MAIN])
m_alt = logit(A, 'late_fail', ALT)
m_ncns = logit(A, 'ncns', [(c_, r_) for c_, r_, _, _ in MAIN])
fe, fe_workers = fe_lpm(A, 'late_fail', [(c_, r_) for c_, r_, _, _ in SIGNALS if c_ in CONTEXT])
print(f"same-worker test uses {fe_workers:,} workers with >=2 bookings in the sample")

rows = []
for col, ref, lab, model in SIGNALS:
    m = m_alt if model == 'alt' else m_main
    for lvl in sorted(A[col].unique()):
        if lvl == ref: continue
        rr, lo, hi, p1, p0 = raw(A, col, lvl, ref)
        k = f"{term(col, ref)}[T.{lvl}]"
        orr, olo, ohi = (np.exp(m.params[k]), *np.exp(m.conf_int().loc[k])) if k in m.params else (np.nan,) * 3
        nk = f"{term(col, ref)}[T.{lvl}]"
        n_or = np.exp(m_ncns.params[nk]) if (model == 'main' and nk in m_ncns.params) else np.nan
        splits = [raw(A[A.half == h], col, lvl, ref)[0] for h in ['Nov 1-15', 'Nov 16-30']] + \
                 [raw(A[A.cleveland == f_], col, lvl, ref)[0] for f_ in [True, False]]
        fe_ = fe.get(f"{col}={lvl}") if col in CONTEXT else None
        rows.append({'signal': lab, 'group': lvl, 'vs': ref, 'share_of_bookings': (A[col] == lvl).mean(),
                     'rate': p1, 'ref_rate': p0, 'RR': rr, 'RR_lo': lo, 'RR_hi': hi, 'adj_OR': orr, 'adj_lo': olo, 'adj_hi': ohi,
                     'same_worker_pp': fe_[0] * 100 if fe_ else np.nan, 'sw_lo': fe_[1] * 100 if fe_ else np.nan, 'sw_hi': fe_[2] * 100 if fe_ else np.nan,
                     'RR_Nov1-15': splits[0], 'RR_Nov16-30': splits[1], 'RR_Cleveland': splits[2], 'RR_other_mkts': splits[3],
                     'adj_OR_for_NCNS': n_or, 'verdict': verdict(rr, lo, hi, olo, ohi, splits, fe_)})

# ---------------------------------------------------------------- bookings made <24h before start
B = e[e.active24 | (e.claim_lead < 24)].copy(); B['cat_short'] = np.where(B.claim_lead < 24, '<24h', 'on books at 24h')
ctrl = [('cat_calloff', '0'), ('cat_ncns', '0'), ('cat_late424', '0'), ('cat_early', '0'), ('cat_volume', '5-19'), ('cat_tenure', '180d+'), ('cat_firstfac', 'repeat'), ('cat_holds', '1-5')]
mb = logit(B, 'late_fail', [('cat_short', 'on books at 24h')] + ctrl)
feb, _ = fe_lpm(B, 'late_fail', [('cat_short', 'on books at 24h'), ('cat_firstfac', 'repeat'), ('cat_holds', '1-5')])
rr, lo, hi, p1, p0 = raw(B, 'cat_short', '<24h', 'on books at 24h')
k = f"{term('cat_short', 'on books at 24h')}[T.<24h]"; orr, olo, ohi = np.exp(mb.params[k]), *np.exp(mb.conf_int().loc[k])
splits = [raw(B[B.half == h], 'cat_short', '<24h', 'on books at 24h')[0] for h in ['Nov 1-15', 'Nov 16-30']] + \
         [raw(B[B.cleveland == f_], 'cat_short', '<24h', 'on books at 24h')[0] for f_ in [True, False]]
fe_ = feb['cat_short=<24h']
rows.append({'signal': 'booked inside the last 24h', 'group': '<24h', 'vs': 'still booked at 24h', 'share_of_bookings': (B.cat_short == '<24h').mean(),
             'rate': p1, 'ref_rate': p0, 'RR': rr, 'RR_lo': lo, 'RR_hi': hi, 'adj_OR': orr, 'adj_lo': olo, 'adj_hi': ohi,
             'same_worker_pp': fe_[0] * 100, 'sw_lo': fe_[1] * 100, 'sw_hi': fe_[2] * 100, 'RR_Nov1-15': splits[0], 'RR_Nov16-30': splits[1],
             'RR_Cleveland': splits[2], 'RR_other_mkts': splits[3], 'adj_OR_for_NCNS': np.nan, 'verdict': verdict(rr, lo, hi, olo, ohi, splits, fe_)})
R = pd.DataFrame(rows)

# ================================================================ Cleveland shift attributes
k3 = ['Facility ID', 'Agent Req', 'Shift Type']
s = s.copy(); s['rel'] = s.charge / s.groupby(k3).charge.transform('median')
s['cat_role'] = s['Agent Req']; s['cat_type'] = s['Shift Type']
s['cat_day'] = np.where(s.Start_loc.dt.dayofweek.isin([6, 0]), 'Sun/Mon', 'Tue-Sat')
s['cat_len'] = np.where(s.dur < 8, '<8h', '8h+')
s['cat_pay'] = np.select([s.rel.isna(), s.rel < 0.95, s.rel <= 1.05], ['unknown', 'below norm', 'at norm'], 'above norm')
s['cat_post'] = np.where(s.post_lead_h < 72, 'posted <72h ahead', 'posted 72h+ ahead')
s['period'] = np.where(s.Start_loc < '2021-12-01', 'Oct-Nov', 'Dec-Jan')
SHIFT = [('cat_role', 'CNA', 'role'), ('cat_type', 'pm', 'shift type'), ('cat_day', 'Tue-Sat', 'start day'),
         ('cat_len', '8h+', 'shift length'), ('cat_pay', 'at norm', 'pay vs facility norm'), ('cat_post', 'posted 72h+ ahead', 'how far ahead it was posted')]
attrs = ['Shift ID'] + [x[0] for x in SHIFT]
CB = c[c.cleveland & c.active24].merge(s[attrs], on='Shift ID')
CB = CB[CB.cat_type != 'custom']
ctrl_c = [('cat_calloff', '0'), ('cat_ncns', '0'), ('cat_volume', '5-19'), ('cat_tenure', '180d+'), ('cat_firstfac', 'repeat'), ('cat_lead', '1-2wk')]
mc = logit(CB, 'late_fail', [(a, r) for a, r, _ in SHIFT] + ctrl_c)
sb = s[s.ever_booked].copy(); sb['lf'] = sb.n_late > 0
print(f"CLEVELAND SHIFT-ATTRIBUTE SAMPLE: {len(CB):,} bookings (Oct-Nov starts, active 24h before start); late-failure rate {CB.late_fail.mean():.3f}; "
      f"stability check on {len(sb):,} booked shifts Oct-Jan")
for col, ref, lab in SHIFT:
    for lvl in sorted(CB[col].unique()):
        if lvl == ref or lvl == 'unknown': continue
        rr, lo, hi, p1, p0 = raw(CB, col, lvl, ref)
        k = f"{term(col, ref)}[T.{lvl}]"
        orr, olo, ohi = (np.exp(mc.params[k]), *np.exp(mc.conf_int().loc[k])) if k in mc.params else (np.nan,) * 3
        splits = [raw(sb[sb.period == p], col, lvl, ref, y='lf')[0] for p in ['Oct-Nov', 'Dec-Jan']]
        rows.append({'signal': f'[Cleveland] {lab}', 'group': lvl, 'vs': ref, 'share_of_bookings': (CB[col] == lvl).mean(), 'rate': p1, 'ref_rate': p0,
                     'RR': rr, 'RR_lo': lo, 'RR_hi': hi, 'adj_OR': orr, 'adj_lo': olo, 'adj_hi': ohi, 'same_worker_pp': np.nan, 'sw_lo': np.nan, 'sw_hi': np.nan,
                     'RR_Nov1-15': np.nan, 'RR_Nov16-30': np.nan, 'RR_Cleveland': splits[0], 'RR_other_mkts': splits[1], 'adj_OR_for_NCNS': np.nan,
                     'verdict': verdict(rr, lo, hi, olo, ohi, splits)})
R = pd.DataFrame(rows)
R = R.rename(columns={'RR_Cleveland': 'RR_Cle / Oct-Nov', 'RR_other_mkts': 'RR_other / Dec-Jan'})

# ================================================================ print
fmt = lambda x, l, h: f"{x:.2f} [{l:.2f}-{h:.2f}]" if np.isfinite(x) else "-"
show = pd.DataFrame({
    'signal': R.signal, 'group': R.group + ' vs ' + R.vs, 'share': R.share_of_bookings.round(3),
    'rate': R.rate.round(3), 'ref': R.ref_rate.round(3),
    'T1 raw RR [95% CI]': [fmt(a, b, c_) for a, b, c_ in zip(R.RR, R.RR_lo, R.RR_hi)],
    'T2 adjusted OR [95% CI]': [fmt(a, b, c_) for a, b, c_ in zip(R.adj_OR, R.adj_lo, R.adj_hi)],
    'T3 same-worker pp [CI]': [f"{a:+.1f} [{b:+.1f},{c_:+.1f}]" if np.isfinite(a) else "-" for a, b, c_ in zip(R.same_worker_pp, R.sw_lo, R.sw_hi)],
    'T4 RR splits': [" / ".join(f"{x:.2f}" if np.isfinite(x) else "-" for x in r) for r in R[['RR_Nov1-15', 'RR_Nov16-30', 'RR_Cle / Oct-Nov', 'RR_other / Dec-Jan']].values],
    'NCNS adj OR': R.adj_OR_for_NCNS.round(2),
    'verdict': R.verdict})
print("=" * 120)
print("T4 splits = Nov 1-15 / Nov 16-30 / Cleveland / other markets  (Cleveland shift attributes: - / - / Oct-Nov / Dec-Jan)")
print(show.to_string(index=False))
if OUT: R.to_csv(f"{OUT}/indicators.csv", index=False)
