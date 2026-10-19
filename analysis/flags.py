"""Early-warning flags for each failure type (NCNS, <4h cancel, 4-24h cancel, any late failure).
Simple, transparent rules -- not a risk score -- judged on catches vs false alarms.  Replaces ncns_flags.py.

Usage: python flags.py <clean_dir> [out_dir]

Design
  * Decision points: 24h before start (time to arrange cover) and 4h before start (more information, less time).
    A booking is in the sample if it is still on the books at the decision point; every signal uses only what was
    logged before that moment.
  * Metrics per rule: flagged share of bookings (workload), recall (share of failures caught), precision (share of
    flagged bookings that really fail), lift (precision / base rate).
  * Rules = single signals plus every "A AND B" / "A OR B" pair. Chosen on Nov 1-15 starts, scored on Nov 16-30 (out of
    time), all markets.
  * "Balanced" rule = highest Youden's J (recall minus false-alarm rate) on Nov 1-15: the standard way to weigh missed
    failures and false alarms equally. "Tight" rule = highest precision that still catches >=10% (on Nov 1-15).
  * Benchmark only: a logistic model on the same signals, to show what simple rules give up.
"""
import sys, os, itertools, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import booking_features, at_decision
warnings.filterwarnings('ignore')
S = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 95)
H = lambda x: pd.Timedelta(hours=x)
TAKE = 0.22
cl = pd.read_pickle(f"{S}/claims.pkl"); cx = pd.read_pickle(f"{S}/cx2.pkl"); s = pd.read_pickle(f"{S}/cle_shifts.pkl")
c = booking_features(cl, cx, with_holds=False)
c['end_act'] = c.cancel_at.where(c.outcome == 'cancel', c.start)
c['lt4'] = c.band == '2_<4h'; c['c424'] = c.band == '3_4-24h'

TARGETS = {  # name: (label column, decision points)
    'NCNS': ('ncns', [24, 4]),
    '<4h cancel': ('lt4', [24, 4]),
    '4-24h cancel': ('c424', [24]),
    'any late failure (NCNS or <24h cancel)': ('late_fail', [24]),
}


def signals(d, offset_h):
    h5 = d.prior_n_t >= 5
    X = {
        '2+ past NCNS': d.k_ncns >= 2, '1+ past NCNS': d.k_ncns >= 1, 'NCNS in last 14 days': d.k_ncns_14d >= 1,
        'past NCNS rate >=10%': h5 & (d.rate_ncns >= 0.10),
        '2+ past call-offs': d.k_calloff >= 2, '1+ past call-off': d.k_calloff >= 1, 'call-off in last 14 days': d.k_calloff_14d >= 1,
        'past <4h-cancel rate >=10%': h5 & (d.rate_lt4 >= 0.10), 'past 4-24h-cancel rate >=10%': h5 & (d.rate_424 >= 0.10),
        'dropped a shift late in last 48h': d.k_drop_48h >= 1,
        'past late-fail rate >=25%': h5 & (d.rate_lf >= 0.25), 'past late-fail rate >=10%': h5 & (d.rate_lf >= 0.10),
        'fewer than 5 past bookings': d.prior_n_t < 5, 'no past bookings at all': d.prior_n_t == 0,
        'first visit to this facility': d.first_fac, 'booked 1-3 days ahead': d.claim_lead < 72,
        'account <30 days old': d.tenure_t < 30,
        'never worked a shift / none in 30 days': d.days_since_worked.isna() | (d.days_since_worked > 30),
        'back-to-back (shift 6-16h earlier)': d.b2b, 'double-booked (shift within 4h)': d.conflict,
        'holding no other shifts': d.holds_t == 0,
    }
    if offset_h < 24: X['booked inside the last 24h'] = d.claim_lead < 24
    return pd.DataFrame(X).astype(bool)


def score(flag, y):
    n = len(flag); f = flag.sum(); tp = (flag & y).sum(); P = max(y.sum(), 1)
    return {'flagged': f / n, 'recall': tp / P, 'precision': tp / max(f, 1), 'lift': (tp / max(f, 1)) / max(y.mean(), 1e-9),
            'youden': tp / P - (f - tp) / max(n - P, 1)}


def pareto(df):
    df = df.sort_values(['flagged', 'recall'], ascending=[True, False]); best = -1; keep = []
    for i, r in df.iterrows():
        if r.recall > best + 1e-9: keep.append(i); best = r.recall
    return df.loc[keep]


cache = {off: at_decision(c, cx, off) for off in [24, 4]}
summary = []
for tname, (ycol, offsets) in TARGETS.items():
    for OFF in offsets:
        d = cache[OFF]; d = d[d.start >= '2021-11-01']
        y = d[ycol].values.astype(bool); X = signals(d, OFF)
        train = (d.start < '2021-11-16').values; test = ~train
        print("\n" + "=" * 140)
        print(f"{tname.upper()}  |  flag raised {OFF}h before start  |  {len(d):,} bookings on the books  |  base rate {y.mean():.2%}  ({y.sum():,} events)")
        print("=" * 140)
        sig = pd.DataFrame([{'signal': k, **score(X[k].values, y)} for k in X.columns]).set_index('signal')
        print("top single signals (all Nov):"); print(sig.sort_values('lift', ascending=False).head(10)[['flagged', 'recall', 'precision', 'lift']].round(3).to_string())
        print("not signals (lift < 1.2):", ", ".join(sig[sig.lift < 1.2].index))
        cols = list(X.columns); rules = {k: X[k].values for k in cols}
        for a, b in itertools.combinations(cols, 2):
            rules[f"{a} AND {b}"] = X[a].values & X[b].values; rules[f"{a} OR {b}"] = X[a].values | X[b].values
        tr = pd.DataFrame([{'rule': k, **score(v[train], y[train])} for k, v in rules.items()]).set_index('rule')
        tr = tr[tr.flagged >= 0.005]
        front = pareto(tr)
        te = pd.DataFrame([{'rule': k, **{f"test_{m}": v for m, v in score(rules[k][test], y[test]).items()}} for k in front.index]).set_index('rule')
        F = front.join(te)
        rows = []
        for budget in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]:
            ok = F[F.flagged <= budget]
            if len(ok): r = ok.iloc[-1]; rows.append({'workload cap': f"<= {budget:.0%}", 'rule (chosen on Nov 1-15)': r.name, 'flagged': r.test_flagged,
                                                      'caught': r.test_recall, 'precision': r.test_precision, 'lift': r.test_lift})
        print("\nTRADE-OFF (test = Nov 16-30):"); print(pd.DataFrame(rows).round(3).to_string(index=False))
        bal = tr.sort_values('youden').iloc[-1].name
        tight_pool = tr[tr.recall >= 0.10]; tight = tight_pool.sort_values('precision').iloc[-1].name
        for lab, rn in [('BALANCED (max Youden J)', bal), ('TIGHT (max precision, >=10% caught)', tight)]:
            sc = score(rules[rn][test], y[test])
            print(f"{lab:<36}: '{rn}' -> flags {sc['flagged']:.1%}, catches {sc['recall']:.1%}, precision {sc['precision']:.1%} ({sc['lift']:.1f}x base)  [test]")
            summary.append({'target': tname, 'decision': f"{OFF}h before", 'pick': lab.split(' ')[0], 'rule': rn, 'base_rate': y[test].mean(),
                            'flagged': sc['flagged'], 'caught': sc['recall'], 'precision': sc['precision'], 'lift': sc['lift']})
        Z = X.astype(float).copy(); Z['const'] = 1.0
        m = sm.Logit(y[train].astype(float), Z[train]).fit(disp=0, method='bfgs', maxiter=500); p = m.predict(Z[test])
        bench = []
        for budget in [0.10, 0.20, 0.35]:
            mf = p >= np.quantile(p, 1 - budget); bench.append(f"{budget:.0%}: model {(mf & y[test]).sum() / y[test].sum():.0%}")
        print("benchmark model catches at the same workload ->", " | ".join(bench))

SUM = pd.DataFrame(summary)
print("\n" + "=" * 140); print("SUMMARY OF PICKS (test half, Nov 16-30)"); print("=" * 140)
print(SUM.round(3).to_string(index=False))
if OUT: SUM.to_csv(f"{OUT}/flags_summary.csv", index=False)

# ================================================================ the one flag that wins for every failure type
print("\n" + "=" * 140)
print("THE BALANCED FLAG FOR EVERY TYPE: 'not yet proven (<5 past bookings) OR past late-fail rate >=10%'  -- 24h before start")
print("=" * 140)
P_UNF = {}
f = s[s.n_fail > 0]
for b, g in f.groupby('last_band'): P_UNF[b] = (g.state == 'unfilled').mean()
d = cache[24]; d = d[d.start >= '2021-11-01'].copy()
d['flag'] = (d.prior_n_t < 5) | ((d.prior_n_t >= 5) & (d.rate_lf >= 0.10))
d['harm'] = d.band.map(P_UNF).where(d.late_fail, 0.0).fillna(0.0)
rows = []
for scope, m in [('all markets, Nov', np.ones(len(d), bool)), ('test half (Nov 16-30)', (d.start >= '2021-11-16').values),
                 ('Cleveland', d.cleveland.values), ('other markets', ~d.cleveland.values)]:
    x = d[m]; fl = x.flag
    row = {'scope': scope, 'bookings': len(x), 'flagged share': fl.mean()}
    for lab, col in [('NCNS', 'ncns'), ('<4h', 'lt4'), ('4-24h', 'c424'), ('any late', 'late_fail')]:
        row[f'{lab}: flagged rate'] = x.loc[fl, col].mean(); row[f'{lab}: unflagged rate'] = x.loc[~fl, col].mean()
        row[f'{lab}: share caught'] = x.loc[fl, col].sum() / x[col].sum()
    row['share of expected unfilled shifts caught'] = x.loc[fl, 'harm'].sum() / x.harm.sum()
    rows.append(row)
print(pd.DataFrame(rows).set_index('scope').T.round(3).to_string())
