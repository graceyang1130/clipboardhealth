"""Role x time segmentation (Cleveland): how late failures, refills and the proposal's triggers differ for CNAs, LVNs
and RNs, and by start window. Feeds role-specific settings of Confirm-or-Release.

Usage: python roles.py <clean_dir> [out_dir] [raw_dir]
  0. ROLE PROFILE (needs raw_dir, the uncleaned build). Every role in the raw shift log, removed ones included: shifts
     posted, facilities, professionals, median charge, cancelled by facility, filled, share of billings, kept/booked.
  1. ROLE x START WINDOW. Booked shifts, late-failure rates by type, share of late failures and lost revenue, share still
     worked, share of late failures after the confirmation deadline, overnight drops for morning starts.
  2. REFILL BY ROLE. Share still worked by notice; hours from drop to re-booking; same-role bench at the facility; pay
     below the facility's norm. A logistic model (worked ~ role x log notice + window) gives each role's chance of being
     worked if released at its window's deadline: RN cancellations are too few for a non-parametric estimate per window.
  3. WORKERS BY ROLE. Who books each role, cross-role work, trigger coverage 24h before start (Oct-Nov Cleveland
     bookings), first-visit and 48h-streak risk ratios, concentration among workers.
  4. BASE-CASE IMPACT BY ROLE. The sizing.py model (30% surface, 5% prevented) with role-specific coverage and refill.
Role is only recorded in the Cleveland shift log, so everything here is Cleveland.
"""
import sys, os, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import booking_features, at_decision, confirmation_deadline
warnings.filterwarnings('ignore')
S = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None; RAW = sys.argv[3] if len(sys.argv) > 3 else None
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
TAKE, ANNUAL = 0.22, 3
LATE = ['1_NCNS', '2_<4h', '3_4-24h']
ROLES = ['CNA', 'LVN', 'RN']; WINS = ['morning', 'afternoon/evening', 'night']
SF, PV = 0.30, 0.05                                                   # base case, as in sizing.py
TZ = 'America/New_York'
to_loc = lambda t: t.dt.tz_localize('UTC').dt.tz_convert(TZ).dt.tz_localize(None)
hdr = lambda t: print("\n" + "=" * 130 + f"\n{t}\n" + "=" * 130)

s = pd.read_pickle(f"{S}/cle_shifts.pkl"); s = s[s['Agent Req'].isin(ROLES)].copy()
s['role'] = s['Agent Req']
s['window'], s['dl_notice'] = confirmation_deadline(s.Start_loc)
s['win'] = s.window.replace({'noon': 'afternoon/evening'})            # too few noon drops to stand alone
med = s.groupby('role').charge.median()
s['gmv'] = s.charge.fillna(s.role.map(med)) * s.Time.clip(lower=0)
s['norm'] = s.groupby(['Facility ID', 'role', 'Shift Type']).charge.transform('median')
s['below_norm'] = s.charge < s.norm - 0.01
s['late'] = (s.n_fail > 0) & s.last_band.isin(LATE)
s['eligible'] = s.late & ((s.last_band == '1_NCNS') | (s.last_lead < s.dl_notice))
s['drop_hr'] = to_loc(s.last_at).dt.hour
B = s[s.ever_booked].copy()
L = B[B.late].copy()

# ================================================================================================ 0. role profile
if RAW:
    hdr("0. ROLE PROFILE (raw Cleveland shift log, all roles; last two columns after cleaning)")
    r = pd.read_pickle(f"{RAW}/cle_shifts.pkl")
    r['grp'] = r['Agent Req'].where(r['Agent Req'].isin(ROLES + ['NURSE', 'CAREGIVER']), 'Other rare')
    r['bill'] = np.where(r.Verified, r.Charge * r.Time, 0.0)
    def profile(d):
        nd = d[~d.deleted]
        return pd.Series({'posted': len(d), 'facilities': d['Facility ID'].nunique(), 'professionals': d['Worker ID'].nunique(),
                          'median_charge_hr': d.Charge.median(), 'cancelled_by_facility': d.deleted.mean(),
                          'filled_of_not_cancelled': nd.Verified.mean(), 'billings': d.bill.sum()})
    RP = r.groupby('grp').apply(profile); RP.loc['All'] = profile(r)
    RP['billing_share'] = RP.billings / RP.loc['All', 'billings']
    RP['pro_pay_hr_at_take'] = RP.median_charge_hr * (1 - TAKE)
    kept = pd.read_pickle(f"{S}/cle_shifts.pkl").groupby('Agent Req').agg(kept=('Shift ID', 'size'), booked=('ever_booked', 'sum'))
    RP = RP.join(kept); RP.loc['All', ['kept', 'booked']] = kept.sum().values
    RP = RP.loc[ROLES + ['NURSE', 'CAREGIVER', 'Other rare', 'All']]
    print(RP.round(3).to_string())
    if OUT: RP.to_csv(f"{OUT}/roles_profile.csv")

# ================================================================================================ 1. role x window
hdr("1. ROLE x START WINDOW (Cleveland booked shifts, Oct 2021-Jan 2022)")


def grid(g):
    late = g[g.late]; unf = late[late.state == 'unfilled']
    mc = late[(late.win == 'morning') & late.last_band.isin(['2_<4h', '3_4-24h'])]
    return pd.Series({
        'booked': len(g), 'late_fail_rate': g.late.mean(),
        'ncns_rate': (g.last_band == '1_NCNS').mean() if len(g) else np.nan, 'calloff_rate': (g.last_band == '2_<4h').mean(),
        'c424_rate': (g.last_band == '3_4-24h').mean(), 'ncns_share_of_late': (late.last_band == '1_NCNS').mean(),
        'late_failures': len(late), 'share_of_all_late': len(late) / len(L),
        'lost_rev_yr': unf.gmv.sum() * TAKE * ANNUAL, 'share_of_lost_rev': unf.gmv.sum() / L.loc[L.state == 'unfilled', 'gmv'].sum(),
        'worked_after_late_failure': late.worked.mean(), 'unfilled_after_late_failure': (late.state == 'unfilled').mean(),
        'share_after_deadline': late.eligible.mean(),
        'morning_late_cancels_overnight': ((mc.drop_hr >= 22) | (mc.drop_hr < 7)).mean() if len(mc) else np.nan})


rows = []
for r in ROLES + ['all']:
    for w in WINS + ['all']:
        g = B[((B.role == r) | (r == 'all')) & ((B.win == w) | (w == 'all'))]
        rows.append({'role': r, 'window': w, **grid(g)})
G = pd.DataFrame(rows); print(G.round(3).to_string(index=False))
print("\nlate cancellations by hour of the drop, morning starts (share of each role's morning late cancels):")
mc = L[(L.win == 'morning') & L.last_band.isin(['2_<4h', '3_4-24h'])].copy()
mc['when'] = pd.cut(mc.drop_hr, [-1, 6, 11, 18, 21, 24], labels=['midnight-7am', '7am-noon', 'noon-7pm', '7-10pm', '10pm-midnight'])
print(pd.crosstab(mc.role, mc.when, normalize='index').round(3).to_string())

# ================================================================================================ 2. refill by role
hdr("2. REFILL BY ROLE")
F = B[B.n_fail > 0].copy()
edges = [0, 4, 24, 72, np.inf]; labs = ['<4h', '4-24h', '24-72h', '>72h']
F['notice'] = np.where(F.last_band == '1_NCNS', 'NCNS', pd.cut(F.last_lead, edges, labels=labs, right=False).astype(str))
print("share of shifts still worked, by role and the notice of the last failure:")
print(F.pivot_table(index='role', columns='notice', values='worked', aggfunc='mean')[['NCNS'] + labs].round(3).to_string())
print(F.pivot_table(index='role', columns='notice', values='worked', aggfunc='size')[['NCNS'] + labs].to_string())

rf = pd.read_pickle(f"{S}/refill.pkl").merge(s[['Shift ID', 'role', 'win']], on='Shift ID')
ld = rf[rf.band.isin(['2_<4h', '3_4-24h'])]
T = ld.groupby('role').agg(late_cancels=('Shift ID', 'size'), rebooked=('refilled', 'mean'),
                           median_hours_to_rebook=('ttr_h', 'median'),
                           rebooked_within_1h=('ttr_h', lambda x: (x <= 1).sum() / len(x)))
print("\nlate cancellations (<24h): share re-booked by another worker, and how fast:"); print(T.round(3).to_string())

# same-role bench: distinct workers of the role who worked at the facility in the 30 days before the start
wk = s[s.worked][['Facility ID', 'role', 'Worker ID', 'Start']]
lc = L[L.last_band.isin(['2_<4h', '3_4-24h'])].copy()
bench = np.zeros(len(lc), int)
for i, (fac, rl, st) in enumerate(zip(lc['Facility ID'], lc.role, lc.Start)):
    x = wk[(wk['Facility ID'] == fac) & (wk.role == rl) & (wk.Start < st) & (wk.Start >= st - pd.Timedelta(days=30))]
    bench[i] = x['Worker ID'].nunique()
lc['bench'] = bench
print("\nsame-role bench at the facility when a late cancellation happens (workers of that role who worked there in the last 30 days):")
print(lc.groupby('role').bench.describe(percentiles=[0.25, 0.5, 0.75])[['count', '25%', '50%', '75%']].to_string())
lc['bench_band'] = pd.cut(lc.bench, [-1, 5, 15, 1000], labels=['0-5', '6-15', '16+'])
print("share still worked after a late cancellation, by role and same-role bench:")
print(lc.pivot_table(index='role', columns='bench_band', values='worked', aggfunc='mean', observed=False).round(3).to_string())
print(lc.pivot_table(index='role', columns='bench_band', values='worked', aggfunc='size', observed=False).to_string())
print("\nshare still worked after a late cancellation, priced below the facility's norm vs at/above:")
print(lc.pivot_table(index='role', columns='below_norm', values='worked', aggfunc=['mean', 'size']).round(3).to_string())

# logistic model: chance a cancelled shift is still worked, by role, notice and window (cancellations with <72h notice)
R = B[(B.n_fail > 0) & B.last_band.isin(['2_<4h', '3_4-24h', '4_24-72h'])].copy()
R['lognotice'] = np.log1p(R.last_lead.clip(lower=0)); R['y'] = R.worked.astype(float)
m = smf.logit('y ~ C(role) * lognotice + C(win)', data=R).fit(disp=0)
print(f"\nrefill model on {len(R):,} cancellations with <72h notice (pseudo-R2 {m.prsquared:.3f}); coefficients:")
print(pd.DataFrame({'coef': m.params, 'p': m.pvalues}).round(3).to_string())
L['p_dl'] = m.predict(L.assign(lognotice=np.log1p(L.dl_notice)))
L['p_now'] = m.predict(L.assign(lognotice=np.log1p(L.last_lead.fillna(0).clip(lower=0))))
P = L[L.eligible].groupby(['role', 'win']).agg(after_deadline=('Shift ID', 'size'), worked_today=('worked', 'mean'),
                                                 worked_if_released_at_deadline=('p_dl', 'mean'))
print("\nlate failures after the deadline: worked today vs modelled chance if released at the deadline:")
print(P.round(3).to_string())

# ================================================================================================ 3. workers by role
hdr("3. WORKERS BY ROLE (Cleveland bookings, Oct-Nov 2021 starts)")
cl = pd.read_pickle(f"{S}/claims.pkl"); cx = pd.read_pickle(f"{S}/cx2.pkl")
c = booking_features(cl, cx, with_holds=False)
role_of = s.set_index('Shift ID').role
cc = c[c.cleveland].copy(); cc['role'] = cc['Shift ID'].map(role_of); cc = cc[cc.role.notna()]
wr = cc.groupby('Worker ID').role.agg(lambda x: set(x))
rows = []
for r in ROLES:
    ws = wr[wr.apply(lambda x: r in x)]
    rows.append({'role': r, 'bookings': (cc.role == r).sum(), 'workers': len(ws),
                 'bookings_per_worker': (cc.role == r).sum() / len(ws),
                 **{f'also books {o}': ws.apply(lambda x: o in x).mean() for o in ROLES if o != r}})
print(pd.DataFrame(rows).round(3).to_string(index=False))

d = at_decision(c, cx, 24); d = d[d.cleveland].copy(); d['role'] = d['Shift ID'].map(role_of); d = d[d.role.notna()]
d['A'] = (d.prior_n_t < 5) | (d.rate_lf >= 0.10); d['B'] = d.k_drop_48h >= 1; d['C'] = d.first_fac.astype(bool)
d['chk'] = d.A | d.B | d.C
rows = []; COVR = {}
for r in ROLES:
    x = d[d.role == r]
    row = {'role': r, 'bookings at 24h': len(x), 'late-fail rate': x.late_fail.mean(), 'NCNS rate': x.ncns.mean(),
           'not yet proven (A)': x.A.mean(), 'B 48h drop': x.B.mean(), 'C first visit': x.C.mean(), 'checked (A|B|C)': x.chk.mean()}
    for lab, col in [('NCNS', 'ncns'), ('<4h', 'lt4'), ('4-24h', 'c424'), ('all late', 'late_fail')]:
        row[f'{lab} covered'] = x.loc[x.chk, col].sum() / max(x[col].sum(), 1)
    row['late rate checked'] = x.loc[x.chk, 'late_fail'].mean(); row['late rate not checked'] = x.loc[~x.chk, 'late_fail'].mean()
    row['first-visit risk ratio'] = x.loc[x.C, 'late_fail'].mean() / x.loc[~x.C, 'late_fail'].mean()
    row['48h-streak risk ratio'] = x.loc[x.B, 'late_fail'].mean() / x.loc[~x.B, 'late_fail'].mean()
    row['proven workers late rate'] = x.loc[~x.A, 'late_fail'].mean()
    k = x[x.late_fail].groupby('Worker ID').size().sort_values(ascending=False)
    row['late failures from top 10% of workers'] = k.head(max(1, int(np.ceil(x['Worker ID'].nunique() * 0.10)))).sum() / max(k.sum(), 1)
    rows.append(row)
    COVR[r] = {'1_NCNS': row['NCNS covered'], '2_<4h': row['<4h covered'], '3_4-24h': row['4-24h covered'], 'all': row['all late covered'],
               'n': {'1_NCNS': x.ncns.sum(), '2_<4h': x.lt4.sum(), '3_4-24h': x.c424.sum()}}
W = pd.DataFrame(rows).set_index('role'); print(W.T.round(3).to_string())

# ================================================================================================ 4. impact by role
hdr("4. BASE-CASE IMPACT BY ROLE (30% surface, 5% prevented; per year)")
# coverage by failure type where the role has >=30 events of that type at 24h, else the role's all-late coverage
L['covg'] = [COVR[r][b] if COVR[r]['n'][b] >= 30 else COVR[r]['all'] for r, b in zip(L.role, L.last_band)]
L['w'] = L.worked.astype(float); L['val'] = L.gmv * TAKE
surf = L.covg * (1 - PV) * SF * L.eligible; prev = L.covg * PV
L['saved'] = prev * (1 - L.w) + surf * (L.p_dl - L.w)
L['nonotice_avoided'] = (prev + surf) * L.last_band.isin(['1_NCNS', '2_<4h'])
rows = []
for r in ROLES + ['all']:
    x = L[(L.role == r) | (r == 'all')]; b = B[(B.role == r) | (r == 'all')]
    unf = (x.state == 'unfilled').sum()
    rows.append({'role': r, 'shifts saved / yr': x.saved.sum() * ANNUAL, 'revenue / yr': (x.saved * x.val).sum() * ANNUAL,
                 'share of saved revenue': np.nan, 'unfilled per 100 booked today': unf / len(b) * 100,
                 'unfilled per 100 booked after': (unf - x.saved.sum()) / len(b) * 100,
                 'NCNS + call-offs avoided': x.nonotice_avoided.sum() / x.last_band.isin(['1_NCNS', '2_<4h']).sum(),
                 'revenue per recovered shift': (x.val * x.saved).sum() / max(x.saved.sum(), 1e-9)})
I = pd.DataFrame(rows).set_index('role'); I['share of saved revenue'] = I['revenue / yr'] / I.loc['all', 'revenue / yr']
print(I.round(3).to_string())

if OUT:
    G.to_csv(f"{OUT}/roles_grid.csv", index=False); P.to_csv(f"{OUT}/roles_refill_at_deadline.csv")
    T.join(lc.groupby('role').bench.median().rename('median_same_role_bench')).to_csv(f"{OUT}/roles_refill.csv")
    W.to_csv(f"{OUT}/roles_workers.csv"); I.to_csv(f"{OUT}/roles_impact.csv")

# ================================================================================================ 5. role-specific settings
hdr("5. ROLE-SPECIFIC SETTINGS: checks without the first-visit trigger; who already covers LVN shifts")
for r in ROLES:
    x = d[d.role == r]; ab = x.A | x.B
    print(f"{r}: checked with A|B|C {x.chk.mean():.1%} (covers {x.loc[x.chk, 'late_fail'].sum() / x.late_fail.sum():.1%} of late failures) | "
          f"A|B only {ab.mean():.1%} (covers {x.loc[ab, 'late_fail'].sum() / x.late_fail.sum():.1%}) | "
          f"first visits: {x.C.sum()} bookings, {x.loc[x.C, 'late_fail'].sum()} late failures")
rn_workers = set(cc.loc[cc.role == 'RN', 'Worker ID'])
lw = s[(s.role == 'LVN') & s.worked]
print(f"worked LVN shifts done by workers who also book RN shifts: {lw['Worker ID'].isin(rn_workers).mean():.1%} of {len(lw):,}")
rb = rf[(rf.role == 'LVN') & rf.band.isin(['2_<4h', '3_4-24h']) & rf.refilled]
print(f"LVN late cancellations re-booked: {len(rb)}")
# how much an earlier deadline would buy each role: modelled chance a released shift is worked, by notice
grid_n = pd.DataFrame([{'role': r, 'win': w, 'lognotice': np.log1p(h), 'notice_h': h} for r in ROLES for w in WINS for h in [2, 6, 12, 24, 48]])
grid_n['p'] = m.predict(grid_n)
print("\nmodelled chance a cancelled shift is still worked, by notice (hours before start):")
print(grid_n.pivot_table(index=['role', 'win'], columns='notice_h', values='p').round(3).to_string())
for r in ROLES:
    x = L[L.role == r]
    for h in [24, 48]:
        p = m.predict(x.assign(lognotice=np.log1p(h)))
        gain = (x.covg * SF * (p - x.w)).sum() * ANNUAL; gain_now = (x.covg * SF * x.eligible * (x.p_dl - x.w)).sum() * ANNUAL
        print(f"{r}: surfacing 30% of covered late failures at {h}h instead of at today's deadlines -> {gain:.0f} vs {gain_now:.0f} shifts/yr")
