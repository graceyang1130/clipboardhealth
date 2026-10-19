"""Segment explorer: size each candidate subgroup and run simple what-if levers.

Usage: python segments.py <work_dir> [out_dir]   (needs clean build.py outputs: claims.pkl, cle_shifts.pkl, cx2.pkl)

Segment families
  A. Booking-time flags (all markets, bookings for Nov 1-30 starts; flags use ONLY history before the booking)
  B. Worker groups (Cleveland; defined on Oct-Nov behaviour, measured on Dec-Jan)
  C. Timing of the failure (Cleveland; what if notice came earlier?)
  D. Shift attributes (Cleveland; where is refill weakest, what if refill improved?)
  E. Facilities (Cleveland; defined Oct-Nov, measured Dec-Jan)
(Raw-vs-clean baseline comparison lives in baselines.py.)

What-if levers are deliberately mechanical; every assumption is a named constant below.
"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import booking_features
S = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)

TAKE = 0.22                 # take rate (given)
ANNUAL = 3                  # 4 months of Cleveland data -> 12 months
LATE = ['1_NCNS', '2_<4h', '3_4-24h']
REPLACE_R = [0.25, 0.5, 0.75]   # share of an excluded group's bookings that other workers would pick up
CUT = 0.5                   # "halve this segment's late failures" lever
CONVERT = [0.10, 0.25, 0.50]    # share of failures converted to earlier notice

cl = pd.read_pickle(f"{S}/claims.pkl"); s = pd.read_pickle(f"{S}/cle_shifts.pkl"); cx = pd.read_pickle(f"{S}/cx2.pkl")

def save(df, name):
    if OUT: df.to_csv(f"{OUT}/segments_{name}.csv")

# =====================================================================================
# Cleveland money + harm weights
# =====================================================================================
s = s.copy()
med = s.groupby('Agent Req').charge.median()
s['gmv'] = s.charge.fillna(s['Agent Req'].map(med)).fillna(med['LVN']) * s.Time.clip(lower=0)
s['period'] = np.where(s.Start_loc < '2021-12-01', 'A', 'B')       # A = Oct-Nov, B = Dec-Jan
s['lost_late'] = (s.state == 'unfilled') & s.last_band.isin(LATE)   # unfilled after a late failure
s['rev_lost'] = np.where(s.lost_late, s.gmv * TAKE, 0.0)
sc = s
f = sc[sc.n_fail > 0]
P_UNFILLED = f.groupby('last_band').apply(lambda g: (g.state == 'unfilled').mean(), include_groups=False)
P_WORKED = f.groupby('last_band').worked.mean()
GMV_PER_LOST = sc.loc[sc.lost_late, 'gmv'].mean()
BASE_REV_YR = sc.rev_lost.sum() * ANNUAL
print("=" * 100)
print(f"Cleveland baseline (clean data): late-failure lost revenue ${sc.rev_lost.sum():,.0f} per 4 mo -> ${BASE_REV_YR:,.0f}/yr; "
      f"avg billings per lost shift ${GMV_PER_LOST:,.0f}")
print("P(shift ends unfilled | last failure band):", P_UNFILLED.round(3).to_dict())
print("P(shift ends worked   | last failure band):", P_WORKED.round(3).to_dict())

# =====================================================================================
# A. Booking-time flags (all markets)
# =====================================================================================
c = booking_features(cl, cx, with_holds=False)
c['harm'] = c.band.map(P_UNFILLED).fillna(0.0)                       # expected unfilled shifts this booking caused
c['harm_late'] = np.where(c.late_fail, c.harm, 0.0)

e = c[c.start >= '2021-11-01'].copy()          # evaluation window: >= 1 month of history available
FLAGS = {
    'first booking at facility':        e.first_fac,
    'booked <72h before start':          e.claim_lead < 72,
    'booked <24h before start':          e.claim_lead < 24,
    'worker account <30 days':           e.tenure_d < 30,
    'any of first-fac / <72h / new':     e.first_fac | (e.claim_lead < 72) | (e.tenure_d < 30),
    'all three':                         e.first_fac & (e.claim_lead < 72) & (e.tenure_d < 30),
    'no resolved history yet':           e.prior_n == 0,
    'prior NCNS >= 1':                   e.prior_ncns >= 1,
    'prior NCNS >= 2':                   e.prior_ncns >= 2,
    'prior call-offs (<4h) >= 2':        e.prior_calloff >= 2,
    'prior late-fail rate >=25% (n>=5)': (e.prior_n >= 5) & (e.prior_rate >= 0.25),
    'core: n>=10 & prior rate <5%':      (e.prior_n >= 10) & (e.prior_rate < 0.05),
    'heavy user: >=20 resolved bookings': e.prior_n >= 20,
}
tot = dict(n=len(e), lf=e.late_fail.sum(), ncns=e.ncns.sum(), harm=e.harm_late.sum(), kept=(e.outcome == 'kept').sum())
base_lfr = e.late_fail.mean()
rows = []
for name, m in FLAGS.items():
    x, rest = e[m], e[~m]
    row = dict(segment=name, bookings=len(x), share_bookings=len(x) / tot['n'],
               late_fail_rate=x.late_fail.mean(), ncns_rate=x.ncns.mean(), rest_rate=rest.late_fail.mean(),
               lift=x.late_fail.mean() / rest.late_fail.mean(),
               share_late_fail=x.late_fail.sum() / tot['lf'], share_ncns=x.ncns.sum() / tot['ncns'],
               share_harm=x.harm_late.sum() / tot['harm'])
    row['halve_seg_$yr'] = CUT * row['share_harm'] * BASE_REV_YR
    # restrict the segment entirely: its bookings disappear, a share r is re-booked by others at the rest's rates
    hr = rest.harm_late.sum() / len(rest); kr = (rest.outcome == 'kept').mean()
    for rr in REPLACE_R:
        harm_after = tot['harm'] - x.harm_late.sum() + rr * len(x) * hr
        kept_after = tot['kept'] - (x.outcome == 'kept').sum() + rr * len(x) * kr
        row[f'restrict_r{rr}_harm_chg'] = harm_after / tot['harm'] - 1
        row[f'restrict_r{rr}_kept_chg'] = kept_after / tot['kept'] - 1
    rows.append(row)
A = pd.DataFrame(rows).set_index('segment')
print("=" * 100)
print(f"A. BOOKING-TIME FLAGS  (all markets, {tot['n']:,} bookings for Nov starts; base late-fail rate {base_lfr:.3f})")
print("   share_harm = share of expected unfilled shifts from late failures; halve_seg_$yr = Cleveland-equivalent revenue if the segment's late failures were halved")
print(A[['bookings', 'share_bookings', 'late_fail_rate', 'ncns_rate', 'lift', 'share_late_fail', 'share_ncns', 'share_harm', 'halve_seg_$yr']]
      .round(3).to_string())
print("\n   Restrict segment entirely: change in harm (late-failure unfilled shifts) vs change in kept bookings, by replacement rate r")
print(A[[col for col in A.columns if col.startswith('restrict')]].round(3).to_string())
save(A, 'A_booking_flags')
# overlap of the three main booking-time flags
fl = pd.DataFrame({'first_fac': e.first_fac, 'short72': e.claim_lead < 72, 'new30': e.tenure_d < 30})
combo = fl.astype(int).astype(str).agg(''.join, axis=1).map(lambda k: '+'.join(n for n, b in zip(fl.columns, k) if b == '1') or 'none')
ov = e.groupby(combo).agg(bookings=('late_fail', 'size'), late_fail_rate=('late_fail', 'mean'), late_fails=('late_fail', 'sum'))
ov['share_bookings'] = ov.bookings / tot['n']; ov['share_late_fail'] = ov.late_fails / tot['lf']
print("\n   Overlap of the three booking-time flags"); print(ov.sort_values('bookings', ascending=False).round(3).to_string())
save(ov, 'A_flag_overlap')

# =====================================================================================
# B. Worker groups (Cleveland; define on Oct-Nov, measure on Dec-Jan)
# =====================================================================================
ec = cx[cx['Shift ID'].isin(set(sc['Shift ID']))].dropna(subset=['Worker ID']).merge(sc[['Shift ID', 'period']], on='Shift ID')
ec['ncns'] = ec.band == '1_NCNS'; ec['calloff'] = ec.band == '2_<4h'
last_w = cx[cx['Shift ID'].isin(set(sc['Shift ID']))].sort_values('at').groupby('Shift ID')['Worker ID'].last()
sc = sc.merge(last_w.rename('last_fail_worker'), left_on='Shift ID', right_index=True, how='left')

def worker_period(p):
    ev_ = ec[ec.period == p].groupby('Worker ID').agg(events=('band', 'size'), lf=('late_fail', 'sum'), ncns=('ncns', 'sum'), calloff=('calloff', 'sum'))
    sp = sc[sc.period == p]
    asg = sp[sp.has_worker].groupby('Worker ID').agg(assigned=('Shift ID', 'size'))
    wk = sp[sp.worked].groupby('Worker ID').agg(worked=('Shift ID', 'size'), hrs=('Time', 'sum'), gmv=('gmv', 'sum'))
    lost = sp[sp.lost_late].groupby('last_fail_worker').rev_lost.sum().rename('rev_lost')
    d = asg.join(ev_, how='outer').join(wk, how='outer').join(lost, how='outer').fillna(0)
    d['bookings'] = d.assigned + d.events
    return d

WA, WB = worker_period('A'), worker_period('B')
WA['rate'] = WA.lf / WA.bookings.where(WA.bookings > 0)
WB = WB.join(WA.add_suffix('_A'), how='left')
inA = WB.bookings_A.fillna(0) > 0
WGROUPS = {
    '2+ NCNS in Oct-Nov':                    WB.ncns_A >= 2,
    '3+ NCNS or 4+ call-offs (2021 deactivation level)': (WB.ncns_A >= 3) | (WB.calloff_A >= 4),
    'late-fail rate >=30% (>=3 bookings)':   (WB.bookings_A >= 3) & (WB.rate_A >= 0.30),
    'core: >=10 bookings & rate <5%':        (WB.bookings_A >= 10) & (WB.rate_A < 0.05),
    'low volume in Oct-Nov (<5 bookings)':   inA & (WB.bookings_A < 5),
    'new in Dec-Jan (no Oct-Nov activity)':  ~inA,
}
tb = WB[['bookings', 'lf', 'ncns', 'worked', 'hrs', 'rev_lost']].sum()
rows = []
for name, m in WGROUPS.items():
    x, rest = WB[m.fillna(False)], WB[~m.fillna(False)]
    row = dict(group=name, workers=len(x), share_bookings=x.bookings.sum() / tb.bookings, late_fail_rate=x.lf.sum() / x.bookings.sum(),
               share_late_fail=x.lf.sum() / tb.lf, share_ncns=x.ncns.sum() / tb.ncns, share_hours_worked=x.hrs.sum() / tb.hrs,
               share_lost_rev=x.rev_lost.sum() / tb.rev_lost)
    y_rest = rest.worked.sum() / rest.bookings.sum(); lost_rest = rest.rev_lost.sum() / rest.bookings.sum()
    for rr in REPLACE_R:
        worked_after = tb.worked - x.worked.sum() + rr * x.bookings.sum() * y_rest
        lost_after = tb.rev_lost - x.rev_lost.sum() + rr * x.bookings.sum() * lost_rest
        row[f'exclude_r{rr}_worked_chg'] = worked_after / tb.worked - 1
        row[f'exclude_r{rr}_lostrev_chg'] = lost_after / tb.rev_lost - 1
    rows.append(row)
B = pd.DataFrame(rows).set_index('group')
print("=" * 100)
print(f"B. WORKER GROUPS (Cleveland; defined on Oct-Nov, measured Dec-Jan: {int(tb.bookings):,} bookings, {int(tb.worked):,} worked shifts)")
print(B[['workers', 'share_bookings', 'late_fail_rate', 'share_late_fail', 'share_ncns', 'share_hours_worked', 'share_lost_rev']].round(3).to_string())
print("\n   Exclude group in Dec-Jan: change in worked shifts vs change in late-failure lost revenue, by replacement rate r")
print(B[[col for col in B.columns if col.startswith('exclude')]].round(3).to_string())
save(B, 'B_worker_groups')

# =====================================================================================
# C. Timing of the failure (Cleveland): what if notice came earlier?
# =====================================================================================
fl_ = sc[sc.n_fail > 0]
n_band = fl_.last_band.value_counts()
moves = [('1_NCNS', '2_<4h', 'NCNS -> any notice (<4h call-off)'),
         ('2_<4h', '3_4-24h', '<4h -> 4-24h notice'),
         ('3_4-24h', '4_24-72h', '4-24h -> 24-72h notice'),
         ('2_<4h', '4_24-72h', '<4h -> 24-72h notice')]
rows = []
for frm, to, lab in moves:
    gain = P_UNFILLED[frm] - P_UNFILLED[to]
    for cv in CONVERT:
        shifts = cv * n_band[frm] * gain
        rows.append(dict(move=lab, converted=cv, failures_in_band=n_band[frm], unfilled_from=P_UNFILLED[frm], unfilled_to=P_UNFILLED[to],
                         shifts_saved_4mo=shifts, rev_saved_yr=shifts * GMV_PER_LOST * TAKE * ANNUAL, share_of_base=shifts * GMV_PER_LOST * TAKE * ANNUAL / BASE_REV_YR))
C = pd.DataFrame(rows)
print("=" * 100); print("C. TIMING LEVERS (Cleveland): convert a share of failures in one band to earlier notice")
print(C.round(3).to_string(index=False)); save(C, 'C_timing')

# =====================================================================================
# D. Shift attributes (Cleveland): where refill is weakest, and a refill lever
# =====================================================================================
k = ['Facility ID', 'Agent Req', 'Shift Type']
sc['rel'] = sc.charge / sc.groupby(k).charge.transform('median')
bk_ = sc[sc.ever_booked].copy()
bk_['dow'] = bk_.Start_loc.dt.day_name().str[:3]
lc = sc[(sc.n_fail > 0) & sc.last_band.isin(['2_<4h', '3_4-24h'])]       # late-cancelled (non-NCNS) shifts
TARGET = lc[lc.rel > 1.1].worked.mean()                                 # worked-after-late-cancel rate when pay is >10% above norm
DIMS = {
    'AM shift': bk_['Shift Type'] == 'am', 'NOC shift': bk_['Shift Type'] == 'noc', 'PM shift': bk_['Shift Type'] == 'pm',
    'CNA': bk_['Agent Req'] == 'CNA', 'LVN': bk_['Agent Req'] == 'LVN', 'RN': bk_['Agent Req'] == 'RN',
    'Sun/Mon start': bk_.dow.isin(['Sun', 'Mon']), 'shift <8h': bk_.dur < 8,
    'pay below facility norm': bk_.rel < 0.95, 'pay >10% above norm': bk_.rel > 1.1,
}
tl = dict(lf=(bk_.n_late > 0).sum(), rev=bk_.rev_lost.sum())
rows = []
for name, m in DIMS.items():
    x = bk_[m]; xl = lc[lc['Shift ID'].isin(x['Shift ID'])]
    cur = xl.worked.mean(); add = max(TARGET - cur, 0) * len(xl)
    rows.append(dict(segment=name, booked=len(x), late_fail_rate=(x.n_late > 0).mean(), share_late_fail=(x.n_late > 0).sum() / tl['lf'],
                     share_lost_rev=x.rev_lost.sum() / tl['rev'], late_cancels=len(xl), worked_after_late_cancel=cur,
                     refill_to_target_shifts_4mo=add, refill_to_target_rev_yr=add * xl.gmv.mean() * TAKE * ANNUAL if len(xl) else 0))
D = pd.DataFrame(rows).set_index('segment')
print("=" * 100)
print(f"D. SHIFT ATTRIBUTES (Cleveland booked shifts). Refill lever: raise worked-after-late-cancel to {TARGET:.3f} (rate seen when pay >10% above norm)")
print(D.round(3).to_string()); save(D, 'D_shift_attributes')
allx = max(TARGET - lc.worked.mean(), 0) * len(lc)
print(f"   all late-cancelled shifts: current {lc.worked.mean():.3f} -> {TARGET:.3f}: +{allx:.0f} worked shifts / 4 mo, "
      f"~${allx * lc.gmv.mean() * TAKE * ANNUAL:,.0f}/yr revenue at base pay")
print(f"   cost check: a 10% premium paid only on late-cancelled shifts that get worked at the target rate "
      f"= ~${0.10 * TARGET * len(lc) * lc.gmv.mean() * ANNUAL:,.0f}/yr of extra billings (who funds it is a design choice)")

# =====================================================================================
# E. Facilities (Cleveland; defined Oct-Nov, measured Dec-Jan)
# =====================================================================================
fa = bk_[bk_.period == 'A'].groupby('Facility ID').agg(n=('n_late', 'size'), lf=('n_late', lambda v: (v > 0).sum()))
p = fa.lf.sum() / fa.n.sum(); fa['z'] = (fa.lf / fa.n - p) / np.sqrt(p * (1 - p) / fa.n)
hi = set(fa[(fa.n >= 30) & (fa.z > 2)].index); lo = set(fa[(fa.n >= 30) & (fa.z < -2)].index)
bB = bk_[bk_.period == 'B']
rows = []
for name, ids in [('high-failure facilities (Oct-Nov z>2)', hi), ('low-failure facilities (Oct-Nov z<-2)', lo)]:
    x = bB[bB['Facility ID'].isin(ids)]
    rows.append(dict(group=name, facilities=len(ids), share_booked_DecJan=len(x) / len(bB), late_fail_rate_DecJan=(x.n_late > 0).mean(),
                     share_late_fail_DecJan=(x.n_late > 0).sum() / (bB.n_late > 0).sum(), share_lost_rev_DecJan=x.rev_lost.sum() / bB.rev_lost.sum()))
E = pd.DataFrame(rows).set_index('group')
print("=" * 100); print(f"E. FACILITIES (Cleveland). Overall Dec-Jan late-fail rate {(bB.n_late > 0).mean():.3f}")
print(E.round(3).to_string()); save(E, 'E_facilities')
