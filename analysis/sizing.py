"""Sizing for the Confirm-or-Release proposal: who gets checked, what it saves, what it costs, how long the pilot runs.

Usage: python sizing.py <clean_dir> [out_dir]

1. WHO GETS CHECKED. Share of bookings hit by each trigger (not-yet-proven flag, late drop in the last 48h, first visit)
   and the share of each failure type they cover. Nov starts, all markets, measured 24h before start.
2. IMPACT. Cleveland shifts with a late failure (Oct-Jan; outcome attributed to the shift's last failure). Each shift
   gets the confirmation deadline of its start window. A covered late failure that happened AFTER the deadline (every
   NCNS; cancels with less notice than the deadline gives) can instead surface at the deadline, and is then worked as
   often as Cleveland shifts dropped with that much notice in the same window. Scenarios set the share of covered late
   failures that surface at the deadline and the share prevented outright. These two shares are ASSUMPTIONS (no data
   on confirmation behaviour exists); everything else is measured.
3. COST. Messages per check (assumed unit cost); a pay floor at the facility's norm on surfaced shifts priced below it
   (upper bound: Clipboard pays the full gap whenever the shift is worked); workers displaced by a release they did
   not want (assumed no-reply rate among bookings that would have been worked).
4. PILOT. Smallest detectable change in the primary metric (booking ends with the shift unfilled after a late
   failure) among checked bookings, with worker-level randomisation (design effect from worker clustering).
"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import booking_features, at_decision, confirmation_deadline
S = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
TAKE, ANNUAL = 0.22, 3                 # 4 months of Cleveland shifts -> per year
LATE = ['1_NCNS', '2_<4h', '3_4-24h']
COST_PER_CHECK = 0.05                  # assumption: two texts, reminder pushes, a call for about 1 in 3 checks
NO_REPLY_KEPT = 0.05                   # assumption: checked bookings that would have been worked but never confirm
SCEN = {'conservative': (0.15, 0.00), 'base': (0.30, 0.05), 'optimistic': (0.50, 0.10)}  # (surface at deadline, prevented)
hdr = lambda t: print("\n" + "=" * 120 + f"\n{t}\n" + "=" * 120)

cl = pd.read_pickle(f"{S}/claims.pkl"); cx = pd.read_pickle(f"{S}/cx2.pkl"); s = pd.read_pickle(f"{S}/cle_shifts.pkl")
c = booking_features(cl, cx, with_holds=False)
CLE_SHARE = c.cleveland.mean()        # Cleveland's share of Oct-Nov bookings, to scale revenue to all markets (assumes Cleveland's loss rates)

# ================================================================================================ 1. who gets checked
hdr("1. WHO GETS CHECKED (bookings still on the books 24h before start, Nov starts)")
d = at_decision(c, cx, 24); d = d[d.start >= '2021-11-01'].copy()
flag = (d.prior_n_t < 5) | (d.rate_lf >= 0.10)
streak = d.k_drop_48h >= 1; first = d.first_fac.astype(bool)
TRIG = {'A not yet proven (<5 past bookings OR late-fail rate >=10%)': flag, 'B late drop in the last 48h': streak,
        'C first visit to this facility': first, 'A or B': flag | streak, 'A or B or C (proposal)': flag | streak | first}
OUTC = {'NCNS': d.ncns, '<4h': d.lt4, '4-24h': d.c424, 'any late': d.late_fail}
rows = []
for scope, m in [('all markets', np.ones(len(d), bool)), ('Cleveland', d.cleveland.values)]:
    for name, t in TRIG.items():
        t = t.values & m; r = {'scope': scope, 'trigger': name, 'share of bookings checked': t.sum() / m.sum()}
        for lab, y in OUTC.items():
            y = y.values.astype(bool); r[f'{lab}: share covered'] = (t & y).sum() / (m & y).sum()
        r['late rate: checked'] = d.late_fail.values[t].mean(); r['late rate: not checked'] = d.late_fail.values[m & ~t].mean()
        rows.append(r)
TR = pd.DataFrame(rows); print(TR.round(3).to_string(index=False))
P = TR[(TR.scope == 'all markets') & (TR.trigger == 'A or B or C (proposal)')].iloc[0]
COV = {'1_NCNS': P['NCNS: share covered'], '2_<4h': P['<4h: share covered'], '3_4-24h': P['4-24h: share covered']}
CHECK_SHARE_CLE = TR[(TR.scope == 'Cleveland') & (TR.trigger == 'A or B or C (proposal)')].iloc[0]['share of bookings checked']
print(f"coverage used in the impact model (all markets): {({k: round(v, 3) for k, v in COV.items()})}")
# booking histories start in Aug-Oct, so some long-time workers look new; how many of the '<5 past bookings' are older accounts?
few = d.prior_n_t < 5
print(f"history check: of bookings flagged for <5 past bookings, {(d.tenure_t[few] > 90).mean():.0%} come from accounts >90 days old "
      f"(late-fail {d.late_fail[few & (d.tenure_t > 90)].mean():.3f} vs {d.late_fail[few & (d.tenure_t <= 90)].mean():.3f} for newer accounts)")

# ================================================================================================ 2. impact
hdr("2. IMPACT (Cleveland shifts with a late failure, Oct-Jan)")
s['window'], s['dl_notice'] = confirmation_deadline(s.Start_loc)
s['est_window'] = s.window.replace({'noon': 'afternoon/evening'})      # too few noon drops to estimate their own refill
med = s.groupby('Agent Req').charge.median()
s['gmv'] = s.charge.fillna(s['Agent Req'].map(med)) * s.Time.clip(lower=0)
s['norm'] = s.groupby(['Facility ID', 'Agent Req', 'Shift Type']).charge.transform('median')
s['floor_gap'] = (s.norm - s.charge).clip(lower=0).fillna(0) * s.Time.clip(lower=0)
L = s[(s.n_fail > 0) & s.last_band.isin(LATE)].copy()
L['eligible'] = (L.last_band == '1_NCNS') | (L.last_lead < L.dl_notice)
R = L[L.last_band.isin(['2_<4h', '3_4-24h'])]                          # reference: cancels and how often they were still worked


def p_at(win, notice):
    r = R[R.est_window == win]
    for bw in (2, 3, 4):
        x = r[(r.last_lead - notice).abs() <= bw]
        if len(x) >= 40: return x.worked.mean()
    return x.worked.mean()


cache = {}
L['p_dl'] = [cache.setdefault((w, round(n * 2) / 2), p_at(w, round(n * 2) / 2)) for w, n in zip(L.est_window, L.dl_notice)]
L['w'] = L.worked.astype(float); L['val'] = L.gmv * TAKE; L['covg'] = L.last_band.map(COV)
booked = int(s.ever_booked.sum()); unf = int(((L.state == 'unfilled')).sum())
print(f"booked shifts: {booked:,} | with a late failure: {len(L):,} | left unfilled after a late failure: {unf:,} "
      f"({unf / booked * 100:.1f} per 100 booked) | revenue per recovered shift ${L.val.mean():.0f} on ${L.gmv.mean():.0f} billings")
W = L.groupby('window').agg(late_failure_shifts=('Shift ID', 'size'), deadline_notice_h=('dl_notice', 'median'),
                            share_after_deadline=('eligible', 'mean'), worked_today=('w', 'mean'),
                            worked_today_after_deadline=('w', lambda x: x[L.loc[x.index, 'eligible']].mean()),
                            worked_if_surfaced_at_deadline=('p_dl', 'mean'))
print("by start window:"); print(W.round(3).to_string())
print("by failure type:"); print(L.groupby('last_band').agg(n=('Shift ID', 'size'), share_after_deadline=('eligible', 'mean'),
                                                          worked_today=('w', 'mean'), worked_if_surfaced=('p_dl', 'mean'), coverage=('covg', 'mean')).round(3).to_string())
rows = []
for name, (sf, pv) in SCEN.items():
    surf = L.covg * (1 - pv) * sf * L.eligible                        # expected share of each shift surfaced at the deadline
    prev = L.covg * pv
    saved = (prev * (1 - L.w) + surf * (L.p_dl - L.w))
    rev = (saved * L.val).sum() * ANNUAL
    floor_cost = (surf * L.p_dl * L.floor_gap).sum() * ANNUAL
    msg_cost = CHECK_SHARE_CLE * booked * ANNUAL * COST_PER_CHECK
    conv = lambda b: ((prev + surf)[L.last_band == b]).sum() / (L.last_band == b).sum()
    rows.append({'scenario': name, 'surface at deadline': sf, 'prevented': pv, 'shifts saved / yr': saved.sum() * ANNUAL,
                 'revenue / yr': rev, 'billings / yr': (saved * L.gmv).sum() * ANNUAL,
                 'unfilled after late failure per 100 booked': (unf - saved.sum()) / booked * 100,
                 'relative cut in that metric': saved.sum() / unf, 'NCNS avoided': conv('1_NCNS'), '<4h drops avoided': conv('2_<4h'),
                 'message cost / yr': msg_cost, 'pay floor cost / yr (upper bound)': floor_cost,
                 'net revenue / yr': rev - msg_cost - floor_cost, 'all-markets revenue / yr': rev / CLE_SHARE})
SC = pd.DataFrame(rows).set_index('scenario'); print("\nscenarios (Cleveland, per year):"); print(SC.T.round(3).to_string())
checks = CHECK_SHARE_CLE * booked * ANNUAL
print(f"\nchecks per year in Cleveland: {checks:,.0f} ({CHECK_SHARE_CLE:.0%} of bookings) | at ${COST_PER_CHECK:.2f}/check: ${checks * COST_PER_CHECK:,.0f}")
disp = checks * NO_REPLY_KEPT * L.p_dl.mean()
print(f"guardrail scale: if {NO_REPLY_KEPT:.0%} of checks are workers who meant to come but never confirm, and a released shift is "
      f"claimed by someone else {L.p_dl.mean():.0%} of the time, ~{disp:,.0f} workers/yr would lose a shift they meant to work")
# break-even: share of covered after-deadline failures that must surface for the program to pay for messages alone
unit = (L.covg * L.eligible * (L.p_dl - L.w) * L.val).sum() * ANNUAL
print(f"break-even: messages are paid for if {checks * COST_PER_CHECK / unit:.1%} of covered after-deadline late failures surface at the deadline")

# sensitivity: regained revenue as the share surfacing early varies (prevention held at 0, so this is early notice alone).
# Revenue is linear in both assumptions, so a refill haircut or a prevention share can be read off the per-unit lines.
lost_rev = (L.loc[L.state == 'unfilled', 'val']).sum() * ANNUAL
nn = L.last_band.isin(['1_NCNS', '2_<4h'])                             # no-notice failures: NCNS and call-offs
rows = []
for sf in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    surf = L.covg * sf * L.eligible; saved = surf * (L.p_dl - L.w); rev = (saved * L.val).sum() * ANNUAL
    cost = msg_cost + (surf * L.p_dl * L.floor_gap).sum() * ANNUAL
    rows.append({'share of covered after-deadline failures surfacing early': sf,
                 'NCNS + call-offs avoided': surf[nn].sum() / nn.sum(),
                 'shifts regained / yr': saved.sum() * ANNUAL, 'share of lost revenue regained': rev / lost_rev,
                 'Cleveland revenue / yr': rev, 'Cleveland net of running cost / yr': rev - cost, 'all-markets revenue / yr': rev / CLE_SHARE})
SE = pd.DataFrame(rows); print("\nsensitivity (Cleveland, per year; nothing prevented outright):"); print(SE.round(3).to_string(index=False))
prev5 = (L.covg * 0.05 * (1 - L.w) * L.val).sum() * ANNUAL
print(f"Cleveland = {CLE_SHARE:.1%} of bookings | revenue lost to late failures: Cleveland ${lost_rev:,.0f}/yr, all markets ~${lost_rev / CLE_SHARE:,.0f}/yr")
print(f"each 5 points of covered late failures prevented outright adds ${prev5:,.0f}/yr in Cleveland (${prev5 / CLE_SHARE:,.0f} all markets)")
print("if released shifts refill only k x as far above today's outcome as history suggests, multiply the revenue columns by k")

# two-way sensitivity: Cleveland revenue / yr over both untested assumptions (same model as the scenarios above)
SF_GRID, PV_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50], [0.00, 0.05, 0.10, 0.15, 0.20]
def revenue(sf, pv):
    surf = L.covg * (1 - pv) * sf * L.eligible; prev = L.covg * pv
    return ((prev * (1 - L.w) + surf * (L.p_dl - L.w)) * L.val).sum() * ANNUAL
SG = pd.DataFrame([[revenue(sf, pv) for pv in PV_GRID] for sf in SF_GRID],
                  index=pd.Index(SF_GRID, name='surface early'), columns=pd.Index(PV_GRID, name='prevented'))
print("\ntwo-way sensitivity, Cleveland revenue / yr (rows: share surfacing early; columns: share prevented outright):")
print(SG.round(0).to_string())
print("same, all markets ($M):"); print((SG / CLE_SHARE / 1e6).round(2).to_string())
print("same, share of lost revenue regained:"); print((SG / lost_rev).round(3).to_string())

# ================================================================================================ 4. pilot
hdr("4. PILOT SIZE (checked Cleveland bookings; 80% power, 2-sided 5%)")
dc = at_decision(c, cx, 24); dc = dc[dc.cleveland].copy()
dc = dc.merge(s[['Shift ID', 'state']], on='Shift ID', how='left')
dc['y_unfilled'] = (dc.late_fail & (dc.state == 'unfilled')).astype(float)      # primary: shift left unfilled after a late failure
dc['y_nonotice'] = (dc.ncns | dc.lt4).astype(float)                              # mechanism: NCNS or <4h call-off
chk = (dc.prior_n_t < 5) | (dc.rate_lf >= 0.10) | (dc.k_drop_48h >= 1) | dc.first_fac.astype(bool)
x = dc[chk & (dc.start >= '2021-10-04') & (dc.start < '2021-11-29')]            # 8 full weeks
per_wk = len(x) / 8


def design_effect(y, grp):
    """worker clustering: 1 + (mean cluster size weighted by size - 1) * ICC (one-way ANOVA estimator)"""
    p0 = y.mean(); g = y.groupby(grp); m = g.size().values; k = len(m); N = m.sum()
    msb = (m * (g.mean().values - p0) ** 2).sum() / (k - 1); msw = (g.var(ddof=0).values * m).sum() / (N - k)
    n0 = (N - (m ** 2).sum() / N) / (k - 1); icc = max((msb - msw) / (msb + (n0 - 1) * msw), 0)
    return 1 + ((m ** 2).sum() / N - 1) * icc, icc


# expected relative cuts among checked bookings (the model's shifts saved are concentrated in checked bookings)
share_unf_checked = (L.covg * (L.state == 'unfilled')).sum() / unf
nn = L.last_band.isin(['1_NCNS', '2_<4h'])
exp = {}
for name, (sf, pv) in SCEN.items():
    exp[('y_unfilled', name)] = SC.loc[name, 'relative cut in that metric'] / share_unf_checked
    exp[('y_nonotice', name)] = (L.covg * (pv + (1 - pv) * sf))[nn].sum() / L.covg[nn].sum()
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80); rows = []
for ycol, lab in [('y_unfilled', 'unfilled after a late failure'), ('y_nonotice', 'NCNS or <4h call-off')]:
    p0 = x[ycol].mean(); deff_w, icc = design_effect(x[ycol], x['Worker ID'])
    print(f"{lab}: rate among checked bookings {p0:.3f} | worker ICC {icc:.3f} -> design effect if randomised by worker {deff_w:.2f} | "
          f"expected cut: conservative {exp[(ycol, 'conservative')]:.1%}, base {exp[(ycol, 'base')]:.1%}, optimistic {exp[(ycol, 'optimistic')]:.1%}")
    for unit, deff in [('booking', 1.0), ('worker', deff_w)]:
        for markets in [1, 4, 6]:
            for wk in [8, 12]:
                n = per_wk * wk * markets / 2; mde = z * np.sqrt(2 * p0 * (1 - p0) * deff / n)
                rows.append({'metric': lab, 'randomised by': unit, 'markets (Cleveland-sized)': markets, 'weeks': wk,
                             'checked bookings per arm': round(n), 'baseline': p0, 'MDE (relative)': mde / p0,
                             'powered for base case': mde / p0 <= exp[(ycol, 'base')]})
print(f"checked Cleveland bookings per week: {per_wk:,.0f}")
PW = pd.DataFrame(rows); print(PW.round(3).to_string(index=False))

if OUT:
    TR.to_csv(f"{OUT}/sizing_triggers.csv", index=False); W.to_csv(f"{OUT}/sizing_windows.csv")
    SC.to_csv(f"{OUT}/sizing_scenarios.csv"); SE.to_csv(f"{OUT}/sizing_sensitivity.csv", index=False); SG.to_csv(f"{OUT}/sizing_sensitivity_grid.csv"); PW.to_csv(f"{OUT}/sizing_pilot.csv", index=False)
