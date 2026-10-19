"""Phase 0: clean the raw logs, then build the analysis tables.

Usage: python build.py <load_dir> <out_dir> [clean|raw]
  load_dir  where load.py cached the raw pickles
  out_dir   where the tables are written (claims.pkl, cle_shifts.pkl, cx2.pkl, bk2.pkl, exclusions.csv)
  clean     (default) applies every rule under CLEANING RULES and logs the rows each one drops
  raw       skips the rules (kept only to show how much cleaning moves the baselines)
"""
import sys, os, pandas as pd, numpy as np
L, O = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else 'clean'
os.makedirs(O, exist_ok=True)
sh = pd.read_pickle(f"{L}/Cleveland_shifts_logs.pkl"); bk = pd.read_pickle(f"{L}/Booking_logs.pkl"); cx = pd.read_pickle(f"{L}/Cancel_logs.pkl")
oid_ts = lambda s: pd.to_datetime(s.str[:8].apply(lambda h: int(h, 16) if isinstance(h, str) else np.nan), unit='s')
TZ = 'America/New_York'
CORE_ROLES = ['CNA', 'LVN', 'RN']
SHOCK_Z = 5          # a day is a shock day if its late-fail rate is > 5 robust SDs above the median day
LATE = ['1_NCNS', '2_<4h', '3_4-24h']

def band(lead, action):
    if action == 'NO_CALL_NO_SHOW': return '1_NCNS'
    if lead < 4: return '2_<4h'
    if lead < 24: return '3_4-24h'
    if lead <= 72: return '4_24-72h'
    return '5_>72h'

log = []
def drop(df, mask, table, rule):
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    log.append(dict(table=table, rule=rule, rows_before=len(df), rows_dropped=int(mask.sum())))
    return df[~mask]

bk = bk.rename(columns={'Created At': 'at', 'Lead Time': 'lead'}); bk['start'] = bk['at'] + pd.to_timedelta(bk.lead, unit='h')
cx = cx.rename(columns={'Created At': 'at', 'Lead Time': 'lead'})
sh_ids_raw = set(sh['Shift ID'])

# ================================ CLEANING RULES (event logs) ================================
if MODE == 'clean':
    bk = drop(bk, bk['Worker ID'].isna(), 'bookings', 'no worker id')
    bk = drop(bk, bk.lead < 0, 'bookings', 'claimed after the shift started (retroactive record)')
    bk = drop(bk, bk.drop(columns='Action ID').duplicated(), 'bookings', 'exact duplicate (all fields but action id)')
    cx = drop(cx, cx['Worker ID'].isna(), 'cancels', 'no worker id')
    cx = drop(cx, ((cx.Action == 'WORKER_CANCEL') & (cx.lead < 0)) | ((cx.Action == 'NO_CALL_NO_SHOW') & (cx.lead > 0)),
              'cancels', 'lead time contradicts the action (cancel after start / NCNS before start)')
    cx = drop(cx, cx.drop(columns='Action ID').duplicated(), 'cancels', 'exact duplicate (all fields but action id)')
    cx = cx.sort_values('at')
    both = cx.groupby(['Shift ID', 'Worker ID']).Action.transform('nunique') > 1
    cx = drop(cx, both & cx.duplicated(['Shift ID', 'Worker ID'], keep='first'), 'cancels',
              'same shift+worker logged as both cancel and NCNS: keep the first-logged event')

cx['band'] = [band(l, a) for l, a in zip(cx.lead, cx.Action)]
cx['late_fail'] = cx.band.isin(LATE)

# ================================ Cleveland shift-level table ================================
s = sh.copy()
if MODE == 'clean':
    s = drop(s, s.Start.isna(), 'shifts', 'missing start time')
    s = drop(s, s['Agent Req'] == 'NURSE', 'shifts', "role 'NURSE': Charge always 0 and never worked (legacy/test shift type)")
    s = drop(s, ~s['Agent Req'].isin(CORE_ROLES), 'shifts', 'non-core roles with <35 shifts each (caregiver, med tech, PT, med aide)')
    dur = (s.End - s.Start).dt.total_seconds() / 3600
    s = drop(s, (dur <= 0) | (dur > 24) | (s.Time <= 0), 'shifts', 'impossible length (<=0h or >24h) or paid hours <=0')
    s = drop(s, s['Created At'] > s.Start, 'shifts', 'posted after it started (retroactive record)')
    s = drop(s, s['Worker ID'].isna() & s.Verified, 'shifts', 'verified with no worker attached (ambiguous outcome)')
for col in ['Start', 'End', 'Created At']:
    s[col + '_loc'] = s[col].dt.tz_localize('UTC').dt.tz_convert(TZ).dt.tz_localize(None)
s['has_worker'] = s['Worker ID'].notna(); s['deleted'] = s.Deleted.fillna(0).astype(bool)
s['worked'] = s.has_worker & s.Verified & ~s.deleted
s['charge'] = s.Charge.replace(0, np.nan)
s['dur'] = (s.End - s.Start).dt.total_seconds() / 3600
s['post_lead_h'] = (s.Start - s['Created At']).dt.total_seconds() / 3600
def add_fail_aggregates(s):
    cxc = cx[cx['Shift ID'].isin(set(s['Shift ID']))]
    agg = cxc.sort_values('at').groupby('Shift ID').agg(n_fail=('Action', 'size'), n_ncns=('band', lambda b: (b == '1_NCNS').sum()),
          n_late=('late_fail', 'sum'), last_band=('band', 'last'), last_at=('at', 'last'), last_lead=('lead', 'last'), min_band=('band', 'min'))
    s = s.drop(columns=[c for c in agg.columns if c in s.columns]).merge(agg, left_on='Shift ID', right_index=True, how='left')
    for col in ['n_fail', 'n_late', 'n_ncns']: s[col] = s[col].fillna(0)
    s['ever_booked'] = s.has_worker | (s.n_fail > 0) | s['Shift ID'].isin(set(bk['Shift ID']))
    return s
s = add_fail_aggregates(s)
if MODE == 'clean':
    b = s[s.ever_booked]
    d = b.groupby(b.Start_loc.dt.date).agg(n=('Shift ID', 'size'), late=('n_late', lambda x: (x > 0).mean()))
    d = d[d.n >= 50]; med = d.late.median(); rsd = 1.4826 * (d.late - med).abs().median()
    shock = sorted(d[(d.late - med) / rsd > SHOCK_Z].index)
    s = drop(s, s.Start_loc.dt.date.isin(shock), 'shifts', f"shock day (late-fail rate > {SHOCK_Z} robust SD above median day): {[str(x) for x in shock]}")
s['state'] = np.select([s.worked, s.deleted & s.has_worker, s.deleted & ~s.has_worker, ~s.has_worker & ~s.deleted],
                       ['worked', 'deleted_after_booked', 'deleted_unassigned', 'unfilled'], 'assigned_unverified')

# events on Cleveland shifts that were cleaned out leave the event logs too
if MODE == 'clean':
    gone = sh_ids_raw - set(s['Shift ID'])
    bk = drop(bk, bk['Shift ID'].isin(gone), 'bookings', 'shift removed by a Cleveland shift rule above')
    cx = drop(cx, cx['Shift ID'].isin(gone), 'cancels', 'shift removed by a Cleveland shift rule above')

# ================================ claim-level cohort (shift starts Oct 1-Nov 30, all markets) ================================
c = bk[bk['Worker ID'].notna()].copy()
c = c[(c.start >= '2021-10-01') & (c.start < '2021-12-01')]
ev = pd.concat([c[['Shift ID', 'Worker ID', 'at', 'Action ID']].assign(kind='claim'),
                cx[['Shift ID', 'Worker ID', 'at', 'Action ID', 'Action', 'lead', 'band']].dropna(subset=['Worker ID']).assign(kind='cancel')])
ev = ev.sort_values(['Shift ID', 'Worker ID', 'at', 'kind'], kind='stable')
g = ev.groupby(['Shift ID', 'Worker ID'], sort=False)
for col in ['kind', 'at', 'Action', 'lead', 'band']: ev['nx_' + col] = g[col].shift(-1)
cl = ev[ev.kind == 'claim'].merge(c[['Action ID', 'Facility ID', 'lead', 'start']], on='Action ID', suffixes=('_x', ''))
cl = cl.drop(columns=['lead_x', 'band', 'Action'])
cl['outcome'] = np.where(cl.nx_kind == 'cancel', np.where(cl.nx_Action == 'NO_CALL_NO_SHOW', 'ncns', 'cancel'), np.where(cl.nx_kind == 'claim', 'superseded', 'kept'))
cl = cl[cl.outcome != 'superseded'].copy()
cl['cancel_at'] = np.where(cl.outcome != 'kept', cl.nx_at, pd.NaT); cl['cancel_lead'] = np.where(cl.outcome != 'kept', cl.nx_lead, np.nan)
cl['band'] = np.where(cl.outcome == 'kept', '0_kept', cl.nx_band)
cl = cl.drop(columns=[x for x in cl.columns if x.startswith('nx_')] + ['kind'])
cl['claim_lead'] = cl['lead']; cl = cl.drop(columns='lead')
cl['tenure_d'] = (cl['at'] - oid_ts(cl['Worker ID'])).dt.total_seconds() / 86400
cl['fac_age_d'] = (cl['at'] - oid_ts(cl['Facility ID'])).dt.total_seconds() / 86400
cl['late_fail'] = cl.band.isin(LATE)
cl['cleveland'] = cl['Facility ID'].isin(set(sh['Facility ID']))

cl.to_pickle(f"{O}/claims.pkl"); s.to_pickle(f"{O}/cle_shifts.pkl"); cx.to_pickle(f"{O}/cx2.pkl"); bk.to_pickle(f"{O}/bk2.pkl")
print(f"mode: {MODE}")
if log:
    lg = pd.DataFrame(log); lg.to_csv(f"{O}/exclusions.csv", index=False)
    print("== exclusion log"); print(lg.to_string(index=False))
print("claims cohort:", len(cl), " outcomes:", cl.outcome.value_counts().to_dict())
print("Cleveland shifts:", len(s), " states:", s.state.value_counts().to_dict(), " ever_booked:", int(s.ever_booked.sum()))
