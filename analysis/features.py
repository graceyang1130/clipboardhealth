"""Booking-time features shared by segments.py and indicators.py.

Every feature uses only information available at the moment the booking was made:
past events are counted only if they were logged before the booking's timestamp.
"""
import numpy as np, pandas as pd


def _sorted_times(df, tcol):
    return {w: np.sort(g[tcol].values) for w, g in df.groupby('Worker ID')}


def _count_before(c, lookup):
    out = np.zeros(len(c), dtype=int); at = c['at'].values
    for w, idx in c.groupby('Worker ID').indices.items():
        t = lookup.get(w)
        if t is not None: out[idx] = np.searchsorted(t, at[idx], side='left')
    return out


def booking_features(cl, cx, with_holds=True):
    c = cl[cl.claim_lead >= 0].copy()
    c['cancel_at'] = pd.to_datetime(c.cancel_at)
    c = c.sort_values('at').reset_index(drop=True)
    c['ncns'] = c.outcome == 'ncns'
    c['late_cancel'] = c.band.isin(['2_<4h', '3_4-24h'])
    c['res_at'] = c.cancel_at.where(c.outcome != 'kept', c.start)       # when the booking's outcome became known
    ev = cx.dropna(subset=['Worker ID'])
    c['prior_ncns'] = _count_before(c, _sorted_times(ev[ev.band == '1_NCNS'], 'at'))
    c['prior_calloff'] = _count_before(c, _sorted_times(ev[ev.band == '2_<4h'], 'at'))
    c['prior_late_cancel'] = _count_before(c, _sorted_times(ev[ev.band == '3_4-24h'], 'at'))
    c['prior_early_cancel'] = _count_before(c, _sorted_times(ev[ev.band == '5_>72h'], 'at'))
    # resolved booking history (count, late failures) known before this booking
    prior_n = np.zeros(len(c), dtype=int); prior_lf = np.zeros(len(c), dtype=int)
    r = c[['Worker ID', 'res_at', 'late_fail']].sort_values('res_at')
    hist = {w: (g.res_at.values, np.cumsum(g.late_fail.values)) for w, g in r.groupby('Worker ID')}
    at = c['at'].values
    for w, idx in c.groupby('Worker ID').indices.items():
        t, cum = hist[w]; k = np.searchsorted(t, at[idx], side='left')
        prior_n[idx] = k; prior_lf[idx] = np.where(k > 0, cum[np.maximum(k - 1, 0)], 0)
    c['prior_n'] = prior_n
    c['prior_rate'] = np.where(prior_n > 0, prior_lf / np.maximum(prior_n, 1), np.nan)
    c['first_fac'] = c.groupby(['Worker ID', 'Facility ID']).cumcount() == 0
    # still on the books 24h before start (the population a late cancellation can come from)
    c['active24'] = (c.claim_lead >= 24) & ~((c.outcome == 'cancel') & (c.cancel_lead >= 24))
    if with_holds:
        # other bookings the worker was holding at the moment of this booking
        end_act = c.cancel_at.where(c.outcome == 'cancel', c.start).values
        holds = np.zeros(len(c), dtype=int)
        for w, idx in c.groupby('Worker ID').indices.items():
            a, e = at[idx], end_act[idx]
            holds[idx] = ((a[None, :] < a[:, None]) & (e[None, :] > a[:, None])).sum(1)
        c['holds'] = holds
    return c


def window_count(d, ev, lo_h, hi_h):
    """number of the worker's events logged in [t - hi_h, t - lo_h)"""
    lookup = {w: np.sort(g['at'].values) for w, g in ev.groupby('Worker ID')}
    out = np.zeros(len(d), dtype=int); t = d['t'].values
    for w, idx in d.groupby('Worker ID').indices.items():
        a = lookup.get(w)
        if a is None: continue
        hi = np.searchsorted(a, t[idx] - np.timedelta64(int(lo_h * 60), 'm'))
        lo = np.searchsorted(a, t[idx] - np.timedelta64(int(hi_h * 60), 'm')) if np.isfinite(hi_h) else 0
        out[idx] = hi - lo
    return out


def at_decision(c, cx, offset_h):
    """bookings still on the books `offset_h` hours before start, with worker history as known at that moment"""
    c = c.copy()
    c['end_act'] = c.cancel_at.where(c.outcome == 'cancel', c.start)
    c['lt4'] = c.band == '2_<4h'; c['c424'] = c.band == '3_4-24h'
    d = c[(c.claim_lead >= offset_h) & ~((c.outcome == 'cancel') & (c.cancel_lead >= offset_h))].copy()
    d['t'] = d.start - pd.Timedelta(hours=offset_h)
    ev = cx.dropna(subset=['Worker ID']); INF = np.inf
    d['k_ncns'] = window_count(d, ev[ev.band == '1_NCNS'], 0, INF)
    d['k_ncns_14d'] = window_count(d, ev[ev.band == '1_NCNS'], 0, 24 * 14)
    d['k_calloff'] = window_count(d, ev[ev.band == '2_<4h'], 0, INF)
    d['k_calloff_14d'] = window_count(d, ev[ev.band == '2_<4h'], 0, 24 * 14)
    d['k_drop_48h'] = window_count(d, ev[(ev.Action == 'WORKER_CANCEL') & (ev.lead < 24)], 0, 48)
    r = c[['Worker ID', 'res_at', 'late_fail', 'ncns', 'lt4', 'c424']].sort_values('res_at')
    hist = {w: (g.res_at.values, np.cumsum(g.late_fail.values), np.cumsum(g.ncns.values), np.cumsum(g.lt4.values), np.cumsum(g.c424.values))
            for w, g in r.groupby('Worker ID')}
    kept = c[c.outcome == 'kept'][['Worker ID', 'start']].sort_values('start')
    kept_t = {w: g.start.values for w, g in kept.groupby('Worker ID')}
    n = len(d); pn = np.zeros(n, int); cums = np.zeros((n, 4), int); gap = np.full(n, np.nan)
    holds = np.zeros(n, int); b2b = np.zeros(n, bool); conflict = np.zeros(n, bool)
    t = d.t.values; st = d.start.values; sid = d['Shift ID'].values
    allw = {w: g for w, g in c.groupby('Worker ID')}
    for w, idx in d.groupby('Worker ID').indices.items():
        tt, *cs = hist[w]; k = np.searchsorted(tt, t[idx]); pn[idx] = k
        for j, arr in enumerate(cs): cums[idx, j] = np.where(k > 0, arr[np.maximum(k - 1, 0)], 0)
        kt = kept_t.get(w)
        if kt is not None:
            j = np.searchsorted(kt, t[idx]); has = j > 0
            gap[idx[has]] = (t[idx][has] - kt[j[has] - 1]) / np.timedelta64(1, 'D')
        g = allw[w]; a_at = g['at'].values; a_end = g.end_act.values; a_st = g.start.values; a_sid = g['Shift ID'].values
        for i in idx:
            act = (a_at < t[i]) & (a_end > t[i]) & (a_sid != sid[i]); holds[i] = act.sum()
            held = (a_sid != sid[i]) & (a_at < t[i]) & (((a_st > t[i]) & act) | ((a_st <= t[i]) & (a_end >= a_st)))
            dh = (st[i] - a_st[held]) / np.timedelta64(1, 'h')
            b2b[i] = ((dh >= 6) & (dh <= 16)).any(); conflict[i] = (np.abs(dh) < 4).any()
    rate = lambda col: np.where(pn > 0, cums[:, col] / np.maximum(pn, 1), np.nan)
    d['prior_n_t'] = pn; d['rate_lf'] = rate(0); d['rate_ncns'] = rate(1); d['rate_lt4'] = rate(2); d['rate_424'] = rate(3)
    d['days_since_worked'] = gap; d['holds_t'] = holds; d['b2b'] = b2b; d['conflict'] = conflict
    d['tenure_t'] = d.tenure_d + (d.t - d['at']).dt.total_seconds() / 86400
    return d


def confirmation_deadline(start_loc):
    """start window and hours between the confirmation deadline and the start (local start times).
    morning 5am-noon: 7pm the evening before | noon-2pm: 9pm the evening before | 2pm-9pm: 6h before |
    night 9pm-5am: 1pm on the day the night begins (the day before for midnight-5am starts)"""
    h = start_loc.dt.hour + start_loc.dt.minute / 60
    window = np.select([(h >= 5) & (h < 12), (h >= 12) & (h < 14), (h >= 14) & (h < 21)], ['morning', 'noon', 'afternoon/evening'], 'night')
    notice = np.select([window == 'morning', window == 'noon', window == 'afternoon/evening', h >= 21], [h + 5, h + 3, 6.0, h - 13], h + 11)
    return window, notice
