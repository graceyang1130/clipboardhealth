"""When should each shift's confirmation deadline be?  (Cleveland, clean data, local time)

Usage: python deadlines.py <clean_dir>
For each start-time window: when late drops happen, and how often a dropped shift still gets worked by notice and by the
clock time of the drop. A good deadline leaves enough notice to refill AND lands while workers are awake.
"""
import sys, numpy as np, pandas as pd
S = sys.argv[1]; pd.set_option("display.width", 220)
s = pd.read_pickle(f"{S}/cle_shifts.pkl")
loc = lambda t: t.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
win = lambda h: pd.cut(h, [-1, 3, 10, 16, 20, 24], labels=['night', 'morning', 'afternoon', 'evening', 'night2']).astype(str).replace('night2', 'night')
b = s[s.ever_booked].copy(); b['window'] = win(b.Start_loc.dt.hour)
print("start-time windows (local start hour): morning 4-10am | afternoon 11am-4pm | evening 5-8pm | night 9pm-3am")
print("booked shifts and most common start times:")
print(b.groupby('window').agg(booked=('Shift ID', 'size'), top_start_hours=('Start_loc', lambda t: t.dt.hour.value_counts().head(3).to_dict())).to_string())
L = s[(s.n_fail > 0) & s.last_band.isin(['2_<4h', '3_4-24h'])].copy()
L['window'] = win(L.Start_loc.dt.hour); L['drop_hr'] = loc(L.last_at).dt.hour
L['notice'] = pd.cut(L.last_lead, [0, 2, 4, 6, 8, 12, 16, 24], right=False)
L['drop_clock'] = pd.cut(L.drop_hr, [-1, 6, 11, 16, 21, 24], labels=['overnight 10pm-7am', 'morning 7-11am', 'midday 12-4pm', 'evening 5-9pm', 'overnight2']).astype(str).replace('overnight2', 'overnight 10pm-7am')
print("\nworked after a <24h drop, by start window x notice (n in brackets)")
t = L.pivot_table(index='notice', columns='window', values='worked', aggfunc=['mean', 'size'], observed=True)
print(pd.DataFrame({w: [f"{m:.2f} [{n:.0f}]" if n == n else "-" for m, n in zip(t['mean'][w], t['size'][w])] for w in t['mean'].columns}, index=t.index.astype(str)).to_string())
print("\nworked after a <24h drop, by start window x clock time of the drop")
t2 = L.pivot_table(index='drop_clock', columns='window', values='worked', aggfunc=['mean', 'size'], observed=True)
print(pd.DataFrame({w: [f"{m:.2f} [{n:.0f}]" if n == n else "-" for m, n in zip(t2['mean'][w], t2['size'][w])] for w in t2['mean'].columns}, index=t2.index).to_string())
print("\nwhen late (<24h) drops happen: share by clock time of the drop, per start window")
print(pd.crosstab(L.drop_clock, L.window, normalize='columns').round(2).to_string())
print("\nmedian notice (h) of late drops per window:", L.groupby('window').last_lead.median().round(1).to_dict())
