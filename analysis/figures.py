"""Appendix figures for the proposal (Cleveland, clean data).

Usage: python figures.py <clean_dir> <outputs_dir> <figure_dir>
  fig_notice.pdf/.png  how often a shift is still worked, by the notice the worker gave; revenue lost per year by failure type
  fig_deadlines.pdf/.png  when late failures happen relative to the shift start, per start window, with the confirmation deadline
Reads outputs/sizing_windows.csv (written by sizing.py) for the deadline annotations.
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import confirmation_deadline
S, O, F = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(F, exist_ok=True)
TAKE, ANNUAL = 0.22, 3
LATE = ['1_NCNS', '2_<4h', '3_4-24h']
ACCENT, GREY, DARK = '#c2410c', '#b8b8b8', '#333333'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': DARK, 'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK})
s = pd.read_pickle(f"{S}/cle_shifts.pkl")
med = s.groupby('Agent Req').charge.median()
s['gmv'] = s.charge.fillna(s['Agent Req'].map(med)) * s.Time.clip(lower=0)

# ================================================================ figure 1: notice decides whether the shift is saved
f = s[s.n_fail > 0].copy()
edges = [0, 1, 2, 4, 8, 12, 24, 48, 72, 168, np.inf]
labels = ['<1h', '1-2h', '2-4h', '4-8h', '8-12h', '12-24h', '1-2d', '2-3d', '3-7d', '7d+']
f['notice'] = np.where(f.last_band == '1_NCNS', 'NCNS', pd.cut(f.last_lead, edges, labels=labels, right=False).astype(str))
order = ['NCNS'] + labels
g = f.groupby('notice').agg(n=('worked', 'size'), worked=('worked', 'mean')).reindex(order)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.3), gridspec_kw={'width_ratios': [2.1, 1]})
cols = [ACCENT if o in order[:7] else GREY for o in order]
a.bar(range(len(order)), g.worked * 100, color=cols, width=0.75)
for i, v in enumerate(g.worked): a.text(i, v * 100 + 1.5, f"{v:.0%}", ha='center', fontsize=7)
a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=40, ha='right', fontsize=6.5)
a.set_ylabel('shift still worked (%)'); a.set_ylim(0, 75)
a.set_xlabel('notice the worker gave (late failures in orange)')
a.set_title('A. Chance the shift is still worked, by notice', loc='left', fontsize=8.5)
L = f[f.last_band.isin(LATE) & (f.state == 'unfilled')]
lost = L.groupby('last_band').gmv.sum() * TAKE * ANNUAL / 1000
unf = f[f.last_band.isin(LATE)].groupby('last_band').apply(lambda x: (x.state == 'unfilled').mean())
names = {'1_NCNS': 'NCNS', '2_<4h': '<4h cancel', '3_4-24h': '4-24h cancel'}
b.barh([names[k] for k in LATE][::-1], [lost[k] for k in LATE][::-1], color=ACCENT, height=0.6)
for i, k in enumerate(LATE[::-1]): b.text(lost[k] + 4, i, f"${lost[k]:.0f}k  ({unf[k]:.0%} unfilled)", va='center', fontsize=7)
b.set_xlim(0, 360); b.set_xlabel('revenue lost per year ($k)')
b.set_title('B. Revenue lost per year', loc='left', fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{F}/fig_notice.pdf"); fig.savefig(f"{F}/fig_notice.png", dpi=200); plt.close(fig)
print("fig_notice:", g.round(3).to_dict(), {k: round(v, 1) for k, v in lost.items()})

# ================================================================ figure 2: when late failures happen vs the deadline
W = pd.read_csv(f"{O}/sizing_windows.csv").set_index('window')
s['window'], s['dl_notice'] = confirmation_deadline(s.Start_loc)
L = s[(s.n_fail > 0) & s.last_band.isin(LATE)].copy()
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
for ax, win, title in zip(axes, ['morning', 'afternoon/evening', 'night'],
                          ['Morning starts (5am-noon)\ndeadline 7pm the evening before', 'Afternoon/evening starts (2-9pm)\ndeadline 6h before start',
                           'Night starts (9pm-5am)\ndeadline 1pm']):
    x = L[L.window == win]; n = len(x)
    bins = np.arange(0, 25, 1)
    cnt, _ = np.histogram(x.loc[x.last_band != '1_NCNS', 'last_lead'], bins=bins)
    dl = W.loc[win, 'deadline_notice_h']
    ax.bar(-1.2, (x.last_band == '1_NCNS').sum() / n * 100, width=1.6, color=ACCENT)
    ax.bar(bins[:-1] + 0.5, cnt / n * 100, width=0.9, color=[ACCENT if b + 0.5 < dl else GREY for b in bins[:-1]])
    ax.axvline(dl, color=DARK, ls='--', lw=0.9)
    ax.text(dl + 0.3, 2, f"deadline (median {dl:.0f}h)", fontsize=6, rotation=90, va='bottom')
    r = W.loc[win]
    ax.text(0.99, 0.97, f"{r.share_after_deadline:.0%} after the deadline\nworked today: {r.worked_today_after_deadline:.0%}\n"
                        f"if released at deadline: {r.worked_if_surfaced_at_deadline:.0%}", fontsize=6, ha='right', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='square,pad=0.25', fc='white', ec='none'))
    ax.set_title(title, loc='left', fontsize=7.5); ax.set_xlim(-2.5, 24.5)
    ax.set_xticks([-1.2, 4, 8, 12, 16, 20, 24]); ax.set_xticklabels(['NCNS', '4h', '8h', '12h', '16h', '20h', '24h'], fontsize=6.5)
axes[0].set_ylabel('share of late failures (%)'); axes[0].set_ylim(0, 40)
fig.supxlabel('hours before start when the worker dropped (orange = after the deadline)', fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{F}/fig_deadlines.pdf"); fig.savefig(f"{F}/fig_deadlines.png", dpi=200); plt.close(fig)
print("fig_deadlines written")
