#!/bin/zsh
# Usage: ./run_all.sh <python with pandas+openpyxl+scipy+statsmodels> <scratch work dir>
# Rebuilds everything from the original xlsx files. Every analysis runs on CLEAN data (rules + exclusion log in build.py);
# a raw build is kept only so baselines.py can show what cleaning changed. Each step's printout goes to outputs/.
PY=$1; W=$2; HERE=${0:A:h}; DATA="$HERE/../Cancellations Data Files"
mkdir -p "$W" "$HERE/outputs"
run() { echo "== $1"; $PY -I "$HERE/$1" "${@:2}" > "$HERE/outputs/${1%.py}.txt" 2>&1 || { echo "FAILED: $1 (see outputs/${1%.py}.txt)"; exit 1; } }
run load.py "$DATA" "$W"
run build.py "$W" "$W/clean" clean        # Phase 0: cleaning rules + exclusion log, claim cohort, Cleveland shift table
$PY -I "$HERE/build.py" "$W" "$W/raw" raw > /dev/null 2>&1 || { echo "FAILED: raw build"; exit 1; }
C="$W/clean"
run baselines.py "$C" "$W/raw" "$HERE/outputs"   # target baselines on clean data (raw alongside)
run p1.py "$C"         # Phase 1: rates, lead-time CDF, refill curve
run p1b.py "$C"        # Phase 1: $ sizing, trend, holidays, facility churn proxy
run p2w.py "$C"        # Phase 2: worker concentration, persistence, tenure, facility familiarity
run p2b.py "$C"        # Phase 2: booking lead, concurrent holds, conflicts, switching, rest gaps
run p2c.py "$C"        # Phase 2: shift/facility attributes, local-time cancel timing
run p3.py "$C"         # Phase 3: threshold bunching, strike reconstruction, enforcement proxy
run p4.py "$C"         # hazard by hour, segment sizing, supply by reliability segment, baselines/MDE
run p5.py "$C"         # segment detail, refill drivers, NCNS signature
run segments.py "$C" "$HERE/outputs"     # subgroup sizing + what-if levers (also writes outputs/segments_*.csv)
run indicators.py "$C" "$HERE/outputs"   # which booking-time signals are attributable higher-risk indicators
run deepdive.py "$C" "$HERE/outputs"    # prevention + recovery deep-dive (prize matrix, streaks, golden hour, overnight gap, NCNS ceiling)
run flags.py "$C" "$HERE/outputs"       # early-warning flags for NCNS / <4h / 4-24h / any late failure (catches vs false alarms)
run advanced.py "$C" "$HERE/outputs"    # penalty RD at policy cut-offs, empirical-Bayes reliability, pay->refill, worker/facility/day variance
run deadlines.py "$C"                  # confirmation-deadline timing by shift start window
run sizing.py "$C" "$HERE/outputs"      # proposal sizing: who gets checked, impact scenarios, cost, pilot power
run roles.py "$C" "$HERE/outputs" "$W/raw"   # role profile, role x start-window segmentation, role-specific settings
run figures.py "$C" "$HERE/outputs" "$HERE/../reporting/figures"   # appendix figures for the report
