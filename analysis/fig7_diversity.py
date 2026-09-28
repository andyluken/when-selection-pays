"""Reproduces Sec. VI: the diversity sweep and the proposal-count null.

  python analysis/fig7_diversity.py
"""
from common import load, scores, gate_product, paired, show
import numpy as np

runs = ["ours_no_selection", "ours_xt2", "ours_xt3", "ours_xt4", "ours_xt5"]
d = {r: load("navhard", r) for r in runs}
base = d["ours_no_selection"]

print(f"\n  DIVERSITY SWEEP (navhard) -- paired against no selection\n")
for r in runs[1:]:
    m = base.merge(d[r][["token", "score"]].rename(columns={"score": "s2"}), on="token")
    show(f"{r:<20} vs no selection", paired(np.asarray(m.s2, float), scores(m)))

print(f"\n  Is the peak real? pairwise around sigma_T = 4\n")
for a, b in [("ours_xt4", "ours_xt3"), ("ours_xt5", "ours_xt4"), ("ours_xt4", "ours_xt2")]:
    m = d[a][["token", "score"]].merge(d[b][["token", "score"]], on="token",
                                       suffixes=("_a", "_b"))
    show(f"{a} vs {b}", paired(np.asarray(m.score_a, float), np.asarray(m.score_b, float)))

# Proposal COUNT is a DIFFERENT knob from diversity, so the K=128 run must be
# paired against K=64 at the SAME sigma_T (both 2.0). Pairing it against xt4
# would conflate the count change with the diversity change.
print(f"\n  Proposal COUNT is saturated (K=64 -> 128, both at sigma_T = 2)\n")
m = d["ours_xt2"][["token", "score"]].merge(
    load("navhard", "ours_k128")[["token", "score"]], on="token", suffixes=("_64", "_128"))
show("K=128 vs K=64 (sigma_T=2)", paired(np.asarray(m.score_128, float),
                                         np.asarray(m.score_64, float)),
     "+0.0012, p=0.49, Wilcoxon negative")
print()
