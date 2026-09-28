"""Reproduces Table I: the navhard comparison, with every row paired on identical tokens.

  python analysis/table1_navhard.py
"""
from common import load, scores, paired, show
import numpy as np

rows = ["gtrs_dense", "ours_xt4", "ours_external_scorer", "ours_no_selection",
        "ours_sensor_only_sel", "ltf", "ours_sensor_only"]
print(f"\n  navhard per-scenario means (composed EPDMS is a two-stage product,\n"
      f"  reported in the paper; the per-scenario mean below is NOT the same number)\n")
for r in rows:
    try:
        s = scores(load("navhard", r))
        print(f"  {r:<24} mean {np.nanmean(s):.4f}   n={np.sum(~np.isnan(s))}")
    except FileNotFoundError as e:
        print(f"  {r:<24} MISSING")

print(f"\n  The paired claims the paper actually makes\n")
def cmp(a, b, label, ref):
    A, B = load("navhard", a), load("navhard", b)
    m = A[["token", "score"]].merge(B[["token", "score"]], on="token", suffixes=("_a", "_b"))
    show(label, paired(np.asarray(m.score_a, float), np.asarray(m.score_b, float)), ref)

cmp("ours_xt4", "ours_no_selection",  "selection vs none",        "+0.0310, t=3.2e-18")
cmp("ours_xt4", "ours_external_scorer","our scorer vs external",  "+0.0004, p=0.90")
cmp("ours_sensor_only_sel", "ours_sensor_only", "selection, sensor-only", "+0.0237, t=1.1e-24")
cmp("ours_sensor_only_sel", "ltf",    "sensor-only vs LTF",       "-0.0052, p=0.24 (a tie)")
print()
