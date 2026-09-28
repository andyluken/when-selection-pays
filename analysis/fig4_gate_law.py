"""Reproduces Sec. IV-B: the gate-saturation law and its regression-to-the-mean control.

  python analysis/fig4_gate_law.py

Paper values: +0.1359 (n=2,049) / +0.0562 (n=280) / -0.0310 (n=3,584);
control against an independent seed: +0.1251 and -0.0269.
"""
import numpy as np
from common import load, scores, gate_product, paired, show

A = load("navhard", "ours_no_selection")         # defines the bins (seed 0)
S = load("navhard", "ours_xt4")                  # selection, sigma_T = 4
B = load("navhard", "ours_no_selection_seedB")   # independent re-draw (seed 1)

m = A.merge(S[["token", "score"]].rename(columns={"score": "s_sel"}), on="token")
m = m.merge(B[["token", "score"]].rename(columns={"score": "s_seedB"}), on="token")
sa, ss, sb = scores(m), np.asarray(m.s_sel, float), np.asarray(m.s_seedB, float)
g = gate_product(m)

print(f"\n  GATE-SATURATION LAW   (navhard, n={len(m)})\n")
bins = {"G = 0        ": g == 0,
        "0 < G <= 0.5 ": (g > 0) & (g <= 0.5),
        "G = 1        ": g == 1}
print(f"  {'bin':14} {'n':>5} {'baseline':>9} {'selected':>9} {'gain':>9}  {'t-test':>9}")
for k, v in bins.items():
    r = paired(ss[v], sa[v])
    print(f"  {k} {v.sum():5d} {sa[v].mean():9.4f} {ss[v].mean():9.4f} "
          f"{r['mean_diff']:+9.4f}  {r.get('t_p', float('nan')):9.2e}")

print(f"\n  CONTROL: vs an INDEPENDENT baseline draw (plays no part in the binning)\n")
for k, v in bins.items():
    show(k.strip() + "  selected - seedB", paired(ss[v], sb[v]))

z = g == 0
print(f"\n  How much of the +{(ss[z]-sa[z]).mean():.4f} at G=0 is the floor?\n")
print(f"    seed-0 (defines the bin)      {sa[z].mean():.4f}   (zero by construction)")
print(f"    seed-1 re-draw, NO scoring    {sb[z].mean():.4f}   <- the floor artifact")
print(f"    selected                      {ss[z].mean():.4f}")
den = ss[z].mean() - sa[z].mean()
print(f"    re-drawing alone accounts for {(sb[z].mean()-sa[z].mean())/den:.1%}")
print(f"    SCORING accounts for          {(ss[z].mean()-sb[z].mean())/den:.1%}")
print(f"    rescue rate, re-draw only     {np.mean(sb[z] > 0):.1%}")
print(f"    rescue rate, selection        {np.mean(ss[z] > 0):.1%}")

print(f"\n  Seed-1 is a valid equivalent baseline:")
r = paired(sb, sa)
print(f"    per-scenario means {sb.mean():.4f} vs {sa.mean():.4f}, "
      f"paired p={r.get('t_p', float('nan')):.2f}, "
      f"differing on {1 - r['tied']/r['n']:.1%} of scenarios\n")
