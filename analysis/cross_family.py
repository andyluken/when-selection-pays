"""Cross-family replication: GTRS-Dense generates, our selection stage disposes.

Reproduces the two claims in Sec. IV-C "Replacing our own generator":
  (1) the AGGREGATE does not replicate (+0.0046, p=0.14), and
  (2) the per-bin MECHANISM does, more sharply than on our own planner
      (Spearman -0.409 vs -0.211), with occupancy x bin-gain reproducing the
      aggregate exactly -- i.e. (1) is a prediction of (2), not a contradiction.

Prediction was registered before the run: harness/PREREGISTRATION_cross_family.md

  python analysis/cross_family.py
"""
import numpy as np
from scipy import stats
from common import load, scores, gate_product

SEL, BASE = "gtrs_props_ourscorer", "gtrs_dense"
OURS = {"G=0": 0.1359, "0<G<1": 0.0562, "G=1": -0.0310}
OURS_OCC = {"G=0": 0.375, "0<G<1": 0.052, "G=1": 0.573}

s, b = load("navhard", SEL), load("navhard", BASE)
m = b.merge(s[["token", "score"]].rename(columns={"score": "s_sel"}), on="token")
base, sel = scores(m), np.asarray(m.s_sel, float)
ok = ~(np.isnan(base) | np.isnan(sel))
G = gate_product(m)[ok]; d = (sel - base)[ok]

print(f"\nAGGREGATE  (n={len(d)}, paired on identical tokens)")
se = d.std(ddof=1) / np.sqrt(len(d))
ci = stats.t.interval(0.95, len(d) - 1, d.mean(), se)
print(f"  paired gain {d.mean():+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
print(f"  t p={stats.ttest_rel(sel[ok], base[ok]).pvalue:.3f}   "
      f"Wilcoxon p={stats.wilcoxon(sel[ok], base[ok]).pvalue:.2e}")
print(f"  better {(d>1e-9).sum()} / worse {(d<-1e-9).sum()} / tied {(abs(d)<=1e-9).sum()}")
print("  -> pre-registered AMBIGUOUS band (0, +0.010): NOT a confirmation.")

print(f"\nPER-BIN MECHANISM  (bins = GTRS-Dense's OWN baseline gate product)")
print(f"  {'bin':<10}{'n':>7}{'gain':>10}{'p':>11}{'ours':>10}")
bins = {"G=0": G <= .001, "0<G<1": (G > .001) & (G <= .999), "G=1": G > .999}
occ, gain = {}, {}
for k, sel_m in bins.items():
    occ[k], gain[k] = sel_m.mean(), d[sel_m].mean()
    p = stats.ttest_1samp(d[sel_m], 0).pvalue
    print(f"  {k:<10}{sel_m.sum():>7}{gain[k]:>+10.4f}{p:>11.1e}{OURS[k]:>+10.4f}")
rho, p = stats.spearmanr(G, d)
print(f"  Spearman(G, gain) = {rho:+.4f}, p={p:.2e}   (ours: -0.2114, p=9.9e-61)")

print(f"\nWHY THE AGGREGATE IS NULL  (occupancy x bin-gain)")
tot = sum(occ[k] * gain[k] for k in occ)
for k in ["G=0", "0<G<1", "G=1"]:
    print(f"  {k:<10} occ {occ[k]:.3f} x gain {gain[k]:+.4f} = {occ[k]*gain[k]:+.4f}"
          f"   | ours occ {OURS_OCC[k]:.3f}")
print(f"  {'sum':<10} {tot:+.4f}   measured {d.mean():+.4f}")
pre = sum(OURS[k] * occ[k] for k in occ)
print(f"\n  pre-registered estimate (OUR bin gains x GTRS occupancy): {pre:+.4f}")
print(f"  the two POSITIVE bins transferred; the miss is the G=1 penalty")
print(f"  deepening {OURS['G=1']:+.4f} -> {gain['G=1']:+.4f}.")
