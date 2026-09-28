"""Reproduces Sec. V: the ordering ceiling and where our proposals sit on it.

  python analysis/fig5_ceiling.py

Fig. 5 is derived from GTRS's published per-trajectory labels (see README:
those are a third-party 30 GB artifact and are not redistributed here). The
derived curves are shipped so the figure and every number quoted from it can
be checked without them.
"""
import pandas as pd
from common import ROOT

for f, title in [("fig5a_scorer_separation.csv", "Fig. 5(a)  scorer ranking vs candidate separation"),
                 ("fig5b_orderability.csv",      "Fig. 5(b)  do the LABELS order the candidates?"),
                 ("proposal_separation.csv",     "Where our own proposals sit (120 navhard scenes)")]:
    print(f"\n  {title}\n")
    print(pd.read_csv(ROOT / "results" / "derived" / f).to_string(index=False, justify="left"))
print("""
  Reading the two together, as Sec. V-B does:
    sigma_T = 2  ->  median  1.36 m  ->  roughly 59% of candidate sets tied
    sigma_T = 4  ->  median  5.94 m  ->  roughly 26% tied
  The ceiling is not a standing limit on the deployed planner; it is the
  regime the diversity knob escapes.
""")
