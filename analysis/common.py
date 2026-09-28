"""Shared helpers for reproducing the paper's numbers from the shipped CSVs.

Everything here runs on CPU from the per-scenario score tables in results/.
No GPU, no simulator, no NAVSIM install required.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
GATES = ["no_at_fault_collisions", "drivable_area_compliance",
         "driving_direction_compliance", "traffic_light_compliance"]


def load(split: str, name: str) -> pd.DataFrame:
    """Load one run's per-scenario table, dropping any aggregate row."""
    f = ROOT / "results" / split / f"{name}.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f}\n  (see README: which tables ship with the repo)")
    d = pd.read_csv(f)
    return d[d.token != "average"].copy()


def scores(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["score"], errors="coerce").values


def gate_product(df: pd.DataFrame, stage: str = "stage_two") -> np.ndarray:
    """Product of the four compliance gates. NaN (no such stage) -> 1.0.

    navhard tokens are EITHER stage-one or stage-two; stage-one-only tokens
    have no stage-two columns and fall in the G=1 bin, which reproduces the
    paper's bin sizes exactly (2,049 / 280 / 3,584).
    """
    g = np.prod([pd.to_numeric(df[f"{c}_{stage}"], errors="coerce").values
                 for c in GATES], axis=0)
    return np.where(np.isnan(g), 1.0, g)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison of a against b. Both tests, as the paper requires."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    d = a - b
    out = {"n": int(ok.sum()), "mean_diff": float(d.mean()),
           "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
           "tied": int((d == 0).sum())}
    if np.any(d != 0):
        t = stats.ttest_rel(a, b)
        out["t_p"] = float(np.ravel(t.pvalue)[0])
        se = d.std(ddof=1) / np.sqrt(len(d))
        out["ci95"] = (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se))
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    return out


def merge(split: str, *names: str) -> pd.DataFrame:
    """Inner-join several runs on token so every comparison is paired."""
    base = load(split, names[0])[["token"] + [c for c in load(split, names[0]).columns
                                              if c != "token"]]
    out = base.rename(columns={"score": f"score__{names[0]}"})
    for n in names[1:]:
        d = load(split, n)[["token", "score"]].rename(columns={"score": f"score__{n}"})
        out = out.merge(d, on="token")
    return out


def show(label: str, r: dict, ref: str = "") -> None:
    ci = r.get("ci95")
    ci_s = f"  CI[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else ""
    print(f"  {label:<38} {r['mean_diff']:+.4f}{ci_s}"
          f"  t={r.get('t_p', float('nan')):.2e}"
          f"  W={r.get('wilcoxon_p', float('nan')):.2e}  n={r['n']}")
    if ref:
        print(f"  {'':38} paper: {ref}")
