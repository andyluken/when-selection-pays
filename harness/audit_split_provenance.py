"""Which components saw the evaluation logs, and what is that worth?

The paper (Sec. III-E, "Data provenance") claims the split boundary was audited
at LOG level rather than inferred from split names. This script is that audit.

Why it cannot be inferred: nuPlan's own `splits/trainval` directory CONTAINS the
logs OpenScene/NAVSIM later designate `test`. Anything built by globbing
`splits/trainval/*.db` therefore touches evaluation logs even though its split
is nominally "trainval". That is exactly what happened to the frozen
self-supervised encoder here, and only to it.

Reported:
  planner training logs  vs evaluation logs ->   0 / 147   (disjoint)
  encoder pre-train logs vs evaluation logs -> 147 / 147   (NOT disjoint)
  share of encoder pre-training frames drawn from evaluation logs -> 10.6%

The overlap is self-supervised only (optical flow + vehicle state; never a
trajectory or a score), the encoder is frozen, and it is identical in both arms
of every paired comparison, so no claim in the paper rests on it. Its total
worth is bounded by the zero-vision ablation at 0.0342 EPDMS -- see
`analysis/table1_navhard.py` for the paired test behind that number.

Level 1 (default, no data needed): verifies the shipped log table reproduces the
numbers quoted in the paper.

    python audit_split_provenance.py

Level 2: regenerate that table from the raw sources, if you have them.

    python audit_split_provenance.py --regenerate \
        --encoder-index   <frames_dir>/index.json \
        --planner-samples <dit_training_data>/samples_k1.json \
        --eval-logs       $OPENSCENE_DATA_ROOT/navsim_logs/test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE.parent / "results" / "derived" / "provenance_logs.csv"

# as quoted in Sec. III-E
PAPER = {"planner_overlap": 0, "encoder_overlap": 147,
         "eval_logs": 147, "encoder_logs": 1249, "planner_logs": 1135,
         "exposure_pct": 10.6}


def regenerate(encoder_index: Path, planner_samples: Path, eval_logs: Path) -> list[dict]:
    enc = {e["log_name"]: e["n_frames"]
           for e in json.loads(encoder_index.read_text())["logs"]}
    s = json.loads(planner_samples.read_text())
    s = s["samples"] if isinstance(s, dict) and "samples" in s else s
    planner = {x["log_name"].replace(".db", "") for x in s}
    ev = {p.replace(".pkl", "").replace(".db", "") for p in os.listdir(eval_logs)}
    return [{"log_name": log,
             "encoder_pretrain_frames": enc.get(log, 0),
             "in_encoder_pretrain": int(log in enc),
             "in_planner_train": int(log in planner),
             "in_eval_split": int(log in ev)}
            for log in sorted(set(enc) | planner | ev)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild the table from raw sources instead of reading it")
    ap.add_argument("--encoder-index", type=Path)
    ap.add_argument("--planner-samples", type=Path)
    ap.add_argument("--eval-logs", type=Path)
    a = ap.parse_args()

    if a.regenerate:
        missing = [n for n in ("encoder_index", "planner_samples", "eval_logs")
                   if getattr(a, n) is None]
        if missing:
            ap.error("--regenerate needs " + ", ".join("--" + m.replace("_", "-")
                                                       for m in missing))
        rows = regenerate(a.encoder_index, a.planner_samples, a.eval_logs)
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        with TABLE.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"regenerated {TABLE.relative_to(HERE.parent)}  ({len(rows)} logs)\n")
    else:
        rows = [{k: (v if k == "log_name" else int(v)) for k, v in r.items()}
                for r in csv.DictReader(TABLE.open())]

    ev = [r for r in rows if r["in_eval_split"]]
    enc = [r for r in rows if r["in_encoder_pretrain"]]
    pln = [r for r in rows if r["in_planner_train"]]
    enc_ev = [r for r in ev if r["in_encoder_pretrain"]]
    pln_ev = [r for r in ev if r["in_planner_train"]]

    frames_all = sum(r["encoder_pretrain_frames"] for r in enc)
    frames_ev = sum(r["encoder_pretrain_frames"] for r in enc_ev)
    exposure = 100.0 * frames_ev / frames_all

    def line(label, got, want, fmt="{:,}"):
        ok = "ok" if (abs(got - want) < 0.05 if isinstance(want, float) else got == want) else "MISMATCH"
        print(f"  {label:<46} {fmt.format(got):>12}   paper {fmt.format(want):>12}   {ok}")

    print("Split provenance audit (paper Sec. III-E)\n")
    print("  component                                       measured        reported")
    line("evaluation logs", len(ev), PAPER["eval_logs"])
    line("planner training logs", len(pln), PAPER["planner_logs"])
    line("  ... of which are evaluation logs", len(pln_ev), PAPER["planner_overlap"])
    line("encoder pre-training logs", len(enc), PAPER["encoder_logs"])
    line("  ... of which are evaluation logs", len(enc_ev), PAPER["encoder_overlap"])
    line("encoder frames from evaluation logs (%)", exposure,
         PAPER["exposure_pct"], "{:.1f}")

    print(f"\n  encoder pre-training frames: {frames_ev:,} of {frames_all:,}")
    print("\n  Supervised components are disjoint from evaluation; the frozen"
          "\n  self-supervised encoder is not. Bounded at 0.0342 EPDMS by the"
          "\n  zero-vision ablation, and it cancels in every paired comparison.")


if __name__ == "__main__":
    main()
