"""Per-scenario inference cost of the selection stage.

What "1.37 M vs 83 M parameters" costs in wall-clock. Three REAL deployed agent
configurations are timed end to end and differenced:

    (a) planner only            num_proposals=1, no scorer
    (b) + our distilled scorer  num_proposals=64, own_scorer_ckpt
    (c) + external GTRS scorer  num_proposals=64, gtrs_scorer_ckpt

Our selection is free because of two design facts rather than luck: the 64
proposals are ONE batched denoising pass (the batch dimension, not 64 forward
passes), and the scorer is a small decoder over the 45-token scene memory the
planner has already computed. The external scorer must run its own V2-99
backbone over eight cameras -- a second perception stack -- which is the whole
+148 ms.

METHODOLOGY (a first attempt got this wrong, so it is worth stating): the three
configurations must be INTERLEAVED, not run back to back. Run sequentially,
config (a) absorbed all the cuDNN autotuning and GPU clock ramp and timed SLOWER
than (b) despite doing strictly less work -- a 17 ms artifact LARGER than the
effect being measured. Here all three agents stay resident, each is warmed on
--warmup tokens, and then per token all three are timed with the starting config
ROTATED, so residual drift is spread evenly rather than charged to whoever ran
first. CUDA is synchronised around every call.

REPORT THE MEDIAN, not the mean. Config (b) has an outlier tail (std 172 ms
against 67 and 78) that lifts its mean to 384.5 while its median is 358.9. The
GTRS contrast is robust either way (+148 median, +149 mean); the "ours is free"
claim rests on the median and is stated as "no measurable latency", not "zero".

Level 1 (default, no GPU): print the measured table shipped with this repo.

    python measure_inference_cost.py

Level 2: re-measure on your own hardware. Absolute numbers are hardware
specific; the DIFFERENCES between configurations are the claim.

    export OPENSCENE_DATA_ROOT=...
    python measure_inference_cost.py --remeasure \
        --agent-config-dir <navsim-clone>/navsim/planning/script/config/common/agent \
        --gtrs-ckpt <path>/gtrs_dense_vov.ckpt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
TABLE = HERE.parent / "results" / "derived" / "inference_cost.csv"


def show(rows: list[dict]) -> None:
    print(f"{'configuration':<40}{'median ms':>11}{'mean':>9}{'p90':>8}{'selection':>12}")
    for r in rows:
        cost = float(r["selection_cost_ms"])
        tag = "—" if r is rows[0] else f"{cost:+.1f} ms"
        print(f"{r['configuration']:<40}{float(r['median_ms']):>11.1f}"
              f"{float(r['mean_ms']):>9.1f}{float(r['p90_ms']):>8.1f}{tag:>12}")
    print(f"\n  n = {rows[0]['n_tokens']} tokens per configuration, interleaved.")
    print("  Ours adds no measurable latency; the external scorer adds ~41%.")


def remeasure(a: argparse.Namespace) -> list[dict]:
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader

    def build(overrides):
        with initialize_config_dir(config_dir=str(a.agent_config_dir), version_base=None):
            agent = instantiate(compose(config_name=a.config_name, overrides=overrides))
        agent.initialize()
        return agent

    def once(agent, scene, agent_input):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent.compute_trajectory(agent_input, scene)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    specs = [
        ("planner only (K=1, no selection)", ["num_proposals=1", "own_scorer_ckpt=null"]),
        ("+ ours, K=64 (1.37M, no new sensors)", ["num_proposals=64"]),
        ("+ GTRS-Dense, K=64 (83M, 8 cameras)",
         ["num_proposals=64", "own_scorer_ckpt=null", f"+gtrs_scorer_ckpt={a.gtrs_ckpt}"]),
    ]
    print("building agents ...", flush=True)
    agents = [(n, build(o)) for n, o in specs]

    root = os.environ["OPENSCENE_DATA_ROOT"]
    loader = SceneLoader(
        data_path=pathlib.Path(f"{root}/navsim_logs/test"),
        original_sensor_path=pathlib.Path(f"{root}/sensor_blobs/test"),
        scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True),
        sensor_config=agents[-1][1].get_sensor_config())
    tokens = loader.tokens[:a.n]
    print(f"caching {len(tokens)} tokens ...", flush=True)
    cache = {t: (loader.get_scene_from_token(t), loader.get_agent_input_from_token(t))
             for t in tokens}

    print(f"warming ({a.warmup} tokens x {len(agents)} agents) ...", flush=True)
    for t in tokens[:a.warmup]:
        scene, agent_input = cache[t]
        for _, agent in agents:
            once(agent, scene, agent_input)

    timings = {n: [] for n, _ in agents}
    for i, t in enumerate(tokens[a.warmup:]):
        scene, agent_input = cache[t]
        for k in [(i + j) % len(agents) for j in range(len(agents))]:   # rotate start
            name, agent = agents[k]
            timings[name].append(once(agent, scene, agent_input))

    out, base = [], None
    for name, _ in agents:
        v = np.array(timings[name])
        base = float(np.median(v)) if base is None else base
        out.append({"configuration": name, "n_tokens": len(v),
                    "median_ms": f"{np.median(v):.1f}", "mean_ms": f"{v.mean():.1f}",
                    "p90_ms": f"{np.percentile(v, 90):.1f}",
                    "std_ms": f"{v.std(ddof=1):.1f}",
                    "selection_cost_ms": f"{np.median(v) - base:+.1f}"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remeasure", action="store_true", help="re-time on this machine")
    ap.add_argument("--agent-config-dir", type=pathlib.Path)
    ap.add_argument("--gtrs-ckpt", type=pathlib.Path)
    ap.add_argument("--config-name", default="selpay_hybrid_xt4")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    if a.remeasure:
        if a.agent_config_dir is None or a.gtrs_ckpt is None:
            ap.error("--remeasure needs --agent-config-dir and --gtrs-ckpt")
        rows = remeasure(a)
        with TABLE.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {TABLE.relative_to(HERE.parent)}\n")
    else:
        rows = list(csv.DictReader(TABLE.open()))
    show(rows)


if __name__ == "__main__":
    main()
