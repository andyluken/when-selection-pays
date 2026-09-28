"""Dump REAL proposal sets at several x_T for one navhard scene.

Produces the data behind the "what diversity looks like" figure: the actual 64
candidates the planner emits, not a schematic. Also reports the measured mean
pairwise final-position separation per x_T, which is what the figure should be
labelled with (the paper's own data says x_T 3/4/5 is a statistical plateau, so
a "sweet spot" label would contradict it).

  python -m epdms_eval.proposal_fan_figure --token <tok> --xts 2 4 5
"""
from __future__ import annotations
import os
import argparse, sys
from pathlib import Path
import numpy as np

NAV = Path(os.environ.get("NAVSIM_ROOT", "./navsim"))  # path to the navsim CLONE
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="selpay_hybrid_xt4")
    ap.add_argument("--proposals", type=int, default=64)
    ap.add_argument("--xts", type=float, nargs="+", default=[2.0, 4.0, 5.0])
    ap.add_argument("--tokens", nargs="+", default=None)
    ap.add_argument("--n-scan", type=int, default=6)
    ap.add_argument("--out", default="epdms_eval/proposal_fans.npz")
    a = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    with initialize_config_dir(
        config_dir=str(NAV / "navsim/planning/script/config/pdm_scoring"), version_base=None
    ):
        cfg = compose(config_name="default_run_pdm_score",
                      overrides=["train_test_split=navhard_two_stage", f"agent={a.agent}",
                                 "worker=sequential", "experiment_name=fanfig"])
    from navsim.common.dataloader import SceneLoader, MetricCacheLoader

    agent = instantiate(cfg.agent); agent.initialize()
    agent._num_proposals = a.proposals
    agent._deterministic_noise = True          # reproducible figure
    mcl = MetricCacheLoader(Path(cfg.metric_cache_path))
    loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=agent.get_sensor_config(),
    )
    avail = [t for t in loader.tokens_stage_one if t in set(mcl.tokens)]
    toks = [t for t in (a.tokens or avail) if t in set(avail)][: a.n_scan]
    print(f"  scanning {len(toks)} token(s); proposals={a.proposals}")

    grab = {}
    if hasattr(agent, "_score_and_select_own"):
        orig = agent._score_and_select_own
        def spy(p):
            grab["p"] = np.asarray(p).copy(); return orig(p)
        agent._score_and_select_own = spy
    else:
        orig = agent._score_and_select
        def spy2(ai, p):
            grab["p"] = np.asarray(p).copy(); return orig(ai, p)
        agent._score_and_select = spy2

    out = {}
    for tok in toks:
        ai = loader.get_agent_input_from_token(tok)
        sc = loader.get_scene_from_token(tok)
        for xt in a.xts:
            agent._xt_scale = float(xt)
            grab.clear()
            try:
                sel = agent.compute_trajectory(ai, sc)
            except Exception as e:
                print(f"    {tok[:10]} xt={xt}: FAILED {type(e).__name__}: {e}"); continue
            if "p" not in grab:
                print(f"    {tok[:10]} xt={xt}: selector not instrumented"); continue
            P = grab["p"]
            fin = P[:, -1, :2]
            d = np.linalg.norm(fin[:, None] - fin[None], axis=-1)
            iu = np.triu_indices(len(fin), 1)
            out[f"{tok}|{xt}"] = P
            out[f"{tok}|{xt}|sel"] = np.asarray(sel.poses)
            print(f"    {tok[:10]} xt={xt:>3}: P{P.shape}  mean pairwise sep "
                  f"{d[iu].mean():6.3f} m   max {d[iu].max():6.3f} m")
    np.savez_compressed(a.out, **out)
    print(f"  wrote {a.out}  ({len(out)} arrays)")


if __name__ == "__main__":
    main()
