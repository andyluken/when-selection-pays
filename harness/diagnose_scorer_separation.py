"""Is our distilled scorer limited by CONDITIONING, or by a train/inference
candidate-distribution mismatch?

The scorer trains on `rng.integers(0, V)` -- uniformly random vocabulary
trajectories, mean pairwise final-position separation 16.7 m. It is deployed
ranking 64 continuous perturbations of ONE diffusion mode, mean separation
5.8 m. GTRS deliberately engineered for this shift (train V_XL, infer V_L,
vocabulary dropout); we did not.

Two hypotheses make DIFFERENT predictions:
  conditioning-limited -> ranking quality is flat in candidate separation
  mismatch-limited     -> ranking quality DEGRADES as candidates get closer

Needs no simulation: GTRS's published labels already score all 16,384
vocabulary trajectories per token, so the true ranking is known for any subset.

Usage (from the harness directory, with NAVSIM on PYTHONPATH):
    python diagnose_scorer_separation.py --tokens 400
"""
from __future__ import annotations
import os
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from nuplan_ad_model.trajectory_scorer import TrajectoryScorer, combine_scores, SUB_METRICS
from nuplan_ad_model.scene_context import SceneContextEncoder
from nuplan_ad_model.semantic_features import SEMANTIC_DIM, SEMANTIC_TOKENS
from nuplan_ad_model.neighbor_agents import NEIGHBOR_PAST_DIM
from nuplan_ad_model.dit_normalization import normalize_neighbor_past
from train_dit_planner import DiTPlannerData


def true_score(lab, rows, idx):
    """EPDMS-shaped truth from the simulator labels: gates multiply, rest weighted."""
    g = lambda m: np.stack([lab[m][r][i] for r, i in zip(rows, idx)])
    gates = (g("no_at_fault_collisions") * g("drivable_area_compliance")
             * g("driving_direction_compliance") * g("traffic_light_compliance"))
    weighted = (5 * g("time_to_collision_within_bound") + 5 * g("ego_progress")
                + 2 * g("lane_keeping")) / 12.0
    return gates * weighted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="k1")
    p.add_argument("--labels", type=Path, default=Path("gtrs_labels/gtrs_labels_k1.npz"))
    p.add_argument("--vocab", type=Path, default=Path(os.environ.get("GTRS_VOCAB", "./gtrs/traj_final/16384.npy")))
    p.add_argument("--scorer", type=Path, default=Path("results_trajectory_scorer/trajectory_scorer.pt"))
    p.add_argument("--planner-checkpoint", type=Path,
                   default=Path("results_dit_planner_k1_margin0.5/dit_planner.pt"))
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--radii", type=float, nargs="+",
                   default=[3.0, 6.0, 12.0, 25.0, 1e9])
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lab = np.load(a.labels, allow_pickle=True)
    lab_tokens = [str(t) for t in lab["tokens"]]
    lab_row = {t: i for i, t in enumerate(lab_tokens)}
    lab_arrays = {m: lab[m] for m in SUB_METRICS}

    vocab = torch.from_numpy(np.load(a.vocab)).float()
    V, H, _ = vocab.shape
    vocab4 = torch.cat([vocab[..., :2], torch.cos(vocab[..., 2:3]),
                        torch.sin(vocab[..., 2:3])], -1)
    tree = cKDTree(vocab[:, -1, :2].numpy())

    ck = torch.load(a.planner_checkpoint, map_location="cpu", weights_only=False)
    ta = ck["args"]
    data = DiTPlannerData(Path("dit_training_data"), a.tag,
                          semantic_model=ta.get("semantic_model", "siglip"))
    tok_index = {s["token"]: i for i, s in enumerate(data.samples)}

    enc = SceneContextEncoder(
        d_model=int(ta["d_model"]), k_vis=int(ta.get("k_vis", 1)), feat_dim=1024,
        corridor_n_points=20, max_neighbors=int(ta["max_neighbors"]),
        neighbor_past_dim=NEIGHBOR_PAST_DIM,
        semantic_dim=SEMANTIC_DIM if ta.get("use_semantic") else None,
        semantic_tokens=SEMANTIC_TOKENS if ta.get("use_semantic") else 0)
    enc.load_state_dict(ck["scene_context"]); enc.to(dev).eval()

    sc = torch.load(a.scorer, map_location="cpu", weights_only=False)
    scorer = TrajectoryScorer(d_model=sc["d_model"], ctx_dim=sc["d_model"],
                              horizon=sc["horizon"])
    scorer.load_state_dict(sc["scorer"]); scorer.to(dev).eval()

    # replicate the trainer's own split exactly -- never diagnose on train data
    usable = [t for t in lab_tokens if t in tok_index]
    val_toks = usable[int(0.9 * len(usable)):][:a.tokens]
    print(f"  scorer {a.scorer}")
    print(f"  val tokens {len(val_toks)}  K={a.k}\n")
    print(f"  {'radius':>9} {'sep(m)':>8} {'capture':>9} {'spearman':>9} "
          f"{'true spread':>12} {'n':>5}")

    rng = np.random.default_rng(0)
    for r in a.radii:
        caps, rhos, seps, spreads = [], [], [], []
        for t in val_toks:
            anchor = int(rng.integers(0, V))
            if r < 1e8:
                pool = tree.query_ball_point(vocab[anchor, -1, :2].numpy(), r)
                if len(pool) < a.k:
                    continue
                idx = rng.choice(np.asarray(pool), a.k, replace=False)
            else:
                idx = rng.integers(0, V, a.k)
            tv = true_score(lab_arrays, [lab_row[t]] * a.k, idx)
            if tv.max() - tv.mean() < 1e-6:
                continue                      # degenerate: nothing to choose
            i = tok_index[t]
            with torch.no_grad():
                sem = (data.semantic[[i]].float().to(dev)
                       if (ta.get("use_semantic") and data.semantic is not None) else None)
                ctx, _ = enc(data.feats[[i]].float().to(dev),
                             data.corridor[[i]].float().to(dev),
                             normalize_neighbor_past(data.neighbor_past[[i]]).to(dev),
                             data.neighbor_mask[[i]].to(dev), semantic=sem)
                lg = scorer(ctx, vocab4[torch.from_numpy(idx)][None].to(dev))
                pred = combine_scores(lg)[0].cpu().numpy()
            caps.append((tv[int(pred.argmax())] - tv.mean()) / (tv.max() - tv.mean()))
            rho = spearmanr(pred, tv).correlation
            if np.isfinite(rho):
                rhos.append(rho)
            fp = vocab[idx, -1, :2].numpy()
            seps.append(np.linalg.norm(fp[:, None] - fp[None], axis=-1).mean())
            spreads.append(tv.max() - tv.mean())
        lbl = "unrestricted" if r > 1e8 else f"{r:.0f} m"
        print(f"  {lbl:>9} {np.mean(seps):8.2f} {np.mean(caps):9.1%} "
              f"{np.mean(rhos):9.3f} {np.mean(spreads):12.4f} {len(caps):5d}")


if __name__ == "__main__":
    main()
