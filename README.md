# When Does Trajectory Selection Pay?

Reproduction package for *When Does Trajectory Selection Pay? A Measurement Study
of Proposal-Scoring Planners* (IEEE IV).

The paper is a **measurement study**: its claims are paired statistical
comparisons between planner configurations, all scored in one harness on
identical scenarios. This repository is built so that **every one of those
claims can be recomputed in about a minute, on a laptop, with no GPU, no
simulator, and no NAVSIM installation** — because the per-scenario score tables
behind every figure and table ship here directly.

---

## TL;DR

```bash
git clone https://github.com/andyluken/when-selection-pays.git
cd when-selection-pays
pip install -r requirements.txt          # numpy, pandas, scipy
cd analysis
python fig4_gate_law.py                  # the gate-saturation law + its control
python table1_navhard.py                 # Table I, every row paired
python fig7_diversity.py                 # the diversity sweep
python fig5_ceiling.py                   # the ordering ceiling
```

Each script prints the paper's reported value next to the recomputed one.

---

## What is here

```
results/
  MANIFEST.csv              every shipped table: split, run, n, per-scenario mean
  navhard/*.csv             16 runs x 5,915 scenarios, per-scenario + per-sub-metric
  navtest_v2/*.csv           6 runs x 12,147 scenarios (v2 EPDMS, single stage)
  navtest_v1/*.csv           6 runs x 12,146 scenarios (v1 PDMS)
  derived/                  the curves behind Fig. 5, and the Fig. 6 proposal sets
analysis/                   scripts that turn the above into the paper's numbers
harness/                    how the tables were produced (needs NAVSIM; see Level 2)
```

Each `results/*/**.csv` has one row per scenario with the composed `score` and
all nine sub-metrics, split by stage where applicable. That is what makes the
paired tests reproducible without re-simulating anything.

**A note on two different numbers.** NAVHARD's headline EPDMS is a *two-stage
product* of group scores, not a mean over scenarios, so the per-scenario mean in
`MANIFEST.csv` is deliberately **not** the composed EPDMS quoted in the paper.
The paper's statistical claims are all paired per-scenario differences, and
those are what `analysis/` recomputes. On NAVTEST (single stage) the mean *is*
the reported score, and you can read them straight out of the manifest.

---

## Level 1 — verify the claims (no GPU, ~1 minute)

| script | reproduces |
|---|---|
| `fig4_gate_law.py` | Sec. IV-B, Fig. 4. Bins by the base policy's gate product and recomputes +0.1359 / +0.0562 / −0.0310, then the regression-to-the-mean control against an independent seed (+0.1251, −0.0269) and the 7.9% floor decomposition. |
| `table1_navhard.py` | Table I, and the four paired claims the text actually makes. |
| `fig7_diversity.py` | Sec. VI: the σ_T sweep, the plateau at σ_T ∈ {3,4,5}, and the K=64→128 null. |
| `fig5_ceiling.py` | Sec. V: the scorer-separation curve, the orderability sweep, and where our proposals sit on that axis. |

`analysis/common.py` holds the shared pieces: gate-product computation, the
paired test (reporting **both** a t-test and a Wilcoxon, as the paper requires,
since the metric's multiplicative structure routinely makes them disagree), and
loading.

### One pairing worth understanding

`ours_k128` is K=128 **at σ_T = 2**, so it must be paired against `ours_xt2`
(K=64, σ_T = 2), not against `ours_xt4`. Pairing it against σ_T = 4 conflates
proposal *count* with proposal *diversity* and produces a spurious significant
negative. The script enforces the correct pairing and says so in a comment;
count and spread are different knobs and the paper is careful to separate them.

---

## Level 2 — re-run the evaluation (needs NAVSIM, data and a GPU)

This regenerates the score tables themselves rather than trusting the ones here.

1. **Install NAVSIM v2.2** from source (the PyPI package named `navsim` is an
   unrelated project). Its Hydra configs are not in the wheel, so you need the
   clone on `PYTHONPATH`:

   ```bash
   git clone https://github.com/autonomousvision/navsim.git
   export PYTHONPATH=/path/to/navsim:/path/to/when-selection-pays/harness
   ```

2. **Get the data.** NAVHARD's two-stage split (`navhard_two_stage`) and
   OpenScene sensor blobs come from NAVSIM's own download scripts. Note that
   NAVHARD stage-one scenes come from the ordinary *test* logs, so
   `OPENSCENE_DATA_ROOT` must contain both.

3. **Build metric caches, then score:**

   ```bash
   export OPENSCENE_DATA_ROOT=... NAVSIM_EXP_ROOT=... \
          NUPLAN_MAPS_ROOT=... NUPLAN_MAP_VERSION=nuplan-maps-v1.0
   python navsim/planning/script/run_metric_caching.py train_test_split=navhard_two_stage
   python navsim/planning/script/run_pdm_score.py \
       train_test_split=navhard_two_stage agent=selpay_hybrid_xt4 \
       worker=sequential experiment_name=my_run
   ```

Four traps that each silently produce a wrong result or none at all, documented
in `harness/NOTES.md`: Hydra resolving configs against the installed NAVSIM
rather than the clone; `experiment_name` being a mandatory field with no
default; Ray's default worker OOM-ing the GPU (use `worker=sequential`);
and NAVHARD's stage-two synthetic sensors being **camera-only**, which
disqualifies any LiDAR-consuming agent from that split entirely.

---

## Level 3 — retrain

Training the planner and the distilled scorer is described in `harness/NOTES.md`.
The scorer is distilled from GTRS's published per-trajectory simulator labels;
see below.

---

## Checkpoints

Model weights are published as **GitHub Release assets** rather than in git
(the frozen ego-motion backbone alone exceeds GitHub's 100 MB file limit):

| file | size | what |
|---|---|---|
| `dit_planner.pt` | 20 MB | the diffusion planner (5.16 M trainable) |
| `trajectory_scorer.pt` | 5.3 MB | the distilled scorer (1.37 M) |
| `laq_ad_v3.pt` | 119 MB | the frozen self-supervised ego-motion encoder |

```bash
mkdir -p checkpoints && cd checkpoints
# download from the Releases page of this repository
```

---

## What is *not* redistributed, and why

- **GTRS's per-trajectory labels** (~30 GB). These are third-party data from
  [NVlabs/GTRS](https://github.com/NVlabs/GTRS) and are what both our scorer's
  training targets and Fig. 5's "true ranking" come from. We ship the *derived*
  curves in `results/derived/` so the figure and every number quoted from it can
  be checked without them.
- **NAVSIM / OpenScene sensor data.** Licensed by their authors; get it from
  NAVSIM's download scripts.
- **Released baseline checkpoints** (GTRS-Dense, DiffusionDrive, TransFuser,
  LTF). We scored the authors' published weights; download them from their own
  repositories. Their per-scenario scores *in our harness* are in `results/`.

---

## Honest scope

Worth knowing before you build on this:

- Our planner is **not** state of the art. It reaches 0.3983 EPDMS on NAVHARD
  against GTRS-Dense's 0.4377, and on NAVTEST it trails every released baseline
  we ran. The contribution is the characterisation of *when* selection pays.
- Our main configuration uses **privileged inputs** (neighbour observations,
  route corridor, traffic-light state). The sensor-only rows are the
  like-for-like comparison; the privileged rows are an upper bound.
- Traffic-light handling is a **deterministic rule** applied after the learned
  planner, not a learned contribution.

---

## Citation

```bibtex
@inproceedings{lubogo2026selection,
  title     = {When Does Trajectory Selection Pay? A Measurement Study of
               Proposal-Scoring Planners},
  author    = {Lubogo, Andrew and Kee, Seok-Cheol},
  booktitle = {Proc. IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2026}
}
```
