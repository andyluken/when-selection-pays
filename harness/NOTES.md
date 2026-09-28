# Harness notes

How the score tables in `results/` were produced, and the traps worth knowing.

## Why a custom harness at all

NAVSIM's own `run_pdm_score.py` is used directly for NAVHARD's two-stage EPDMS —
we do not reimplement the scorer. For the single-stage NAVTEST protocols we call
NAVSIM's scoring primitives (`PDMSimulator`, `PDMScorer`, `pdm_score`,
`MetricCacheProcessor`) from our own driver, because the pip wheel omits the
bundled Hydra YAML configs and its CLI entry points therefore do not run as
installed. The constructor defaults we rely on are the official values.

## Traps

1. **Hydra resolves against the *installed* NAVSIM** unless the clone is first on
   `PYTHONPATH`. Symptom: a confusing `Could not load 'default_dataset_paths'`.
2. **`experiment_name` is a mandatory `???` field** with no default.
3. **Ray's default worker OOMs the GPU** — roughly 24 workers each load the model.
   Use `worker=sequential`; GPU memory is the binding constraint, not CPU.
4. **NAVHARD stage-two synthetic sensors are camera-only.** There are zero
   `MergedPointCloud` directories in that split, so any LiDAR-consuming agent
   fails 100% of stage two. This is a property of the benchmark, not of an
   implementation.
5. **`OPENSCENE_DATA_ROOT` must be assembled.** NAVHARD's stage-*one* scenes come
   from the ordinary test logs, not from the NAVHARD download.
6. **Scenario "tokens" are scoped to `num_history_frames`.** The same log position
   yields a *different* token identity under a different history requirement, so
   two configurations with different history lengths do not share a token set.

## Reproducing the seed control (Sec. IV-B)

The regression-to-the-mean control needs a second, independent baseline draw.
The agent takes a `noise_seed` argument (default `0`, which reproduces every
earlier run bit-for-bit) and includes it in its name so the two runs cannot
overwrite each other's score table:

```yaml
# agent config, identical to the baseline except for the seed
deterministic_noise: true
noise_seed: 1
```

```bash
python navsim/planning/script/run_pdm_score.py \
    train_test_split=navhard_two_stage agent=selpay_noselection_seedB \
    worker=sequential experiment_name=noselection_seedB
```

Then `analysis/fig4_gate_law.py` bins on the seed-0 run and compares the
selected run against the seed-1 draw, which plays no part in defining the bins.
