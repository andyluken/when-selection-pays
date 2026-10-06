# Pre-registration: cross-family test of the gate-saturation effect
Written 2026-10-06, BEFORE running any selection on a second planner.

## Claim under test
Selection's value is monotone in how far the base policy's multiplicative gate
product G sits from saturation. Measured WITHIN our DiT planner (navhard, n=5915,
stage-two gates): gain +0.1359 at G=0, +0.0562 at 0<G<1, -0.0310 at G=1.

## Baselines measured (navhard stage two, n=5464, from each planner's own run)
| planner | family | G | G=0 | 0<G<1 | G=1 |
|---|---|---|---|---|---|
| LTF            | single-shot regression | 0.5546 | 41.6% | 5.9% | 52.5% |
| ours (DiT)     | joint diffusion        | 0.5992 | 37.5% | 5.2% | 57.3% |
| GTRS-Dense     | fixed vocabulary       | 0.6894 | 28.2% | 5.7% | 66.1% |

## Prediction (our bin gains x GTRS's own bin occupancy)
Selection applied to GTRS-Dense proposals should yield **+0.021 EPDMS**,
i.e. ~58% of what the same machinery yields on our planner (+0.036 predicted,
+0.031 measured paired).

## Falsification
- A GTRS gain at or above our planner's (>= +0.036) REFUTES the effect as a
  cross-family predictor.
- A gain near zero or negative would mean the effect over-predicts for
  high-saturation planners -- reportable, but a weaker form of the claim.
- Bands: confirm if +0.010 to +0.032; refute if >= +0.036 or <= 0.

## Protocol fixed in advance
- navhard two-stage, all 5915 scenarios, official run_pdm_score.py.
- Paired against the already-completed GTRS-Dense baseline on identical tokens.
- Report BOTH paired t-test and Wilcoxon (they disagree elsewhere in this work).
- Our distilled scorer, unchanged. No retuning for GTRS.

---

## Amendment, 2026-10-06, after smoke test, BEFORE the scoring run

A covariate was measured during implementation and is recorded here before any
EPDMS number exists for this configuration.

**GTRS-Dense's top-64 proposals are far tighter than ours.** Mean pairwise
final-position separation is **1.61 m** (6 navtest tokens) against our
diffusion's **~5.9 m** at xT=4.0 -- a 3.7x difference. This is intrinsic to the
family: a top-K drawn by argmax from a 16,384-entry vocabulary returns
near-neighbours, whereas raising the diffusion prior's noise scale widens the
set by construction.

**Why this matters.** Our own ordering-ceiling result says selection's headroom
falls with candidate separation independently of gate saturation (63.5% of
64-candidate sets exactly tied at ~0.9 m, 26% at 6 m). So two distinct
mechanisms now push the GTRS gain down, and the headline prediction (+0.021)
isolates only the first.

**Consequence for interpretation, fixed in advance:**
- Gain in [+0.010, +0.032] -> CONFIRMS; the separation confound did not bite.
- Gain >= +0.036 -> REFUTES the effect as a cross-family predictor.
- Gain in (0, +0.010) -> AMBIGUOUS. Directionally consistent, but we will NOT
  claim confirmation: the shortfall is attributable to gate saturation or to the
  tighter candidate set and this run cannot separate them. Report as such.
- Gain <= 0 -> the effect over-predicts for high-saturation planners; report.

**Not run, and why.** Widening GTRS's set (e.g. sampling spread-out entries
rather than its own top-K) would decouple the two mechanisms, but the proposals
would no longer be the ones GTRS would actually emit, so it answers a different
question. It is the natural follow-up, not a substitute.

## Protocol actually executed
- agent config `gtrs_props_ourscorer.yaml`; K=64; GTRS infers on its full 16,384
  vocabulary; our k1 scene context + 1.37M distilled scorer; argmax.
- stop-line layer OFF, smoothing lambda=0 -> isolates selection as the ONLY
  difference from the completed GTRS-Dense baseline (`epdms_gtrs_gpu`).
- Emission grid is GTRS's native 40 poses @ 0.1 s, matching that baseline, so no
  resampling artifact is charged to selection.
