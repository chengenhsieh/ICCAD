# ICCAD 2026 FloorSet — Diffusion Floorplanning Pipeline

Diffusion-model-based VLSI floorplanning solution for the [ICCAD 2026 FloorSet contest](https://github.com/IntelLabs/FloorSet). Generates a raw block layout with a force-guided DDIM diffusion sampler, then legalizes it with a teammate-contributed constrained-optimization legalizer (with a deterministic, hard-constraint-safe fallback for the rare case it can't converge).

## Contents

| File | Purpose |
|---|---|
| `config.py` | Model / training configuration |
| `dataset.py` | Training dataset handling |
| `model.py` | GNN-style diffusion model (`FloorplanDiffusionModel`) |
| `diffusion.py` | Forward/reverse diffusion process; DDIM/EDM samplers with force guidance (pin, grouping, repulsion, boundary) |
| `train.py` | Training loop |
| `inference.py` | Diffusion generation (`generate_floorplan`) + the original `legalize_lff` fallback legalizer; shared by both `my_optimizer.py` and `inference_v2.py` |
| `inference_v2.py` | Adapter that swaps the legalize stage for the teammate's `legalize_sample` (see `teammate_legalizer/`), with an automatic fallback to `inference.py`'s `legalize_lff` if the convex solve can't converge |
| `teammate_legalizer/` | Vendored, unmodified legalizer from a teammate's repo (ICCAD2026-Problem-C/diffusion-floorplanner) — the primary legalization path since v6.0 |
| `my_optimizer.py` | The actual contest submission entry point (`MyOptimizer`) |
| `utils.py` | `legalize_lff` implementation, metrics (HPWL, overlap, soft violations), plotting |
| `checkpoints/model_epoch300_overlap_v4.pt` | Trained model weights (tracked via Git LFS) |
| `CHANGELOG.md` | Full per-version history of what changed, why, and the measured result — the most detailed and up-to-date record of this project's decisions |
| `method.md` | Technical write-up of the training + inference design (Traditional Chinese) |
| `requirements.txt` | Python dependencies (includes `cvxpy`, needed by `legalize_sample`) |

## Setup

This repo holds the custom pipeline code only — it expects to sit inside a checkout of the official [IntelLabs/FloorSet](https://github.com/IntelLabs/FloorSet) repo at `FloorSet/iccad2026contest/`, since the official `iccad2026_evaluate.py` (not included here) and the validation dataset live one directory up (`../`).

1. Clone `IntelLabs/FloorSet`, drop these files into its `iccad2026contest/` folder (replacing the originals except `iccad2026_evaluate.py` / `optimizer_template.py`, which should stay the official versions).
2. Install **[Git LFS](https://git-lfs.com/)** before cloning *this* repo, or run `git lfs pull` afterward — otherwise `checkpoints/model_epoch300_overlap_v4.pt` will just be a small pointer file instead of the real ~137MB weights.
3. Python dependencies: `pip install -r requirements.txt` (torch, numpy, shapely, matplotlib, tqdm, requests, cvxpy).

## Running

```bash
cd iccad2026contest
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
```

Runs the 100-sample validation set through `MyOptimizer` and writes `my_optimizer_results.json` (per-case metrics) plus `my_optimizer_solutions.json` (raw positions).

For quick manual inspection of a few samples with either legalizer directly (bypasses `my_optimizer.py`'s production settings):
```bash
python inference.py          # legalize_lff path
python inference_v2.py       # legalize_sample path
```

## Pipeline

1. **Diffusion generation** (`diffusion.py`, shared by both legalize paths) — force-guided DDIM sampler. Runs a batch of best-of-N candidates together (candidates are ~free on GPU since they share one batch, not run sequentially), applies pin/grouping/repulsion/boundary forces during sampling, then does a short physics-only "post-repel" phase.
2. **Legalization** — since v6.0, the primary path is the teammate's `legalize_sample` (`teammate_legalizer/compaction.py`, vendored unmodified), reached through `inference_v2.py`. It's a constrained convex-optimization legalizer: precedence-graph reconstruction + solve (3 rounds), grouping/boundary penalties baked directly into the objective, followed by a sequence of feasibility-gated finishing passes (snap-to-contact, boundary pull, MIB reshaping, gap repair). If its solve can't converge — this happens on samples where the raw diffusion output has unusually severe overlap — `inference_v2.py` detects the leftover overlap and falls back to this project's own `legalize_lff` (`utils.py`): a deterministic, single-pass, Less-Flexibility-First placement using MAXRECTS free-rectangle bin packing with weighted-median positioning. `legalize_lff` guarantees, by construction, zero overlap and exact preplaced/fixed-shape dimensions no matter how bad the input is — it's what makes the fallback safe, at the cost of generally worse area/HPWL than a successful `legalize_sample` solve.

See `CHANGELOG.md` for the full version history (why `legalize_sample` was adopted, the v6.1 infeasible-solve bug and fix, and everything tried since).

## Current results

**Local validation set (100 samples)** — single official-evaluate run, current production settings:

| Metric | Value |
|---|---|
| Area gap vs. optimal | +4.85% |
| HPWL gap vs. optimal | +8.60% |
| Soft-constraint violation rate (V_relative) | 0.0253 |
| Avg. time / sample | 1.669s |
| Infeasible (hard constraints violated) | 0 / 100 |

**Official final result (hidden test set, 100 samples, 2026-09-02)** — from the organizers' `final_evaluation_results.json`, real `RuntimeFactor` included:

| Metric | Value |
|---|---|
| Area gap vs. optimal | +24.48% |
| HPWL gap vs. optimal | +46.20% |
| Soft-constraint violation rate (V_relative) | 0.1877 |
| Avg. time / sample | 1.319s |
| Infeasible (hard constraints violated) | 0 / 100 (100% feasible) |
| **Total Score (official, real RuntimeFactor)** | **2.0186** |

The hidden-test numbers are noticeably worse than the validation-set snapshot above. Root cause, confirmed by per-sample inspection: 14/100 hidden-test samples hit the `legalize_sample`-infeasible → `legalize_lff`-fallback path described above (vs. only 2/100 on the validation set) — the diffusion model's raw output apparently has severe overlap on a larger fraction of the hidden test distribution than the validation set led us to expect. All 100 samples stayed fully feasible either way; the fallback trades area/HPWL quality for that guarantee. See `method.md`'s "最終結果" section for the full breakdown.
