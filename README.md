# ICCAD 2026 FloorSet — Diffusion Floorplanning Pipeline

Diffusion-model-based VLSI floorplanning solution for the [ICCAD 2026 FloorSet contest](https://github.com/IntelLabs/FloorSet). Generates a raw block layout with a force-guided DDIM diffusion sampler, then legalizes it with a deterministic, hard-constraint-safe packing algorithm.

## Contents

| File | Purpose |
|---|---|
| `config.py` | Model / training configuration |
| `dataset.py` | Training dataset handling |
| `model.py` | GNN-style diffusion model (`FloorplanDiffusionModel`) |
| `diffusion.py` | Forward/reverse diffusion process; DDIM/EDM samplers with force guidance (pin, grouping, repulsion, boundary) |
| `train.py` | Training loop |
| `inference.py` | End-to-end inference: diffusion generation → legalization → evaluation → JSON export |
| `utils.py` | Legalization algorithm, metrics (HPWL, overlap, soft violations), plotting |
| `checkpoints/model_epoch300_overlap_v4.pt` | Trained model weights (tracked via Git LFS) |

## Setup

This repo holds the custom pipeline code only — it expects to sit inside a checkout of the official [IntelLabs/FloorSet](https://github.com/IntelLabs/FloorSet) repo at `FloorSet/iccad2026contest/`, since `inference.py` imports `litetestLoader.py` and reads the validation dataset from one directory up (`../`).

1. Clone `IntelLabs/FloorSet`, drop these files into its `iccad2026contest/` folder (replacing the originals).
2. Install **[Git LFS](https://git-lfs.com/)** before cloning *this* repo, or run `git lfs pull` afterward — otherwise `checkpoints/model_epoch300_overlap_v4.pt` will just be a small pointer file instead of the real 131MB weights.
3. Python dependencies: `torch`, `numpy`, `matplotlib`.

## Running

```bash
cd iccad2026contest
python inference.py
```

Runs the 100-sample validation set and writes raw/legalized solution JSONs plus a timing summary to `../json/`.

## Pipeline

1. **Diffusion generation** — force-guided DDIM sampler (`diffusion.py`). Runs a batch of best-of-N candidates together (candidates are ~free on GPU since they share one batch, not run sequentially), applies pin/grouping/repulsion/boundary forces during sampling, then does a short physics-only "post-repel" phase.
2. **Legalization** (`utils.py: legalize_lff`) — deterministic, single-pass, Less-Flexibility-First-style placement using MAXRECTS free-rectangle bin packing with weighted-median (L1-optimal) positioning per block. Guarantees, by construction:
   - zero overlap between blocks
   - preplaced blocks keep their exact position and shape
   - fixed-shape blocks keep their exact shape
   - all blocks stay within 1% of their target area
   - all blocks stay within the pin-derived bounding box
   Followed by compaction passes (`compact_merge_clusters`, `compact_reinsert`, `compact_positions`) that reduce wasted space without ever being able to reintroduce a hard-constraint violation.

## Current results (100-sample validation set)

| Metric | Value |
|---|---|
| Area gap vs. optimal | ~24.8% |
| HPWL gap vs. optimal | ~15.9% |
| Soft-constraint violation rate (V_relative) | ~0.110 |
| Avg. time / sample | ~2.7s (diffusion ~1.4s + legalize ~1.2s) |
| Infeasible (hard constraints violated) | 0 / 100 |
