# teammate_legalizer

Vendored, **unmodified** copies of the legalization module from the teammate's
repo: `ICCAD2026-Problem-C/diffusion-floorplanner`
(`diffusion_floorplanner/{compaction,config,scoring,energy,canvas}.py`),
fetched from commit `d5a6a3669e6c0a525ca5d6b08e22038f7a73cd19` on `main` (see
`.vendor_commit_sha`). `energy.py`/`canvas.py` weren't in the original plan --
`compaction.py` imports `energy.grouping_components` lazily (inside function
bodies, at 3 call sites), so a static top-of-file dependency scan missed it;
found by actually running `legalize_sample()` and hitting `ModuleNotFoundError`
on a real sample with grouping constraints. `energy.py` in turn imports
`canvas._build_padded_groups` at module load time.

Do not hand-edit these files. If a fix or update is needed, re-fetch from the
source repo and re-vendor, so this stays a clean copy of their tested code.

**Entry point**: `compaction.legalize_sample(coords, is_pp, is_fs, area_target,
target_ll, mib=..., grouping=..., boundary=..., mode='hybrid', ...)` — see its
docstring in `compaction.py` for the full contract. Used by `inference_v2.py`
in the parent directory as an alternative to `utils.legalize_lff`.

`clamps.py` (their sampler-side hard-clamp logic) was intentionally NOT
vendored — it's only used during their own model's diffusion sampling, which
we don't use; our own model already has equivalent clamping in `diffusion.py`.

Requires `cvxpy` + the `CLARABEL` solver (already a project dependency, added
in v5.39 — see `requirements.txt`).
