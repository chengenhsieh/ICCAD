"""Centralized configuration.

All hyperparameters and constants used across the package live here. Importing
this module is side-effect-free except for reading torch (for `DEVICE`); no
files are created here.

The values mirror the v0.15 notebook one-to-one, MINUS the CFG hyperparameters
which were removed in v0.15. The v0.16 series adds:
- SPECTRAL_INIT_INTERIOR_SCALE : post-QP inward nudge to reduce initial border pileup

v0.17 adds:
- OVERLAP_LINEAR_MIX   : blend quartic loss with bilinear for non-vanishing small-overlap gradient
- USE_FLIP_AUG         : random x/y/180-degree flip augmentation during training

v1.5.4 removes the last DDPM-era knobs (TIMESTEPS, NOISE_PARAM, CFG_DROP_PROB,
OVERLAP_SNR_POWER, OVERLAP_T_THRESHOLD). EDM has been the only sampler since the v1.5.4
inference cleanup and is now the only training parameterization too, so nothing read them.
NOTE: dropping CFG_DROP_PROB is the one change here that alters TRAINING -- 10% of steps
used to null out the conditioning to learn a CFG prior that inference no longer uses. The
deployable model (release/v1.3/v1.3_model.pth) was trained WITH it and is unaffected; a
retrain from this commit will differ from the v1.3 recipe by exactly this.
"""

import os

import torch


# --------------- Floorplan constants ---------------
FEATURE_DIM  = 3
POS_DIM      = 2

# --------------- Edge attribute layout -------------
# [weight, is_netlist, is_spatial_centroid, is_spatial_edge,
#  is_pin_edge, gap_x, gap_y, touches_preplaced]
# Slot 7 (touches_preplaced) was added so the GNN can learn obstacle-aware
# message-passing at preplaced boundaries without having to infer that from
# motion patterns alone.
EDGE_ATTR_DIM = 8

# --------------- Dataset selection -----------------
# Full block-count range (FloorSet-Lite spans 21..120). To cover ALL counts without
# the full 907k set, train_loader samples a fixed PER-BLOCK count (even coverage)
# instead of a random cap over the combined pool (which over-represents the larger,
# more-populated block counts). TRAIN_PER_BLOCK indices are drawn from each count.
# Env-overridable for tail fine-tuning: TRAIN_BLOCK_LO/HI narrow the count range,
# TRAIN_PER_BLOCK sets density. Defaults = full range 21-120 @ 2700/count (so EVAL,
# which reads TRAIN_BLOCK_COUNTS, still spans the whole official set unless overridden).
_TB_LO = int(os.environ.get('TRAIN_BLOCK_LO', 21))
_TB_HI = int(os.environ.get('TRAIN_BLOCK_HI', 120))
TRAIN_BLOCK_COUNTS = list(range(_TB_LO, _TB_HI + 1))
TRAIN_PER_BLOCK    = int(os.environ.get('TRAIN_PER_BLOCK', 2700))   # 3x density (270k total). The 900/block run (~90k, = old
# single-range total) was WORSE than the 30-40 specialist EVERYWHERE: per-block density
# fell ~10x (8940->900) and that undertraining beat coverage. 2700/block (~3.3x below the
# specialist's density) tests whether more density closes the gap. Enabled by the v0.20
# precompute (PrebuiltDataset): ~2h vs ~13h on the old per-epoch-rebuild pipeline.

# --------------- Noise parameterization: EDM (the only one) ----------------
# Karras et al. (2022) "Elucidating...": VE noising x_sigma = x0 + sigma*eps, input/output
# preconditioning (c_skip/c_out/c_in/c_noise), sigma-dependent loss weighting lambda(sigma),
# and a lognormal training-sigma distribution centered where the net learns most.
# sigma_data was calibrated on a full-range scan (tools/archived_experiments/edm_calibrate_sigma.py):
#   pooled RMS over movable [cx,cy,log_ar] = 0.566 (~= Karras' image default 0.5, so
#   the standard defaults transfer; sigma_min/max scaled by sigma_data/0.5 = 1.13).
# (v1.5.4: NOISE_PARAM/TIMESTEPS are gone -- the 'ddpm' arm they selected no longer exists.)
EDM_SIGMA_DATA = float(os.environ.get('EDM_SIGMA_DATA', 0.566))
EDM_SIGMA_MIN  = float(os.environ.get('EDM_SIGMA_MIN', 0.0023))
EDM_SIGMA_MAX  = float(os.environ.get('EDM_SIGMA_MAX', 90.0))
EDM_P_MEAN     = float(os.environ.get('EDM_P_MEAN', -1.2))      # ln-sigma ~ N(P_mean, P_std^2)
EDM_P_STD      = float(os.environ.get('EDM_P_STD', 1.2))
EDM_RHO        = float(os.environ.get('EDM_RHO', 7.0))          # Heun schedule curvature
# EDM_STEPS: Heun sampler steps (NFE ~ 2*steps). 100 ADOPTED (Phase-2 stack sweep,
# 2026-06-24): clean-stack overlap 561(@32)->254(@64)->169.6(@100) on the 270k model,
# beating v1.2's 254.6 by -33% AT LOWER runtime (2.95 vs ~3.7 s/case). Trajectory
# resolution is the lever (the v1.2 DDIM_STEPS lesson, EDM analogue).
EDM_STEPS      = int(os.environ.get('EDM_STEPS', 100))
# --- EDM inference-stack knobs (v1.3 deployable, TUNED via CRN coordinate descent on the
# 270k model; clean stack 169.6 @STEPS100 -> 126.4 official 100 r3) ---
# REPEL: S_CHURN 1.5 = best-of-N trajectory diversity, the DDIM_ETA analogue (0->169.6,
# 1.0->140.8, 1.5->~117-126 plateau).
# 2026-08-06 (parameter sweep): 2.0 -> 1.0. The 2.0 "knee" was fitted to RAW SAMPLER overlap in
# v1.3, before M9 re-derived the packing; the full-surface ablation then sampled only {0, 2.0} and
# concluded "off is better". Both readings missed that the end-to-end curve is single-peaked with
# its optimum INSIDE the interval, and that 4.0 is a cliff (+24.68%, V_grp +142).
# ALONE this is worth nothing -- -0.27% real at 2/3, retracted as a standalone. It earns its place
# only TOGETHER with LAM_BND_ADD, where the pair is -3.85% neutral / -3.46% real on 3/3 seeds
# against -2.95%/-2.34% for LAM_BND_ADD by itself. Do not "simplify" this back to 2.0 without
# re-gating the PAIR: the interaction, not the value, is what was measured.
# See docs/experiments/PARAMETER_SWEEP.md (Tier 3).
EDM_REPEL_STRENGTH   = float(os.environ.get('EDM_REPEL_STRENGTH', 1.0))    # in-traj repulsion
EDM_GATE_SIGMA_FRAC  = float(os.environ.get('EDM_GATE_SIGMA_FRAC', 1.0))   # ops on when sigma < frac*sigma_data
EDM_RELEASE_FRAC     = float(os.environ.get('EDM_RELEASE_FRAC', 0.15))     # release pin/group below frac*sigma_data
EDM_REFINE_SIGMA_FRAC = float(os.environ.get('EDM_REFINE_SIGMA_FRAC', 1.0))  # renoise to frac*sigma_data
# 2026-08-06 (parameter sweep): 1.5 -> 3.0. Set in v1.1 for a pipeline with no M9 legalizer and
# never re-tuned since. gamma = min(EDM_S_CHURN / steps, sqrt(2)-1), so at the shipped 16 steps
# this is gamma 0.094 -> 0.188. The phase-4 factorial established that gamma, NOT step count, is
# what carries the quality of `--edm-steps 16` (gamma alone -0.58% at 3/3; steps alone -0.19% at
# 2/3, inside the control spread).
# FOUR independent campaigns, every one of them 3/3 or 2/2 and all in the -0.57..-0.81 band:
#   phase-4 factorial  -0.58% neutral 3/3 / -0.74% real 3/3
#   Tier 2 Round A     -0.81% neutral 2/2 / -0.97% real 2/2
#   Tier 3b            -0.57% neutral 3/3 (on top of LAM_BND_ADD + repel), s/case -2.2%
# Any single one of those sits at or just under its campaign's control spread; the case rests on
# the repetition, not on any one number. Runtime is neutral-to-better, so the real score cannot
# take it back. Do NOT stack it with EDM_RELEASE_FRAC/EDM_GATE_SIGMA_FRAC: all three together
# measured +0.24% at 1/3, WORSE than either member -- they are substitutes.
EDM_S_CHURN          = float(os.environ.get('EDM_S_CHURN', 3.0))           # Heun stochasticity (best-of-N diversity)
# Runtime: bf16 autocast around the inference model forward (eval-only; training stays fp32).
# The EDM path is already CFG-free (no 2x forward like the DDPM stack). ADOPTED default-on
# (A/B: -5.6% sec/case, quality-neutral within noise).
USE_AMP_INFER        = (os.environ.get('AMP_INFER', '1') == '1')
# Runtime cleanup: the model already knows the graph count from the per-graph
# noise tensor. Passing that count into global pooling avoids batch_idx.max().item(),
# an otherwise forced CUDA->CPU synchronization in every GlobalContextModule.
USE_STATIC_GLOBAL_BATCH = (os.environ.get('STATIC_GLOBAL_BATCH', '1') == '1')
# --- Tier-1 WIN: multi-level RESTART sampling (test/edm-restart, 2026-07-01) ---
# v1.3 re-noise refinement is a SINGLE-level restart: renoise to 1.0*sigma_data,
# REFINE_ROUNDS times (saturates ~95 overlap by round 3). Restart (Xu et al. 2023)
# alternates renoise-up + ODE-down at a DESCENDING ladder of sigma levels: each lower
# level renoises a smaller perturbation and re-denoises a shorter low-sigma tail, giving
# the in-traj repulsion more passes to settle overlaps WITHOUT disturbing coarse
# structure. When EDM_RESTART=1, replace the single-level refine with EDM_RESTART_LEVELS
# (fractions of sigma_data, descending), EDM_RESTART_ITERS restarts per level; all
# candidates are pooled (monotone). =0 => exact v1.3.
#   OFFICIAL-100 r2 (vs v1.3 baseline 118.7): 4-level 31.4 (-73%), 5-level 16.6 (-86%),
#   7-level 13.4 (-89%). Attribution is CANDIDATE-COUNT-MATCHED: at 3 states the descending
#   ladder is 63.7 vs single-level refine's 95.3 (-33%) => the win is the SCHEDULE, not more
#   best-of-N draws. HPWL/BBox stay ~flat (+2%), BndViol improves (0.059->0.054), GrpViol
#   rises mildly (2.78->3.90). 5-level is the quality/runtime knee (-86% at +40% sec/case;
#   5->7 buys only -2.6 overlap for +30% runtime). Default schedule = the tuned 5-level.
#   v1.4: EDM_RESTART default ON (the deployable, overlap 16.6); EDM_RESTART=0 = v1.3 sampler.
EDM_RESTART        = (os.environ.get('EDM_RESTART', '1') == '1')
EDM_RESTART_LEVELS = os.environ.get('EDM_RESTART_LEVELS', '1.0,0.5,0.25,0.125,0.0625')  # x sigma_data, descending (tuned)
EDM_RESTART_ITERS  = int(os.environ.get('EDM_RESTART_ITERS', 1))       # restarts per level

# Submission runtime profile. Keep EDM_STEPS=100 as the clean model/evaluation default so
# historical sampler studies remain comparable. infer.py uses the adaptive profile: most
# cases legalize after one cheap first draw; only failed cases escalate to 64 then 100.
#
# 32 -> 16 on 2026-08-04. This IMPROVES QUALITY as well as runtime, which is not what a step
# reduction normally does: -1.30% neutral (the R=1 score, so pure quality) and -4.03% real on
# 3/3 seeds, reproduced across THREE independent campaigns (-1.34%/-4.14%, -1.30%/-4.03%,
# -1.23% neutral), 100/100 feasible every time, at -0.42 s/case. V_bnd 87.7 -> 76.0,
# weighted V_rel 0.0455 -> 0.0410, hpwl ratio 1.1398 -> 1.1259; only area_ratio worsens
# (1.0348 -> 1.0368).
#
# WHY fewer steps is better, measured as a 2x2 factorial on {steps} x {gamma}: the sampler sets
#     gamma = min(EDM_S_CHURN / num_steps, sqrt(2)-1)
# so the step count and the per-step stochasticity are INVERSELY COUPLED -- halving the steps
# doubles the churn. The two effects are separable and almost exactly additive: halving the
# steps is worth -0.54%, doubling gamma is worth -0.69%, and their sum (-1.23%) equals the
# measured combination (-1.23%). So this constant is doing two things at once, and roughly
# 56% of its benefit is the diversity increase rather than the step reduction.
#
# Consequence for anyone tuning this: EDM_S_CHURN has NOT been re-tuned since v1.1, when it was
# set for a pipeline with no M9 legalizer. EDM_S_CHURN=3.0 at 32 steps captures the diversity
# half on its own (-0.60% neutral) but has only ONE campaign behind it and NO validated real
# score -- do not ship it without a real-runtime gate. Note also that raising S_CHURN while
# steps are at 16 OVERSHOOTS: gamma 0.1875 measured -1.05%, worse than this setting's -1.23%.
# See docs/experiments/ABLATION_FULL_SURFACE.md.
SUBMISSION_EDM_STEPS = int(os.environ.get('SUBMISSION_EDM_STEPS', 16))
# Unchanged at 64,100: every measurement of the 16-step first draw kept this escalation, and
# the two have never been varied together.
SUBMISSION_RETRY_EDM_STEPS = os.environ.get('SUBMISSION_RETRY_EDM_STEPS', '64,100')

# Exact grouping abutment must not enlarge the solution outline. The zero-growth cap removed
# every snap-induced bbox expansion on the paired official-100 screen and improved corrected-v10
# size-weighted neutral cost by 2.10% while preserving 100/100 hard feasibility.
SNAP_MAX_AREA_GROWTH = float(os.environ.get('SNAP_MAX_AREA_GROWTH', '0.0'))
# NOT a stale area guard -- a BOUNDARY guard, re-measured under v1.5.11's tighter packing.
# Loosening it to 0.25% or 1% is a literal no-op (the declines it makes want a median +10.6%
# growth); REMOVING it loses 0.5-1.2% on three paired pools and the damage lands in V_boundary,
# because a grown bbox moves the walls out from under the blocks that were flush on them.
# See docs/experiments/VIOLATION_TRACKS.md (G1).

# MIB shape equality (v1.5.12). The M9 solve gives every member of an MIB group ONE (w,h)
# variable pair and delivers V_mib = 0; the post-solve gap step then splits it, because it
# shrinks free blocks 0.1% and leaves immutable ones exact. SNAP_MIB makes the gap uniform per
# GROUP and adds the gated repair for the fallback path. Paired replay of five independent
# pools: -1.57 / -2.13 / -1.63 / -2.23 / -3.02%, V_mib -78..-92%, feasibility unchanged
# case-for-case, 0-1 case worse per pool; V_mib = 0 end-to-end on three fresh draws.
SNAP_MIB = os.environ.get('SNAP_MIB', '1') != '0'

# PAID RE-RANKING BUDGET (v1.5.12). K candidates are legalized per draw and the best feasible
# one is kept under the GT-free proxy. K=6 over K=3, three fresh seeds each, 600/600 feasible:
# real-runtime score 1.0427 -> 1.0151 (-2.65%), neutral 1.4803 -> 1.4499, and runtime does NOT
# rise (1.16 -> 1.12 s/case) because the solves are independent and LEGALIZE_WORKERS absorbs
# them. The score gap sits inside the seed spread; what does separate cleanly is the term the
# selector optimises -- V_boundary 217-227 (K=3) vs 185-192 (K=6), V_grouping 384-403 vs
# 347-357, no overlap on either.
# ASSUMES >= 6 SPARE CORES. With fewer, the extra solves serialize into the contest's
# max(0.7, R^0.3) runtime term and the trade reverses. See docs/experiments/VIOLATION_TRACKS.md (Track S).
# TARGETED GAP REPAIR (v1.5.13). `aware_gaps` holds grouped/boundary-flagged contacts at exact
# touching; when one of them rounds past the scorer's 1e-6 overlap threshold the shipped code
# discarded the whole layout and re-gapped EVERY free block. Measured over 192 solved cases that
# fires on 29% of them, and the offence is a MEDIAN OF ONE overlapping pair out of ~2775 (max 3,
# always overlap, never the area tolerance) -- while the wholesale re-gap costs V_grouping 6.74
# per case against 4.36, and V_boundary 3.09 against 2.33. Gapping only the offenders instead:
# -3.31% mean over five paired pools (all five clear the -2% gate), V_grp -10.3%, V_bnd -2.8%,
# feasibility unchanged, and runtime FLAT because it replaces a wholesale re-gap with a cheaper
# targeted one. See compaction.targeted_gaps and docs/experiments/VIOLATION_PROPOSALS.md.
GAP_REPAIR = os.environ.get('GAP_REPAIR', '1') != '0'

# WEIGHTED SELECTION BUDGET (v1.5.12, P2). The official aggregate weights each case by
# exp(n/12), so the ten largest cases carry 56.6% of the whole score and the single n=120 case
# carries 8.0% -- and those cases are also the WORST (mean V_rel 0.153 against 0.115 for the
# rest). A uniform K spends the same budget on an n=21 case with weight 1e-5. TOPK_WEIGHTED
# scales K by exp((n-60)/24), clipped to [TOPK_MIN, TOPK_MAX], so an n=21 case gets K=3 and an
# n>=100 case gets K=24. Three seeds against uniform K=6: -8.91% neutral, -7.67% REAL, the best
# arm measured; uniform K=12 is -6.46%/-6.45%. Runtime stays under the contest's floor
# (max(0.7, R^0.3) is flat for R <= 0.316; R median 0.297 here).
# ASSUMES >= 12 SPARE CORES. See docs/experiments/VIOLATION_RESULTS.md.
SELECT_TOPK = int(os.environ.get('SELECT_TOPK', 6))
TOPK_WEIGHTED = os.environ.get('TOPK_WEIGHTED', '1') != '0'
TOPK_MIN = int(os.environ.get('TOPK_MIN', 3))
# 24 -> 12 on 2026-08-06 (USER DECISION). This is the one release change that is a BET, not a
# measurement: it deliberately SELLS quality to buy runtime.
#   quality cost   +1.00% neutral at 0/3 on the current stack (Tier 3b), matching the paired
#                  prediction of +1.15% for the 101+ band that carries 81% of the official weight
#   runtime gain   -22.0% s/case ratio on the current stack (-12.5% when it was first gated)
#   real score     -0.85% and -1.95% on two independent 3-seed gates, 6/6 seeds, no crossover
# It pays only if the contest field's median runtime is at or below our stand-in: R_i enters as
# max(0.7, R_i^0.3), so a saving is worth nothing on cases already under the floor and a lot on
# cases above it. The user retained it as opt-in on 2026-08-03 for exactly that reason and
# reversed the decision on 2026-08-06. If the field turns out slower than assumed, THIS is the
# first default to put back to 24 -- the other v1.5.15 changes are quality wins at neutral or
# negative runtime and carry no such conditionality.
# K=12 is also the largest budget fitting ONE wave through 12 workers.
TOPK_MAX = int(os.environ.get('TOPK_MAX', 12))
LEGALIZE_WORKERS = int(os.environ.get('LEGALIZE_WORKERS', 12))

# CONSTRAINT-AWARE PRECEDENCE (v1.5.13, B2 + E7). The precedence graph is COMPLETE over pairs,
# so ordering alone can make a soft constraint UNREACHABLE: a foreign block ordered between two
# grouping-group members forces a gap of at least that block's width, and a block ordered on a
# flagged block's boundary side holds it off the wall by its whole width. Neither is a geometry
# problem and no post-solve hill-climb can repair either -- but both are removable by moving one
# edge to the other chain, which costs no bbox and, because it rides the graph `_iterate`
# re-infers every round, no extra solve. Paired over five pools: b2 -1.97%, e7 -1.24%, together
# -3.29% with 6 previously-infeasible cases RESCUED and none lost. 'b3' (route a group pair onto
# its abutment axis) measured a wash and is available but off. See docs/experiments/VIOLATION_REDUCTION_RESULTS.md.
GRP_DAG = tuple(x for x in os.environ.get('GRP_DAG', 'b2,e7').split(',') if x)

# FINISHERS TO A JOINT FIXED POINT (v1.5.13, P7). `snap_groups` and `pull_boundary` interact
# through the bounding box -- a snap that grows it moves every wall, a pull that moves a wall
# changes which snaps are affordable -- so running each once gives whichever runs second the last
# word. Both are individually monotone under the same priced score, so iterating cannot lose
# ground. -5.56% over five paired pools with ZERO cases worse on any pool, and it is superadditive
# with GRP_DAG (-9.87% together against -5.56 and -1.90 apart), because a finisher can only
# compound when the corridors it re-tries are not forbidden by the program.
FINISH_ROUNDS = int(os.environ.get('FINISH_ROUNDS', 4))

# BATCH WALL PULL (v1.5.13, E3). `pull_boundary` moves one block at a time under a monotone
# priced gate, so it can never take a step that needs two blocks to move together -- and since
# the scorer's bbox is SELF-REFERENTIAL, that is the common case: the block holding a wall out is
# usually itself flagged to that wall, so neither moves alone and both move together. One extra
# priced move per wall, translating every movable block flagged to it at once: V_boundary -44%
# for +10% legalize time, which is better than the in-solve P4 constraint on BOTH axes.
PULL_BATCH = os.environ.get('PULL_BATCH', '1') != '0'

# BOUNDARY RESHAPE (v1.5.8, adopted; DEFAULT OFF since the 2026-08-04 full-surface ablation).
# Lets a free non-MIB block GROW area-exactly toward its wall when it cannot slide. It was worth
# ~-14% of the residual V_bnd when adopted, on an exactly-paired run over 200 stored placements
# (52 touched, 52 better, 0 worse). It is now QUALITY-NULL: +0.03% paired (2/3) and -0.05%
# neutral (1/3) on fresh seeds, touching 1-7 cases per 100. PULL_BATCH, FINISH_ROUNDS and
# TRUE_BBOX have since closed the residual it used to reach -- the clearest interaction effect in
# that campaign. Turned off as DEAD WEIGHT, not as a measured win: its apparent -1.58% real cost
# (3/3 on one campaign) did NOT reproduce on fresh seeds (-0.30%, 2/3, sign flipping per seed).
# `--pull-reshape` restores it. See docs/experiments/ABLATION_FULL_SURFACE.md.
PULL_RESHAPE = os.environ.get('PULL_RESHAPE', '0') != '0'

# SNAP CROSS-BORDER MARGIN (v1.5.13, A2). The scorer's grouping predicate needs a shared border of
# POSITIVE LENGTH; `snap_groups` used to aim for 25% of the smaller block, and bought that margin
# with a CROSS-AXIS shift that walked the moving component into a neighbour. Audited on the
# v1.5.12 artifact, that margin alone declines 28 merges that are otherwise legal and free -- more
# than the pass limit, the bbox cap and the priced test combined (which decline zero). Asking for
# the scorer's own predicate and nothing more: -0.50% alone over five pools, and about -1.1% more
# on top of GRP_DAG, whose emptied corridors it stops the snap from re-entering.
SNAP_CROSS_FRAC = float(os.environ.get('SNAP_CROSS_FRAC', 1e-6))

# SNAP CORRIDOR CLEARING (G2). 88% of `snap_groups` declines are a foreign block occupying the
# destination, not a price or a cap. Rather than give up, evict the obstructor and re-test under
# the SAME feasibility and priced gates as the rigid move. Each entry of CLEAR_MOVES is a
# different eviction: `push` carries the obstructors along, `step` evicts them sideways,
# `reshape`/`split`/`thin` reshape one area-exactly.
# DEFAULT ('step',) since v1.5.14. `step` carries most of the effect; the rest are opt-in.
# Adopted on a pre-registered REAL-runtime gate (docs/experiments/SNAP_CLEAR_GATE.md), because its
# only prior number was a PAIRED -1.34%, and paired quality wins are exactly what the
# max(0.7, R^0.3) term took back from `--cluster`, `--grp-hard` and the candidate mix.
# Measured on top of TRUE_BBOX: -0.86% neutral / -0.44% real, better on 3/3 seeds on BOTH
# metrics, 300/300 feasible, V_mib 0, V_grp 191.3 -> 181.7, for +1.89% wall.
# The real gain is nearer -0.2%: seed 7303 supplied half the total while its V_grp barely moved,
# so part of that seed is draw variance rather than the mechanism.
SNAP_CLEAR = tuple(v for v in os.environ.get('SNAP_CLEAR', 'step').split(',') if v)

# H3 -- WIRELENGTH-AWARE PRECEDENCE TIE-BREAK (docs/experiments/GROUPING_HPWL_CAMPAIGN.md 5.1).
# Default OFF. `PREC_SEP_RULE` decides ~67% of all precedence edges, for pairs already separated
# on BOTH axes, and BOTH choices are legal -- v1.5.11 showed that this one free degree of freedom
# is the whole game for utilization. H3 points the same lever at wirelength: for a heavy netlist
# pair, order it on the axis that leaves the smaller forced separation, so the solve can still
# bring the two together. The value is the TOP FRACTION of each case's b2b pairs by summed weight
# that the override applies to; 0 = off (the shipped v1.5.11 tie-break everywhere).
# 0.10 is where the H13 autopsy put the money: the heaviest weight decile is 37.6% of the
# netlist's weight mass but only 14.5% of GT's wirelength -- GT abuts those pairs -- and it
# carries 79% of our b2b excess, at 1.6x GT's normalised centre distance.
SEP_NET_Q = float(os.environ.get('SEP_NET_Q', 0.0))

# J1/H10 -- ONE EXACT ACCEPTANCE TEST EVERYWHERE. Default OFF.
# `snap_groups` and `pull_boundary` accept a move on `(1 + ALPHA*area_gap) * exp(BETA*V_rel)`,
# but the contest sums `area_gap` and `hpwl_gap` inside ONE ALPHA. So a move that abuts a group
# or reaches a wall by dragging a heavily-netted block across the layout is priced as free.
# This is the third instance of the same defect: B8-a found it in the candidate-acceptance test
# (where pricing wirelength turned 34 better / 10 worse into 34 / 0) and H1 found it in the
# selection proxy. Turning it on hands the contest's own b2b/p2b/pins rows to every post-solve
# hill-climb. It costs an O(rows) HPWL evaluation per candidate move, which is why it is a flag.
PRICED_HPWL = os.environ.get('PRICED_HPWL', '0') != '0'

# TWO-LEVEL CLUSTER SOLVE (B8, docs/experiments/VIOLATION_REDUCTION_PLAN.md section 5.B8). Default OFF.
# GROUP_SUPERMODULE_SCREEN.md rejected the POST-HOC form -- rigidly reinserting an already-packed
# group into an already-packed layout accepted zero placements, because M9's output has no
# outline-preserving slot. Its own diagnosis was that a supermodule "would have to participate in
# the global packing topology from the start". This is that constructive form: solve each owed
# group as its own small sub-floorplan, hand the result to the global solve as ONE fixed-shape
# node, then expand it in place. The group's abutment is then true by construction rather than
# repaired, and no foreign block can be ordered inside it because it is not a set of nodes at all.
# Runs as an extra CANDIDATE alongside the ordinary solve and is chosen only if it wins the same
# priced (area, V_rel) comparison G3 uses, so it can never lose a case -- it can only cost time.
CLUSTER_SOLVE = os.environ.get('CLUSTER_SOLVE', '0') != '0'
# Only groups that are actually UNSATISFIED after the ordinary finish are clustered: a satisfied
# group needs no help and clustering it only removes freedom from the global packing.
CLUSTER_MIN_SIZE = int(os.environ.get('CLUSTER_MIN_SIZE', 2))
CLUSTER_MAX_SIZE = int(os.environ.get('CLUSTER_MAX_SIZE', 12))
# Reject a cluster whose own rectangle is emptier than this (sum member area / W*H). The measured
# density of a group's CURRENT footprint on the shipped v1.5.13 artifact is a median of 0.502, so
# anything above that is strictly less space than the group already occupies.
CLUSTER_MIN_DENSITY = float(os.environ.get('CLUSTER_MIN_DENSITY', 0.85))
# CONDITIONAL ESCALATION (the E1b pattern). The extra global solve is charged in full, so run it
# only where there is enough owed grouping to repay it. Measured over five paired pools: the
# whole -1.62% survives a threshold of 2, 97.5% of it survives 3 (at 24% less added time) and
# 93% survives 4 (at 37% less). Below 3 the arm is paying a solve to accept nothing.
CLUSTER_MIN_OWED = int(os.environ.get('CLUSTER_MIN_OWED', 3))
# A sub-instance is k <= 12 blocks with no preplaced, where the expensive finishes have almost
# nothing left to find. Measured over five paired pools, cutting the sub-solve to iters=2 /
# finish_rounds=1 / no gap-repair / no reshape scored -1.62% against the full sub-solve's -1.60%
# -- i.e. marginally BETTER -- while removing 17% of the arm's added time. The same saving
# attempted on the REDUCED GLOBAL solve (cluster_iters=1) is a bad trade and is not taken: it
# gives up 37% of the win (-1.00%) to save 31% of the cost.
CLUSTER_SUB_LIGHT = os.environ.get('CLUSTER_SUB_LIGHT', '1') != '0'
# BATCHED SUB-SOLVE. A 7-block cluster sub-solve measured 17.8 ms against CLARABEL's own
# sub-millisecond time: it is essentially all cvxpy canonicalisation, which is paid per PROBLEM.
# A case's clusters share no block and no constraint, so stacking them into ONE program with a
# per-cluster outline gives the identical solution -- the program separates -- for one
# canonicalisation instead of one per group. Pure implementation, no quality trade.
# Default OFF: measured end-to-end it trades 0.30pp of neutral quality for 2.4pp of wall
# clock, and the runtime penalty is governed by how many cases CROSS the R floor rather
# than by the wall clock itself -- the saving moved only 2 of 100 cases back under it, so
# the quality cost is not repaid. Kept for the runtime-bound case (see CLUSTER_SOLVE_SCREEN 6b).
CLUSTER_BATCH = os.environ.get('CLUSTER_BATCH', '0') != '0'
# WHITESPACE PRE-GATE. The candidate is accepted on only ~45% of the cases the arm runs on, and
# the sub-solves (34% of its cost) are enough to predict most of the rest: a cluster charges the
# packing its own whitespace, so if the clusters already waste more than this share of the
# whitespace the base layout HAS, the reduced global solve (the other 61%) is money spent to be
# declined. None = no pre-gate.
# 0.35 measured over five paired pools: -1.616% against the ungated -1.622% (99.6% of the win)
# for 19% less added time. 0.20 starts costing quality (-1.558%).
CLUSTER_MAX_WASTE = (None if os.environ.get('CLUSTER_MAX_WASTE', '0.35') in ('', 'none')
                     else float(os.environ.get('CLUSTER_MAX_WASTE', '0.35')))

# SOFT-PIN RESEED (v1.5.10 default; gated on two paired official-100 pools).
# When the M9 conic solve is infeasible the cause is always the same: the precedence graph
# inferred from the raw draw is inconsistent with the PINNED preplaced coordinates. Dropping
# the `x[pp] == llx0[pp]` equality rescued 31 of 31 observed failures, so the instance is
# solvable and only the inferred ordering is wrong. SOLVE_RESEED_ROUNDS > 0 re-solves with the
# pins as an L1 penalty (always feasible), snaps preplaced back onto their exact pins, and
# re-runs the ordinary hard-pin solve from that layout. SOLVE_RESEED_LAM is that penalty's
# weight: large enough that preplaced barely move, small enough to stay well conditioned.
# 2 rounds: the second round rescues cases the first does not, and a third never fired in the
# 600-draw probe. Runs ONLY on the failure path, so solved cases stay bitwise identical.
SOLVE_RESEED_ROUNDS = int(os.environ.get('SOLVE_RESEED_ROUNDS', '2'))
SOLVE_RESEED_LAM = float(os.environ.get('SOLVE_RESEED_LAM', '50.0'))


# --------------- Model architecture ----------------
# CAPACITY BUMP (2026-06-03): HIDDEN_DIM 128->192 test. The 6x density run showed the
# big-count tail (101-120) is CAPACITY-bound not data-bound (more data REGRESSED the tail
# +5.4% while small counts gained). Widen the per-node hidden state (NOT depth — the
# GlobalContextModule already covers receptive field). Requires retrain (shape change).
HIDDEN_DIM       = 192
COND_DIM         = 64
POS_ENC_DIM      = 32
NUM_BLOCKS       = 3
LAYERS_PER_BLOCK = 2
FFN_MULT         = 4
GLOBAL_EVERY     = 1

# --------------- Energy conditioning ---------------
LAMBDA_HPWL      = 1.0
LAMBDA_OVERLAP   = 1.0
LAMBDA_BBOX_EREL = 0.5
KAPPA            = 5.0

# --------------- Manifold Constraint Guidance ------

# --------------- Edge Pruning ----------------------
# Env-overridable for the netlist-edge sweep (does a more complex/big graph want LESS
# pruning = more retained b2b edges?). '0'/'none' => no pruning (keep all netlist edges).
# Applied in build_sample_data, so an env override changes the eval graphs => inference-only.
_ETK = os.environ.get('EDGE_TOPK') or None   # unset or empty -> default
EDGE_TOPK = (8 if _ETK is None else (None if _ETK.lower() in ('0', 'none') else int(_ETK)))

# --------------- Spatial K-NN edges ----------------
# Env-overridable for the KNN-edge sweep (does scaling spatial K with block count help
# the intrinsic tail?). Attention aggregation (softmax over incoming edges) is degree-robust,
# so changing K at inference is only mildly OOD -> an inference-only sweep is meaningful.
SPATIAL_K_CENTROID = int(os.environ.get('SPATIAL_K_CENTROID', 4))
SPATIAL_K_EDGE     = int(os.environ.get('SPATIAL_K_EDGE', 4))
USE_SPATIAL_EDGES  = True

# --------------- Pin / p2b configuration -----------
USE_PINS         = True
PIN_EDGE_TOPK    = None

# --------------- Boundary constraint feature (experiment a) ---------
# Feed placement_constraints[:,4] (boundary code: L=1,R=2,T=4,B=8, corners=sums)
# as a 4-bit [L,R,T,B] node FEATURE so the GNN knows the perimeter ring of the
# layout (the dataset flags every boundary block; boundary is a SOFT contest
# constraint = "block edge touches the bounding-box edge / be the extreme block").
# input_dim grows by 4 -> requires retrain.
USE_BOUNDARY_FEAT = True

# Experiment (b): explicit boundary objective (on top of the feature).
# Training: a soft loss penalizing a flagged block's relevant edge for not
# coinciding with the layout bounding-box edge (the contest 'boundary' soft
# constraint, measured against OUR solution's bbox -> self-referential).
# Inference: a positional nudge biasing flagged blocks toward the matching
# extreme of the current layout during low-t DDIM steps.
# Attribution run (test/boundary-feat-fix): boundary FEATURE + pin_centroid fix only
# — the explicit boundary loss + inference nudge are OFF, to isolate how much of the
# boundary-loss champion (~713) was the fix vs the loss/nudge (boundary-feat was ~867).
USE_BOUNDARY_LOSS         = False  # exp (b) loss OFF for this attribution run
LAMBDA_BOUNDARY_TRAIN     = 0.1
BOUNDARY_WARMUP_START     = 5
BOUNDARY_WARMUP_END       = 15
# E2: nudge strongly overlap-beneficial; 0.65 was the prior principled max.
# v1.1 (inference sweep, tools/sweep_inference.py): retuned 0.65->0.8 — on the official
# 100 this cut overlap a further ~5% AND IMPROVED BndViol (0.13->0.08); both-better, robust
# across all block-count bands. See the experiment ledger ("v1.1 inference profile").
# 2026-08-05: made env-overridable for the parameter sweep. Values UNCHANGED — these three were
# tuned in v1.1 against a pipeline that shipped the sampler's output nearly raw, and have never
# been re-measured since M9 began re-deriving the packing. Without the env read the sweep arms
# would have silently run the default and reported a null.
BOUNDARY_NUDGE_STRENGTH   = float(os.environ.get('BOUNDARY_NUDGE_STRENGTH', 0.80))
BOUNDARY_NUDGE_TARGET_EMA = float(os.environ.get('BOUNDARY_NUDGE_TARGET_EMA', 0.5))  # E4: EMA-smooth the nudge target extreme (0=off)
# E5: at HIGH global REPEL (1.5) the ring-focused repulsion is NOT subsumed — it adds
# on top (overlap 232->223) and sharply cuts variance (+/-12->+/-4) and boundary-viol.
BOUNDARY_TANGENTIAL_SPREAD = float(os.environ.get('BOUNDARY_TANGENTIAL_SPREAD', 0.5))
# (Removed 2026-06-14: BOUNDARY_NUDGE_PERCENTILE / PCT_END / PCT_HOLD_FRAC — the per-edge-
# flagged percentile target experiment (2026-06-01) was negative; the nudge always used the
# true bbox extreme (q=1.0). See git history if revisiting.)

# --------------- MIB clamp (soft constraint, inference) ----------
# MIB groups (placement_constraints col 2, id>0) must have IDENTICAL (w,h). Members
# always share the same fixed area (verified 100/100 official), so equalizing
# log_aspect_ratio within a group yields identical (w,h) -> V_mib=0, at zero cost to
# the hard area constraint. Handles arbitrarily many groups (test set has Q=1).
# Inference-only; no retrain. Applied to x[:,2] (log_ar) each step in the window
# the WHOLE schedule so repulsion settles the resized shapes; a final exact snap after the
# loop guarantees V_mib=0. Env-overridable for the sweep. Default off.
# ADOPTED (2026-06-02): mean + sustained is default-on. The former MIB_CLAMP_START_FRAC knob
# was removed in v1.5.11: adoption fixed it at 1.0, which made it a no-op that nothing read.
# Official 100 r3: MibViol 3.60->0.00 (exactly, via the fixed-shape-aware target) at NO
# overlap cost (748.7->736.6, -1.6%; BndViol 0.128->0.123) — a free soft-constraint win.
USE_MIB_CLAMP        = (os.environ.get('MIB_CLAMP', '1') == '1')
MIB_CLAMP_STRENGTH   = float(os.environ.get('MIB_CLAMP_STRENGTH', '1.0'))  # frac toward target/step
MIB_CLAMP_METHOD     = os.environ.get('MIB_CLAMP_METHOD', 'mean')          # 'mean' | 'median'

# --------------- Grouping force (soft constraint, inference) ----------
# Grouping groups (placement_constraints col 3, id>0) must ABUT into ONE connected
# component (V_grouping = sum_p (c_p-1)). A SUSTAINED attraction (init-only washes out)
# on movable group members each low-t step, BEFORE repulsion so it settles the overlap
# the pull creates. Env-overridable for the Tier-1 sweep. Default off.
#   METHOD: 'centroid' (members->group centroid), 'component' (disconnected comps->
#           largest comp), 'nearest' (member->nearest same-group member).
#   GATED:  1 = apply only to members NOT touching a same-group member (release once
#           abutting, hand off to repulsion); 0 = always-on.
#   Window: active for T_grp_lo <= t < T_grp where T_grp=START_FRAC*T, T_grp_lo=END_FRAC*T.
#           END_FRAC>0 RELEASES the force for the final low-t steps (early-then-release).
# ADOPTED (2026-06-02): component + gated + early-release (strength 0.30, END_FRAC 0.2)
# is default-on. Official 100 r3: GrpViol 6.85->3.40 (-50%) at NEUTRAL overlap (733.7->734.5,
# within +/-6-11 noise). EARLY-RELEASE is the key: sustained costs +overlap (s0.10 sust = 807),
# but releasing the pull for the final low-t steps lets repulsion tile the pre-clustered groups
# -> overlap-neutral. component (merge disconnected comps) beats centroid/nearest. Pushing
# strength/lower-release cuts GrpViol further but trades overlap; the rest is the legalizer's.
USE_GROUP_FORCE        = (os.environ.get('GROUP_FORCE', '1') == '1')
GROUP_FORCE_METHOD     = os.environ.get('GROUP_METHOD', 'component')        # centroid|component|nearest
GROUP_FORCE_STRENGTH   = float(os.environ.get('GROUP_STRENGTH', '0.30'))
GROUP_FORCE_GATED      = (os.environ.get('GROUP_GATED', '1') == '1')
GROUP_FORCE_START_FRAC = float(os.environ.get('GROUP_START_FRAC', '1.0'))
GROUP_FORCE_END_FRAC   = float(os.environ.get('GROUP_END_FRAC', '0.2'))

# --------------- Hard-constraint clamping ----------
USE_PREPLACED_CLAMP   = True
USE_FIXED_SHAPE_CLAMP = True

# --------------- Canvas boundary handling -----------
# HARD clamp of block edges to the boundary expanded by CANVAS_BOUNDARY_MARGIN_FRAC.
# The contest has no boundary-overflow penalty, so the 5% margin gives the downstream
# legalizer a little slack. (Removed 2026-06-14: the v0.16 CANVAS_SOFT_BOUNDARY soft
# quadratic/exponential push + canvas_soft_push() — superseded by this hard clamp; see
# git history if revisiting.)
CANVAS_BOUNDARY_MARGIN_FRAC = 0.05   # boundary expanded 5% past the pin-defined border

# --------------- Canvas normalization (v0.13) -------
# When >=2 pins span both x AND y, the pin bounding box defines the canvas:
# pins are mapped to [-CANVAS_FRAC, +CANVAS_FRAC]^2, leaving a small margin.
# When pins are absent / degenerate, fall back to constant-total-area
# normalization (the v0.12 behaviour).
USE_PIN_DEFINED_CANVAS = True
CANVAS_FRAC            = 0.95    # margin: pins at +/-0.95 in normalized coords

# --------------- Training Overlap Loss -------------
LAMBDA_OVERLAP_TRAIN  = 0.3
OVERLAP_WARMUP_START  = 5
OVERLAP_WARMUP_END    = 15
OVERLAP_SAMPLE_K      = 32
OVERLAP_USE_QUARTIC   = True
# v0.17: blend quartic (rdx²·rdy²) with bilinear (rdx·rdy) at this mix ratio.
# Bilinear gives non-vanishing gradient for tiny overlaps where quartic ≈ 0.
# 0.0 = pure quartic (v0.16), 1.0 = pure bilinear.
OVERLAP_LINEAR_MIX    = 0.10
OVERLAP_MARGIN        = 0.02
OVERLAP_NORMALISE_BY_SIZE = True
OVERLAP_MAX_WEIGHT    = 0.5
# Asymmetric multiplier applied to movable-vs-preplaced overlap pairs. Pairs
# where exactly one endpoint is preplaced count this many times more than
# movable-vs-movable pairs (so the model is rewarded extra for routing around
# preplaced obstacles). 1.0 disables the asymmetry. Used in both training
# (compute_overlap_loss_sampled) and in inference s_cons_max guidance.
PREPLACED_OVERLAP_WEIGHT = 4.0

# --------------- Spectral / Quadratic Init ---------
# Anchored quadratic placement (preferred when pins are available): solve
# (L_bb + alpha*D_p + alpha_pp*D_pp + eps*I) P = alpha*Q_p + alpha_pp*Q_pp
# where L_bb is the b2b block Laplacian, D_p / Q_p come from p2b pin anchors,
# and D_pp / Q_pp come from preplaced-block anchors at their GT positions.
# PIN_ANCHOR_ALPHA = relative weight of p2b vs b2b edges (your knob: "higher
# p2b weight").
# PREPLACED_ANCHOR_WEIGHT_MULT = extra factor on top of PIN_ANCHOR_ALPHA so
# preplaced anchors strongly dominate (we know exactly where they go).
# QUAD_REG_EPS = diagonal regularisation so the system is always solvable
# even when a connected component has no anchors.
# v0.16: after the QP solve, scale all block positions toward the origin by
# this factor. Prevents heavily-pinned blocks from piling up on the canvas
# border in the initial state (many initial overlaps → slow DDIM convergence).
# 1.0 = no nudge (v0.15 behaviour); 0.90 = pull 10% toward centre.
# Spectral-init GROUP attraction (experiment, 2026-06-01). 0.0 = off (baseline). >0 adds a
# clique-Laplacian term over same-grouping-id blocks (col3) to the quadratic-init matrix M, so
# same-group blocks START clustered (cheapest, least-overlap-damaging way to bias the SOFT
# grouping/abutment constraint — no per-step force fighting repulsion). b2b weights are ~1e-3
# and alpha*Dp ~1e-2, so a weight ~0.1-1 makes grouping competitive without collapsing groups.

# --------------- Direct Repulsion ------------------
USE_REPULSION      = True
# E10 (2026-05-31): REPEL_STRENGTH was badly UNDER-tuned at 0.5. It is the single
# biggest overlap lever found — and unlike the nudge it does NOT inflate BBox (it
# tiles tighter within the same footprint). Sweep (nudge 0.65, 20 val, r2/r3):
#   0.5->438  0.75->388  1.0->323  1.25->276  1.5->258  2.0->244(+/-23)  2.5->382  3.0->720
# Monotonic down to ~2.0 then a hard CLIFF at 2.5 (repulsion overshoots -> oscillation).
# v1.1 (full-range inference sweep, tools/sweep_inference.py): on the OFFICIAL 100 the knee
# is 1.75 (631+/-6), strictly below the cliff (2.0 regresses to 684 here) and LOWER variance
# than 1.5 (+/-6 vs +/-12). Retuned 1.5->1.75. (The old 1.5 note was from the 20-sample dev
# split; the official-100 sweep moved the knee one step up.) See the experiment ledger.
REPEL_STRENGTH     = 1.75

# --------------- Pin-centroid force (Test 2, experimental) ---------
# Each block receives its weighted pin-centroid (cx,cy) plus a has-pins flag as
# input features. Training adds a pin-attraction loss and inference applies a
# positional force toward that centroid during the low-sigma EDM window.
# PIN_FORCE_STRENGTH=0.0 disables only the inference force (feature+loss remain).
PIN_FORCE_STRENGTH   = 0.15   # inference: fraction of (target - pos) per low-t step
PIN_ATTRACT_LAMBDA   = 0.05   # training: weight on the pin-attraction loss
PIN_ATTRACT_WARMUP_START = 5
PIN_ATTRACT_WARMUP_END   = 15

# --------------- DDIM sampling ---------------------
# v1.2 (2026-06-10): 50->100. More DDIM steps = a finer reverse trajectory = lower overlap
# (official 100: 518->339, -35%) for ~2x wall-clock; a BETTER compute lever than best-of-N
# (which saturated at 6). The cleanest "spend the runtime budget" knob. See the experiment ledger.
# v0.17: η=0.3 adds mild stochasticity so Best-of-N candidates explore different
# local minima (η=0 was fully deterministic; all N copies were identical up to
# repulsion jitter). η=1.0 recovers full DDPM.
# v1.1 (inference sweep): raised 0.3->0.5. MORE stochasticity = MORE diverse Best-of-N
# candidates = lower min overlap (official 100: 633->582, monotone 0.0<0.15<0.3<0.5). The
# init type barely matters but candidate DIVERSITY does — eta is the diversity lever.

# --------------- Best-of-N -------------------------
# v1.1: raised 4->6. Best-of-N overlap is MONOTONE in candidate count (official 100, distilled
# base: N4 560 -> N6 519 -> N8 510) — the count, not the init type, is the lever. 6 is the
# chosen quality/RuntimeFactor knee (+34% wall-clock per case vs N=4 for -7% overlap; N=8 adds
# only -2% more for another +30% time). Size-allocating N by block count was a WASH at equal
# budget (small designs benefit from more candidates too). See the experiment ledger.
BEST_OF_N          = int(os.environ.get('BEST_OF_N', 6))

# CANDIDATE SELECTION RULE. The pool is BEST_OF_N inits x every pooled restart state = 36
# candidates at the shipping profile, and exactly one is kept.
#   'overlap'    = argmin raw pairwise overlap. The pre-v1.5.8 rule, and STALE since v1.5.4:
#                  M9 legalization drives overlap to zero on every candidate, so raw overlap
#                  now only proxies "easy to legalize" (Spearman rho 0.21 vs final cost).
#   'cheap_pack' = rank by bbox * exp(BETA * V_rel) of a SOCP-free longest-path pack
#                  (compaction.compact_positions, the legalizer's own position-only fallback).
#                  2.4 ms per candidate -- ~86 ms per case against ~5 s, so effectively free.
# Screened on 3600 legalized candidates x 2 independent draws: priced -4.09% / -3.87%, 57/21
# and 56/29 per-case. Adding an HPWL term or an overlap tie-break did NOT replicate.
# See docs/experiments/RERANK_SCREEN.md.
SELECT_RULE        = os.environ.get('SELECT_RULE', 'cheap_pack')

# --------------- Tier-1 inference experiments (2026-06-10) ----------
# (#1) RE-NOISE REFINEMENT — ADOPTED v1.2 (default-ON, rounds=2): after best-of-N, renoise the
# batch to t=REFINE_T_FRAC*T and re-denoise that short low-t tail, REFINE_ROUNDS times; ALL
# candidates (original + every refined round) are pooled and the global-min-overlap one is
# returned (monotone by construction — refinement can only help). Official 100: 518->409 (-21%),
# and it IMPROVES bnd+grp. Stacks with DDIM_STEPS=100 (the two compose). 0 rounds = off.
REFINE_ROUNDS      = int(os.environ.get('REFINE_ROUNDS', 2))
# (v1.5.4: CFG_DROP_PROB removed. It nulled the conditioning on 10% of training steps to
# learn an unconditional prior for classifier-free guidance -- but CFG_SCALE went with the
# DDPM sampler in the v1.5.4 inference cleanup, so the EDM path never draws on that prior.
# It was a 10% tax on the training signal for an unused capability. See release/v1.5.4/.)

# --------------- Flip Augmentation (v0.17) ---------
# Randomly reflect the normalized layout (blocks + pins) across canvas axes
# during training. Breaks the directional bias in FloorSet GT solutions.
# 25% no-op, 25% x-flip, 25% y-flip, 25% 180-degree rotation.
USE_FLIP_AUG      = True

# --------------- Training --------------------------
# Full-range training: a random batch from blocks 21..120 averages ~2x the nodes of
# the old 30..40 range, and 120-block graphs are ~4x a 30-block graph -> a physical
# batch of 1024 risks OOM (esp. on the shared GPU). Use a smaller PHYSICAL batch with
# gradient accumulation so the EFFECTIVE batch (1024) and optimization match the champion.
BATCH_SIZE        = 384
ACCUM_STEPS       = 1     # v0.20: was 3 (eff. 1152). Accumulation does NOT help a
# data-bound run (util is set by PHYSICAL batch, not effective) — it only adds wasted
# micro-steps. ACCUM=1 -> effective batch = physical 384 (healthy for diffusion). Keep
# physical 384 (a bigger physical batch risks OOM on an all-120-block batch).
# v0.19: full-range batches starve the fast GNN step at batch 256 (util ~28%, spiky);
# 384 gives longer GPU bursts (~16GB, safe), 20 workers (of 24 cores) feed it faster.
# Still data-pipeline-bound (~31% util); the real fix is the precompute .pt cache
# (roadmap item 8) — worker count alone doesn't help (IPC of large full-range graphs).
EPOCHS            = 100
LR                = 1e-4
EMA_SMOOTH        = 0.95
SAVE_EVERY_EPOCHS = 10

# --------------- Tier-1 training techniques (test/edm) -------------
# A/B'd on the 90k screen on top of the EDM base. All env-gated / default-off so the
# control = plain EDM. See docs/experiments/EDM_SCREEN.md.
# (1) MODEL-WEIGHT EMA (Polyak): maintain an EMA of the weights, eval on the averaged
# weights (evaluate.py --use-ema). Standard diffusion best practice (EDM/EDM2) that was
# NOT implemented here — EMA_SMOOTH only smooths the LOGGED loss. Cheap, variance-reducing.
# ADOPTED default-on (test/edm Tier-1 screen, 2026-06-22): -9.2% bare overlap on the 90k
# EDM screen (2958->2685, CRN raw-weights sanity 2984~=baseline), lower variance, free at
# inference. Train with EMA, EVAL WITH --use-ema (raw weights are still saved too).
USE_EMA   = (os.environ.get('USE_EMA', '1') == '1')
EMA_DECAY = float(os.environ.get('EMA_DECAY', 0.9999))

# --------------- Data loading ----------------------
USE_EREL_CACHE = True   # memoise the deterministic E_rel score across epochs
NUM_WORKERS    = 8      # v0.20: with the PrebuiltDataset cache, per-worker work is a
# trivial index+clone+flip (~0.1ms) -> GPU-bound at 98% with FAR fewer workers. 20 workers
# was not only unnecessary but DANGEROUS on this SHARED box: each persistent worker COW-copies
# the cache's Python-object header pages on access (~2G/worker), so 20 workers inflated RSS by
# ~40G (33G cache -> ~79G total) and a transient spike from another user's job globally
# OOM-killed the 3x run at epoch 20 (dmesg: global_oom, killed pid 2130104). 8 workers keep
# util ~98% while cutting COW to ~16G. (Real fix for >385k / full-set: disk cache, roadmap 8b.)
# NOTE: "received 0 items of ancdata" comes from a low file-descriptor limit, not
# from this setting. Run with `ulimit -n` >= ~65535 (the system default shell has
# 1048576; an old login/tmux session may still carry the legacy 1024 → crash).
PIN_MEMORY     = True

# --------------- Filesystem defaults ---------------
# These are sourced from env vars at import time but creation of these
# directories is deferred to the entrypoints (train.py, evaluate.py, infer.py)
# so importing this module never touches disk.
PROJECT_DIR    = os.environ.get('FLOORPLAN_PROJECT',
                                 os.path.expanduser('~/floorplan_project'))
# Dataset + index JSONs live here (floorset_lite/, LiteTensorDataTest/,
# train_full_ranges.json, train_split.json, val_split.json). Loaders are vendored
# in diffusion_floorplanner/floorset_data.py — the external floorset-io repo is no longer used.
DATA_DIR       = os.environ.get('FLOORSET_DATA',
                                 os.path.join(PROJECT_DIR, 'data'))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
RESULTS_DIR    = os.environ.get('RESULTS_DIR',
                                 os.path.join(PROJECT_DIR, 'results'))

# --------------- Dataset cache (v0.21) -------------
# Two precompute backends build every graph ONCE (build_sample_data is deterministic):
#   USE_DISK_CACHE=False -> PrebuiltDataset: whole cache in RAM (~136KB/graph -> 33G @270k)
#       + per-worker COW (~2G/worker). Fast but the RAM/COW ceiling OOMs the shared box
#       past ~385k graphs (full set ~118G is impossible).
#   USE_DISK_CACHE=True (DEFAULT)  -> DiskCachedDataset: graphs serialized into ONE blob
#       file on disk + an in-RAM offset index; __getitem__ pread+deserializes one graph on
#       demand. Resident ANON RAM stays tiny (index 4MB + prefetch batches) — the blob is
#       paged in as RECLAIMABLE OS page cache, so it never counts toward the OOM ceiling.
#       Scales to the full set (only disk grows). REBUILT EVERY RUN (no persistence, no
#       validation) -> a fresh blob can never be stale; removed at process exit.
USE_DISK_CACHE = True
DISK_CACHE_DIR = os.environ.get('FLOORPLAN_DISK_CACHE',
                                 os.path.join(PROJECT_DIR, 'cache'))

# --------------- Visualisation toggle --------------
USE_LIVE_PLOT  = False     # IPython live-plot during training (off by default)
VIZ_B2B_ALPHA  = 0.08      # b2b line alpha in eval/export images

# --------------- Device ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# H5/H6 -- FROZEN-TOPOLOGY WIRELENGTH RE-SOLVE. Default OFF.
# Every wirelength arm before this one acted on the precedence graph (H3, `SEP_NET_Q`) or on a
# post-solve move's price (J1, `PRICED_HPWL`) and moved `hpwl_gap` by ~0. The reason is the same
# in both cases and it is structural: the M9 program minimises `W + H`, so nothing in the
# pipeline ever optimises wire. This puts the scorer's own term in the OBJECTIVE instead.
# With the precedence graph held fixed at the layout the ordinary solve produced, the feasible
# region is a polyhedron and weighted Manhattan distance between block centres is convex, so one
# extra solve returns the EXACTLY HPWL-optimal placement inside the shipped topology.
# `HPWL_EPS` is H6: the fraction by which the outline's PERIMETER may grow. 0.0 is H5 proper --
# the wirelength gain must come out of slack that already existed. A tuple sweeps several and
# lets the priced acceptance test pick, at one solve each.
# Each entry is `kind@eps`: `hpwl` (H5/H6, the scorer's b2b wirelength), `bnd` (H8, the distance
# of every boundary-flagged block to its required wall -- P4's hard constraint, PRICED instead,
# so it can never make the program infeasible), or `both`. `eps` is the fraction the outline's
# perimeter may grow. Each entry costs ONE extra solve and enters the priced acceptance test as
# an ordinary candidate, so a bad trade is declined rather than shipped.
RESOLVE = tuple(v for v in os.environ.get('RESOLVE', '').split(',') if v)

# D. Minimise the TRUE outline (W + H - X0 - Y0) rather than the outline measured from the
# ORIGIN. The two are the same thing only when the packing can slide until min(x) = 0, and a
# preplaced equality in the solve frame is exactly what stops it. Costs no extra solve.
# DEFAULT ON since v1.5.14: -1.52% neutral / -1.10% real on 3/3 fresh seeds with a control
# beside them, 300/300 feasible, V_bnd -32% (119/119/121 -> 82/83/78), for +1.6% wall. It is a
# structural correction with no tuned constant. See docs/experiments/BOUNDARY_RESIDUAL_CAMPAIGN.md.
TRUE_BBOX = os.environ.get('TRUE_BBOX', '1') != '0'
# B. Weight of the boundary-flag distance term ADDED to that objective (H8 without a second
# solve). Normalised to mean distance per flagged edge. 0 = off.
# DEFAULT ON since 2026-08-06 (parameter sweep): the single largest lever the sweep found.
# -2.95% neutral / -2.34% real on 3/3 fresh seeds with a control beside them, 100/100 feasible,
# V_bnd 73.7 -> 38.0, and paired replay over 300 stored cases agreed closely (-3.24%).
# The value is NOT knife-edge: 768..16384 is one flat plateau within noise (-2.3% to -3.2%
# paired), and 1024 vs 4096 are indistinguishable end-to-end (-2.43% vs -2.34% real). 4096 is
# chosen because it is the value the adopted PAIR was gated at, not because it beat 1024.
# Because the term is normalised per flagged edge, a "large" weight here is not a large number in
# the objective -- it is roughly "one unit of mean boundary distance is worth 4096 units of
# outline", and the solve still returns 100/100 feasible at 16384.
LAM_BND_ADD = float(os.environ.get('LAM_BND_ADD', '4096'))
# A. Candidate mix: every second candidate uses this `LAM_BND_ADD` instead. Substitution, so K
# and the wave count are unchanged; the priced selection keeps the better objective per case.
ALT_LAM_BND_ADD = float(os.environ.get('ALT_LAM_BND_ADD', '0'))
