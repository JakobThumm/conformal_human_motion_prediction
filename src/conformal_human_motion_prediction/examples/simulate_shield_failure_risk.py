"""Estimate the robot shield's dangerous-failure rate by *factorising* the rare event.

Why not count it directly. ``examples.simulate_robot_shield`` replays ~6e4 recorded human motion
windows against ~3.5e6 random robot placements (~2e13 "test cycles") and puts a Clopper-Pearson
bound on the 0-2 observed dangerous failures. Those cycles are **not independent**: the same finite
motion corpus is reused ~1e8 times, so a binomial bound over cycles is not a statement about the
world. The estimate below never counts the rare event over dependent cycles. It factorises it:

    PFH_D [1/h] = N_h * P(F) * P(D | F),        N_h = 3600 s / t_cycle                        (1)

A dangerous failure needs BOTH (a) the motion prediction to fail -- the true human occupancy
escapes the predicted set -- AND (b) that failed prediction to actually produce a dangerous
contact. The factorisation is *exact*, not an approximation, because of the lemma

    verified AND contact  =>  the truth left the predicted set,                                (2)

i.e. the shield can only be fooled inside the prediction-failure set F (if the truth stayed inside
the predicted set, a verified -- contact-free-against-the-prediction -- trajectory is contact-free
against the truth). ``--verify_lemma N`` tests (2) empirically on non-failure windows.

The two factors are measured on completely different footings:

  * ``lambda_f`` -- prediction-failure *episodes* per hour of operation. Measured from the recorded
    data: hundreds of events, so a tight interval. The per-window failure indicator is a **duty
    cycle**, not a rate; the rate is rebuilt at the *verification* cadence, i.e. one trial per
    SARA-shield planning cycle ``t_cycle`` (NOT per prediction -- a prediction is reused for
    ``K_P`` cycles and each inherits its failure status), so with ``N_h = 3600 / t_cycle``:

        p_hat    = mean(failure indicator)  over windows sub-sampled with --failure_stride
        lambda_f = p_hat * N_h / L                    [1/h]                                    (3)

    where ``L`` = mean failure-*episode* length in consecutive windows (measured from the
    run-length distribution of the indicator in dataset order; ~2.3 windows = 94 ms on the H36M
    validation split). Sub-sampling buys independent observations, it does *not* change the rate.
    The rate is the WINDOW rate -- never an episode rate. P(D|F) is measured per window, so the
    two must be paired per window; dividing by the episode length would price a whole episode as
    one window and understate PFH_D. Keeping the window rate over-counts a contact that persists
    across an episode, which is the safe direction. Window dependence is handled on the INTERVAL
    instead: the count is deflated to ``n_eff = N_F / (2 * L_corr)`` with ``L_corr`` the
    integrated autocorrelation time of the failure indicator and the 2 the loader's two phase
    offsets. The episode statistics are kept as the evidence behind ``L_corr``.

  * ``P(d | f_i)`` -- probability that failure ``f_i`` becomes a dangerous contact. Computed by
    replaying ONLY the failing windows against random robot placements, with the *same* geometry,
    culling hierarchy and GPU backend the shield simulation uses. This is an integral over a
    placement distribution we *choose* (area-uniform in a ``--pose_radius`` disk, random yaw, z
    offset), so its precision is limited by compute (``--num_robot_poses``), not by data.

    One placement is applied to ALL failing windows x trajectories, so those trials are a
    cluster, not independent draws, and a binomial bound over their product count would be
    anti-conservative. The bound therefore collapses a placement to the indicator "was ANY
    failing window dangerous here?" and is taken over the placements, which we did draw i.i.d.
    It is valid for the failure-averaged probability because for every placement
    ``mean_i 1{dangerous(i,p)} <= 1{any dangerous at p}``.

Both factors are one-sided Clopper-Pearson upper limits. They hold jointly with confidence
``(1-eps_F)(1-eps_{D|F}) = 1 - --epsilon_d`` because the recorded motions and the generated
placements are independent sources of randomness; the budget is split symmetrically. The
PRIMARY dangerous event is verified-AND-contact at ANY speed (matching the shield's verification
condition); the speed-gated variant, the point estimates, the placement ``sup`` and the bootstrap
over the observed failure set are all reported as diagnostics. The safety budget is inverted as

    exposure budget = 1e-6 / PFH_D                                                             (4)

the fraction of an operating hour of human-in-workspace exposure that keeps the hourly dangerous-
failure probability under the PL d / PFH_D = 1e-6 line (also printed in s/h and min/h).

Units: all lengths metres, all rates 1/h, ``t_cycle`` and horizon times seconds. A "window" is one
recorded motion-prediction sample (one 400 ms prediction from one measured pose).

Run::

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -m \
        conformal_human_motion_prediction.examples.simulate_shield_failure_risk \
        --backend gpu --num_robot_poses 2000 --verify_lemma 200
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time

import cloudpickle
import numpy as np
from tqdm import tqdm

# Importing this module also performs its argv-peeking jax-platform setup (--backend gpu keeps the
# GPU and enables x64; anything else pins jax to the CPU), so it must happen before jax is touched.
from conformal_human_motion_prediction.examples import simulate_robot_shield as srs
from conformal_human_motion_prediction.examples.simulate_robot_shield import (
    GATE_BLOCK,
    _skipped_pose_counts,
    build_shield_state,
    pose_active_sets_batched,
    root_dir,
    run_pose_pure,
    sample_robot_poses,
    write_results_csv,
)
from conformal_human_motion_prediction.generate_plots.conformal_results_common import pl_from_pfh

# PL d / SIL 2 line: the ISO 13849-1 PFH_D a safety function must stay under. Used only to invert
# the composed rate into an exposure budget (eq. 4).
PFH_D_TARGET = 1e-6
ACF_LAGS = (1, 2, 5, 10, 20, 50)

MODULE = "conformal_human_motion_prediction.examples.simulate_shield_failure_risk"
# Placements are drawn in blocks: the runs that tighten the bound to a useful level are ~4e8
# placements, where one [P,4] float64 array would be 13 GB. A run at or below this size is drawn
# in ONE call, so it stays bit-identical to the un-blocked version for a given seed.
POSE_DRAW_BLOCK = 1_000_000
# Per-placement statistics are accumulated ONLINE (running max + a coarse histogram + the count
# of placements with >=1 dangerous window), so memory does not grow with the placement count.
PLACEMENT_HIST_BINS = 20
PLACEMENT_TOP_KEEP = 1000
# Shard workers each load the (GB-scale) results cloudpickle; starting them slightly apart keeps
# those transient peaks (and 16 simultaneous JIT compiles) from piling up.
SHARD_STAGGER_S = 0.5
# Counter fields pooled across shards (all additive).
COUNTER_KEYS = ("total_pairs", "n_verified", "n_contact", "n_unsafe", "n_verified_contact",
                "n_verified_unsafe", "n_poses_skipped", "active")
# Flags the launcher owns; they must not be forwarded to the shard workers.
LAUNCHER_ONLY_FLAGS = ("--shards", "--shard_id", "--shard_out", "--shard_progress",
                       "--results_csv", "--save_per_failure", "--num_robot_poses")


# --------------------------------------------------------------------------- step 2: the set F


def escape_margins(pred_c, pred_r, true_c, true_r):
    """Per-window worst violation of predicted-set containment, in metres (>0 = escape).

    ``escape[m,s,j] = ||true_c - pred_c|| + true_r - pred_r`` is >0 exactly when the true
    occupancy sphere is not contained in the predicted one, so

        failure(m) = max over horizon steps s>=1 and joints j of escape[m,s,j]  > 0.

    Step 0 is the current *measured* pose, not a prediction, hence excluded. (s,j) entries that
    are invalid by the codebase convention (all-zero centre -> zero radius, cf.
    ``compute_human_occupancies`` / ``conformal_calibration.flatten_valid``) are skipped: there the
    two spheres are not comparable and a raw margin would read as a huge spurious escape.

    Written in terms of the occupancy arrays only, so it is identical for every predicted-set type
    (conditional-conformal, affine fallback, SARA).

    Returns ``(max_escape [M], step [M], joint [M])``; ``step``/``joint`` locate the worst
    violation and are -1 (with ``max_escape = -inf``) for a window with no valid entry at all.
    """
    d = np.linalg.norm(true_c[:, 1:] - pred_c[:, 1:], axis=-1)          # [M,PH,J] m
    esc = d + true_r[:, 1:] - pred_r[:, 1:]                             # [M,PH,J] m
    valid = (pred_r[:, 1:] > 0.0) & (true_r[:, 1:] > 0.0)               # [M,PH,J]
    esc = np.where(valid, esc, -np.inf)
    M, _, J = esc.shape
    flat = esc.reshape(M, -1)
    arg = flat.argmax(axis=1)
    mx = flat[np.arange(M), arg]
    step = arg // J + 1                                                 # +1: s=0 was dropped
    joint = arg % J
    none_valid = ~valid.reshape(M, -1).any(axis=1)
    return (np.where(none_valid, -np.inf, mx), np.where(none_valid, -1, step),
            np.where(none_valid, -1, joint))


# --------------------------------------------------------------------------- step 3: lambda_f


def run_lengths(flag):
    """Lengths of the maximal runs of True in ``flag`` -- the failure *episodes* in dataset order.

    Consecutive windows are 1/fps apart and overlap heavily, so a single escape typically shows up
    in several of them; the episode is the physical event, the window is the sampling grid.
    """
    f = np.asarray(flag).astype(np.int8)
    if f.size == 0 or not f.any():
        return np.zeros(0, dtype=np.int64)
    d = np.diff(np.concatenate([[0], f, [0]]).astype(np.int64))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def autocorrelation(x, lags=ACF_LAGS):
    """Sample autocorrelation of a 0/1 series at the given lags (dataset order)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    v = float(x @ x)
    return {int(L): (float(x[:-L] @ x[L:] / v) if v > 0.0 and 0 < L < x.size else float("nan"))
            for L in lags}


def clopper_pearson_upper(k, n, eps):
    """One-sided Clopper-Pearson UPPER limit at confidence ``1 - eps``: B^-1(1-eps; k+1, n-k).

    This is the limit the PFH_D bound needs for both of its factors. ``k`` may be fractional (the
    P(F) side feeds an effective sample size), which the Beta quantile handles.
    """
    from scipy.stats import beta
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - eps, k + 1.0, n - k))


def split_confidence(epsilon_d):
    """Symmetric budget split: eps_F = eps_{D|F} = 1 - sqrt(1 - eps_D).

    The two bounds hold simultaneously with confidence (1-eps_F)(1-eps_{D|F}) = 1-eps_D because
    the recorded motions and the generated placements are independent sources of randomness. Both
    limits depend on their level only logarithmically, so the allocation barely matters -- the
    symmetric split is chosen for being the one that needs no justification.
    """
    half = 1.0 - (1.0 - epsilon_d) ** 0.5
    return half, half


def integrated_autocorr_time(flag, max_lag=1000, c=5.0):
    """Integrated autocorrelation time ``L_corr = 1 + 2 * sum_k rho_k`` of a 0/1 series.

    The sum is truncated by Sokal automatic windowing: stop at the smallest lag ``M`` with
    ``M >= c * L_corr(M)``, which cuts the sum off once the window is comfortably wider than the
    correlation it is measuring and so avoids accumulating the noise in the far tail of the
    sample ACF.

    This is the *design effect* of the overlapping-window sampling: a run of ``N`` correlated
    windows carries about as much information as ``N / L_corr`` independent ones, so it is what
    the binomial interval on P(F) must be deflated by. It deliberately does NOT touch the point
    estimate ``k_F / N_F``.

    Computed on the series in dataset order, which concatenates per-file, per-phase-offset blocks;
    lagged pairs that straddle a block boundary pair up unrelated segments and dilute rho. With
    O(100) blocks over O(6e4) windows those pairs are ~2% of the total at the lags that matter, so
    the dilution is negligible -- but it biases ``L_corr`` *down*, so prefer ``--l_corr`` if a
    hand-checked value is available.
    """
    x = np.asarray(flag, dtype=np.float64)
    n = x.size
    x = x - x.mean()
    v = float(x @ x)
    if n < 8 or v <= 0.0:
        return 1.0
    tau = 1.0
    for k in range(1, int(min(max_lag, n // 4)) + 1):
        tau += 2.0 * float(x[:-k] @ x[k:]) / v
        if k >= c * tau:
            break
    return float(max(1.0, tau))


def clopper_pearson(k, n, alpha=0.05):
    """Two-sided Clopper-Pearson interval for a binomial proportion.

    ``k`` may be fractional: we deflate the raw window count to an effective sample size (the H36M
    loader emits both 50->25 fps phase offsets, so every motion appears twice, 20 ms apart), and
    the Beta quantiles are defined for non-integer shapes.
    """
    from scipy.stats import beta
    lo = 0.0 if k <= 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1.0))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - alpha / 2.0, k + 1.0, n - k))
    return lo, hi


# --------------------------------------------------------------------------- step 4: the replay


class _Progress:
    """Placement-progress sink: a tqdm bar in-process, a one-line file inside a shard worker.

    Shards must not interleave 16 tqdm bars on one terminal, so a worker instead rewrites a tiny
    ``<n_done> <n_total>`` file that the launcher polls to drive a single bar.
    """

    def __init__(self, total, desc, path=None):
        self.total = int(total)
        self.n = 0
        self.path = path
        self.bar = None if path else tqdm(total=self.total, desc=desc, unit="pose",
                                          dynamic_ncols=True, mininterval=2.0)
        self._last_write = 0.0

    def update(self, n, postfix=None):
        self.n += int(n)
        if self.bar is not None:
            self.bar.update(int(n))
            if (self.bar.n % 100 < n or self.bar.n == self.total) and postfix:
                self.bar.set_postfix_str(postfix, refresh=False)
        elif time.monotonic() - self._last_write > 1.0 or self.n >= self.total:
            self._last_write = time.monotonic()
            with open(self.path, "w") as f:      # tiny + rewritten in place: cheap to poll
                f.write(f"{self.n} {self.total}\n")

    def close(self):
        if self.bar is not None:
            self.bar.close()
        elif self.path:
            with open(self.path, "w") as f:
                f.write(f"{self.n} {self.total}\n")


def pose_block_factory(rng, n_poses, args):
    """Return a zero-arg callable that yields the SAME placement blocks on every call.

    What is snapshotted is the rng *state*, not the placements: the lemma pass has to replay the
    identical placements, and materialising them is not an option at 4e8 placements. With
    ``n_poses <= POSE_DRAW_BLOCK`` the whole set comes from a single ``sample_robot_poses`` call,
    i.e. bit-identical to the un-blocked draw for that seed.
    """
    state = rng.bit_generator.state
    blk = n_poses if n_poses <= POSE_DRAW_BLOCK else POSE_DRAW_BLOCK

    def blocks():
        r = np.random.default_rng()
        r.bit_generator.state = state
        done = 0
        while done < n_poses:
            n = min(blk, n_poses - done)
            yield sample_robot_poses(n, args.pose_radius, args.pose_z_offset, r)
            done += n

    return blocks


def new_placement_stats(M):
    """Zeroed online per-placement accumulators (see :func:`replay_placements`)."""
    return dict(k_dangerous=0, k_contact=0, sup_dangerous=0.0, sup_contact=0.0,
                argsup_dangerous=-1, hist=np.zeros(PLACEMENT_HIST_BINS, dtype=np.int64), top=[],
                act_hist=np.zeros(int(M) + 1, dtype=np.int64))


def replay_placements(ctx, args, n_poses, pose_blocks, progress):
    """Replay the shield over ``n_poses`` placements for the human sub-population held in ``ctx``.

    Identical machinery to ``simulate_robot_shield``: the level 1-3 bounding-sphere gate on the
    CPU, then either the cKDTree narrow phase (``run_pose_pure``) or the dense GPU kernel. Both
    backends are asked for the lightweight ``(traj_idx, motion_idx, unsafe)`` triples of every
    verified-but-contact pair, which is what gives per-window and per-placement resolution instead
    of the aggregate counts the shield simulation reports.

    On the GPU backend the gate runs for a whole ``GATE_BLOCK`` of placements at once
    (:func:`pose_active_sets_batched`). The per-pose gate used to dominate: at M ~ 1e2..1e3 the
    test is a few flops, so one Python iteration per placement cost more than the arithmetic, and
    the ~88 % of placements culled at level 1 are now accounted for in bulk rather than one
    counter dict each.

    ``pose_blocks`` is an iterable of [b,4] placement arrays (drawn lazily, see
    :func:`pose_block_factory`).

    Returns ``(counters, per_window, per_placement)``. ``per_window`` counts, per window, the
    (placement, trajectory) pairs that ended dangerous / in contact. ``per_placement`` holds
    ONLINE statistics only -- ``sup``/``argsup`` (running max of ``mean_i 1{dangerous(i, p)}``),
    ``k_dangerous`` (placements with >=1 such window, what the binomial placement bound needs), a
    coarse histogram, the first ``PLACEMENT_TOP_KEEP`` non-zero placements and the distribution of
    active motions per surviving placement -- so nothing here scales with the placement count.
    """
    st, M, n_traj = ctx.st, ctx.M, ctx.n_traj
    st.failure_triples = True          # cheap sibling of --save_failures: indices, no geometry
    P = int(n_poses)
    dang_w = np.zeros(M, dtype=np.int64)
    cont_w = np.zeros(M, dtype=np.int64)
    tot = dict.fromkeys(COUNTER_KEYS, 0)
    pl = new_placement_stats(M)

    def accumulate(p, cc):
        for key in tot:
            tot[key] += cc[key]
        trip = cc.get("failures") or []
        if not trip:
            return                     # nothing failed here -> every placement statistic stays 0
        mi = np.array([t[1] for t in trip], dtype=np.int64)
        un = np.array([t[2] for t in trip], dtype=bool)
        np.add.at(cont_w, mi, 1)
        np.add.at(dang_w, mi[un], 1)
        c_share = np.unique(mi).size / M
        d_share = np.unique(mi[un]).size / M
        pl["k_contact"] += 1
        pl["sup_contact"] = max(pl["sup_contact"], c_share)
        if d_share > 0.0:
            pl["k_dangerous"] += 1
            pl["hist"][min(int(d_share * PLACEMENT_HIST_BINS), PLACEMENT_HIST_BINS - 1)] += 1
            if d_share > pl["sup_dangerous"]:
                pl["sup_dangerous"], pl["argsup_dangerous"] = d_share, int(p)
        if len(pl["top"]) < PLACEMENT_TOP_KEEP:
            pl["top"].append((int(p), d_share, c_share))

    def postfix():
        return (f"dangerous={int(dang_w.sum()):,} contact={int(cont_w.sum()):,} "
                f"L1-culled={tot['n_poses_skipped']:,}")

    ev = None
    if args.backend == "gpu":
        from conformal_human_motion_prediction.examples.shield_gpu import GpuShieldEvaluator
        ev = GpuShieldEvaluator(st, n_traj, dtype=args.gpu_dtype, a_chunk=args.gpu_a_chunk,
                                capture_failures=True)
    p0 = 0
    for pblock in pose_blocks:
        if ev is not None:
            for b0 in range(0, len(pblock), GATE_BLOCK):
                sub = pblock[b0: b0 + GATE_BLOCK]
                skip, actives = pose_active_sets_batched(st, sub)
                n_sk = int(skip.sum())
                if n_sk:
                    bulk = _skipped_pose_counts(st, n_traj, n_sk)
                    for key in tot:
                        tot[key] += bulk[key]
                    progress.update(n_sk, postfix())
                alive = np.flatnonzero(~skip)
                for pose, ai, loc in zip(sub[alive], actives, alive):
                    pl["act_hist"][ai.size] += 1
                    accumulate(p0 + b0 + int(loc), ev.eval_active(pose, ai))
                    progress.update(1, postfix())
        elif args.num_workers > 1:
            srs._WORKER_ST = st        # set before forking: workers inherit it copy-on-write
            with mp.get_context("fork").Pool(min(args.num_workers, len(pblock))) as pool:
                for loc, cc in pool.imap_unordered(srs._pose_worker, list(enumerate(pblock))):
                    if not cc["n_poses_skipped"]:
                        pl["act_hist"][cc["active"]] += 1
                    accumulate(p0 + loc, cc)
                    progress.update(1, postfix())
        else:
            for loc, pose in enumerate(pblock):
                cc = run_pose_pure(st, pose)
                if not cc["n_poses_skipped"]:
                    pl["act_hist"][cc["active"]] += 1
                accumulate(p0 + loc, cc)
                progress.update(1, postfix())
        p0 += len(pblock)
    progress.close()
    assert p0 == P, f"placement stream yielded {p0} of {P} placements"
    return tot, dict(dangerous=dang_w, contact=cont_w), pl


def active_motion_summary(act_hist):
    """(n_survivors, mean, median, p95) of the per-surviving-placement active-motion count.

    Read off the exact histogram (one bin per possible count), so it costs O(M) memory instead of
    one entry per placement. Drives the ``--gpu_a_chunk`` hint: the GPU kernel pads every launch
    to ``a_chunk`` motions, so a chunk far above the mean is mostly wasted work, and far below it
    multiplies the per-launch overhead.
    """
    n = int(act_hist.sum())
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    counts = np.arange(act_hist.size)
    mean = float((counts * act_hist).sum() / n)
    cum = np.cumsum(act_hist)
    median = float(np.searchsorted(cum, 0.5 * n, side="left"))
    p95 = float(np.searchsorted(cum, 0.95 * n, side="left"))
    return n, mean, median, p95


def suggested_a_chunk(mean_active):
    """Nearest power of two to ~1.5x the mean active-motion count (the measured optimum).

    Nearest, not next-above: at M = 271 the mean is 87, 1.5x is 130, and the measured U-curve
    bottoms out at 128 (0.69 ms per active pose) rather than 256 (0.82 ms).
    """
    if not np.isfinite(mean_active) or mean_active <= 0:
        return 128
    return int(2 ** int(round(np.log2(max(8.0, 1.5 * mean_active)))))


def placement_upper_bound(k, P, confidence=0.95):
    """One-sided Clopper-Pearson upper limit on P(dangerous | placement), over placements.

    Unlike the shield simulation's test cycles, the placements really ARE i.i.d. draws (we drew
    them), so a binomial bound over them is legitimate. Collapsing a placement to the indicator
    "was ANY failing window dangerous here?" is conservative, because for every placement
    ``mean_i 1{dangerous(i, p)} <= 1{any dangerous at p}``; the bound on the mean of the right
    side therefore bounds the failure-averaged probability too. This is what carries the result
    when the brute force sees zero events and the point estimate is only "below the resolution".

    Returns the upper limit for ``k`` dangerous placements out of ``P``. Both are pooled totals
    when the replay was sharded -- the shards draw independent placement streams, so their counts
    simply add and the bound is computed once on the sum.
    """
    from scipy.stats import beta
    k, P = int(k), int(P)
    return 1.0 if k >= P else float(beta.ppf(confidence, k + 1, P - k))


def concentration(p_per_failure):
    """Share of ``sum_i P(d|f_i)`` carried by the top 1 / 3 / 10 failures (hostage diagnostic)."""
    s = float(p_per_failure.sum())
    if s <= 0.0:
        return {1: float("nan"), 3: float("nan"), 10: float("nan")}
    srt = np.sort(p_per_failure)[::-1]
    return {k: float(srt[:k].sum() / s) for k in (1, 3, 10)}


def bootstrap_mean(values, reps, seed):
    """Bootstrap the mean of ``values`` by resampling *which failures were observed*.

    Captures the sampling of the observed failure set only -- NOT between-subject variability
    (H36M has one subject per split) and not the placement integral (a modelling choice).
    """
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 0:
        return np.zeros(0)
    draws = rng.integers(0, n, size=(int(reps), n))
    return values[draws].mean(axis=1)


# --------------------------------------------------------------------------- reporting helpers


def pl_label(lambda_d):
    """ISO 13849-1 PL for a composed rate, with the zero-event case called out.

    ``lambda_d == 0`` means the placement replay saw no dangerous event at this resolution, not
    that the rate is zero; reading that as "PL e" would be exactly the over-claim this script
    exists to avoid.
    """
    if lambda_d <= 0.0:
        return ("not determined (0 dangerous events at this placement resolution -- use the "
                "placement-level bound below)")
    return pl_from_pfh(lambda_d)


def exposure_budget(lambda_d):
    """Eq. (4): the share of an operating hour of exposure that stays under the PL d line.

    A zero rate (nothing observed at this resolution) means the target is met at any exposure;
    ``inf`` keeps that distinguishable from "one full hour is fine but no more".
    """
    return PFH_D_TARGET / lambda_d if lambda_d > 0.0 else float("inf")


def fmt_budget(budget_h):
    """Exposure budget as 'fraction of an operating hour (s/h, min/h)'."""
    if not np.isfinite(budget_h):
        return "unbounded (no dangerous failure observed)"
    if budget_h >= 1.0:
        return f"{budget_h:.3g} h/h (>= full exposure: the target is met continuously)"
    return f"{budget_h:.3e} h/h = {3600.0 * budget_h:.3g} s/h = {60.0 * budget_h:.4g} min/h"


# --------------------------------------------------------------------------- shard parallelism


def failure_set_hash(idx):
    """Short digest of the evaluated failure-window indices.

    Every shard must price the SAME set F; the digest lets the launcher assert that instead of
    trusting that the shards happened to see the same results file and flags.
    """
    return hashlib.sha1(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()[:16]


def lemma_windows(args, flag):
    """The non-failure windows the lemma test replays -- seeded, so every shard picks the same."""
    non_fail = np.flatnonzero(~flag)
    n = min(int(args.verify_lemma), non_fail.size)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.random.default_rng(args.seed + 99).choice(non_fail, n, replace=False))


def build_failure_state(args, results=None):
    """Load the results (unless handed in), identify F, and build the shield state restricted to F.

    Shared by the single-process path, the launcher (which needs F for steps 2-3) and every shard
    worker, so all three derive the failure set with exactly the same code.

    Returns ``(ctx, results, fail_stats)``; ``fail_stats`` carries the full-window failure
    indicator, the escape margins, the (step, joint) of each worst violation and ``evaluated`` =
    the window indices the placement replay sees.
    """
    results_file = (args.results_file if os.path.isabs(args.results_file)
                    else os.path.join(root_dir, args.results_file))
    if results is None:
        print(f"Loading human results from {results_file} ...")
        with open(results_file, "rb") as f:
            results = cloudpickle.load(f)

    fail_stats = {}

    def select_failures(horizon_times, pred_c, pred_r, true_c, true_r):
        """Subset hook: identify F and record the lambda_f statistics of the full window set."""
        mx, step, joint = escape_margins(pred_c, pred_r, true_c, true_r)
        flag = mx > 0.0
        fail_stats.update(flag=flag, escape=mx, step=step, joint=joint,
                          dt=float(horizon_times[1] - horizon_times[0]))
        idx = np.flatnonzero(flag)
        if args.max_failures_eval is not None and idx.size > args.max_failures_eval:
            sel = np.random.default_rng(args.seed + 1).choice(idx, args.max_failures_eval,
                                                              replace=False)
            idx = np.sort(sel)
            print(f"  --max_failures_eval: replaying {idx.size} of the failures only")
        fail_stats["evaluated"] = idx
        return idx

    ctx = build_shield_state(args, results=results, subset=select_failures)
    return ctx, results, fail_stats


def shard_rng(seed, shards, shard_id):
    """Independent placement stream per shard (numpy's SeedSequence spawn, not seed arithmetic)."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(shards)[shard_id])


def save_shard(path, n_placements, tot, per_window, pl, M, fhash, lemma):
    """Write one shard's RAW counts (never a formatted report) for the launcher to pool."""
    np.savez(
        path,
        n_placements=np.int64(n_placements), M=np.int64(M), fail_hash=np.array(fhash),
        counters=np.array([tot[k] for k in COUNTER_KEYS], dtype=np.int64),
        dangerous_per_failure=per_window["dangerous"], contact_per_failure=per_window["contact"],
        k_dangerous=np.int64(pl["k_dangerous"]), k_contact=np.int64(pl["k_contact"]),
        sup_dangerous=np.float64(pl["sup_dangerous"]), sup_contact=np.float64(pl["sup_contact"]),
        argsup_dangerous=np.int64(pl["argsup_dangerous"]), hist=pl["hist"], act_hist=pl["act_hist"],
        top=np.asarray(pl["top"], dtype=np.float64).reshape(-1, 3),
        lemma=np.array([lemma["n"], lemma["cycles"], lemma["n_verified_contact"],
                        lemma["n_verified_unsafe"]], dtype=np.int64),
    )


def pool_shards(paths, M, fhash):
    """Pool raw shard counts into the same objects the single-process replay returns.

    Exact pooling rules: every count ADDS (the shards draw independent placement streams, so
    their placements are disjoint i.i.d. draws), the sup-over-placements is the MAX of the shard
    maxima, and nothing from steps 2-3 is touched -- the failure set, duty cycle, episode length
    and autocorrelation do not depend on placements and are computed once by the launcher.
    """
    tot = dict.fromkeys(COUNTER_KEYS, 0)
    per_window = dict(dangerous=np.zeros(M, np.int64), contact=np.zeros(M, np.int64))
    pl = new_placement_stats(M)
    lemma = dict(n=0, cycles=0, n_verified_contact=0, n_verified_unsafe=0)
    n_total = 0
    for path in paths:
        d = np.load(path, allow_pickle=False)
        if int(d["M"]) != M or str(d["fail_hash"]) != fhash:
            raise SystemExit(
                f"Shard {path} priced a different failure set (M={int(d['M'])}, "
                f"hash={str(d['fail_hash'])}) than the launcher (M={M}, hash={fhash}) -- "
                f"refusing to pool.")
        n_total += int(d["n_placements"])
        for k, v in zip(COUNTER_KEYS, d["counters"]):
            tot[k] += int(v)
        per_window["dangerous"] += d["dangerous_per_failure"]
        per_window["contact"] += d["contact_per_failure"]
        pl["k_dangerous"] += int(d["k_dangerous"])
        pl["k_contact"] += int(d["k_contact"])
        if float(d["sup_dangerous"]) > pl["sup_dangerous"]:
            pl["sup_dangerous"] = float(d["sup_dangerous"])
            # NB: shard-local placement index (each shard numbers its own share from 0).
            pl["argsup_dangerous"] = int(d["argsup_dangerous"])
        pl["sup_contact"] = max(pl["sup_contact"], float(d["sup_contact"]))
        pl["hist"] += d["hist"]
        pl["act_hist"] += d["act_hist"]
        pl["top"].extend(map(tuple, d["top"][: max(0, PLACEMENT_TOP_KEEP - len(pl["top"]))]))
        lem = d["lemma"]
        lemma["n"] = max(lemma["n"], int(lem[0]))          # same window set in every shard
        lemma["cycles"] += int(lem[1])                     # ... over disjoint placements
        lemma["n_verified_contact"] += int(lem[2])
        lemma["n_verified_unsafe"] += int(lem[3])
    return n_total, tot, per_window, pl, lemma


def strip_flags(argv, names):
    """Drop ``--flag value`` / ``--flag=value`` pairs from a raw argv list."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        base = a.split("=", 1)[0]
        if base in names:
            i += 1 if "=" in a else 2
            continue
        out.append(a)
        i += 1
    return out


def read_progress(path):
    """Placements done according to a shard's progress file (0 if it has not written one yet)."""
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, IndexError, ValueError):
        return 0


def run_shard_worker(args):
    """Shard worker: replay this shard's placements and write raw counts. No reporting."""
    ctx, results, fail_stats = build_failure_state(args)
    sel = lemma_windows(args, fail_stats["flag"])
    # Build both shield states up front, then drop the (GB-scale) results dict: 16 shard workers
    # each keeping it alive for the whole replay would be tens of GB held for nothing.
    ctx_l = build_shield_state(args, results=results, subset=sel) if sel.size else None
    del results
    n = args.num_robot_poses
    blocks = pose_block_factory(shard_rng(args.seed, args.shards, args.shard_id), n, args)
    print(f"shard {args.shard_id}/{args.shards}: M={ctx.M}, {n:,} placements")
    tot, per_window, pl = replay_placements(
        ctx, args, n, blocks(), _Progress(n, "placements", path=args.shard_progress))

    lemma = dict(n=0, cycles=0, n_verified_contact=0, n_verified_unsafe=0)
    if ctx_l is not None:
        # The lemma test has to run over the SAME placements, so each shard checks its own share.
        prog_l = _Progress(n, "lemma",
                           path=None if args.shard_progress is None else args.shard_progress + ".l")
        tot_l, _, _ = replay_placements(ctx_l, args, n, blocks(), prog_l)
        lemma = dict(n=int(sel.size), cycles=int(tot_l["total_pairs"]),
                     n_verified_contact=int(tot_l["n_verified_contact"]),
                     n_verified_unsafe=int(tot_l["n_verified_unsafe"]))
    save_shard(args.shard_out, n, tot, per_window, pl, ctx.M,
               failure_set_hash(fail_stats["evaluated"]), lemma)
    print(f"shard {args.shard_id}: wrote {args.shard_out}")


def launch_shards(args, n_poses, M, fhash, workdir):
    """Spawn ``args.shards`` worker processes over disjoint placement shares and pool their counts.

    Subprocesses, not ``fork``: this process has already initialised CUDA through JAX, and forking
    a CUDA context is undefined behaviour. Each shard gets its own log, its own placement seed
    stream and a ``floor``/``ceil`` share of ``n_poses`` that sums to exactly ``n_poses``.
    """
    shards = int(args.shards)
    sizes = [n_poses // shards + (1 if k < n_poses % shards else 0) for k in range(shards)]
    assert sum(sizes) == n_poses, (sizes, n_poses)
    passthrough = strip_flags(sys.argv[1:], LAUNCHER_ONLY_FLAGS)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    procs, outs, progs, logs = [], [], [], []
    print(f"\nLaunching {shards} shard(s) of {sizes[0]:,}(+/-1) placements each "
          f"(logs in {workdir}) ...")
    for k, n in enumerate(sizes):
        out = os.path.join(workdir, f"shard{k:03d}.npz")
        prog = os.path.join(workdir, f"shard{k:03d}.progress")
        log = os.path.join(workdir, f"shard{k:03d}.log")
        cmd = [sys.executable, "-u", "-m", MODULE, *passthrough,
               "--num_robot_poses", str(n), "--shards", str(shards), "--shard_id", str(k),
               "--shard_out", out, "--shard_progress", prog]
        lf = open(log, "w")
        procs.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
        outs.append(out)
        progs.append(prog)
        logs.append(log)
        if k + 1 < shards:
            time.sleep(SHARD_STAGGER_S)   # stagger the (GB-scale) results-file loads

    # One bar for all shards: the workers write progress files instead of their own bars.
    bar = tqdm(total=n_poses * (2 if args.verify_lemma > 0 else 1), desc=f"{shards} shards",
               unit="pose", dynamic_ncols=True, mininterval=2.0)
    while any(pr.poll() is None for pr, _ in procs):
        time.sleep(1.0)
        done = sum(read_progress(f) + read_progress(f + ".l") for f in progs)
        bar.update(max(0, done - bar.n))
    bar.update(max(0, bar.total - bar.n))     # a shard may exit before its last progress write
    bar.close()
    for (pr, lf), log in zip(procs, logs):
        lf.close()
        if pr.returncode != 0:
            with open(log) as f:                   # surface the traceback, not just the exit code
                print("".join(f.readlines()[-20:]))
            raise SystemExit(f"Shard failed (exit {pr.returncode}); log: {log}")
    return pool_shards(outs, M, fhash)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- inputs (names/defaults mirror simulate_robot_shield) ----
    parser.add_argument("--results_file", type=str,
                        default="results/motion_prediction/motion_prediction_results_validation.cloudpickle")
    parser.add_argument("--conformal_calibrator", type=str,
                        default="models/motion_prediction/conformal_calibration/conformal_calibrator.npz",
                        help="Conditional-conformal calibrator .npz forming the predicted human set "
                             "radius. '' / 'none' (or a missing file) falls back to the affine "
                             "calibration; --human_set sara ignores it entirely.")
    parser.add_argument("--robot_csv", type=str,
                        default="datasets/robot_reachable_sets/ablation_conformal_prediction_sets_panda_4ms.csv")
    parser.add_argument("--config", type=str, default="h36m", choices=["h36m", "rgbd_yolo"])
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Deployment cadence f_deploy (Hz): predictions issued per second. "
                             "Also the window spacing of the recorded data, hence the horizon dt.")
    parser.add_argument("--human_set", type=str, default="conformal", choices=["conformal", "sara"],
                        help="Predicted human occupancy model whose failures are being priced.")
    parser.add_argument("--mask_ood", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ood_threshold", type=float, default=None,
                        help="Override OOD_THRESHOLD (head-specific; see simulate_robot_shield).")
    parser.add_argument("--mask_too_fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot_origin", type=str, default="0,0,0")
    parser.add_argument("--overapprox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion_group_size", type=int, default=50)
    parser.add_argument("--max_human_samples", type=int, default=None,
                        help="Cap eligible windows (random subset) before failures are identified. "
                             "Only for smoke runs: it destroys the dataset-order episode structure "
                             "that lambda_f's declustering is measured from.")
    parser.add_argument("--max_robot_timesteps", type=int, default=None)
    parser.add_argument("--robot_stride", type=int, default=25,
                        help="Evaluate every Nth monitored trajectory (25 on the 4 ms grid = 100 ms "
                             "apart).")
    parser.add_argument("--t_cycle", type=float, default=None,
                        help="Safety-function cycle time (s); default: the robot planning grid.")
    # ---- placement integral ----
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpu_dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--gpu_a_chunk", type=int, default=128,
                        help="Active motions per GPU kernel launch. Every launch is padded to "
                             "this width, so the throughput optimum is ~1.5x the mean active-"
                             "motion count at the nearest power of two (128 measured best at "
                             "M = 271 failing windows); the run prints the observed counts and a "
                             "suggestion, which must be re-tuned when the failure count changes.")
    parser.add_argument("--shards", type=int, default=1,
                        help="Split the placement integral over N worker processes (subprocesses, "
                             "each with its own independent placement stream and GPU context). "
                             "The launcher pools their raw counts and reports once. 1 = "
                             "single-process, which reproduces the un-sharded numbers exactly.")
    parser.add_argument("--shard_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_out", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_progress", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_dir", type=str, default=None,
                        help="Directory for shard logs / raw-count npz files (default: a temp "
                             "dir that is kept, so the per-shard logs survive the run).")
    parser.add_argument("--num_robot_poses", type=int, default=1000,
                        help="Random robot base placements the failing windows are replayed "
                             "against. Sets the resolution of both the average- and the "
                             "worst-placement estimate.")
    parser.add_argument("--pose_radius", type=float, default=10.0,
                        help="Radius (m) of the disk in which (x, y) base positions are sampled.")
    parser.add_argument("--pose_z_offset", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Processes for the CPU backend (poses are independent).")
    parser.add_argument("--seed", type=int, default=0)
    # ---- this script's own knobs ----
    parser.add_argument("--failure_stride", type=int, default=20,
                        help="Sub-sample every Nth window for p_hat, so the observations entering "
                             "the binomial interval are not the same overlapping prediction.")
    parser.add_argument("--epsilon_d", type=float, default=1e-5,
                        help="TOTAL confidence budget of the PFH_D bound: it holds with "
                             "confidence 1 - epsilon_d. Split symmetrically over the two factors, "
                             "eps_F = eps_(D|F) = 1 - sqrt(1 - epsilon_d). Default 1e-5 = "
                             "99.999%% joint confidence.")
    parser.add_argument("--l_corr", type=float, default=None,
                        help="Override the integrated autocorrelation time used to deflate the "
                             "window count to an effective sample size for the interval on P(F). "
                             "Default: measured from the failure indicator by Sokal windowing. "
                             "Raise it for a sensitivity check; it does not affect the point "
                             "estimate k_F / N_F.")
    parser.add_argument("--max_failures_eval", type=int, default=None,
                        help="Cap the failure set the placement replay uses (smoke runs). The "
                             "lambda_f side still uses every window.")
    parser.add_argument("--verify_lemma", type=int, default=0,
                        help="Replay N randomly chosen NON-failure windows and assert that zero "
                             "verified-AND-contact events occur -- the empirical test of the lemma "
                             "'verified AND contact => prediction failure' the factorisation rests "
                             "on. 0 = skip.")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap replications over the observed failure set.")
    parser.add_argument("--save_per_failure", type=str, default=None,
                        help="Path to a .npz: per-failure escape size, (step, joint), P(d|f_i) and "
                             "the results-file row index, for later analysis.")
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Append this run's summary as one row (header-migrating, so a CSV "
                             "written before a column was added stays aligned).")
    # build_shield_state also consults the --save_failures geometry dump, which this script does
    # not expose (it keeps only the index triples).
    parser.set_defaults(save_failures=None, max_failures=0)
    args = parser.parse_args()
    if args.shard_id is not None:
        # Hidden worker mode: no report, just raw counts for the launcher (see launch_shards).
        run_shard_worker(args)
        return

    # ------------------------------------------------------------------ load + identify failures
    ctx, results, fail_stats = build_failure_state(args)
    flag = fail_stats["flag"]
    escape = fail_stats["escape"]
    n_windows = int(flag.size)
    n_failures = int(flag.sum())
    if n_failures == 0:
        raise SystemExit("No prediction failures in this results file -- nothing to price. "
                         "(lambda_f = 0 with this window set; the rate is bounded only by the "
                         "binomial upper limit, which this script does not model.)")

    # p_hat on a strided sub-sample: adjacent windows are the same physical event seen again.
    stride = max(1, args.failure_stride)
    sub = flag[::stride]
    n_sub = int(sub.size)
    k_sub = int(sub.sum())
    # P(F) point estimate over EVERY window, i.e. k_F / |Z_test| -- the estimator the PFH_D
    # derivation uses. --failure_stride is only a diagnostic sub-sample printed beside it.
    p_hat = n_failures / n_windows if n_windows else float("nan")
    p_hat_sub = k_sub / n_sub if n_sub else float("nan")
    # Dependence enters the INTERVAL, not the estimate: deflate the window count to an effective
    # sample size by two measured factors.
    #   L_corr -- integrated autocorrelation time of the failure indicator. Adjacent windows share
    #             K_I - 1 input poses and re-observe the same physical escape.
    #   2      -- the loader emits both 50->25 fps phase offsets (h36m_motion_prediction.py: the
    #             `for offset in [0, 1]` loop is OUTER), so every motion is in the series twice:
    #             20 ms apart in time, one contiguous block apart in row order -- and therefore
    #             NOT removed by sub-sampling at any stride.
    L_corr = float(args.l_corr) if args.l_corr is not None else integrated_autocorr_time(flag)
    n_eff = n_windows / (2.0 * L_corr)
    k_eff = p_hat * n_eff
    p_lo, p_hi = clopper_pearson(k_eff, n_eff)          # diagnostic 95% two-sided
    eps_f, eps_dgf = split_confidence(args.epsilon_d)
    p_f_up = clopper_pearson_upper(k_eff, n_eff, eps_f)  # the bound that enters PFH_D

    episodes = run_lengths(flag)
    L_meas = float(episodes.mean()) if episodes.size else 1.0
    acf = autocorrelation(flag)
    f_deploy = float(args.fps)
    # N_h -- verification cycles per operating hour. The safety function is evaluated once per
    # SARA-shield planning cycle t_cycle, NOT once per prediction: each prediction is reused for
    # 1 / (f_deploy * t_cycle) = K_P consecutive cycles and every one of them inherits the failure
    # status of the prediction it runs on, so P(F) is identical at both rates and only this
    # conversion factor changes. Counting each cycle separately over-approximates a contact that
    # persists across several cycles, which keeps the rate an upper bound.
    # (Using f_deploy here instead understates the rate by exactly K_P.)
    N_h = 3600.0 / float(ctx.t_cycle)
    # The rate is the WINDOW rate, never an episode rate: P(D|F) is measured per window, so the
    # two must be paired per window. Dividing by the episode length would price a whole episode as
    # a single window and understate PFH_D; keeping the window rate over-counts a contact that
    # persists across an episode, which is the safe direction (union bound). The episode
    # statistics below are retained as the diagnostic evidence behind L_corr, not as a rate.
    lam_f_point = p_hat * N_h
    lam_f_up = p_f_up * N_h
    p_f_i = N_h / n_windows                               # 1/h attributed to one observed failure

    print("\n=============== Prediction-failure set F (step 2/3) ===============")
    print(f"Predicted human occupancy model : {args.human_set}"
          + (" (ISO 13855 constant-velocity set)" if args.human_set == "sara"
             else (" (conditional-conformal set)" if ctx.calibrator is not None
                   else " (affine/raw covariance set)")))
    print(f"Windows (after OOD/too-fast filtering) : {n_windows:,}")
    print(f"Prediction failures (truth escapes the predicted set at some horizon step, joint): "
          f"{n_failures:,}  ({100.0 * n_failures / n_windows:.4f}% of windows)")
    esc_f = escape[flag]
    print(f"  escape size (m): median {np.median(esc_f):.4f}  mean {esc_f.mean():.4f}  "
          f"p95 {np.quantile(esc_f, 0.95):.4f}  max {esc_f.max():.4f}")
    st_f, jt_f = fail_stats["step"][flag], fail_stats["joint"][flag]
    print(f"  worst-violation horizon step: median {int(np.median(st_f))} of {ctx.S - 1} "
          f"(= {float(np.median(st_f)) * fail_stats['dt'] * 1000:.0f} ms); "
          f"joints hit (idx:count): "
          + ", ".join(f"{j}:{c}" for j, c in zip(*np.unique(jt_f, return_counts=True))))
    print(f"P_hat(F) = k_F / N_F = {p_hat:.4e}   95% Clopper-Pearson [{p_lo:.4e}, {p_hi:.4e}]")
    print(f"  integrated autocorrelation time L_corr = {L_corr:.2f} windows "
          f"(= {1000.0 * L_corr / f_deploy:.0f} ms)"
          + ("  <- --l_corr override" if args.l_corr is not None else "  (Sokal windowing)"))
    print(f"  effective sample size n_eff = N_F / (2 L_corr) = {n_eff:,.0f}, k_eff = {k_eff:,.1f}"
          f"   (the 2 = both 25 fps phase offsets)")
    # A large gap between the strided and the all-window rate warns that the stride landed on an
    # unlucky phase; the stride no longer drives anything that is reported.
    print(f"  diagnostic: every {stride}th window -> N_sub = {n_sub:,}, k = {k_sub:,}, "
          f"p_hat_sub = {p_hat_sub:.4e} "
          f"({p_hat_sub / p_hat if p_hat else float('nan'):.2f}x the all-window rate)")
    print(f"Failure episodes (runs of consecutive failing windows, dataset order): "
          f"{episodes.size:,}")
    print(f"  length L: mean {L_meas:.3f} windows (= {1000.0 * L_meas / f_deploy:.0f} ms), "
          f"median {np.median(episodes) if episodes.size else float('nan'):.1f}, "
          f"max {episodes.max() if episodes.size else 0}")
    print("  indicator autocorrelation: "
          + ", ".join(f"lag {L}: {v:+.3f}" for L, v in acf.items()))
    print(f"N_h = 3600 s / t_cycle = {N_h:,.0f} verification cycles per operating hour "
          f"(t_cycle = {ctx.t_cycle:g} s = {1.0 / (f_deploy * float(ctx.t_cycle)):.0f} cycles "
          f"per prediction)")
    print(f"P(F) upper bound at 1-eps_F = {1.0 - eps_f:.8g}: {p_f_up:.4e}")
    print(f"  -> N_h * P_hat(F) = {lam_f_point:.4g} 1/h (point)   "
          f"N_h * P_up(F) = {lam_f_up:.4g} 1/h (bound)")
    print(f"Rate attributed to one observed failure: P(f_i) = {p_f_i:.4g} 1/h")
    print("===================================================================")

    # ------------------------------------------------------------------ placement integral
    n_traj, M_f = ctx.n_traj, ctx.M
    P = int(args.num_robot_poses)
    fhash = failure_set_hash(fail_stats["evaluated"])
    print(f"\nReplaying {M_f:,} failing window(s) x {n_traj} monitored trajectory(ies) against "
          f"{P:,} random robot placement(s) on the {args.backend} backend "
          f"(xy area-uniform in a {args.pose_radius:g} m disk, z +/-{args.pose_z_offset:g} m, "
          f"yaw +/-pi)" + (f", split over {args.shards} shards" if args.shards > 1 else "")
          + f" [failure set {fhash}] ...")
    shard_lemma, pose_blocks = None, None
    args.shards = max(1, min(args.shards, P))   # no point spawning workers with no placements
    if args.shards > 1:
        workdir = args.shard_dir or tempfile.mkdtemp(prefix="shield_shards_")
        os.makedirs(workdir, exist_ok=True)
        n_done, tot, per_window, per_placement, shard_lemma = launch_shards(
            args, P, M_f, fhash, workdir)
        if n_done != P:
            raise SystemExit(f"Shards covered {n_done:,} placements, expected {P:,}")
    else:
        # Single process: the placements come from ctx.rng exactly as the un-sharded script drew
        # them, so a fixed seed reproduces earlier runs bit-for-bit.
        pose_blocks = pose_block_factory(ctx.rng, P, args)
        tot, per_window, per_placement = replay_placements(
            ctx, args, P, pose_blocks(), _Progress(P, "placements"))
    n_pl_traj = P * n_traj
    p_d = per_window["dangerous"] / n_pl_traj              # P(d | f_i)
    p_c = per_window["contact"] / n_pl_traj                # P(contact | f_i), any speed
    E_d, E_c = float(p_d.mean()), float(p_c.mean())
    sup_d = float(per_placement["sup_dangerous"])
    sup_c = float(per_placement["sup_contact"])
    conc = concentration(p_d)
    l3_share = tot["active"] / (P * M_f) if M_f else float("nan")
    n_surv, act_mean, act_med, act_p95 = active_motion_summary(per_placement["act_hist"])

    print("\n============= P(d | f): placement replay of the failures (step 4) =============")
    print(f"Test cycles in the replay (placement x trajectory x failure): {tot['total_pairs']:,}")
    print(f"Placements culled at level 1 (robot never reaches any failing human): "
          f"{tot['n_poses_skipped']:,}/{P:,} "
          f"({100.0 * tot['n_poses_skipped'] / P:.2f}%)")
    print(f"Level-3 active (placement, failure) pairs: {tot['active']:,}/{P * M_f:,} "
          f"({100.0 * l3_share:.3f}%) -- the human is inside the robot's swept workspace")
    print(f"Verified {tot['n_verified']:,} | contact {tot['n_contact']:,} | "
          f"unsafe contact {tot['n_unsafe']:,}")
    if args.backend == "gpu":
        # The GPU kernel pads every launch to --gpu_a_chunk motions, so the throughput optimum
        # tracks the *mean* active count; it has to be re-tuned whenever the failure count moves
        # (e.g. ~477 failing windows on the test split instead of ~271 here).
        sug = suggested_a_chunk(act_mean)
        print(f"Active motions per surviving placement ({n_surv:,} survivors): median "
              f"{act_med:.0f}, mean {act_mean:.1f}, p95 {act_p95:.0f} -> --gpu_a_chunk {sug} "
              f"(~1.5x mean, nearest power of two; running with {args.gpu_a_chunk})")
    print(f"Dangerous (verified AND unsafe contact, link speed > {ctx.v_robot} m/s): "
          f"{tot['n_verified_unsafe']:,}   [in {int((p_d > 0).sum()):,} of {M_f:,} failures]")
    print(f"Secondary (verified AND contact, any speed)  : {tot['n_verified_contact']:,}   "
          f"[in {int((p_c > 0).sum()):,} of {M_f:,} failures]")
    print("-------------------------------------------------------------------------------")
    print(f"E_f[P(d|f)]  average placement : {E_d:.4e}"
          f"   (secondary, any contact: {E_c:.4e})")
    print(f"sup_p mean_i 1{{dangerous}}      : {sup_d:.4e}"
          f"   (secondary, any contact: {sup_c:.4e})")
    print("  sup is a MAX OVER THE SAMPLED PLACEMENT GRID -> a lower bound on the true sup; it "
          "grows with --num_robot_poses.")
    print(f"Concentration of sum_i P(d|f_i): top 1 = {100 * conc[1]:.1f}%, "
          f"top 3 = {100 * conc[3]:.1f}%, top 10 = {100 * conc[10]:.1f}%")
    if np.isfinite(conc[3]) and conc[3] > 0.5:
        print("  !! WARNING: the top 3 failures carry >50% of the risk -- this estimate is hostage")
        print("  !! to a handful of events. Treat E_f[P(d|f)] as indicative only and read the")
        print("  !! bootstrap interval below (it will be correspondingly wide).")
    print("=" * 79)

    # ------------------------------------------------------------------ compose + invert
    # PFH_D <= N_h * P_up(F) * P_up(D|F), each factor a one-sided Clopper-Pearson limit, holding
    # jointly at (1-eps_F)(1-eps_{D|F}) = 1-eps_D. The PRIMARY dangerous event is verified-AND-
    # contact at ANY speed (matching the shield's verification condition); the speed-gated variant
    # is a diagnostic.
    k_pl = int(per_placement["k_contact"])       # placements with >=1 verified-AND-contact window
    k_pl_unsafe = int(per_placement["k_dangerous"])
    # Collapsing a placement to "was ANY failing window dangerous here?" is what makes this bound
    # legitimate: one placement is applied to all M failing windows x n_traj trajectories, so the
    # trials within a placement are a CLUSTER, not independent draws, and a binomial bound over
    # M*n_traj*P would be anti-conservative. Over placements, which we drew i.i.d., it is exact --
    # and it bounds the failure-averaged probability because for every placement
    # mean_i 1{dangerous(i,p)} <= 1{any dangerous at p}.
    p_dgf_up = placement_upper_bound(k_pl, P, confidence=1.0 - eps_dgf)
    p_dgf_up_unsafe = placement_upper_bound(k_pl_unsafe, P, confidence=1.0 - eps_dgf)
    pfh_up = lam_f_up * p_dgf_up
    budget_up = exposure_budget(pfh_up)
    res_floor = 1.0 / n_pl_traj

    print("\n================== PFH_D = N_h * P(F) * P(D|F) (step 5) ==================")
    print(f"N_h                      = {N_h:,.0f} verification cycles per operating hour")
    print(f"P_up(F)     [1-eps_F     = {1.0 - eps_f:.8g}] = {p_f_up:.4e}   "
          f"(point {p_hat:.4e}, n_eff {n_eff:,.0f})")
    print(f"P_up(D|F)   [1-eps_(D|F) = {1.0 - eps_dgf:.8g}] = {p_dgf_up:.4e}   "
          f"({k_pl:,} of {P:,} placements had >=1 verified-AND-contact window)")
    print("-" * 118)
    print(f">>> PFH_D <= {pfh_up:.4e} 1/h   at joint confidence "
          f"{100.0 * (1.0 - args.epsilon_d):.6g}%   -> PL {pl_label(pfh_up)}")
    print(f">>> Exposure budget = {PFH_D_TARGET:g} / PFH_D = {fmt_budget(budget_up)}")
    print("    (the share of an operating hour with a human in the robot's workspace that keeps "
          "PFH_D under the PL d line)")
    print("-" * 118)
    print("Diagnostics (NOT the reported bound):")
    print(f"  speed-gated variant (verified AND unsafe contact, > {ctx.v_robot} m/s): "
          f"P_up(D|F) = {p_dgf_up_unsafe:.4e} -> PFH_D <= {lam_f_up * p_dgf_up_unsafe:.4e} 1/h, "
          f"budget {fmt_budget(exposure_budget(lam_f_up * p_dgf_up_unsafe))}")
    print(f"  point estimates: E_f[P(d|f)] = {E_c:.4e} (any speed) / {E_d:.4e} (speed-gated); "
          f"N_h * P_hat(F) * E_c = {lam_f_point * E_c:.4e} 1/h")
    print(f"  sup over the sampled placement grid: {sup_c:.4e} (any speed) / {sup_d:.4e} "
          f"(speed-gated) -- a LOWER bound on the true sup, grows with --num_robot_poses")
    print(f"  failure episodes: {episodes.size:,} runs, mean length {L_meas:.2f} windows -- "
          f"evidence behind L_corr = {L_corr:.2f}, NOT used as a rate")
    boot = bootstrap_mean(p_c, args.bootstrap, args.seed + 7)
    b_lo, b_hi = float("nan"), float("nan")
    if boot.size:
        b_lo, b_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        print(f"  bootstrap over WHICH failures were observed ({args.bootstrap:,} reps): "
              f"E_f[P(d|f)] 95% [{b_lo:.4e}, {b_hi:.4e}]")
        print("    (a different uncertainty from the two composed above -- the empirical failure "
              "mixture -- so it is not folded into the bound)")
    print(f"  Monte-Carlo resolution: smallest non-zero P(d|f_i) the grid can express is "
          f"1/(placements x trajectories) = {res_floor:.3e}"
          + ("  <- the point estimates above are 0, i.e. BELOW RESOLUTION; the bound is the "
             "usable number and tightens as 1/--num_robot_poses" if E_c == 0.0 else ""))
    print("-" * 118)
    print("Caveats: the bootstrap captures the sampling of WHICH failures were observed, but NOT")
    print("between-subject variability -- H36M has ONE subject per split (train = S1,S6,S7,S8,S9 /")
    print("validation = S11 / test = S5), so nothing here bounds subject-to-subject spread. The")
    print("placement integral's statistical error IS carried, by the binomial bound over the")
    print("i.i.d. placements; what it cannot capture is the choice of placement distribution.")
    print("=" * 118)

    # ------------------------------------------------------------------ step 6: the lemma test
    lemma = None
    if args.verify_lemma > 0:
        sel = lemma_windows(args, flag)
        n_lem = int(sel.size)
        print(f"\n--- Lemma test: replaying {n_lem:,} NON-failure window(s) over the same "
              f"{P:,} placement(s) ---")
        print("Expectation: ZERO verified-AND-contact events. 'verified AND contact => the truth "
              "left the predicted set'")
        print("is what makes the factorisation exact; if it fires, the failure definition misses "
              "something the shield can be fooled by.")
        per_window_l, ctx_l = None, None
        if shard_lemma is not None:
            # Each shard ran the same non-failure windows over its own placement share; the
            # violation counts add, but the per-window breakdown is not pooled.
            lemma = shard_lemma
        else:
            ctx_l = build_shield_state(args, results=results, subset=sel)
            tot_l, per_window_l, _ = replay_placements(ctx_l, args, P, pose_blocks(),
                                                       _Progress(P, "lemma"))
            lemma = dict(n=n_lem, cycles=int(tot_l["total_pairs"]),
                         n_verified_contact=int(tot_l["n_verified_contact"]),
                         n_verified_unsafe=int(tot_l["n_verified_unsafe"]))
        n_bad = int(lemma["n_verified_contact"])
        if n_bad == 0:
            print(f"LEMMA HOLDS: 0 verified-AND-contact in {lemma['cycles']:,} test cycles "
                  f"over {n_lem:,} non-failure windows.")
        else:
            print("!" * 100)
            print(f"!! LEMMA VIOLATED: {n_bad:,} verified-AND-contact event(s) "
                  f"({lemma['n_verified_unsafe']:,} of them unsafe) on NON-failure windows!")
            if per_window_l is None:
                print("!! Per-window detail is not pooled across shards -- re-run the same seed "
                      "with --shards 1 to locate the offending windows.")
            else:
                print("!! Offending windows (sub-sample index / results-file row / escape "
                      "margin m):")
                for b in np.flatnonzero(per_window_l["contact"] > 0)[:20]:
                    print(f"!!   sub {int(b)} -> results row {int(ctx_l.keep_idx[b])}, "
                          f"escape {escape[sel[b]]:+.4f} m, "
                          f"{int(per_window_l['contact'][b]):,} contact cycles")
            print("!! => the escape test does NOT capture everything the shield can be fooled by, "
                  "so lambda_d above is NOT an upper bound. Do not report it.")
            print("!" * 100)

    # ------------------------------------------------------------------ artefacts
    if args.save_per_failure:
        out = (args.save_per_failure if os.path.isabs(args.save_per_failure)
               else os.path.join(root_dir, args.save_per_failure))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        ev_idx = fail_stats["evaluated"]   # window indices, aligned with p_d / per_window
        np.savez(out, window_idx=ev_idx, results_row=ctx.keep_idx, escape_m=escape[ev_idx],
                 escape_step=fail_stats["step"][ev_idx],
                 escape_joint=fail_stats["joint"][ev_idx],
                 p_d_given_f=p_d, p_contact_given_f=p_c,
                 n_dangerous=per_window["dangerous"], n_contact=per_window["contact"],
                 n_placement_traj=np.int64(n_pl_traj), n_placements=np.int64(P),
                 placement_hist=per_placement["hist"],
                 placement_act_hist=per_placement["act_hist"],
                 placement_top=np.asarray(per_placement["top"], dtype=np.float64).reshape(-1, 3),
                 placement_sup_dangerous=np.float64(sup_d),
                 placement_sup_contact=np.float64(sup_c),
                 placement_argsup_dangerous=np.int64(per_placement["argsup_dangerous"]),
                 placement_k_dangerous=np.int64(per_placement["k_dangerous"]),
                 placement_k_contact=np.int64(per_placement["k_contact"]))
        print(f"\nSaved per-failure detail to {out} (np.load(path))")

    if args.results_csv:
        used_cal = None if args.human_set == "sara" else ctx.calibrator
        if args.human_set == "sara":
            set_kind = "sara"
        elif used_cal is None:
            set_kind = "affine" if args.calibrate else "raw"
        else:
            # "max_conformal" / "uncalibrated" are what conformal_results_common.
            # shield_method_key keys the ablation rows on; without the mode mapping the alpha_max
            # and no-calibration rows are indistinguishable from the conditional one and collapse
            # onto the same table row. Keep in sync with simulate_robot_shield.
            set_kind = {"max": "max_conformal", "uncalibrated": "uncalibrated"}.get(
                used_cal.get("mode"), "conditional_conformal")
        row = dict(
            results_file=os.path.basename(args.results_file), human_set=args.human_set,
            set_kind=set_kind,
            set_likelihood=(float(used_cal["level"]) if used_cal is not None
                            else float(ctx.set_likelihood)),
            mask_ood=args.mask_ood, mask_too_fast=args.mask_too_fast, backend=args.backend,
            seed=args.seed, fps=f_deploy, failure_stride=stride, robot_stride=args.robot_stride,
            n_trajectories=n_traj, num_robot_poses=P, shards=args.shards,
            gpu_a_chunk=args.gpu_a_chunk, pose_radius=args.pose_radius,
            pose_z_offset=args.pose_z_offset,
            n_windows=n_windows, n_failures=n_failures, n_failures_evaluated=M_f,
            n_sub=n_sub, k_sub=k_sub, p_hat_sub=p_hat_sub,
            p_hat=p_hat, p_hat_ci_lo=p_lo, p_hat_ci_hi=p_hi,
            l_corr=L_corr, l_corr_source=("override" if args.l_corr is not None else "measured"),
            n_eff=n_eff, N_h=N_h,
            episode_len_mean=L_meas,
            episode_len_median=float(np.median(episodes)) if episodes.size else float("nan"),
            episode_len_max=int(episodes.max()) if episodes.size else 0,
            n_episodes=int(episodes.size),
            escape_median_m=float(np.median(esc_f)), escape_max_m=float(esc_f.max()),
            epsilon_d=args.epsilon_d, epsilon_f=eps_f, epsilon_d_given_f=eps_dgf,
            p_f_upper=p_f_up, lambda_f_point=lam_f_point, lambda_f_upper=lam_f_up,
            p_d_given_f_upper=p_dgf_up, p_d_given_f_upper_unsafe=p_dgf_up_unsafe,
            pfh_d_upper=pfh_up, pl=pl_label(pfh_up),
            exposure_budget_h=budget_up, exposure_budget_s_per_h=3600.0 * budget_up,
            placement_k_contact=k_pl, placement_k_dangerous=k_pl_unsafe, p_f_i=p_f_i,
            replay_cycles=tot["total_pairs"], n_poses_skipped=tot["n_poses_skipped"],
            pct_poses_skipped=100.0 * tot["n_poses_skipped"] / P,
            n_l3_active=tot["active"], pct_l3_active=100.0 * l3_share,
            n_dangerous=tot["n_verified_unsafe"], n_verified_contact=tot["n_verified_contact"],
            E_f_P_d_given_f=E_d, E_f_P_contact_given_f=E_c,
            E_f_ci_lo=b_lo, E_f_ci_hi=b_hi,
            sup_P_d_given_f=sup_d, sup_P_contact_given_f=sup_c,
            conc_top1=conc[1], conc_top3=conc[3], conc_top10=conc[10],
            resolution_floor_P_d_given_f=res_floor,
        )
        row["lemma_n_windows"] = lemma["n"] if lemma else 0
        row["lemma_n_verified_contact"] = lemma["n_verified_contact"] if lemma else -1
        out_csv = (args.results_csv if os.path.isabs(args.results_csv)
                   else os.path.join(root_dir, args.results_csv))
        write_results_csv(out_csv, row, fieldnames=list(row.keys()))
        print(f"\nAppended run summary to {out_csv}")


if __name__ == "__main__":
    main()
