"""Conditional (Mondrian/CQR-style) conformal calibration of the predicted spherical set radius.

Motivation (see evaluate_covariance_failures.py for the diagnosis): the conformal set under-covers
in a way that the global affine calibration (COV_CALIBRATION_CT/IT/FACTORS) cannot fix, because the
under-coverage is *conditional*:
  (M1) at HIGH last-input uncertainty the model's set saturates and is overconfident (arms/wrists);
  (M2) some joints have a heavy prediction-error tail at LOW input uncertainty (legs/ankles).

A global multiplicative/affine factor can't add input-dependence (M1) or reshape a per-joint tail
(M2) without ballooning the well-behaved majority (and the learned r_model is deliberately << the
SARA reach bound V_HUMAN_ISO*t, so a velocity-bound floor is off the table).

This module replaces the affine calibration with a CONDITIONAL conformal recalibration of the set
radius, conditioned on (input-uncertainty bin x joint x horizon frame):

    r_cal = max(r_model + q_hat(input_unc_bin, joint, frame), 0)

where q_hat is the split-conformal additive quantile (CQR residual e = ||pred - true|| - r_model) at
level SET_LIKELIHOOD, computed per group on a held-out CALIBRATION split. Mondrian grouping gives
per-group (conditional) coverage where the group is dense; sparse groups fall back up a hierarchy
(joint,frame,bin) -> (joint,bin) -> (bin) -> global, and an optional isotonic (PAVA) smoothing makes
q_hat monotone in input-uncertainty so the data-starved high-uncertainty tail extrapolates sanely.
Because q_hat can be negative, OVER-covered strata are tightened (better availability) while
UNDER-covered strata are inflated -- the point is to spend volume only where coverage is missing.

Honest split discipline: fit on a calibration split, report on a disjoint test split (default:
split the validation results 50/50 by sample; or pass --calib_file/--test_file explicitly).

Run::

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu .venv/bin/python -m \
        conformal_human_motion_prediction.motion_prediction.conformal_calibration \
        --results_file results/motion_prediction/motion_prediction_results_validation.cloudpickle
"""
import argparse
import os
from pathlib import Path

import cloudpickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conformal_human_motion_prediction.motion_prediction.inference_helper import calibrate_covariance_matrices
from conformal_human_motion_prediction.utils.eval_utils import (
    compute_sara_predictions,
    convert_covariance_matrices_to_set,
)
from conformal_human_motion_prediction.pose_estimation.h36m_settings import JOINT_NAMES_13

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


# --------------------------------------------------------------------------- conformal primitives


def conformal_add_quantile(scores, level):
    """Split-conformal additive quantile: smallest q with empirical coverage >= level (finite sample).

    Returns (q, n). q is np.inf when the group is too small to reach ``level`` (k = ceil((n+1)*level)
    exceeds n) -- callers treat that as "insufficient data" and fall back to a coarser group.
    """
    n = scores.size
    if n == 0:
        return np.nan, 0
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return np.inf, n
    return float(np.partition(scores, k - 1)[k - 1]), n


def grouped_quantiles(score, codes, n_codes, level):
    """Per-group conformal additive quantile. ``codes`` in [0, n_codes); returns (q[n_codes], n[...])."""
    q = np.full(n_codes, np.nan)
    cnt = np.zeros(n_codes, dtype=np.int64)
    if score.size == 0:
        return q, cnt
    order = np.argsort(codes, kind="stable")
    sc, ss = codes[order], score[order]
    uniq, start = np.unique(sc, return_index=True)
    bounds = np.append(start, sc.size)
    for i, g in enumerate(uniq):
        q[g], cnt[g] = conformal_add_quantile(ss[bounds[i]:bounds[i + 1]], level)
    return q, cnt


def pava_nondecreasing(y, w):
    """Weighted isotonic (non-decreasing) regression via pool-adjacent-violators. Ignores NaN weights."""
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            wn = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / max(wn, 1e-12)); wts.append(wn); cnts.append(c1 + c2)
    out, pos = np.empty(len(y)), 0
    for v, c in zip(vals, cnts):
        out[pos:pos + c] = v; pos += c
    return out


# --------------------------------------------------------------------------- calibrator


def fit_calibrator(error, r_model, input_unc, joint_idx, frame_idx, J, T,
                   level, n_bins, n_min, monotone, tail_edges=(0.3, 0.5, 0.75)):
    """Fit the conditional conformal grid q_grid[J,T,B] from flat calibration arrays (all valid).

    error, r_model in mm; input_unc in m. ``n_bins`` equal-mass quantile bins cover the bulk; the
    explicit ``tail_edges`` (m) add dedicated high-input-uncertainty bins so the rare but
    safety-critical tail (where the bulk quantile bins would otherwise lump it) is calibrated on its
    own (via the joint/frame-pooled fallback). Returns dict with bin_edges, q_grid, n_grid, meta.
    """
    qedges = np.quantile(input_unc, np.linspace(0, 1, n_bins + 1))[1:-1]      # bulk inner edges (m)
    tail = np.array([e for e in tail_edges if e > (qedges.max() if qedges.size else 0)])
    edges = np.unique(np.concatenate([qedges, tail]))                          # B-1 inner edges (m)
    B = edges.size + 1
    b = np.clip(np.searchsorted(edges, input_unc, side="right"), 0, B - 1)
    score = error - r_model  # CQR additive residual

    # three grouping levels for hierarchical fallback
    q_full, n_full = grouped_quantiles(score, (joint_idx * T + frame_idx) * B + b, J * T * B, level)
    q_jb, n_jb = grouped_quantiles(score, joint_idx * B + b, J * B, level)
    q_b, n_b = grouped_quantiles(score, b, B, level)
    q_g, _ = conformal_add_quantile(score, level)
    q_full = q_full.reshape(J, T, B); n_full = n_full.reshape(J, T, B)
    q_jb = q_jb.reshape(J, B); n_jb = n_jb.reshape(J, B)

    q_grid = np.empty((J, T, B))
    src = np.empty((J, T, B), dtype="<U6")
    for j in range(J):
        for t in range(T):
            for bb in range(B):
                if n_full[j, t, bb] >= n_min and np.isfinite(q_full[j, t, bb]):
                    q_grid[j, t, bb], src[j, t, bb] = q_full[j, t, bb], "jtb"
                elif n_jb[j, bb] >= n_min and np.isfinite(q_jb[j, bb]):
                    q_grid[j, t, bb], src[j, t, bb] = q_jb[j, bb], "jb"
                elif n_b[bb] >= n_min and np.isfinite(q_b[bb]):
                    q_grid[j, t, bb], src[j, t, bb] = q_b[bb], "b"
                else:
                    q_grid[j, t, bb], src[j, t, bb] = (q_g if np.isfinite(q_g) else np.nanmax(q_b)), "g"
            if monotone:  # enforce q_hat non-decreasing in input-uncertainty bin
                q_grid[j, t] = pava_nondecreasing(q_grid[j, t], np.maximum(n_full[j, t], 1.0))
    return dict(bin_edges=edges, q_grid=q_grid, n_grid=n_full, source=src,
                level=level, J=J, T=T, B=B)


def apply_calibrator(calib, r_model, input_unc, joint_idx, frame_idx):
    """Calibrated radius (mm) = max(r_model + q_hat, 0). Vectorized over flat arrays."""
    b = np.clip(np.searchsorted(calib["bin_edges"], input_unc, side="right"), 0, calib["B"] - 1)
    return np.maximum(r_model + calib["q_grid"][joint_idx, frame_idx, b], 0.0)


# --------------------------------------------------------------------------- data / eval helpers


def load_results(path):
    with open(path, "rb") as f:
        r = cloudpickle.load(f)
    pred = np.asarray(r["predictions"], dtype=np.float64)
    tgt = np.asarray(r["targets"], dtype=np.float64)
    cov = np.asarray(r["covariance_matrices"], dtype=np.float64)
    li = np.asarray(r["last_input_poses"], dtype=np.float64)
    N, T, J, _ = pred.shape
    if li.shape[1] < J * 3 + J * 9:
        raise SystemExit("last_input_poses lacks the covariance block (need the input_uncertainty pipeline).")
    in_cov = li[:, J * 3:J * 3 + J * 9].reshape(N, J, 3, 3)
    last_poses = li[:, :J * 3].reshape(N, J, 3)  # [N, J, 3]
    return pred, tgt, cov, in_cov, last_poses


def flatten_valid(pred, tgt, cov_radius, in_set, level=None):
    """Return flat (error, r_model, input_unc, joint_idx, frame_idx) over valid joint-frames."""
    N, T, J, _ = pred.shape
    valid = ~(np.all(pred == 0.0, axis=(2, 3)) | np.all(tgt == 0.0, axis=(2, 3)))  # [N,T]
    valid = np.repeat(valid[:, :, None], J, axis=2)                                # [N,T,J]
    error = np.linalg.norm(pred - tgt, axis=-1)                                    # [N,T,J] mm
    in_TJ = np.repeat(in_set[:, None, :], T, axis=1)                               # [N,T,J] m
    jj = np.broadcast_to(np.arange(J)[None, None, :], (N, T, J))
    tt = np.broadcast_to(np.arange(T)[None, :, None], (N, T, J))
    m = valid
    return (error[m], cov_radius[m], in_TJ[m], jj[m].astype(np.int64), tt[m].astype(np.int64),
            valid)


def coverage_volume(error, radius):
    cov = float((error <= radius).mean())
    vol = 4.0 / 3.0 * np.pi * np.power(np.mean(radius) / 1000.0, 3.0)
    return cov, vol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_file", type=str,
                    default="results/motion_prediction/motion_prediction_results_validation.cloudpickle",
                    help="Single results file to split into calib/test (by sample).")
    ap.add_argument("--calib_file", type=str, default=None,
                    help="Explicit calibration results file (overrides the split of --results_file).")
    ap.add_argument("--test_file", type=str, default=None,
                    help="Explicit test results file (overrides the split of --results_file).")
    ap.add_argument("--calib_frac", type=float, default=0.5, help="Fraction of samples used to calibrate.")
    ap.add_argument("--config", type=str, default="h36m", choices=["h36m", "rgbd_yolo"])
    ap.add_argument("--likelihood", type=float, default=None,
                    help="Target coverage / confidence for the conformal set and the base radius "
                         "(overrides the config's SET_LIKELIHOOD). E.g. 0.99, 0.995, 0.999.")
    ap.add_argument("--n_bins", type=int, default=8, help="Equal-mass input-uncertainty quantile bins for the bulk.")
    ap.add_argument("--tail_edges", type=float, nargs="*", default=[0.3, 0.5, 0.75],
                    help="Extra explicit input-uncertainty bin edges (m) for the rare high-uncertainty "
                         "tail, where equal-mass bins would lump the safety-critical cases together.")
    ap.add_argument("--n_min", type=int, default=200, help="Min calibration points to trust a group's quantile.")
    ap.add_argument("--monotone", action=argparse.BooleanOptionalAction, default=True,
                    help="Enforce q_hat non-decreasing in input uncertainty (isotonic smoothing).")
    ap.add_argument("--base", type=str, default="raw", choices=["raw", "affine"],
                    help="Base radius the conformal correction sits on: raw model covariance (correct "
                         "for P2-pinball self-calibrated models, e.g. cov_p2p4) or the "
                         "affine-calibrated covariance (legacy NLL-only models).")
    ap.add_argument("--baseline", type=str, default="raw", choices=["raw", "affine"],
                    help="What to compare conformal against in the report: 'raw' = the model's own "
                         "(self-calibrated) radius -- the right comparison for P2-trained models; "
                         "'affine' = the legacy affine-calibrated radius. Use 'raw' for cov_p2p4 "
                         "(affine would double-calibrate a model that already predicts the radius).")
    ap.add_argument("--output_dir", type=str, default="results/motion_prediction/conformal_calibration",
                    help="Directory for the diagnostic figure.")
    ap.add_argument("--calibrator_path", type=str,
                    default="models/motion_prediction/conformal_calibration/conformal_calibrator.npz",
                    help="Path to write the deployable calibrator .npz (the file the pipeline loads).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.config == "rgbd_yolo":
        from conformal_human_motion_prediction.motion_prediction.rgbd_yolo_settings import (
            COV_CALIBRATION_FACTORS, COV_CALIBRATION_CT, COV_CALIBRATION_IT, SET_LIKELIHOOD)
    else:
        from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
            COV_CALIBRATION_FACTORS, COV_CALIBRATION_CT, COV_CALIBRATION_IT, SET_LIKELIHOOD)
    level = args.likelihood if args.likelihood is not None else SET_LIKELIHOOD
    out_dir = os.path.join(root_dir, args.output_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def rmodel(cov, base):
        c = calibrate_covariance_matrices(cov, COV_CALIBRATION_CT, COV_CALIBRATION_IT,
                                          COV_CALIBRATION_FACTORS) if base == "affine" else cov
        return convert_covariance_matrices_to_set(c, likelihood=level)  # [N,T,J] mm

    def prep(path):
        pred, tgt, cov, in_cov, last_poses = load_results(os.path.join(root_dir, path) if not os.path.isabs(path) else path)
        in_set = convert_covariance_matrices_to_set(in_cov, level) / 1000.0  # [N,J] m
        r_conf_base = rmodel(cov, args.base)
        r_baseline = rmodel(cov, args.baseline)  # comparison baseline (raw self-calibrated / affine)
        # Compute SARA shield radius with per-joint input uncertainty
        N, T, J, _ = pred.shape
        dt = 1.0 / 25.0  # FPS = 25 from h36m_settings
        pred_horizon_times = [(t + 1) * dt for t in range(T)]
        _, r_sara = compute_sara_predictions(
            last_input_poses=last_poses,
            prediction_horizon_times=pred_horizon_times,
            v_human=2.0,  # V_HUMAN_ISO from h36m_settings
            measurement_uncertainty=in_set,
        )  # [N, T, J] mm
        return pred, tgt, in_set, r_conf_base, r_baseline, r_sara

    # ----- assemble calibration / test splits --------------------------------------------------
    if args.calib_file and args.test_file:
        print(f"Calibrating on {args.calib_file}\nTesting on    {args.test_file}")
        cal = prep(args.calib_file)
        tst = prep(args.test_file)
    else:
        print(f"Splitting {args.results_file} by sample ({args.calib_frac:.0%} calib / rest test)")
        pred, tgt, in_set, r_conf_base, r_affine, r_sara = prep(args.results_file)
        N = pred.shape[0]
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(N)
        n_cal = int(round(args.calib_frac * N))
        ci, ti = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
        cal = tuple(a[ci] for a in (pred, tgt, in_set, r_conf_base, r_affine, r_sara))
        tst = tuple(a[ti] for a in (pred, tgt, in_set, r_conf_base, r_affine, r_sara))
    cpred, ctgt, cin, cbase, _, _ = cal
    tpred, ttgt, tin, tbase, tbaseline, tsara = tst
    J = cpred.shape[2]; T = cpred.shape[1]
    print(f"  calib samples={cpred.shape[0]}  test samples={tpred.shape[0]}  J={J} T={T}  "
          f"level={level}  base={args.base}")

    # ----- fit on calibration split ------------------------------------------------------------
    err_c, rm_c, in_c, j_c, t_c, _ = flatten_valid(cpred, ctgt, cbase, cin)
    calib = fit_calibrator(err_c, rm_c, in_c, j_c, t_c, J, T, level, args.n_bins, args.n_min,
                           args.monotone, tuple(args.tail_edges))
    B = calib["q_grid"].shape[2]
    print(f"  fitted q_grid[{J},{T},{B}]; input-unc bin edges (m): "
          f"{np.round(calib['bin_edges'], 3).tolist()}")
    src_counts = {s: int((calib['source'] == s).sum()) for s in ("jtb", "jb", "b", "g")}
    print(f"  group source (finest->coarsest): {src_counts}")

    # ----- evaluate on test split: baseline (affine) vs conditional conformal vs SARA ------------------
    err_t, rm_t, in_t, j_t, t_t, valid_t = flatten_valid(tpred, ttgt, tbase, tin)
    _, rbase_t, _, _, _, _ = flatten_valid(tpred, ttgt, tbaseline, tin)
    _, rsara_t, _, _, _, _ = flatten_valid(tpred, ttgt, tsara, tin)
    r_conf_t = apply_calibrator(calib, rm_t, in_t, j_t, t_t)

    cov_base, vol_base = coverage_volume(err_t, rbase_t)
    cov_con, vol_con = coverage_volume(err_t, r_conf_t)
    cov_sara, vol_sara = coverage_volume(err_t, rsara_t)
    print("\n==================== TEST-split coverage / volume ====================")
    print(f"target coverage (SET_LIKELIHOOD) = {level:.4f}")
    base_label = f"baseline ({args.baseline})"
    print(f"{'method':>26} {'coverage':>10} {'mean vol (m^3)':>15} {'mean radius (m)':>16}")
    print(f"{base_label:>26} {100 * cov_base:>9.3f}% {vol_base:>15.5f} {rbase_t.mean()/1000:>16.4f}")
    print(f"{'conditional conformal':>26} {100 * cov_con:>9.3f}% {vol_con:>15.5f} {r_conf_t.mean()/1000:>16.4f}")
    print(f"{'SARA shield':>26} {100 * cov_sara:>9.3f}% {vol_sara:>15.5f} {rsara_t.mean()/1000:>16.4f}")

    # coverage by input-uncertainty bin (the M1 test)
    edges = calib["bin_edges"]
    bt = np.clip(np.searchsorted(edges, in_t, side="right"), 0, B - 1)
    print(f"\nCoverage by input-uncertainty bin ({args.baseline} -> conformal -> SARA):")
    print(f"  {'bin (m)':>16} {'n':>10} {'base':>9} {'conformal':>10} {'SARA':>9} {'base vol':>9} {'con vol':>9} {'sara vol':>9}")
    lab_edges = np.concatenate([[0.0], edges, [np.inf]])
    for bb in range(B):
        m = bt == bb
        if not m.any():
            continue
        ca, _ = coverage_volume(err_t[m], rbase_t[m])
        cc, _ = coverage_volume(err_t[m], r_conf_t[m])
        cs, _ = coverage_volume(err_t[m], rsara_t[m])
        va = 4/3*np.pi*(rbase_t[m].mean()/1000)**3
        vc = 4/3*np.pi*(r_conf_t[m].mean()/1000)**3
        vs = 4/3*np.pi*(rsara_t[m].mean()/1000)**3
        lab = f"[{lab_edges[bb]:.2f},{lab_edges[bb+1]:.2f})" if np.isfinite(lab_edges[bb+1]) else f">={lab_edges[bb]:.2f}"
        print(f"  {lab:>16} {int(m.sum()):>10,} {100*ca:>8.2f}% {100*cc:>9.2f}% {100*cs:>8.2f}% {va:>9.4f} {vc:>9.4f} {vs:>9.4f}")

    # per-joint coverage (the M2 test)
    print(f"\nPer-joint coverage ({args.baseline} -> conformal -> SARA), sorted by baseline coverage:")
    print(f"  {'joint':>10} {'base':>9} {'conformal':>10} {'SARA':>9} {'base vol':>9} {'con vol':>9} {'sara vol':>9}")
    rows = []
    for j in range(J):
        m = j_t == j
        ca, va = coverage_volume(err_t[m], rbase_t[m])
        cc, vc = coverage_volume(err_t[m], r_conf_t[m])
        cs, vs = coverage_volume(err_t[m], rsara_t[m])
        rows.append((j, ca, cc, cs, va, vc, vs))
    for j, ca, cc, cs, va, vc, vs in sorted(rows, key=lambda r: r[1]):
        print(f"  {JOINT_NAMES_13[j]:>10} {100*ca:>8.2f}% {100*cc:>9.2f}% {100*cs:>8.2f}% {va:>9.4f} {vc:>9.4f} {vs:>9.4f}")
    print("=" * 90)

    # ----- figure -----------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    bins_x = range(B)
    ca_bin = [coverage_volume(err_t[bt == bb], rbase_t[bt == bb])[0] * 100 for bb in bins_x]
    cc_bin = [coverage_volume(err_t[bt == bb], r_conf_t[bt == bb])[0] * 100 for bb in bins_x]
    cs_bin = [coverage_volume(err_t[bt == bb], rsara_t[bt == bb])[0] * 100 for bb in bins_x]
    labels = [f"[{lab_edges[bb]:.2f},{lab_edges[bb+1]:.2f})" if np.isfinite(lab_edges[bb+1])
              else f">={lab_edges[bb]:.2f}" for bb in bins_x]
    ax[0].plot(bins_x, ca_bin, "o-", color="#e45756", label=args.baseline)
    ax[0].plot(bins_x, cc_bin, "s-", color="#4c78a8", label="conformal")
    ax[0].plot(bins_x, cs_bin, "^-", color="#59a14f", label="SARA")
    ax[0].axhline(100 * level, color="k", ls="--", lw=1, label=f"target {100*level:.1f}%")
    ax[0].set_xticks(list(bins_x)); ax[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax[0].set_title("Coverage vs input uncertainty (M1)"); ax[0].set_xlabel("input-unc bin (m)")
    ax[0].set_ylabel("coverage (%)"); ax[0].legend(fontsize=8)

    jo = [r[0] for r in sorted(rows, key=lambda r: r[1])]
    ax[1].plot(range(J), [100 * rows[j][1] for j in jo], "o-", color="#e45756", label=args.baseline)
    ax[1].plot(range(J), [100 * rows[j][2] for j in jo], "s-", color="#4c78a8", label="conformal")
    ax[1].plot(range(J), [100 * rows[j][3] for j in jo], "^-", color="#59a14f", label="SARA")
    ax[1].axhline(100 * level, color="k", ls="--", lw=1, label=f"target {100*level:.1f}%")
    ax[1].set_xticks(range(J)); ax[1].set_xticklabels([JOINT_NAMES_13[j] for j in jo], rotation=45, ha="right", fontsize=8)
    ax[1].set_title("Per-joint coverage (M2)"); ax[1].set_ylabel("coverage (%)"); ax[1].legend(fontsize=8)

    # q_hat vs input-uncertainty bin, per joint (the learned conditional correction)
    for j in range(J):
        ax[2].plot(range(B), calib["q_grid"][j].mean(axis=0) / 1000.0, label=JOINT_NAMES_13[j], lw=1)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("Learned q_hat (m) vs input-unc bin\n(>0 inflate, <0 tighten)")
    ax[2].set_xlabel("input-unc bin"); ax[2].set_ylabel("q_hat (m), frame-mean")
    ax[2].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Conditional conformal calibration + SARA comparison  (target {100*level:.1f}%, base={args.base})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(out_dir, "conformal_calibration.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved figure to {fig_path}")

    # ----- persist the deployable calibrator ---------------------------------------------------
    cal_path = os.path.join(root_dir, args.calibrator_path)
    Path(cal_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cal_path, bin_edges=calib["bin_edges"], q_grid=calib["q_grid"], n_grid=calib["n_grid"],
             level=level, J=J, T=T, B=B, base=args.base)
    print(f"Saved calibrator to {cal_path}  (apply: r_cal = max(r_model + q_grid[j,t,bin(input_unc)], 0))")


if __name__ == "__main__":
    main()
