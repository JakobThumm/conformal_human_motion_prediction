"""Summarize the coverage sweep: baseline vs each variant on the SAME validation split, using the
deployed AFFINE calibration set (so the numbers isolate the training change, not the parallel
conditional-conformal calibrator). One consistent pass -> one table.

Run: .venv/bin/python -m experiments.coverage.summarize   (or: python experiments/coverage/summarize.py)
"""
import os
import numpy as np
import cloudpickle

from conformal_human_motion_prediction.motion_prediction.inference_helper import calibrate_covariance_matrices
from conformal_human_motion_prediction.utils.eval_utils import convert_covariance_matrices_to_set
from conformal_human_motion_prediction.pose_estimation.h36m_settings import JOINT_NAMES_13
from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
    COV_CALIBRATION_FACTORS, COV_CALIBRATION_CT, COV_CALIBRATION_IT, SET_LIKELIHOOD)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

RUNS = [
    ("baseline(final_model)", "results/motion_prediction/motion_prediction_results_validation.cloudpickle"),
    ("control(retrain)",      "results/coverage_experiments/cov_control/motion_prediction_results_validation.cloudpickle"),
    ("P1 input-noise",        "results/coverage_experiments/cov_p1_noise/motion_prediction_results_validation.cloudpickle"),
    ("P2 pinball",            "results/coverage_experiments/cov_p2_pinball/motion_prediction_results_validation.cloudpickle"),
    ("P1+P2",                 "results/coverage_experiments/cov_p1p2/motion_prediction_results_validation.cloudpickle"),
    ("P1+P2+P4",              "results/coverage_experiments/cov_p1p2p4/motion_prediction_results_validation.cloudpickle"),
    ("P2+P4 (no P1)",         "results/coverage_experiments/cov_p2p4/motion_prediction_results_validation.cloudpickle"),
    ("P2(hi)+P4 (no P1)",     "results/coverage_experiments/cov_p2hi_p4/motion_prediction_results_validation.cloudpickle"),
]

ANKLES = [JOINT_NAMES_13.index("LAnkle"), JOINT_NAMES_13.index("RAnkle")]
WRISTS = [JOINT_NAMES_13.index("LWrist"), JOINT_NAMES_13.index("RWrist")]
STRATA = [(0.30, 0.50), (0.50, 0.75), (0.75, 1.00), (1.00, np.inf)]


def analyze(path):
    with open(os.path.join(ROOT, path), "rb") as f:
        r = cloudpickle.load(f)
    pred = np.asarray(r["predictions"], np.float64)
    tgt = np.asarray(r["targets"], np.float64)
    cov = np.asarray(r["covariance_matrices"], np.float64)
    last = np.asarray(r["last_input_poses"], np.float64)
    N, T, J, _ = pred.shape

    in_cov = last[:, J * 3: J * 3 + J * 9].reshape(N, J, 3, 3)
    in_set = convert_covariance_matrices_to_set(in_cov, SET_LIKELIHOOD) / 1000.0  # [N,J] m

    if os.environ.get("AFFINE", "1") != "0":
        cov = calibrate_covariance_matrices(cov, COV_CALIBRATION_CT, COV_CALIBRATION_IT, COV_CALIBRATION_FACTORS)
    radius = convert_covariance_matrices_to_set(cov, likelihood=SET_LIKELIHOOD)   # [N,T,J] mm
    dist = np.linalg.norm(pred - tgt, axis=-1)                                    # [N,T,J] mm
    within = dist <= radius

    valid_TF = ~(np.all(pred == 0.0, axis=(2, 3)) | np.all(tgt == 0.0, axis=(2, 3)))
    valid = np.repeat(valid_TF[:, :, None], J, axis=2)
    in_set_TJ = np.repeat(in_set[:, None, :], T, axis=1)

    cflat = within[valid]
    inflat = in_set_TJ[valid]
    rflat = radius[valid] / 1000.0
    dflat = dist[valid] / 1000.0

    out = {}
    out["overall_cov"] = 100 * cflat.mean()
    out["fail_pct"] = 100 * (1 - cflat.mean())
    out["mpjpe_mm"] = 1000 * dflat.mean()
    out["mean_vol"] = 4 / 3 * np.pi * (rflat.mean() ** 3)
    out["corr"] = float(np.corrcoef(inflat, rflat)[0, 1])
    for lo, hi in STRATA:
        m = (inflat >= lo) & (inflat < hi)
        out[f"cov_{lo:.2f}_{hi:.2f}"] = 100 * cflat[m].mean() if m.any() else float("nan")
    # per-joint coverage
    pj = {}
    for j in range(J):
        vj = valid[:, :, j]
        pj[j] = 100 * within[:, :, j][vj].mean()
    out["LAnkle"] = pj[ANKLES[0]]
    out["RAnkle"] = pj[ANKLES[1]]
    out["LWrist"] = pj[WRISTS[0]]
    out["RWrist"] = pj[WRISTS[1]]
    out["worst_joint"] = JOINT_NAMES_13[min(pj, key=pj.get)]
    out["worst_cov"] = min(pj.values())
    return out


def main():
    rows = []
    for label, path in RUNS:
        if not os.path.exists(os.path.join(ROOT, path)):
            print(f"  (skip {label}: missing {path})")
            continue
        rows.append((label, analyze(path)))

    def fmt(v, w=7, p=2):
        return f"{v:>{w}.{p}f}" if isinstance(v, float) and not np.isnan(v) else f"{'n/a':>{w}}"

    cols = [("overall_cov", "overallCov%"), ("cov_0.50_0.75", "cov[.5,.75)"),
            ("cov_0.75_1.00", "cov[.75,1)"), ("cov_1.00_inf", "cov>=1.0"),
            ("RAnkle", "RAnkle%"), ("LAnkle", "LAnkle%"), ("RWrist", "RWrist%"),
            ("corr", "in->out r"), ("mean_vol", "meanVol m3"), ("mpjpe_mm", "MPJPE mm")]
    hdr = f"{'variant':>22} | " + " | ".join(f"{h:>11}" for _, h in cols)
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, o in rows:
        cells = []
        for key, _ in cols:
            v = o.get(key, float("nan"))
            if key == "corr":
                cells.append(f"{v:>11.3f}")
            elif key == "mean_vol":
                cells.append(f"{v:>11.4f}")
            else:
                cells.append(f"{v:>11.2f}")
        print(f"{label:>22} | " + " | ".join(cells))
    print(f"\n(worst joint per variant: " +
          ", ".join(f"{l}={o['worst_joint']}:{o['worst_cov']:.2f}%" for l, o in rows) + ")")
    print(f"target SET_LIKELIHOOD = {100*SET_LIKELIHOOD:.1f}%")


if __name__ == "__main__":
    main()
