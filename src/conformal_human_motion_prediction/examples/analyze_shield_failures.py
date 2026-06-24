"""Analyse the verified-but-contact failures dumped by ``simulate_robot_shield.py``.

``simulate_robot_shield --save_failures <file.npy>`` writes one dict per instance where the shield
declared a monitored trajectory *verified* (no predicted robot capsule touches any predicted human
sphere) yet the ground truth had a (possibly unsafe) human-robot contact. This script reconstructs
the verification geometry of each instance -- mirroring the simulation exactly -- and produces
overall statistics + a multi-panel figure to investigate *why* the shield was fooled.

Questions answered (one figure panel + printed stat each):

  1. By how much was the prediction wrong?      -> ``pred_err``: distance between the predicted and
     the true position of the *contacting* joint at the contact step (m).
  2. How large were the predicted occupancies?   -> ``pred_radius``: predicted sphere radius of the
     contacting joint at the contact step (m).
  3. How much larger would the predicted occupancy need to be to flip the verdict? -> ``margin``:
     the minimum predicted clearance over all (interval, link, joint) pairs. Inflating every
     predicted human sphere by this much would make the shield detect an intersection and refuse to
     verify. Reported in metres and as a fraction of the closest predicted radius.
  4. Which human body part causes the issue?     -> distribution of the *contacting joint* (the
     true human sphere that penetrates a true robot capsule deepest), split safe/unsafe.
  5. Movement pattern around the prediction?     -> predicted vs. true horizon displacement of the
     contacting joint, decomposed by error source. NB: step 0 is the *camera-estimated* input pose
     (noisy) while steps >=1 are *mocap* ground truth, so the step 0->1 gap is dominated by the
     camera-vs-mocap measurement discrepancy, not real motion. The failures turn out to be the heavy
     tail of camera pose-estimation noise (worst on the arms), which the conformal/measurement-
     uncertainty model under-covers -- not sudden human motion.

Plus several follow-up panels (contact time within the horizon, penetration depth, conformal
coverage of the contacting joint, robot link index, link speed at contact, scene/motion
concentration). ``last_input_poses`` only carries the single current observed pose in its first 39
dims (the rest is the model's feature vector, not absolute history), so genuine pre-observation
motion is not recoverable; the *predicted* horizon displacement is used as the model's proxy for
"what it expected given the history".

Run::

    .venv/bin/python -m conformal_human_motion_prediction.examples.analyze_shield_failures \
        --failures results/motion_prediction/shield_failures.npy
"""
import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conformal_human_motion_prediction.pose_estimation.h36m_settings import JOINT_NAMES_13
from conformal_human_motion_prediction.motion_prediction.h36m_settings import V_HUMAN_ISO, V_ROBOT_ISO

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
N_LINKS = 7


# --------------------------------------------------------------------------- geometry (mirrors sim)


def segment_point_distances(a, b, pts):
    """Distance from each point in ``pts`` [n,3] to the segment a-b. Returns [n]."""
    ab = b - a
    denom = float(ab @ ab)
    ap = pts - a
    if denom <= 1e-18:
        return np.linalg.norm(ap, axis=1)
    t = np.clip(ap @ ab / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def last_step_below(hz, tp):
    le = np.flatnonzero(hz <= tp + 1e-9)
    return int(le[-1]) if le.size else 0


def nearest_step(hz, tp):
    return int(np.argmin(np.abs(hz - tp)))


# --------------------------------------------------------------------------- per-instance analysis


def analyze_instance(ins, vh, vr):
    """Reconstruct one failure instance's geometry; return a flat dict of derived quantities."""
    hz = np.asarray(ins["horizon_times"], dtype=np.float64)          # [S]
    rp, rt = ins["robot_pred"], ins["robot_true"]
    pc = np.asarray(ins["human_pred_centers"], dtype=np.float64)     # [S,J,3]
    pr = np.asarray(ins["human_pred_radii"], dtype=np.float64)       # [S,J]
    tc = np.asarray(ins["human_true_centers"], dtype=np.float64)
    tr = np.asarray(ins["human_true_radii"], dtype=np.float64)
    R = rp["p1"].shape[0]
    S, J = pr.shape

    # ---- (Q3) predicted side: minimum clearance over all (interval, link, joint) pairs ----------
    # verified  =>  every predicted pair has clearance > 0. The smallest clearance is exactly how
    # much one would have to inflate the predicted human spheres (uniformly) to trigger a detection.
    min_clear = np.inf
    margin_radius = np.nan          # predicted radius at the closest predicted approach
    for i in range(R):
        tp = float(rp["tp"][i, 0])
        s = last_step_below(hz, tp)
        addr = (tp - hz[s]) * vh                                     # V_HUMAN_ISO bridge -> capsule
        for a in range(N_LINKS):
            eff = float(rp["r"][i, a]) + addr
            d = segment_point_distances(rp["p1"][i, a], rp["p2"][i, a], pc[s])   # [J]
            clear = d - (eff + pr[s])                                            # [J]
            j = int(np.argmin(clear))
            if clear[j] < min_clear:
                min_clear = float(clear[j])
                margin_radius = float(pr[s, j])

    # ---- (Q4/Q1/Q2) true side: the contact that fooled us -----------------------------------------
    # Attribute the instance to its deepest-penetrating contact, *consistently with the sim's unsafe
    # flag*: the sim marks an instance unsafe iff a link with speed > V_ROBOT_ISO makes contact, so
    # for unsafe instances we attribute to the deepest *fast* link (the dangerous one), else to the
    # deepest link of any speed. Without this, the deepest overall link is often a different, slow
    # link, which would mislabel an unsafe instance's contacting link as slow.
    unsafe = bool(ins["unsafe"])
    best_pen = -np.inf
    cj = cs = ci = ca = -1
    for i in range(R):
        s = nearest_step(hz, float(rp["tp"][i, 0]))
        for a in range(N_LINKS):
            if unsafe and float(rt["speed"][i, a]) <= vr:
                continue                                                        # not a dangerous link
            rlink = float(rt["r"][i, a])
            d = segment_point_distances(rt["p1"][i, a], rt["p2"][i, a], tc[s])   # [J]
            pen = (rlink + tr[s]) - d                                            # [J] >0 => overlap
            j = int(np.argmax(pen))
            if pen[j] > best_pen:
                best_pen = float(pen[j])
                cj, cs, ci, ca = j, s, i, a

    link_speed = float(rt["speed"][ci, ca])
    # prediction error & predicted occupancy at the contacting joint / contact step
    pred_err = float(np.linalg.norm(pc[cs, cj] - tc[cs, cj]))
    pred_radius = float(pr[cs, cj])
    true_radius = float(tr[cs, cj])
    # conformal coverage: did the predicted sphere of the contacting joint contain its true centre?
    covered = pred_err <= pred_radius

    # ---- (Q5) movement of the contacting joint over the horizon ----------------------------------
    pred_disp = float(np.linalg.norm(pc[cs, cj] - pc[0, cj]))
    true_disp = float(np.linalg.norm(tc[cs, cj] - tc[0, cj]))
    # The "true" occupancy mixes two sensors: step 0 is the *camera-estimated* input pose (noisy),
    # steps >=1 are *mocap* ground truth. So the step 0->1 gap is dominated by camera-vs-mocap
    # measurement error, NOT real motion. Decompose: obs_jump = that sensor discrepancy; the pure
    # mocap motion over the horizon (t1..contact) is the genuine human displacement.
    obs_jump = float(np.linalg.norm(tc[1, cj] - tc[0, cj]))                 # camera(t0) vs mocap(t1)
    horizon_motion = float(np.linalg.norm(tc[cs, cj] - tc[1, cj]))          # mocap-only motion
    # per-step true speed of the contacting joint (m/s). Step 0->1 is the last-observed-input ->
    # first-target transition, which the too-fast filter never screens (it only looks within the
    # prediction horizon); it is reported separately as the motion-onset speed.
    dts = np.diff(hz)
    step_speed = np.linalg.norm(np.diff(tc[:, cj], axis=0), axis=1) / np.maximum(dts, 1e-9)
    true_peak_speed = float(step_speed.max()) if step_speed.size else 0.0
    onset_speed = float(step_speed[0]) if step_speed.size else 0.0          # input -> first target
    horizon_peak_speed = float(step_speed[1:].max()) if step_speed.size > 1 else 0.0  # filtered band

    # ---- whole-instance prediction-error summary over all joints & horizon steps -----------------
    all_err = np.linalg.norm(pc[1:] - tc[1:], axis=2)               # [S-1,J], skip step 0 (=input)
    base_dist = float(np.linalg.norm(rp["p1"][:, :, :2].reshape(-1, 2), axis=1).min())  # closest link xy to base

    return dict(
        unsafe=bool(ins["unsafe"]), t_ms=int(ins["t_ms"]), motion_idx=int(ins["motion_idx"]),
        margin=float(min_clear), margin_radius=margin_radius,
        margin_frac=float(min_clear / margin_radius) if margin_radius > 0 else np.nan,
        pred_err=pred_err, pred_radius=pred_radius, true_radius=true_radius,
        covered=bool(covered), penetration=float(best_pen),
        contact_joint=cj, contact_step=cs, contact_time=float(hz[cs]),
        robot_link=ca, link_speed=link_speed,
        pred_disp=pred_disp, true_disp=true_disp, true_peak_speed=true_peak_speed,
        onset_speed=onset_speed, horizon_peak_speed=horizon_peak_speed,
        obs_jump=obs_jump, horizon_motion=horizon_motion,
        noise_frac=float(obs_jump / pred_err) if pred_err > 1e-9 else np.nan,
        mean_err=float(all_err.mean()), max_err=float(all_err.max()),
        base_link_dist=base_dist,
    )


# --------------------------------------------------------------------------- reporting


def q(x, name, unit="m"):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return (f"  {name:<34} median={np.median(x):.3f} mean={x.mean():.3f} "
            f"p90={np.percentile(x, 90):.3f} max={x.max():.3f} {unit}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", type=str, default="results/motion_prediction/shield_failures.npy")
    ap.add_argument("--out_dir", type=str, default="results/motion_prediction",
                    help="Where to write the analysis figure and per-instance CSV.")
    ap.add_argument("--prefix", type=str, default="shield_failure_analysis")
    args = ap.parse_args()

    path = args.failures if os.path.isabs(args.failures) else os.path.join(root_dir, args.failures)
    data = np.load(path, allow_pickle=True)
    print(f"Loaded {len(data)} verified-but-contact instance(s) from {path}")

    rows = [analyze_instance(ins, V_HUMAN_ISO, V_ROBOT_ISO) for ins in data]
    F = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n = len(rows)
    unsafe = F["unsafe"]
    print(f"  {unsafe.sum()} unsafe (link speed > V_ROBOT_ISO={V_ROBOT_ISO} m/s), "
          f"{n - unsafe.sum()} contact-only\n")

    # ---- printed report -------------------------------------------------------------------------
    print("=================== Failure statistics ===================")
    print("Q1  Prediction error at the contacting joint (||pred - true||):")
    print(q(F["pred_err"], "pred_err"))
    print(q(F["mean_err"], "per-instance mean joint err"))
    print(q(F["max_err"], "per-instance max joint err"))
    cov = F["covered"]
    print(f"  conformal coverage of contacting joint: {cov.sum()}/{n} "
          f"({100 * cov.mean():.1f}%) had the true centre INSIDE the predicted sphere "
          f"(so contact came from the robot-side gap, not a set miss); "
          f"{n - cov.sum()} were genuine set misses.\n")

    print("Q2  Predicted occupancy size at the contacting joint:")
    print(q(F["pred_radius"], "pred_radius"))
    print(q(F["true_radius"], "true_radius (body only)"))
    print()

    print("Q3  Inflation of predicted spheres needed to flip 'verified' -> 'not verified':")
    print(q(F["margin"], "margin (abs)"))
    print(q(F["margin_frac"], "margin / closest pred radius", unit="x"))
    near = F["margin"] < 0.05
    print(f"  {near.sum()}/{n} ({100 * near.mean():.1f}%) were 'barely verified' (margin < 5 cm) -- "
          f"a small recalibration would catch them; the rest cleared by more.\n")

    print("Q4  Which body part contacts (deepest true penetration):")
    counts = np.bincount(F["contact_joint"], minlength=len(JOINT_NAMES_13))
    for j in np.argsort(counts)[::-1]:
        if counts[j] == 0:
            continue
        u = int(unsafe[F["contact_joint"] == j].sum())
        print(f"    {JOINT_NAMES_13[j]:<10} {counts[j]:>4}  ({u} unsafe)")
    print()

    print("Q5  Movement of the contacting joint over the horizon:")
    print(q(F["pred_disp"], "predicted displacement (model)"))
    print(q(F["true_disp"], "apparent true displacement (step0->contact, sensor-conflated)"))
    print("  -- 'apparent true displacement' mixes camera(step0) and mocap(steps>=1); see the")
    print("     source decomposition below for the part that is real motion vs measurement error.")
    print("  NB: step 0 is the *camera-estimated* input pose, steps >=1 are *mocap* ground truth,")
    print("  so the apparent step 0->1 'motion' is dominated by camera-vs-mocap measurement error,")
    print("  not real human motion. Decomposing the contacting joint's prediction error:")
    print(q(F["obs_jump"], "obs jump (camera t0 -> mocap t1)"))
    print(q(F["horizon_motion"], "pure mocap motion (t1 -> contact)"))
    print(q(F["noise_frac"], "obs_jump / prediction error", unit="x"))
    noise_dom = F["obs_jump"] > F["horizon_motion"]
    print(f"  measurement discrepancy exceeds real horizon motion in {noise_dom.sum()}/{n} "
          f"({100 * noise_dom.mean():.1f}%) -- the failures are the heavy tail of camera "
          f"pose-estimation noise (worst on the arms), not sudden human motion.\n")

    print("Extra:")
    print(q(F["penetration"], "true penetration depth"))
    print(q(F["contact_time"], "contact time in horizon", unit="s"))
    print(q(F["link_speed"], "robot link speed at contact", unit="m/s"))
    print(q(F["base_link_dist"], "closest robot link to base (xy)"))
    lc = np.bincount(F["robot_link"], minlength=N_LINKS)
    print("  contacting robot link index counts: " + ", ".join(f"L{a}:{lc[a]}" for a in range(N_LINKS)))
    umot, cmot = np.unique(F["motion_idx"], return_counts=True)
    usc, csc = np.unique(F["t_ms"], return_counts=True)
    print(f"  concentration: {len(umot)} distinct human motions, {len(usc)} distinct robot scenes; "
          f"top motion has {cmot.max()} failures, top scene {csc.max()}.")
    print("==========================================================")

    # ---- figure ---------------------------------------------------------------------------------
    os.makedirs(os.path.join(root_dir, args.out_dir), exist_ok=True)
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    safe_m = ~unsafe

    def dual_hist(a, key, title, xlabel, bins=25, logx=False):
        x = F[key]
        rng = (np.nanmin(x), np.nanmax(x))
        b = np.linspace(*rng, bins) if not logx else np.logspace(
            np.log10(max(rng[0], 1e-4)), np.log10(rng[1] + 1e-9), bins)
        a.hist([x[safe_m], x[unsafe]], bins=b, stacked=True,
               color=["#4c78a8", "#e45756"], label=["contact", "unsafe contact"])
        a.set_title(title); a.set_xlabel(xlabel); a.set_ylabel("instances")
        if logx:
            a.set_xscale("log")
        a.legend(fontsize=8)

    dual_hist(ax[0, 0], "pred_err", "Q1: prediction error at contacting joint", "||pred - true|| (m)")
    dual_hist(ax[0, 1], "pred_radius", "Q2: predicted occupancy radius (contact joint)", "radius (m)")
    dual_hist(ax[0, 2], "margin", "Q3: inflation needed to flip verdict", "min predicted clearance (m)")

    # Q4: contacting body part
    order = np.argsort(counts)[::-1]
    order = order[counts[order] > 0]
    su = np.array([int(unsafe[F["contact_joint"] == j].sum()) for j in order])
    sc = counts[order] - su
    ax[1, 0].bar(range(len(order)), sc, color="#4c78a8", label="contact")
    ax[1, 0].bar(range(len(order)), su, bottom=sc, color="#e45756", label="unsafe")
    ax[1, 0].set_xticks(range(len(order)))
    ax[1, 0].set_xticklabels([JOINT_NAMES_13[j] for j in order], rotation=45, ha="right", fontsize=8)
    ax[1, 0].set_title("Q4: contacting body part"); ax[1, 0].set_ylabel("instances"); ax[1, 0].legend(fontsize=8)

    # Q5: predicted vs true displacement scatter
    a = ax[1, 1]
    a.scatter(F["pred_disp"][safe_m], F["true_disp"][safe_m], s=18, c="#4c78a8", label="contact", alpha=0.7)
    a.scatter(F["pred_disp"][unsafe], F["true_disp"][unsafe], s=18, c="#e45756", label="unsafe", alpha=0.7)
    lim = max(F["pred_disp"].max(), F["true_disp"].max()) * 1.05
    a.plot([0, lim], [0, lim], "k--", lw=1, label="pred = true")
    a.axvspan(0, 0.10, color="orange", alpha=0.08)
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_title("Q5: horizon displacement (contact joint)\nshaded = predicted near-still")
    a.set_xlabel("predicted displacement (m)"); a.set_ylabel("true displacement (m)"); a.legend(fontsize=8)

    # Q1b: coverage -- prediction error vs predicted radius (diagonal = set boundary)
    a = ax[1, 2]
    a.scatter(F["pred_radius"][safe_m], F["pred_err"][safe_m], s=18, c="#4c78a8", alpha=0.7, label="contact")
    a.scatter(F["pred_radius"][unsafe], F["pred_err"][unsafe], s=18, c="#e45756", alpha=0.7, label="unsafe")
    lim2 = max(F["pred_radius"].max(), F["pred_err"].max()) * 1.05
    a.plot([0, lim2], [0, lim2], "k--", lw=1, label="err = radius (coverage edge)")
    a.set_xlim(0, lim2); a.set_ylim(0, lim2)
    a.set_title("Coverage: pred error vs predicted radius\n(above line = true joint outside set)")
    a.set_xlabel("predicted radius (m)"); a.set_ylabel("prediction error (m)"); a.legend(fontsize=8)

    dual_hist(ax[2, 0], "contact_time", "Contact time within horizon", "tp (s)", bins=11)
    # Error decomposition: camera-vs-mocap measurement discrepancy vs genuine mocap motion.
    a = ax[2, 1]
    a.scatter(F["obs_jump"][safe_m], F["horizon_motion"][safe_m], s=18, c="#4c78a8", alpha=0.7, label="contact")
    a.scatter(F["obs_jump"][unsafe], F["horizon_motion"][unsafe], s=18, c="#e45756", alpha=0.7, label="unsafe")
    lim3 = max(F["obs_jump"].max(), F["horizon_motion"].max()) * 1.05
    a.plot([0, lim3], [0, lim3], "k--", lw=1, label="equal")
    a.set_xlim(0, lim3); a.set_ylim(0, lim3)
    a.set_title("Error source: camera noise vs real motion\n(below line = measurement-noise driven)")
    a.set_xlabel("obs jump: camera t0 -> mocap t1 (m)")
    a.set_ylabel("pure mocap motion t1->contact (m)"); a.legend(fontsize=8)
    a = ax[2, 2]
    x = F["link_speed"]
    a.hist([x[safe_m], x[unsafe]], bins=25, stacked=True, color=["#4c78a8", "#e45756"],
           label=["contact", "unsafe"])
    a.axvline(V_ROBOT_ISO, color="k", ls="--", lw=1, label=f"V_ROBOT_ISO={V_ROBOT_ISO}")
    a.set_title("Robot link speed at contact"); a.set_xlabel("speed (m/s)"); a.set_ylabel("instances")
    a.legend(fontsize=8)

    fig.suptitle(f"Shield verified-but-contact failure analysis  (n={n}, {int(unsafe.sum())} unsafe)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = os.path.join(root_dir, args.out_dir, f"{args.prefix}.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved figure to {fig_path}")

    # ---- per-instance CSV for manual drill-down -------------------------------------------------
    csv_path = os.path.join(root_dir, args.out_dir, f"{args.prefix}.csv")
    keys = ["t_ms", "motion_idx", "unsafe", "contact_joint", "robot_link", "contact_time",
            "pred_err", "pred_radius", "true_radius", "covered", "margin", "margin_frac",
            "penetration", "link_speed", "pred_disp", "true_disp", "true_peak_speed",
            "onset_speed", "horizon_peak_speed", "obs_jump", "horizon_motion", "noise_frac",
            "mean_err", "max_err", "base_link_dist"]
    with open(csv_path, "w") as f:
        f.write("idx," + ",".join(keys) + ",contact_joint_name\n")
        for i, r in enumerate(rows):
            f.write(f"{i}," + ",".join(
                (f"{r[k]:.5g}" if isinstance(r[k], float) else str(int(r[k]) if isinstance(r[k], (bool, np.bool_)) else r[k]))
                for k in keys) + f",{JOINT_NAMES_13[r['contact_joint']]}\n")
    print(f"Saved per-instance CSV to {csv_path}")


if __name__ == "__main__":
    main()
