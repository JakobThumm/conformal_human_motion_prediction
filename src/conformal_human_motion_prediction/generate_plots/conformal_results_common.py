"""Shared helpers for the final conformal-prediction results tables.

Three deliverables share the same method set (our conditional-conformal sets, the single-threshold
max-score ablation, and the ISO 13855 baseline -- with and without OOD filtering) and the same CSV
sources:
  * the standalone conformal prediction-set table (coverage / volume),
  * the standalone robot-shield safety table (c_safe / contacts / PFH_D / PL),
  * the combined table that fuses both.

This module centralizes the method definitions, the LaTeX row labels, and the readers for both
CSV kinds so the three generators stay consistent (same row order, same method -> row mapping).
"""
import csv
import json
import re
import math
import os

# Canonical method keys, in table row order. The ``ours_max_*`` rows are the single-threshold
# ablation of our conditional calibration (one global alpha_max over all joint-timesteps, see
# motion_prediction.conformal_calibration --method max); the ``ours_uncal_*`` rows are the
# no-calibration ablation (fixed training-time factor sqrt(chi2_3(1-eps)) on the predicted
# covariance, --method uncalibrated). Each ablation sits next to the ``ours_*`` row it ablates so
# the coverage/volume cost of dropping the calibration is read off directly.
METHODS = ["iso_no_ood", "ours_uncal_no_ood", "ours_max_no_ood", "ours_no_ood",
           "ours_uncal_ood", "ours_max_ood", "ours_ood"]

# LaTeX row label per method.
METHOD_LABELS = {
    "iso_no_ood": r"ISO 13855",
    "ours_uncal_no_ood": r"Ours $\left(\alpha_{k, \chi}^j\right)$ with OOD inputs",
    "ours_max_no_ood": r"Ours ($\alpha_{\max}$) with OOD inputs",
    "ours_no_ood": r"Ours with OOD inputs",
    "ours_uncal_ood": r"Ours $\left(\alpha_{k, \chi}^j\right)$ OOD filtered",
    "ours_max_ood": r"Ours ($\alpha_{\max}$) OOD filtered",
    "ours_ood": r"Ours OOD filtered",
}

# Conformal coverage CSV file per method (written by examples.motion_prediction).
COVERAGE_CSV = {
    "iso_no_ood": "coverage_stats_sara.csv",
    "ours_uncal_no_ood": "coverage_stats_conformal_prediction_sets_uncalibrated.csv",
    "ours_max_no_ood": "coverage_stats_conformal_prediction_sets_max.csv",
    "ours_no_ood": "coverage_stats_conformal_prediction_sets.csv",
    "ours_uncal_ood": "coverage_stats_conformal_prediction_sets_uncalibrated_ood_filtered.csv",
    "ours_max_ood": "coverage_stats_conformal_prediction_sets_max_ood_filtered.csv",
    "ours_ood": "coverage_stats_conformal_prediction_sets_ood_filtered.csv",
}


def select_methods(spec=None):
    """Canonical-order subset of :data:`METHODS` from a comma/space separated ``spec``.

    ``None``/``""``/``"all"`` selects everything. The returned order is always the canonical
    METHODS order, never the order the spec happens to list -- a table's row order is a layout
    decision, independent of which rows a sweep produced (or of the order it produced them in).
    """
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return list(METHODS)
    want = {t for t in re.split(r"[,\s]+", str(spec).strip()) if t}
    unknown = sorted(want - set(METHODS))
    if unknown:
        raise SystemExit(f"unknown method key(s) {unknown}; known keys: {METHODS}")
    return [m for m in METHODS if m in want]


def prune_to_methods(by_method, methods):
    """Drop entries whose method key is not in ``methods`` (the row loops skip what is absent)."""
    keep = set(methods)
    return {k: v for k, v in by_method.items() if k in keep}


def _as_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def shield_method_key(row):
    """Map a shield CSV row to its canonical method key via (human_set, set_kind, mask_ood).

    ``set_kind == "max_conformal"`` marks a run driven by a max-score (single-threshold) calibrator
    and ``set_kind == "uncalibrated"`` one driven by the no-calibration ablation; rows written
    before those modes existed have no such value and map to the conditional rows.
    """
    human_set = str(row.get("human_set", "conformal")).strip().lower()
    if human_set == "sara":
        return "iso_no_ood"
    ood = _as_bool(row.get("mask_ood", "False"))
    set_kind = str(row.get("set_kind", "")).strip().lower()
    if set_kind == "max_conformal":
        return "ours_max_ood" if ood else "ours_max_no_ood"
    if set_kind == "uncalibrated":
        return "ours_uncal_ood" if ood else "ours_uncal_no_ood"
    return "ours_ood" if ood else "ours_no_ood"


def read_coverage_stats(csv_path):
    """Read a coverage CSV -> dict with coverage_percent, volume_m3, volume_std_m3.

    ``coverage_percent`` is the marginal (per joint-timestep) rate; ``coverage_per_prediction_
    percent`` the family-wise one (a prediction counts as covered only if all of its T*J spheres
    hold). The latter is NaN for CSVs written before it was recorded, which the table generators
    render as an absent cell.
    """
    with open(csv_path, newline="") as f:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(f)}
    return {
        "coverage_percent": float(stats["overall_coverage_percent"]),
        "coverage_per_prediction_percent": float(
            stats.get("overall_coverage_per_prediction_percent", "nan")),
        "n_predictions": int(float(stats.get("n_predictions", 0))),
        "volume_m3": float(stats["overall_volume_m3"]),
        "volume_std_m3": float(stats.get("overall_volume_std_m3", "nan")),
        "volume_p5_m3": float(stats.get("overall_volume_p5_m3", "nan")),
        "volume_p50_m3": float(stats.get("overall_volume_p50_m3", "nan")),
        "volume_p95_m3": float(stats.get("overall_volume_p95_m3", "nan")),
    }


def read_coverage_by_method(results_dir):
    """Read all available per-method coverage CSVs from ``results_dir``. Missing files are skipped."""
    out = {}
    for key, fname in COVERAGE_CSV.items():
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            out[key] = read_coverage_stats(path)
    return out


# ISO 13849-1 Performance Levels, keyed by the PFH_D band [lo, hi) (failures per hour).
PL_BANDS = [
    ("e", 1e-8, 1e-7),
    ("d", 1e-7, 1e-6),
    ("c", 1e-6, 3e-6),
    ("b", 3e-6, 1e-5),
    ("a", 1e-5, 1e-4),
]


def pl_from_pfh(pfh):
    """Map a PFH_D value (1/h) to the achievable ISO 13849-1 Performance Level."""
    if pfh < PL_BANDS[0][1]:
        return "e (better than required)"
    for name, lo, hi in PL_BANDS:
        if lo <= pfh < hi:
            return name
    return "none (worse than PL a)"


def pfh_d_upper_bound(N, k, t_cycle, confidence):
    """One-sided upper confidence bound on PFH_D from k dangerous failures in N test cycles.

    A dangerous failure is the shield verifying a trajectory that the ground truth shows in
    contact with the human (``n_verified_contact``) -- the same event the results tables report,
    so the failure count and the bound derived from it never drift apart.

    Each cycle is a Bernoulli trial; the per-cycle dangerous-failure probability gets the exact
    Clopper-Pearson upper limit p_up = Beta.ppf(confidence, k+1, N-k) (for k=0 this is the closed
    form 1-(1-C)^(1/N)). Converted to an hourly rate via the cycle time:
        PFH_D = PFC_D * (3600 s/h) / t_cycle.
    Returns (pfc_d_upper, pfh_d_upper).
    """
    from scipy.stats import beta
    pfc = 1.0 if k >= N else float(beta.ppf(confidence, k + 1, N - k))
    return pfc, pfc * 3600.0 / t_cycle


def confidence_tag(confidence):
    """CSV column suffix for a Clopper-Pearson confidence (``pfh_d_<tag>``/``pl_<tag>``).

    Fixed point with trailing zeros trimmed, but never fewer than 4 decimals, so the historical
    4-decimal tags stay byte-identical (0.99 -> ``0.9900``) while deeper confidences keep the
    digits that distinguish them (0.99999 -> ``0.99999``, which ``.4f`` would collapse to 1.0000).
    """
    frac = f"{float(confidence):.10f}".split(".")[1].rstrip("0")
    return f"{float(confidence):.{max(4, len(frac))}f}"


def fmt_confidence_percent(confidence):
    """Confidence as a percentage string: 0.9999 -> ``'99.99'``, 0.999999 -> ``'99.9999'``."""
    pct = float(confidence) * 100.0
    frac = f"{pct:.8f}".split(".")[1].rstrip("0")
    return f"{pct:.{max(2, len(frac))}f}"


def read_shield_by_method(csv_path, confidence=0.9999):
    """Read the shield results CSV -> {method_key: row-dict with parsed shield fields}.

    ``confidence`` selects which Clopper-Pearson PFH_D / PL columns to surface. If several rows map
    to the same method, the last one wins (fresh sweeps overwrite the CSV, so this is unusual).

    PFH_D / PL are recomputed here from the reported failure count (``n_verified_contact``, N and
    t_cycle) rather than trusted from the CSV, so the bound can never disagree with the failure
    count printed next to it in the table. A CSV cell that disagrees with the recomputation is a
    stale row (written by an older simulate_robot_shield) and only triggers a warning.
    """
    tag = confidence_tag(confidence)
    pfh_key = f"pfh_d_{tag}"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    if pfh_key not in rows[0]:
        avail = sorted(k[len("pfh_d_"):] for k in rows[0] if k.startswith("pfh_d_"))
        raise SystemExit(f"confidence {tag} not in {csv_path}; available: {avail}")
    out = {}
    stale = []
    for r in rows:
        if not str(r.get(pfh_key, "")).strip():
            # Row written before this confidence level existed -- its cell is empty, not zero.
            stale.append(shield_method_key(r))
            continue
        n_contact = int(float(r["n_verified_contact"]))
        total_pairs = int(float(r["total_pairs"]))
        t_cycle = float(r["t_cycle"])
        _, pfh = pfh_d_upper_bound(total_pairs, n_contact, t_cycle, confidence)
        pfh_csv = float(r[pfh_key])
        if not math.isclose(pfh, pfh_csv, rel_tol=1e-6):
            print(f"warning: {csv_path}: {shield_method_key(r)} row has a stale PFH_D column "
                  f"({pfh_csv:.3e}); using {pfh:.3e} recomputed from n_verified_contact="
                  f"{n_contact:,} at confidence {tag}")
        out[shield_method_key(r)] = {
            "pct_verified": float(r["pct_verified"]),
            "n_verified_unsafe": int(float(r["n_verified_unsafe"])),
            "n_verified_contact": n_contact,
            "pfh_d": pfh,
            "pl": pl_from_pfh(pfh),
            "total_pairs": total_pairs,
            "t_cycle": t_cycle,
        }
    if stale and not out:
        raise SystemExit(f"No row in {csv_path} has confidence {tag} (empty cells for "
                         f"{sorted(set(stale))}); re-run simulate_robot_shield for those methods.")
    if stale:
        print(f"warning: skipping rows without confidence {tag}: {sorted(set(stale))}")
    return out


def fmt_sci(x):
    """Format a float as LaTeX scientific notation, e.g. 4.87e-07 -> '4.87 \\times 10^{-7}'."""
    x = float(x)
    if x <= 0:
        return "0"
    exp, mant = 0, x
    while mant >= 10.0:
        mant /= 10.0
        exp += 1
    while mant < 1.0:
        mant *= 10.0
        exp -= 1
    return f"{mant:.2f} \\times 10^{{{exp}}}"


def _mantissa_exp(x):
    """Decompose x into (signed mantissa in [1,10), integer exponent). x must be nonzero."""
    exp, mant = 0, abs(x)
    while mant >= 10.0:
        mant /= 10.0
        exp += 1
    while mant < 1.0:
        mant *= 10.0
        exp -= 1
    return (-mant if x < 0 else mant), exp


def fmt_num(x, digits=2):
    """Format a float for siunitx \\num, e.g. 4.87e-07 -> '4.87e-7', 2.175e13 -> '2.175e13'.

    Clean exponent (no leading zero, no '+') so \\num renders it as scientific notation.
    """
    x = float(x)
    if x == 0:
        return "0"
    mant, exp = _mantissa_exp(x)
    return f"{mant:.{digits}f}e{exp}"


def sci_cell(x, is_bold=False, digits=2):
    """Scientific-notation table cell. Non-bold uses siunitx \\num; bold uses an explicit
    \\mathbf{m \\times 10^{e}} (siunitx \\num does not bold cleanly)."""
    x = float(x)
    if not is_bold:
        return "0" if x == 0 else f"\\num{{{fmt_num(x, digits)}}}"
    if x == 0:
        return r"\textbf{0}"
    mant, exp = _mantissa_exp(x)
    return f"$\\mathbf{{{mant:.{digits}f} \\times 10^{{{exp}}}}}$"


def coverage_cells(coverage_percent, is_best, digits=1):
    """LaTeX (miss-rate, nines-of-reliability) cell pair for one coverage percentage.

    Returns ``("X", "X")`` when the value is absent (NaN), matching how the generators render a
    method whose CSV half is missing -- so an older coverage CSV without the per-prediction rate
    leaves a blank pair of cells instead of formatting a NaN.
    """
    if coverage_percent is None or math.isnan(coverage_percent):
        return "X", "X"
    k = nines_of_reliability(coverage_percent)
    return (sci_cell(miss_rate(coverage_percent), is_best, digits=digits),
            bold(r"$\infty$" if math.isinf(k) else f"{k:.2f}", is_best))


def best_coverage_method(by_method, methods, key="coverage_percent"):
    """Method with the highest coverage under ``key``, ignoring methods whose value is absent."""
    present = [m for m in methods if m in by_method and not math.isnan(by_method[m].get(key, float("nan")))]
    return max(present, key=lambda m: by_method[m][key]) if present else None


def miss_rate(coverage_percent):
    """Miss-rate p_miss = 1 - p_coverage (rate of a prediction landing outside the set)."""
    return 1.0 - float(coverage_percent) / 100.0


def nines_of_reliability(coverage_percent):
    """Nines of reliability k = -log10(p_miss); inf if the empirical miss-rate is 0."""
    pm = miss_rate(coverage_percent)
    return math.inf if pm <= 0 else -math.log10(pm)


def fmt_pl(pl):
    """'d' -> 'PL d'; 'e (better than required)' -> 'PL e'; 'none (worse than PL a)' -> 'none'."""
    tok = str(pl).strip().split()[0]
    return "none" if tok.lower() == "none" else f"PL {tok}"


def bold(s, is_bold):
    return r"\textbf{" + s + "}" if is_bold else s


# --- OOD detector metrics -----------------------------------------------------------------
# The two sketched-Lanczos detectors of the pipeline, in table row order, with the JSON written
# by the corresponding ID-vs-OOD example script.
OOD_DETECTORS = ["pose", "motion"]

# Row label per detector. The ID set is H36M for both, so only the OOD set is named here; the
# caption carries the ID sets and the sample counts.
OOD_DETECTOR_LABELS = {
    "pose": r"Pose (tiger-pose)",
    "motion": r"Motion (time-shuffled)",
}

OOD_METRIC_KEYS = ("auroc", "tpr", "fpr", "tnr", "fnr")


def read_ood_detection_metrics(json_path):
    """Read one OOD-detector metrics JSON (see ``utils.eval_utils.save_ood_detection_metrics``).

    Args:
        json_path: Path to the JSON written by ``examples.id_vs_ood_pose_prediction`` or
            ``examples.id_vs_ood_motion_prediction``.
    Returns:
        The parsed dict, or None if the file does not exist or predates the rate fields (an older
        run that only recorded AUROC/AUPRC) -- so a partial results dir degrades to no OOD table
        instead of a table of blanks.
    """
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        d = json.load(f)
    if any(d.get(k) is None for k in OOD_METRIC_KEYS):
        print(f"{json_path}: no threshold-based rates (re-run with --ood_threshold); skipping.")
        return None
    return d


def read_ood_detectors(paths):
    """Read the per-detector metrics JSONs.

    Args:
        paths: {detector_key: json_path}.
    Returns:
        {detector_key: metrics dict} for the files that exist and carry the rate fields.
    """
    out = {}
    for key in OOD_DETECTORS:
        m = read_ood_detection_metrics(paths.get(key))
        if m is not None:
            out[key] = m
    return out


def pct_cell(x, digits=2):
    """Rate in [0, 1] as a percentage table cell, e.g. 0.0025 -> '0.25'."""
    return f"{100.0 * float(x):.{digits}f}"
