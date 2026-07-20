"""Shared helpers for the final conformal-prediction results tables.

Three deliverables share the same three methods and the same CSV sources:
  * the standalone conformal prediction-set table (coverage / volume),
  * the standalone robot-shield safety table (c_safe / contacts / PFH_D / PL),
  * the combined table that fuses both.

This module centralizes the method definitions, the LaTeX row labels, and the readers for both
CSV kinds so the three generators stay consistent (same row order, same method -> row mapping).
"""
import csv
import math
import os

# Canonical method keys, in table row order.
METHODS = ["iso_no_ood", "ours_no_ood", "ours_ood"]

# LaTeX row label per method.
METHOD_LABELS = {
    "iso_no_ood": r"ISO 13855",
    "ours_no_ood": r"Ours with OOD inputs",
    "ours_ood": r"Ours OOD filtered",
}

# Conformal coverage CSV file per method (written by examples.motion_prediction).
COVERAGE_CSV = {
    "iso_no_ood": "coverage_stats_sara.csv",
    "ours_no_ood": "coverage_stats_conformal_prediction_sets.csv",
    "ours_ood": "coverage_stats_conformal_prediction_sets_ood_filtered.csv",
}


def _as_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def shield_method_key(row):
    """Map a shield CSV row to its canonical method key via (human_set, mask_ood)."""
    human_set = str(row.get("human_set", "conformal")).strip().lower()
    if human_set == "sara":
        return "iso_no_ood"
    return "ours_ood" if _as_bool(row.get("mask_ood", "False")) else "ours_no_ood"


def read_coverage_stats(csv_path):
    """Read a coverage CSV -> dict with coverage_percent, volume_m3, volume_std_m3."""
    with open(csv_path, newline="") as f:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(f)}
    return {
        "coverage_percent": float(stats["overall_coverage_percent"]),
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


def read_shield_by_method(csv_path, confidence=0.9999):
    """Read the shield results CSV -> {method_key: row-dict with parsed shield fields}.

    ``confidence`` selects which Clopper-Pearson PFH_D / PL columns to surface. If several rows map
    to the same method, the last one wins (fresh sweeps overwrite the CSV, so this is unusual).
    """
    tag = f"{float(confidence):.4f}"
    pfh_key, pl_key = f"pfh_d_{tag}", f"pl_{tag}"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    if pfh_key not in rows[0]:
        avail = sorted(k[len("pfh_d_"):] for k in rows[0] if k.startswith("pfh_d_"))
        raise SystemExit(f"confidence {tag} not in {csv_path}; available: {avail}")
    out = {}
    for r in rows:
        out[shield_method_key(r)] = {
            "pct_verified": float(r["pct_verified"]),
            "n_verified_unsafe": int(float(r["n_verified_unsafe"])),
            "n_verified_contact": int(float(r["n_verified_contact"])),
            "pfh_d": float(r[pfh_key]),
            "pl": str(r[pl_key]),
            "total_pairs": int(float(r["total_pairs"])),
            "t_cycle": float(r["t_cycle"]),
        }
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
