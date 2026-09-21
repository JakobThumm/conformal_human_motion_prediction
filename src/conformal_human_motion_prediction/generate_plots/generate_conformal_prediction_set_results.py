"""Standalone conformal prediction-set table (Coverage % / Volume m^3), one row per method.

Reads the per-method coverage CSVs written by ``examples.motion_prediction`` in one results dir:
  * coverage_stats_sara.csv                                       -> ISO 13855 without OOD filtered
  * coverage_stats_conformal_prediction_sets.csv                  -> Ours without OOD filtered
  * coverage_stats_conformal_prediction_sets_ood_filtered.csv     -> Ours with OOD filtered
  * coverage_stats_conformal_prediction_sets_max.csv              -> Ours (alpha_max ablation), no OOD filter
  * coverage_stats_conformal_prediction_sets_max_ood_filtered.csv -> Ours (alpha_max ablation), OOD filtered
  * coverage_stats_conformal_prediction_sets_uncalibrated.csv     -> Ours (no-calibration ablation), no OOD filter
  * coverage_stats_conformal_prediction_sets_uncalibrated_ood_filtered.csv
                                                                  -> Ours (no-calibration ablation), OOD filtered
Missing files are simply skipped, so a run without the ablations still produces the table.

The file also carries a second table with the OOD detection performance (AUROC plus the
TPR/FPR/TNR/FNR at the deployed threshold) of the two sketched-Lanczos detectors, read from the
metrics JSONs written by ``examples.id_vs_ood_pose_prediction`` and
``examples.id_vs_ood_motion_prediction``. It is omitted if neither JSON is present.

Coverage is reported at two granularities -- marginal (per joint-timestep sphere) and per whole
prediction (all T*J spheres of a sample hold at once) -- and Volume as the percentiles of the
individual per-sphere volumes.

Usage::

    python -m conformal_human_motion_prediction.generate_plots.generate_conformal_prediction_set_results \
        --results_dir results/final/conformal_prediction_sets
"""
import argparse
import os

from conformal_human_motion_prediction.generate_plots.conformal_results_common import (
    METHODS, METHOD_LABELS, OOD_DETECTORS, OOD_DETECTOR_LABELS, best_coverage_method, bold,
    coverage_cells, pct_cell, prune_to_methods, read_coverage_by_method, read_ood_detectors,
    select_methods,
)

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "../../../results/final/conformal_prediction_sets",
)
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "../../..")
DEFAULT_POSE_OOD_METRICS = os.path.join(
    _REPO_ROOT, "results/pose_prediction_ood/pose_ood_detection_metrics.json")
DEFAULT_MOTION_OOD_METRICS = os.path.join(
    _REPO_ROOT,
    "results/motion_prediction_ood_randproj/dct_pose_transformer_randproj_score_fn_results.json")


def generate_table(cov):
    """Build the LaTeX table string from ``cov`` = {method_key: coverage-stats dict}.

    The Volume column reports the 5 / 50 / 95 percentiles of the per-sphere volume (robust to the
    heavy OOD tail); the lowest value in each column is bolded.
    """
    present = [m for m in METHODS if m in cov]
    best_cov = best_coverage_method(cov, present)
    best_cov_pred = best_coverage_method(cov, present, "coverage_per_prediction_percent")
    best_p = {q: (min(present, key=lambda m: cov[m][f"volume_{q}_m3"]) if present else None)
              for q in ("p5", "p50", "p95")}

    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Conformal prediction set test results on H36M. Coverage is reported as the "
        r"miss-rate (rate of a ground-truth position landing outside its predicted set) and the "
        r"nines of reliability ($-\log_{10}$ miss-rate), at two granularities: per joint-timestep "
        r"(one trial per predicted sphere) and per prediction (a prediction misses if ANY of its "
        r"$T \times J$ spheres does, the event a downstream shield depends on); volume as the "
        r"5/50/95 percentiles of the per-sphere volume.}",
        r"    \label{tab:conformal_prediction_set}",
        r"    \begin{tabular}{lccccccc}",
        r"        \toprule",
        r"        \multirow{3}[5]{*}{\textbf{Method}} & \multicolumn{4}{c}{Coverage} & "
        r"\multicolumn{3}{c}{$\downarrow$ Volume ($m^3$)} \\",
        r"        \cmidrule(lr){2-5} \cmidrule(lr){6-8}",
        r"         & \multicolumn{2}{c}{per joint-timestep} & \multicolumn{2}{c}{per prediction} "
        r"& & & \\",
        r"        \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"         & $\downarrow$ Miss-rate & $\uparrow$ 9s & $\downarrow$ Miss-rate & "
        r"$\uparrow$ 9s & 5\% & 50\% & 95\% \\",
        r"        \midrule",
    ]
    for m in METHODS:
        if m not in cov:
            continue
        c = cov[m]
        miss_s, nines_s = coverage_cells(c["coverage_percent"], m == best_cov)
        miss_p, nines_p = coverage_cells(c["coverage_per_prediction_percent"], m == best_cov_pred)
        v5 = bold(f"{c['volume_p5_m3']:.3f}", m == best_p["p5"])
        v50 = bold(f"{c['volume_p50_m3']:.3f}", m == best_p["p50"])
        v95 = bold(f"{c['volume_p95_m3']:.3f}", m == best_p["p95"])
        lines.append(f"        {METHOD_LABELS[m]} & {miss_s} & {nines_s} & {miss_p} & {nines_p} & "
                     f"{v5} & {v50} & {v95} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def generate_ood_table(det):
    """Build the LaTeX OOD-detection table from ``det`` = {detector_key: metrics dict}.

    One row per detector. OOD is the positive class (the detector flags a sample when its score
    exceeds the threshold), so TPR is the OOD detection rate, FNR the rate of OOD inputs that slip
    through, and FPR the false-alarm rate on in-distribution inputs. The two detectors are scored
    against different OOD sets and at different thresholds, so the rows are not a ranking --
    each threshold is stated in its own column.
    """
    present = [d for d in OOD_DETECTORS if d in det]
    counts = "; ".join(
        f"{OOD_DETECTOR_LABELS[d].split(' (')[0].lower()}: {det[d]['id_dataset']} "
        f"($n = {det[d]['n_id']}$) vs.\\ {det[d]['ood_dataset']} ($n = {det[d]['n_ood']}$)"
        for d in present)
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Out-of-distribution detection performance of the two sketched-Lanczos "
        r"detectors. OOD is the positive class: a sample is flagged when its score exceeds the "
        r"detector's deployed threshold $\tau$, so TPR is the rate of OOD inputs caught, FNR the "
        r"rate that slips through, and FPR the false-alarm rate on in-distribution inputs. AUROC "
        r"is threshold-free; the four rates are evaluated at $\tau$. Datasets -- " + counts +
        r".}",
        r"    \label{tab:ood_detection}",
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        \textbf{Detector} & $\uparrow$ AUROC & $\tau$ & $\uparrow$ TPR (\%) & "
        r"$\downarrow$ FPR (\%) & $\uparrow$ TNR (\%) & $\downarrow$ FNR (\%) \\",
        r"        \midrule",
    ]
    for d in present:
        m = det[d]
        lines.append(
            f"        {OOD_DETECTOR_LABELS[d]} & {m['auroc']:.4f} & {m['threshold']:g} & "
            f"{pct_cell(m['tpr'])} & {pct_cell(m['fpr'])} & {pct_cell(m['tnr'])} & "
            f"{pct_cell(m['fnr'])} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                   help="Directory holding the three per-method coverage CSVs.")
    p.add_argument("--methods", default="all",
                   help="Method rows to include: 'all' (default) or a comma/space separated list of method keys (see METHODS in conformal_results_common).")
    p.add_argument("--output", default=None,
                   help="Output .tex path (default: <results_dir>/conformal_prediction_set_results.tex).")
    p.add_argument("--pose_ood_metrics", default=DEFAULT_POSE_OOD_METRICS,
                   help="Pose OOD detection metrics JSON (examples.id_vs_ood_pose_prediction). "
                        "Pass '' to drop the pose row.")
    p.add_argument("--motion_ood_metrics", default=DEFAULT_MOTION_OOD_METRICS,
                   help="Motion OOD detection metrics JSON (examples.id_vs_ood_motion_prediction). "
                        "Pass '' to drop the motion row.")
    args = p.parse_args()

    cov = prune_to_methods(read_coverage_by_method(args.results_dir),
                           select_methods(args.methods))
    if not cov:
        raise SystemExit(f"No coverage CSVs found in {args.results_dir}")
    table = generate_table(cov)

    # The OOD-detection table is appended to the same file; it comes from the ID-vs-OOD runs, not
    # from the coverage CSVs, so a results dir without those JSONs still yields the first table.
    det = read_ood_detectors({"pose": args.pose_ood_metrics, "motion": args.motion_ood_metrics})
    if det:
        table = table + "\n" + generate_ood_table(det)
    else:
        print("No OOD detection metrics JSON found — writing the coverage table only.")

    out = args.output or os.path.join(args.results_dir, "conformal_prediction_set_results.tex")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(table)
    print(f"Saved table to {out}\n")
    print(table)


if __name__ == "__main__":
    main()
