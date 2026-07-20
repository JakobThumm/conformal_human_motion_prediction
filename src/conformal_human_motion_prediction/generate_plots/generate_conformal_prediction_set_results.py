"""Standalone conformal prediction-set table (Coverage % / Volume m^3) for the three methods.

Reads the per-method coverage CSVs written by ``examples.motion_prediction`` in one results dir:
  * coverage_stats_sara.csv                                   -> ISO 13855 without OOD filtered
  * coverage_stats_conformal_prediction_sets.csv              -> Ours without OOD filtered
  * coverage_stats_conformal_prediction_sets_ood_filtered.csv -> Ours with OOD filtered

Coverage is reported to 4 decimals; Volume as mean +/- std over the individual per-sphere volumes.

Usage::

    python -m conformal_human_motion_prediction.generate_plots.generate_conformal_prediction_set_results \
        --results_dir results/final/conformal_prediction_sets
"""
import argparse
import math
import os

from conformal_human_motion_prediction.generate_plots.conformal_results_common import (
    METHODS, METHOD_LABELS, bold, miss_rate, nines_of_reliability, read_coverage_by_method, sci_cell,
)

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "../../../results/final/conformal_prediction_sets",
)


def generate_table(cov):
    """Build the LaTeX table string from ``cov`` = {method_key: coverage-stats dict}.

    The Volume column reports the 5 / 50 / 95 percentiles of the per-sphere volume (robust to the
    heavy OOD tail); the lowest value in each column is bolded.
    """
    present = [m for m in METHODS if m in cov]
    best_cov = max(present, key=lambda m: cov[m]["coverage_percent"]) if present else None
    best_p = {q: (min(present, key=lambda m: cov[m][f"volume_{q}_m3"]) if present else None)
              for q in ("p5", "p50", "p95")}

    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Conformal prediction set test results on H36M. Coverage is reported as the "
        r"miss-rate (rate of prediction outside of the predicted set) and the nines of reliability "
        r"($-\log_{10}$ miss-rate); volume as the 5/50/95 percentiles of the per-sphere volume.}",
        r"    \label{tab:conformal_prediction_set}",
        r"    \begin{tabular}{lccccc}",
        r"        \toprule",
        r"        \multirow{2}[3]{*}{\textbf{Method}} & \multicolumn{2}{c}{Coverage} & "
        r"\multicolumn{3}{c}{$\downarrow$ Volume ($m^3$)} \\",
        r"        \cmidrule(lr){2-3} \cmidrule(lr){4-6}",
        r"         & $\downarrow$ Miss-rate & $\uparrow$ 9s of reliability & 5\% & 50\% & 95\% \\",
        r"        \midrule",
    ]
    for m in METHODS:
        if m not in cov:
            continue
        c = cov[m]
        miss_s = sci_cell(miss_rate(c["coverage_percent"]), m == best_cov, digits=1)
        k = nines_of_reliability(c["coverage_percent"])
        nines_s = bold(r"$\infty$" if math.isinf(k) else f"{k:.2f}", m == best_cov)
        v5 = bold(f"{c['volume_p5_m3']:.3f}", m == best_p["p5"])
        v50 = bold(f"{c['volume_p50_m3']:.3f}", m == best_p["p50"])
        v95 = bold(f"{c['volume_p95_m3']:.3f}", m == best_p["p95"])
        lines.append(f"        {METHOD_LABELS[m]} & {miss_s} & {nines_s} & {v5} & {v50} & {v95} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                   help="Directory holding the three per-method coverage CSVs.")
    p.add_argument("--output", default=None,
                   help="Output .tex path (default: <results_dir>/conformal_prediction_set_results.tex).")
    args = p.parse_args()

    cov = read_coverage_by_method(args.results_dir)
    if not cov:
        raise SystemExit(f"No coverage CSVs found in {args.results_dir}")
    table = generate_table(cov)
    out = args.output or os.path.join(args.results_dir, "conformal_prediction_set_results.tex")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(table)
    print(f"Saved table to {out}\n")
    print(table)


if __name__ == "__main__":
    main()
