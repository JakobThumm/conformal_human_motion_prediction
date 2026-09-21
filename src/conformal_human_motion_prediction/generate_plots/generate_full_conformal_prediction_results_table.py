"""Combined final results table: motion-prediction evaluation + certification simulation on H36M.

Fuses the two standalone tables into the paper's ``tab:all_conformal_results``:
  * the conformal prediction-set coverage / volume, from the per-method coverage CSVs
    (results_dir, written by examples.motion_prediction), and
  * the robot-shield safety columns (c_safe, c_safe & contact, PFH_D, PL), from the shield
    results CSV (written by examples.simulate_robot_shield --results_csv).

Both halves are keyed by the same three methods (see conformal_results_common), so this only reads
the CSVs the two upstream scripts already produced -- run them first.

Usage::

    python -m conformal_human_motion_prediction.generate_plots.generate_full_conformal_prediction_results_table \
        --coverage_dir results/final/conformal_prediction_sets \
        --shield_csv results/final/robot_shield/shield_results.csv \
        --output results/final/all_conformal_results.tex
"""
import argparse
import os

from conformal_human_motion_prediction.generate_plots.conformal_results_common import (
    METHODS, prune_to_methods, select_methods, METHOD_LABELS, best_coverage_method, bold,
    coverage_cells, fmt_num, read_coverage_by_method, read_shield_by_method, sci_cell,
)

# Above this many verified-but-contact cases the count is shown in scientific notation.
_CONTACT_SCI_THRESHOLD = 10000


def generate_table(cov, shield, confidence):
    """Build the combined LaTeX table from coverage + shield dicts (both keyed by method)."""
    cov_present = [m for m in METHODS if m in cov]
    sh_present = [m for m in METHODS if m in shield]
    # Miss-rate (lower better) and nines of reliability (higher better) both rank by coverage.
    best_cov = best_coverage_method(cov, cov_present)
    best_p = {q: (min(cov_present, key=lambda m: cov[m][f"volume_{q}_m3"]) if cov_present else None)
              for q in ("p5", "p50", "p95")}
    best_safe = max(sh_present, key=lambda m: shield[m]["pct_verified"]) if sh_present else None
    best_contact = min(sh_present, key=lambda m: shield[m]["n_verified_contact"]) if sh_present else None
    best_pfh = min(sh_present, key=lambda m: shield[m]["pfh_d"]) if sh_present else None

    n = shield[best_pfh]["total_pairs"] if best_pfh else 0

    def contact_cell(k, is_bold):
        body = (f"\\num{{{fmt_num(k, 1)}}}" if k >= _CONTACT_SCI_THRESHOLD else f"{k:,}")
        return bold(body, is_bold)

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Motion prediction evaluation and certification simulation on H36M test data. "
        r"The first columns report the per joint-timestep miss-rate (rate of a ground-truth "
        r"position landing outside the predicted set), the nines of reliability, and the volume "
        r"(\num{5}/\num{50}/\num{95} percentiles of the per-sphere volume) of the predicted sets. "
        r"The last three columns report the results on $N = \num{" + fmt_num(n, 0) + r"}$ simulated "
        r"HRC test cycles, measuring how often SARA shield verified the monitored trajectory as "
        r"safe ($c_{\text{safe}}$) and the number of contacts despite a verified trajectory "
        r"($c_{\text{safe}} \land \text{contact}$), and the resulting dangerous-failure rate.}",
        r"    \label{tab:all_conformal_results}",
        r"    \begin{tabular}{lcccccccc}",
        r"        \toprule",
        r"        \multirow{2}[3]{*}{\textbf{Method}} & \multicolumn{2}{c}{Coverage} & "
        r"\multicolumn{3}{c}{$\downarrow$ Volume (\unit{\cubic\meter})} & "
        r"\multirow{2}[3]{*}{$\uparrow$ $c_{\text{safe}}$ (\%)} & "
        r"\multirow{2}[3]{*}{$\downarrow$ $c_{\text{safe}} \land \text{contact}$} & "
        r"\multirow{2}[3]{*}{$\downarrow$ PFH$_\text{D}$ (1/h)} \\",
        r"        \cmidrule(lr){2-3} \cmidrule(lr){4-6}",
        r"         & $\downarrow$ Miss-rate & $\uparrow$ 9s of reliability & 5\% & 50\% & 95\% "
        r"& & & \\",
        r"        \midrule",
    ]
    for m in METHODS:
        if m not in cov and m not in shield:
            continue
        # coverage cells: miss-rate + nines of reliability
        if m in cov:
            c = cov[m]
            miss_s, nines_s = coverage_cells(c["coverage_percent"], m == best_cov)
            v5 = bold(f"{c['volume_p5_m3']:.3f}", m == best_p["p5"])
            v50 = bold(f"{c['volume_p50_m3']:.3f}", m == best_p["p50"])
            v95 = bold(f"{c['volume_p95_m3']:.3f}", m == best_p["p95"])
        else:
            miss_s = nines_s = v5 = v50 = v95 = "X"
        # shield cells
        if m in shield:
            s = shield[m]
            safe_s = bold(f"{s['pct_verified']:.2f}", m == best_safe)
            contact_s = contact_cell(s["n_verified_contact"], m == best_contact)
            pfh_s = sci_cell(s["pfh_d"], m == best_pfh, digits=2)
        else:
            safe_s = contact_s = pfh_s = "X"
        lines.append(
            f"        {METHOD_LABELS[m]} & {miss_s} & {nines_s} & "
            f"{v5} & {v50} & {v95} & {safe_s} & {contact_s} & {pfh_s} \\\\"
        )
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coverage_dir", default="results/final/conformal_prediction_sets",
                   help="Directory with the three per-method coverage CSVs.")
    p.add_argument("--shield_csv", default="results/final/robot_shield/shield_results.csv",
                   help="Shield results CSV written by simulate_robot_shield --results_csv.")
    p.add_argument("--methods", default="all",
                   help="Method rows to include: 'all' (default) or a comma/space separated "
                        "list of method keys (see METHODS in conformal_results_common).")
    p.add_argument("--output", default="results/final/all_conformal_results.tex",
                   help="Output .tex path for the combined table.")
    p.add_argument("--confidence", type=float, default=0.9999,
                   help="Which Clopper-Pearson confidence column to report (must be in the CSV).")
    args = p.parse_args()

    methods = select_methods(args.methods)
    cov = prune_to_methods(read_coverage_by_method(args.coverage_dir), methods)
    shield = prune_to_methods(read_shield_by_method(args.shield_csv, args.confidence), methods)
    if not cov and not shield:
        raise SystemExit("No coverage or shield data found — run the two upstream scripts first.")
    table = generate_table(cov, shield, args.confidence)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(table)
    print(f"Saved combined table to {args.output}\n")
    print(table)


if __name__ == "__main__":
    main()
