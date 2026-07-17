"""Standalone robot-shield safety table for the three methods.

Each CSV row (from ``examples.simulate_robot_shield --results_csv``) is one shield run; the run's
method is derived from (human_set, mask_ood):
  * human_set=sara                     -> ISO 13855 without OOD filtered
  * human_set=conformal, mask_ood=off  -> Ours without OOD filtered
  * human_set=conformal, mask_ood=on   -> Ours with OOD filtered

The table reports, per method, how often the shield verified the monitored trajectory as safe
(c_safe), the number of contacts despite a verified trajectory (c_safe & contact), and the ISO
13849-1 PFH_D upper bound + achievable Performance Level at a chosen Clopper-Pearson confidence.

Usage::

    python -m conformal_human_motion_prediction.generate_plots.generate_robot_shield_results \
        --csv results/final/robot_shield/shield_results.csv \
        --output results/final/robot_shield/robot_shield_safety.tex --confidence 0.9999
"""
import argparse
import os

from conformal_human_motion_prediction.generate_plots.conformal_results_common import (
    METHODS, METHOD_LABELS, bold, fmt_num, fmt_pl, read_shield_by_method,
)


def generate_shield_table(csv_path, confidence=0.9999):
    """Build the LaTeX shield table string from a shield results CSV (one row per method)."""
    shield = read_shield_by_method(csv_path, confidence)
    present = [m for m in METHODS if m in shield]
    if not present:
        raise SystemExit(f"No recognized method rows in {csv_path}")
    best_safe = max(present, key=lambda m: shield[m]["pct_verified"])
    best_contact = min(present, key=lambda m: shield[m]["n_verified_contact"])
    best_pfh = min(present, key=lambda m: shield[m]["pfh_d"])

    n = shield[best_pfh]["total_pairs"]
    t_cycle = shield[best_pfh]["t_cycle"]
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Certification simulation on H36M test data over $N = \num{" + fmt_num(n, 3) + r"}$ "
        r"simulated HRC test cycles ($t_\mathrm{cycle} = " + f"{t_cycle:g}" + r"$ s). "
        r"PFH$_D$ is the one-sided Clopper-Pearson upper bound at confidence "
        f"${float(confidence) * 100:.2f}\\%$.}}",
        r"    \label{tab:robot_shield_safety}",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        \textbf{Method} & $\uparrow$ $c_{\text{safe}}$ (\%) & "
        r"$\downarrow$ $c_{\text{safe}} \land \text{contact}$ & "
        r"$\downarrow$ PFH$_D$ (1/h) & PL \\",
        r"        \midrule",
    ]
    for m in METHODS:
        if m not in shield:
            continue
        s = shield[m]
        safe_s = bold(f"{s['pct_verified']:.2f}", m == best_safe)
        contact_s = bold(f"{s['n_verified_contact']:,}", m == best_contact)
        pfh_s = bold(f"\\num{{{fmt_num(s['pfh_d'])}}}", m == best_pfh)
        pl_s = bold(fmt_pl(s["pl"]), m == best_pfh)
        lines.append(f"        {METHOD_LABELS[m]} & {safe_s} & {contact_s} & {pfh_s} & {pl_s} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/final/robot_shield/shield_results.csv",
                   help="Shield results CSV written by simulate_robot_shield --results_csv.")
    p.add_argument("--output", default=None,
                   help="Output .tex path (default: alongside the CSV as <stem>.tex).")
    p.add_argument("--confidence", type=float, default=0.9999,
                   help="Which Clopper-Pearson confidence column to report (must be in the CSV).")
    args = p.parse_args()

    table = generate_shield_table(args.csv, confidence=args.confidence)
    out = args.output or os.path.splitext(args.csv)[0] + ".tex"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(table)
    print(f"Saved table to {out}\n")
    print(table)


if __name__ == "__main__":
    main()
