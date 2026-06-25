"""Turn the robot-shield results CSV (from ``examples.simulate_robot_shield --results_csv``) into a
LaTeX results table.

Each CSV row is one shield run (typically one conformal *set likelihood*). The table reports, per
run, how often the shield verifies a trajectory as safe, the ground-truth (unsafe-)contact rates,
the number of *dangerous failures* (verified-but-unsafe-contact), and the resulting ISO 13849-1
PFH_D upper bound + achievable Performance Level at a chosen Clopper-Pearson confidence.

Usage::

    python -m conformal_human_motion_prediction.generate_plots.generate_robot_shield_results \
        --csv results/motion_prediction/shield_results.csv \
        --output results/final/robot_shield/robot_shield_safety.tex
"""
import argparse
import csv
import os


def _read_rows(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    return rows


def _fmt_sci(x):
    """Format a float as LaTeX scientific notation, e.g. 2.21e-06 -> '2.21 \\times 10^{-6}'."""
    x = float(x)
    if x <= 0:
        return "0"
    exp = 0
    mant = x
    while mant >= 10.0:
        mant /= 10.0; exp += 1
    while mant < 1.0:
        mant *= 10.0; exp -= 1
    return f"{mant:.2f} \\times 10^{{{exp}}}"


def _fmt_pl(pl):
    """'c' -> 'PL c'; 'e (better than required)' -> 'PL e'; 'none (worse than PL a)' -> 'none'."""
    tok = str(pl).strip().split()[0]
    return "none" if tok.lower() == "none" else f"PL {tok}"


def generate_shield_table(csv_path, confidence=0.9999):
    """Build the LaTeX table string from a shield results CSV (one row per run).

    ``confidence`` selects which Clopper-Pearson column (``pfh_d_<C>`` / ``pl_<C>``) to report; it
    must match a confidence the simulation wrote (0.9900 / 0.9990 / 0.9999 by default).
    """
    rows = _read_rows(csv_path)
    tag = f"{float(confidence):.4f}"
    pfh_key, pl_key = f"pfh_d_{tag}", f"pl_{tag}"
    if pfh_key not in rows[0]:
        avail = sorted(k[len("pfh_d_"):] for k in rows[0] if k.startswith("pfh_d_"))
        raise SystemExit(f"confidence {tag} not in CSV; available: {avail}")

    rows.sort(key=lambda r: float(r["set_likelihood"]))
    best = min(range(len(rows)), key=lambda i: float(rows[i][pfh_key]))  # lowest PFH_D = best

    # Context for the caption (assumed shared across rows; taken from the best row).
    n = int(float(rows[best]["total_pairs"]))
    t_cycle = float(rows[best]["t_cycle"])

    def cell(s, is_best):
        return r"\textbf{" + s + "}" if is_best else s

    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \caption{Robot-shield safety evaluation on H36M. Each row is one conformal "
        r"prediction-set likelihood; a \emph{dangerous failure} is a trajectory the shield verifies "
        r"as safe while the ground truth has an unsafe contact "
        r"($v_\mathrm{robot} > V_\mathrm{ISO}$). PFH$_D$ is the one-sided Clopper-Pearson upper "
        f"bound at confidence ${float(confidence)*100:.2f}\\%$ over $N = {n:,}$ test cycles "
        f"($t_\\mathrm{{cycle}} = {t_cycle:g}$ s).}}",
        r"    \label{tab:robot_shield_safety}",
        r"    \begin{tabular}{lccccc}",
        r"        \toprule",
        r"        Set likelihood & $\uparrow$ Verified (\%) & Unsafe contact (\%) & "
        r"$\downarrow$ Dangerous failures & $\downarrow$ PFH$_D$ (1/h) & PL \\",
        r"        \midrule",
    ]
    for i, r in enumerate(rows):
        is_best = i == best
        sl = f"{float(r['set_likelihood']) * 100:.2f}\\%"
        ver = f"{float(r['pct_verified']):.2f}"
        unsafe = f"{float(r['pct_unsafe']):.3f}"
        k = f"{int(float(r['n_verified_unsafe'])):,}"
        pfh = f"${_fmt_sci(r[pfh_key])}$"
        pl = _fmt_pl(r[pl_key])
        lines.append(
            f"        {cell(sl, is_best)} & {ver} & {unsafe} & "
            f"{cell(k, is_best)} & {cell(pfh, is_best)} & {cell(pl, is_best)} \\\\"
        )
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/motion_prediction/shield_results.csv",
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
