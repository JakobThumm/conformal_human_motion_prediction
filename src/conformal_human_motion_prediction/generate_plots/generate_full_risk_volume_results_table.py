"""Combined results table using the VOLUME-WEIGHTED risk estimator's certification columns.

Mirrors :mod:`generate_full_conformal_prediction_results_table`, which fuses the conformal
prediction-set coverage/volume half with the *naive* shield simulation's certification half
(``c_safe``, ``c_safe & contact``, PFH_D over N dependent test cycles). Here the right half instead
reports the factorisation that ``examples.simulate_shield_failure_risk_volume`` measures:

    PFH_D <= N_h * P_up(F) * [V(W)/V(F)] * P_up(D | F, W)

so the table shows *where* each method's risk comes from rather than one pooled count:

  * ``P(F)``      -- one-sided Clopper-Pearson upper limit on the prediction-failure probability,
                     computed on the recorded windows and deflated to
                     ``n_eff = N_F / (2 L_corr)`` for the overlap between windows. DATA-limited.
  * ``P(W|F)``    -- the witness-region volume ratio, in closed form (no Monte-Carlo error). Purely
                     geometric, and it differs per method only because each method fails on
                     *different* windows, whose true-occupancy spheres differ in size.
  * ``k_D``       -- dangerous trials observed among the ``N_W`` i.i.d. trials drawn from W (NOT
                     the uniform-placement count; that is ``N_D = N_W / P(W|F)``). These
                     trials really are independent draws, so their binomial bound is exact rather
                     than the cluster-collapsed one the naive estimator has to use.
  * ``PFH_D``/PL  -- the composed bound at joint confidence ``1 - eps_D``.

Both halves are keyed by the same method keys as the naive table (see
``conformal_results_common.shield_method_key``), so this only reads CSVs the upstream scripts
already produced -- run them first.

Usage::

    python -m conformal_human_motion_prediction.generate_plots.generate_full_risk_volume_results_table \
        --coverage_dir results/final/conformal_prediction_sets \
        --risk_csv results/final/robot_shield_risk_volume/shield_risk_volume_results.csv \
        --output results/final/all_conformal_results_risk_volume.tex
"""
import argparse
import csv
import os

from conformal_human_motion_prediction.generate_plots.conformal_results_common import (
    METHOD_LABELS, METHODS, best_coverage_method, bold, coverage_cells, fmt_pl,
    prune_to_methods, read_coverage_by_method, sci_cell, select_methods, shield_method_key,
)

# ISO 13849-1 PL d line. A row whose PFH_D bound exceeds it is coloured red: it does not certify.
PFH_D_TARGET = 1e-6
# Rendered below a \midrule as a separate block AND excluded from the best-in-column bolding.
# The no-calibration ablation replaces the conformal quantile with the analytic Gaussian factor
# alpha_{k,chi}^j, so it carries no coverage guarantee at all and fails the PL d line; letting it
# win "smallest volume" against methods that do certify would misread the table.
ABLATION_BLOCK = ("ours_uncal_ood", "ours_uncal_no_ood")
# Row order for THIS table, which is the paper's tab:all_conformal_results: the two
# large-set/low-PFH_D methods first, then the two conditional-conformal rows, then the ablation
# block below the rule. Anything not listed falls back to the shared METHODS order.
ROW_ORDER = ("iso_no_ood", "ours_max_ood", "ours_no_ood", "ours_ood")
# Presentation-only row labels for this table (shorter than the canonical METHOD_LABELS, which the
# naive table still uses). "No conformal prediction" names what the alpha_{k,chi}^j ablation IS
# rather than which method it ablates, since it sits in its own block.
LABEL_OVERRIDE = {
    "ours_max_ood": r"Ours ($\alpha_{\max}$)",
    "ours_uncal_ood": r"No conformal prediction $\left(\alpha_{k, \chi}^j\right)$",
}


def row_label(m):
    return LABEL_OVERRIDE.get(m, METHOD_LABELS[m])


def order_key(m):
    """Sort key putting ROW_ORDER first, then the shared METHODS order."""
    return (ROW_ORDER.index(m), 0) if m in ROW_ORDER else (len(ROW_ORDER), METHODS.index(m))


def read_risk_volume_by_method(csv_path):
    """Read the volume-estimator CSV -> ``{method_key: parsed row}``.

    The CSV is appended to, so a method may appear several times (a smoke run followed by the
    production sweep, say); the LAST row for a method wins, matching
    ``conformal_results_common.read_shield_by_method``. Rows are keyed by the same
    ``shield_method_key`` the naive table uses, which needs ``set_kind`` to carry
    ``max_conformal`` / ``uncalibrated`` for the two ablations -- a CSV written before
    simulate_shield_failure_risk_volume recorded those will collapse all three conformal rows onto
    one key, which this reports rather than silently mislabelling.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    need = ("p_f_upper", "p_w", "k_D", "n_trials", "pfh_d_upper", "pl")
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit(f"{csv_path} lacks {missing} -- it was not written by "
                         f"examples.simulate_shield_failure_risk_volume.")
    out, seen = {}, {}
    for i, r in enumerate(rows):
        key = shield_method_key(r)
        seen.setdefault(key, []).append(i)
        out[key] = {
            "n_windows": int(float(r["n_windows"])),
            "k_F": int(float(r["k_F"])),
            "p_f_hat": float(r["p_f_hat"]),
            "p_f_upper": float(r["p_f_upper"]),
            "l_corr": float(r.get("l_corr", "nan")),
            "n_eff": float(r.get("n_eff", "nan")),
            "p_w": float(r["p_w"]),
            "n_trials": int(float(r["n_trials"])),
            "k_D": int(float(r["k_D"])),
            "p_dgw_upper": float(r["p_d_given_f_w_upper"]),
            "pfh_d": float(r["pfh_d_upper"]),
            "pfh_d_hat": float(r.get("pfh_d_hat", "nan")),
            "pl": r["pl"],
            "epsilon_d": float(r.get("epsilon_d", "nan")),
            "t_cycle": float(r.get("t_cycle", "nan")),
            "calibrator": r.get("conformal_calibrator", ""),
        }
    for key, idxs in seen.items():
        if len(idxs) > 1:
            print(f"note: {key} appears in {len(idxs)} rows of {csv_path} "
                  f"(CSV lines {[i + 2 for i in idxs]}); using the last one "
                  f"(N_W = {out[key]['n_trials']:,}, k_D = {out[key]['k_D']:,})")
    return out


def generate_table(cov, risk):
    """Build the combined LaTeX table from coverage + volume-risk dicts (both keyed by method)."""
    main = [m for m in METHODS if m not in ABLATION_BLOCK]
    cov_present = [m for m in main if m in cov]
    rk_present = [m for m in main if m in risk]
    best_cov = best_coverage_method(cov, cov_present)
    best_p = {q: (min(cov_present, key=lambda m: cov[m][f"volume_{q}_m3"]) if cov_present else None)
              for q in ("p5", "p50", "p95")}
    best_pf = min(rk_present, key=lambda m: risk[m]["p_f_upper"]) if rk_present else None
    best_pfh = min(rk_present, key=lambda m: risk[m]["pfh_d"]) if rk_present else None
    # k_D ties at 0 whenever the rare event stays unresolved. Bold on the VALUE so every tied row
    # is marked, rather than singling out whichever one comes first.
    min_kd = min((risk[m]["k_D"] for m in rk_present), default=None)

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Motion prediction results on H36M test data and PFH$_\text{D}$ "
        r"evaluation in random placement experiments.}",
        r"    \label{tab:all_conformal_results}",
        r"    \begin{tabular}{lcccccccc}",
        r"        \toprule",
        r"        \multirow{2}[3]{*}{Method} & \multicolumn{2}{c}{Coverage} & "
        r"\multicolumn{3}{c}{$\downarrow$ Volume (\unit{\cubic\meter})} & "
        r"\multirow{2}[3]{*}{$\downarrow$ $P(F)$} & "
        r"\multirow{2}[3]{*}{$\downarrow$ $k_D$} & "
        r"\multirow{2}[3]{*}{$\downarrow$ PFH$_\text{D}$ (1/h)} \\",
        r"        \cmidrule(lr){2-3} \cmidrule(lr){4-6}",
        r"         & $\downarrow$ Miss rate & $\uparrow$ Nines of reliability & 5\% & 50\% "
        r"& 95\% & & & \\",
        r"        \midrule",
    ]

    def render(m):
        """One LaTeX row. ``X`` marks a half with no CSV; the best-in-column marks skip the
        ablation block (see ABLATION_BLOCK)."""
        if m in cov:
            c = cov[m]
            miss_s, nines_s = coverage_cells(c["coverage_percent"], m == best_cov)
            v5 = bold(f"{c['volume_p5_m3']:.3f}", m == best_p["p5"])
            v50 = bold(f"{c['volume_p50_m3']:.3f}", m == best_p["p50"])
            v95 = bold(f"{c['volume_p95_m3']:.3f}", m == best_p["p95"])
        else:
            miss_s = nines_s = v5 = v50 = v95 = "X"
        if m in risk:
            r = risk[m]
            pf_s = sci_cell(r["p_f_upper"], m == best_pf, digits=2)
            kd = r["k_D"]
            # siunitx does the thousands separator; below 1000 a bare digit string is cleaner.
            kd_s = bold(f"\\num{{{kd}}}" if kd >= 1000 else f"{kd}", kd == min_kd)
            pfh_s = sci_cell(r["pfh_d"], m == best_pfh, digits=2)
            if r["pfh_d"] >= PFH_D_TARGET:      # does not reach PL d -> flag it
                pfh_s = r"\textcolor{red}{" + pfh_s + "}"
        else:
            pf_s = kd_s = pfh_s = "X"
        return (f"        {row_label(m)} & {miss_s} & {nines_s} & "
                f"{v5} & {v50} & {v95} & {pf_s} & {kd_s} & {pfh_s} \\\\")

    for m in sorted(main, key=order_key):
        if m in cov or m in risk:
            lines.append(render(m))
    tail = [m for m in ABLATION_BLOCK if m in cov or m in risk]
    if tail:
        lines.append(r"        \midrule")
        lines += [render(m) for m in sorted(tail, key=METHODS.index)]
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def print_summary(risk):
    """Plain-text audit of the factorisation behind each row (not part of the .tex).

    The short caption deliberately omits the run constants, so they are printed here: they now
    have to be stated in the prose instead, or 'k_D = 4' has no denominator.
    """
    present = [m for m in METHODS if m in risk]
    ref = risk[present[0]] if present else {}
    t_cycle = ref.get("t_cycle", float("nan"))
    # N_W, t_cycle and eps_D are per-RUN constants; P(W|F) -- and hence the equivalent uniform
    # count N_D = N_W / P(W|F) -- is per-METHOD, because each method fails on different windows.
    # So N_D belongs in the per-row table below, not here.
    for key, label in (("n_trials", "N_W"), ("t_cycle", "t_cycle"), ("epsilon_d", "eps_D")):
        vals = {risk[m][key] for m in present}
        if len(vals) > 1:
            print(f"warning: rows disagree on {label}: {sorted(vals)} -- the shared constants "
                  f"below are taken from {row_label(present[0])}")
    print(f"\nrun constants NOT in the caption -- state these in the text:"
          f"\n  N_W = {ref.get('n_trials', 0):,} trials drawn from W"
          f"\n  t_cycle = {t_cycle:g} s -> N_h = {3600.0 / t_cycle:,.0f} 1/h, joint confidence "
          f"{100.0 * (1.0 - ref.get('epsilon_d', float('nan'))):.3f}%")
    print(f"\n{'method':38s} {'N_F':>8s} {'k_F':>7s} {'L_corr':>7s} {'P_up(F)':>10s} "
          f"{'P(W|F)':>10s} {'N_W':>12s} {'N_D=N_W/P(W|F)':>15s} {'k_D':>8s} "
          f"{'PFH_D up':>11s} PL")
    for m in sorted(METHODS, key=order_key):
        if m not in risk:
            continue
        r = risk[m]
        print(f"{row_label(m)[:38]:38s} {r['n_windows']:>8,} {r['k_F']:>7,} "
              f"{r['l_corr']:>7.2f} {r['p_f_upper']:>10.3e} {r['p_w']:>10.3e} "
              f"{r['n_trials']:>12,} {r['n_trials'] / r['p_w']:>15,.0f} "
              f"{r['k_D']:>8,} {r['pfh_d']:>11.3e} {fmt_pl(r['pl'])}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coverage_dir", default="results/final/conformal_prediction_sets",
                   help="Directory with the per-method coverage CSVs.")
    p.add_argument("--risk_csv",
                   default="results/final/robot_shield_risk_volume/shield_risk_volume_results.csv",
                   help="CSV written by simulate_shield_failure_risk_volume --results_csv.")
    p.add_argument("--methods", default="all",
                   help="Method rows to include: 'all' (default) or a comma/space separated list "
                        "of method keys (see METHODS in conformal_results_common).")
    p.add_argument("--output", default="results/final/all_conformal_results_risk_volume.tex",
                   help="Output .tex path for the combined table.")
    args = p.parse_args()

    methods = select_methods(args.methods)
    cov = prune_to_methods(read_coverage_by_method(args.coverage_dir), methods)
    risk = prune_to_methods(read_risk_volume_by_method(args.risk_csv), methods)
    if not cov and not risk:
        raise SystemExit("No coverage or risk data found -- run the two upstream scripts first.")
    missing = [m for m in methods if m not in risk]
    if missing:
        print(f"note: no volume-risk row for {missing} -- those cells are rendered 'X'")
    table = generate_table(cov, risk)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(table)
    print_summary(risk)
    print(f"\nSaved combined table to {args.output}\n")
    print(table)


if __name__ == "__main__":
    main()
