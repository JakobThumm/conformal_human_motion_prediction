"""Build the per-stage runtime LaTeX table from the benchmark CSV.

Input:  results/final/runtime/runtime_stages.csv  (examples/benchmark_pipeline_runtime.py)
Output: runtime.tex + runtime_sentence.tex next to it.
"""
import argparse
import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../../results/final/runtime")

# The six pipeline stages, in execution order, with their table labels.
STAGE_LABELS = [
    ("pose_2d", r"2D pose estimation $f_{\text{2D}}$"),
    ("ood_pose", r"Pose OOD detection $\mathrm{SLU}_{\text{2D}}$"),
    ("triangulate", r"Triangulation"),
    ("motion", r"Motion prediction $f_{\text{mot}}$"),
    ("ood_motion", r"Motion OOD detection $\mathrm{SLU}_{\text{mot}}$"),
    ("set", r"Prediction-set computation"),
]


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return {row["stage"]: row for row in csv.DictReader(f)}


def pretty_device(name):
    """'NVIDIA GeForce RTX 5090' -> 'NVIDIA RTX 5090' (the marketing tier is noise here)."""
    return name.replace("GeForce ", "").strip()


def generate_table(rows):
    device = pretty_device(rows["total"]["device"])
    n = rows["total"]["n"]
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Per-stage runtime of our pipeline, measured over \num{" + n +
        r"} pipeline steps on a single " + device + r" after warm-up. The 2D pose estimation"
        r" stage includes human detection and covers both camera views.}",
        r"    \label{tab:runtime}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        Stage & Mean [\si{\milli\second}] & Median [\si{\milli\second}] \\",
        r"        \midrule",
    ]
    for key, label in STAGE_LABELS:
        r = rows[key]
        lines.append(
            f"        {label} & \\num{{{float(r['mean_ms']):.2f}}} $\\pm$ "
            f"\\num{{{float(r['std_ms']):.2f}}} & \\num{{{float(r['median_ms']):.2f}}} \\\\"
        )
    total = rows["total"]
    lines += [
        r"        \midrule",
        f"        Total & \\num{{{float(total['mean_ms']):.2f}}} & "
        f"\\num{{{float(total['median_ms']):.2f}}} \\\\",
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def generate_sentence(rows):
    total = rows["total"]
    mean = float(total["mean_ms"])
    ood = sum(float(rows[k]["mean_ms"]) for k in ("ood_pose", "ood_motion"))
    return (
        f"One pipeline step takes \\SI{{{mean:.0f}}}{{\\milli\\second}} on average "
        f"(\\SI{{{float(total['median_ms']):.0f}}}{{\\milli\\second}} median, "
        f"\\SI{{{1000.0 / mean:.0f}}}{{\\hertz}}) on a single "
        f"{pretty_device(total['device'])}, of which "
        f"\\SI{{{ood:.0f}}}{{\\milli\\second}} (\\SI{{{100.0 * ood / mean:.0f}}}{{\\percent}}) "
        f"are spent on the two OOD monitors.\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate the per-stage runtime LaTeX table")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.path.dirname(__file__), "../../..", results_dir)
    csv_path = os.path.join(results_dir, "runtime_stages.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found -- run examples.benchmark_pipeline_runtime first.")
    rows = read_rows(csv_path)

    table = generate_table(rows)
    table_path = os.path.join(results_dir, "runtime.tex")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Saved table to {table_path}\n")
    print(table)

    sentence = generate_sentence(rows)
    sentence_path = os.path.join(results_dir, "runtime_sentence.tex")
    with open(sentence_path, "w") as f:
        f.write(sentence)
    print(f"Saved sentence to {sentence_path}\n")
    print(sentence)

    wall = rows.get("measured_wall")
    if wall is not None:
        print(f"Cross-check: measured end-to-end {float(wall['mean_ms']):.2f} ms mean vs. "
              f"stage sum {float(rows['total']['mean_ms']):.2f} ms")


if __name__ == "__main__":
    main()
