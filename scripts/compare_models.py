"""
compare_models.py
-----------------
Convenience wrapper that runs evaluate.py for BOTH the baseline model
and the fine-tuned model, then produces a side-by-side comparison table
and a bar chart saved to results/comparison.png.

Usage
~~~~~
    python scripts/compare_models.py \\
        --baseline_dir   Helsinki-NLP/opus-mt-en-nl \\
        --finetuned_dir  models/marian-en-nl-ft/final \\
        --flores_tsv     data/flores/devtest.tsv \\
        --wmt_tsv        data/wmt_it/test.tsv    \\
        --custom_tsv     data/custom/test.tsv    \\
        --output_dir     results/

Requirements
~~~~~~~~~~~~
    pip install matplotlib pandas
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_evaluate(model_dir: str, tag: str, args) -> dict:
    """Call evaluate.py as a subprocess and load its JSON output."""
    out_dir = Path(args.output_dir) / tag
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--model_dir",  model_dir,
        "--flores_tsv", args.flores_tsv,
        "--wmt_tsv",    args.wmt_tsv,
        "--custom_tsv", args.custom_tsv,
        "--output_dir", str(out_dir),
        "--batch_size", str(args.batch_size),
        "--num_beams",  str(args.num_beams),
    ]
    if args.use_comet:
        cmd.append("--use_comet")

    print(f"\n{'#'*60}")
    print(f"  Running evaluation for: {tag}")
    print(f"{'#'*60}")
    subprocess.run(cmd, check=True)

    results_file = out_dir / "metrics_summary.json"
    with open(results_file) as f:
        return json.load(f)["results"]


def build_comparison_table(baseline: dict, finetuned: dict) -> pd.DataFrame:
    rows = []
    for split in set(list(baseline.keys()) + list(finetuned.keys())):
        for metric in ["BLEU", "chrF2", "TER"]:
            b_val = baseline.get(split, {}).get(metric)
            f_val = finetuned.get(split, {}).get(metric)
            if b_val is None or f_val is None:
                continue
            # For TER, lower is better so delta sign is inverted
            delta = f_val - b_val if metric != "TER" else b_val - f_val
            rows.append({
                "Split":    split,
                "Metric":   metric,
                "Baseline": b_val,
                "Fine-tuned": f_val,
                "Δ (ft-base)": round(f_val - b_val, 2),
                "Improved": "✓" if delta > 0 else ("=" if delta == 0 else "✗"),
            })
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed — skipping chart.")
        return

    metrics  = df["Metric"].unique()
    splits   = df["Split"].unique()
    n_splits = len(splits)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    colors = {"Baseline": "#4C72B0", "Fine-tuned": "#DD8452"}
    bar_width = 0.35

    for ax, metric in zip(axes, metrics):
        sub = df[df["Metric"] == metric].copy()
        x = range(len(sub))
        ax.bar([i - bar_width / 2 for i in x], sub["Baseline"],
               bar_width, label="Baseline", color=colors["Baseline"])
        ax.bar([i + bar_width / 2 for i in x], sub["Fine-tuned"],
               bar_width, label="Fine-tuned", color=colors["Fine-tuned"])
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["Split"].tolist(), rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        if metric == "TER":
            ax.set_ylabel("Score (lower = better)")
        else:
            ax.set_ylabel("Score (higher = better)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("Baseline vs Fine-tuned: EN→NL Translation Quality",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart_path = output_dir / "comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {chart_path}")


def main():
    p = argparse.ArgumentParser(description="Compare baseline vs fine-tuned")
    p.add_argument("--baseline_dir",  default="Helsinki-NLP/opus-mt-en-nl")
    p.add_argument("--finetuned_dir", required=True)
    p.add_argument("--flores_tsv",    default="data/flores/devtest.tsv")
    p.add_argument("--wmt_tsv",       default="data/wmt_it/test.tsv")
    p.add_argument("--custom_tsv",    default="data/custom/test.tsv")
    p.add_argument("--output_dir",    default="results/")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--num_beams",     type=int, default=4)
    p.add_argument("--use_comet",     action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_results  = run_evaluate(args.baseline_dir,  "baseline",  args)
    finetuned_results = run_evaluate(args.finetuned_dir, "finetuned", args)

    df = build_comparison_table(baseline_results, finetuned_results)

    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    # Save
    table_path = out_dir / "comparison_table.tsv"
    df.to_csv(table_path, sep="\t", index=False)
    print(f"\n✓ Table saved → {table_path}")

    plot_comparison(df, out_dir)


if __name__ == "__main__":
    main()
