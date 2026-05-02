"""
evaluate.py
-----------
Evaluates a fine-tuned (or baseline) MarianMT model on all three test sets:

  1. FLORES-200 devtest   — general-domain (news, Wikipedia)
  2. WMT IT domain test   — in-domain software (OPUS proxy)
  3. Custom dataset       — provided Dataset_Challenge_1.xlsx

Metrics reported
~~~~~~~~~~~~~~~~
  • BLEU      (sacreBLEU, tokenized)
  • chrF2     (character n-gram F-score, robust to morphological variation)
  • TER       (Translation Edit Rate — lower is better)
  • COMET     (optional neural metric; requires 'unbabel-comet' package)

Usage
~~~~~
    # Evaluate fine-tuned model on all sets
    python scripts/evaluate.py \\
        --model_dir    models/marian-en-nl-ft/final \\
        --flores_tsv   data/flores/devtest.tsv       \\
        --wmt_tsv      data/wmt_it/test.tsv          \\
        --custom_tsv   data/custom/test.tsv          \\
        --output_dir   results/

    # Compare baseline vs fine-tuned (run twice, different --model_dir)
    python scripts/evaluate.py --model_dir Helsinki-NLP/opus-mt-en-nl ...

Requirements
~~~~~~~~~~~~
    pip install transformers sacrebleu sentencepiece pandas torch
    pip install unbabel-comet  # optional, for COMET
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# COMET (optional)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Translation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: str, device: str):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model     = MarianMTModel.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


def translate_batch(
    srcs: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: str,
    batch_size: int = 32,
    num_beams: int  = 4,
    max_length: int = 256,
) -> list[str]:
    """Translate a list of source strings, returning hypothesis strings."""
    all_preds = []
    for i in range(0, len(srcs), batch_size):
        chunk = srcs[i : i + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_preds.extend(decoded)
        if (i // batch_size) % 10 == 0:
            print(f"  Translated {min(i + batch_size, len(srcs))}/{len(srcs)}")
    return all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    hypotheses: list[str],
    references: list[str],
    sources: list[str] | None = None,
    use_comet: bool = False,
) -> dict:
    """Return a dict with BLEU, chrF2, TER (and optionally COMET)."""
    assert len(hypotheses) == len(references), "Length mismatch"

    bleu  = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf  = sacrebleu.corpus_chrf(hypotheses, [references])
    ter   = sacrebleu.corpus_ter(hypotheses, [references])

    metrics = {
        "num_sentences": len(hypotheses),
        "BLEU":  round(bleu.score, 2),
        "BLEU_BP": round(bleu.bp, 4),
        "chrF2": round(chrf.score, 2),
        "TER":   round(ter.score,  2),
    }

    if use_comet and COMET_AVAILABLE and sources is not None:
        print("  Computing COMET (this may take a minute)…")
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model      = load_from_checkpoint(comet_model_path)
        comet_data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        comet_output = comet_model.predict(comet_data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
        metrics["COMET"] = round(float(comet_output["system_score"]), 4)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Per-sentence detail
# ─────────────────────────────────────────────────────────────────────────────

def sentence_level_df(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
) -> pd.DataFrame:
    """Build a DataFrame with per-sentence BLEU and chrF2."""
    rows = []
    for src, hyp, ref in zip(sources, hypotheses, references):
        s_bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
        s_chrf = sacrebleu.sentence_chrf(hyp, [ref]).score
        rows.append({
            "source":     src,
            "hypothesis": hyp,
            "reference":  ref,
            "sent_BLEU":  round(s_bleu, 2),
            "sent_chrF2": round(s_chrf, 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Load TSV
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv(path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str).dropna()
    srcs = df.iloc[:, 0].tolist()
    refs = df.iloc[:, 1].tolist()
    return srcs, refs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one split
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(
    name: str,
    tsv_path: str,
    tokenizer,
    model,
    device: str,
    output_dir: Path,
    use_comet: bool,
    batch_size: int,
    num_beams: int,
) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Evaluating : {name}")
    print(f"  File       : {tsv_path}")

    srcs, refs = load_tsv(tsv_path)
    print(f"  Sentences  : {len(srcs)}")

    hyps = translate_batch(srcs, tokenizer, model, device,
                           batch_size=batch_size, num_beams=num_beams)

    metrics = compute_metrics(hyps, refs, sources=srcs, use_comet=use_comet)

    print(f"\n  Results for [{name}]:")
    print(f"    BLEU  : {metrics['BLEU']}")
    print(f"    chrF2 : {metrics['chrF2']}")
    print(f"    TER   : {metrics['TER']}")
    if "COMET" in metrics:
        print(f"    COMET : {metrics['COMET']}")

    # Save detailed per-sentence results
    detail_df = sentence_level_df(srcs, hyps, refs)
    detail_path = output_dir / f"{name.replace(' ', '_')}_detail.tsv"
    detail_df.to_csv(detail_path, sep="\t", index=False)
    print(f"\n  Per-sentence details → {detail_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MT model on all test sets")
    p.add_argument("--model_dir",   required=True,
                   help="Path to fine-tuned model dir OR HuggingFace model ID")
    p.add_argument("--flores_tsv",  default="data/flores/devtest.tsv")
    p.add_argument("--wmt_tsv",     default="data/wmt_it/test.tsv")
    p.add_argument("--custom_tsv",  default="data/custom/test.tsv")
    p.add_argument("--output_dir",  default="results/")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_beams",   type=int, default=4)
    p.add_argument("--use_comet",   action="store_true",
                   help="Compute COMET score (requires unbabel-comet)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice   : {device}")
    print(f"Model    : {args.model_dir}")

    tokenizer, model = load_model(args.model_dir, device)

    all_results = {}

    test_splits = [
        ("FLORES_devtest",   args.flores_tsv),
        ("WMT_IT_test",      args.wmt_tsv),
        ("Custom_Challenge1",args.custom_tsv),
    ]

    for name, path in test_splits:
        if not Path(path).exists():
            print(f"\n[SKIP] {name}: file not found at {path}")
            continue
        metrics = evaluate_split(
            name, path, tokenizer, model, device,
            output_dir, args.use_comet,
            args.batch_size, args.num_beams,
        )
        all_results[name] = metrics

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    header = f"{'Split':<25} {'BLEU':>7} {'chrF2':>7} {'TER':>7}"
    if args.use_comet and COMET_AVAILABLE:
        header += f" {'COMET':>8}"
    print(header)
    print("─" * 60)

    for name, m in all_results.items():
        row = f"{name:<25} {m['BLEU']:>7} {m['chrF2']:>7} {m['TER']:>7}"
        if "COMET" in m:
            row += f" {m['COMET']:>8}"
        print(row)

    # ── Save JSON ────────────────────────────────────────────────────────────
    results_path = output_dir / "metrics_summary.json"
    with open(results_path, "w") as f:
        json.dump(
            {"model": args.model_dir, "results": all_results},
            f, indent=2
        )
    print(f"\n✓ Summary saved → {results_path}")


if __name__ == "__main__":
    main()
