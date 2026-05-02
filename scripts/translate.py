"""
translate.py
------------
Simple inference script: translates a list of English strings to Dutch
using a fine-tuned (or baseline) MarianMT model.

Usage
~~~~~
    # Translate a single sentence
    python scripts/translate.py \\
        --model_dir models/marian-en-nl-ft/final \\
        --text "Hold the Control key to start drag & drop"

    # Translate all lines in a text file (one sentence per line)
    python scripts/translate.py \\
        --model_dir models/marian-en-nl-ft/final \\
        --input_file my_strings.txt \\
        --output_file my_strings_nl.txt

    # Translate a TSV (reads 'src' column, writes 'hypothesis' column)
    python scripts/translate.py \\
        --model_dir models/marian-en-nl-ft/final \\
        --input_tsv  data/custom/test.tsv \\
        --output_tsv results/custom_hypotheses.tsv

Requirements
~~~~~~~~~~~~
    pip install transformers torch sentencepiece
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer


def load_model(model_dir: str, device: str):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


def translate(
    texts: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: str,
    batch_size: int = 32,
    num_beams: int  = 4,
    max_length: int = 256,
) -> list[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
            )
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results


def main():
    p = argparse.ArgumentParser(description="EN→NL translation inference")
    p.add_argument("--model_dir",   required=True)
    p.add_argument("--text",        default=None,
                   help="Single sentence to translate (printed to stdout)")
    p.add_argument("--input_file",  default=None,
                   help="Plain text file (one sentence per line)")
    p.add_argument("--output_file", default=None)
    p.add_argument("--input_tsv",   default=None,
                   help="TSV with 'src' or first column as source")
    p.add_argument("--output_tsv",  default=None)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_beams",   type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_dir, device)

    # ── Single sentence ──────────────────────────────────────────────────────
    if args.text:
        hyps = translate([args.text], tokenizer, model, device,
                         args.batch_size, args.num_beams)
        print(hyps[0])
        return

    # ── Plain text file ──────────────────────────────────────────────────────
    if args.input_file:
        lines = Path(args.input_file).read_text(encoding="utf-8").splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        hyps  = translate(lines, tokenizer, model, device,
                          args.batch_size, args.num_beams)
        output = "\n".join(hyps)
        if args.output_file:
            Path(args.output_file).write_text(output, encoding="utf-8")
            print(f"✓ Written to {args.output_file}")
        else:
            print(output)
        return

    # ── TSV ──────────────────────────────────────────────────────────────────
    if args.input_tsv:
        import pandas as pd
        df   = pd.read_csv(args.input_tsv, sep="\t", dtype=str).dropna()
        srcs = df.iloc[:, 0].tolist()
        hyps = translate(srcs, tokenizer, model, device,
                         args.batch_size, args.num_beams)
        df["hypothesis"] = hyps
        out_path = args.output_tsv or args.input_tsv.replace(".tsv", "_translated.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        print(f"✓ Saved to {out_path}")
        return

    print("Nothing to do. Pass --text, --input_file, or --input_tsv.")
    sys.exit(1)


if __name__ == "__main__":
    main()
