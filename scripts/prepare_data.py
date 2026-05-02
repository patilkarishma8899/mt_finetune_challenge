"""
prepare_data.py
---------------
Downloads and prepares all three datasets required for Challenge 1:

  1. WMT 2016 IT domain data  (training + in-domain test)
  2. FLORES-200 devtest       (general-domain test)
  3. Custom dataset           (Dataset_Challenge_1.xlsx — provided test set)

Outputs everything as plain TSV files under  data/  so that train.py
can load them without touching HuggingFace datasets at runtime.

Usage
-----
    python scripts/prepare_data.py --data_dir data/

Requirements
------------
    pip install datasets openpyxl pandas sacremoses
"""

import argparse
import os
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_tsv(pairs: list[tuple[str, str]], path: Path) -> None:
    """Save a list of (src, tgt) pairs as a two-column TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("src\ttgt\n")
        for src, tgt in pairs:
            src = src.replace("\t", " ").replace("\n", " ").strip()
            tgt = tgt.replace("\t", " ").replace("\n", " ").strip()
            if src and tgt:
                f.write(f"{src}\t{tgt}\n")
    print(f"  Saved {len(pairs):>6,} pairs → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. WMT 2016 IT domain  (en → nl)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_wmt16_it(data_dir: Path) -> None:
    """
    Loads the WMT 2016 IT-domain translation memory from HuggingFace.
    The 'wmt16' config for en-nl only has news domain; we use the
    'opus_books' and 'ccaligned_multilingual' or the dedicated IT-domain
    dataset from OPUS (opus100 en-nl subset as proxy when wmt16 IT
    is unavailable directly).

    Strategy used here:
      - Primary  : datasets 'wmt16' with language_pair='de-en' is not
                   available for nl; instead we load the OPUS IT corpus
                   via the 'Helsinki-NLP/opus-100' dataset (en-nl split)
                   which is derived from OPUS and covers software / IT text.
      - We keep a large training split and a held-out dev split.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("\n[1/3] Downloading OPUS-100 en-nl (IT/software proxy for WMT-IT)…")
    # opus-100 en-nl contains ~1M sentence pairs from IT, subtitle, news domains
    ds = load_dataset("Helsinki-NLP/opus-100", "en-nl", trust_remote_code=True)

    def extract(split_name):
        split = ds[split_name]
        pairs = []
        for ex in split:
            t = ex.get("translation", {})
            src = t.get("en", "").strip()
            tgt = t.get("nl", "").strip()
            if src and tgt and len(src) < 500 and len(tgt) < 500:
                pairs.append((src, tgt))
        return pairs

    train_pairs = extract("train")
    # Limit to 200k for manageable fine-tuning; increase if GPU allows
    train_pairs = train_pairs[:200_000]
    val_pairs   = extract("validation")
    test_pairs  = extract("test")

    save_tsv(train_pairs, data_dir / "wmt_it" / "train.tsv")
    save_tsv(val_pairs,   data_dir / "wmt_it" / "val.tsv")
    save_tsv(test_pairs,  data_dir / "wmt_it" / "test.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FLORES-200 devtest  (general domain)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_flores(data_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("\n[2/3] Downloading FLORES-200 devtest (eng_Latn → nld_Latn)…")
    # FLORES-200 is sentence-aligned across all languages; we load eng + nld
    flores_en = load_dataset("facebook/flores", "eng_Latn", trust_remote_code=True)
    flores_nl = load_dataset("facebook/flores", "nld_Latn", trust_remote_code=True)

    pairs = []
    for split in ["devtest"]:
        en_rows = flores_en[split]
        nl_rows = flores_nl[split]
        for en_ex, nl_ex in zip(en_rows, nl_rows):
            src = en_ex["sentence"].strip()
            tgt = nl_ex["sentence"].strip()
            if src and tgt:
                pairs.append((src, tgt))

    save_tsv(pairs, data_dir / "flores" / "devtest.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom dataset  (Dataset_Challenge_1.xlsx)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_custom(xlsx_path: Path, data_dir: Path) -> None:
    print(f"\n[3/3] Loading custom dataset from {xlsx_path}…")
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    src_col = "English Source"
    tgt_col = "Reference Translation"

    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError(
            f"Expected columns '{src_col}' and '{tgt_col}', "
            f"got: {list(df.columns)}"
        )

    pairs = []
    for _, row in df.iterrows():
        src = str(row[src_col]).strip()
        tgt = str(row[tgt_col]).strip()
        if src and tgt and src != "nan" and tgt != "nan":
            pairs.append((src, tgt))

    save_tsv(pairs, data_dir / "custom" / "test.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare all MT datasets")
    parser.add_argument("--data_dir",   default="data",
                        help="Root directory to write TSV files into")
    parser.add_argument("--xlsx_path",  default="data/Dataset_Challenge_1.xlsx",
                        help="Path to the provided custom test set XLSX")
    parser.add_argument("--skip_wmt",   action="store_true",
                        help="Skip downloading WMT/OPUS data (use if already done)")
    parser.add_argument("--skip_flores",action="store_true",
                        help="Skip downloading FLORES data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_wmt:
        prepare_wmt16_it(data_dir)

    if not args.skip_flores:
        prepare_flores(data_dir)

    xlsx_path = Path(args.xlsx_path)
    if xlsx_path.exists():
        prepare_custom(xlsx_path, data_dir)
    else:
        print(f"\n[3/3] SKIPPED — custom XLSX not found at {xlsx_path}")
        print("      Copy Dataset_Challenge_1.xlsx there and re-run, or pass --xlsx_path")

    print("\n✓ All datasets prepared.")


if __name__ == "__main__":
    main()
