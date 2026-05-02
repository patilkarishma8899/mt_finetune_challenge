"""
train.py
--------
Fine-tunes Helsinki-NLP/opus-mt-en-nl (MarianMT, ~74M params) on the
software/IT domain using PyTorch Lightning.

Architecture
~~~~~~~~~~~~
  • Base model   : Helsinki-NLP/opus-mt-en-nl
                   — encoder-decoder Transformer pre-trained on OPUS en-nl
                   — already produces good general EN→NL; fine-tuning adapts
                     it to mobile/software UI text
  • Framework    : PyTorch Lightning (LightningModule + Trainer)
  • Tokenizer    : MarianTokenizer (SentencePiece, bundled with the model)

Training strategy
~~~~~~~~~~~~~~~~~
  • Full-parameter fine-tuning (small model, ~74M params — fits on 1 GPU)
  • Label smoothing = 0.1  to prevent overconfidence on small IT corpus
  • Linear warmup for 500 steps, then cosine decay
  • Checkpoint on best validation BLEU

Usage
~~~~~
    # Minimal (all defaults)
    python scripts/train.py

    # Custom
    python scripts/train.py \\
        --train_tsv   data/wmt_it/train.tsv \\
        --val_tsv     data/wmt_it/val.tsv   \\
        --output_dir  models/marian-en-nl-ft \\
        --max_epochs  5                      \\
        --batch_size  32                     \\
        --lr          3e-5                   \\
        --max_src_len 128

Requirements
~~~~~~~~~~~~
    pip install torch pytorch-lightning transformers sacrebleu sentencepiece
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)

# Optional: nice progress bars
try:
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """Reads a two-column TSV (src, tgt) and tokenises on the fly."""

    def __init__(
        self,
        tsv_path: str,
        tokenizer: MarianTokenizer,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        self.tokenizer   = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        df = pd.read_csv(tsv_path, sep="\t", dtype=str).dropna()
        # Accept either "src/tgt" or "English Source/Reference Translation"
        if "src" in df.columns:
            self.srcs = df["src"].tolist()
            self.tgts = df["tgt"].tolist()
        else:
            self.srcs = df.iloc[:, 0].tolist()
            self.tgts = df.iloc[:, 1].tolist()

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        return self.srcs[idx], self.tgts[idx]

    def collate_fn(self, batch):
        srcs, tgts = zip(*batch)

        model_inputs = self.tokenizer(
            list(srcs),
            max_length=self.max_src_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                list(tgts),
                max_length=self.max_tgt_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        # Replace padding token id in labels with -100 (ignored by loss)
        label_ids = labels["input_ids"].clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels":         label_ids,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────

class MarianFineTuner(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.tokenizer = MarianTokenizer.from_pretrained(hparams.model_name)
        self.model     = MarianMTModel.from_pretrained(hparams.model_name)

        # Enable label smoothing via the model config
        self.model.config.label_smoothing_factor = hparams.label_smoothing

        self._train_dataset = None
        self._val_dataset   = None

    # ── data ────────────────────────────────────────────────────────────────

    def _make_dataset(self, tsv_path):
        return TranslationDataset(
            tsv_path,
            self.tokenizer,
            max_src_len=self.hparams.max_src_len,
            max_tgt_len=self.hparams.max_tgt_len,
        )

    def train_dataloader(self):
        if self._train_dataset is None:
            self._train_dataset = self._make_dataset(self.hparams.train_tsv)
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self._train_dataset.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self._val_dataset is None:
            self._val_dataset = self._make_dataset(self.hparams.val_tsv)
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._val_dataset.collate_fn,
            pin_memory=True,
        )

    # ── forward / steps ─────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Compute BLEU on a sample (full set evaluated in evaluate.py)
        if batch_idx == 0:
            import sacrebleu
            preds = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=4,
                max_length=self.hparams.max_tgt_len,
            )
            decoded_preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True
            )
            # Recover reference strings from label ids
            label_ids = batch["labels"].clone()
            label_ids[label_ids == -100] = self.tokenizer.pad_token_id
            decoded_refs = self.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )
            bleu = sacrebleu.corpus_bleu(
                decoded_preds, [decoded_refs]
            ).score
            self.log("val/bleu_sample", bleu, prog_bar=True)

        return loss

    # ── optimiser ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr
        )

        # Estimate total steps
        steps_per_epoch = (
            len(self.train_dataloader()) // self.hparams.accumulate_grad_batches
        )
        total_steps = steps_per_epoch * self.hparams.max_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", "frequency": 1}],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MarianMT en→nl")
    # Model
    p.add_argument("--model_name",    default="Helsinki-NLP/opus-mt-en-nl",
                   help="HuggingFace model ID")
    # Data
    p.add_argument("--train_tsv",     default="data/wmt_it/train.tsv")
    p.add_argument("--val_tsv",       default="data/wmt_it/val.tsv")
    p.add_argument("--max_src_len",   type=int, default=128)
    p.add_argument("--max_tgt_len",   type=int, default=128)
    # Training
    p.add_argument("--output_dir",    default="models/marian-en-nl-ft")
    p.add_argument("--max_epochs",    type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--warmup_steps",  type=int,   default=500)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--precision",     default="16-mixed",
                   help="Trainer precision: '32', '16-mixed', 'bf16-mixed'")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--patience",      type=int,   default=3,
                   help="Early stopping patience (epochs)")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = MarianFineTuner(args)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="best-{epoch:02d}-{val/bleu_sample:.2f}",
            monitor="val/bleu_sample",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/bleu_sample",
            patience=args.patience,
            mode="max",
            verbose=True,
        ),
    ]

    logger = CSVLogger(save_dir=args.output_dir, name="logs")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",          # GPU if available, else CPU
        devices="auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5,      # validate twice per epoch
        gradient_clip_val=1.0,
    )

    print(f"\n{'='*60}")
    print(f"  Model  : {args.model_name}")
    print(f"  Epochs : {args.max_epochs}")
    print(f"  Batch  : {args.batch_size}")
    print(f"  LR     : {args.lr}")
    print(f"  Output : {args.output_dir}")
    print(f"{'='*60}\n")

    trainer.fit(model)

    # Save final model + tokenizer in HuggingFace format
    final_path = Path(args.output_dir) / "final"
    model.model.save_pretrained(final_path)
    model.tokenizer.save_pretrained(final_path)
    print(f"\n✓ Final model saved to {final_path}")


if __name__ == "__main__":
    main()
