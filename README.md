# mt_finetune_challenge 
## Overview

This project implements a domain-specific fine-tuning pipeline for a Transformer-based machine translation model (English → Dutch), focusing on the software/technical domain.

The goal is to improve translation quality for domain-specific content compared to a general-purpose pretrained model.

## Objectives
Fine-tune a pretrained Transformer model for software domain translation
Evaluate performance on:
General dataset (FLORES devtest)
Domain-specific dataset (provided dataset)
Compare results before and after fine-tuning

## Model Used
Model: Helsinki-NLP/opus-mt-en-nl
Architecture: Encoder-Decoder Transformer (MarianMT)

## Project Structure
mt-finetune-challenge1

├─ train.py              # Script for fine-tuning the model

├─ evaluate.py           # Script for evaluation and BLEU score

├─README.md             # Project documentation

├─ results.txt           # Evaluation results

├─ Dataset_Challenge_1.xlsx   # Provided test dataset

## HoW to run 
### Run Training
```
 python scripts/train.py
 ```
### Run Evaluation
```
python evaluate.py
```
## result

| Model Type | Dataset          | BLEU Score |
| ---------- | ---------------- | ---------- |
| Pretrained | General (FLORES) | 28.5       |
| Pretrained | Software Domain  | 24.2       |
| Fine-tuned | General (FLORES) | 27.1       |
| Fine-tuned | Software Domain  | 36.8  |
