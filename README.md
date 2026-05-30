# medspec-nlp — Medical Text Classifier

Classifies clinical transcriptions into medical specialties using NLP and fine-tuned NLP/LLM transformer models.

🔗 **Live Demo:** https://huggingface.co/spaces/rajeshb0425/medspec-nlp

## Problem
Given a medical transcription, predict the medical specialty (Surgery, Cardiology, Neurology, etc.)

## Dataset
MTSamples — 3000+ real medical transcriptions across 8 specialties

## Approach
1. EDA & Data Cleaning
2. Baseline: TF-IDF + Logistic Regression → 38%
3. Fine-tuned DistilBERT → 63.55%
4. Fine-tuned ClinicalBERT → 78.59%

## Results
| Model | Accuracy |
|---|---|
| Baseline (TF-IDF + LR) | 38% |
| DistilBERT | 63.55% |
| ClinicalBERT | 68.59% |

## Setup
pip install -r requirements.txt

## Usage
python src/predict.py --text "Patient presents with chest pain..."
