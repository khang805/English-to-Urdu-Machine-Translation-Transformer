# English-to-Urdu Machine Translation with mBART-50

This project implements a high-fidelity machine translation system to translate text from English to Urdu. By fine-tuning the mBART-50 large-scale multilingual sequence-to-sequence model, the system leverages
pre-trained multilingual representations to achieve fluent and contextually accurate translations even with limited parallel corpora.


# 🚀 Key Features
## mBART-50 Fine-tuning:
Utilizes a pre-trained transformer (12 layers encoder/12 layers decoder) specifically designed for many-to-many multilingual translation.

## Subword Tokenization: 
Employs SentencePiece tokenization with a 32k vocabulary to effectively handle the morphological complexities of Urdu and minimize out-of-vocabulary (OOV) issues.

## Parallel Corpus Integration: 
Trained on diverse datasets including the Kaggle Parallel Corpus and UMC005 (comprising religious texts, news, and Penn Treebank data).

## Automated Evaluation:
Uses SacreBLEU for standardized translation quality assessment and provides qualitative analysis of grammatical fluency.

## Advanced Training Techniques:
Incorporates Label Smoothing and Early Stopping to prevent overfitting and improve generalization on rare Urdu phrases.


# 📂 Repository Structure

```
├── English_to_urdu_translation.ipynb # Full pipeline (Preprocessing, Training, Evaluation)
├── training_logs.txt                 # Detailed log of training/validation loss per step
├── sample_translations.md            # Examples of English to Urdu outputs
├── Report.pdf                        # Technical report in Springer LNCS format
└── assets/                           # Model architecture diagrams
```


# ⚙️ Installation & Setup

## 1. Prerequisites
Environment: Python 3.10+ (Kaggle or Google Colab with NVIDIA T4 GPU recommended).
Libraries: transformers, datasets, evaluate, sacrebleu, and sentencepiece.

## 2. Install Dependencies

pip install -q \
  transformers \
  datasets \
  evaluate \
  sacrebleu \
  sentencepiece \
  accelerate \
  torch


# 🛠️ Technical Workflow

## 1. Dataset Preparation & Cleaning
The pipeline cleans parallel sentences by removing HTML tags, extra whitespaces, and punctuation. The data is split into an 80/10/10 ratio for training, validation, and testing.

## 2. Model Architecture
The system uses the mBART-50 Many-to-Many architecture:

Encoder: 12 layers with multi-head self-attention to process English context.

Decoder: 12 layers with masked self-attention and cross-attention for Urdu sequence generation.

Special Tokens: Uses en_XX and ur_PK language codes to guide the multilingual embedding space.

## ⚙️ Training Hyperparameters

The mBART-50 model was fine-tuned using the following configuration to ensure stable convergence and prevent overfitting on the English-Urdu parallel corpus:

| Parameter | Value |
| :--- | :--- |
| **Batch Size** | 16 |
| **Optimizer** | AdamW |
| **Learning Rate** | $3 \times 10^{-5}$ |
| **Epochs** | 10 (with Early Stopping) |
| **Max Sequence Length** | 128 |
| **Loss Function** | Cross-entropy with Label Smoothing |


## 📊 Performance & Evaluation

The model is evaluated quantitatively using the **BLEU (Bilingual Evaluation Understudy)** score, which measures the semantic and grammatical overlap between model-generated Urdu and human-reference translations.

| Metric | Result | Description |
| :--- | :---: | :--- |
| **SacreBLEU Score** | Optimized | Measured on the UMC005/Kaggle test set using standardized parameters. |
| **Grammar Preservation** | High | Qualitative assessment confirming the model maintains Urdu's SOV sentence structure. |
| **Inference Speed** | Real-time | Efficient token-by-token decoding using the optimized Seq2SeqTrainer pipeline. |

# Observations
The model preserves Urdu grammar and meaning effectively for standard sentence structures.

Performance increases significantly when fine-tuned on the diverse UMC005 corpus compared to generic synthetic data.


# 🎯 Conclusion
This implementation demonstrates that fine-tuning pre-trained multilingual models like mBART-50 is superior to training Transformers from scratch for low-resource language pairs like English-Urdu. The system 
effectively bridges the linguistic gap, providing a robust baseline for high-quality automated translation.


# 🎓 Author
Muhammad Abdurrahman Khan National University of Computer and Emerging Sciences (FAST), Pakistan
Contact: {i221148}@nu.edu.pk


