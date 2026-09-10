# Cross-Dataset Generalization of Fake Review Detection Models

**A Comparative Study Across Synthetic (AI-Generated), Human-Deceptive, and E-Commerce Reviews**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://huggingface.co/)
[![Tests Passing](https://img.shields.io/badge/pytest-48%20passed-brightgreen.svg)](tests/)
[![Verification](https://img.shields.io/badge/truth%20checks-18%2F18%20passed-brightgreen.svg)](scripts/verify_research_truth.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Tugas Akhir / Undergraduate Thesis Project**  
> Author: Tiffany Christabel Anggriawan  
> Program Studi Informatika, Universitas Ciputra Surabaya

---

## 1. Research Overview & Problem Formulation

Fake online reviews severely distort consumer trust and digital marketplace equilibrium. While state-of-the-art Natural Language Processing (NLP) models achieve high within-domain accuracy on fake review benchmarks, their real-world efficacy depends on **cross-dataset generalization**.

### Core Research Questions
1. **Cross-Domain Generalization**: Do detection models trained on **AI-generated synthetic reviews (Amazon GPT-2)** generalize to **human-written deceptive reviews (DOSC MTurk)**, and vice versa?
2. **Representation Resilience**: Can contextual transformer representations (**BERT-base**) mitigate the cross-domain performance collapse experienced by classical surface $n$-gram baselines?
3. **Forensic Failure Modes**: What linguistic divergence, vocabulary isolation, and syntactic artifacts cause cross-dataset transfer breakdown?

---

## 2. Experimental Benchmark Datasets

| Dataset | Domain | Nature of Fake Reviews | Size | Ground Truth Balance | Key Leakage Control Applied |
|---|---|---|---|---|---|
| **Amazon Reviews** (Salminen et al.) | E-Commerce | **Synthetic AI** (GPT-2 generated) | 40,432 rows | 50% CG / 50% OR | **Prefix-Group De-contamination**: GroupShuffleSplit on $\ge 80$-char prefixes to eliminate prompt template leakage |
| **DOSC** (Ott et al.) | Hotel Hospitality | **Human Deceptive** (Crowdsourced MTurk) | 1,596 rows | 50% Deceptive / 50% Truthful | **Strict Leakage Removal**: Immediate drop of `source` column (100% target leakage: MTurk=deceptive, Web=truthful) + exact deduplication |

---

## 3. Key Research Findings

### 3.1 Symmetric Cross-Dataset Evaluation Transfer Matrix

| Model ID | Family | Algorithm | Training Domain | Evaluation Domain | Within F1 | Cross F1 | $\Delta$ F1 (Degradation) | Within ROC-AUC | Cross ROC-AUC | Generalization Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **`A_logreg`** | Classical | Logistic Regression | AMAZON | DOSC | **0.9578** | **0.0000** | **+0.9578** | 0.9920 | 0.5629 | **Catastrophic Failure** |
| **`A_svm`** | Classical | Linear SVC | AMAZON | DOSC | **0.9573** | **0.0000** | **+0.9573** | 0.9921 | 0.5827 | **Catastrophic Failure** |
| **`A_bert`** | Transformer | BERT-base-uncased | AMAZON | DOSC | **0.9305** | **0.1236** | **+0.8069** | 0.9867 | 0.5668 | **Catastrophic Failure** |
| **`B_logreg`** | Classical | Logistic Regression | DOSC | AMAZON | **0.9177** | **0.1709** | **+0.7468** | 0.9675 | 0.4452 | **Catastrophic Failure** |
| **`B_svm`** | Classical | Linear SVC | DOSC | AMAZON | **0.9108** | **0.1619** | **+0.7489** | 0.9680 | 0.4343 | **Catastrophic Failure** |
| **`B_bert`** | Transformer | BERT-base-uncased | DOSC | AMAZON | **0.8045** | **0.6549** | **+0.1496** | 0.8849 | 0.6131 | **Moderate Degradation** |

### 3.2 Linguistic Forensics & Breakthrough Insights
- **Asymmetric Transferability**: Fine-tuning BERT on human deception (`B_bert`) transfers moderately well to synthetic reviews ($F1 = 0.6549$, retaining $>81\%$ capability), while fine-tuning on synthetic reviews (`A_bert`) collapses on human deception ($F1 = 0.1236$) due to shortcut learning on generator artifacts.
- **Disjoint Deception Feature Spaces (2.0% Overlap)**: Across the top-50 deception cues identified by `A_logreg` and `B_logreg`, **only 1 phrase (`'i will'`) was shared**. Spearman rank correlation across 24,773 shared features is essentially zero ($\rho = 0.0167, R^2 = 0.00028$).
- **Pronoun Polarity Inversion**: The first-person pronoun `"i"` functions in polar opposite directions:
  - In Amazon GPT-2 reviews, `"i"` is a strong signal of a **genuine** review (GPT-2 underproduces personal grounding: weight $-4.973$).
  - In DOSC human deception, `"i"` and `"my"` are strong signals of a **deceptive** review (human impostors overcompensate to establish artificial credibility: weight $+1.794$).
- **Failure Mode Taxonomy**:
  - **82.5%** of $A \rightarrow B$ misclassifications are caused by **Template Absence (TA)**: models trained on synthetic reviews learn prompt-repetition artifacts that human impostors do not exhibit.
  - **62.5%** of $B \rightarrow A$ misclassifications are caused by **Length/Complexity Bias (LC)**: DOSC reviews are narrative-heavy (median ~135 words), whereas Amazon reviews are compact (median ~40 words).

### 3.3 Preemptive Defense Ablation Studies
1. **Named-Entity & Geographic Masking**: Masking 3,200 occurrences of `chicago` and hotel names (`[LOCATION]`) shifted cross-domain F1 from $0.1709 \to 0.1200$, proving that geographic confounders are not the primary failure driver; pronoun inversion remains the dominant obstacle.
2. **Youden's J Threshold Calibration**: Calibrating optimal decision thresholds $\tau^*$ on validation splits failed to rescue any cross-domain model (`A_logreg` remained $F1 = 0.0000$), proving that failure is **representational (orthogonal spaces)**, not decisional.

---

## 4. Publication-Ready Visualizations (300 DPI)

All figures are automatically generated in `results/figures/`:

1. `results/figures/feature_divergence.png`: Top-15 positive & negative feature weights comparing Amazon vs. DOSC.
2. `results/figures/vocab_overlap_venn.png`: Lexical isolation and vocabulary overlap Venn diagram (79% OOV).
3. `results/figures/length_distributions.png`: KDE word length distributions across genuine and deceptive classes.
4. `results/figures/confusion_matrices.png`: 8-panel heatmaps across within-domain and cross-domain transfer.
5. `results/figures/roc_curves.png`: Within-domain vs. cross-domain ROC curves including the sub-chance curve.
6. `results/figures/delta_f1_comparison.png`: Performance drop ($\Delta F1$) bar chart comparing all 6 models.
7. `results/figures/error_taxonomy.png`: Prevalence of failure modes (TA, LC, VM, SS, SD).
8. `results/figures/psycholinguistic_scatter.png`: Pronoun usage vs. spatial detail density scatter plot.

---

## 5. Repository Architecture

```
fake_review_detection/
├── config/
│   └── config.yaml                 # Master configuration (seed=42, all hyperparams)
├── data/
│   ├── raw/                        # Untouched raw CSVs
│   ├── processed/                  # Clean parquet files (amazon_clean, dosc_clean)
│   └── splits/                     # Reproducible split JSONs with SHA-256 checksums
├── src/
│   ├── data/                       # Ingest, prefix grouping, dedup, PyTorch dataset
│   ├── features/                   # Dual TF-IDF, psycholinguistic, text stats
│   ├── models/                     # Classical baselines, transformer fine-tuning, registry
│   ├── evaluation/                 # Metrics, cross-eval matrix, error analysis taxonomy
│   └── visualization/              # 8 publication-ready 300 DPI plotting functions
├── scripts/
│   ├── run_preprocessing.py        # End-to-end data preparation & verification gates
│   ├── run_training.py             # Classical baseline training & cross-validation
│   ├── run_transformers.py         # Transformer fine-tuning runner (MPS/CUDA)
│   ├── run_cross_eval.py           # Cross-dataset transfer evaluation & matrix generator
│   ├── run_forensics.py            # Forensic failure analysis & figure generator
│   ├── verify_research_truth.py    # 18/18 empirical truth and data hygiene checks
│   └── run_ablation_studies.py     # Geographic entity masking & Youden's J calibration
├── tests/                          # 48 automated unit tests (100% passing)
├── results/
│   ├── figures/                    # 8 publication-ready 300 DPI figures
│   ├── metrics/                    # Serialized CSV and JSON metrics
│   ├── models/                     # Serialized model bundles and transformer checkpoints
│   ├── MASTER_RESEARCH_MONOGRAPH.md# Consolidates all context, findings, proofs, and Bab 3-5 drafts
│   ├── academic_polish_ablation_report.md
│   ├── research_truth_verification_report.md
│   └── audit_verification_summary.md
├── requirements.txt                # Pinned dependency requirements
└── README.md
```

---

## 6. Quickstart & Execution

### 6.1 Environment Setup
```bash
git clone https://github.com/catnipconnoisseur/fake-review-generalization.git
cd fake-review-generalization

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Run Preprocessing & Splitting (Phase 1)
```bash
python scripts/run_preprocessing.py
```

### 6.3 Train Classical Baselines (Phase 2)
```bash
python scripts/run_training.py
```

### 6.4 Fine-Tune Transformers (Phase 3)
```bash
# Fine-tune BERT on DOSC (uses Apple Silicon MPS acceleration)
python scripts/run_transformers.py --model B_bert --epochs 2
```

### 6.5 Execute Cross-Dataset Transfer Matrix (Phase 4)
```bash
python scripts/run_cross_eval.py
```

### 6.6 Run Forensic Failure Mode Analysis & Generate Figures (Phase 5)
```bash
python scripts/run_forensics.py
```

### 6.7 Run Automated Unit Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

### 6.8 Run Research Truth & Sanity Verification (18 Checks)
```bash
python scripts/verify_research_truth.py
```

### 6.9 Run Preemptive Defense Ablation Studies
```bash
python scripts/run_ablation_studies.py
```

---

## 7. License & Citation

This project is licensed under the MIT License.

```bibtex
@thesis{anggriawan2026crossdataset,
  title={Cross-Dataset Generalization of Fake Review Detection Models: A Comparative Study Across Synthetic (AI-Generated), Human-Deceptive, and E-Commerce Reviews},
  author={Anggriawan, Tiffany Christabel},
  school={Universitas Ciputra Surabaya},
  year={2026}
}
```
