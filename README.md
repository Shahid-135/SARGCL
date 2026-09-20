# State-Aware Hypergraph Representation Learning with Difficulty-Regulated Contrastive Alignment for Multimodal Understanding

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📌 Abstract

Real-world multimodal data are characterized by weak, noisy, and heterogeneous image–text correspondence, posing significant challenges for high-stakes classification tasks such as disaster response, safety monitoring, and situational awareness. Existing multimodal approaches rely on uniformly enforced alignment objectives that fail to capture entity-level physical state variations, higher-order relational structures, and cross-modal alignment uncertainty.

We propose **SARGCL** (**S**tate-**A**ware **H**yper**G**raph **C**ontrastive **L**earning), an end-to-end framework with difficulty-regulated alignment:
1. **State-Aware Visual Nodes:** Visual content is decomposed into region-level entity proposals via Faster R-CNN and textually grounded using conditioned BLIP captioning to capture fine-grained physical and situational states (e.g., *"damaged bridge"* vs. *"intact bridge"*).
2. **Textual Syntactic Hypergraphs:** Textual content is mapped into hypergraphs using dependency parsing to capture high-order predicate–argument structures beyond pairwise edges.
3. **Cross-Modal Hypergraph Refinement:** Textual global context prunes semantically irrelevant visual hyperedges, mitigating background visual noise.
4. **Modality-Specific HGATs:** Dual Hypergraph Attention Networks encode higher-order topological relations within each modality.
5. **Difficulty-Regulated Contrastive Alignment:** Entropy-regularized Optimal Transport (Sinkhorn–Knopp) measures instance-level cross-modal alignment difficulty, dynamically modulating the temperature schedule $\tau(C)$ in hierarchical contrastive learning (LHC-CL).
6. **Dirichlet Uncertainty Fusion:** Multimodal representations are adaptively aggregated using evidential deep learning to estimate epistemic uncertainty.

![SARGCL Framework Architecture](figures/arch.png)

---

## 🏛️ Method Overview

### 1. State-Aware Entity Grounding
Rather than compressing images into global vectors or detecting ungrounded bounding boxes, SARGCL detects key entities and generates state descriptions via BLIP captioning. This projects visual entities directly into the semantic language space of CLIP:

![State-Aware Entity Illustration](figures/state.png)

### 2. Syntactic & Cross-Modal Hypergraphs
- **Text:** Dependency trees from spaCy are decomposed into hyperedges grouping head tokens with their syntactic dependents.
- **Vision:** Visual entities form hyperedges based on feature affinity, refined by cross-modal gating against global text context.

### 3. Difficulty-Regulated Optimal Transport Alignment
Standard contrastive learning applies uniform attraction across all pairs, leading to gradient explosion on noisy or partially paired samples. SARGCL computes the Optimal Transport cost $\mathcal{W}(z^V, z^T)$ as a **detached difficulty signal**, modulating temperature $\tau$:

$$\tau(C) = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \cdot \text{MLP}(C)$$

This formally guarantees softer gradient updates for ambiguous samples and sharper gradients for well-aligned pairs.

---

## 📁 Repository Structure

```text
SARGCL/
├── configs/
│   └── default.yaml                   # Default training hyperparameters
├── data/
│   └── README.md                      # Dataset acquisition and preparation guide
├── figures/                           # Paper diagrams and analysis plots
│   ├── arch.png                       # Full framework architecture
│   ├── state.png                      # State-aware entity grounding
│   ├── hps.png                        # Hyperparameter sensitivity
│   ├── hyperparameter.png
│   ├── bubble_chart.png
│   └── hypergraph_nodes1.png
├── qualitative/                       # Qualitative case studies and visualizations
│   ├── comparison_analysis/
│   └── sample_32_disagree/
├── results/                           # Pre-computed evaluation logs and reports
│   ├── test_classification_report.csv
│   ├── test_classification_report.txt
│   ├── test_confusion_matrix.csv
│   └── training_history.csv
├── sargcl/                            # Core modular Python package
│   ├── __init__.py
│   ├── dataset.py                     # DisasterDataset, CachedSARGCLDataset, collate_fns
│   ├── encoders.py                    # CLIP global feature extractors
│   ├── evaluate.py                    # Evaluation metrics, OT checks, uncertainty analysis
│   ├── hypergraph.py                  # VisualHypergraphRefiner, HGATLayer, HGATEncoder
│   ├── model.py                       # SARGCL and CachedSARGCL model implementations
│   ├── ot.py                          # Sinkhorn-Knopp Optimal Transport alignment
│   ├── text_extractor.py              # spaCy dependency-based textual hypergraph builder
│   ├── utils.py                       # Seed setting, similarity, and logging utilities
│   └── visual_extractor.py            # Faster R-CNN + BLIP state-aware node extractor
├── scripts/
│   ├── train.py                       # End-to-end training script
│   ├── train_cached.py                # Accelerated two-stage feature cached training
│   └── evaluate.py                    # Standalone checkpoint evaluation
├── .gitignore
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Shahid-135/SARGCL.git
cd SARGCL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download spaCy English language model
python -m spacy download en_core_web_sm
```

### 2. Data Preparation

See [`data/README.md`](data/README.md) for detailed dataset instructions.

Format your data as tab-separated (`.tsv`) files with the following columns:
- `image`: Relative path to image file
- `tweet_text`: Raw tweet text
- `label`: Binary classification target (`0` or `1`)

Organize the directory as follows:
```text
data/
├── task_informative_text_img_agreed_lab_train.tsv
├── task_informative_text_img_agreed_lab_dev.tsv
├── task_informative_text_img_agreed_lab_test.tsv
└── data_image/
    ├── 906018789631787012_0.jpg
    └── ...
```

---

## 🏋️ Training

### Option A: Accelerated Two-Stage Training (Recommended)

Running Faster R-CNN and BLIP on-the-fly for every epoch can be computationally demanding. Use `scripts/train_cached.py` to extract and freeze the multimodal representations once, then train the active SARGCL modules (HGATs, refiners, alignment MLPs, classifier) for 50 epochs in minutes:

```bash
python scripts/train_cached.py \
    --train_csv data/task_informative_text_img_agreed_lab_train.tsv \
    --valid_csv data/task_informative_text_img_agreed_lab_dev.tsv \
    --test_csv  data/task_informative_text_img_agreed_lab_test.tsv \
    --image_dir data/data_image \
    --cache_dir ./sargcl_feature_cache \
    --output_dir outputs/sargcl_cached_run \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-5
```

### Option B: Standard End-to-End Training

Train the complete architecture end-to-end:

```bash
python scripts/train.py \
    --train_csv data/task_informative_text_img_agreed_lab_train.tsv \
    --valid_csv data/task_informative_text_img_agreed_lab_dev.tsv \
    --test_csv  data/task_informative_text_img_agreed_lab_test.tsv \
    --image_dir data/data_image \
    --output_dir outputs/sargcl_full_run \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-5 \
    --lambda_hc_cl 0.1 \
    --lambda_ot 0.1
```

---

## 🔍 Evaluation

To evaluate a trained checkpoint and generate test metrics, confusion matrix, and uncertainty analysis:

```bash
python scripts/evaluate.py \
    --test_csv data/task_informative_text_img_agreed_lab_test.tsv \
    --image_dir data/data_image \
    --checkpoint outputs/sargcl_cached_run/best_sargcl_cached_model.pth \
    --output_dir outputs/eval_results \
    --batch_size 8 \
    --run_uncertainty_analysis \
    --measure_inference_time
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
