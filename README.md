# Detecting LLM-Generated Text Using Linguistic Features

This project investigates whether machine-generated text can be distinguished from human-written text using traditional linguistic features and classical machine learning models.

**Dataset**: [human-ai-parallel-corpus-biber](https://huggingface.co/datasets/human-ai-parallel-corpus-biber)  
**Models**: Random Forest, XGBoost, MLP  
**Techniques**: PCA, t-SNE, supervised learning

---

## Project Objective

Large language models (LLMs) can produce human-like text, but subtle linguistic patterns remain. This project tests whether structural linguistic cues—like clause type and syntactic density—can distinguish LLM-generated from human-written text.

---

## Problem & Findings

**Problem**

As LLMs become widespread, distinguishing machine- from human-written text is essential for academic integrity, authorship verification, and trust online. Most existing detectors rely on model-specific signals, which are hard to generalize across architectures.

**Findings**

This project demonstrates that classic ML models trained on linguistic features can reliably classify LLM-generated text:

- XGBoost reached 70.65% test accuracy across 7 LLM sources
- Predictive features include clause complexity, “that” clause frequency, and type-token ratio
- GPT-4 and human texts showed strong separability; LLaMA variants were harder to differentiate
- The feature-based approach is interpretable and architecture-agnostic, offering a robust alternative to black-box detection

---

## File Overview
```
detect-llm-generated-text/
├── llm_text_classifier.ipynb # Full notebook: loading, feature extraction, modeling, visualization
├── requirements.txt # Python dependencies
└── README.md
```

---

## Approach Summary

1. **Dataset**: Loaded from Hugging Face; includes paired human/LLM text samples with genre labels
2. **Preprocessing**: Extracted 66 linguistic features based on Biber's taxonomy
3. **Dimensionality Reduction**: PCA + t-SNE for 2D visualization
4. **Modeling**: Trained Random Forest, XGBoost, and MLP classifiers
5. **Evaluation**: Visual and quantitative results confirm high classification accuracy

---

## Requirements
```
datasets
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install dependencies:
```
pip install -r requirements.txt
```

---

## References
* https://arxiv.org/abs/2410.16107 Park, J., & Choi, Y. (2024). Detecting AI-Generated Text with Human-Like Linguistic Features. arXiv:2410.16107
* https://huggingface.co/datasets/browndw/human-ai-parallel-corpus Human-AI Parallel Corpus – HuggingFace
* https://www.english-corpora.org/coca/ Corpus of Contemporary American English (COCA)
* https://www.english-corpora.org/coca/help/coca2020_overview.pdf COCA Overview PDF


---

## Contact
* Created by Jaeun Park