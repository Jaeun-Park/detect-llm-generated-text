# Detecting LLM-Generated Text Using Linguistic Features

This project investigates whether machine-generated text can be distinguished from human-written text using traditional linguistic features and classical machine learning models.

**Dataset**: [human-ai-parallel-corpus-biber](https://huggingface.co/datasets/human-ai-parallel-corpus-biber)  
**Models**: Random Forest, XGBoost, MLP  
**Techniques**: PCA, t-SNE, supervised learning

---

## Project Objective

Can LLM-generated text be distinguished from human writing using linguistic structure alone? This project evaluates whether features like clause type, syntactic density, and function word usage can effectively classify text source using classical ML models.

---

## Research Basis & Contributions

This project builds on and extends experiments from:

Park, J., & Choi, Y. (2024). Detecting AI-Generated Text with Human-Like Linguistic Features. arXiv:2410.16107

The original work proposed that features such as clause complexity, syntactic density, and function word ratios could effectively differentiate AI- from human-generated text.

This project:
- Reproduces the core findings using open-access data
- Validates across multiple LLM families (GPT, LLaMA, Mistral)
- Tests new classifiers (XGBoost, MLP)
- Explores genre generalization
- Visualizes boundaries via PCA and t-SNE

These results support the robustness of feature-based detection and demonstrate how interpretable models can serve as architecture-agnostic tools for identifying LLM-generated content.

---

## Problem & Findings

**Problem**

As LLMs become widespread, distinguishing machine- from human-written text is essential for academic integrity, authorship verification, and trust online. Most existing detectors rely on model-specific signals, which are hard to generalize across architectures.

**Findings**

- XGBoost achieved 70.65% test accuracy across 7 LLM sources
- Most predictive features: clause complexity, “that” clause frequency, type-token ratio
- GPT-4 vs. human texts were easier to separate; LLaMA variants were more challenging
- Linguistic-feature-based models offer a transparent, generalizable alternative to proprietary detectors

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

1.	**Dataset**: Paired human/LLM text samples (multiple genres)
2.	**Preprocessing**: Extracted 66 linguistic features from each text using Biber’s taxonomy
3.	**Dimensionality Reduction**: Visualized clusters using PCA and t-SNE
4.	**Modeling**: Trained Random Forest, XGBoost, and MLP
5.	**Evaluation**: Quantified performance and visual separation

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

## Key References

- Park, J., & Choi, Y. (2024). [Detecting AI-Generated Text with Human-Like Linguistic Features](https://arxiv.org/abs/2410.16107)
- [Human-AI Parallel Corpus – HuggingFace](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus)
- [Corpus of Contemporary American English (COCA)](https://www.english-corpora.org/coca/)

---

## Contact
* Created by Jaeun Park