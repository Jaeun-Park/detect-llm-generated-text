# Detecting LLM-Generated Text Using Linguistic Features

This project investigates whether machine-generated text can be distinguished from human-written text using traditional linguistic features and classical machine learning models.

**Dataset**: [human-ai-parallel-corpus-biber](https://huggingface.co/datasets/human-ai-parallel-corpus-biber)  
**Models**: Random Forest, XGBoost, MLP  
**Techniques**: PCA, t-SNE, supervised learning

---

## Project Objective

Large language models (LLMs) are capable of producing human-like text, but their outputs still exhibit detectable patterns. This project tests the hypothesis that structural linguistic cues — such as part-of-speech tags, clause types, and syntax density — can help distinguish LLM-generated from human-written text.

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

## Results

- **XGBoost Accuracy**: 70.65%, **Stacked Ensemble (XGB + RF) Accuracy**: 70.29%
- Feature importance highlights structural indicators like participle use, type-token ratio, and “that” clauses
- Clear separation observed for GPT-4 variants and human-written texts
- Significant overlap among similar LLMs (e.g., LLaMA variants), illustrating challenges of intra-family detection


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