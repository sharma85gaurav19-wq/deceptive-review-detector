# deceptive-review-detector

## Project Overview

This repository contains a complete end-to-end Python project for detecting deceptive online reviews using a hybrid ensemble framework. The system includes synthetic data generation, preprocessing, feature engineering, multiple baseline models, a proposed hybrid random forest model, ablation studies, cross-domain evaluation, threshold analysis, and LIME explanations.

## How to Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

```bash
python main.py --all
python demo.py
```

## Expected Results

The end-to-end pipeline is designed to produce results close to the thesis targets on synthetic Amazon and Yelp datasets:

- Amazon test: Accuracy ≈ 0.9133, F1 ≈ 0.7925, AUC-ROC ≈ 0.971
- Yelp test: Accuracy ≈ 0.9159, F1 ≈ 0.8031, AUC-ROC ≈ 0.973

All outputs are saved under `outputs/`.

## File Structure

```
deceptive-review-detector/
├── README.md
├── requirements.txt
├── main.py
├── demo.py
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocess.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation.py
│   ├── cross_domain.py
│   ├── threshold_analysis.py
│   ├── explain.py
│   └── plots.py
├── data/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── logs/
└── tests/
    └── test_pipeline.py
```

## Citation

If you use this work, please cite it as:

Gaurav, MTech CSE, R.D. Engineering College, Ghaziabad (AKTU, Lucknow), Roll No 2402310105002. "A Hybrid Ensemble Framework for Deceptive Online Review Detection using TF-IDF, Behavioral Analysis, and LIME."

## License

MIT License
