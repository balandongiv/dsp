## Week 12 – Supervised Data Driven Signal Analysis and Pattern Recognition

This lesson introduces supervised learning techniques for classifying fault types in sensor signals.  You will construct a labelled dataset from synthetic vibration data, extract time‑domain and spectral features, train at least two classifiers and evaluate them using accuracy and false‑alarm rate metrics.

### How to Run

```
python demo.py
```

### Expected Output

Running the script trains logistic regression and random forest classifiers on the synthetic dataset and reports accuracy and false‑alarm rate.  Example output:

```
Logistic regression (C=1.0) accuracy: 1.00, false‑alarm rate: 0.00
Random forest accuracy: 1.00, false‑alarm rate: 0.00
```

Numbers may vary slightly depending on random seed and data realisation.

### Completion Criteria

1. Build a labelled dataset using windowed signals and extract meaningful features (e.g., RMS, kurtosis, bandpower).
2. Train at least two supervised classifiers (e.g., logistic regression and random forest) and perform a small hyperparameter sweep.
3. Report accuracy, confusion matrix and false‑alarm rate at a chosen threshold or class weighting.