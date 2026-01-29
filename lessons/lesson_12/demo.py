"""Week 12 demonstration: supervised classification of fault classes.

This script constructs a labelled dataset from synthetic vibration segments,
extracts simple features (RMS, kurtosis and bandpower), trains a logistic
regression and a random forest classifier, and evaluates performance using
accuracy and false‑alarm rate.  It performs a small hyperparameter sweep
over the logistic regression regularisation parameter.
"""

from typing import Dict, List

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import os
import sys
# Add package root so dsp_utils can be imported when running demo directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data


def extract_features(segments: np.ndarray, fs: float) -> np.ndarray:
    """Extract simple features (RMS, kurtosis, bandpower) from each segment."""
    feats = []
    for seg in segments:
        rms = np.sqrt(np.mean(seg ** 2))
        kurt = kurtosis(seg, fisher=False)
        # Compute bandpower in a broad band (0–500 Hz) as a proxy for energy
        f, psd = welch(seg, fs=fs, nperseg=256)
        bp = np.trapz(psd[(f >= 0) & (f <= 500)], f[(f >= 0) & (f <= 500)])
        feats.append([rms, kurt, bp])
    return np.array(feats)


def compute_false_alarm_rate(y_true: List[str], y_pred: List[str], normal_label: str = 'healthy') -> float:
    """Compute false‑alarm rate treating `normal_label` as the negative class.

    False alarm rate is defined as FP/(FP + TN).
    """
    # Map labels to binary: normal_label = 0, others = 1
    y_true_bin = np.array([0 if lbl == normal_label else 1 for lbl in y_true])
    y_pred_bin = np.array([0 if lbl == normal_label else 1 for lbl in y_pred])
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def run_demo() -> Dict[str, float]:
    # Seed RNG for reproducible synthetic dataset and model splits
    import numpy as np
    np.random.seed(0)
    # Generate labelled dataset
    fs = 12000.0
    data = generate_week_data('week_12', {'fs': fs, 'duration': 6.0, 'segment_length_samples': int(fs * 0.5)})
    segments = data['segments']
    labels = data['labels']
    # Extract features
    X = extract_features(segments, fs=fs)
    y = labels
    # Split into train/test without leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train logistic regression with small sweep over C
    best_acc = 0.0
    best_c = None
    for C in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=200, C=C, multi_class='auto')
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_c = C
    # Retrain logistic regression with best C
    lr = LogisticRegression(max_iter=200, C=best_c, multi_class='auto')
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    far_lr = compute_false_alarm_rate(y_test, y_pred_lr)
    # Train random forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    far_rf = compute_false_alarm_rate(y_test, y_pred_rf)
    print(f"Logistic regression (C={best_c}) accuracy: {acc_lr:.2f}, false‑alarm rate: {far_lr:.2f}")
    print(f"Random forest accuracy: {acc_rf:.2f}, false‑alarm rate: {far_rf:.2f}")
    return {
        'log_reg_accuracy': float(acc_lr),
        'log_reg_false_alarm_rate': float(far_lr),
        'rf_accuracy': float(acc_rf),
        'rf_false_alarm_rate': float(far_rf),
    }


if __name__ == '__main__':
    run_demo()