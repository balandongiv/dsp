"""Week 13 demonstration: unsupervised anomaly detection.

This script trains unsupervised anomaly detectors on healthy data only and
computes anomaly scores on a mixed dataset.  Two models are used: PCA
reconstruction error (distance in principal component space) and Isolation
Forest.  A threshold is selected to achieve a target false‑alarm rate on a
validation set.  A simple regime feature (segment index modulo two) is
added to illustrate regime‑aware mitigation.
"""

from typing import Dict

import numpy as np
from scipy.stats import kurtosis
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import os
import sys
# Insert package root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data


def extract_features(segments: np.ndarray, fs: float) -> np.ndarray:
    feats = []
    for i, seg in enumerate(segments):
        rms = np.sqrt(np.mean(seg ** 2))
        kurt = kurtosis(seg, fisher=False)
        f, psd = welch(seg, fs=fs, nperseg=256)
        bp = np.trapz(psd[(f >= 0) & (f <= 500)], f[(f >= 0) & (f <= 500)])
        # Regime feature: simple proxy for operating regime (even/odd segment index)
        regime = i % 2
        feats.append([rms, kurt, bp, regime])
    return np.array(feats)


def run_demo() -> Dict[str, float]:
    # Seed RNG for reproducibility of synthetic anomaly segments and models
    import numpy as np
    np.random.seed(0)
    fs = 5000.0
    data = generate_week_data('week_13', {'fs': fs, 'duration': 12.0, 'segment_length_samples': int(fs * 0.5)})
    segments = data['segments']
    labels = data['labels']
    X = extract_features(segments, fs=fs)
    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Partition into healthy and mixed sets
    healthy_idx = np.where(labels == 'healthy')[0]
    mixed_idx = np.where(labels != 'healthy')[0]
    X_train = X_scaled[healthy_idx]
    X_test = X_scaled
    y_test = np.array([0 if lbl == 'healthy' else 1 for lbl in labels])
    # PCA model
    pca = PCA(n_components=2)
    pca.fit(X_train)
    # Reconstruction error as anomaly score
    X_proj = pca.inverse_transform(pca.transform(X_test))
    recon_error = np.sum((X_test - X_proj) ** 2, axis=1)
    # Determine threshold at 95th percentile of healthy errors
    healthy_errors = recon_error[healthy_idx]
    threshold_pca = np.percentile(healthy_errors, 95.0)
    y_pred_pca = (recon_error > threshold_pca).astype(int)
    # Compute false alarm rate (among healthy samples in test set)
    false_alarm_pca = np.sum((y_pred_pca == 1) & (y_test == 0)) / max(np.sum(y_test == 0), 1)
    # Isolation Forest model
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train)
    scores = -iso.score_samples(X_test)
    threshold_if = np.percentile(scores[healthy_idx], 95.0)
    y_pred_if = (scores > threshold_if).astype(int)
    false_alarm_if = np.sum((y_pred_if == 1) & (y_test == 0)) / max(np.sum(y_test == 0), 1)
    print(f"PCA false‑alarm rate: {false_alarm_pca:.2f}")
    print(f"Isolation Forest false‑alarm rate: {false_alarm_if:.2f}")
    return {
        'pca_false_alarm_rate': float(false_alarm_pca),
        'if_false_alarm_rate': float(false_alarm_if),
    }


if __name__ == '__main__':
    run_demo()