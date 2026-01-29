"""Week 14 demonstration: capstone time‑frequency neural pipeline.

This script builds a capstone fault‑classification pipeline using a
time‑frequency representation (log‑scaled STFT spectrogram) as input to a
multi‑layer perceptron (MLP) classifier.  It performs a small sweep over
hidden layer sizes, reports accuracy and false‑alarm rate and measures
approximate per‑segment inference latency.
"""

from typing import Dict, List

import time
import numpy as np
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import os
import sys
# Ensure dsp_utils module is importable during direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data


def extract_spectrogram_features(segments: np.ndarray, fs: float, nperseg: int = 256,
                                 noverlap: int = 128) -> np.ndarray:
    """Compute log‑magnitude spectrogram features for each segment and flatten."""
    features = []
    for seg in segments:
        f, t_vec, Zxx = stft(seg, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
        spec_mag = np.abs(Zxx)
        # Log‑scale to stabilise magnitude range
        log_spec = np.log1p(spec_mag)
        # Flatten spectrogram (time × frequency)
        features.append(log_spec.flatten())
    return np.array(features)


def compute_false_alarm_rate(y_true: List[str], y_pred: List[str], normal_label: str = 'healthy') -> float:
    y_true_bin = np.array([0 if lbl == normal_label else 1 for lbl in y_true])
    y_pred_bin = np.array([0 if lbl == normal_label else 1 for lbl in y_pred])
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def run_demo() -> Dict[str, float]:
    # Seed RNG for reproducible synthetic segments and model training
    import numpy as np
    np.random.seed(0)
    # Generate dataset
    fs = 16000.0
    data = generate_week_data('week_14', {'fs': fs, 'duration': 8.0, 'segment_length_samples': int(fs * 0.5)})
    segments = data['segments']
    labels = data['labels']
    # Extract log‑spectrogram features
    X = extract_spectrogram_features(segments, fs=fs, nperseg=256, noverlap=128)
    y = labels
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    # Standardise
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Parameter sweep over hidden layer sizes
    candidates = [(50,), (100,), (50, 50)]
    best_acc = 0.0
    best_config = None
    for hidden in candidates:
        clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, max_iter=200, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_config = hidden
    # Train final model
    mlp = MLPClassifier(hidden_layer_sizes=best_config, alpha=1e-4, max_iter=200, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    far = compute_false_alarm_rate(y_test, y_pred)
    # Estimate latency: time to process one segment end‑to‑end (feature extraction + inference)
    seg = segments[0]
    start = time.time()
    feat = extract_spectrogram_features(np.array([seg]), fs=fs, nperseg=256, noverlap=128)
    feat_scaled = scaler.transform(feat)
    _ = mlp.predict(feat_scaled)
    end = time.time()
    latency_ms = (end - start) * 1000.0
    print(f"MLP hidden layers {best_config}, accuracy: {acc:.2f}, false‑alarm rate: {far:.2f}, latency: {latency_ms:.1f} ms")
    return {
        'mlp_accuracy': float(acc),
        'mlp_false_alarm_rate': float(far),
        'latency_ms': float(latency_ms),
    }


if __name__ == '__main__':
    run_demo()