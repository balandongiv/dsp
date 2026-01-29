"""Completion checks for Week 14.

The check ensures that the MLP pipeline reports accuracy, false‑alarm rate and
latency.  Latency should be a finite positive number and false‑alarm rate
should be between 0 and 1.
"""

import os
import sys
# Append package root for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo, extract_spectrogram_features, compute_false_alarm_rate
from dsp_utils import generate_week_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def test_capstone_pipeline() -> None:
    """Check that the MLP metrics are present and within valid ranges."""
    results = run_demo()
    assert 'mlp_accuracy' in results and 'mlp_false_alarm_rate' in results and 'latency_ms' in results, \
        "Missing metrics from capstone pipeline"
    assert 0.0 <= results['mlp_accuracy'] <= 1.0, \
        "Accuracy must be between 0 and 1"
    assert 0.0 <= results['mlp_false_alarm_rate'] <= 1.0, \
        "False-alarm rate must be between 0 and 1"
    assert results['latency_ms'] > 0.0, \
        "Latency must be positive"

def test_coarser_spectrogram_degradation() -> None:
    """Use coarser spectrogram parameters and verify that accuracy does not exceed the baseline."""
    baseline = run_demo()
    baseline_acc = baseline['mlp_accuracy']
    fs = 16000.0
    data = generate_week_data('week_14', {
        'fs': fs,
        'duration': 8.0,
        'segment_length_samples': int(fs * 0.5),
    })
    segments = data['segments']
    labels = data['labels']
    # Compute spectrogram features with larger windows (coarser time-frequency resolution)
    features = extract_spectrogram_features(segments, fs=fs, nperseg=512, noverlap=256)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4, random_state=42, stratify=labels
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train a simple MLP with a single hidden layer (as in baseline)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-4, max_iter=200, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    assert acc <= baseline_acc + 1e-6, \
        "Coarser spectrogram features should not outperform the baseline accuracy"


if __name__ == '__main__':
    test_capstone_pipeline()
    test_coarser_spectrogram_degradation()
    print('All Week 14 checks passed.')