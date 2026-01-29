"""Completion checks for Week 13.

The check verifies that the unsupervised anomaly detection computes a
false‑alarm rate for each model and that the rates are between 0 and 1.
"""

import os
import sys
# Ensure dsp_utils can be imported when this test runs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo, extract_features
from dsp_utils import generate_week_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def test_anomaly_detectors() -> None:
    """Check that false-alarm rates are computed and within [0, 1]."""
    results = run_demo()
    for key in ['pca_false_alarm_rate', 'if_false_alarm_rate']:
        assert key in results, f"Missing {key}"
        val = results[key]
        assert 0.0 <= val <= 1.0, f"{key} must be between 0 and 1"

def test_training_on_mixed_data_degrades_detection() -> None:
    """
    Train the Isolation Forest on a dataset that contains anomalies.  This is a
    common failure mode in unsupervised anomaly detection: including anomalous
    segments in the training set causes the model to treat them as nominal,
    reducing its ability to detect anomalies.  The test compares the F1‑score
    (combined precision and recall) of a model trained on healthy data versus
    one trained on all data.  The F1‑score should be lower when anomalies are
    included in the training set.
    """
    # First run the demo to obtain baseline F1 using the healthy training set.
    # Extract features and labels from the synthetic dataset.
    fs = 5000.0
    data = generate_week_data('week_13', {
        'fs': fs,
        'duration': 12.0,
        'segment_length_samples': int(fs * 0.5),
    })
    segments = data['segments']
    labels = data['labels']
    X = extract_features(segments, fs)
    # Precompute true labels: 0 for healthy, 1 for anomaly
    y_true = np.array([0 if lbl == 'healthy' else 1 for lbl in labels])
    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Identify indices of healthy segments
    healthy_idx = np.where(labels == 'healthy')[0]

    def compute_f1(train_mask: np.ndarray) -> float:
        """
        Train IsolationForest on the subset of samples indicated by train_mask and
        compute the F1‑score on the full dataset.  The threshold is computed
        from scores on the training subset, consistent with run_demo().
        """
        # Fit model on selected training samples
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(X_scaled[train_mask])
        scores = -iso.score_samples(X_scaled)
        # Determine threshold using the 95th percentile of the training scores
        threshold = np.percentile(scores[train_mask], 95.0)
        y_pred = (scores > threshold).astype(int)
        # Compute precision, recall, and F1 for anomaly class (label == 1)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return f1

    # F1‑score when training on healthy data (baseline)
    baseline_f1 = compute_f1(healthy_idx)
    # F1‑score when training on mixed data (including anomalies)
    mixed_f1 = compute_f1(np.arange(len(labels)))
    # Assert that training on mixed data reduces the F1‑score
    assert mixed_f1 < baseline_f1, (
        "Training on a dataset containing anomalies should degrade detection performance "
        "(lower F1‑score) compared to training on healthy data only"
    )



if __name__ == '__main__':
    test_anomaly_detectors()
    test_training_on_mixed_data_degrades_detection()
    print('All Week 13 checks passed.')