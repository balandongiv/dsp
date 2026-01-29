"""Completion checks for Week 12.

The check verifies that the demonstration trains at least two models and
computes accuracy and false‑alarm rate for each.  It ensures that the
false‑alarm rate is a finite value between 0 and 1.
"""

import os
import sys
# Add package root for direct test execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo, extract_features
from dsp_utils import generate_week_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def test_supervised_models() -> None:
    """Ensure both logistic regression and random forest metrics are present and within [0, 1]."""
    results = run_demo()
    for key in ['log_reg_accuracy', 'log_reg_false_alarm_rate', 'rf_accuracy', 'rf_false_alarm_rate']:
        assert key in results, f"Missing {key} in results"
        value = results[key]
        assert 0.0 <= value <= 1.0, f"{key} must be between 0 and 1"

def test_random_label_performance() -> None:
    """Shuffle labels randomly and verify that model accuracy drops relative to the baseline."""
    baseline = run_demo()
    baseline_acc = baseline['log_reg_accuracy']
    # Generate a fresh dataset
    fs = 12000.0
    data = generate_week_data('week_12', {
        'fs': fs,
        'duration': 6.0,
        'segment_length_samples': int(fs * 0.5),
    })
    segments = data['segments']
    labels = data['labels']
    features = extract_features(segments, fs)
    rng = np.random.default_rng(0)
    shuffled_labels = rng.permutation(labels)
    # Train logistic regression on the shuffled labels
    X_train, X_test, y_train, y_test = train_test_split(
        features, shuffled_labels, test_size=0.4, random_state=42, stratify=shuffled_labels
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=200, multi_class='auto')
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    assert acc < baseline_acc, \
        "Model trained on random labels should have lower accuracy than the baseline logistic regression"


if __name__ == '__main__':
    test_supervised_models()
    test_random_label_performance()
    print('All Week 12 checks passed.')