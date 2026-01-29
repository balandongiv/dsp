"""Completion checks for Week 1.

These tests verify that the demonstration runs and produces the required
metrics.  Running this module should not raise any assertions when the
learner’s code meets the completion criteria.
"""

import os
import sys
# Ensure the package root is importable for direct test execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
from dsp_utils.metrics import snr_db


def test_snr_computation() -> None:
    """Ensure that `run_demo` returns SNR values for each experiment."""
    results = run_demo()
    assert isinstance(results, list) and results, "run_demo should return a non‑empty list"
    for result in results:
        assert 'snr_db' in result, "Each result dict must contain an 'snr_db' field"
        assert isinstance(result['snr_db'], float), "SNR value must be a float"

def test_low_sampling_failure() -> None:
    """Check that using a very low sampling rate yields significantly worse SNR."""
    baseline = run_demo()
    best_snr = max(r['snr_db'] for r in baseline)
    # Use an excessively low sampling rate to illustrate aliasing
    low_fs = 200.0
    data = generate_week_data('week_01', {'fs': low_fs, 'duration': 2.0})
    x_clean = data['x_clean']
    x_noisy = data['x']
    snr_low = snr_db(x_clean, x_noisy - x_clean)
    # Expect at least 20% degradation relative to the best baseline SNR
    assert snr_low < best_snr * 0.8, "Low sampling rate should produce markedly lower SNR"


if __name__ == '__main__':
    test_snr_computation()
    test_low_sampling_failure()
    print('All Week 1 checks passed.')