"""Completion checks for Week 8.

The check ensures that the CWT‑based impulse detection returns a detection rate
between 0 and 1 and a non‑negative false alarm rate.
"""

import os
import sys
# Enable imports when running tests directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
import numpy as np
from scipy.signal import cwt, ricker


def test_cwt_detection() -> None:
    """Ensure that the CWT-based impulse detection returns valid rates."""
    results = run_demo()
    assert 0.0 <= results['impulse_detection_rate'] <= 1.0, \
        "Detection rate must be between 0 and 1"
    assert results['false_alarm_rate'] >= 0.0, \
        "False alarm rate must be non‑negative"

def test_narrow_width_detection() -> None:
    """Use a narrower range of widths to illustrate reduced detection sensitivity."""
    baseline = run_demo()
    baseline_rate = baseline['impulse_detection_rate']
    fs = 5000.0
    data = generate_week_data('week_08', {'fs': fs, 'duration': 1.0})
    x = data['x']
    # Use fewer scales for the wavelet transform
    widths_small = np.arange(1, 6)
    cwt_matrix = cwt(x, ricker, widths_small)
    scalogram = np.abs(cwt_matrix)
    energy = scalogram.sum(axis=0)
    threshold = energy.mean() + 3.0 * energy.std()
    detections = np.where(energy > threshold)[0]
    detection_rate_small = min(len(detections) / 5, 1.0)
    assert detection_rate_small <= baseline_rate, \
        "Using fewer widths should not produce a higher detection rate than the baseline"


if __name__ == '__main__':
    test_cwt_detection()
    test_narrow_width_detection()
    print('All Week 8 checks passed.')