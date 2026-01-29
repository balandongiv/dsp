"""Week 8 demonstration: Continuous wavelet transform (CWT) for impulse detection.

This script applies the Ricker wavelet to a synthetic signal containing
impulsive events, computes the scalogram magnitude and estimates an
impulse detection rate based on a simple threshold.  Learners can modify
the `WIDTHS` array or switch to a different mother wavelet (e.g. Morlet) to
explore multi‑resolution behaviour.
"""

from typing import Dict

import numpy as np
from scipy.signal import cwt, ricker

import os
import sys
# Allow direct execution by appending package root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data

# Wavelet scales (widths) to use in the CWT
WIDTHS = np.arange(1, 31)


def run_demo() -> Dict[str, float]:
    # Fixed seed for reproducible impulse generation
    import numpy as np
    np.random.seed(0)
    fs = 5000.0
    data = generate_week_data('week_08', {'fs': fs, 'duration': 1.0})
    x = data['x']
    # Compute CWT using the Ricker wavelet
    cwt_matrix = cwt(x, ricker, WIDTHS)
    scalogram = np.abs(cwt_matrix)
    # Aggregate across scales to form an energy envelope over time
    energy = scalogram.sum(axis=0)
    # Detect peaks above a threshold (mean + 3*std)
    threshold = energy.mean() + 3.0 * energy.std()
    detections = np.where(energy > threshold)[0]
    detection_count = len(detections)
    # Assume approximately 5 impulses in the synthetic data
    expected_impulses = 5
    detection_rate = min(detection_count / expected_impulses, 1.0)
    false_alarms = max(detection_count - expected_impulses, 0)
    false_alarm_rate = false_alarms / max(expected_impulses, 1)
    print(f"Detected impulses: {detection_count}")
    print(f"Detection rate (clipped): {detection_rate:.2f}")
    print(f"False alarm rate: {false_alarm_rate:.2f}")
    return {
        'impulse_detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
    }


if __name__ == '__main__':
    run_demo()