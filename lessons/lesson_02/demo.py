"""Week 2 demonstration: sampling and anti‑aliasing effects.

This script generates a multi‑tone vibration signal, samples it at a high
frequency and then downsamples it to a lower rate using `scipy.signal.resample`.
It computes the signal‑to‑noise ratio (SNR) of the high‑rate and low‑rate
signals relative to the clean signal and derives a simple aliasing level
metric as the difference between the two SNR values.  Learners can modify
`FS_HIGH` and `FS_LOW` to explore different sampling scenarios.
"""

from typing import Dict

import numpy as np
from scipy.signal import resample

import os
import sys
# Ensure the package root is available for imports when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data
from dsp_utils.metrics import snr_db

# High and low sampling rates (Hz)
FS_HIGH = 2000
FS_LOW = 500


def run_demo() -> Dict[str, float]:
    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(0)
    """Run the Week 2 demonstration.

    Returns a dictionary with the SNR at the high sampling rate, the SNR at
    the downsampled rate and a simple aliasing level metric.
    """
    # Generate data sampled at the high rate
    data_high = generate_week_data('week_02', {'fs': FS_HIGH, 'duration': 1.0})
    x_clean_high = data_high['x_clean']
    x_high = data_high['x']
    # Downsample the noisy signal
    num_samples = int(len(x_high) * FS_LOW / FS_HIGH)
    x_low = resample(x_high, num_samples)
    # Downsample the clean signal to serve as a reference
    x_clean_low = resample(x_clean_high, num_samples)
    # Compute SNR for each sampling rate
    snr_high = snr_db(x_clean_high, x_high - x_clean_high)
    snr_low = snr_db(x_clean_low, x_low - x_clean_low)
    # Aliasing level defined as the drop in SNR when sampling slower
    aliasing_level = snr_high - snr_low
    print(f"High FS = {FS_HIGH} Hz → SNR = {snr_high:.2f} dB")
    print(f"Low  FS = {FS_LOW} Hz → SNR = {snr_low:.2f} dB")
    print(f"Aliasing level (SNR drop) = {aliasing_level:.2f} dB")
    return {
        'snr_high': snr_high,
        'snr_low': snr_low,
        'aliasing_level': aliasing_level,
    }


if __name__ == '__main__':
    run_demo()