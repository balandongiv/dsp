"""Week 7 demonstration: Short‑Time Fourier Transform (STFT).

This script computes the STFT of a chirp signal and reports approximate
time and frequency resolution based on the chosen window length and number
of FFT points.  Learners can modify `WINDOW_LENGTH` and `NFFT` to explore
the trade‑off between temporal and spectral detail.
"""

from typing import Dict

import numpy as np
from scipy.signal import stft

import os
import sys
# Add repository root to sys.path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data

# STFT parameters
WINDOW_LENGTH = 256
NFFT = 256
OVERLAP = WINDOW_LENGTH // 2


def run_demo() -> Dict[str, float]:
    # Seed RNG to ensure deterministic chirp and noise
    import numpy as np
    np.random.seed(0)
    # Generate chirp signal for STFT analysis
    fs = 4000.0
    data = generate_week_data('week_07', {'fs': fs, 'duration': 2.0})
    x = data['x']
    # Compute STFT
    f, t, Zxx = stft(x, fs=fs, window='hann', nperseg=WINDOW_LENGTH, noverlap=OVERLAP, nfft=NFFT)
    # Estimate time and frequency resolution
    time_resolution = WINDOW_LENGTH / fs
    frequency_resolution = fs / NFFT
    print(f"Time resolution (approx): {time_resolution:.4f} s")
    print(f"Frequency resolution (approx): {frequency_resolution:.2f} Hz")
    # Identify maximum amplitude to ensure STFT computed
    max_amp = np.abs(Zxx).max()
    return {
        'time_resolution': time_resolution,
        'frequency_resolution': frequency_resolution,
        'max_spectrogram_amplitude': float(max_amp),
    }


if __name__ == '__main__':
    run_demo()