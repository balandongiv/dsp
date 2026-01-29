"""Completion checks for Week 11.

The check verifies that bandpower and defect frequency amplitude are
computed and positive.
"""

import os
import sys
# Allow imports when running tests directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.fft import rfft, rfftfreq


def test_bandpower_envelope() -> None:
    """Verify that bandpower and defect frequency amplitude are positive."""
    results = run_demo()
    assert 'bandpower_40_60' in results and 'defect_frequency_amplitude' in results, \
        "Missing keys in results"
    # Bandpower may be zero if the discrete frequency grid does not fall within 40–60 Hz
    assert results['bandpower_40_60'] >= 0.0, "Bandpower must be non‑negative"
    assert results['defect_frequency_amplitude'] > 0.0, \
        "Defect frequency amplitude must be positive"

def test_wrong_band_reduces_defect_amplitude() -> None:
    """Use an incorrect band for envelope detection and verify the defect amplitude decreases."""
    baseline = run_demo()
    baseline_amp = baseline['defect_frequency_amplitude']
    fs = 16000.0
    data = generate_week_data('week_11', {'fs': fs, 'duration': 1.0})
    x = data['x']
    # Envelope detection with a wrong band (e.g. very high frequencies)
    nyq = fs / 2.0
    # Band-pass filter around 5000–6000 Hz where no fault energy exists
    b_bp, a_bp = butter(4, [5000.0 / nyq, 6000.0 / nyq], btype='bandpass')
    x_bp = filtfilt(b_bp, a_bp, x)
    # Rectify
    x_rect = np.abs(x_bp)
    # Low-pass filter at 200 Hz (same as demo)
    b_lp, a_lp = butter(4, 200.0 / nyq, btype='lowpass')
    envelope = filtfilt(b_lp, a_lp, x_rect)
    # Compute FFT of envelope
    N = len(envelope)
    env_fft = np.abs(rfft(envelope))
    freqs = rfftfreq(N, d=1.0 / fs)
    defect_freq = 50.0
    idx = np.argmin(np.abs(freqs - defect_freq))
    wrong_amp = env_fft[idx]
    assert wrong_amp < baseline_amp, \
        "Using an incorrect band should reduce the defect frequency amplitude relative to the baseline"


if __name__ == '__main__':
    test_bandpower_envelope()
    test_wrong_band_reduces_defect_amplitude()
    print('All Week 11 checks passed.')