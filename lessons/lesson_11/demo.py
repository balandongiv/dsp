"""Week 11 demonstration: bandpower and envelope detection.

This script computes bandpower for a synthetic bearing signal and performs
envelope detection by band‑pass filtering, rectifying and low‑pass
filtering.  It then computes the FFT of the envelope and reports the
amplitude of the expected defect frequency.
"""

from typing import Dict

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.fft import rfft, rfftfreq

import os
import sys
# Ensure direct execution can import dsp_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data


def bandpower(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    f, psd = welch(signal, fs=fs, nperseg=1024)
    idx = np.logical_and(f >= fmin, f <= fmax)
    return np.trapz(psd[idx], f[idx])


def envelope_detection(x: np.ndarray, fs: float, band: tuple, lp_cutoff: float) -> np.ndarray:
    # Band‑pass filter
    nyq = fs / 2.0
    b_bp, a_bp = butter(4, [band[0] / nyq, band[1] / nyq], btype='bandpass')
    x_bp = filtfilt(b_bp, a_bp, x)
    # Rectify
    x_rect = np.abs(x_bp)
    # Low‑pass filter
    b_lp, a_lp = butter(4, lp_cutoff / nyq, btype='lowpass')
    envelope = filtfilt(b_lp, a_lp, x_rect)
    return envelope


def run_demo() -> Dict[str, float]:
    # Seed RNG for reproducible bearing fault simulation
    import numpy as np
    np.random.seed(0)
    fs = 16000.0
    data = generate_week_data('week_11', {'fs': fs, 'duration': 1.0})
    x = data['x']
    # Compute bandpower in a low‑frequency band (e.g., fault modulation band)
    bp = bandpower(x, fs, 40.0, 60.0)
    # Perform envelope detection: isolate carrier band then demodulate
    envelope = envelope_detection(x, fs, band=(1800.0, 2200.0), lp_cutoff=200.0)
    # Compute spectrum of the envelope
    N = len(envelope)
    env_fft = np.abs(rfft(envelope))
    freqs = rfftfreq(N, d=1.0 / fs)
    # Find amplitude at defect frequency (50 Hz)
    defect_freq = 50.0
    idx = np.argmin(np.abs(freqs - defect_freq))
    defect_amp = env_fft[idx]
    print(f"Bandpower 40–60 Hz: {bp:.4e}")
    print(f"Envelope defect frequency amplitude (50 Hz): {defect_amp:.4e}")
    return {
        'bandpower_40_60': float(bp),
        'defect_frequency_amplitude': float(defect_amp),
    }


if __name__ == '__main__':
    run_demo()