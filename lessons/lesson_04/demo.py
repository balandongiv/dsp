"""Week 4 demonstration: FIR filter design and linear phase.

This script designs a low‑pass FIR filter using a Hamming window, applies it
to a noisy multi‑tone signal and computes the SNR improvement.  Learners can
experiment with the filter order and cutoff frequency to trade off between
attenuation and computational cost.
"""

from typing import Dict

import numpy as np
from scipy.signal import firwin, lfilter

import os
import sys
# Make sure dsp_utils can be imported when this script is run as a standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data
from dsp_utils.metrics import snr_db

# FIR filter parameters
FILTER_ORDER = 101
CUTOFF_HZ = 1000.0  # Low‑pass cutoff frequency in Hz


def run_demo() -> Dict[str, float]:
    # Fix the random seed for reproducible noise in the synthetic signal
    import numpy as np
    np.random.seed(0)
    """Run the Week 4 demonstration.

    Returns a dictionary containing the SNR before and after filtering and
    the improvement in dB.
    """
    # Generate synthetic optical signal with low‑frequency and high‑frequency components
    fs = 10000.0
    data = generate_week_data('week_04', {'fs': fs, 'duration': 1.0})
    x_clean = data['x_clean']
    x_noisy = data['x']
    # Design a low‑pass FIR filter (linear phase) using a Hamming window
    nyq = fs / 2.0
    normalized_cutoff = CUTOFF_HZ / nyq
    coeffs = firwin(FILTER_ORDER, normalized_cutoff, window='hamming')
    # Apply the filter to the noisy signal
    x_filt = lfilter(coeffs, 1.0, x_noisy)
    # Also filter the clean signal so that the reference is in the same band
    x_clean_filt = lfilter(coeffs, 1.0, x_clean)
    # Align filtered and clean signals (filter introduces group delay of (order‑1)/2 samples)
    delay = (FILTER_ORDER - 1) // 2
    # Baseline SNR computed on unfiltered signals after aligning for group delay
    x_clean_aligned = x_clean[delay:]
    x_noisy_aligned = x_noisy[delay:]
    # Reference for after filtering is the filtered clean signal
    x_clean_filt_aligned = x_clean_filt[delay:]
    x_filt_aligned = x_filt[delay:]
    # Compute SNR before filtering (relative to original clean signal)
    snr_before = snr_db(x_clean_aligned, x_noisy_aligned - x_clean_aligned)
    # Compute SNR after filtering relative to the low‑pass reference
    snr_after = snr_db(x_clean_filt_aligned, x_filt_aligned - x_clean_filt_aligned)
    improvement = snr_after - snr_before
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after  filtering: {snr_after:.2f} dB")
    print(f"Improvement: {improvement:.2f} dB")
    return {
        'snr_before': snr_before,
        'snr_after': snr_after,
        'improvement': improvement,
    }


if __name__ == '__main__':
    run_demo()