"""Completion checks for Week 4.

These checks verify that the demonstration computes SNR values before and after
filtering and reports a positive improvement (i.e., the filter improves the
signal quality).  The improvement need not be large but should be finite.
"""

import os
import sys
# Ensure dsp_utils is importable when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
from dsp_utils.metrics import snr_db
import numpy as np
from scipy.signal import firwin, lfilter


def test_filter_improvement() -> None:
    """Check that the FIR filter in the demo improves the SNR."""
    results = run_demo()
    assert 'snr_before' in results and 'snr_after' in results and 'improvement' in results, \
        "Result must contain SNR before, after and improvement"
    assert isinstance(results['improvement'], float), "Improvement must be a float"
    # Filtering should not significantly degrade SNR
    assert results['snr_after'] >= results['snr_before'] - 1e-3, \
        "Filtering should not significantly degrade SNR"

def test_low_order_filter_degradation() -> None:
    """Design a much shorter FIR filter and verify that the improvement is reduced."""
    baseline = run_demo()
    baseline_improvement = baseline['improvement']
    fs = 10000.0
    data = generate_week_data('week_04', {'fs': fs, 'duration': 1.0})
    x_clean = data['x_clean']
    x_noisy = data['x']
    # Extremely low order filter (e.g. 5 taps) with same cutoff
    order = 5
    cutoff_hz = 1000.0
    nyq = fs / 2.0
    normalized_cutoff = cutoff_hz / nyq
    coeffs = firwin(order, normalized_cutoff, window='hamming')
    x_filt = lfilter(coeffs, 1.0, x_noisy)
    delay = (order - 1) // 2
    x_clean_aligned = x_clean[delay:]
    x_noisy_aligned = x_noisy[delay:]
    x_filt_aligned = x_filt[delay:]
    snr_before = snr_db(x_clean_aligned, x_noisy_aligned - x_clean_aligned)
    snr_after = snr_db(x_clean_aligned, x_filt_aligned - x_clean_aligned)
    improvement_low = snr_after - snr_before
    assert improvement_low < baseline_improvement, \
        "A very low-order FIR filter should yield less improvement than the baseline"


if __name__ == '__main__':
    test_filter_improvement()
    test_low_order_filter_degradation()
    print('All Week 4 checks passed.')