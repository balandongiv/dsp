"""Completion checks for Week 7.

The check ensures that the STFT is computed and that time and frequency
resolution metrics are sensible numbers.  It also verifies that the
spectrogram contains non‑zero energy.
"""

import os
import sys
# Add package root so dsp_utils resolves when running this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
from scipy.signal import stft


def test_stft_metrics() -> None:
    """Ensure STFT time and frequency resolutions are sensible and energy is non-zero."""
    results = run_demo()
    assert 'time_resolution' in results and 'frequency_resolution' in results, \
        "Missing resolution metrics"
    assert results['time_resolution'] > 0.0, "Time resolution must be positive"
    assert results['frequency_resolution'] > 0.0, "Frequency resolution must be positive"
    assert results['max_spectrogram_amplitude'] > 0.0, \
        "Spectrogram should contain non‑zero energy"

def test_resolution_tradeoff() -> None:
    """Verify that increasing the FFT size improves frequency resolution."""
    baseline = run_demo()
    baseline_freq_res = baseline['frequency_resolution']
    fs = 4000.0
    data = generate_week_data('week_07', {'fs': fs, 'duration': 2.0})
    x = data['x']
    # Compute STFT with a larger FFT size while keeping window length constant
    nfft = 512
    _, _, Zxx = stft(x, fs=fs, window='hann', nperseg=256, noverlap=128, nfft=nfft)
    freq_res = fs / nfft
    assert freq_res < baseline_freq_res, \
        "A larger FFT size should yield finer frequency resolution than the baseline"


if __name__ == '__main__':
    test_stft_metrics()
    test_resolution_tradeoff()
    print('All Week 7 checks passed.')