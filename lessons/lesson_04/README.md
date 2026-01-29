## Week 4 – FIR Filter Design and Linear Phase

This lesson introduces finite impulse response (FIR) filters as a black‑box tool for removing high‑frequency noise while preserving waveform shape.  You will design a low‑pass FIR filter using the window method, apply it to a noisy photodiode‑like signal and observe the improvement in SNR.

### How to Run

```
python demo.py
```

### Expected Output

Running `demo.py` prints the SNR before and after filtering, as well as the improvement in decibels.  For example:

```
SNR before filtering: 37.5 dB
SNR after  filtering: 44.3 dB
Improvement: 6.8 dB
```

Values may vary depending on the noise realisation and filter parameters.

### Completion Criteria

1. Design an FIR filter (low‑pass or band‑pass) using SciPy’s `firwin` function.
2. Apply the filter to the synthetic data and report an improvement in SNR or RMSE relative to the unfiltered signal.
3. Comment on how filter order affects computational cost and the transition band.