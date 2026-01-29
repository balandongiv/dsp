## Week 2 – Sampling and Anti‑Aliasing

This lesson focuses on the operational aspects of the sampling theorem and anti‑aliasing.  You will sample a synthetic vibration signal at different rates, apply a simple anti‑aliasing filter via resampling and compare the resulting spectra.  The demonstration computes the signal‑to‑noise ratio before and after down‑sampling and reports an approximate aliasing level.

### How to Run

```
python demo.py
```

### Expected Output

The script prints SNR values and an aliasing level when comparing high and low sampling rates.  An example output is:

```
High FS = 2000 Hz → SNR = 39.3 dB
Low  FS = 500 Hz → SNR = 45.6 dB
Aliasing level (SNR drop) = −6.3 dB
```

Exact values may vary.

### Completion Criteria

To complete Week 2, your submission must:

1. Show the effect of at least two sampling rates on the quality of the sampled signal.
2. Compute and report SNR or aliasing level for each sampling rate.
3. Comment on why the chosen anti‑aliasing strategy works or fails.