## Week 5 – IIR Filter Design and Stability

This lesson introduces infinite impulse response (IIR) filters, including Butterworth and Chebyshev designs.  You will design a band‑pass IIR filter to isolate a low‑frequency oscillation in a synthetic thermocouple signal and evaluate stability and phase distortion.  The demonstration computes the SNR before and after filtering and prints the filter’s pole locations to confirm stability.

### How to Run

```
python demo.py
```

### Expected Output

The script prints the SNR before and after filtering along with the improvement and reports whether the designed filter is stable.  For example:

```
SNR before filtering: 37.1 dB
SNR after  filtering: 42.3 dB
Improvement: 5.2 dB
Filter stability: stable
```

If the filter is unstable or has too much ripple, the script will show degraded improvement or mark the filter as unstable.

### Completion Criteria

1. Design an IIR filter (Butterworth or Chebyshev) with specified passband and stopband.
2. Apply the filter to the signal and report SNR improvement.
3. Inspect the filter’s poles to ensure they lie inside the unit circle and comment on phase distortion.