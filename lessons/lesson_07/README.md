## Week 7 – Time–Frequency Analysis with STFT

This lesson uses the short‑time Fourier transform (STFT) to analyse non‑stationary signals.  You will compute spectrograms for a chirp signal, explore the trade‑off between time and frequency resolution by varying the window length and overlap and identify transient events.

### How to Run

```
python demo.py
```

### Expected Output

The script prints approximate time and frequency resolutions based on the chosen window length and overlap.  It may output:

```
Time resolution (approx): 0.0640 s
Frequency resolution (approx): 15.62 Hz
```

It also displays spectrograms illustrating how varying the window affects resolution.

### Completion Criteria

1. Compute the STFT of a non‑stationary signal and visualise or inspect the spectrogram.
2. Adjust the window length and overlap to observe time–frequency trade‑offs.
3. Report quantitative measures of time and frequency resolution.