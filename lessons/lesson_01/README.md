## Week 1 – Introduction to Digital Signal Processing

This lesson introduces DSP as an information‑management workflow for instrumentation.  You will explore how sampling rate, bit depth and anti‑alias filtering affect the digitisation of sensor signals and how to quantify distortions such as quantisation noise and clipping.  The provided demonstration synthesises a multi‑tone signal with a transient event, simulates noise and sampling, and reports the signal‑to‑noise ratio.

### How to Run

```
python demo.py
```

The script will generate synthetic motor‑current‑like data, add noise and a transient event, and compute the SNR between the clean and noisy signals.  You can adjust the sampling rate and noise level by modifying the parameters at the top of `demo.py`.

### Expected Output

Running `python demo.py` prints the SNR values (in decibels) for each sampling rate tested.  An example output is:

```
Sampling rate 1000 Hz → SNR = 38.3 dB
Sampling rate 5000 Hz → SNR = 38.2 dB
```

Your results may differ slightly depending on random noise.

### Completion Criteria

To satisfy the Week 1 completion criteria, ensure that your submission:

1. Sweeps at least two sampling rates or bit depths (simulated via the `fs` parameter).
2. Reports at least one metric (e.g., SNR) for each scenario.
3. Includes a short justification in your report about which information you preserved and what was allowed to be lost.