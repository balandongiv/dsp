## Week 10 – Statistical Features From Time Domain

This lesson covers time‑domain statistical features for condition monitoring, including RMS, crest factor, kurtosis and skewness.  You will compute these features for a synthetic vibration signal with impulsive events, compare them for healthy and faulty scenarios and interpret their physical meaning.

### How to Run

```
python demo.py
```

### Expected Output

The script computes statistical features and prints them for the synthetic vibration signal.  Example output:

```
RMS: 0.7760
Peak: 2.3727
Crest factor: 3.06
Kurtosis: 1.92
Skewness: 0.09
```

The values will differ based on the random data but should reveal larger crest factor and kurtosis when impulses are present.

### Completion Criteria

1. Compute RMS, peak, crest factor, kurtosis and skewness for the given signal.
2. Plot or print the feature values and discuss how they change with faults.
3. Report at least one metric trend over time or across scenarios.