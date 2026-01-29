"""Synthetic data generators for the DSP course.

Each week of the course requires a particular type of synthetic signal.  The
functions in this module provide minimal examples that match the spirit of
the syllabus without trying to exhaustively model every industrial
phenomenon.  Instructors and learners can modify these functions to
experiment with different frequencies, noise levels and transient events.
"""

from typing import Tuple, List, Optional, Dict

import numpy as np

def generate_sine_wave(freqs: List[float], amps: Optional[List[float]] = None,
                       phase: Optional[List[float]] = None, duration: float = 1.0,
                       fs: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a multi‑tone sine wave.

    Parameters
    ----------
    freqs : list of float
        Frequencies of the sine components in hertz.
    amps : list of float, optional
        Amplitudes for each component.  Defaults to all ones.
    phase : list of float, optional
        Phase offsets for each component in radians.  Defaults to zeros.
    duration : float
        Duration of the signal in seconds.
    fs : float
        Sampling rate in samples per second.

    Returns
    -------
    t : numpy.ndarray
        Time vector.
    signal : numpy.ndarray
        The generated multi‑tone signal.
    """
    if amps is None:
        amps = [1.0 for _ in freqs]
    if phase is None:
        phase = [0.0 for _ in freqs]
    t = np.arange(0, duration, 1.0 / fs)
    signal = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, phase):
        signal += a * np.sin(2.0 * np.pi * f * t + p)
    return t, signal


def add_gaussian_noise(signal: np.ndarray, std: float) -> np.ndarray:
    """Add zero‑mean Gaussian noise to a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal.
    std : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    noisy_signal : numpy.ndarray
        Signal with added noise.
    """
    return signal + std * np.random.randn(*signal.shape)


def generate_week_data(week_id: str, params: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """Generate synthetic data for a given week.

    This high‑level generator dispatches to simple signal models for
    different weeks.  It accepts an optional parameter dictionary to
    override defaults such as sampling rate, duration and noise level.

    The returned dictionary contains one or more keys depending on the
    modality (e.g. vibration, current, temperature).  At a minimum it
    returns a time vector (`t`) and a primary signal (`x`).

    Parameters
    ----------
    week_id : str
        Course week identifier (e.g. "week_01", "week_02", ...).
    params : dict, optional
        Override default parameters such as `fs`, `duration`, `noise_std`.

    Returns
    -------
    data : dict
        Dictionary containing at least `t` and `x`.  Additional keys
        (`y`, `z`, etc.) may be present for multichannel weeks.
    """
    if params is None:
        params = {}
    # default global parameters
    fs = params.get('fs', 5000.0)
    duration = params.get('duration', 1.0)
    noise_std = params.get('noise_std', 0.01)
    # Select a basic signal based on the week
    if week_id == 'week_01':
        # Motor current + vibration fundamental and harmonics
        freqs = [50.0, 150.0, 300.0]
        amps = [1.0, 0.5, 0.2]
        t, x_clean = generate_sine_wave(freqs, amps, duration=duration, fs=fs)
        # Add a Gaussian pulse as a transient event
        pulse_center = duration / 2.0
        pulse_width = 0.02
        pulse = np.exp(-0.5 * ((t - pulse_center) / pulse_width) ** 2)
        x_clean += 0.5 * pulse
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_02':
        # Two sine tones for sampling/aliasing demonstration
        freqs = [50.0, 200.0]
        amps = [1.0, 0.8]
        t, x_clean = generate_sine_wave(freqs, amps, duration=duration, fs=fs)
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_03':
        # Multi‑tone for FFT analysis
        freqs = [120.0, 300.0]
        amps = [1.0, 0.7]
        t, x_clean = generate_sine_wave(freqs, amps, duration=duration, fs=fs)
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_04':
        # Low‑pass + high‑frequency noise for FIR filtering
        freqs = [500.0, 2000.0]
        amps = [1.0, 0.3]
        t, x_clean = generate_sine_wave(freqs, amps, duration=duration, fs=fs)
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_05':
        # Thermocouple with slow drift and oscillation for IIR filters
        t = np.arange(0, duration, 1.0 / fs)
        drift = 0.1 * t  # slow drift
        osc = 0.5 * np.sin(2.0 * np.pi * 5.0 * t)
        emi = 0.1 * np.sin(2.0 * np.pi * 60.0 * t)
        x_clean = drift + osc + emi
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_06':
        # Gearbox vibration with multiple tones for PSD/Welch
        freqs = [60.0, 150.0, 250.0]
        amps = [1.0, 0.6, 0.4]
        t, x_clean = generate_sine_wave(freqs, amps, duration=duration, fs=fs)
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_07':
        # STFT demonstration with chirp
        t = np.arange(0, duration, 1.0 / fs)
        # Linear chirp from 100 to 1000 Hz
        k = (1000.0 - 100.0) / duration
        chirp = np.sin(2.0 * np.pi * (100.0 * t + 0.5 * k * t ** 2))
        x_clean = chirp
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_08':
        # Impulsive events for wavelet analysis
        t = np.arange(0, duration, 1.0 / fs)
        x_clean = np.zeros_like(t)
        # generate random impulses
        rng = np.random.default_rng()
        for center in rng.uniform(0, duration, size=5):
            x_clean += np.exp(-0.5 * ((t - center) / 0.005) ** 2)
        # baseline low‑frequency trend
        x_clean += 0.2 * np.sin(2.0 * np.pi * 10.0 * t)
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_09':
        # Two channels for multichannel analysis
        t = np.arange(0, duration, 1.0 / fs)
        # Shared 100 Hz tone
        shared = np.sin(2.0 * np.pi * 100.0 * t)
        ch1 = shared + 0.5 * np.sin(2.0 * np.pi * 150.0 * t)
        ch2 = shared + 0.5 * np.sin(2.0 * np.pi * 200.0 * t + np.pi / 4)
        x1 = add_gaussian_noise(ch1, noise_std)
        x2 = add_gaussian_noise(ch2, noise_std)
        return {'t': t, 'x1': x1, 'x2': x2, 'x1_clean': ch1, 'x2_clean': ch2}
    elif week_id == 'week_10':
        # Bearing vibration with impulsive noise events for time‑domain features
        t = np.arange(0, duration, 1.0 / fs)
        base = np.sin(2.0 * np.pi * 100.0 * t)
        # Add random impulses
        rng = np.random.default_rng()
        impulses = np.zeros_like(t)
        for center in rng.uniform(0, duration, size=10):
            impulses += 0.5 * np.exp(-100.0 * (t - center) ** 2)
        x_clean = base + impulses
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_11':
        # Bearing fault with band‑limited impacts for spectral features
        t = np.arange(0, duration, 1.0 / fs)
        carrier = np.sin(2.0 * np.pi * 2000.0 * t)
        # Envelope modulated by a lower frequency representing fault impacts
        mod_freq = 50.0
        envelope = 1.0 + 0.5 * np.sin(2.0 * np.pi * mod_freq * t)
        x_clean = envelope * carrier
        # Add random EMI tone
        x_clean += 0.2 * np.sin(2.0 * np.pi * 60.0 * t)
        x = add_gaussian_noise(x_clean, noise_std)
        return {'t': t, 'x': x, 'x_clean': x_clean}
    elif week_id == 'week_12':
        # Data for supervised classification (single channel)
        # Generate segments with different faults (A: 1x tone; B: 1x + 2x; C: impulsive bursts)
        segment_length = int(params.get('segment_length_samples', fs * 0.5))
        num_segments = int(duration * fs // segment_length)
        segments = []
        labels = []
        for i in range(num_segments):
            t_seg = np.arange(0, segment_length) / fs
            if i % 4 == 0:
                # healthy: base tone
                sig = np.sin(2.0 * np.pi * 50.0 * t_seg)
                label = 'healthy'
            elif i % 4 == 1:
                # class A: imbalance (strong 1x)
                sig = 1.5 * np.sin(2.0 * np.pi * 50.0 * t_seg)
                label = 'class_A'
            elif i % 4 == 2:
                # class B: misalignment (1x + 2x)
                sig = np.sin(2.0 * np.pi * 50.0 * t_seg) + 0.8 * np.sin(2.0 * np.pi * 100.0 * t_seg)
                label = 'class_B'
            else:
                # class C: bearing fault (impulses)
                sig = np.sin(2.0 * np.pi * 50.0 * t_seg)
                # add one impulse at random location
                center = np.random.uniform(0.1, t_seg[-1] - 0.1)
                sig += np.exp(-100.0 * (t_seg - center) ** 2)
                label = 'class_C'
            sig_noisy = add_gaussian_noise(sig, noise_std)
            segments.append(sig_noisy)
            labels.append(label)
        return {'segments': np.array(segments), 'labels': np.array(labels)}
    elif week_id == 'week_13':
        # Data for unsupervised anomaly detection
        # Generate healthy baseline segments and a few anomalous segments with impulses
        segment_length = int(params.get('segment_length_samples', fs * 0.5))
        num_segments = int(duration * fs // segment_length)
        segments = []
        labels = []
        for i in range(num_segments):
            t_seg = np.arange(0, segment_length) / fs
            sig = np.sin(2.0 * np.pi * 50.0 * t_seg)
            # add drift
            sig += 0.001 * np.arange(len(sig))
            if i % 6 == 0:
                # anomaly: random impulse
                center = np.random.uniform(0.1, t_seg[-1] - 0.1)
                sig += 0.5 * np.exp(-100.0 * (t_seg - center) ** 2)
                labels.append('anomaly')
            else:
                labels.append('healthy')
            seg_noisy = add_gaussian_noise(sig, noise_std)
            segments.append(seg_noisy)
        return {'segments': np.array(segments), 'labels': np.array(labels)}
    elif week_id == 'week_14':
        # Capstone: time–frequency data for neural model
        # Create a few segments with spectrogram‑friendly signals
        segment_length = int(params.get('segment_length_samples', fs * 0.5))
        num_segments = int(duration * fs // segment_length)
        segments = []
        labels = []
        for i in range(num_segments):
            t_seg = np.arange(0, segment_length) / fs
            if i % 3 == 0:
                # healthy: stable tone
                sig = np.sin(2.0 * np.pi * 100.0 * t_seg)
                label = 'healthy'
            elif i % 3 == 1:
                # fault: short bursts
                sig = np.sin(2.0 * np.pi * 100.0 * t_seg)
                # add multiple bursts
                for center in [0.1, 0.3, 0.45]:
                    sig += 0.5 * np.exp(-200.0 * (t_seg - center) ** 2)
                label = 'fault'
            else:
                # regime change: base tone shifts frequency
                freq = 100.0 + 20.0 * (i % 5)
                sig = np.sin(2.0 * np.pi * freq * t_seg)
                label = 'regime'
            seg_noisy = add_gaussian_noise(sig, noise_std)
            segments.append(seg_noisy)
            labels.append(label)
        return {'segments': np.array(segments), 'labels': np.array(labels)}
    else:
        raise ValueError(f"Unsupported week_id: {week_id}")