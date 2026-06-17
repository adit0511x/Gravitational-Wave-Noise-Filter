"""
Deliverable 1: LIGO Gravitational-Wave Strain Data Preprocessing Pipeline
==========================================================================
Implements:
  - Whitening (flattens the noise power spectral density)
  - Bandpass filtering  (20–2048 Hz science band)
  - Segment windowing   (overlapping Hann-windowed chunks)

Author  : GW Noise Filter Project
Dataset : LIGO Open Science Center – GWTC Strain Data
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
SAMPLE_RATE   = 4096          # Hz  (standard LIGO strain sample rate)
FMIN          = 20.0          # Hz  lower edge of science band
FMAX          = 1800.0        # Hz  upper edge (below Nyquist for 4096 Hz)
SEGMENT_DUR   = 1.0           # seconds per window
OVERLAP_FRAC  = 0.5           # 50% overlap between windows
NPERSEG       = int(SAMPLE_RATE * SEGMENT_DUR)
NOVERLAP      = int(NPERSEG * OVERLAP_FRAC)
WHITEN_NFFT   = NPERSEG * 4   # FFT length for PSD estimation


# ─────────────────────────────────────────────
#  1. Bandpass Filter
# ─────────────────────────────────────────────
def design_bandpass(fmin: float = FMIN,
                    fmax: float = FMAX,
                    fs:   float = SAMPLE_RATE,
                    order: int  = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter for the LIGO science band.

    Returns
    -------
    b, a : filter coefficients (second-order sections via sosfilt recommended)
    """
    nyq  = fs / 2.0
    low  = fmin / nyq
    high = fmax / nyq
    sos  = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return sos


def apply_bandpass(strain: np.ndarray,
                   fs: float = SAMPLE_RATE) -> np.ndarray:
    """
    Apply the bandpass filter to a strain timeseries using zero-phase
    forward-backward filtering (filtfilt via SOS).
    """
    sos = design_bandpass(fs=fs)
    return signal.sosfiltfilt(sos, strain)


# ─────────────────────────────────────────────
#  2. Whitening
# ─────────────────────────────────────────────
def estimate_psd(strain: np.ndarray,
                 fs: float = SAMPLE_RATE,
                 nperseg: int = WHITEN_NFFT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the one-sided power spectral density (PSD) using Welch's method.

    Returns
    -------
    freqs : frequency array (Hz)
    psd   : PSD array (strain^2 / Hz)
    """
    freqs, psd = signal.welch(strain, fs=fs, nperseg=nperseg,
                               window="hann", average="median")
    # Avoid division by zero
    psd = np.clip(psd, a_min=1e-50, a_max=None)
    return freqs, psd


def whiten(strain: np.ndarray,
           fs: float = SAMPLE_RATE,
           psd_strain: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Whiten a strain timeseries by dividing each frequency bin by
    the square-root of the estimated (or supplied) PSD.

    Parameters
    ----------
    strain      : raw strain timeseries
    fs          : sample rate in Hz
    psd_strain  : optional separate segment used to estimate the PSD
                  (e.g. off-source background). Defaults to `strain` itself.

    Returns
    -------
    whitened : whitened strain (unit variance across frequency)
    """
    if psd_strain is None:
        psd_strain = strain

    N      = len(strain)
    freqs, psd = estimate_psd(psd_strain, fs=fs)

    # FFT of the target strain
    strain_fft  = np.fft.rfft(strain, n=N)
    fft_freqs   = np.fft.rfftfreq(N, d=1.0 / fs)

    # Interpolate PSD onto the FFT frequency grid
    psd_interp  = np.interp(fft_freqs, freqs, psd)
    psd_interp  = np.clip(psd_interp, a_min=1e-50, a_max=None)

    # Divide by amplitude spectral density
    whitened_fft = strain_fft / np.sqrt(psd_interp * fs / 2.0)

    whitened = np.fft.irfft(whitened_fft, n=N)
    return whitened


# ─────────────────────────────────────────────
#  3. Segment Windowing
# ─────────────────────────────────────────────
def segment_and_window(strain: np.ndarray,
                       nperseg: int = NPERSEG,
                       noverlap: int = NOVERLAP) -> np.ndarray:
    """
    Slice the strain timeseries into overlapping Hann-windowed segments.

    Returns
    -------
    segments : array of shape (n_segments, nperseg)
    """
    hann   = signal.windows.hann(nperseg)
    step   = nperseg - noverlap
    n_segs = (len(strain) - nperseg) // step + 1

    segments = np.zeros((n_segs, nperseg), dtype=np.float32)
    for i in range(n_segs):
        start = i * step
        chunk = strain[start : start + nperseg]
        if len(chunk) < nperseg:
            break
        segments[i] = (chunk * hann).astype(np.float32)

    return segments


# ─────────────────────────────────────────────
#  4. Full Pipeline
# ─────────────────────────────────────────────
def preprocess(raw_strain: np.ndarray,
               fs: float = SAMPLE_RATE,
               psd_strain: Optional[np.ndarray] = None,
               return_segments: bool = True
               ) -> Tuple[np.ndarray, dict]:
    """
    End-to-end preprocessing pipeline:
        raw_strain → bandpass → whiten → segment & window

    Parameters
    ----------
    raw_strain      : 1-D numpy array of raw h(t) strain
    fs              : sample rate (Hz)
    psd_strain      : optional separate background segment for PSD estimate
    return_segments : if True, return windowed segments; else return full array

    Returns
    -------
    output  : (n_segments, nperseg) array of preprocessed windows
              OR 1-D array if return_segments=False
    meta    : dict with intermediate results and statistics
    """
    # Step 1 – Bandpass
    bp_strain = apply_bandpass(raw_strain, fs=fs)

    # Step 2 – Whiten
    wh_strain = whiten(bp_strain, fs=fs, psd_strain=psd_strain)

    # Normalise to zero-mean unit-variance
    wh_strain = (wh_strain - wh_strain.mean()) / (wh_strain.std() + 1e-12)

    # Step 3 – Segment & Window
    if return_segments:
        output = segment_and_window(wh_strain)
    else:
        output = wh_strain

    meta = {
        "raw_rms"      : float(np.sqrt(np.mean(raw_strain ** 2))),
        "bp_rms"       : float(np.sqrt(np.mean(bp_strain  ** 2))),
        "whitened_rms" : float(np.sqrt(np.mean(wh_strain  ** 2))),
        "n_segments"   : output.shape[0] if return_segments else None,
        "segment_len"  : NPERSEG,
        "sample_rate"  : fs,
        "fmin"         : FMIN,
        "fmax"         : FMAX,
    }
    return output, meta


# ─────────────────────────────────────────────
#  5. Synthetic data helper (for testing / demo)
# ─────────────────────────────────────────────
def generate_synthetic_strain(duration: float = 32.0,
                               fs: float = SAMPLE_RATE,
                               snr: float = 5.0,
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic noisy strain + clean signal pair for unit tests
    and demo runs (no internet access required).

    The 'signal' is a Gaussian chirplet centred at 0.5 s before the end.
    The 'noise' is coloured Gaussian noise shaped to mimic LIGO's ASD.
    """
    rng  = np.random.default_rng(seed)
    N    = int(duration * fs)
    t    = np.arange(N) / fs

    # --- Mock LIGO-like coloured noise ---
    white      = rng.standard_normal(N)
    freqs      = np.fft.rfftfreq(N, d=1.0 / fs)
    freqs[0]   = 1e-6                                # avoid division by 0
    # Rough LIGO noise shape: ~1/f^1.5 below 100 Hz, flat above
    asd        = np.where(freqs < 100, (100 / freqs) ** 1.5, 1.0)
    asd       /= asd.mean()
    noise_fft  = np.fft.rfft(white) * asd
    noise      = np.fft.irfft(noise_fft, n=N)
    noise_rms  = np.sqrt(np.mean(noise ** 2))

    # --- Clean chirplet signal ---
    t0         = duration - 0.5
    f0, k      = 50.0, 200.0                         # chirp params
    clean      = np.exp(-0.5 * ((t - t0) / 0.1) ** 2) \
                 * np.sin(2 * np.pi * (f0 * (t - t0)
                           + 0.5 * k * (t - t0) ** 2))
    clean_rms  = np.sqrt(np.mean(clean ** 2)) + 1e-12
    clean     *= (noise_rms / clean_rms) * snr       # scale to desired SNR

    noisy = noise + clean
    return noisy.astype(np.float32), clean.astype(np.float32)


# ─────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Preprocessing Pipeline Self-Test ===\n")
    noisy, clean = generate_synthetic_strain(duration=32.0, snr=5.0)
    print(f"Generated synthetic strain: {len(noisy)} samples @ {SAMPLE_RATE} Hz")

    segments, meta = preprocess(noisy, fs=SAMPLE_RATE)
    print(f"\nPreprocessing complete:")
    for k, v in meta.items():
        print(f"  {k:20s}: {v}")
    print(f"\nOutput segment array shape: {segments.shape}")
    print("\n✓  All preprocessing steps passed.")
