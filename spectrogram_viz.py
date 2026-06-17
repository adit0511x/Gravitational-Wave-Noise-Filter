"""
Deliverable 4: Spectrogram Visualisations – Residual Noise Suppression
=======================================================================
Generates side-by-side spectrograms of:
  (a) Raw noisy strain
  (b) Bandpass + whitened strain
  (c) Autoencoder-reconstructed (denoised) strain
  (d) Residual (noisy − reconstructed)

Plots noise power reduction across frequency bands as an additional panel.

Output: outputs/spectrograms.png
"""

import numpy as np
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import (generate_synthetic_strain, preprocess,
                            apply_bandpass, whiten,
                            SAMPLE_RATE, NPERSEG, FMIN, FMAX)


# ──────────────────────────────────────────────────────────────────────────────
#  Style constants
# ──────────────────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
CYAN    = "#00d4ff"
ORANGE  = "#ff6b35"
GREEN   = "#39d353"
GRAY    = "#8b949e"
LIGHT   = "#c9d1d9"


# ──────────────────────────────────────────────────────────────────────────────
#  Spectrogram helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_spectrogram(strain: np.ndarray, fs: float = SAMPLE_RATE,
                        nperseg: int = 256) -> tuple:
    """Return (t, f, Sxx_dB) with dB-scaled power."""
    f, t, Sxx = sp_signal.spectrogram(strain, fs=fs, nperseg=nperseg,
                                       noverlap=nperseg // 2,
                                       window="hann")
    Sxx_dB = 10 * np.log10(np.clip(Sxx, 1e-30, None))
    # Restrict to science band
    mask = (f >= FMIN) & (f <= FMAX)
    return t, f[mask], Sxx_dB[mask]


def noise_power_by_band(Sxx_dB: np.ndarray, freqs: np.ndarray,
                         bands: list) -> np.ndarray:
    """Mean power per frequency band."""
    powers = []
    for flo, fhi in bands:
        mask = (freqs >= flo) & (freqs < fhi)
        if mask.any():
            powers.append(float(np.mean(Sxx_dB[mask])))
        else:
            powers.append(np.nan)
    return np.array(powers)


# ──────────────────────────────────────────────────────────────────────────────
#  Main plotting function
# ──────────────────────────────────────────────────────────────────────────────
def plot_spectrograms(noisy_raw: np.ndarray,
                      clean_raw: np.ndarray,
                      recon_raw: np.ndarray,
                      fs:        float = SAMPLE_RATE,
                      out_path:  str   = "outputs/spectrograms.png",
                      event_name: str  = "Synthetic GW Event"):
    """
    Four-panel spectrogram plot + noise power reduction chart.
    """
    # ── Compute spectrograms ──────────────────────────────────────────────────
    t_n, f_n, S_noisy = compute_spectrogram(noisy_raw, fs)
    t_b, f_b, S_bp    = compute_spectrogram(apply_bandpass(noisy_raw, fs), fs)
    t_r, f_r, S_recon = compute_spectrogram(recon_raw,  fs)
    t_c, f_c, S_clean = compute_spectrogram(clean_raw,  fs)

    residual = noisy_raw[:len(recon_raw)] - recon_raw
    _, f_res, S_resid = compute_spectrogram(residual, fs)

    # Noise power reduction per band
    BANDS = [(20, 60), (60, 150), (150, 350), (350, 750), (750, 2048)]
    band_labels = ["20–60", "60–150", "150–350", "350–750", "750–2048"]

    power_noisy = noise_power_by_band(S_noisy, f_n, BANDS)
    power_recon = noise_power_by_band(S_recon, f_r, BANDS)
    power_reduction = power_noisy - power_recon      # positive = suppression

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                             hspace=0.45, wspace=0.35,
                             left=0.06, right=0.97, top=0.92, bottom=0.08)

    PANELS = [
        (S_noisy, f_n, t_n, "Raw Noisy Strain",          ORANGE, 0, 0),
        (S_bp,    f_b, t_b, "Bandpass + Whitened",        CYAN,   0, 1),
        (S_recon, f_r, t_r, "Autoencoder Reconstructed",  GREEN,  0, 2),
        (S_resid, f_res, t_n[:min(len(t_n),S_resid.shape[1])],
                             "Residual Noise",             GRAY,   0, 3),
    ]

    # Global colour scale
    vmin = min(S.min() for S, *_ in [(S_noisy,), (S_recon,), (S_clean,)])
    vmax = max(S.max() for S, *_ in [(S_noisy,), (S_recon,), (S_clean,)])
    norm = Normalize(vmin=vmin, vmax=vmax)

    for (S, f, t, title, accent, row, col) in PANELS:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(DARK_BG)
        T = t[:S.shape[1]]
        im = ax.pcolormesh(T, f, S[:, :len(T)], cmap="inferno",
                           norm=norm, shading="auto")
        ax.set_title(title, color=accent, fontsize=10, pad=6)
        ax.set_xlabel("Time (s)", color=LIGHT, fontsize=8)
        ax.set_ylabel("Frequency (Hz)", color=LIGHT, fontsize=8)
        ax.tick_params(colors=LIGHT, labelsize=7)
        ax.spines[:].set_color(GRAY)
        ax.set_yscale("log")
        ax.set_ylim(FMIN, FMAX)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.ax.tick_params(colors=LIGHT, labelsize=7)
        cb.set_label("Power (dB)", color=LIGHT, fontsize=7)

    # ── Row 2: Time-domain comparison (last 4 s) ──────────────────────────
    ax_td = fig.add_subplot(gs[1, :2])
    ax_td.set_facecolor(DARK_BG)
    fs_int = int(fs)
    t_arr  = np.arange(len(noisy_raw)) / fs
    t_zoom = t_arr[-4 * fs_int:]
    ax_td.plot(t_zoom, noisy_raw[-4 * fs_int:],
               color=ORANGE, alpha=0.6, linewidth=0.8, label="Noisy")
    ax_td.plot(t_zoom, recon_raw[-4 * fs_int:len(noisy_raw)],
               color=GREEN,  alpha=0.9, linewidth=1.0, label="Reconstructed")
    ax_td.plot(t_zoom, clean_raw[-4 * fs_int:],
               color=CYAN,   alpha=0.7, linewidth=0.8, linestyle="--",
               label="Clean (truth)")
    ax_td.set_title("Time-Domain: Final 4 s", color=LIGHT, fontsize=10)
    ax_td.set_xlabel("Time (s)", color=LIGHT, fontsize=8)
    ax_td.set_ylabel("Strain (normalised)", color=LIGHT, fontsize=8)
    ax_td.tick_params(colors=LIGHT)
    ax_td.spines[:].set_color(GRAY)
    ax_td.legend(facecolor="#1c2128", labelcolor=LIGHT, edgecolor=GRAY,
                 fontsize=8)
    ax_td.grid(color=GRAY, alpha=0.2)

    # ── Row 2: Noise Power Reduction by Band ──────────────────────────────
    ax_pwr = fig.add_subplot(gs[1, 2:])
    ax_pwr.set_facecolor(DARK_BG)
    x = np.arange(len(BANDS))
    colors = [GREEN if v >= 0 else ORANGE for v in power_reduction]
    ax_pwr.bar(x, power_reduction, color=colors, alpha=0.85, zorder=3)
    ax_pwr.axhline(0, color=GRAY, linewidth=1)
    ax_pwr.set_xticks(x)
    ax_pwr.set_xticklabels([f"{b}\nHz" for b in band_labels],
                            color=LIGHT, fontsize=8)
    ax_pwr.set_ylabel("Noise Power Reduction (dB)", color=LIGHT)
    ax_pwr.set_title("Noise Suppression by Frequency Band", color=LIGHT, pad=6)
    ax_pwr.tick_params(colors=LIGHT)
    ax_pwr.spines[:].set_color(GRAY)
    ax_pwr.grid(axis="y", color=GRAY, alpha=0.3, zorder=0)
    for xi, (v, c) in enumerate(zip(power_reduction, colors)):
        ax_pwr.text(xi, v + 0.3, f"{v:+.1f}", ha="center", color=c, fontsize=9)

    # ── Row 3: PSD comparison ─────────────────────────────────────────────
    ax_psd = fig.add_subplot(gs[2, :])
    ax_psd.set_facecolor(DARK_BG)
    from scipy.signal import welch
    fn, Pn = welch(noisy_raw, fs=fs, nperseg=NPERSEG)
    fr, Pr = welch(recon_raw, fs=fs, nperseg=NPERSEG)
    fc, Pc = welch(clean_raw, fs=fs, nperseg=NPERSEG)
    band_m = (fn >= FMIN) & (fn <= FMAX)
    ax_psd.semilogy(fn[band_m], Pn[band_m], color=ORANGE, alpha=0.8,
                    linewidth=1.2, label="Noisy")
    ax_psd.semilogy(fr[band_m], Pr[band_m], color=GREEN,  alpha=0.9,
                    linewidth=1.2, label="Reconstructed")
    ax_psd.semilogy(fc[band_m], Pc[band_m], color=CYAN,   alpha=0.7,
                    linewidth=1.0, linestyle="--", label="Clean (truth)")
    ax_psd.set_xlabel("Frequency (Hz)", color=LIGHT)
    ax_psd.set_ylabel("PSD  [strain²/Hz]", color=LIGHT)
    ax_psd.set_title("Power Spectral Density Comparison", color=LIGHT, pad=6)
    ax_psd.tick_params(colors=LIGHT)
    ax_psd.spines[:].set_color(GRAY)
    ax_psd.legend(facecolor="#1c2128", labelcolor=LIGHT, edgecolor=GRAY,
                  fontsize=9)
    ax_psd.grid(color=GRAY, alpha=0.2, which="both")
    ax_psd.set_xscale("log")

    # ── Super title ───────────────────────────────────────────────────────
    fig.suptitle(
        f"Gravitational Wave Noise Filter — Spectrogram Analysis\n{event_name}",
        color=CYAN, fontsize=14, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"  Spectrogram saved → {out_path}")
    return power_reduction


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from scipy.signal import wiener

    print("=== Spectrogram Visualisation ===\n")
    noisy_raw, clean_raw = generate_synthetic_strain(duration=32.0,
                                                     snr=6.0, seed=7)
    # Simulate reconstruction with Wiener filter (baseline)
    recon_raw = wiener(noisy_raw, mysize=15)

    pr = plot_spectrograms(noisy_raw, clean_raw, recon_raw,
                           event_name="Synthetic BBH Merger (SNR ≈ 6)")

    total_reduction = np.nanmean(pr)
    pct_reduction   = (1 - 10 ** (-total_reduction / 10)) * 100
    print(f"\n  Mean noise power reduction : {total_reduction:.2f} dB")
    print(f"  Equivalent power reduction : {pct_reduction:.1f}%  "
          f"(target ≥ 40%)")
    print("\n✓  Spectrogram visualisation complete.")
