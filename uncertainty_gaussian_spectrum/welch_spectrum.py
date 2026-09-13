"""Welch PSD with Gaussian-process checks and chi-squared uncertainties.

Import and call ``welch_spectrum_with_checks_and_uncertainties``. Theory: the
companion notebook ``uncertainty_gaussian_spectrum.ipynb``.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "WelchSpectrumResult",
    "estimate_welch_edf",
    "welch_spectrum_with_checks_and_uncertainties",
]


class WelchSpectrumResult(NamedTuple):
    """Unpackable result of ``welch_spectrum_with_checks_and_uncertainties``.

    ``uncertainty_factors`` are multiplicative: at the requested two-sided
    confidence level,

        lower = welch_spectrum * uncertainty_factors[0]
        upper = welch_spectrum * uncertainty_factors[1]

    On a semilogy plot those factors are a constant vertical interval.
    They apply to interior bins (DC / Nyquist have fewer degrees of freedom).
    Adjacent bins are correlated (taper ENBW); do not treat them as independent
    when integrating the spectrum.
    """

    frequency: NDArray[np.floating]
    welch_spectrum: NDArray[np.floating]
    uncertainty_factors: tuple[float, float]
    welch_interval: tuple[NDArray[np.floating], NDArray[np.floating]]
    edof: float


def estimate_welch_edf(
    n_total: int,
    nperseg: int,
    noverlap: int,
    window: str | ArrayLike = "hann",
) -> tuple[float, int]:
    """Effective chi-squared degrees of freedom for a Welch PSD.

    Welch / Satterthwaite:

        ν = 2K / (1 + 2 Σ_{m=1}^{K-1} (1 - m/K) ρ(m S)^2)

    with hop S = nperseg - noverlap and ρ the lag-m overlap correlation of the
    taper. For Hann and 50% overlap this is exactly 36 K² / (19 K - 1).
    """
    hop = nperseg - noverlap
    if hop <= 0:
        raise ValueError("noverlap must be strictly less than nperseg.")

    K = int((n_total - noverlap) // hop)
    if K <= 0:
        raise ValueError("Signal is too short for the chosen segment configuration.")

    if isinstance(window, str):
        w = signal.get_window(window, nperseg)
    else:
        w = np.asarray(window, dtype=float)
        if w.shape != (nperseg,):
            raise ValueError("Custom window must have length nperseg.")

    window_energy = np.sum(w**2)
    if window_energy <= 0:
        raise ValueError("Window energy must be positive.")

    rho_sum = 0.0
    for m in range(1, K):
        shift_samples = m * hop
        if shift_samples >= nperseg:
            break
        rho_m = np.sum(w[shift_samples:] * w[:-shift_samples]) / window_energy
        rho_sum += (1.0 - m / K) * (rho_m**2)

    edf = (2.0 * K) / (1.0 + 2.0 * rho_sum)
    return float(edf), K


def _as_real_1d(x: ArrayLike) -> NDArray[np.floating]:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("signal must be a 1-D array of real values.")
    if np.iscomplexobj(arr):
        raise ValueError("signal must be real; got complex values.")
    if arr.size < 2:
        raise ValueError("signal must contain at least 2 samples.")
    if not np.isfinite(arr).all():
        raise ValueError("signal contains NaN or inf.")
    return arr.astype(float, copy=False)


def _chi2_scale_factors(edof: float, confidence_level: float) -> tuple[float, float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1), e.g. 0.95.")
    alpha = 1.0 - confidence_level
    lower_chi = stats.chi2.ppf(alpha / 2.0, df=edof)
    upper_chi = stats.chi2.ppf(1.0 - alpha / 2.0, df=edof)
    scale_lower = edof / upper_chi
    scale_upper = edof / lower_chi
    return float(scale_lower), float(scale_upper)


def _gaussianity_metrics(x: NDArray[np.floating]) -> dict:
    mu, sigma = stats.norm.fit(x)
    skewness = float(stats.skew(x))
    ex_kurtosis = float(stats.kurtosis(x))
    agostino_stat, agostino_p = stats.normaltest(x)
    skew_ok = abs(skewness) <= 1.0
    kurt_ok = abs(ex_kurtosis) <= 2.0
    return {
        "n": int(x.size),
        "mu": float(mu),
        "sigma": float(sigma),
        "skewness": skewness,
        "excess_kurtosis": ex_kurtosis,
        "agostino_stat": float(agostino_stat),
        "agostino_p": float(agostino_p),
        "practically_normal": bool(skew_ok and kurt_ok),
    }


def _acf_one_sided(x: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    centered = x - np.mean(x)
    acf = signal.correlate(centered, centered, mode="full", method="fft")
    acf = acf[x.size - 1 :]
    if acf[0] == 0:
        raise ValueError("Signal has zero variance; autocorrelation is undefined.")
    normalized = acf / acf[0]
    envelope = np.abs(signal.hilbert(normalized))
    return normalized, envelope


def _acf_decay_time(
    envelope: NDArray[np.floating], sample_rate: float, threshold: float = 0.2
) -> float:
    below = np.flatnonzero(envelope < threshold)
    if below.size == 0:
        return (envelope.size - 1) / sample_rate
    return float(below[0] / sample_rate)


def _stationarity_windows(
    x: NDArray[np.floating], nperseg: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    n_win = x.size // nperseg
    if n_win < 2:
        raise ValueError("Need at least 2 non-overlapping segments for a stationarity check.")
    windows = x[: n_win * nperseg].reshape(n_win, nperseg)
    return windows.mean(axis=1), windows.std(axis=1, ddof=0)


def _plot_gaussianity(x: NDArray[np.floating], metrics: dict) -> None:
    import matplotlib.pyplot as plt

    mu, sigma = metrics["mu"], metrics["sigma"]
    shape_results = (
        f"Distribution metrics:\n"
        f"Sample size: {metrics['n']}\n"
        f"Skewness: {metrics['skewness']:.4f}\n"
        f"Excess kurtosis: {metrics['excess_kurtosis']:.4f}\n\n"
        f"Practical verdict: "
        f"{'near-Gaussian' if metrics['practically_normal'] else 'shape not near-Gaussian'}"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.hist(
        x, bins=25, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Empirical"
    )
    grid = np.linspace(np.min(x), np.max(x), 200)
    ax1.plot(
        grid,
        stats.norm.pdf(grid, mu, sigma),
        "r-",
        linewidth=2,
        label=f"Fitted Gaussian\n($\\mu$={mu:.3g}, $\\sigma$={sigma:.3g})",
    )
    ax1.text(
        0.05,
        0.55,
        shape_results,
        transform=ax1.transAxes,
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5"),
    )
    ax1.set_title("Empirical distribution vs. Gaussian")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    stats.probplot(x, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("skyblue")
    ax2.get_lines()[0].set_markeredgecolor("black")
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[1].set_color("red")
    ax2.get_lines()[1].set_linewidth(2)
    ax2.set_title("Normal Q-Q plot")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def _plot_acf(
    lags_s: NDArray[np.floating],
    acf: NDArray[np.floating],
    envelope: NDArray[np.floating],
    tau_s: float,
    t_seg_s: float,
) -> None:
    import matplotlib.pyplot as plt

    bartlett = 1.96 / np.sqrt(acf.size)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharey=True)
    xmax_long = float(lags_s[-1])
    xmax_short = min(xmax_long, max(60.0, 1.2 * t_seg_s, 4.0 * tau_s))
    for ax, xmax in zip(axes, (xmax_long, xmax_short)):
        ax.plot(lags_s, acf, label="ACF", alpha=0.5)
        ax.plot(lags_s, envelope, label="Hilbert envelope", color="red", linewidth=2)
        ax.axhline(bartlett, color="gray", linestyle=":", linewidth=1, label="Bartlett ~95%")
        ax.axhline(-bartlett, color="gray", linestyle=":", linewidth=1)
        ax.axvline(tau_s, color="black", linestyle="--", linewidth=1, label=f"$\\tau_{{acf}}$={tau_s:.2g}s")
        ax.axvline(t_seg_s, color="tab:green", linestyle="--", linewidth=1, label=f"$T_{{seg}}$={t_seg_s:.2g}s")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_title("Autocorrelation (long / short lag)")
    fig.tight_layout()
    plt.show()


def _plot_stationarity(
    means: NDArray[np.floating], stds: NDArray[np.floating], t_seg_s: float
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(1, means.size + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(x, means, "o-", color="tab:blue", label="Mean")
    ax2.plot(x, stds, "s--", color="tab:orange", label="Std")
    ax1.set_xlabel("Non-overlapping window number")
    ax1.set_ylabel("Mean", color="tab:blue")
    ax2.set_ylabel("Standard deviation", color="tab:orange")
    ax1.set_title(f"Quasi-stationarity check ($T_{{seg}}$={t_seg_s:.3g} s)")
    fig.tight_layout()
    plt.show()


def _plot_welch(
    frequency: NDArray[np.floating],
    psd: NDArray[np.floating],
    psd_lo: NDArray[np.floating],
    psd_hi: NDArray[np.floating],
    scale_lower: float,
    scale_upper: float,
    confidence_level: float,
    edof: float,
    n_segments: int,
    overlap: float,
    window_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.semilogy(frequency, psd, label="Welch PSD")
    ax.fill_between(
        frequency,
        np.clip(psd_lo, 1e-300, None),
        np.clip(psd_hi, 1e-300, None),
        alpha=0.25,
        label=f"{100 * confidence_level:.0f}% $\\chi^2$ interval",
    )
    # Constant-length legend bar on log-y (multiplicative interval).
    pos = max(1, psd.size // 4)
    x_bar = frequency[pos]
    y_bar = psd[pos]
    ax.errorbar(
        [x_bar],
        [y_bar],
        yerr=[[y_bar - y_bar * scale_lower], [y_bar * scale_upper - y_bar]],
        fmt="x",
        color="crimson",
        markersize=8,
        capsize=5,
        elinewidth=2,
        label=(
            f"{100 * confidence_level:.0f}% $\\chi^2$, {n_segments} segments, "
            f"{100 * overlap:.0f}% overlap, {window_label}, $\\nu$={edof:.1f}"
        ),
    )
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("PSD")
    ax.legend()
    fig.tight_layout()
    plt.show()


def welch_spectrum_with_checks_and_uncertainties(
    signal_values: ArrayLike,
    sample_rate: float,
    window: str | ArrayLike = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    confidence_level: float = 0.95,
    verbose: bool = True,
    plot: bool = False,
    detrend: str | None = "constant",
    scaling: str = "density",
) -> WelchSpectrumResult:
    """Welch PSD of a real series, with checks and chi-squared interval.

    Always runs: (i) large-N Gaussianity (skew / excess kurtosis / Q-Q, not a
    formal normality test), (ii) autocorrelation time vs segment length,
    (iii) mean/std stationarity on non-overlapping segments, (iv) Welch PSD
    with a two-sided χ² interval at ``confidence_level`` using the general
    overlap-taper EDF. Checks warn; they do not abort the estimate.

    Parameters
    ----------
    signal_values :
        1-D real array.
    sample_rate :
        Sampling frequency $f_s$ in Hz.
    window, nperseg, noverlap, detrend, scaling :
        Passed to ``scipy.signal.welch`` (``nperseg`` defaults to 256 as in
        SciPy; ``noverlap`` defaults to 50% of ``nperseg``).
    confidence_level :
        Two-sided level for the χ² interval, e.g. 0.95.
    verbose, plot :
        Print summaries and/or show the same diagnostic figures as the notebook.

    Returns
    -------
    WelchSpectrumResult
        ``frequency, welch_spectrum, (scale_lo, scale_hi), (psd_lo, psd_hi), edof``.
        Relative χ² factors are independent of ``scaling`` (density vs spectrum).
    """
    x = _as_real_1d(signal_values)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    if nperseg is None:
        nperseg = 256
    nperseg = int(nperseg)
    if nperseg < 2 or nperseg > x.size:
        raise ValueError("nperseg must satisfy 2 <= nperseg <= len(signal).")

    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg.")

    t_seg_s = nperseg / sample_rate
    overlap_frac = noverlap / nperseg
    window_label = window if isinstance(window, str) else "custom"

    edof, n_segments = estimate_welch_edf(x.size, nperseg, noverlap, window=window)
    scale_lo, scale_hi = _chi2_scale_factors(edof, confidence_level)

    # --- i) Gaussianity (large-N practical check) ---
    gauss = _gaussianity_metrics(x)
    if verbose:
        print("=== Gaussianity (large-N practical check) ===")
        print(
            "A formal test (D'Agostino K² / KS) almost always rejects on long "
            "real records; use skew, excess kurtosis, and a Q-Q plot."
        )
        print(f"N = {gauss['n']}, mean = {gauss['mu']:.6g}, std = {gauss['sigma']:.6g}")
        print(
            f"D'Agostino K² statistic = {gauss['agostino_stat']:.4f}, "
            f"p = {gauss['agostino_p']:.4g} "
            f"({'reject Gaussian at 5%' if gauss['agostino_p'] <= 0.05 else 'do not reject at 5%'})"
        )
        print(f"Skewness = {gauss['skewness']:.4f} (practical |skew| <= 1)")
        print(f"Excess kurtosis = {gauss['excess_kurtosis']:.4f} (practical |excess kurtosis| <= 2)")
        if gauss["practically_normal"]:
            print("Verdict: near-Gaussian marginally. Necessary, not sufficient for a Gaussian process.")
        else:
            print(
                "WARNING: marginal shape is not near-Gaussian; "
                "χ² periodogram intervals may be a poor approximation."
            )
        print()
    if plot:
        _plot_gaussianity(x, gauss)

    # --- ii) Autocorrelation time vs segment length ---
    acf, envelope = _acf_one_sided(x)
    lags_s = np.arange(acf.size) / sample_rate
    tau_s = _acf_decay_time(envelope, sample_rate)
    acf_ok = t_seg_s >= 3.0 * tau_s
    if verbose:
        print("=== Autocorrelation time vs segment length ===")
        print(
            "Envelope is the Hilbert envelope of the one-sided ACF (visual decay, "
            "not a theorem). τ_acf = first lag where the envelope drops below 0.2."
        )
        print(f"τ_acf ≈ {tau_s:.4g} s, T_seg = {t_seg_s:.4g} s, T_seg / τ_acf = {t_seg_s / tau_s:.3g}")
        if acf_ok:
            print("Verdict: segment length is comfortably above the ACF decay time.")
        else:
            print(
                "WARNING: T_seg is not >> τ_acf; overlapping Welch segments may "
                "not be effectively independent, so ν is optimistic."
            )
        print()
    if plot:
        _plot_acf(lags_s, acf, envelope, tau_s, t_seg_s)

    # --- iii) Stationarity ---
    means, stds = _stationarity_windows(x, nperseg)
    typical_std = float(np.mean(stds))
    mean_drift = float(np.ptp(means) / typical_std) if typical_std > 0 else 0.0
    std_spread = float(np.ptp(stds) / typical_std) if typical_std > 0 else 0.0
    stat_ok = mean_drift <= 0.5 and std_spread <= 0.3
    if verbose:
        print("=== Stationarity (non-overlapping segments of length nperseg) ===")
        print(
            "Mean drift is (max mean − min mean) / typical window std — use this, "
            "not the relative change of a large DC offset (e.g. gravity)."
        )
        print(f"{means.size} windows of {t_seg_s:.4g} s")
        print(f"Window means: min={means.min():.6g}, max={means.max():.6g}, drift/σ = {mean_drift:.3g}")
        print(f"Window stds:  min={stds.min():.6g}, max={stds.max():.6g}, range/mean = {std_spread:.3g}")
        if stat_ok:
            print("Verdict: no large trend in mean (in σ units) or in variance.")
        else:
            print("WARNING: mean and/or variance look non-stationary across segments.")
        print()
    if plot:
        _plot_stationarity(means, stds, t_seg_s)

    # --- iv) Welch + χ² interval ---
    frequency, psd = signal.welch(
        x,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=detrend,
        return_onesided=True,
        scaling=scaling,
        average="mean",
    )
    psd_lo = psd * scale_lo
    psd_hi = psd * scale_hi
    if verbose:
        print("=== Welch estimate ===")
        print(
            f"window={window_label}, nperseg={nperseg}, noverlap={noverlap} "
            f"({100 * overlap_frac:.1f}%), K={n_segments} segments"
        )
        print(f"EDF ν = {edof:.4g}  (independent segments would give 2K = {2 * n_segments})")
        print(
            f"{100 * confidence_level:.0f}% multiplicative interval: "
            f"[{scale_lo:.4g}, {scale_hi:.4g}] × PSD"
        )
        print(
            "Shaded band / factors apply to interior bins. DC and Nyquist have "
            "fewer dof. Adjacent bins are correlated (taper ENBW)."
        )
        print()
    if plot:
        _plot_welch(
            frequency,
            psd,
            psd_lo,
            psd_hi,
            scale_lo,
            scale_hi,
            confidence_level,
            edof,
            n_segments,
            overlap_frac,
            window_label,
        )

    return WelchSpectrumResult(
        frequency=frequency,
        welch_spectrum=psd,
        uncertainty_factors=(scale_lo, scale_hi),
        welch_interval=(psd_lo, psd_hi),
        edof=edof,
    )
