import numpy as np
import scipy.signal as signal
from scipy.fft import next_fast_len


def estimate_bicoherence_edf(n_total, segment_length, n_overlap, window='hann', use_next_fftlength=True):
    """
    Estimates the effective number of independent segments (K) and the
    corresponding degrees of freedom (dof = 2K) for a bicoherence estimate.

    The overlap-correction formula used here is the third-order analogue of
    Welch's equivalent-dof formula (rho^3 instead of rho^2). It is *not* in
    Elgar and Sebert (1989), who used non-overlapping, untapered records and
    took dof = 2N. The same window-correlation construction appears in the
    second-order (PSD) case in e.g. Percival & Walden; the cube on rho for
    the bispectrum / bicoherence is the natural extension discussed in the
    overlapping-segment literature (see also Elgar and Guza, 1988, IEEE
    Trans. Acoust. Speech Signal Process. 36, 1667-1668).

        K = p / (1 + 2 * sum_{m=1}^{p-1} (1 - m/p) * rho(mS)^3)

    with p the (raw, non-independent) number of segments, S the hop size
    between consecutive segment starts, and rho(mS) the window overlap
    correlation coefficient at a shift of m*S samples:

        rho(mS) = sum_n w[n] w[n+mS] / sum_n w[n]^2

    Segment counting matches split_signal_into_segments in bicoherence.py:
        p = 1 + (n_total - segment_length) // hop
      (= (n_total - n_overlap) // hop), with segment_length taken *after*
      the optional next_fast_len bump (see use_next_fftlength below), exactly
      as split_signal_into_segments does.

    Parameters:
    -----------
    n_total : int
        Total length of the input signal (number of samples).
    segment_length : int
        Length of each segment used for the bispectrum FFTs.
    n_overlap : int
        Number of overlapping points between consecutive segments.
    window : str or array_like
        Desired window to use (e.g., 'hann', 'hamming', 'boxcar'). Built with
        the symmetric convention (fftbins=False), matching the windows used
        by compute_auto_bicoherence/compute_auto_biphase in bicoherence.py
        (e.g. np.hanning), not scipy's default periodic/fftbins=True window.
        If array_like, it must be of length `segment_length` and should be
        the exact same window array used in the bicoherence computation.
    use_next_fftlength : bool
        Whether segment_length should be bumped up to the next fast FFT
        length, mirroring the same option in split_signal_into_segments /
        compute_auto_bicoherence (bicoherence.py). Must be set consistently
        with the call to compute_auto_bicoherence, otherwise the effective
        segment length assumed here will not match the one actually used,
        and the K / dof estimate will be wrong.

    Returns:
    --------
    K : float
        The effective number of independent segments (equivalent dof modifier),
        as used e.g. in the Rayleigh/gamma bicoherence noise-floor formulas
        (mean(b^2) = 1/K, etc.) in the notebook's "Statistics" section.
    dof : float
        The effective degrees of freedom, dof = 2 * K.
    p : int
        The raw (non-independent) number of segments used.
    """
    if use_next_fftlength:
        segment_length = next_fast_len(segment_length)

    hop = segment_length - n_overlap
    if hop <= 0:
        raise ValueError("n_overlap must be strictly less than segment_length.")

    p = int(1 + (n_total - segment_length) // hop)
    if p <= 0:
        raise ValueError("Signal length n_total is too short for the chosen segment configuration.")

    if isinstance(window, str):
        # fftbins=False: symmetric window, matching e.g. np.hanning as used
        # by compute_auto_bicoherence/compute_auto_biphase in bicoherence.py
        w = signal.get_window(window, segment_length, fftbins=False)
    else:
        w = np.asarray(window)
        if len(w) != segment_length:
            raise ValueError("Custom window array must have length equal to segment_length.")

    window_energy = np.sum(w ** 2)
    rho_sum = 0.0

    for m in range(1, p):
        shift_samples = m * hop
        if shift_samples >= segment_length:
            break

        overlap_corr = np.sum(w[shift_samples:] * w[:-shift_samples])
        rho_m = overlap_corr / window_energy
        rho_sum += (1.0 - m / p) * (rho_m ** 3)

    K = p / (1.0 + 2.0 * rho_sum)
    dof = 2.0 * K

    return K, dof, p


def null_b2_threshold(K, p_false=0.05):
    """
    Approximate upper tail of b^2 under true b^2 = 0.

    For a Gaussian process, 2 K b^2 ~ chi^2_2, so P(b^2 > x) ~ exp(-K x).
    Returns x such that P(b^2 > x) = p_false (default 5%).
    """
    return -np.log(p_false) / K
