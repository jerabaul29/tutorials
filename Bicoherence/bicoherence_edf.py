import numpy as np
import scipy.signal as signal
import polars as pl


def estimate_bicoherence_edf(n_total, segment_length, n_overlap, window='hann'):
    """
    Estimates the effective number of independent segments (K) and the
    corresponding degrees of freedom (dof = 2K) for a bicoherence estimate,
    following the notation used in the Bicoherence.ipynb "Statistics" section
    (Elgar and Sebert, 1989):

        K = p / (1 + 2 * sum_{m=1}^{p-1} (1 - m/p) * rho(mS)^3)

    with p the (raw, non-independent) number of segments, S the hop size
    between consecutive segment starts, and rho(mS) the window overlap
    correlation coefficient at a shift of m*S samples:

        rho(mS) = sum_n w[n] w[n+mS] / sum_n w[n]^2

    Note the cube (not the square, as in the analogous Welch PSD dof formula)
    on rho: bicoherence is a third-order spectral quantity, so the
    correlation between segments enters to the third power instead of the
    second.

    Parameters:
    -----------
    n_total : int
        Total length of the input signal (number of samples).
    segment_length : int
        Length of each segment used for the bispectrum FFTs.
    n_overlap : int
        Number of overlapping points between consecutive segments.
    window : str or array_like
        Desired window to use (e.g., 'hann', 'hamming', 'boxcar').
        If array_like, it must be of length `segment_length`.

    Returns:
    --------
    K : float
        The effective number of independent segments (equivalent dof modifier),
        as used e.g. in the Rayleigh/gamma bicoherence noise-floor formulas
        (mean(b) = sqrt(pi/(4K)), etc.) in the notebook's "Statistics" section.
    dof : float
        The effective degrees of freedom, dof = 2 * K.
    p : int
        The raw (non-independent) number of segments used.
    """
    # 1. Calculate the shift (hop size) and (raw) number of segments
    hop = segment_length - n_overlap
    if hop <= 0:
        raise ValueError("n_overlap must be strictly less than segment_length.")

    p = int((n_total - n_overlap) // hop)
    if p <= 0:
        raise ValueError("Signal length n_total is too short for the chosen segment configuration.")

    # 2. Generate or extract the window function
    if isinstance(window, str):
        w = signal.get_window(window, segment_length)
    else:
        w = np.asarray(window)
        if len(w) != segment_length:
            raise ValueError("Custom window array must have length equal to segment_length.")

    # 3. Compute the correlation sum representing the overlap correlations
    window_energy = np.sum(w ** 2)
    rho_sum = 0.0

    # Iterate through all possible segment shifts
    for m in range(1, p):
        shift_samples = m * hop
        if shift_samples >= segment_length:
            break  # no more overlap between segments spaced this far apart

        # overlapping region between window and its shifted counterpart
        overlap_corr = np.sum(w[shift_samples:] * w[:-shift_samples])
        rho_m = overlap_corr / window_energy

        # weighted contribution of this lag, cubed (third-order quantity)
        rho_sum += (1.0 - m / p) * (rho_m ** 3)

    # 4. Effective number of independent segments K, and dof = 2K
    K = p / (1.0 + 2.0 * rho_sum)
    dof = 2.0 * K

    return K, dof, p
