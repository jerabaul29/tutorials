import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import math
from scipy.fft import next_fast_len
from scipy.stats import norm

from bicoherence import split_signal_into_segments, compute_auto_bispectrum


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


def compute_bicoherence_with_confidence(
    signal_data: npt.NDArray,
    sample_frequency: float,
    segment_length: int,
    n_overlap: int,
    use_next_fftlength: bool = True,
    window=np.hanning,
    f1_range=None,
    f2_range=None,
    confidence: float = 0.95,
    bicoherence_threshold: float = 0.4,
    verbose: bool = True,
):
    """
    Compute the auto-bicoherence b^2 and auto-biphase of a signal, together
    with their approximate confidence intervals at a given confidence level,
    following Elgar and Sebert (1989) (see the notebook's "Statistics"
    section for the formulas used).

    This function does everything end to end: it takes the same signal /
    segmentation / windowing arguments as compute_auto_bicoherence and
    compute_auto_biphase (bicoherence.py), computes the point estimates
    (b^2 and biphase, Kim-Powers `square_norm` convention), estimates the
    effective number of independent segments K and degrees of freedom
    (via estimate_bicoherence_edf), and combines all of this into:

        i)   a boolean map, True where b^2 is not significantly different
             from 0 at the given confidence (i.e. the null b^2=0 cannot be
             rejected there),
        ii)  a pair of maps (lower, upper) for b^2: both 0 where (i) is
             True, otherwise the confidence bounds on b^2,
        iii) a pair of maps (lower, upper) for the biphase (rad): both NaN
             where (i) is True (the biphase is meaningless when there is no
             significant coupling), otherwise the confidence bounds. In
             addition, and independently of (i), the biphase is also hidden
             (set to NaN) wherever the point estimate b^2 <= bicoherence_threshold
             (a plain display threshold, not a statistical test), since the
             biphase is only meaningful where b^2 is reasonably high, and
             wherever the biphase confidence interval half-width is >= pi,
             i.e. it wraps all the way around the circle and is therefore
             uninformative.

    Uncertainty formulas used (Elgar and Sebert 1989, their eqs. 4-5, 8-9;
    see also the notebook's "Statistics" section) are semi-empirical
    Gaussian / bias-variance approximations, accurate for moderate to large
    dof; they are used here as a "plug-in" approximation, substituting the
    (bias-corrected) point estimate for the unknown true b^2:

        bias-corrected estimate:  b2_c = clip(b2_hat - (2/dof) (1-b2_hat)^2, 0, 1)
        Var[b2_hat]            =  (4 b2_c / dof) (1 - b2_c)^3
        Var[biphase_hat]       =  (1/dof) (1/b2_c - 1)

    Confidence bounds are then b2_hat +/- z * sqrt(Var[b2_hat]) (clipped to
    [0, 1]) and biphase_hat +/- z * sqrt(Var[biphase_hat]) (wrapped to
    [-pi, pi]), with z the two-sided normal quantile for `confidence`.

    Arguments:
        - signal_data: the input signal on which to compute the bicoherence / biphase
        - sample_frequency: the sample frequency of the signal
        - segment_length: the length of individual segments on which to take the FFTs
        - n_overlap: the number of samples of overlap between consecutive segments
        - use_next_fftlength: whether or not to use the next length for which FFT is fast, default True
        - window: the FFT windowing algorithm to use; None is no windowing; defaults to np.hanning
        - f1_range: the range of frequencies for f1; None is as wide as possible
        - f2_range: same as f1 but for f2
        - confidence: the confidence level to use throughout (default 0.95, i.e. 95%)
        - bicoherence_threshold: a plain display threshold on the b^2 point
            estimate (default 0.4); the biphase is hidden (NaN) wherever
            bicoherence_hat <= bicoherence_threshold, in addition to being
            hidden wherever bicoherence_likely_zero is True. Unlike
            `confidence`, this is not a statistical test, just a convenience
            cutoff to only show the biphase where b^2 is reasonably high
            (see also compute_auto_biphase's hide_low_bicoherence_threshold)
        - verbose: if True (default), print a short summary while running:
            number of raw segments, effective number of independent segments,
            dof, and the null b^2 threshold used for the "likely zero" test

    Returns:
        - frequencies_1, frequencies_2: the frequency axes
        - bicoherence_hat: the b^2 point estimate (Kim-Powers `square_norm`)
        - biphase_hat: the biphase point estimate (rad), `mean_then_angle` method
        - bicoherence_likely_zero: bool array, True where b^2 is not
            significantly different from 0 at the given confidence
        - (bicoherence_lower, bicoherence_upper): b^2 confidence bounds; 0
            wherever bicoherence_likely_zero is True
        - (biphase_lower, biphase_upper): biphase confidence bounds (rad);
            NaN wherever bicoherence_likely_zero is True, wherever
            bicoherence_hat <= bicoherence_threshold, or wherever the
            biphase confidence interval half-width is >= pi (wraps around)
    """
    assert isinstance(signal_data, np.ndarray)
    assert signal_data.ndim == 1
    assert np.issubdtype(signal_data.dtype, np.floating) or np.issubdtype(signal_data.dtype, np.integer)

    assert isinstance(sample_frequency, float)
    assert sample_frequency > 0.0

    assert isinstance(segment_length, int)
    assert isinstance(n_overlap, int)
    assert isinstance(use_next_fftlength, bool)

    assert 0.0 < confidence < 1.0
    assert 0.0 <= bicoherence_threshold <= 1.0

    p_false = 1.0 - confidence

    # actual segment length used by split_signal_into_segments, needed to build
    # a matching window array for estimate_bicoherence_edf
    used_segment_length = next_fast_len(segment_length) if use_next_fftlength else segment_length
    if window is not None:
        window_array = window(used_segment_length)
    else:
        window_array = np.ones(used_segment_length)

    K, dof, p = estimate_bicoherence_edf(
        len(signal_data), segment_length, n_overlap,
        window=window_array, use_next_fftlength=use_next_fftlength,
    )

    array_of_signals = split_signal_into_segments(signal_data, segment_length, n_overlap, use_next_fftlength)
    n_segments = array_of_signals.shape[0]
    assert n_segments == p

    list_Ff1 = []
    list_Ff2 = []
    list_Ff3 = []
    for crrt_segment_index in range(n_segments):
        crrt_segment = array_of_signals[crrt_segment_index, :]
        frequencies_1, frequencies_2, (Ff1, Ff2, Ff3) = compute_auto_bispectrum(
            crrt_segment, sample_frequency, window, f1_range, f2_range, output="list",
        )
        list_Ff1.append(Ff1)
        list_Ff2.append(Ff2)
        list_Ff3.append(Ff3)

    array_Ff1 = np.array(list_Ff1)
    array_Ff2 = np.array(list_Ff2)
    array_Ff3 = np.array(list_Ff3)
    array_bispectrums = array_Ff1 * array_Ff2 * array_Ff3

    # Kim-Powers b^2, same formula as compute_auto_bicoherence(method="square_norm")
    sum_bispectrums = np.sum(array_bispectrums, axis=0)
    bicoherence_hat = np.abs(sum_bispectrums) ** 2 / (
        np.sum(np.abs(array_Ff1 * array_Ff2) ** 2, axis=0) * np.sum(np.abs(array_Ff3) ** 2, axis=0)
    )

    # biphase, mean_then_angle method (recommended, see compute_auto_biphase)
    biphase_hat = np.angle(sum_bispectrums, deg=False)

    # null b^2=0 threshold at the requested confidence, and the "likely zero" mask
    b2_threshold = null_b2_threshold(K, p_false=p_false)
    bicoherence_likely_zero = bicoherence_hat <= b2_threshold

    if verbose:
        print(f"compute_bicoherence_with_confidence: n_total={len(signal_data)}, segment_length={used_segment_length}, n_overlap={n_overlap}")
        print(f"    raw (non-independent) segments p={p}")
        print(f"    effective independent segments K={K:.2f}, dof=2K={dof:.2f}")
        print(f"    confidence={100 * confidence:.1f}%  ->  null b^2 threshold={b2_threshold:.4g}")
        print(f"    fraction of bins with b^2 likely 0 at this confidence: {np.mean(bicoherence_likely_zero):.1%}")

    # bias-corrected plug-in estimate of the true b^2, used in the variance formulas below
    bicoherence_corrected = np.clip(bicoherence_hat - (2.0 / dof) * (1.0 - bicoherence_hat) ** 2, 0.0, 1.0)

    z = norm.ppf(1.0 - p_false / 2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        var_b2 = (4.0 * bicoherence_corrected / dof) * (1.0 - bicoherence_corrected) ** 3
        var_biphase = (1.0 / dof) * (1.0 / bicoherence_corrected - 1.0)

    b2_margin = z * np.sqrt(var_b2)
    bicoherence_lower = np.clip(bicoherence_hat - b2_margin, 0.0, 1.0)
    bicoherence_upper = np.clip(bicoherence_hat + b2_margin, 0.0, 1.0)

    # var_biphase (and hence biphase_margin) is +inf wherever bicoherence_corrected == 0;
    # those bins are always inside bicoherence_likely_zero and get overwritten with NaN
    # below, so silence the resulting (harmless) inf/nan warnings here
    with np.errstate(invalid="ignore"):
        biphase_margin = z * np.sqrt(var_biphase)
        biphase_lower = np.angle(np.exp(1j * (biphase_hat - biphase_margin)), deg=False)
        biphase_upper = np.angle(np.exp(1j * (biphase_hat + biphase_margin)), deg=False)

    # a biphase confidence interval half-width >= pi wraps all the way around the
    # circle: the bounds would then be meaningless (any phase is within them), so hide
    biphase_wraps_around = biphase_margin >= math.pi

    # hide the biphase (but not b^2 itself) wherever: not statistically significant,
    # below the plain display threshold, or its confidence interval wraps around
    hide_biphase = bicoherence_likely_zero | (bicoherence_hat <= bicoherence_threshold) | biphase_wraps_around

    if verbose:
        print(f"    biphase hidden (b^2 likely 0, b^2 <= {bicoherence_threshold}, or CI wraps around): {np.mean(hide_biphase):.1%} of bins")

    bicoherence_lower = np.where(bicoherence_likely_zero, 0.0, bicoherence_lower)
    bicoherence_upper = np.where(bicoherence_likely_zero, 0.0, bicoherence_upper)
    biphase_lower = np.where(hide_biphase, np.nan, biphase_lower)
    biphase_upper = np.where(hide_biphase, np.nan, biphase_upper)

    return (
        frequencies_1, frequencies_2,
        bicoherence_hat, biphase_hat,
        bicoherence_likely_zero,
        (bicoherence_lower, bicoherence_upper),
        (biphase_lower, biphase_upper),
    )
