import numpy as np
import numpy.typing as npt
import math

from scipy.fft import next_fast_len


def generate_realization_from_PSD(psd: npt.NDArray, segment_length: int, n_overlap: int, total_number_segments: int, window=np.hanning, use_next_fftlength: bool = True, noise_level: float = 1.0, cut_half_windows=True, real=True, phase=None, phase_noise_level: float = math.pi / 16.0):
    """Generate a pseudo-random realization of a given PSD, using ifft,
    applies a random Gaussian scaling to each bin amplitude to enforce stochasticity,
    applies either random phases (if phase=None) or Gaussian phase spreading around `phase`,
    and uses a windowed segment method to avoid creating phase locking across the whole signal length.
    Arguments:
        - psd: the psd for which to generate a random realization
        - segment_length: the segment length used with ifft to generate each segment of data
        - n_overlap: the overlap between consecutive segments; if None, set to segment_length//2
        - total_number_segments: the number of segments to use when generating the signal; this
            together with segment_length and n_overlap will determine the generated signal length
        - window: the fft window to use, a callable window(N) or None for no window
        - use_next_fftlength: whether or not to use the next fast fft compatible segment length
        - noise_level: the std of the gaussian multiplicative factor applied to bin amplitudes for fft-ing
        - cut_half_windows: whether to remove a half window of signal at the start and end to avoid transients
        - real: whether to output a real signal, by taking the real part of the ifft
        - phase: if None use independent uniform phases on [0, 2 pi); otherwise use the provided phase
            plus Gaussian noise with std phase_noise_level. Unit: radians; should match shape of psd
        - phase_noise_level: std of the Gaussian phase spreading across segments (radians)
    Returns:
        - output: the produced signal
    """

    assert isinstance(psd, np.ndarray)
    assert psd.ndim == 1
    assert np.issubdtype(psd.dtype, np.floating) or np.issubdtype(psd.dtype, np.integer)
    assert np.all(psd >= 0.)
    assert isinstance(segment_length, int)
    assert segment_length > 0
    assert isinstance(n_overlap, int) or n_overlap is None
    if n_overlap is None:
        n_overlap = segment_length // 2
    assert n_overlap < segment_length
    assert isinstance(total_number_segments, int)
    assert total_number_segments > 0
    assert isinstance(use_next_fftlength, bool)
    assert isinstance(noise_level, float)
    assert noise_level >= 0
    assert isinstance(cut_half_windows, bool)
    assert isinstance(phase, np.ndarray) or phase is None
    if phase is not None:
        assert len(psd) == len(phase)
        assert phase.ndim == 1
        assert np.issubdtype(phase.dtype, np.floating) or np.issubdtype(phase.dtype, np.integer)
    assert isinstance(phase_noise_level, float)

    if use_next_fftlength:
        segment_length = next_fast_len(segment_length)

    if len(psd) > segment_length:
        print(f"WARNING: {len(psd)=}, but {segment_length=}; cutting the psd")
        psd = psd[:segment_length]
        if phase is not None:
            phase = phase[:segment_length]

    if len(psd) < segment_length:
        print(f"WARNING: {len(psd)=}, but {segment_length=}; extending the psd with mean value")
        psd_new = np.full((segment_length,), np.mean(psd))
        psd_new[:len(psd)] = psd
        psd = psd_new
        if phase is not None:
            phase_new = np.full((segment_length,), np.mean(phase))
            phase_new[:len(phase)] = phase
            phase = phase_new

    psd = psd * np.sqrt(segment_length) * 2.0 * (np.pi)**2

    if window is not None:
        window = window(segment_length)
    else:
        window = np.ones(segment_length)

    hop = segment_length - n_overlap
    length_output = segment_length + total_number_segments * hop
    output = np.full((length_output,), 0.0 + 0.0j)

    for crrt_window in range(total_number_segments + 1):
        crrt_start = crrt_window * hop
        crrt_end = segment_length + crrt_start

        crrt_noisy_spectrum = np.sqrt(np.copy(psd))
        crrt_noisy_spectrum = crrt_noisy_spectrum * np.random.normal(1, noise_level, (segment_length,))

        if phase is None:
            crrt_noisy_phase = np.random.uniform(0, 2.0 * np.pi, (segment_length,))
        else:
            crrt_noisy_phase = phase + np.random.normal(0.0, phase_noise_level, (segment_length,))
        crrt_noisy_phase_complex = np.exp(1j * crrt_noisy_phase)

        crrt_noisy_spectrum = crrt_noisy_spectrum * crrt_noisy_phase_complex

        crrt_noisy_signal = np.fft.ifft(crrt_noisy_spectrum)
        crrt_noisy_signal = crrt_noisy_signal * window

        output[crrt_start:crrt_end] += crrt_noisy_signal

    if cut_half_windows:
        output = output[segment_length // 2:-segment_length // 2]

    if real:
        output = np.real(output)

    return output
