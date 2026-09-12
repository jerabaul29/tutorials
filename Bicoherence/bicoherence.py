import numpy as np
import numpy.typing as npt

import math

from scipy import fftpack

import matplotlib.pyplot as plt

# A verbose, possibly relatively slow, un-optimized implementation of real-signal bispectrum and bi-coherence.
# The goal is correctness and ease-of-reading, not speed!
# Still, we use FFTs in reasonable ways, and we do vectorized operations not loops! :)
# But we do not cache anything, including FFT coeffs when repeating the same FFT many times; we compute same coefficients possibly twice (symmetry not taken into account); etc.


def split_signal_into_segments(signal: npt.NDArray, segment_length: int, n_overlap: int, use_next_fftlength: bool =True) ->  npt.NDArray:
    """Split a signal into segments:
    Arguments:
        - signal: the signal to split, a 1d numpy array
        - segment_length: the (minimum if use_next_fftlength is True) segment length
        - overlap: the int number of overlap samples between 2 consecutive segments
        - use_next_fftlength: True if should use the next segment length that allows fast FFT, False to force the current segment length
    Returns:
        - array_of_signals: such that array_of_signals[0, :] is the first segment of length segment_length, etc"""

    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert isinstance(segment_length, int)
    assert isinstance(n_overlap, int)
    assert isinstance (use_next_fftlength, bool)
    
    if use_next_fftlength:
        segment_length = fftpack.next_fast_len(segment_length)

    start_segments = np.arange(0, len(signal)-segment_length, segment_length-n_overlap)
    nbr_segments = len(start_segments)
    array_of_signals = np.full([nbr_segments, segment_length], np.nan)

    for crrt_segment in range(nbr_segments):
        array_of_signals[crrt_segment, :] = signal[start_segments[crrt_segment]: start_segments[crrt_segment]+segment_length]
    
    return array_of_signals


array_of_signals = split_signal_into_segments(np.arange(0, 10, 1), segment_length=4, n_overlap=2, use_next_fftlength=True)
res = np.array(
    [[0., 1., 2., 3.],
     [2., 3., 4., 5.],
     [4., 5., 6., 7.]])
np.testing.assert_allclose(res, array_of_signals)
array_of_signals = split_signal_into_segments(np.arange(0, 12, 1), segment_length=7, n_overlap=5, use_next_fftlength=True)
res = np.array(
    [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
     [ 3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])
np.testing.assert_allclose(res, array_of_signals)


def find_first_greater_or_equal_index(value:float, array: npt.NDArray):
    """Find the index of the first element in the array that is greater than the specified value.
    Arguments:
        - array (numpy.ndarray): The array to search.
        - value (float): The value to compare against.
    Returns:
        - int: The index of the first element greater than the specified value, the last index."""

    bool_array = array > value
    if bool_array.any():
        argmax = np.argmax(bool_array)
        return argmax
    else:
        return len(array)-1


def get_f_range_index(f_range, frequencies: npt.NDArray):
    """Get the range of index in frequencies that cover f_range. We assume that frequencies is an array of rfft
    frequencies from rfftfreq. We always skip the 0-frequency and the last frequency.
    Arguments:
        - f_range: the range of frequencies to cover, either [f1, f2] with f1<f2 and both are floats, or None
        - frequencies: the array of frequencies from rfftfreq
    Returns:
        - (idx_min, idx_max): the range of indexes so that frequencies[idx_min:idx_max] cover f_range"""

    assert isinstance(frequencies, np.ndarray)
    assert frequencies.ndim == 1
    np.testing.assert_approx_equal(0.0, frequencies[0])
    
    if f_range is None:
        return (1, len(frequencies)-1)
    else:
        assert isinstance(f_range, list)
        assert len(f_range) == 2
        assert isinstance(f_range[0], float)
        assert isinstance(f_range[1], float)
        assert f_range[0] < f_range[1]
        idx_min = find_first_greater_or_equal_index(f_range[0], frequencies)
        if idx_min > 1:
            idx_min = idx_min - 1
        idx_min = max(1, idx_min)
        idx_max = find_first_greater_or_equal_index(f_range[1], frequencies)
        if idx_max < len(frequencies)-2:
            idx_max = idx_max + 1
        idx_max = min(idx_max, len(frequencies)-1)
        return (int(idx_min), int(idx_max))


assert get_f_range_index(None, np.array([0., 1., 2., 3., 4., 5.])) == (1, 5)
assert get_f_range_index([-1.0, 7.0], np.array([0., 1., 2., 3., 4., 5.])) == (1, 5)
assert get_f_range_index([2., 3.], np.array([0., 1., 2., 3., 4., 5.])) == (2, 4)

def compute_auto_bispectrum(signal: npt.NDArray, sample_frequency: float, window=np.hanning, f1_range=None, f2_range=None, output="product"):
    """Compute a single bispectrum for a real signal.
    Arguments:
        - signal: the signal on which to compute the bispectrum
        - sample_frequency: the sampling frequency
        - window: the windowing algorithm, a window function from
            https://numpy.org/doc/stable/reference/routines.window.html ,
            or None for no window
        - f1_range: the range [min_f1, max_f1] over which take the bisepctrum; if None use all frequencies
        - f2_range: the range [min_f1, max_f2] over which take the bispectrum; if None use all frequencies
        - output: whether the output should be the "product" F(f1).F(f2).F*(f1+f2), or the "list" [F(f1).F(f2), F*(f1+f2)].
    Returns:
        - frequencies: the frequencies of the bispectrum
        - bispectrum: the bispectrum as a np array (if using "product", or as a list of its components (if using "list")"""
    
    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert np.issubdtype(signal.dtype, np.floating) or np.issubdtype(signal.dtype, np.integer)
    assert isinstance(sample_frequency, float)
    assert sample_frequency > 0.0

    if window is not None:
        window = window(len(signal))
        signal = signal * window

    rfft = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1.0/sample_frequency)

    (min_index_f1, max_index_f1) = get_f_range_index(f1_range, frequencies)
    (min_index_f2, max_index_f2) = get_f_range_index(f2_range, frequencies)

    indexes_f1 = np.arange(min_index_f1, max_index_f1)
    frequencies_1 = frequencies[indexes_f1]
    rfft1 = rfft[indexes_f1]
    
    indexes_f2 = np.arange(min_index_f2, max_index_f2)
    frequencies_2 = frequencies[indexes_f2]
    rfft2 = rfft[indexes_f2]

    indexes_f2 = np.transpose(indexes_f2[np.newaxis])
    rfft2 = np.transpose(rfft2[np.newaxis])
    
    indexes_f3 = indexes_f1 + indexes_f2
    
    original_shape = indexes_f3.shape
    indexes_f3 = indexes_f3.flatten()
    mask_indexes_f3 = indexes_f3 >= len(rfft)
    indexes_f3[mask_indexes_f3] = 0
    Ff3 = rfft[indexes_f3]
    Ff3[mask_indexes_f3] = np.nan
    Ff3 = np.conjugate(Ff3)
    Ff3 = Ff3.reshape(original_shape)

    if output == "product":
        Ff1Ff2 = rfft1 * rfft2
        bispectrum_out = np.transpose(Ff1Ff2 * Ff3)
        return frequencies_1, frequencies_2, bispectrum_out

    elif output == "list":
        Ff1 = rfft1 * np.ones((1, len(rfft2)))
        Ff2 = rfft2 * np.ones((len(rfft1), 1))
        return frequencies_1, frequencies_2, [np.transpose(Ff1), np.transpose(Ff2), np.transpose(Ff3)]

    else:
        raise RuntimeError(f"Unknown {output =}")


f1, f2, bsp = compute_auto_bispectrum(signal=np.array([1, 2, 3, 2, 1, 1, 2, 3]), sample_frequency=1.0, window=np.hanning, f1_range=None, f2_range=None)
f1_res = np.array([0.125, 0.25 , 0.375])
f2_res = np.array([0.125, 0.25 , 0.375])
bsp_res = np.array([
    [12.96642115-15.52622105j,  3.38154335 -0.93617016j, -0.17457407 -0.58808224j],
    [ 3.38154335 -0.93617016j,  0.2048933  -0.62593076j, np.nan        ],
    [-0.17457407 -0.58808224j,         np.nan        , np.nan        ]
    ])
np.testing.assert_allclose(f1, f1_res)
np.testing.assert_allclose(f2, f2_res)
np.testing.assert_allclose(bsp, bsp_res)

f1, f2, bsp = compute_auto_bispectrum(signal=np.array([1, 2, 3, 2, 1, 1, 2, 3]), sample_frequency=1.0, window=np.hanning, f1_range=[0.1,0.3], f2_range=None)
f1_res = np.array([0.125, 0.25])
f2_res = np.array([0.125, 0.25 , 0.375])
bsp_res = np.array([
    [12.96642115-15.52622105j,  3.38154335 -0.93617016j, -0.17457407 -0.58808224j],
    [ 3.38154335 -0.93617016j,  0.2048933  -0.62593076j, np.nan        ],
    ])
np.testing.assert_allclose(f1, f1_res)
np.testing.assert_allclose(f2, f2_res)
np.testing.assert_allclose(bsp, bsp_res)

f1, f2, bsp = compute_auto_bispectrum(signal=np.array([1, 2, 3, 2, 1, 1, 2, 3]), sample_frequency=1.0, window=np.hanning, f1_range=None, f2_range=[0.1,0.3])
f1_res = np.array([0.125, 0.25 , 0.375])
f2_res = np.array([0.125, 0.25])
bsp_res = np.array([
    [12.96642115-15.52622105j,  3.38154335 -0.93617016j],
    [ 3.38154335 -0.93617016j,  0.2048933  -0.62593076j],
    [-0.17457407 -0.58808224j,         np.nan          ]
    ])
np.testing.assert_allclose(f1, f1_res)
np.testing.assert_allclose(f2, f2_res)
np.testing.assert_allclose(bsp, bsp_res)

# TODO: go through, comment, warning a bit messy and a bit too many options...

def compute_auto_biphase(signal: npt.NDArray, sample_frequency: float, segment_length: int, n_overlap: int, use_next_fftlength: bool=True, window=np.hanning, f1_range=None, f2_range=None, biphase_method="mean_then_angle", hide_low_bicoherence_threshold=0.5, low_bicoherence_threshold_method="square_norm", plot_distribution_peak=False, plot_distribution_peak_method="square_norm"):
    """Compute the auto-biphase. It is computed by averaging over the segments
    Arguments:
        - signal: the input signal on which to compute the biphase
        - sample_frequency: the sample frequency of the signal
        - segment_length: the length of individual segments on which to take the FFTs
        - n_overlap: the number of samples of overlap between consecutive segments
        - use_next_fftlength: whether or not to use the next length for which FFT is fast, default True
        - window: the FFT windowing algorithm to use; None is no windowing; defaults to np.hanning
        - f1_range: the range of frequencies for f1 in the biphase; None is as wide as possible
        - f2_range: same as f1 but for f2
        - biphase_method: either 1) "mean_then_angle" or 2) "angle_then_mean"; only 1) is recommended, see comments
        - hide_low_bicoherence_threshold: None if show all biphase, or a threshold for min bicoherence otherwise - bins with lower bicoherence are replaced with NaN
        - low_bicoherence_threshold_method: method to use to compute the bicoherence; should be a valid compute_auto_bicoherence method; only used for hide_low_bicoherence_threshold not None
        - plot_distribution_peak: if should plot the distribution of phases in a hist plot for the bin that has the maximum bicoherence; only used for angle_then_mean
        - plot_distribution_peak_method: method to use to compute the bicoherence; should be a valid compute_auto_bicoherence method; only used for angle_then_mean
    Returns:
        - frequencies_1: the frequencies 1 for the biphase
        - frequencies_2: the frequencies 2 for the biphase
        - auto_biphase_mean: the auto-biphase array; unit is rad
    """

    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert np.issubdtype(signal.dtype, np.floating) or np.issubdtype(signal.dtype, np.integer)
    
    assert isinstance(sample_frequency, float)
    assert sample_frequency > 0.0

    assert isinstance(segment_length, int)
    
    assert isinstance(n_overlap, int)
    
    assert isinstance(use_next_fftlength, bool)

    if biphase_method == "mean_then_angle":
        pass
    elif biphase_method == "angle_then_mean":
        print("")
        print("WARNING: angle_then_mean is not recommended!!!")
        print("WARNING: see the documentation and comment for 'compute_auto_biphase' method!!!")
        print("")

    if hide_low_bicoherence_threshold is not None:
        assert hide_low_bicoherence_threshold >= 0.0
        assert hide_low_bicoherence_threshold <= 1.0

    if hide_low_bicoherence_threshold is None:
        print("")
        print("WARNING: hide_low_bicoherence_theshold=None is not recommended!!!")
        print("This will plot the biphase everywhere, independently of bicoherence value")
        print("Remember that biphase is only reliable where bicoherence is high enough")
        print("")

    # split the signal in segments
    array_of_signals = split_signal_into_segments(signal, segment_length, n_overlap, use_next_fftlength)
    n_segments = array_of_signals.shape[0]

    # compute the auto bispectrum on each segment
    # put result in a new np.array
    list_bispectrums = []
    for crrt_segment_index in range(n_segments):
        crrt_segment = array_of_signals[crrt_segment_index, :]
        frequencies_1, frequencies_2, bispectrum_out = compute_auto_bispectrum(crrt_segment, sample_frequency, window, f1_range, f2_range, output="product")
        list_bispectrums.append(bispectrum_out)

    # averaging when there is some wrapping and not messing up conventions and signs and offsets is tricky...
    # 
    # first, there are 2 options: 1) average then angle, and 2) angle then average; option 1) average then angle should be more robust, because when taking the angle of
    #    some stochastic quantities, the wrapping around the unit circle will distribute values everywhere and may create angles that are almost 360 degrees off and mess
    #    up the mean
    #
    # second, there are different angle conventions - it is very easy to get a 90 degrees more or less, a factor + or - wrong, etc, if having subtle
    #    mismatch in conventions / choices between the different parts of the work... make sure to double check to your application and the exact conventions used!
    
    # option 1: average then take the angle; this can be done "brutally"; just be careful of conventions
    if biphase_method == "mean_then_angle":
        auto_biphase_mean = np.angle(np.mean(np.array(list_bispectrums), axis=0), deg=False)

    # option 2: angle then average; then we need to take only the relevant non wrapped angles, otherwise things get messy...
    #     note that close to the wrapping, this will always be messy :(
    #     this is dis-recommended, we keep this only for documentation purpose
    if biphase_method == "angle_then_mean":
        # look at the list_bispectrums
        list_biphases = [np.angle(crrt_bispectrum, deg=False) for crrt_bispectrum in list_bispectrums]
    
        auto_biphase_mean = np.mean(list_biphases, axis=0)
        
        nbins = min(len(auto_biphase_mean) / 100, 32)
    
        # show the difficulty with wrapping... there are wrapped values due to stochasticity, and this messes up / drags around the mean
        if plot_distribution_peak:
            f1, f2, bicoh = compute_auto_bicoherence(signal=signal, sample_frequency=sample_frequency, segment_length=segment_length, n_overlap=n_overlap, use_next_fftlength=use_next_fftlength, window=window, f1_range=f1_range, f2_range=f2_range, method=plot_distribution_peak_method)
            row, col = np.unravel_index(np.argmax(bicoh), bicoh.shape)
            list_values = [180.0 / math.pi * (np.angle(crrt_bispectrum[row, col], deg=False)) for crrt_bispectrum in list_bispectrums]
            # print(list_values)
    
            fig, ax = plt.subplots()
            ax.hist(list_values, bins=32, linewidth=0.5, edgecolor="white")
            # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 56), yticks=np.linspace(0, 56, 9))
            plt.show()

    if hide_low_bicoherence_threshold is not None:
        f1, f2, bicoh = compute_auto_bicoherence(signal=signal, sample_frequency=sample_frequency, segment_length=segment_length, n_overlap=n_overlap, use_next_fftlength=use_next_fftlength, window=window, f1_range=f1_range, f2_range=f2_range, method=low_bicoherence_threshold_method)

        auto_biphase_mean = np.where(bicoh > hide_low_bicoherence_threshold, auto_biphase_mean, np.nan)
    
    return frequencies_1, frequencies_2, auto_biphase_mean


def compute_auto_bicoherence(signal: npt.NDArray, sample_frequency: float, segment_length: int, n_overlap: int, use_next_fftlength: bool=True, window=np.hanning, f1_range=None, f2_range=None, method="square_norm"):
    """Compute the auto-bicoherence.
    Arguments:
        - signal: the input signal on which to compute the bicoherence
        - sample_frequency: the sample frequency of the signal
        - segment_length: the length of individual segments on which to take the FFTs
        - n_overlap: the number of samples of overlap between consecutive segments
        - use_next_fftlength: whether or not to use the next length for which FFT is fast, default True
        - window: the FFT windowing algorithm to use; None is no windowing; defaults to np.hanning
        - f1_range: the range of frequencies for f1 in the bicoherence; None is as wide as possible
        - f2_range: same as f1 but for f2
        - method: how to combine the bispectra on all segments into the bicoherence, possible values: "absolute_norm", "square_norm", "square_norm_indep"
    Returns:
        - frequencies_1: the frequencies 1 for the bicoherence
        - frequencies_2: the frequencies 2 for the bicoherence
        - auto_bicoherence: the auto-bicoherence array
    """
    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert np.issubdtype(signal.dtype, np.floating) or np.issubdtype(signal.dtype, np.integer)
    
    assert isinstance(sample_frequency, float)
    assert sample_frequency > 0.0

    assert isinstance(segment_length, int)
    
    assert isinstance(n_overlap, int)
    
    assert isinstance(use_next_fftlength, bool)

    assert method == "absolute_norm" or method == "square_norm" or method == "square_norm_indep"

    if method == "absolute_norm":
        output = "product"
    elif method == "square_norm":
        output = "list"
    elif method == "square_norm_indep":
        output = "list"
        print("WARNING!! this methoed is a buggy way to do it, that does not work - mathematically wrong!!")
        print("WARNING!! this method will not work, do not use it if not just for illustration purposes!!")
    else:
        raise RuntimeError("Unknown method!")
    
    # split the signal in segments
    array_of_signals = split_signal_into_segments(signal, segment_length, n_overlap, use_next_fftlength)
    n_segments = array_of_signals.shape[0]
    
    # compute the auto bispectrum on each segment
    # put result in a new np.array
    list_bispectrums = []
    for crrt_segment_index in range(n_segments):
        crrt_segment = array_of_signals[crrt_segment_index, :]
        frequencies_1, frequencies_2, bispectrum_out = compute_auto_bispectrum(crrt_segment, sample_frequency, window, f1_range, f2_range, output=output)
        list_bispectrums.append(bispectrum_out)
    
    # average into the bicoherence with the normalization chose
    if method == "absolute_norm":
        array_bispectrums = np.array(list_bispectrums)
        num = np.abs(np.sum(array_bispectrums, axis=0))
        denum = np.sum(np.abs(array_bispectrums), axis=0)
        auto_bicoherence = num / denum
    elif method == "square_norm":
        array_Ff1 = np.array([elem[0] for elem in list_bispectrums])
        array_Ff2 = np.array([elem[1] for elem in list_bispectrums])
        array_Ff3 = np.array([elem[2] for elem in list_bispectrums])
        num = np.power(np.abs(np.sum(array_Ff1 * array_Ff2 * array_Ff3, axis=0)), 2)
        denum = np.mean(np.power(np.abs(array_Ff1 * array_Ff2), 2), axis=0) * np.mean(np.power(np.abs(array_Ff3), 2), axis=0)  * (len(list_bispectrums)**2)  # same normalization as in wiki, except need a N**2... not sure why
        auto_bicoherence = np.sqrt(num / denum)
    elif method == "square_norm_indep":
        array_Ff1 = np.array([elem[0] for elem in list_bispectrums])
        array_Ff2 = np.array([elem[1] for elem in list_bispectrums])
        array_Ff3 = np.array([elem[2] for elem in list_bispectrums])
        num = np.power(np.abs(np.sum(array_Ff1 * array_Ff2 * array_Ff3, axis=0)), 2)
        denum = np.mean(np.power(np.abs(array_Ff1), 2), axis=0) * np.mean(np.power(np.abs(array_Ff2), 2), axis=0) * np.mean(np.power(np.abs(array_Ff3), 2), axis=0) * (len(list_bispectrums)**2)  # this actually does not work, for a good reason
        auto_bicoherence = np.sqrt(num / denum)
    else:
        raise RuntimeError("Unknown method!")

    assert np.all(np.logical_or(auto_bicoherence >= 0., np.isnan(auto_bicoherence)))
    # assert np.all(np.logical_or(auto_bicoherence <= 1., np.isnan(auto_bicoherence)))
    
    return frequencies_1, frequencies_2, auto_bicoherence
