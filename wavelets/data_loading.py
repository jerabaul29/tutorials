from pathlib import Path

from dataclasses import dataclass
from typing import ClassVar

import datetime

import scipy.io

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tqdm

import compress_pickle as cpkl

import time
import os

import pytz

from icecream import ic
ic.configureOutput(prefix="", outputFunction=print)

# ------------------------------------------------------------------------------------------
print("***** Put the interpreter in UTC, to make sure no TZ issues")
os.environ["TZ"] = "UTC"
time.tzset()


@dataclass
class GeophoneMatData:
    skip: ClassVar[int] = 250  # we skip the very start and end of each binary file: ignore 250 points
    date: np.ndarray  # date in Matlab's datenum format (days since the year 0)
    datetimes: datetime.datetime  # the corresponding datetimes
    t: np.ndarray  # time in seconds
    G1: np.ndarray  # X gain 5
    G2: np.ndarray  # Y gain 5
    G3: np.ndarray  # Z gain 5
    G4: np.ndarray  # Z gain 50
    G5: np.ndarray  # Z gain 1000
    # G1_filtered: np.ndarray  # X gain 5
    # G2_filtered: np.ndarray  # Y gain 5
    # G3_filtered: np.ndarray  # Z gain 5
    # G4_filtered: np.ndarray  # Z gain 50
    G5_filtered: np.ndarray  # Z gain 1000


def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    day_fraction = datetime.timedelta(days=matlab_datenum % 1) - datetime.timedelta(days=366)
    return day + day_fraction


def sliding_filter_nsigma(np_array_in, nsigma=3.0, side_half_width=5, skip_around=4):
    """Perform a sliding filter, on points of indexes
    [idx-side_half_width; idx+side_half_width], to remove outliers. I.e.,
    the [idx] point gets removed if it is more than nsigma deviations away
    from the mean of the whole segment.

    np_array_in should have a shape (nbr_of_entries,).

    return the filtered array and the list of indexes where filtered out

    """

    np_array = np.copy(np_array_in)
    array_len = np_array.shape[0]

    list_filtered_indexes = []

    middle_point_index_start = side_half_width
    middle_point_index_end = array_len - side_half_width - 1

    for crrt_middle_index in tqdm.tqdm(range(middle_point_index_start, middle_point_index_end+1, 1)):
        crrt_left_included = crrt_middle_index - side_half_width - skip_around
        crrt_right_included = crrt_middle_index + side_half_width + skip_around
        crrt_array_data = \
            np.concatenate(
                [np_array_in[crrt_left_included:crrt_middle_index-skip_around],
                 np_array_in[crrt_middle_index+1+skip_around:crrt_right_included+1]]
            )
        mean = np.mean(crrt_array_data)
        std = np.std(crrt_array_data)
        if np.abs(np_array[crrt_middle_index] - mean) > nsigma * std:
            np_array[crrt_middle_index] = mean  # we play a bit with fire: simply do a mean interpolation
            list_filtered_indexes.append(crrt_middle_index)

    return np_array, list_filtered_indexes


def multi_filter_nsigma(np_array_in):
    filtered_once = sliding_filter_nsigma(np_array_in, nsigma=3.0, side_half_width=6, skip_around=5)[0]
    filtered_twice = sliding_filter_nsigma(filtered_once, nsigma=3.0, side_half_width=3, skip_around=1)[0]
    return filtered_twice


def load_preprocess_geophone_matfile(path_to_matfile: Path) -> GeophoneMatData:
    path_to_cpkl_data = path_to_matfile.with_suffix(".cpkl")

    mat = scipy.io.loadmat(path_to_matfile)

    """
    # Just a way to check what the metadata about this mat file is,
    # and the fields contained by the mat file itself.
    ic(mat.keys())
    ic(mat["__header__"])
    ic(mat["__version__"])
    ic(mat["__globals__"])
    ic(mat["Data"].dtype)
    """

    if not path_to_cpkl_data.is_file():
        extracted_data = GeophoneMatData(
            date=mat["Data"]["date"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip],
            datetimes=[matlab2datetime(timestamp) for timestamp in mat["Data"]["date"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip]],
            t=mat["Data"]["t"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip],
            G1=mat["Data"]["G1"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip],
            G2=mat["Data"]["G2"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip],
            G3=mat["Data"]["G3"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip],
            G4=mat["Data"]["G4"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip],
            G5=mat["Data"]["G5"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip],
            # G1_filtered=sliding_filter_nsigma(mat["Data"]["G1"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip].astype(np.float32), 3.0, 3)[0],
            # G2_filtered=sliding_filter_nsigma(mat["Data"]["G2"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip].astype(np.float32), 3.0, 3)[0],
            # G3_filtered=sliding_filter_nsigma(mat["Data"]["G3"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip].astype(np.float32), 3.0, 3)[0],
            # G4_filtered=sliding_filter_nsigma(mat["Data"]["G4"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip].astype(np.float32), 3.0, 3)[0],
            G5_filtered=multi_filter_nsigma(mat["Data"]["G5"][0][0][:, 0][GeophoneMatData.skip:-GeophoneMatData.skip].astype(np.float32)),
        )

        with open(path_to_cpkl_data, "bw") as fh:
            cpkl.dump(extracted_data, fh, compression="lzma", set_default_extension=False)
    else:
        with open(path_to_cpkl_data, "br") as fh:
            extracted_data = cpkl.load(fh, compression="lzma")

    return extracted_data


def show_Z_x1000(geophone_mat_data: GeophoneMatData):
    datetime_start = geophone_mat_data.datetimes[0].astimezone(pytz.UTC)

    def converter_mdate_to_totalsecs(crrt_mdate):
        return (mdates.num2date(crrt_mdate)-datetime_start).total_seconds()

    v_converter_mdate_to_totalsecs = np.vectorize(converter_mdate_to_totalsecs)

    def converter_sec_to_mdate(crrt_sec):
        return mdates.date2num(datetime_start + datetime.timedelta(seconds=crrt_sec))

    v_converter_sec_to_mdate = np.vectorize(converter_sec_to_mdate)

    def mdate_to_sec(arr_mdate):
        if arr_mdate.size == 0:
            return []
        res = v_converter_mdate_to_totalsecs(arr_mdate)
        return res

    def sec_to_mdate(arr_sec):
        if arr_sec.size == 0:
            return []
        res = v_converter_sec_to_mdate(arr_sec)
        return res

    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(geophone_mat_data.datetimes, geophone_mat_data.G5, linestyle="-", color="k", label="G5 raw")
    ax.plot(geophone_mat_data.datetimes, geophone_mat_data.G5_filtered, linestyle="-", color="b", label="G5 filtered")

    ax.set_ylabel("ADC 12 bits reading")

    secax = ax.secondary_xaxis('top', functions=(mdate_to_sec, sec_to_mdate))
    # secax = ax.secondary_xaxis('top', functions=(sec_to_mdate, mdate_to_sec))
    secax.set_xlabel('time [s]')

    plt.legend()


# sampling freq: 1kHz
geophone_fs = 1e3
geophone_sampling_period = 1.0 / geophone_fs
