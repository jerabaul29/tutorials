# %%
# if need to run this, use mamba (already on the machine), use the dev env; install extra packages as needed with mamba, only using the conda-forge channel (should already be the default)
# %%

from loguru import logger
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats
from scipy import integrate
import math

from scipy.optimize import fsolve

# bispectrum with the polycoherence 1-file script
from polycoherence import _plot_signal, polycoherence, plot_polycoherence

from bicoherence import compute_auto_bicoherence, compute_auto_biphase

import geopy.distance

# %%

pl_data = pl.read_csv("./VN100_with_Kalman_ned.csv")
pl_data = pl_data.with_columns(datetime = pl.col("timestamp").str.to_datetime())
pl_data = pl_data.with_columns(posix_time = pl.col("datetime").dt.epoch())

# %%

pl_data.columns

# %%

print(pl_data.tail(10).glimpse(max_items_per_column=1, return_as_string=True))

# %%

# list of IMUs in the dataset
list_IMUs = ["VN1005", "VN1002", "VN1007"]

# constant sample rate
sample_rate_hz = 10

# segments start at minutes 0 modulo 20
segment_minutes_modulo_start = 20

# each segment has a duration that is a power of 2 to make FFTs more effective
points_per_segment = 2**14
seconds_per_segment = points_per_segment / sample_rate_hz
print(f"{seconds_per_segment = }")
print(f"{seconds_per_segment/60.0 = }")

# find the indices for the start and end of the segments
start_segment = pl_data["posix_time"] // (1_000_000 / sample_rate_hz) % (segment_minutes_modulo_start*60*sample_rate_hz) == 0
start_segment_idx = np.where(start_segment)[0]
start_segment_idx = start_segment_idx[:-1]

# %%

# some quick looks at the data

# plot acc_d_m/s/s for all 3 IMUs over the whole time
# plot acc_d_m/s/s for all 3 IMUs over the whole time

crrt_segment_number = 3
acc_list = ["acc_d_m/s/s"]
# acc_list = ["acc_n_m/s/s", "acc_e_m/s/s"]
crrt_data = pl_data[start_segment_idx[crrt_segment_number]: start_segment_idx[crrt_segment_number]+points_per_segment]

plt.figure()
for crrt_acc in acc_list:
    for crrt_IMU in list_IMUs:
        plt.plot(crrt_data["datetime"], crrt_data[f"{crrt_IMU}_{crrt_acc}"], label=f"{crrt_acc}: {crrt_IMU}")
plt.ylabel("acc [m/s/s]")
plt.legend()
plt.show()

# %%

# reasonable FFT and similar params

fs = sample_rate_hz
window = "hann"
nperseg = int(points_per_segment / 2**3)
noverlap = int(nperseg / 2)
nfft = nperseg
detrend = "constant"
return_onesided = True
scaling = "density"
average = "mean"

# %%

# look at the Welch spectra
# for this, take a new Welch spectrum every 20 mins
# do a Welch spectrum based on these 20 mins of data

max_range = len(start_segment_idx)
# max_range = 10
# segments_list = list(range(max_range))

segments_list = [2, 16]

for crrt_segment_number in segments_list:

    plt.figure()

    # list_accs = ["acc_d_m/s/s", "acc_n_m/s/s", "acc_e_m/s/s"]
    list_accs = ["acc_d_m/s/s"]

    for crrt_acc in list_accs:

        for crrt_IMU in list_IMUs:
            # crrt_IMU = "VN1002"
            idx_segment_start = start_segment_idx[crrt_segment_number]
            idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment
            crrt_acc_d = pl_data[idx_segment_start: idx_segment_end][f"{crrt_IMU}_{crrt_acc}"]
            crrt_start = pl_data.row(idx_segment_start, named=True)["timestamp"][:-7]
            crrt_end = pl_data.row(idx_segment_end, named=True)["timestamp"][:-7]

            crrt_acc_d = crrt_acc_d.to_numpy()

            crrt_f, crrt_welch_psd_acc_d = signal.welch(crrt_acc_d, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, average=average)

            crrt_welch_psd_elev = crrt_welch_psd_acc_d * (2.0 * math.pi * crrt_f)**(-4)
            indf_min = 15
            indf_max = 60
            print(f"{crrt_f[indf_min]=}")
            print(f"{crrt_f[indf_max]=}")
            m0 = integrate.trapezoid(crrt_welch_psd_elev[indf_min:indf_max], crrt_f[indf_min:indf_max])
            swh = 4.0 * math.sqrt(m0)
            print(f"{swh=}")

            plt.plot(crrt_f, crrt_welch_psd_acc_d, label=f"{crrt_acc}_{crrt_IMU}")

    plt.axhline(6e-5, color="black", linewidth=2.0, linestyle="--", label="noise floor")

    plt.xlim([0.05, 0.6])
    plt.ylim([1e-5, 2e-2])

    plt.xlabel("f [Hz]")
    plt.ylabel("PSD$_a$ [(m/s$^2$)$^2$/hz]")
    # plt.title(f"Tempelfjorden {crrt_start} to {crrt_end}")
    plt.title(f"Tempelfjorden segment {crrt_segment_number}")
    plt.legend()

    plt.tight_layout()
    plt.semilogy()

    plt.show()

    print("")

    plt.savefig(f"figs/PSD_Tempelfjorden_2015_segment_{crrt_segment_number}.png")

pass

# %%

crrt_data.select(["datetime", "VN1002_acc_d_m/s/s"]).write_csv("segment_3.csv")
crrt_data.select(["datetime", "VN1005_acc_d_m/s/s"]).write_csv("segment_3_VN1005.csv")
crrt_data.select(["datetime", "VN1007_acc_d_m/s/s"]).write_csv("segment_3_VN1007.csv")

# %%

# look at the Welch spectra
# for this, take a new Welch spectrum every 20 mins
# do a Welch spectrum based on these 20 mins of data

max_range = len(start_segment_idx)
# max_range = 10
# segments_list = list(range(max_range))

segments_list = [2, 16]

for crrt_segment_number in segments_list:

    plt.figure()

    # list_accs = ["acc_d_m/s/s", "acc_n_m/s/s", "acc_e_m/s/s"]
    list_accs = ["acc_d_m/s/s"]

    for crrt_acc in list_accs:

        for crrt_IMU in list_IMUs:
            # crrt_IMU = "VN1002"
            idx_segment_start = start_segment_idx[crrt_segment_number]
            idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment
            crrt_acc_d = pl_data[idx_segment_start: idx_segment_end][f"{crrt_IMU}_{crrt_acc}"]
            crrt_start = pl_data.row(idx_segment_start, named=True)["timestamp"][:-7]
            crrt_end = pl_data.row(idx_segment_end, named=True)["timestamp"][:-7]

            crrt_acc_d = crrt_acc_d.to_numpy()
            
            crrt_f, crrt_welch_psd_acc_d = signal.welch(crrt_acc_d, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, average=average)

            plt.loglog(crrt_f, crrt_welch_psd_acc_d, label=f"{crrt_acc}_{crrt_IMU}")

    plt.axhline(6e-5, color="black", linewidth=2.0, linestyle="--", label="noise floor")

    plt.xlim([0.05, 0.6])
    plt.ylim([1e-5, 2e-2])

    plt.xlabel("f [Hz]")
    plt.ylabel("PSD$_a$ [(m/s$^2$)$^2$/hz]")
    # plt.title(f"Tempelfjorden {crrt_start} to {crrt_end}")
    plt.title(f"Tempelfjorden segment {crrt_segment_number}")
    plt.legend()

    plt.tight_layout()
    # plt.semilogy()

    plt.savefig(f"figs/PSD_Tempelfjorden_2015_segment_{crrt_segment_number}.png")

plt.show()

# %%

# show the different frequency component contributions?
# use filter to separate the main peak from the secondary peak, and plot the signals then
# NOTE: this is not as easy as just looking if something clear is present; this does not produce anything useful it seems?

filter_order = 5
swell_frequency_band = [0.05, 0.25]
harmonic_frequency_band = [0.25, 0.45]
btype = "bandpass"
analog = False
output = "sos"
fs = sample_rate_hz

sos_swell = signal.butter(filter_order, swell_frequency_band, btype=btype, analog=analog, output=output, fs=fs)
sos_harmonic = signal.butter(filter_order, harmonic_frequency_band, btype=btype, analog=analog, output=output, fs=fs)

# for each segment, show the filtered parts of the signal
# max_range = len(start_segment_idx)
max_range = 5

for crrt_segment_number in range(max_range):
    print(f"{crrt_segment_number = }")

    for crrt_IMU in list_IMUs:
        print(f"{crrt_IMU = }")

        dict_acc_filtered = {}

        list_accs = ["acc_d_m/s/s"]
        # list_accs = ["acc_d_m/s/s", "acc_n_m/s/s", "acc_e_m/s/s"]
        for crrt_acc in list_accs:
            print(f"{crrt_acc = }")

            idx_segment_start = start_segment_idx[crrt_segment_number]
            idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment
            crrt_acc_signal = np.array(pl_data[idx_segment_start: idx_segment_end][f"{crrt_IMU}_{crrt_acc}"])
            crrt_acc_signal = crrt_acc_signal - np.mean(crrt_acc_signal)

            crrt_start = pl_data.row(idx_segment_start, named=True)["timestamp"][:-7]
            crrt_end = pl_data.row(idx_segment_end, named=True)["timestamp"][:-7]

            crrt_acc_filtered_swell = signal.sosfiltfilt(sos_swell, crrt_acc_signal)
            crrt_acc_filtered_harmonic = signal.sosfiltfilt(sos_harmonic, crrt_acc_signal)

            if True:
                plt.figure()
                plt.plot(crrt_acc_signal, label="non filtered")
                plt.plot(crrt_acc_filtered_swell, label="filtered swell")
                plt.plot(crrt_acc_filtered_harmonic, label="filtered harmonic")
                plt.plot(crrt_acc_filtered_swell + crrt_acc_filtered_harmonic, label="filtered swell + harmonic")
                plt.title(crrt_acc)
                plt.legend()
                plt.show()

pass


# %%

# look at the presence of collisions

# NOTE: we cannot reliably observe collisions

# compute the residual:
# 5th order Butterworth filter frequency cutoff 0.5Hz (a bit higher than Lars, closer to open water)
filter_order = 5
cutoff_frequency_hz = [0.5]
btype = "lowpass"
analog = False
output = "sos"
fs = sample_rate_hz

sos = signal.butter(filter_order, cutoff_frequency_hz, btype=btype, analog=analog, output=output, fs=fs)

# for each segment, compute the difference between the filtered signal and the signal
# max_range = len(start_segment_idx)
max_range = 5

for crrt_segment_number in range(max_range):
    print(f"{crrt_segment_number = }")

    for crrt_IMU in list_IMUs:
        print(f"{crrt_IMU = }")

        dict_acc_filtered = {}

        for crrt_acc in ["acc_d_m/s/s", "acc_n_m/s/s", "acc_e_m/s/s"]:
            print(f"{crrt_acc = }")

            idx_segment_start = start_segment_idx[crrt_segment_number]
            idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment
            crrt_acc_signal = pl_data[idx_segment_start: idx_segment_end][f"{crrt_IMU}_{crrt_acc}"]

            crrt_start = pl_data.row(idx_segment_start, named=True)["timestamp"][:-7]
            crrt_end = pl_data.row(idx_segment_end, named=True)["timestamp"][:-7]

            crrt_acc_filtered = signal.sosfiltfilt(sos, crrt_acc_signal)
            crrt_acc_residual = crrt_acc_signal - crrt_acc_filtered

            if True:
                plt.figure()
                plt.plot(crrt_acc_signal, label="non filtered")
                plt.plot(crrt_acc_filtered, label="filtered")
                plt.title(crrt_acc)
                plt.legend()
                plt.show()

            dict_acc_filtered[crrt_acc] = {}
            dict_acc_filtered[crrt_acc]["raw"] = crrt_acc_signal
            dict_acc_filtered[crrt_acc]["filtered"] = crrt_acc_filtered
            dict_acc_filtered[crrt_acc]["residual"] = crrt_acc_residual

            is_normal_test = stats.normaltest(crrt_acc_residual)
            print(f"{is_normal_test}")

        if True:
            plt.figure()
            plt.scatter(dict_acc_filtered["acc_n_m/s/s"]["residual"], dict_acc_filtered["acc_e_m/s/s"]["residual"])
            plt.xlabel("acc_n_m/s/s")
            plt.ylabel("acc_e_m/s/s")

            plt.figure()
            residual_acc_horizontal = np.sqrt(dict_acc_filtered["acc_n_m/s/s"]["residual"]**2 + dict_acc_filtered["acc_e_m/s/s"]["residual"]**2)
            plt.scatter(residual_acc_horizontal, dict_acc_filtered["acc_d_m/s/s"]["residual"])
            plt.xlabel("acc_h_m/s/s")
            plt.ylabel("acc_d_m/s/s")

            plt.show()

pass

# %%

# NOTE: - relation between bispectrum and 3-point autocorrelation function
# NOTE: - why it makes sense to look at the bispectrum to try to identify non-linearity?
# NOTE: - see bispectrum and / vs. bicoherence: https://en.wikipedia.org/wiki/Bispectrum and https://en.wikipedia.org/wiki/Bicoherence
# NOTE: - see https://en.wikipedia.org/wiki/Bispectrum and https://en.wikipedia.org/wiki/Bicoherence

# for each segment, compute the bispectrum / related quantities
# list_segments = [2, 22]
# list_segments = range(0, 7, 1)
# list_segments = range(len(start_segment_idx))
list_segments = [2, 16]

for crrt_segment_number in list_segments:
    print("")
    print(f"{crrt_segment_number = }")

    # use_IMUs = list_IMUs
    # IMU 5 (most outside) never has the harmonic present, 2 and 7 have more or less the same: only look at 2
    use_IMUs = ["VN1002"]
    
    for crrt_IMU in use_IMUs:
        print(f"{crrt_IMU = }")

        dict_acc_filtered = {}

        # list_accs = ["acc_d_m/s/s", "acc_n_m/s/s", "acc_e_m/s/s"]
        list_accs = ["acc_d_m/s/s"]

        for crrt_acc in list_accs:
            print(f"{crrt_acc = }")

            idx_segment_start = start_segment_idx[crrt_segment_number]
            idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment
            crrt_acc_signal = pl_data[idx_segment_start: idx_segment_end][f"{crrt_IMU}_{crrt_acc}"]

            crrt_start = pl_data.row(idx_segment_start, named=True)["timestamp"][:-7]
            crrt_end = pl_data.row(idx_segment_end, named=True)["timestamp"][:-7]

            # bispectrum with polycoherence package
            if True:
                data = crrt_acc_signal
                fs = sample_rate_hz
                window = "hann"
                nperseg = int(points_per_segment / 2**5)
                noverlap = int(0.5 * nperseg)
                nfft = nperseg
                norm = 2
            
                freq1, freq2, bispec = polycoherence(data=data, fs=sample_rate_hz, norm=norm, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nperseg, flim1=(0.05, 0.5), flim2=(0.05, 0.5))
                df1 = freq1[1] - freq1[0]
                df2 = freq2[1] - freq2[0]
                freq1 = np.append(freq1, freq1[-1] + df1) - 0.5 * df1
                freq2 = np.append(freq2, freq2[-1] + df2) - 0.5 * df2

                length_i = np.shape(bispec)[0]
                length_j = np.shape(bispec)[1]

                max_bicoherence = np.max(np.abs(bispec))
                print(f"{max_bicoherence = }")

                for i in range(length_i):
                    for j in range(i):
                        bispec[i, j] = math.nan

                plt.figure()
                # plt.pcolormesh(freq2, freq1, np.abs(bispec), norm=matplotlib.colors.LogNorm())
                plt.pcolormesh(freq2, freq1, np.abs(bispec), vmin=0, vmax=0.7)
                # plt.pcolormesh(freq2, freq1, np.abs(bispec))
                plt.title(f"Tempelfjorden segment {crrt_segment_number}")
                plt.xlabel('f [Hz]')
                plt.ylabel('f [Hz]')
                cbar = plt.colorbar()
                cbar.set_label('bicoherence', fontsize=12)

                plt.savefig(f"figs/polycoherence_norm2_{crrt_segment_number}_{crrt_IMU}_accd.png")

                plt.show()

                for j in range(length_j):
                    for i in range(j):
                        f1 = freq1[i]
                        f2 = freq2[j]
                        bicoh = np.abs(bispec[i, j])
                        if bicoh > 0.5:
                            print(f"{f1 = }")
                            print(f"{f2 = }")
                            print(f"{f1 + f2 = }")
                            print(f"{bicoh = }")

pass

# %%

# bicoherence with my code

# for each segment, compute the bispectrum / related quantities
# list_segments = [2, 22]
# list_segments = range(0, 7, 1)
list_segments = [2, 16]
# list_segments = range(len(start_segment_idx))

for crrt_segment_number in list_segments:
    print(f"{crrt_segment_number = }")

    # use_IMUs = list_IMUs
    # IMU 5 (most outside) never has the harmonic present, 2 and 7 have more or less the same: only look at 2
    use_IMUs = ["VN1002"]
    
    for crrt_IMU in use_IMUs:
        print(f"{crrt_IMU = }")

        dict_acc_filtered = {}

        # list_accs = ["acc_d_m/s/s", "acc_n_m/s/s", "acc_e_m/s/s"]
        list_accs = ["acc_d_m/s/s"]

        for crrt_acc in list_accs:
            print(f"{crrt_acc = }")

            idx_segment_start = start_segment_idx[crrt_segment_number]
            idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment
            crrt_acc_signal = np.array(pl_data[idx_segment_start: idx_segment_end][f"{crrt_IMU}_{crrt_acc}"])
            crrt_acc_signal = crrt_acc_signal - np.mean(crrt_acc_signal)

            crrt_start = pl_data.row(idx_segment_start, named=True)["timestamp"][:-7]
            crrt_end = pl_data.row(idx_segment_end, named=True)["timestamp"][:-7]

            # bispectrum with my code
            if True:
                # method = "absolute_norm"
                # method = "square_norm"
                method = "square_norm_indep"

                f1_s3, f2_s3, ch_s3 = compute_auto_bicoherence(crrt_acc_signal, sample_frequency=float(sample_rate_hz), segment_length=nperseg, n_overlap=noverlap, f1_range=[0.05, 0.4], f2_range=[0.05, 0.4], method=method)

                cleaned_ch_s3 = np.transpose(ch_s3)

                for i in range(cleaned_ch_s3.shape[0]):
                    for j in range(cleaned_ch_s3.shape[1]):
                        if i > j:
                            cleaned_ch_s3[i,j] = math.nan

                plt.figure()
                plt.pcolormesh(f1_s3, f2_s3, cleaned_ch_s3, shading='nearest', vmin=0.0, vmax=0.9)
                cbar = plt.colorbar()
                cbar.set_label('bicoherence', fontsize=12)
                plt.xlabel("f [Hz]")
                plt.ylabel("f [Hz]")
                plt.show()

pass

# conclusion: same results as the polycoherence package :)

## %

# Look at the biphase

# for each segment, compute the biphase
# list_segments = [2, 22]
# list_segments = range(0, 7, 1)
list_segments = [2, 16]
# list_segments = range(len(start_segment_idx))

for crrt_segment_number in list_segments:
    print("T15")
    print(f"{crrt_segment_number = }")

    # use_IMUs = list_IMUs
    # IMU 5 (most outside) never has the harmonic present, 2 and 7 have more or less the same: only look at 2
    use_IMUs = ["VN1002"]
    
    for crrt_IMU in use_IMUs:
        print(f"{crrt_IMU = }")

        dict_acc_filtered = {}

        # list_accs = ["acc_d_m/s/s", "acc_n_m/s/s", "acc_e_m/s/s"]
        list_accs = ["acc_d_m/s/s"]

        for crrt_acc in list_accs:
            print(f"{crrt_acc = }")

            idx_segment_start = start_segment_idx[crrt_segment_number]
            idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment
            crrt_acc_signal = np.array(pl_data[idx_segment_start: idx_segment_end][f"{crrt_IMU}_{crrt_acc}"])
            crrt_acc_signal = crrt_acc_signal - np.mean(crrt_acc_signal)

            crrt_start = pl_data.row(idx_segment_start, named=True)["timestamp"][:-7]
            crrt_end = pl_data.row(idx_segment_end, named=True)["timestamp"][:-7]

            # biphase with my code
            if True:

                f1_s3, f2_s3, ph_s3 = compute_auto_biphase(crrt_acc_signal, sample_frequency=float(sample_rate_hz), segment_length=nperseg, n_overlap=noverlap, f1_range=[0.05, 0.4], f2_range=[0.05, 0.4], hide_low_bicoherence_threshold=0.4, biphase_method="mean_then_angle")

                cleaned_ph_s3 = np.transpose(ph_s3) * 180.0 / math.pi
                # we use opposite sign conventions in the "wave community" vs the "engineering community":
                # forward FFT in engineering (like numpy fft) is obtained by multiplying by e(-1j k . t / N), assuming a signal basis of e(1j omega . t)
                # for the waves, we use the convention for the signal basis e(1j [k . x - omega . t]) ; note the - vs + before the omega . t term
                # fix this conventions mismatch by switching the signs
                cleaned_ph_s3 = -cleaned_ph_s3

                for i in range(cleaned_ph_s3.shape[0]):
                    for j in range(cleaned_ph_s3.shape[1]):
                        if i > j:
                            cleaned_ph_s3[i,j] = math.nan

                        if i <= j:
                            if np.isfinite(cleaned_ph_s3[i, j]):
                                print(f"f1={float(f1_s3[i]):.2f}Hz; f2={float(f2_s3[j]):.2f}Hz | bp={float(cleaned_ph_s3[i,j]):.2f} deg")

                if np.isnan(cleaned_ph_s3).all():
                    mean_ph_s3 = math.nan
                else:
                    mean_ph_s3 = np.nanmean(cleaned_ph_s3)

                if np.isfinite(mean_ph_s3):
                    print(f"{mean_ph_s3=:.2f} [deg]")
                
                plt.figure()
                plt.pcolormesh(f1_s3, f2_s3, cleaned_ph_s3, shading='nearest', vmin=0, vmax=180)
                cbar = plt.colorbar()
                cbar.set_label('biphase [deg]', fontsize=12)
                plt.xlabel("f [Hz]")
                plt.ylabel("f [Hz]")
                plt.title(f"T15 segment {crrt_segment_number} IMU {crrt_IMU} biphase")
                plt.show()

pass

# %%

# cross spectrum analysis to vizualize the effective dispersion relation

# GPS positions:
# sensors are listed as follows
# VN1002 = ext 1 (wired) 78 22.604' N, 16 54.175' E
# Converted to decimal : 78.376733 16.902917
pos_vn1002 = [78.376733, 16.902917]
# VN1003 = ext 2 (wired) 78 22.063' N, 16 54.222' E
# Converted to decimal : 78.367717 16.9037
post_vn1003 = [78.367717, 16.9037]
# VN1004 = T5 (wireless) 78 22.590 N, 16 54.003' E
# Converted to decimal : 78.3765 16.90005
pos_vn1004 = [78.3765, 16.90005]
# VN1005 = T1 (wireless) 78 22.596 N, 16 54.044' E
# Converted to decimal : 78.3766 16.900733
pos_vn1005 = [78.3766, 16.900733]
# VN1006 = T3 (wireless) didn't work
# VN1007 = internal - gps position can be obtained from $GPRMC strings from data file
# From 150325_190000.00.GPS1:  7822.6038 N,01654.1880 E,
# Converted to decimal : 78.3767300 16.9031303
pos_vn1007 = [78.3767300, 16.9031303]

dist_7_2_meters = geopy.distance.geodesic(pos_vn1007, pos_vn1002).m
dist_5_2_meters = geopy.distance.geodesic(pos_vn1005, pos_vn1002).m
dist_5_7_meters = geopy.distance.geodesic(pos_vn1005, pos_vn1007).m

# list_segments = [2]
# list_segments = [2, 22]
# list_segments = [1, 2, 3, 4, 5, 7, 10, 15]
list_segments = range(0, 21, 1)
# list_segments = range(len(start_segment_idx))

# parameters for the matplotlib csd functions
NFFT = 2048
# NFFT = 4096
Fs = sample_rate_hz
detrend = "mean"
# noverlap = NFFT//2
noverlap = int(0.9 * NFFT)
window = np.hanning(NFFT)
# window = np.hamming(NFFT)

print(f"{points_per_segment=}")
print(f"{NFFT=}")
print(f"{noverlap=}")
print(f"{window=}")

list_valid_frequency_hz = []
list_valid_k = []
list_valid_frequency_hz_highnoise = []
list_valid_k_highnoise = []

for crrt_segment_number in list_segments:
    print("")
    print(f"{crrt_segment_number = }")

    acc_field = "acc_d_m/s/s"

    # reminder: ordering of the IMUs: out to in:
    # 5
    # 2
    # 7

    # nice fig 1
    if True:
        imu_out = "VN1005"  # the far one
        imu_in = "VN1007"  # the one furthest in
        # distance_imus = dist_5_7_meters
        distance_imus = 72.0

    # nice fig 2
    if False:
        imu_out = "VN1002"
        imu_in = "VN1007"
        # distance_imus = dist_7_2_meters
        distance_imus = 7.0

    if False:
        imu_out = "VN1005"
        imu_in = "VN1002"
        distance_imus = dist_5_2_meters

    print(f"{imu_in = }")
    print(f"{imu_out = }")
    print(f"{distance_imus = }")
    
    idx_segment_start = start_segment_idx[crrt_segment_number]
    idx_segment_end = start_segment_idx[crrt_segment_number]+points_per_segment

    acc_out = np.array(pl_data[idx_segment_start: idx_segment_end][f"{imu_out}_{crrt_acc}"])
    acc_in = np.array(pl_data[idx_segment_start: idx_segment_end][f"{imu_in}_{crrt_acc}"])

    # NOTE: this uses the matplotlib functions; see:
    # https://stackoverflow.com/a/21669766 , from which this is
    # heavily inspired :)

    # coherence plot
    # note this API is a bit unusual: from the documentation:
    # plotting, the power is plotted as decibels, though Cxy itself is returned...
    plt.figure()
    csdxy, fcsd = plt.csd(acc_in, acc_out, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap)
    plt.xlim([0.05, 1.0])
    plt.ylabel('CSD magnitude (db)')
    plt.xlabel("f [Hz]")
    plt.tight_layout()
    # plt.show()

    phase_rad = np.angle(csdxy)
    phase_deg = 180.0 / math.pi * phase_rad
    
    # coherence
    plt.figure()
    cxy, fcoh = plt.cohere(acc_in, acc_out, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap)
    plt.xlabel("f [Hz]")
    plt.ylabel("cherence")
    plt.xlim([0.05, 1.0])
    # plt.show()

    # nice fig 1
    valid_coherence = np.where(np.logical_and(np.logical_and(cxy>0.25, fcoh < 0.25), fcoh>0.08))
    valid_coherence_highnoise = None
    # we only have high frequency harmonic from segment 2 up to segment 5
    if crrt_segment_number < 5 and crrt_segment_number > 0:
        valid_coherence_highnoise = np.where(np.logical_and(np.logical_and(cxy>0.15, fcoh < 0.45), fcoh>0.08))

    # valid_coherence = np.where(np.logical_and(np.logical_and(cxy>0.3, fcoh < 0.45), fcoh>0.08))
    # nce fig 2
    # valid_coherence = np.where(np.logical_and(np.logical_and(cxy>0.4, fcoh < 0.4), fcoh>0.08))

    # Plot twin plot
    fig, ax1 = plt.subplots()
    # plot on ax1 the coherence
    ax1.plot(fcoh, cxy, 'b-')
    ax1.set_xlabel("f [Hz]")
    # ax1.set_ylim([0.05, 0.5])
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Coherence', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    # plot on ax2 the phase
    ax2 = ax1.twinx()
    ax2.plot(fcoh[valid_coherence], phase_deg[valid_coherence], 'r.')
    ax2.set_ylabel('Phase (degrees)', color='r')
    # ax2.set_ylim([-200,200])
    # ax2.set_yticklabels([-180,-135,-90,-45,0,45,90,135,180])

    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax1.grid(True)
    plt.xlim([0.05, 0.5])
    # plt.show()

    # angle of the waves relative to the 2 buoys
    angle_buoys_waves_deg = 0.0
    angle_buoys_waves_rad = math.pi / 180.0 * angle_buoys_waves_deg

    # k: obtained from a bit of trigonometry
    valid_frequency_hz = fcoh[valid_coherence]
    valid_phase_rad = phase_rad[valid_coherence]

    # plt.close('all')
    plt.figure()
    plt.plot(fcoh[valid_coherence], valid_phase_rad, label="before unwrap")

    # better method: method 2
    list_valid_phase_rad_unwrapped = [valid_phase_rad[0]]
    crrt_unwrap_value = 0.0
    previous_valid_phase_rad = valid_phase_rad[0]
    if previous_valid_phase_rad <= -math.pi/2:
        previous_valid_phase_rad += 2.0 * math.pi
        crrt_unwrap_value += 2.0*math.pi

    # previous_valid_phase_rad = 0.0
    # if previous_valid_phase_rad >= math.pi:
    #     crrt_unwrap_value = -math.pi
    #     previous_valid_phase_rad += crrt_unwrap_value
    threshold_2pi_wrap = 1.0
    for crrt_valid_phase_rad in valid_phase_rad[1:]:
        crrt_valid_phase_rad = crrt_valid_phase_rad + crrt_unwrap_value
        if crrt_valid_phase_rad - previous_valid_phase_rad <= -threshold_2pi_wrap*math.pi:
            crrt_unwrap_value += 2.0 * math.pi
            crrt_valid_phase_rad = crrt_valid_phase_rad + 2.0 * math.pi
        if crrt_valid_phase_rad - previous_valid_phase_rad >= threshold_2pi_wrap*math.pi:
            crrt_unwrap_value -= 2.0 * math.pi
            crrt_valid_phase_rad = crrt_valid_phase_rad - 2.0 * math.pi
        list_valid_phase_rad_unwrapped.append(crrt_valid_phase_rad)
        previous_valid_phase_rad = crrt_valid_phase_rad

    valid_phase_rad = np.array(list_valid_phase_rad_unwrapped)

    # method 1
    # def unwrap_phase(phase_rad):
    #     while True:
    #         if phase_rad > 2.0 * math.pi:
    #             phase_rad = phase_rad - 2.0 * math.pi
    #         elif phase_rad < -0.0 * math.pi:
    #             phase_rad = phase_rad + 2.0 * math.pi
    #         else:
    #             return phase_rad


    # valid_phase_rad = np.array([unwrap_phase(crrt_phase_rad) for crrt_phase_rad in valid_phase_rad])

    plt.plot(fcoh[valid_coherence], valid_phase_rad, label="after unwrap")
    plt.legend()
    # plt.show()
    
    valid_k = valid_phase_rad / math.cos(angle_buoys_waves_rad) / distance_imus

    list_valid_frequency_hz.append(valid_frequency_hz)
    list_valid_k.append(valid_k)

    # ---

    if valid_coherence_highnoise is not None:
        # k: obtained from a bit of trigonometry
        valid_frequency_hz = fcoh[valid_coherence_highnoise]
        valid_phase_rad = phase_rad[valid_coherence_highnoise]
    
        # better method: method 2
        list_valid_phase_rad_unwrapped = [valid_phase_rad[0]]
        crrt_unwrap_value = 0.0
        previous_valid_phase_rad = valid_phase_rad[0]
        if previous_valid_phase_rad <= -math.pi/2:
            previous_valid_phase_rad += 2.0 * math.pi
            crrt_unwrap_value += 2.0*math.pi

        # previous_valid_phase_rad = 0.0
        # if previous_valid_phase_rad >= math.pi:
        #     crrt_unwrap_value = -math.pi
        #     previous_valid_phase_rad += crrt_unwrap_value
        threshold_2pi_wrap = 1.0
        for crrt_valid_phase_rad in valid_phase_rad[1:]:
            crrt_valid_phase_rad = crrt_valid_phase_rad + crrt_unwrap_value
            if crrt_valid_phase_rad - previous_valid_phase_rad <= -threshold_2pi_wrap*math.pi:
                crrt_unwrap_value += 2.0 * math.pi
                crrt_valid_phase_rad = crrt_valid_phase_rad + 2.0 * math.pi
            if crrt_valid_phase_rad - previous_valid_phase_rad >= threshold_2pi_wrap*math.pi:
                crrt_unwrap_value -= 2.0 * math.pi
                crrt_valid_phase_rad = crrt_valid_phase_rad - 2.0 * math.pi
            list_valid_phase_rad_unwrapped.append(crrt_valid_phase_rad)
            previous_valid_phase_rad = crrt_valid_phase_rad

        valid_phase_rad = np.array(list_valid_phase_rad_unwrapped)

        # method 1
        # def unwrap_phase(phase_rad):
        #     while True:
        #         if phase_rad > 2.0 * math.pi:
        #             phase_rad = phase_rad - 2.0 * math.pi
        #         elif phase_rad < -0.0 * math.pi:
        #             phase_rad = phase_rad + 2.0 * math.pi
        #         else:
        #             return phase_rad


        # valid_phase_rad = np.array([unwrap_phase(crrt_phase_rad) for crrt_phase_rad in valid_phase_rad])

        # plt.plot(fcoh[valid_coherence], valid_phase_rad, label="after unwrap")
        # plt.legend()
        # plt.show()
    
        valid_k = valid_phase_rad / math.cos(angle_buoys_waves_rad) / distance_imus

        # # plot k vs. f
        # plt.figure()
        # plt.plot(valid_frequency_hz, valid_k, linestyle="", marker="o")
        # plt.xlabel("freq [Hz]")
        # plt.ylabel("k [rad/m]")
        # plt.show()

        # # plot all together
        # plt.figure()
        # plt.plot(valid_frequency_hz, valid_k, linestyle="", marker="o", label="csd")
        # plt.plot(fcoh, open_water_dispersion, label="open water")
        # plt.xlabel("freq [Hz]")
        # plt.ylabel("k [rad/m]")
        # plt.xlim([0.05, 0.3])
        # plt.ylim([-0.05, 0.2])
        # plt.legend()
        # # plt.show()

        list_valid_frequency_hz_highnoise.append(valid_frequency_hz)
        list_valid_k_highnoise.append(valid_k)
    else:
        list_valid_frequency_hz_highnoise.append([-1])
        list_valid_k_highnoise.append([-1])

    

plt.close('all')

plt.figure()

depth_m = 80
g_const = 9.81

# dispersion relation for open water

def dispersion_relation_open_water(omega, k, depth):
    return omega**2 - g_const * k * math.tanh(k * depth)

def solve_dispersion_relation_open_water(f, depth_m):
    omega = 2 * math.pi * f

    def residual(k):
        return dispersion_relation_open_water(omega, k, depth_m)

    initial_guess_k = omega**2 / g_const
    k_solution = fsolve(residual, initial_guess_k)

    return k_solution
        
open_water_dispersion = [solve_dispersion_relation_open_water(crrt_f, depth_m) for crrt_f in fcoh]

plt.plot(fcoh, open_water_dispersion, label="open water", linewidth=2.0, color="black")

# dispersion relation for thin plate

E = 3e9
rho_w = 1025
nu = 0.3
P = 0
rho_i = 920

D_ref = E * (0.6)**3 / (rho_w * 12 * (1 - nu**2))

for h, linestyle in zip([0.4, 0.6, 0.8], ["--", "-.", ":"]):
    D = E * h**3 / (rho_w * 12 * (1 - nu**2))

    Q = P * h / rho_w

    M = rho_i * h / rho_w

    D_over_D_ref = D / D_ref

    def dispersion_relation_elastic_plate(omega, k, depth, D, Q, M):
        if math.fabs(k) < 0.01:
            k = 0.01
        return omega**2 - (g_const*k + D*k**5 - Q*k**3) / (1.0/math.tanh(k*depth) + k*M)

    def solve_dispersion_relation_elastic_plate(f, depth_m):
        omega = 2 * math.pi * f

        def residual(k):
            return dispersion_relation_elastic_plate(omega, k, depth_m, D, Q, M)

        initial_guess_k = omega**2 / g_const
        k_solution = fsolve(residual, initial_guess_k)

        return k_solution
        
    elastic_plate_dispersion = [solve_dispersion_relation_elastic_plate(crrt_f, depth_m) for crrt_f in fcoh]

    plt.plot(fcoh, elastic_plate_dispersion, label=f"elastic plate {h}m; B/B_ref={round(D_over_D_ref, 2)}", linewidth=2.0, color="black", linestyle=linestyle)

cmap = plt.cm.viridis
list_colors = cmap(np.linspace(0, 1, len(list_segments)))

for (crrt_valid_frequency_hz, crrt_valid_k, crrt_segment, crrt_color) in zip(list_valid_frequency_hz, list_valid_k, list_segments, list_colors):
    if crrt_segment in [0, 5, 10, 15, 20]:
        plt.plot(crrt_valid_frequency_hz, crrt_valid_k, linestyle="", marker="o", label=f"csd segment {crrt_segment}", color=crrt_color)
    else:
        plt.plot(crrt_valid_frequency_hz, crrt_valid_k, linestyle="", marker="o", color=crrt_color)
for (crrt_valid_frequency_hz, crrt_valid_k, crrt_segment, crrt_color) in zip(list_valid_frequency_hz_highnoise, list_valid_k_highnoise, list_segments, list_colors):
    if crrt_segment in [2]:
        plt.plot(crrt_valid_frequency_hz, crrt_valid_k, linestyle="", marker="x", label=f"low coherence segments 0 to 4", color=crrt_color)
    else:
        plt.plot(crrt_valid_frequency_hz, crrt_valid_k, linestyle="", marker="x", color=crrt_color)
for (crrt_valid_frequency_hz, crrt_valid_k, crrt_segment, crrt_color) in zip(list_valid_frequency_hz, list_valid_k, list_segments, list_colors):
    plt.plot(crrt_valid_frequency_hz, crrt_valid_k, linestyle="", marker="o", color=crrt_color)
plt.xlabel("f [Hz]")
plt.ylabel("k [rad/m]")
# nice fig 1
# plt.xlim([0.06, 0.23])
plt.xlim([0.06, 0.45])
# nice fig 2
# plt.xlim([0.06, 0.4])
# nice fig 1
plt.ylim([-0.01, 0.2])
# nice fig 2
# plt.ylim([-0.01, 0.5])
plt.legend(framealpha=1.0)
plt.show()
pass

# %%

# export as CSV for Jie

for (crrt_valid_frequency_hz, crrt_valid_k, crrt_segment, crrt_color) in zip(list_valid_frequency_hz, list_valid_k, list_segments, list_colors):
    crrt_filename = f"valid_frequency_bins_segment{crrt_segment}_normalnoise.csv"

    df = pl.DataFrame({
                          "valid_frequency_hz": crrt_valid_frequency_hz,
                          "valid_k_radoverm": crrt_valid_k,
                      })

    df.write_csv(crrt_filename)

for (crrt_valid_frequency_hz, crrt_valid_k, crrt_segment, crrt_color) in zip(list_valid_frequency_hz_highnoise, list_valid_k_highnoise, list_segments, list_colors):
    crrt_filename = f"valid_frequency_bins_segment{crrt_segment}_highnoise.csv"

    df = pl.DataFrame({
                          "valid_frequency_hz": crrt_valid_frequency_hz,
                          "valid_k_radoverm": crrt_valid_k,
                      })

    df.write_csv(crrt_filename)

pass

# %%

for crrt_segment in range(21):
    plt.figure()

    crrt_filename = f"valid_frequency_bins_segment{crrt_segment}_normalnoise.csv"
    df = pl.read_csv(crrt_filename)
    plt.plot(df["valid_frequency_hz"], df["valid_k_radoverm"], label="normalnoise")

    crrt_filename = f"valid_frequency_bins_segment{crrt_segment}_highnoise.csv"
    df = pl.read_csv(crrt_filename)
    plt.plot(df["valid_frequency_hz"], df["valid_k_radoverm"], label="highnoise")

    plt.title(f"segment {crrt_segment=}")

    plt.legend()
    plt.xlim([0.05, 0.55])
    plt.ylim([-0.005, 0.25])
    
    plt.show()
    
pass

# %%

plt.figure()
for crrt_segment in range(21):

    crrt_filename = f"valid_frequency_bins_segment{crrt_segment}_normalnoise.csv"
    df = pl.read_csv(crrt_filename)
    plt.plot(df["valid_frequency_hz"], df["valid_k_radoverm"], label="normalnoise")

    crrt_filename = f"valid_frequency_bins_segment{crrt_segment}_highnoise.csv"
    df = pl.read_csv(crrt_filename)
    plt.plot(df["valid_frequency_hz"], df["valid_k_radoverm"], label="highnoise")

# plt.legend()
plt.xlim([0.05, 0.55])
plt.ylim([-0.005, 0.25])

plt.show()
   

# %%

# look at wave triads, and compare with the field data
# part 1: compute

# the actual best dispersion relation from the comparison to the data is for 0.6m
E = 3e9
rho_w = 1025
nu = 0.3
P = 0
rho_i = 920
h = 0.6

D = E * h**3 / (rho_w * 12 * (1 - nu**2))

Q = P * h / rho_w

M = rho_i * h / rho_w

def dispersion_relation_elastic_plate(omega, k, depth, D, Q, M):
    # special case due to the tanh
    if math.fabs(k) < 1e-12:
        k = 1e-12
    return omega**2 - (g_const*k + D*k**5 - Q*k**3) / (1.0/math.tanh(k*depth) + k*M)

def solve_dispersion_relation_elastic_plate_f(f, depth_m):
    omega = 2 * math.pi * f

    def residual(k):
        return dispersion_relation_elastic_plate(omega, k, depth_m, D, Q, M)

    initial_guess_k = omega**2 / g_const
    k_solution = fsolve(residual, initial_guess_k)

    return k_solution


def solve_dispersion_relation_elastic_plate_k(k, depth_m):

    def residual(omega):
        return dispersion_relation_elastic_plate(omega, k, depth_m, D, Q, M)

    initial_guess_omega = math.sqrt(k * g_const)
    omega_solution = fsolve(residual, initial_guess_omega)
    f_solution = omega_solution / 2.0 / math.pi

    return f_solution


# compute the triads
# this requires:
# f1 + f2 = f3
# k1 + k2 = k3
# main peak for active f (relevant for 1 and 2) is [0.08-0.2] at most, ie active k main peak [0.025-0.175] at most
# we plot the triads as: k3 the horizontal one which is also the longest one, k1 and k2 on top of each others to make k3
# things are symmetric, we only consider the upper right quadrant

# from the bicoherence plot:
# there is not a single exact answer - this is a cluster of points

# example 1
# f1_bicoherence = 0.15
# f2_bicoherence = 0.17
# f3_bicoherence = f1_bicoherence + f2_bicoherence

# example 2
f1_bicoherence = 0.145
f2_bicoherence = 0.175
f3_bicoherence = f1_bicoherence + f2_bicoherence

k3_bicoherence = float(solve_dispersion_relation_elastic_plate_f(f3_bicoherence, depth_m))
k2_bicoherence = float(solve_dispersion_relation_elastic_plate_f(f2_bicoherence, depth_m))
k1_bicoherence = float(solve_dispersion_relation_elastic_plate_f(f1_bicoherence, depth_m))
print(f"{f3_bicoherence = }")
print(f"{k3_bicoherence = }")

# list_k3 = [0.155]
# list_k3 = np.arange(0.08, 0.20+0.01, 0.010)
list_k3 = [k3_bicoherence, 0.12, 0.13, 0.15, 0.17, 0.19, 0.21]

list_triads_k3 = []
list_level_lines_x = []
list_level_lines_y = []

for k3 in list_k3:
    f_3 = solve_dispersion_relation_elastic_plate_k(k3, depth_m)

    print(f"{f_3 = }")

    min_k1x = 0.0001
    max_k1x = 0.22

    # min_k1y = -max_k1x/2.0
    min_k1y = 0.0001
    max_k1y = max_k1x / 1.0

    # npoints_x = 80
    # npoints_y = 75
    resolution_factor = 5
    npoints_x = resolution_factor*80
    npoints_y = resolution_factor*75

    k1test_x = np.arange(min_k1x, max_k1x, (max_k1x-min_k1x)/npoints_x)
    k1test_y = np.arange(min_k1y, max_k1y, (max_k1y-min_k1y)/npoints_y)

    k2test_x = k3 - k1test_x
    k2test_y = -k1test_y

    residual_f = np.full((len(k1test_x), len(k1test_y)), math.nan)
    f_1_arr = np.full((len(k1test_x), len(k1test_y)), math.nan)
    f_2_arr = np.full((len(k1test_x), len(k1test_y)), math.nan)

    for i in range(len(k1test_x)):
        for j in range(len(k1test_y)):
            k1 = math.sqrt(k1test_x[i]**2 + k1test_y[j]**2)
            k2 = math.sqrt(k2test_x[i]**2 + k2test_y[j]**2)
            f_1 = solve_dispersion_relation_elastic_plate_k(k1, depth_m)
            f_2 = solve_dispersion_relation_elastic_plate_k(k2, depth_m)
            f_1_arr[i, j] = f_1
            f_2_arr[i, j] = f_2
            residual_f[i, j] = f_3 - f_1 - f_2

    print(f"{np.max(residual_f) = }")
    print(f"{np.min(residual_f) = }")

    if np.max(residual_f) > 0 and np.min(residual_f) < 0:

        plt.figure()
        contour_set = plt.contour(k1test_x, k1test_y, np.transpose(residual_f), color="red", levels=[0])
        plt.scatter([k3], [0], marker="o")
        # plt.colorbar()
        plt.legend()
        plt.xlim([0.0, 1.1*k3])
        # plt.show()

        level_line_0 = contour_set.allsegs[0][0]
        list_level_lines_x.append(level_line_0[:, 0])
        list_level_lines_y.append(level_line_0[:, 1])
        list_triads_k3.append(k3)

pass

# %%

# part 2: plot

plt.close('all')

cmap = plt.cm.viridis
list_colors = cmap(np.linspace(0, 1, len(list_triads_k3)))

plt.figure()
idx_match_bicoherence = 0
crrt_idx = -1
for (crrt_k, crrt_lines_x, crrt_lines_y, crrt_color) in zip(list_triads_k3, list_level_lines_x, list_level_lines_y, list_colors):
    crrt_idx += 1
    if crrt_idx == idx_match_bicoherence:
        continue
    crrt_rounded_k = round(crrt_k, 4)
    plt.scatter([crrt_k], [0], color=crrt_color, label=f"k3={crrt_rounded_k}", marker="*")
    plt.plot(crrt_lines_x, crrt_lines_y, color=crrt_color)
#
# plt.scatter([list_triads_k3[idx_match_bicoherence]], [0], label=f"k3b={round(k3_bicoherence, 3)}", marker="o", color="k")
plt.plot(list_level_lines_x[idx_match_bicoherence], list_level_lines_y[idx_match_bicoherence], label=r'$\mathbf{k}$1 locii, k3_b=0.139', linestyle="-", color="k")
#

plt.plot([0, k3_bicoherence], [0, 0], color="k", label=r"$\mathbf{k}$3_b", linestyle="--")

residual_k1_bicoherence = np.abs(np.sqrt(np.array(list_level_lines_x[idx_match_bicoherence])**2 + np.array(list_level_lines_y[idx_match_bicoherence])**2) - k1_bicoherence)
idx_match_k1_bicoherence = int(np.argmin(residual_k1_bicoherence))
plt.plot([0, list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence]], [0, list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence]], color="r", linestyle="-.", label=r"$\mathbf{k}$1_b")
plt.plot([list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence], k3_bicoherence], [list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence], 0], color="r", linestyle=":", label=r"$\mathbf{k}$2_b")

# first triad
# consistency checks
k1 = math.sqrt((list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence])**2 + (list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence])**2)
k2 = math.sqrt((k3_bicoherence - list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence])**2 + (list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence])**2)
k3 = k3_bicoherence
# k1 + k2 = k3 is ok by construction of the vectors
f1 = solve_dispersion_relation_elastic_plate_k(k1, depth_m)
f2 = solve_dispersion_relation_elastic_plate_k(k2, depth_m)
f3 = solve_dispersion_relation_elastic_plate_k(k3, depth_m)
print("summary first triad match - should match closely, note there may be minor mismatches due to the numerical rounding + discrete plane discretization, this is as expected")
print(f"{f1 = }")
print(f"{f2 = }")
print(f"{f3 = }")
print(f"{f1 + f2 = }")

residual_k1_bicoherence = np.abs(np.sqrt(np.array(list_level_lines_x[idx_match_bicoherence])**2 + np.array(list_level_lines_y[idx_match_bicoherence])**2) - k2_bicoherence)
idx_match_k1_bicoherence = int(np.argmin(residual_k1_bicoherence))
plt.plot([0, list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence]], [0, list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence]], color="r", linestyle="-.")
plt.plot([list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence], k3_bicoherence], [list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence], 0], color="r", linestyle=":")

# second triad
# consistency checks
k1 = math.sqrt((list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence])**2 + (list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence])**2)
k2 = math.sqrt((k3_bicoherence - list_level_lines_x[idx_match_bicoherence][idx_match_k1_bicoherence])**2 + (list_level_lines_y[idx_match_bicoherence][idx_match_k1_bicoherence])**2)
k3 = k3_bicoherence
# k1 + k2 = k3 is ok by construction of the vectors
f1 = solve_dispersion_relation_elastic_plate_k(k1, depth_m)
f2 = solve_dispersion_relation_elastic_plate_k(k2, depth_m)
f3 = solve_dispersion_relation_elastic_plate_k(k3, depth_m)
print("summary second triad match - should match closely, note there may be minor mismatches due to the numerical rounding + discrete plane discretization, this is as expected")
print(f"{f1 = }")
print(f"{f2 = }")
print(f"{f3 = }")
print(f"{f1 + f2 = }")

#
plt.xlim([0.0, 0.22])
plt.ylim([-0.001, 0.12])
plt.gca().set_aspect('equal')
plt.xlabel("kx [rad/m]")
plt.ylabel("ky [rad/m]")
plt.legend(bbox_to_anchor=(1, 1.0))
plt.tight_layout()
plt.show()

pass

# %%
