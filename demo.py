"""
Copyright (c) Baptiste Caramiaux, Etienne Thoret
Please cite us if you use this script :)
All rights reserved

"""

import matplotlib.pylab as plt
import auditory
import utils
import numpy as np
import scipy.io as sio
import plotslib
import os


####
audio, fs = utils.audio_data(
    os.path.expanduser(
        "~/Research/Sleep Deprivation Detection using voice/scripts/preprocess/processed_audio/segment_1.wav"
    )
)
rates_vec = [
    -32,
    -22.6,
    -16,
    -11.3,
    -8,
    -5.70,
    -4,
    -2,
    -1,
    -0.5,
    -0.25,
    0.25,
    0.5,
    1,
    2,
    4,
    5.70,
    8,
    11.3,
    16,
    22.6,
    32,
]
# scales_vec = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66]
scales_vec = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00]
strf, auditory_spectrogram_, mod_scale, scale_rate = auditory.strf(
    audio, audio_fs=fs, duration=5, rates=rates_vec, scales=scales_vec
)
plt.imshow(
    np.transpose(auditory_spectrogram_[:][1:80]),
    aspect="auto",
    interpolation="gaussian",
    origin="lower",
)
plt.show()

strf_shape = ", ".join(map(str, strf.shape))
print("strf shape: " + strf_shape)
avgvec = plotslib.strf2avgvec(strf)
strf_scale_rate, strf_freq_rate, strf_freq_scale = plotslib.avgvec2strfavg(
    avgvec, nbScales=len(scales_vec), nbRates=len(rates_vec)
)
plotslib.plotStrfavgEqual(
    strf_scale_rate, strf_freq_rate, strf_freq_scale, cmap="seismic"
)
print(f"strf_scale_rate: {strf_scale_rate}")

print()

print(f"strf_freq_rate: {strf_freq_rate}")

print()

print(f"strf_freq_scale: {strf_freq_scale}")
