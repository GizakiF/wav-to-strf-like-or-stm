"""
Copyright (c) Baptiste Caramiaux, Etienne Thoret
Please cite us if you use this script :)
All rights reserved

"""

import numpy as np
import math
import aifc
import matplotlib.pylab as plt
import wave
from scipy.io import wavfile


def raised_cosine(x, mu, s):
    return 1 / 2 / s * (1 + np.cos((x - mu) / s * math.pi)) * s


def nextpow2(n):
    counter = 1
    while math.pow(2, counter) < n:
        counter += 1
    return counter


def angle(compl_values):
    real_values = np.array([x.real for x in compl_values])
    imag_values = np.array([x.imag for x in compl_values])
    return np.arctan2(imag_values, real_values)


def sigmoid(x, fac):
    """
    Compute sigmoidal function
    """
    y = x
    if fac > 0:
        y = 1.0 / (1.0 + np.exp(-y / fac))
    elif fac == 0:
        y = y > 0
    elif fac == -1:
        y = np.max(y, 0)
    elif fac == -3:
        raise ValueError("not implemented")
    return y


def get_dissimalrity_matrix(folder_path="../ext/data/"):
    return np.loadtxt(folder_path + "/dissimilarity_matrix.txt")


def audio_data(filename):
    ext = filename.split(".")[-1].lower()

    if ext == "aiff":
        aif = aifc.open(filename, "r")
        fs_wav = aif.getframerate()
        wavtemp = aif.readframes(aif.getnframes())
        wavtemp = np.fromstring(wavtemp, np.short).byteswap() / 32767.0
        return wavtemp, fs_wav

    elif ext == "wav":
        fs_wav, wavtemp = wavfile.read(filename)
        if wavtemp.dtype == np.int16:
            wavtemp = wavtemp / 32767.0  # Normalize to [-1, 1]
        elif wavtemp.dtype == np.int32:
            wavtemp = wavtemp / 2147483647.0
        elif wavtemp.dtype == np.uint8:
            wavtemp = (wavtemp - 128) / 128.0
        return wavtemp, fs_wav

    else:
        raise ValueError("Unsupported file format. Please use .wav or .aiff")


if __name__ == "__main__":
    plt.plot(raised_cosine(np.arange(2205), 2205, 2205))
    plt.show()
