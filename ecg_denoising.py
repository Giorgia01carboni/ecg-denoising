import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ecg = pd.read_csv("noisyEcgFromWearable.csv")
ecg_values = ecg.iloc[:, 0].values
N = len(ecg)
# fs = sample rate
fs = 500

time_vector = []
# 500 number of samples per second
delta_t = 1 / fs
# Time vector: from 0 to number of total samples N, incrementing by \delta_t
# for each sample
time_vector = np.arange(0, delta_t*N, delta_t)

plt.plot(time_vector, ecg_values)
plt.show()

# Window selected: Hamming
# Window length L = (4 / 20 Hz) * 500 Hz (should be approximated to first odd number 101)

L = int(np.ceil((4 / 20) * fs))
if L % 2 == 0:
    L += 1

# window vector
w = np.hamming(L)

plt.plot(w)
plt.title("Hamming Window (Zero-Phase)")
plt.show()

# cutoff frequencies:
# Lower cutoff frequency: 0.5Hz
# Upper cutoff frequency: 40Hz

# Normalized (divided by the Nyquist frequency):
upper_cutoff = 40 / (fs/2)
lower_cutoff = 0.5 / (fs/2)

# Filter vector h
time_vector_h = np.arange(-L/2, L/2)
lower_sinc = np.sinc(2 * time_vector_h * lower_cutoff)
upper_sinc = np.sinc(2 * time_vector_h * upper_cutoff)

h = upper_sinc - lower_sinc

h_filtered = w * h

# Filter ecg using circular convolution
ecg_filtered = np.convolve(ecg_values, h_filtered, mode='same')
plt.plot(ecg_filtered)
plt.show()