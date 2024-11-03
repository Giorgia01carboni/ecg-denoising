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
plt.title("Original ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.show()

# Window selected: Hamming
# Window length L = (4 / 20 Hz) * 500 Hz (should be approximated to first odd number 101)
# Window Length is based on transition band width

L = int(np.ceil((4 / 20) * fs))
if L % 2 == 0:
    L += 1      # Makes sure L value is odd

# window vector
w = np.hamming(L)

# padding with zeros
w_padded = np.pad(w, (0, N - len(w)), 'constant')

# shifting padded window to have max value at the centre
max_index = np.argmax(w_padded)
w_zero_phase = np.roll(w_padded, -max_index + N // 2)

plt.plot(w_zero_phase)
plt.title("Hamming Window (Zero-Phase)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()

# cutoff frequencies for the bandpass filter:
# Lower cutoff frequency: 0.5Hz
# Upper cutoff frequency: 40Hz

# Normalized (divided by the Nyquist frequency):
upper_cutoff = 40 / (fs/2)
lower_cutoff = 0.5 / (fs/2)

# Time vector for filter h
time_vector_h = np.arange(-L//2, L//2)
lower_sinc = np.sinc(2 * time_vector_h * upper_cutoff)          # allow frequencies below 40 Hz to pass

# delta function and design of high-pass filter
delta = np.zeros(N)
delta[0] = 1

h_low = np.sinc(2 * time_vector_h * lower_cutoff)      # low-pass filter with a cutoff at 0.5 Hz
h_low_padded = np.pad(h_low, (0, N - len(h_low)), 'constant')
h_high = delta - h_low_padded          # blocks frequencies below 0.5 Hz and allows frequencies above 0.5 Hz to pass

# bandpass filter * zero-phase window = impulse response
h_bandpass = np.fft.ifft(np.fft.fft(h_low_padded) * np.fft.fft(h_high)).real
h_filtered = h_bandpass * w_zero_phase

H = np.fft.fft(h_bandpass)
H_filtered = np.fft.fft(h_bandpass * w_zero_phase)

# Magnitude and phase of the DFT
freqs = np.fft.fftfreq(len(H), d = 1/fs)
magnitude = 20 * np.log10(np.abs(H))
phase = np.angle(H)
plt.plot(freqs, magnitude)
plt.show()

plt.plot(freqs, phase)
plt.show()

# Filter ecg using circular convolution
ecg_filtered = np.convolve(ecg_values, H_filtered, mode='same')
plt.plot(time_vector, ecg_filtered)
plt.show()