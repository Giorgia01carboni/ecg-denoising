import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ecg = pd.read_csv("noisyEcgFromWearable.csv")
ecg_values = ecg.iloc[:, 0].values
N = len(ecg)

time_vector = []
# 500 number of samples per second
delta_t = 1 / 500

time_vector = np.arange(0, delta_t*N, delta_t)

plt.plot(time_vector, ecg_values)
plt.show()