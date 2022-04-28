import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal

data = pd.read_csv('Data/datatraining.txt', header=0)
len_min = len(data)
len_day = len_min/24/60
print("Number of days: ", len_day)

# temp = [x for x in data.iloc[:,1].values]
temp = [x-19 for x in data["Temperature"].values]

tempN = temp - np.mean(temp)
plt.figure()
plt.plot(tempN)

fs = 24*60

f, Pxx = signal.periodogram(tempN, fs = fs, window='hanning', scaling='spectrum')
plt.figure(figsize = (6, 5))
plt.plot(f, Pxx)
plt.yscale('log')
plt.xlabel('Frequency [cycle/day]')
plt.show()

for amp_arg in np.argsort(np.abs(Pxx))[::-1][1:3]:
    day = 1 / f[amp_arg]
    print("frequency: {}; {} days/cycle".format(f[amp_arg],day))