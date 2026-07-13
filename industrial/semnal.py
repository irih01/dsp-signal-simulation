import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parametri
fs = 1000          # Hz – rata de eșantionare
T = 2.0            # secunde
N = int(fs * T)
t = np.linspace(0, T, N, endpoint=False)




# Componente semnal de 5 - 50 Hz
signal_1 = 0.7 * np.sin(2 * np.pi * 5 * t)     # 5 Hz (lent)
signal_2 = 0.3 * np.sin(2 * np.pi * 50 * t)    # 50 Hz
noise = 0.1 * np.random.randn(N)

signal = signal_1 + signal_2 + noise

# Fereastra Hann
window = np.hanning(N)
singal_w = signal * window

# FFT
fft_vals = np.fft.rfft(singal_w)
freqs = np.fft.rfftfreq(N, 1/fs)

# Magnitudine
magn = np.abs(fft_vals)


# Plot semnal timp
plt.figure()
plt.plot(t, signal)
plt.title("Semnal simulat 5 - 50 Hz (timp)")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")
plt.show()

#Plot FFT
plt.figure()
plt.plot(freqs, magn)
plt.title("Spectru FFT")
plt.xlabel("Frecventa [Hz]")
plt.ylabel("Magnitudine")
plt.xlim(0, 100)
plt.show()




#  fs = 1000      # Hz
#  T = 1.0        # 1 secunda
#  t = np.linspace(0, T, int(fs*T), endpoint=False)

#  # Semnal de retea electrica (50 Hz)
#  powerline = 0.3 * np.sin(2 * np.pi * 50 * t)
# pnoise = powerline + noise

# plt.figure(figsize=(10,3))
# plt.plot(t, pnoise)
# plt.xlabel("Timp [s]")
# plt.ylabel("Amplitudine")
# plt.title("Semnal de rețea electrică (50 Hz)")
# plt.grid()
# plt.tight_layout()
# plt.show()
