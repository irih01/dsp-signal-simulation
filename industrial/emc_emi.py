import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parametri semnal EMI
# =========================
A = 1.0          # amplitudine (V sau V/m)
tp = 2e-9        # durata impuls (2 ns)
T = 20e-9        # perioadă (20 ns)
N_pulses = 10

# =========================
# Parametri simulare
# =========================
fs = 50e9        # 50 GHz
duration = N_pulses * T
t = np.arange(0, duration, 1/fs)

# =========================
# Generare tren de impulsuri
# =========================
signal = np.zeros_like(t)
for k in range(N_pulses):
    start = int(k * T * fs)
    end = int((k * T + tp) * fs)
    signal[start:end] = A

# =========================
# FFT
# =========================
freqs = np.fft.rfftfreq(len(signal), 1/fs)
spectrum = np.abs(np.fft.rfft(signal)) / len(signal)

# =========================
# Plot timp
# =========================
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, signal)
plt.title("EMI impulsiv – tren de impulsuri rapide")
plt.xlabel("Timp [ns]")
plt.ylabel("Amplitudine")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Plot frecvență
# =========================
plt.figure(figsize=(12, 4))
plt.plot(freqs / 1e6, spectrum)
plt.xlim(0, 3000)
plt.title("Spectru EMI – impulsuri rapide")
plt.xlabel("Frecvență [MHz]")
plt.ylabel("Magnitudine")
plt.grid(True)
plt.tight_layout()
plt.show()
