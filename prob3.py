import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parametri fizici
# =========================
E_peak = 1.0        # V/m
tp = 10e-9          # durata pulsului: 10 ns
T = 100e-9          # perioada: 100 ns
N_pulses = 3        # număr pulsuri

D = tp / T          # duty cycle
f0 = 1 / T          # frecvență fundamentală

# =========================
# Parametri simulare
# =========================
fs = 100e9          # 100 GHz
duration = N_pulses * T
t = np.arange(0, duration, 1/fs)

# =========================
# Generare pulsuri dreptunghiulare
# =========================
signal = np.zeros_like(t)

for k in range(N_pulses):
    start = int(k * T * fs)
    end = int((k * T + tp) * fs)
    signal[start:end] = E_peak

# =========================
# FFT numeric
# =========================
N = len(signal)
freqs = np.fft.rfftfreq(N, 1/fs)
spectrum = np.abs(np.fft.rfft(signal)) / N

# =========================
# Spectru teoretic (envelopa sinc)
# =========================
# |E(f)| = E0 * D * |sinc(pi f tp)|
sinc_envelope = E_peak * D * np.abs(
    np.sinc(freqs * tp)
)

# =========================
# Plot domeniul timp
# =========================
plt.figure(figsize=(12, 4))
plt.plot(t * 1e9, signal, linewidth=2)
plt.title("Câmp electric – 3 pulsuri dreptunghiulare")
plt.xlabel("Timp [ns]")
plt.ylabel("E [V/m]")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Plot domeniul frecvență
# =========================
plt.figure(figsize=(12, 4))

plt.plot(freqs / 1e6, spectrum, label="FFT numeric", alpha=0.7)
plt.plot(freqs / 1e6, sinc_envelope, 'r--',
         label="Envelopă teoretică sinc")

# Linii spectrale teoretice
harmonics = np.arange(1, 50) * f0
plt.vlines(harmonics / 1e6, 0,
           E_peak * D, colors='gray', alpha=0.3)

plt.xlim(0, 1000)   # până la 1 GHz
plt.title("Spectrul de frecvență – pulsuri repetitive")
plt.xlabel("Frecvență [MHz]")
plt.ylabel("Magnitudine |E(f)|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
