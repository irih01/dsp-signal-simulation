import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parametri PWM
# =========================
A = 1.0            # amplitudine
f_pwm = 10e3       # frecvență PWM: 10 kHz
duty = 0.4         # factor de umplere (40%)

# =========================
# Parametri simulare
# =========================
fs = 1e6           # 1 MHz
duration = 0.02    # 20 ms
t = np.arange(0, duration, 1/fs)

# =========================
# Generare PWM
# =========================
signal = A * ((t % (1/f_pwm)) < (duty / f_pwm))

# =========================
# FFT
# =========================
freqs = np.fft.rfftfreq(len(signal), 1/fs)
spectrum = np.abs(np.fft.rfft(signal)) / len(signal)

# =========================
# Plot timp
# =========================
plt.figure(figsize=(12, 4))
plt.plot(t[:2000] * 1e3, signal[:2000])
plt.title("Semnal PWM industrial – domeniul timp")
plt.xlabel("Timp [ms]")
plt.ylabel("Amplitudine")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Plot frecvență
# =========================
plt.figure(figsize=(12, 4))
plt.stem(freqs / 1e3, spectrum, basefmt=" ")
plt.xlim(0, 100)
plt.title("Spectru semnal PWM industrial")
plt.xlabel("Frecvență [kHz]")
plt.ylabel("Magnitudine")
plt.grid(True)
plt.tight_layout()
plt.show()
