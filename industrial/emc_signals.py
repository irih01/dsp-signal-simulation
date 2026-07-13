import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from utils import fft_analysis


class EMCSignal:
    """
    Pipeline EMC/EMI
    - semnal util
    - interferență PWM
    - interferență impulsivă
    - analiză spectrală
    - filtrare
    - metrici (SNR)
    """

    def __init__(self, fs=1_000_000, duration=0.01):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # ==================================
    # Semnal util
    # ==================================
    def generate_clean(self):
        return 0.5 * np.sin(2 * np.pi * 5 * self.t)

    def generate_ramp_singal(self, slope=1.0):
        return slope * self.t
    # ==================================
    # Interferență PWM
    # ==================================
    def add_pwm(self, signal, f_pwm=20_000, duty=0.4, amp=0.3):
        pwm = amp * ((self.t % (1/f_pwm)) < (duty / f_pwm))
        return signal + pwm

    # ==================================
    # Interferență impulsivă EMI
    # ==================================
    def add_emi_impulses(self, signal, n_pulses=20, width_us=2, amp=0.5):
        emi = np.zeros_like(signal)
        width = int(width_us * 1e-6 * self.fs)
        idx = np.random.choice(len(signal), n_pulses, replace=False)

        for i in idx:
            emi[i:i+width] = amp

        return signal + emi

    # ==================================
    # Filtru low-pass
    # ==================================
    def lowpass(self, signal, cutoff=100):
        nyq = self.fs / 2
        b, a = butter(4, cutoff/nyq, btype="low")
        return filtfilt(b, a, signal)

    # ==================================
    # SNR
    # ==================================
    def compute_snr(self, clean, noisy):
        noise = noisy - clean
        return 10 * np.log10(np.var(clean) / np.var(noise))

    # ==================================
    # Plot
    # ==================================
    def plot(self, clean, disturbed, filtered):
        freqs, fft_clean = fft_analysis(clean, self.fs)
        _, fft_dist = fft_analysis(disturbed, self.fs)
        _, fft_filt = fft_analysis(filtered, self.fs)

        plt.figure(figsize=(12, 6))

        plt.subplot(2,1,1)
        plt.plot(self.t*1e3, clean, label="Semnal util")
        plt.plot(self.t*1e3, disturbed, label="Afectat EMI", alpha=0.6)
        plt.plot(self.t*1e3, filtered, label="Filtrat", linewidth=2)
        plt.xlim(0, 5)
        plt.legend()
        plt.title("Semnal industrial – domeniul timp")
        plt.xlabel("Timp [ms]")
        plt.ylabel("Amplitudine")

        plt.subplot(2,1,2)
        plt.plot(freqs, fft_clean, label="Util")
        plt.plot(freqs, fft_dist, label="Afectat", alpha=0.6)
        plt.plot(freqs, fft_filt, label="Filtrat", linewidth=2)
        plt.xlim(0, 200_000)
        plt.legend()
        plt.title("Semnal industrial – domeniul frecvență")
        plt.xlabel("Frecvență [Hz]")

        plt.tight_layout()
        plt.show()

    def plot_db(self, clean, noisy, filtered):
        snr_before = self.compute_snr(clean, noisy)
        snr_after = self.compute_snr(clean, filtered)

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        ax[0].plot(self.t, clean, label="Semnal util (rampă)")
        ax[0].plot(self.t, noisy, alpha=0.6, label="Semnal perturbat")
        ax[0].plot(self.t, filtered, linewidth=2, label="Semnal filtrat")
        ax[0].set_title(
            f"Domeniul timp | SNR inițial={snr_before:.2f} dB → SNR filtrat={snr_after:.2f} dB"
        )
        ax[0].legend()

        freqs = np.fft.rfftfreq(len(noisy), 1/self.fs)
        ax[1].plot(freqs, np.abs(np.fft.rfft(noisy)), alpha=0.5, label="Spectru perturbat")
        ax[1].plot(freqs, np.abs(np.fft.rfft(filtered)), linewidth=2, label="Spectru filtrat")
        ax[1].set_xlim(0, 5000)
        ax[1].legend()
        ax[1].set_xlabel("Frecvență [Hz]")
        ax[1].set_ylabel("Magnitudine")

        plt.tight_layout()
        plt.show()