import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from utils import fft_analysis


class EnergySignal:
    """
    Pipeline energetic
    - semnal rețea ideal
    - distorsiuni armonice
    - analiză spectrală
    - filtrare
    - metrici (THD)
    """

    def __init__(self, fs=5000, duration=1.0, f0=50):
        self.fs = fs
        self.duration = duration
        self.f0 = f0
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # ==================================
    # Semnal ideal
    # ==================================
    def generate_clean(self, amplitude=1.0):
        return amplitude * np.sin(2 * np.pi * self.f0 * self.t)

    # ==================================
    # Distorsiuni armonice
    # ==================================
    def add_harmonics(self, signal, harmonics=[3, 5, 7], level=0.2):
        distorted = signal.copy()
        for h in harmonics:
            distorted += level * np.sin(2 * np.pi * h * self.f0 * self.t)
        return distorted

    # ==================================
    # Filtru notch (50 Hz)
    # ==================================
    def notch_filter(self, signal, Q=30):
        b, a = iirnotch(self.f0, Q, self.fs)
        return filtfilt(b, a, signal)
    
    def bandpass(self, signal, low=45, high=55, order=4):
        nyq = self.fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, signal)

    # ==================================
    # THD
    # ==================================
    def compute_thd(self, signal):
        freqs, spectrum = fft_analysis(signal, self.fs)
        fund = spectrum[np.argmin(np.abs(freqs - self.f0))]
        harm = np.sum(
            spectrum[(freqs > self.f0 * 1.5) & (freqs < self.f0 * 10)]**2
        )
        return np.sqrt(harm) / fund

    # ==================================
    # Plot
    # ==================================
    def plot(self, clean, distorted, filtered):
        freqs, fft_clean = fft_analysis(clean, self.fs)
        _, fft_dist = fft_analysis(distorted, self.fs)
        _, fft_filt = fft_analysis(filtered, self.fs)

        plt.figure(figsize=(12, 6))

        plt.subplot(2,1,1)
        plt.plot(self.t, clean, label="Ideal")
        plt.plot(self.t, distorted, label="Distorsionat", alpha=0.6)
        plt.plot(self.t, filtered, label="Filtrat", linewidth=2)
        plt.xlim(0, 0.1)
        plt.legend()
        plt.title("Semnal rețea – domeniul timp")
        plt.xlabel("Timp [s]")
        plt.ylabel("Amplitudine")

        plt.subplot(2,1,2)
        plt.plot(freqs, fft_clean, label="Ideal")
        plt.plot(freqs, fft_dist, label="Distorsionat", alpha=0.6)
        plt.plot(freqs, fft_filt, label="Filtrat", linewidth=2)
        plt.xlim(0, 500)
        plt.legend()
        plt.title("Semnal rețea – domeniul frecvență")
        plt.xlabel("Frecvență [Hz]")

        plt.tight_layout()
        plt.show()

    def plot_db(self, raw, filtered):
        freqs = np.fft.rfftfreq(len(raw), 1/self.fs)
        fft_raw = np.abs(np.fft.rfft(raw))
        fft_filt = np.abs(np.fft.rfft(filtered))

        thd_raw = self.compute_thd(raw)
        thd_filt = self.compute_thd(filtered)

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        ax[0].plot(self.t, raw, label="Semnal distorsionat", alpha=0.5)
        ax[0].plot(self.t, filtered, label="Semnal filtrat", linewidth=2)
        ax[0].set_title(
            f"Domeniul timp | THD inițial={thd_raw:.3f}, THD filtrat={thd_filt:.3f}"
        )
        ax[0].legend()
        ax[0].set_xlabel("Timp [s]")
        ax[0].set_ylabel("Amplitudine")

        ax[1].plot(freqs, fft_raw, alpha=0.5, label="Spectru distorsionat")
        ax[1].plot(freqs, fft_filt, linewidth=2, label="Spectru filtrat")
        ax[1].set_xlim(0, 300)
        ax[1].set_xlabel("Frecvență [Hz]")
        ax[1].set_ylabel("Magnitudine")
        ax[1].legend()

        plt.tight_layout()
        plt.show()
