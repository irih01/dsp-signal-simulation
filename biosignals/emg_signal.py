import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import rms, mean_frequency, median_frequency

class EMGSignal:
    def __init__(self, fs: int, duration: float):
        self.fs = fs
        self.duration = duration
        self.time_vector = self.generate_time_vector()

    def generate_time_vector(self):
        return np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)

    def generate_activ(self, start=0.5, end=1.5):
        activation = np.zeros_like(self.time_vector)
        activation[(self.time_vector > start) & (self.time_vector < end)] = 1.0
        return activation

    def generate_emg_signal(
            self,
            activation,
            base_noise_lvl=0.1,
            muscle_noise_lvl=0.6,
            pline_fq=50,
            pline_amp=0.2):
        emg_base = base_noise_lvl * np.random.randn(len(self.time_vector))
        emg_active = activation * muscle_noise_lvl * np.random.randn(len(self.time_vector))
        powerline = pline_amp * np.sin(2 * np.pi * pline_fq * self.time_vector)

        return emg_base + emg_active + powerline
    

    def generate_fatigue_profile(self, time, start=0.5, end=1.5, strength=0.6):
        '''Oboseala musculară se manifestă prin:

            scăderea frecvenței mediane

            creșterea amplitudinii RMS inițial, apoi scădere

            conținut spectral mutat spre frecvențe joase
        '''
        fatigue = np.ones_like(time)
        idx = (time > start) & (time < end)
        fatigue[idx] = 1 - strength * (time[idx] - start) / (end - start)
        return fatigue

    def generate_emg_with_fatigue(
        self,
        t,
        activation,
        fatigue,
        base_noise_level=0.1,
        muscle_noise_level=0.6,
        powerline_freq=50,
        powerline_amp=0.2
    ):
        emg_base = base_noise_level * np.random.randn(len(t))
        emg_active = activation * fatigue * muscle_noise_level * np.random.randn(len(t))
        powerline = powerline_amp * np.sin(2 * np.pi * powerline_freq * t)

        return emg_base + emg_active + powerline
    
    @staticmethod
    def prepare_dataset(segments, labels):
        return np.array(segments), np.array(labels)

    @staticmethod
    def train_classifier(X, y):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ])
        pipeline.fit(X, y)
        return pipeline
    
    @staticmethod
    def extract_segment_features(signal, fs, window_size=0.25):
        step = int(window_size * fs)
        features = []

        for i in range(0, len(signal) - step, step):
            seg = signal[i:i+step]
            freqs = np.fft.rfftfreq(len(seg), 1/fs)
            spectrum = np.abs(np.fft.rfft(seg))

            features.append([
                rms(seg),
                mean_frequency(freqs, spectrum),
                median_frequency(freqs, spectrum)
            ])
        return np.array(features)

    
    def plot_time_domain(self, t, raw, filtered):
        plt.figure(figsize=(12, 4), facecolor="black")
        ax = plt.gca()
        ax.set_facecolor("black")

        # GLOW
        plt.plot(t, raw, color="#ffaa00", linewidth=4, alpha=0.15)
        plt.plot(t, raw, color="#ffaa00", alpha=0.6, label="EMG măsurat")

        # GLOW FILTERED
        plt.plot(t, filtered, color="#00ffd5", linewidth=4, alpha=0.2)
        plt.plot(t, filtered, color="#00ffd5", linewidth=1.8, label="EMG filtrat")

        plt.legend(facecolor="black", edgecolor="white", labelcolor="white")
        plt.title("EMG - Domeniul timp", color="white")

        ax.tick_params(color="white")
        ax.spines[:].set_color("white")

        plt.xlabel("Timp [s]", color="white")
        plt.ylabel("Amplitudine", color="white")
        plt.grid(alpha=0.15)
        plt.tight_layout()
        plt.show()

    def plot_freq_domain(self, freqs, fft_raw, fft_filt, fmax=500):
        plt.figure(figsize=(12, 4), facecolor="black")
        ax = plt.gca()
        ax.set_facecolor("black")

        # Glow FFT brut
        plt.plot(freqs, fft_raw, color="#ff8800", linewidth=4, alpha=0.15)
        plt.plot(freqs, fft_raw, color="#ff8800", alpha=0.6, label="FFT EMG măsurat")

        # Glow FFT filtrat
        plt.plot(freqs, fft_filt, color="#00ffd5", linewidth=4, alpha=0.2)
        plt.plot(freqs, fft_filt, color="#00ffd5", linewidth=1.8, label="FFT EMG filtrat")

        ax.tick_params(colors="white")
        ax.spines[:].set_color("white")

        plt.xlim(0, fmax)
        plt.legend(facecolor="black", edgecolor="white", labelcolor="white")
        plt.title("EMG – domeniul frecvență", color="white")
        plt.xlabel("Frecvență [Hz]", color="white")
        plt.ylabel("Magnitudine", color="white")
        plt.tight_layout()
        plt.show()