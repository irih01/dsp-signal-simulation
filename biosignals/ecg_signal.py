from scipy.signal import butter, filtfilt, find_peaks

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import fft_analysis, bandpass
import matplotlib.pyplot as plt
import numpy as np



class ECGSignal:
    """
    Pipeline ECG
    - simulare
    - zgomot
    - filtrare
    - detectie QRS
    - extragere caracteristici
    - clasificare ML
    """

    def __init__(self, fs: int=1000, duration: float=5, heart_rate=72):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, fs * duration, endpoint=False)
        self.hr = heart_rate
        self.T = 60 / heart_rate
        self.model = self._build_model()
 

    # ==================================
    #   Simulare ECG (P-Q-R-S-T)
    # ==================================
    @staticmethod
    def ecg_waveform(t):
        return (
            0.1 * np.exp(-((t - 0.20) / 0.025)**2)   +   # P
           -0.15 * np.exp(-((t - 0.38) / 0.010)**2)  +   # Q
            1.00 * np.exp(-((t - 0.40) / 0.012)**2)  +   # R
           -0.25 * np.exp(-((t - 0.43) / 0.010)**2)  +   # S
            0.30 * np.exp(-((t - 0.60) / 0.040)**2)      # T
        )
    


    def gen_ecg(self):
        ecg = np.zeros_like(self.t)
        for beat in np.arange(0, self.duration, self.T):
            ecg += self.ecg_waveform(self.t - beat)
        return ecg
    


    # =================================================
    #                   Generare ECG 
    # ==================================================
    def generate_ecg(self, heart_rate=72, arrhytmia=False):
        ecg = np.zeros_like(self.t)
        time = 0

        while time < self.t[-1]:
            # perioada unei batai
            rr = 60 / heart_rate 
            if arrhytmia:
                rr += np.random.uniform(-0.15, 0.15)

            ecg += self.ecg_waveform(self.t - time)
            time += rr
        return ecg
    
    
    # ========================================
    #    ECG cu ritm variabil (aritmie)
    # ========================================
    def generate_ecg_arrhytmia(self, base_hr=70):
        ecg = np.zeros_like(self.t)
        time = 0

        while time < self.t[-1]:
            # variatie RR (aritmie sinusala)
            rr = 60 / base_hr + np.random.uniform(-0.15, 0.15)

            ecg += self.ecg_waveform(self.t - time)
            time += rr
        return ecg
    
    def generate_ecg_type(self, kind="normal", base_hr=70):
        ecg = np.zeros_like(self.t)
        time = 0

        while time < self.t[-1]:
            rr = 60 / base_hr

            # -------------------------
            if kind == "normal":
                pass

            elif kind == "sinus_arrhythmia":
                rr += np.random.uniform(-0.15, 0.15)

            elif kind == "tachycardia":
                rr = 60 / np.random.uniform(100, 140)

            elif kind == "bradycardia":
                rr = 60 / np.random.uniform(40, 55)

            elif kind == "atrial_fibrillation":
                rr += np.random.uniform(-0.35, 0.35)

            elif kind == "premature_beat":
                if np.random.rand() < 0.15:
                    rr *= 0.5   # bătaie prematură

            elif kind == "missed_beat":
                if np.random.rand() < 0.1:
                    time += rr * 2
                    continue

            # -------------------------
            ecg += self.ecg_waveform(self.t - time)
            time += rr

        return ecg





    # ==============================================
    #        Zgomot + interferenta 50 Hz
    # ==============================================
    def add_noise(self, ecg, powerline_freq=50, powerline_amp = 0.2, show_powerline=False):
        powerline = powerline_amp * np.sin(2 * np.pi * powerline_freq * self.t)
        if show_powerline:
            plt.figure(figsize=(12,4))

            # glow
            plt.plot(self.t,powerline,color="#ff4444",linewidth=4,alpha=0.15)
            # linie principala
            plt.plot(self.t,powerline,color="#ff4444",linewidth=1.5,label=f"Interferență rețea {powerline_freq} Hz")

            plt.xlim(0, 0.1)  # zoom ca sa se vada clar frecventa
            plt.xlabel("Timp [s]")
            plt.ylabel("Amplitudine")
            plt.title("Semnal de interferență – powerline 50 Hz")
            plt.grid(alpha=0.15)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return ecg + powerline 
    
    def add_motion_artifact(self, ecg):
        drift = 0.3 * np.sin(2 * np.pi * 0.2 * self.t)
        emg = 0.05 * np.random.randn(len(self.t))
        return ecg + emg + drift


    # ========================================================
    #       Filtrare ECG (trece-banda 0.5 - 40 Hz)
    #           < 0.5 Hz - drift bazal
    #              40 Hz - zgomot 
    # ========================================================
    def ecg_bandpass(self, data, low: float=0.5, high=40, order=4):
        """
        Filtrare ECG (trece-banda 0.5 - 40 Hz)
        
        :param data: semnal
        :param low: < 0.5 Hz - drift bazal
        :param high: 40 Hz - zgomot
    
        """
        nyq = self.fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data)



    # =================================
    #  Trece-banda QRS (5 - 15 Hz) 
    # =================================
    def bandpass_qrs(self, ecg):
        return self.ecg_bandpass(ecg, 5, 15, order=3) 
        


    # ====================================================
    #                   Pan-Tompkins 
    # =====================================================
    def pan_tompkins(self, ecg):
        filtered = self.bandpass_qrs(ecg)

        derivatie = np.diff(filtered, prepend=filtered[0])
        squared = derivatie ** 2

        window = int(0.15 * self.fs)
        integrata = np.convolve(squared, np.ones(window)/window, mode='same')

        # Prag adaptiv
        prag = 0.5 * np.mean(integrata)
        varfuri, _ = find_peaks(integrata, height=prag, distance=0.3*self.fs)

        return varfuri, integrata, filtered



    # ===================================
    #                HRV
    # ===================================
    def extract_hrv(self, r_peaks):
        if len(r_peaks) < 2:
            return [0.0, 0.0, 0.0, 0]
        
        rr = np.diff(r_peaks) / self.fs
        if len(rr) < 2:
            return [0, 0, 0, len(rr)]

        return [
            np.mean(rr),                        # RR mediu
            np.std(rr),                         # SDNN
            np.sqrt(np.mean(np.diff(rr)**2)),   # RMSSD
            len(rr)                             # Nr batai
        ]




    # ====================================================
    #                   Spectral Features
    # ====================================================
    def spectral_features(self, signal):
        freqs, spectrum = fft_analysis(signal, self.fs)

        low = np.sum(spectrum[(freqs >= 0.5) & (freqs < 5)])
        mid = np.sum(spectrum[(freqs >= 5) & (freqs < 15)])
        high = np.sum(spectrum[(freqs >= 15) & (freqs < 40)])

        return [low, mid, high]
    


    # =========================================
    #                   ALL
    # =========================================
    def extract_features(self, ecg):
        ecg_filt = bandpass(ecg, 0.5, 40, self.fs)
        r_peaks, _, _ = self.pan_tompkins(ecg_filt)

        if len(r_peaks) < 3:
            return None
        
        hrv = self.extract_hrv(r_peaks)
        spec = self.spectral_features(ecg_filt)

        feautures = np.array(hrv + spec)
        if np.any(np.isnan(feautures)):
            return None

        return feautures
    



    def detect_arr_beats(self, r_peaks, k=1.5, show=True):
        rr = np.diff(r_peaks) / self.fs
        rr_mean = np.mean(rr)
        rr_std = np.std(rr)

        arr_idx = np.where(np.abs(rr - rr_mean) > k * rr_std)[0]
        if show:
            plt.figure(figsize=(10,3))
            plt.plot(rr, marker="o")
            plt.axhline(rr_mean, color="g", linestyle="--", label="RR mediu")
            plt.axhline(rr_mean + 1.5*rr_std, color="r", linestyle="--")
            plt.axhline(rr_mean - 1.5*rr_std, color="r", linestyle="--")
            plt.title("Variabilitatea intervalelor RR (HRV)")
            plt.xlabel("Index bătaie")
            plt.ylabel("RR [s]")
            plt.legend()
            plt.grid()
            plt.show()

        return r_peaks[arr_idx + 1]


    # ========================================================================
    #                               ML
    # =========================================================================
    def slide_win_prediction(self, ecg, model, window_sec=2.0, step_sec=0.5):
        win_len = int(window_sec * self.fs)
        step_len = int(step_sec * self.fs)

        predictions = []

        for start in range(0, len(ecg) - win_len, step_len):
            end = start + win_len

            segment = ecg[start:end]
            feautures = self.extract_features(segment)

            if feautures is None:
                continue

            pred = model.predict([feautures])[0]

            predictions.append({
                "start": self.t[start],
                "end": self.t[end],
                "label": pred
            })

        return predictions
    



    def generate_multiclas(self, n_per_class=50):
        X, y = [], []

        classes = {
            "normal": 0,
            "sinus_arrhythmia": 1,
            "tachycardia": 2,
            "bradycardia": 3,
            "atrial_fibrillation": 4,
            "premature_beat": 5
        }

        for label, cls in classes.items():
            for _ in range(n_per_class):
                ecg = self.generate_ecg_type(label)
                ecg = self.add_noise(ecg)

                features = self.extract_features(ecg)
                if features is None:
                    continue

                if not np.any(np.isnan(features)):
                    X.append(features)
                    y.append(cls)

        return np.array(X), np.array(y), classes




    def generate_dataset(self, n_samples=100):
        X = []
        y = []

        for _ in range(n_samples):
            # ECG normal
            ecg_n = self.add_noise(self.generate_ecg(self.hr, False))
            X.append(self.extract_features(ecg_n))
            y.append(0)

            # ECG aritmic
            ecg_a = self.add_noise(self.generate_ecg(self.hr, True))
            X.append(self.extract_features(ecg_a))
            y.append(1)

        return np.array(X), np.array(y)


    def _build_model(self):
        return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    multi_class="multinomial",
                    max_iter=2000
                ))
            ])
    
    


    # ==============================================================
    #                           PLOTS
    # ==============================================================
    def plot(self, raw, filtered, freqs_raw, fft_raw, fft_filt):
        # Domeniul timp
        plt.subplot(2,1,1)
        plt.plot(self.t, raw, label="ECG măsurat", alpha=0.5)
        plt.plot(self.t, filtered, label="ECG filtrat", linewidth=2)
        plt.xlim(0, 3)
        plt.legend()
        plt.title("ECG realist – domeniul timp")
        plt.xlabel("Timp [s]")
        plt.ylabel("Amplitudine")

        # Domeniul frecvent
        plt.subplot(2,1,2)
        plt.plot(freqs_raw, fft_raw, label="FFT măsurat", alpha=0.5)
        plt.plot(freqs_raw, fft_filt, label="FFT filtrat", linewidth=2)
        plt.xlim(0, 100)
        plt.legend()
        plt.title("ECG realist – domeniul frecvență")
        plt.xlabel("Frecvență [Hz]")

        plt.tight_layout()
        plt.show()


    # Varf
    def plot_peak(self, r_peaks, ecg_filtered):
        plt.figure(figsize=(12,4))
        plt.plot(self.t, ecg_filtered, label="ECG filtrat")
        plt.plot(self.t[r_peaks], ecg_filtered[r_peaks], "ro", label="R-peaks")
        plt.xlim(0, 3)
        plt.legend()
        plt.title("Detecția vârfurilor R (Pan–Tompkins)")
        plt.xlabel("Timp [s]")
        plt.ylabel("Amplitudine")
        plt.show()
    
    # Aritmia
    def plot_aritmia(self, ecg_filtered, ecg_tachy, ecg_brady):
        plt.figure(figsize=(12,6))

        plt.subplot(3,1,1)
        plt.plot(self.t, ecg_filtered)
        plt.title("ECG normal")

        plt.subplot(3,1,2)
        plt.plot(self.t, ecg_tachy)
        plt.title("ECG – tahicardie")

        plt.subplot(3,1,3)
        plt.plot(self.t, ecg_brady)
        plt.title("ECG – bradicardie")
        plt.tight_layout()
        plt.show()


    def plot_detects(self, r_peaks, arr_peaks, ecg_filtered, ecg_tachy, ecg_brady):
        plt.figure(figsize=(12,6))

        plt.subplot(3,1,1)
        plt.plot(self.t, ecg_filtered)
        plt.title("ECG normal")

        plt.subplot(3,1,2)
        plt.plot(self.t, ecg_tachy)
        plt.title("ECG – tahicardie")

        plt.subplot(3,1,3)
        plt.plot(self.t, ecg_brady)
        plt.title("ECG – bradicardie")

        plt.tight_layout()
        plt.show()