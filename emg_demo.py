from biosignals import EMGSignal
from utils import bandpass, rms, fft_analysis
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

fs = 2000
duration = 2.0

emg = EMGSignal(fs, duration)
t = emg.generate_time_vector()
activation = emg.generate_activ()
fatigue = emg.generate_fatigue_profile(t)

# Semnal
emg_signal = emg.generate_emg_signal(activation)
emg_filt = bandpass(emg_signal, 20, 450, fs)

emg_fatigue = emg.generate_emg_with_fatigue(t, activation, fatigue)
emg_fatigue_filt = bandpass(emg_fatigue, 20, 450, fs)


# Timp
emg.plot_time_domain(t, emg_signal, emg_filt)
emg.plot_time_domain(t, emg_fatigue, emg_fatigue_filt)

# FFT
freqs, fft = fft_analysis(emg_signal, fs)
_, fft_fatigue = fft_analysis(emg_fatigue_filt, fs)

emg.plot_freq_domain(freqs, fft, fft_fatigue)




# ========================
# Extragere ML
# ========================

X_normal = emg.extract_segment_features(emg_filt, fs)
X_fatigue = emg.extract_segment_features(emg_fatigue, fs)

y_normal = np.zeros(len(X_normal))
y_fatigue = np.ones(len(X_fatigue))

X = np.vstack((X_normal, X_fatigue))
y = np.hstack((y_normal, y_fatigue))

# =========================
# Antrenare clasificator
# =========================
model = emg.train_classifier(X, y)


# =========================
# Evaluare
# =========================
y_pred = model.predict(X)

print("=== Classification report ===")
print(classification_report(y, y_pred, target_names=["Normal", "Fatigue"]))

print("=== Confusion matrix ===")
print(confusion_matrix(y, y_pred))


# =========================
# Vizualizare feature space
# =========================
plt.figure(figsize=(6, 5))
plt.scatter(X_normal[:, 1], X_normal[:, 2], label="Normal", alpha=0.7)
plt.scatter(X_fatigue[:, 1], X_fatigue[:, 2], label="Fatigue", alpha=0.7)
plt.xlabel("Mean Frequency [Hz]")
plt.ylabel("Median Frequency [Hz]")
plt.title("Separare EMG Normal vs Oboseală")
plt.legend()
plt.tight_layout()
plt.show()