from biosignals import ECGSignal
from utils import fft_analysis,  beat_from_r_peaks, wobble_bass, sidechain, ecg_riser, ecg_impact, ecg_snare, growl_bass
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


def plot_ecg(t, signal, title):
    plt.figure(figsize=(12,4))

    #glow
    plt.plot(t, signal, color="#ffcc00", linewidth=4, alpha=0.15)
    #linia principala
    plt.plot(t, signal, color="#ffcc00", linewidth=1.5)

    plt.xlabel("Timp [s]")
    plt.ylabel("Amplitudine")
    plt.title(title)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()

def plot_fft_ecg(freq, signal, title):
    plt.figure(figsize=(12,4))
    plt.plot(freq, signal, color="#ff8800", linewidth=4, alpha=0.15)
    plt.plot(freq, signal, color="#ff8800", linewidth=1.5)

    plt.xlim(0, 100)
    plt.title(title)
    plt.xlabel("Frecvență [Hz]")
    plt.ylabel("Ampltitudine")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()

def set_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": "#0A0A0A",
        "axes.facecolor": "#0A0A0A",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#646464",
        "font.size": 12
    })

set_dark_theme()


# Parametri generali
fs = 1000            # Hz
duration = 5         # secunde


ecg = ECGSignal(fs, duration)
powerline = 0.2 * np.sin(2 * np.pi * 50 * ecg.t)
freq, pw_fft = fft_analysis(powerline, fs)
plot_fft_ecg(freq,pw_fft, "FFT interferențe rețea")

# ====================================
# 1.          ECG IDEAL
# ====================================
ecg_clean = ecg.gen_ecg()
_, fft_clean = fft_analysis(ecg_clean, fs)


# ===============================================
# 2.   ECG + ZGOMOT LINIE ELECTRICA (50Hz)
# ===============================================
ecg_powerline = ecg.add_noise(ecg_clean, show_powerline=False)
_, fft_powerline = fft_analysis(ecg_powerline, fs)

ecg_pw_filt = ecg.ecg_bandpass(ecg_powerline, 0.5, 40)
qrs_pw = ecg.bandpass_qrs(ecg_pw_filt)


# ================================================
# 3.    ECG + ZGOMOT + ARTEFACT MISCARE
# =================================================
ecg_motion_only = ecg.add_motion_artifact(ecg_clean)
motion_only_freq, fft_motion_only = fft_analysis(ecg_motion_only, fs)

# plot_ecg(ecg.t, ecg_motion_only, "ECG doar cu artefacte de mișcare fără drift")
# plot_fft_ecg(motion_only_freq, fft_motion_only, "FFT ECG daor artefacte de mișcare fără drift")

ecg_motion = ecg.add_motion_artifact(ecg_powerline)
_, fft_motion = fft_analysis(ecg_motion, fs)

# ====================================
# 4.          ECG FILTRAT
# ====================================
ecg_filtered = ecg.ecg_bandpass(ecg_motion, 0.5, 40)
ecg_qrs = ecg.bandpass_qrs(ecg_filtered)

# FFT
freqs, fft_raw = fft_analysis(ecg_motion, fs)
_, fft_filt = fft_analysis(ecg_filtered, fs)

# Varfuri
r_peaks, integrated, ecg_qrs = ecg.pan_tompkins(ecg_filtered)
arr_peaks = ecg.detect_arr_beats(r_peaks, show=False)





# ===================================================================================================================================
#                       ECG MUSIC
# ====================================================================================================================================
def create_music():
    audio_fs = 44100
    duration = ecg.duration
    base_freq = 40

    ecg_fs = ecg.fs

    r_times_sec = r_peaks / ecg_fs          # timp real
    r_audio_idx = (r_times_sec * audio_fs).astype(int)

    DROP_PEAK = 5
    drop_idx = r_audio_idx[DROP_PEAK]


    # ECG textura
    ecg_audio = np.interp(
        np.linspace(0, duration, int(audio_fs * duration)),
        ecg.t,
        ecg_filtered
    )

    #ecg_audio = resample(ecg, int(len(ecg) * audio_fs / fs))
    ecg_audio /= np.max(np.abs(ecg_audio))


    t = np.arange(len(ecg_audio)) / audio_fs

    # Beat din R-peaks (kick biometric)
    kick = beat_from_r_peaks(
        np.zeros_like(ecg_audio),
        r_peaks,
        fs_ecg=fs,
        audio_fs=audio_fs,
        freq=50
    )

    # Bass wobble
    bass = wobble_bass(duration, audio_fs, base_freq=40, wobble_rate=8, amp=1.2)
    # bass += 0.8 * np.sin(2 * np.pi * base_freq * 3 * ecg.t)


    # === STRUCTURA ===
    intro_len = int(1.5 * audio_fs)
    build_len = int(1.0 * audio_fs)

    track = np.zeros_like(ecg_audio)
    track = np.tanh(2.5 * track)


    # INTRO (ECG + reverb vibe)
    track[:intro_len] = 0.4 * ecg_audio[:intro_len]

    # BUILD-UP (pitch rise)
    for i in range(build_len):
        idx = intro_len + i
        if idx >= len(track): break
        track[idx] = ecg_audio[idx] * (i / build_len)

    # sub, mid = split_bass(duration, audio_fs)

    # mid = np.tanh(3.5 * mid)   # distorsie agresiva
    # bass =  0.5 * sub



    # DROP 🔥
    drop_start = intro_len + build_len
    track[drop_start:] = (
        3 * bass[drop_start:] +
        0.6 * kick[drop_start:]
    )

    growl = growl_bass(ecg_audio, t)
    snare = ecg_snare(r_peaks, len(t), fs, audio_fs)
    riser = ecg_riser(r_peaks, len(t), fs, audio_fs)
    impact = ecg_impact(r_peaks, len(t), fs, audio_fs)

    track = (
        1.4 * growl +
        1.0 * bass +
        0.8 * snare +
        1.2 * impact
    )


    # Sidechain pe bass
    track = sidechain(track, r_peaks, fs, audio_fs, depth=0.85)

    # Gain global
    # gain = 0.4 # incearca 2.0 – 4.0
    # track *= gain
    #track = np.tanh(track)

    # Normalizare
    track /= np.max(np.abs(track))



    from scipy.io.wavfile import write
    write("ECG_GROWL.wav", audio_fs, np.int16(track * 32767))
    print("Track creat")



# plot_ecg(ecg.t, ecg_clean, "ECG ideal")
# plot_ecg(ecg.t, ecg_powerline, "ECG cu interferente EM (50 Hz)")
# # plot_ecg(ecg.t, ecg_pw_filt, "ECG dupa filtrare digitala (0.5 - 40 Hz)")
# plot_ecg(ecg.t, ecg_motion, "ECG cu interferente + artefact de miscare")
# plot_ecg(ecg.t, ecg_filtered, "ECG dupa filtrare zgomote")
# #plot_ecg(ecg.t, qrs_pw, "QRS")

# plot_fft_ecg(freqs, fft_clean, "FFT ECG curat")
# plot_fft_ecg(freqs, fft_powerline, "FFT ECG cu interferente EM (50 Hz)")
# # plot_fft_ecg(freqs, fft_raw, "FFT ECG cu artefacte + powerline")
# plot_fft_ecg(freqs, fft_motion, "FFT ECG cu artefacte + powerline")
# plot_fft_ecg(freqs, fft_filt, "FFT ECG dupa filtrare")
# ecg.plot(ecg_motion, ecg_filtered, freqs, fft_raw, fft_filt)
# ecg.plot_peak(r_peaks, ecg_filtered)
# ========================================
# ML
# ========================================
# X, y, classes = ecg.generate_multiclas(n_per_class=80)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# ecg.model.fit(X_train, y_train)
# y_pred = ecg.model.predict(X_test)


# target_names = list(classes.keys())



# test_signals = {
#     "Normal": ecg.generate_ecg_type("normal"),
#     "Sinus": ecg.generate_ecg_type("sinus_arrhytmia"),
#     "Tachycardia": ecg.generate_ecg_type("tachycardia"),
#     "Bradycardia": ecg.generate_ecg_type("bradycardia"),
#     "Atrial fibrillation": ecg.generate_ecg_type("atrial_fibrillation"),
#     "Premature beats": ecg.generate_ecg_type("premature_beat")
# }

# predictions = ecg.slide_win_prediction(
#     ecg_filtered,
#     ecg.model,
#     window_sec=2,
#     step_sec=0.5
# )

# colors = {
#     1: "orange",
#     2: "red",
#     3: "blue",
#     4: "purple",
#     5: "green"
# }


# def test_signal(ecg_obj, signal, title):
#     signal = ecg_obj.add_noise(signal)
#     signal_f = ecg_obj.ecg_bandpass(signal)

#     predictions = ecg_obj.slide_win_prediction(
#         signal_f,
#         ecg_obj.model,
#         window_sec=2,
#         step_sec=0.5
#     )

#     plt.figure(figsize=(12,4))
#     plt.plot(ecg_obj.t, signal_f, label="ECG filtrat")

#     for p in predictions:
#         if p["label"] != 0:
#             plt.axvspan(
#                 p["start"],
#                 p["end"],
#                 color="red",
#                 alpha=0.3
#             )

#     plt.title(f"Detectare ML – {title}")
#     plt.xlabel("Timp [s]")
#     plt.ylabel("Amplitudine")
#     plt.grid()
#     plt.show()

# for name, sig in test_signals.items():
#     test_signal(ecg, sig, name)

