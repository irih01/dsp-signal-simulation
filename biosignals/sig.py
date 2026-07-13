import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write

# Parametri generali
fs = 1000            # frecventa de esantionare [Hz]
T = 5                # durata semnalului [s]
t = np.linspace(0, T, fs*T, endpoint=False)

# Semnal simplificat (componenta joasa)
signal_ambint = 0.8 * np.sin(2 * np.pi * 1.2 * t)   

# Semnal bioelectric lent (planta)
plant_signal = 0.3 * np.sin(2 * np.pi * 0.1 * t)

# Semnal electromagnetic util total
signal_clean = signal_ambint + plant_signal

# Zgomot electromagnetic de fond
noise_white = 0.2 * np.random.randn(len(t))

# Interferenta retelei electrice (50 Hz)
powerline_noise = 0.4 * np.sin(2 * np.pi * 50 * t)



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




plt.figure()
plt.plot(t, powerline_noise)
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")
plt.title("Zgomot rețea electrică (domeniul timp)")
plt.show()


# Semnal masurat (realist)
signal_measured = plant_signal + noise_white + powerline_noise

plt.figure()
plt.plot(t, signal_measured)
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")
plt.title("Semnal electromagnetic măsurat virtual (domeniul timp)")
plt.show()


# FFT
N = len(signal_measured)
fft_vals = np.fft.fft(signal_measured)
fft_freq = np.fft.fftfreq(N, 1/fs)

# Spectru de amplitudine (jumatate pozitiva)
positive_freqs = fft_freq[:N//2]
magnitude = np.abs(fft_vals[:N//2]) / N

plt.figure()
plt.plot(positive_freqs, magnitude)
plt.xlabel("Frecvență [Hz]")
plt.ylabel("Amplitudine")
plt.title("Spectrul de frecvență al semnalului electromagnetic")
plt.xlim(0, 100)
plt.show()



def bandstop_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data)

# Eliminare interferenta 50 Hz
signal_filtered = bandstop_filter(signal_measured, 48, 52, fs)
fft_filtered = np.fft.fft(signal_filtered)
magnitude_filtered = np.abs(fft_filtered[:N//2]) / N

plt.figure()
plt.plot(positive_freqs, magnitude_filtered)
plt.xlabel("Frecvență [Hz]")
plt.ylabel("Amplitudine")
plt.title("Spectru după reducerea interferenței de 50 Hz")
plt.xlim(0, 100)
plt.show()


'''Analiza în domeniul frecvență evidențiază prezența componentelor
 joase asociate semnalului biomedical și variațiilor bioelectrice lente, 
 precum și componenta dominantă de 50 Hz corespunzătoare interferenței rețelei 
 electrice. Aplicarea filtrării de tip band-stop conduce la reducerea semnificativă a interferenței,
 îmbunătățind raportul semnal–zgomot și facilitând interpretarea componentelor electromagnetice de interes.'''

# # Normalizare semnal filtrat
# signal_audio = signal_filtered / np.max(np.abs(signal_filtered))

# # Scalare pentru audio (int16)
# signal_audio_int16 = np.int16(signal_audio * 32767)

# audio_fs = 44100  # frecventa audio standard

# # Resampling simplu prin interpolare
# t_audio = np.linspace(0, T, audio_fs * T, endpoint=False)
# signal_audio_resampled = np.interp(t_audio, t, signal_audio)

# signal_audio_resampled_int16 = np.int16(signal_audio_resampled * 32767)

# write("sonificare_semnal_em.wav", audio_fs, signal_audio_resampled_int16)

# fft_audio = np.fft.fft(signal_audio_resampled)
# freq_audio = np.fft.fftfreq(len(fft_audio), 1/audio_fs)

# plt.figure()
# plt.plot(freq_audio[:len(freq_audio)//2], 
#          np.abs(fft_audio[:len(freq_audio)//2]) / len(fft_audio))
# plt.xlabel("Frecvență [Hz]")
# plt.ylabel("Amplitudine")
# plt.title("Spectrul semnalului sonificat")
# plt.xlim(0, 2000)
# plt.show()
