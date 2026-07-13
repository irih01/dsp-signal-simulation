import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.io.wavfile import write

def rms(signal):
    return np.sqrt(np.mean(signal ** 2))


# =========================================
#               Analiza FFT
# =========================================
def fft_analysis(signal, fs):
    """
    0.5 - 5 Hz ritm cardiac
    5 - 15 Hz QRS
    15- 40 Hz zgomot muscular
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    spectrum = np.abs(np.fft.rfft(signal))
    return freqs, spectrum



def bandpass(data, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)



# ==============================================
#        Zgomot + interferenta 50 Hz
# ==============================================
def add_noise(signal, t, powerline_freq=50, powerline_amp=0.2):
    powerline = powerline_amp * np.sin(2 * np.pi * powerline_freq * t)
    return signal + powerline 

def mean_frequency(freqs, spectrum):
    return np.sum(freqs * spectrum) / np.sum(spectrum)

def median_frequency(freqs, spectrum):
    cumulative = np.cumsum(spectrum)
    return freqs[np.where(cumulative >= cumulative[-1] / 2)[0][0]]

def notch_filter(data, freq, fs, Q=30):
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, data)

