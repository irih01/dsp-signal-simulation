import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.io.wavfile import write


def add_beat_to_audio(
    audio_signal,
    audio_fs,
    bpm=72,
    tone_freq=220,
    beat_duration=0.05,
    beat_amp=0.4
):
    """
    Adauga un beat (nota sinusoidala) peste un semnal audio.

    bpm           : batai pe minut
    tone_freq     : frecventa notei (Hz)
    beat_duration : durata beat-ului (sec)
    """

    beat_signal = np.zeros_like(audio_signal)

    samples_per_beat = int((60 / bpm) * audio_fs)
    beat_len = int(beat_duration * audio_fs)

    for i in range(0, len(audio_signal), samples_per_beat):
        t = np.linspace(0, beat_duration, beat_len, endpoint=False)
        beat = beat_amp * np.sin(2 * np.pi * tone_freq * t)

        end = min(i + beat_len, len(audio_signal))
        beat_signal[i:end] = beat[:end - i]

    return audio_signal + beat_signal



def sonify_ecg(
    ecg_signal,
    fs_ecg,
    filename,
    bpm=72,
    tone_freq=220,
    audio_fs=44100
):
    # Offset + normalizare
    ecg_signal = ecg_signal - np.mean(ecg_signal)
    ecg_norm = ecg_signal / np.max(np.abs(ecg_signal))

    # Resampling
    duration = len(ecg_norm) / fs_ecg
    t_ecg = np.linspace(0, duration, len(ecg_norm))
    t_audio = np.linspace(0, duration, int(audio_fs * duration))
    ecg_audio = np.interp(t_audio, t_ecg, ecg_norm)

    # Adaugare beat
    ecg_with_beat = add_beat_to_audio(
        ecg_audio,
        audio_fs,
        bpm=bpm,
        tone_freq=tone_freq
    )

    # Normalizare finala
    ecg_with_beat /= np.max(np.abs(ecg_with_beat))
    ecg_int16 = np.int16(ecg_with_beat * 32767)

    write(filename, audio_fs, ecg_int16)
    print(f"[OK] ECG + beat salvat: {filename}")


def beat_from_r_peaks(
    audio_signal,
    r_peaks,
    fs_ecg,
    audio_fs,
    freq=60,          # kick low
    decay=0.04,
    amp=0.8
):
    beat_signal = np.zeros_like(audio_signal)
    ratio = audio_fs / fs_ecg

    for rp in r_peaks:
        idx = int(rp * ratio)
        length = int(decay * audio_fs)

        t = np.linspace(0, decay, length, endpoint=False)
        envelope = np.exp(-8 * t)
        beat = amp * envelope * np.sin(2 * np.pi * freq * t)

        end = min(idx + length, len(audio_signal))
        beat_signal[idx:end] += beat[:end - idx]

    return audio_signal + beat_signal


def ecg_to_melody(
    ecg_signal,
    fs_ecg,
    audio_fs,
    scale=[220, 247, 262, 294, 330, 349, 392],  # A minor-ish
    amp=0.3
):
    duration = len(ecg_signal) / fs_ecg
    t_audio = np.linspace(0, duration, int(audio_fs * duration))
    t_ecg = np.linspace(0, duration, len(ecg_signal))

    ecg_interp = np.interp(t_audio, t_ecg, ecg_signal)
    ecg_norm = (ecg_interp - np.min(ecg_interp)) / np.ptp(ecg_interp)

    melody = np.zeros_like(ecg_norm)

    note_len = int(0.12 * audio_fs)

    for i in range(0, len(ecg_norm), note_len):
        val = ecg_norm[i]
        freq = scale[int(val * (len(scale) - 1))]

        t = np.linspace(0, note_len / audio_fs, note_len, endpoint=False)
        tone = amp * np.sin(2 * np.pi * freq * t)

        end = min(i + note_len, len(melody))
        melody[i:end] += tone[:end - i]

    return melody



def wobble_bass(
    duration,
    audio_fs,
    base_freq=55,        # bass fundamental
    wobble_rate=4,       # Hz (LFO)
    amp=0.8
):
    t = np.linspace(0, duration, int(audio_fs * duration), endpoint=False)

    # LFO pentru wobble
    lfo = 0.5 * (1 + np.sin(2 * np.pi * wobble_rate * t))

    # Bass fundamental + armonici
    bass = np.sin(2 * np.pi * base_freq * t)
    bass += 0.8 * np.sin(2 * np.pi * base_freq * 2 * t)
    bass += 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)

    return amp * lfo * bass


def sidechain(signal, r_peaks, fs_ecg, audio_fs, depth=0.6):
    env = np.ones_like(signal)
    ratio = audio_fs / fs_ecg

    for rp in r_peaks:
        idx = int(rp * ratio)
        length = int(0.12 * audio_fs)

        if idx + length < len(env):
            env[idx:idx+length] *= np.linspace(1-depth, 1, length)

    return signal * env

def split_bass(duration, audio_fs):
    t = np.linspace(0, duration, int(audio_fs * duration), endpoint=False)

    # SUB (mono, curat)
    sub = np.sin(2 * np.pi * 40 * t)

    # MID-BASS (grit)
    mid = np.sin(2 * np.pi * 90 * t)
    mid += 0.6 * np.sin(2 * np.pi * 180 * t)

    return sub, mid

def growl_bass(ecg_resampled, t, base_freq=55):
    # FM modulation depth from ECG
    mod = 30 * ecg_resampled

    carrier = np.sin(2 * np.pi * base_freq * t + mod)
    harmonic = np.sin(2 * np.pi * base_freq * 2 * t + 0.5 * mod)

    growl = carrier + 0.7 * harmonic

    # Distorsie agresiva
    growl = np.tanh(4.5 * growl)

    return growl


def ecg_snare(r_peaks, length, fs, audio_fs):
    snare = np.zeros(length)
    decay = np.exp(-np.linspace(0, 1, int(0.08 * audio_fs)) * 25)

    noise = np.random.randn(len(decay)) * decay

    for r in r_peaks:
        idx = int(r / fs * audio_fs) + int(0.04 * audio_fs)
        if idx + len(decay) < length:
            snare[idx:idx+len(decay)] += noise

    return snare

def ecg_riser(r_peaks, length, fs, audio_fs):
    riser = np.zeros(length)

    for i in range(1, len(r_peaks)):
        start = int(r_peaks[i-1] / fs * audio_fs)
        end = int(r_peaks[i] / fs * audio_fs)

        dur = end - start
        if dur < audio_fs * 0.2:
            continue

        t = np.linspace(0, 1, dur)
        sweep = np.sin(2 * np.pi * (200 + 1200 * t) * t)
        env = t**2

        riser[start:end] += sweep * env * 0.4

    return riser


def ecg_impact(r_peaks, length, fs, audio_fs):
    impact = np.zeros(length)
    decay = np.exp(-np.linspace(0, 1, int(0.12 * audio_fs)) * 18)

    for r in r_peaks:
        idx = int(r / fs * audio_fs)
        if idx + len(decay) < length:
            impact[idx:idx+len(decay)] += decay * np.sin(2*np.pi*60*np.linspace(0,1,len(decay)))

    return impact
