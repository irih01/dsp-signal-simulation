from industrial import EMCSignal, EnergySignal

def demo_energy():
    print("=== DEMO Domeniul Energetic ===")

    energy = EnergySignal(fs=5000, duration=1.0, f0=50)

    # 1. Semnal ideal
    clean = energy.generate_clean(amplitude=1.0)

    # 2. Semnal distorsionat (armonici)
    distorted = energy.add_harmonics(clean)

    # 3. Filtrare
    bp = energy.bandpass(distorted)
    filtered = energy.notch_filter(bp)

    # 4. Metrici
    thd_before = energy.compute_thd(distorted)
    thd_after = energy.compute_thd(filtered)


    # 5. Plot
    energy.plot(clean, distorted, filtered)
    energy.plot_db(distorted, filtered)



def demo_emc():
    print("=== DEMO Domeniul EMC / EMI ===")

    emc = EMCSignal(fs=100_000, duration=0.02)

    # 1. Semnal util curat
    clean = emc.generate_ramp_singal()

    # 2. Interferență electronică de putere (PWM)
    with_pwm = emc.add_pwm(
        clean,
        f_pwm=20_000,
        duty=0.4,
        amp=0.3
    )

    # 3. Interferență EMI impulsivă
    disturbed = emc.add_emi_impulses(
        with_pwm,
        n_pulses=20,
        width_us=2,
        amp=0.5
    )

    # 4. Filtrare
    filtered = emc.lowpass(disturbed, cutoff=100)

    # 5. Metrici
    snr_before = emc.compute_snr(clean, disturbed)
    snr_after = emc.compute_snr(clean, filtered)

    print(f"SNR înainte de filtrare: {snr_before:.2f} dB")
    print(f"SNR după filtrare:     {snr_after:.2f} dB")

    # 6. Plot
    emc.plot(clean, disturbed, filtered)
    emc.plot_db(clean, disturbed, filtered)



if __name__ == "__main__":
    demo_energy()
    demo_emc()