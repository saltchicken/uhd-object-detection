import signal
import numpy as np
import argparse

class SignalHandler:
    """
    Replaces the global RUNNING variable and handler function in all scripts.
    """
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.handler)
    
    def handler(self, signum, frame):
        print("\n--> Signal caught. Shutting down...")
        self.running = False


def get_standard_args(description, default_freq=915e6, default_rate=1e6, default_gain=60):
    """
    Allows runtime configuration: python app.py --freq 915e6 --gain 70
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--freq", type=float, default=default_freq, help="Center Frequency (Hz)")
    parser.add_argument("--rate", type=float, default=default_rate, help="Sample Rate (Hz)")
    parser.add_argument("--gain", type=float, default=default_gain, help="RX/TX Gain (dB)")
    return parser.parse_args()

def generate_chirp_probe(length):
    t = np.arange(length)
    k = 1.0 
    chirp = np.exp(1j * np.pi * k * (t**2 / length))
    window = np.hanning(length)
    return (chirp * window).astype(np.complex64) * 0.7


def correlate_and_detect(rx_chunk, probe_sequence):
    """
    Consolidates the correlation, magnitude, and SNR calculation 
    used in channel_sounding, csi_analysis, and object_detection.
    """
    correlation = np.correlate(rx_chunk, probe_sequence, mode='valid')
    mag = np.abs(correlation)
    peak_idx = np.argmax(mag)
    peak_val = mag[peak_idx]
    
    # Estimate noise floor (simple approach: look before the peak)
    if peak_idx > 20:
        noise_region = mag[:peak_idx-10]
        noise_floor = np.mean(noise_region) if len(noise_region) > 0 else 1e-9
    else:
        noise_floor = 1e-9
        
    snr_linear = peak_val / (noise_floor + 1e-12)
    snr_db = 10 * np.log10(snr_linear)
    
    return {
        "correlation": correlation,
        "mag": mag,
        "peak_idx": peak_idx,
        "peak_val": peak_val,
        "snr_db": snr_db,
        "noise_floor": noise_floor
    }

def calculate_csi_metrics(cir_window, sample_rate):
    pdp = np.abs(cir_window)**2
    thresh = np.max(pdp) * 0.1
    valid_indices = np.where(pdp > thresh)[0]
    
    if len(valid_indices) < 2: 
        rms_delay = 0.0
        coherence_bw = sample_rate 
    else:
        first_path = valid_indices[0]
        pdp_clean = pdp[valid_indices]
        delays_sec = (valid_indices - first_path) / sample_rate
        total_power = np.sum(pdp_clean)
        mean_delay = np.sum(pdp_clean * delays_sec) / total_power
        sq_delay_error = (delays_sec - mean_delay)**2
        rms_delay = np.sqrt(np.sum(pdp_clean * sq_delay_error) / total_power)
        
        if rms_delay > 1e-12:
            coherence_bw = 1.0 / (5.0 * rms_delay)
        else:
            coherence_bw = sample_rate

    cfr_complex = np.fft.fftshift(np.fft.fft(cir_window))
    cfr_mag_linear = np.abs(cfr_complex)
    cfr_mag_db = 20 * np.log10(cfr_mag_linear + 1e-12)
    
    return {
        "rms_delay_us": rms_delay * 1e6, 
        "coherence_bw_khz": coherence_bw / 1e3, 
        "cfr_db": cfr_mag_db,
        "cfr_linear": cfr_mag_linear,
        "pdp": pdp 
    }








