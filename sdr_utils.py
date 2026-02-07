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

def ascii_sparkline(data, width=40):
    if len(data) == 0: return ""
    chunk_size = max(1, len(data) // width)
    reduced = [np.max(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)]
    reduced = reduced[:width]
    chars = " _.-=oO#"
    m = np.max(reduced)
    if m == 0: return "_" * width
    line = ""
    for val in reduced:
        idx = int((val / m) * (len(chars) - 1))
        line += chars[idx]
    return line

def ascii_bar_chart(data, width=40):
    if len(data) == 0: return ""
    d_min, d_max = np.min(data), np.max(data)
    if d_max == d_min: norm_data = np.zeros_like(data)
    else: norm_data = (data - d_min) / (d_max - d_min)
    chunk = len(norm_data) // width
    if chunk < 1: chunk = 1
    resampled = [np.mean(norm_data[i:i+chunk]) for i in range(0, len(norm_data), chunk)][:width]
    chars = "  ▂▃▄▅▆▇█"
    line = ""
    for val in resampled:
        idx = int(val * (len(chars) - 1))
        line += chars[idx]
    return line

def ascii_compass(angle_deg):
    width = 50
    center_idx = width // 2
    norm = (angle_deg + 90) / 180
    pos = int(norm * (width - 1))
    chars = ['-'] * width
    chars[center_idx] = '|'
    pos = max(0, min(width-1, pos))
    chars[pos] = 'O'
    return "".join(chars)

def ascii_dual_gauge(value, max_val, width=40):
    """
    ‼️ Added for Doppler Radar.
    Visualizes positive (Approaching) vs negative (Receding) values.
    [-10 .... 0 .... +10]
    """
    mid = width // 2
    chars = [' '] * width
    chars[mid] = '|'
    
    # Normalize value to fit half-width
    norm_val = value / max_val
    bar_len = int(abs(norm_val) * (mid - 1))
    bar_len = min(mid - 1, bar_len)
    
    if value > 0:
        # Fill right side
        for i in range(mid + 1, mid + 1 + bar_len):
            chars[i] = '>'
    elif value < 0:
        # Fill left side
        for i in range(mid - 1, mid - 1 - bar_len, -1):
            chars[i] = '<'
            
    return "".join(chars)

def ascii_density_map(data, min_db=-90, max_db=-30):
    """
    Maps an array of dB values to density characters.
    """
    # Characters ordered by increasing density/visual weight
    chars = " .:-=+*#%@"
    line = ""
    for val in data:
        if val < min_db:
            norm = 0.0
        elif val > max_db:
            norm = 1.0
        else:
            norm = (val - min_db) / (max_db - min_db)
            
        idx = int(norm * (len(chars) - 1))
        line += chars[idx]
    return line


def calculate_steering_phase(angle_deg, frequency, spacing_meters):
    """
    Calculates the required phase shift (in radians) for a uniform linear array
    to steer the beam to 'angle_deg' (where 0 is boresight).
    """
    wavelength = 3e8 / frequency
    # Path difference d * sin(theta)
    # Phase = 2pi * (d * sin(theta) / lambda)
    theta_rad = np.radians(angle_deg)
    phase_rad = (2 * np.pi * spacing_meters * np.sin(theta_rad)) / wavelength
    return phase_rad


def apply_beamforming(ch0_samples, ch1_samples, steering_phase_rad, cal_offset_rad=0.0):
    """
    Combines two channels constructively for a given steering angle.
    ch0 is reference. ch1 is shifted to align with ch0.
    """
    # Total phase correction = Steering Phase + Calibration Offset
    # If Ch1 leads Ch0 by 'phi', we multiply Ch1 by exp(-j*phi) to delay it back to sync.
    total_phase = steering_phase_rad + cal_offset_rad
    
    weight = np.exp(-1j * total_phase)
    
    # Beamformed Signal = Ch0 + (Ch1 * Weight)
    # Scale by 0.707 to keep power normalized relative to single element peak potential
    beamformed = (ch0_samples + (ch1_samples * weight)) * 0.707
    
    return beamformed
