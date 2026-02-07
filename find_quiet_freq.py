import uhd
import numpy as np
import time
import sys
from usrp_driver import B210UnifiedDriver
import sdr_utils

# Configuration for 5.8 GHz ISM Band
START_FREQ = 5.725e9
STOP_FREQ = 5.875e9
STEP_SIZE = 2e6  # 2 MHz steps (standard wifi channel spacing overlap)
SAMPLE_RATE = 1e6
GAIN = 60

sig_handler = sdr_utils.SignalHandler()

def get_simple_bar(value, min_val=-100, max_val=-30, width=30):
    norm = (value - min_val) / (max_val - min_val)
    norm = max(0.0, min(1.0, norm))
    filled = int(norm * width)
    return "█" * filled + "░" * (width - filled)

def run_scan():
    print(f"--> Initializing Scanner...")
    # Initialize at start freq
    driver = B210UnifiedDriver(START_FREQ, SAMPLE_RATE, GAIN)
    usrp = driver.initialize()
    rx_streamer = driver.get_rx_streamer()

    buff_len = 2048
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    current_freq = START_FREQ
    lowest_power = 0.0
    best_freq = START_FREQ
    
    scan_results = []

    print(f"\nScanning {START_FREQ/1e9:.3f} GHz to {STOP_FREQ/1e9:.3f} GHz...\n")
    print(f"{'FREQ (GHz)':<12} | {'PWR (dB)':<10} | {'SIGNAL STRENGTH'}")
    print("-" * 50)

    try:
        while current_freq <= STOP_FREQ and sig_handler.running:
            # 1. Tune
            driver.tune_frequency(current_freq)
            
            # 2. Flush buffer (clear samples from previous freq)
            for _ in range(3):
                rx_streamer.recv(recv_buffer, metadata, 0.01)

            # 3. Measure
            avg_pwr_linear = 0
            readings = 0
            
            # Take average of 5 readings
            for _ in range(5):
                samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
                if samps > 0:
                    data = recv_buffer[0][:samps]
                    pwr = np.mean(np.abs(data)**2)
                    avg_pwr_linear += pwr
                    readings += 1
            
            if readings > 0:
                final_pwr = avg_pwr_linear / readings
                pwr_db = 10 * np.log10(final_pwr + 1e-12)
                
                # Track best
                scan_results.append((current_freq, pwr_db))
                
                bar = get_simple_bar(pwr_db)
                print(f"{current_freq/1e9:.3f}       | {pwr_db:6.1f}     | {bar}")

            current_freq += STEP_SIZE
            
    except KeyboardInterrupt:
        pass
    finally:
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

    # Analysis
    if scan_results:
        # Sort by power (ascending)
        best = min(scan_results, key=lambda x: x[1])
        worst = max(scan_results, key=lambda x: x[1])
        
        print("\n" + "="*50)
        print(f"✅ BEST FREQ:  {best[0]/1e9:.3f} GHz ({best[1]:.1f} dB)")
        print(f"❌ WORST FREQ: {worst[0]/1e9:.3f} GHz ({worst[1]:.1f} dB)")
        print("="*50)
        print(f"Suggestion: Update 'args' in main.py to use {best[0]:.1e}")

if __name__ == "__main__":
    run_scan()
