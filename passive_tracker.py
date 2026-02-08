import uhd
import numpy as np
import sys
import time
import signal
import threading

from usrp_driver import B210UnifiedDriver
import sdr_utils

# ‼️ CHANGED: Default freq set to Channel 149 (5.745 GHz) as a common router channel.
# Check your router's specific channel and update this arg.
args = sdr_utils.get_standard_args("Passive Object Detection", default_freq=5.745e9, default_gain=70)

# ‼️ CHANGED: Threshold logic is now relative to noise floor (SNR), not absolute voltage
SNR_THRESHOLD_DB = 5.0  

TRAINING_FRAMES = 30            
CSI_WIN_SIZE = 64               

sig_handler = sdr_utils.SignalHandler()

# ‼️ REMOVED: ControllableTransmitter class and PROBE_TX definitions. 
# This script is strictly passive.

class RFModel:
    def __init__(self):
        self.profiles = {} # Dictionary to store 'label' -> 'mean_vector'

    def train(self, label, data_matrix):
        if len(data_matrix) == 0:
            print(f"  [Model] ⚠️ No data captured for '{label}'. Training skipped.")
            return

        # Average across all captured frames to remove noise
        mean_vector = np.mean(np.array(data_matrix), axis=0)
        self.profiles[label] = mean_vector
        print(f"  [Model] Learned '{label}' from {len(data_matrix)} frames.")

    def predict(self, current_vector):
        if not self.profiles:
            return "Uncalibrated", 0.0

        best_label = None
        min_dist = float('inf')

        for label, profile in self.profiles.items():
            dist = np.linalg.norm(current_vector - profile)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        return best_label, min_dist


def extract_passive_feature(rx_chunk):
    """
    ‼️ CHANGED: Uses energy burst detection instead of chirp correlation.
    """
    # 1. Detect if there is a packet/burst in this chunk
    res = sdr_utils.detect_energy_burst(rx_chunk, threshold_factor=10**(SNR_THRESHOLD_DB/10))
    
    if res['is_burst']:
        peak_idx = res['peak_idx']
        
        # 2. Extract a window around the peak energy
        # For passive WiFi, we capture the burst structure itself
        start_idx = peak_idx
        end_idx = start_idx + CSI_WIN_SIZE
        
        captured_burst = np.zeros(CSI_WIN_SIZE, dtype=np.complex64)
        
        # Safe array slicing
        src_end = min(len(rx_chunk), end_idx)
        copy_len = src_end - start_idx
        
        if copy_len > 0:
             captured_burst[:copy_len] = rx_chunk[start_idx:src_end]

        # 3. Use the raw burst as the "Channel Impulse Response" proxy
        # Since we don't know the transmitted data, we assume the router sends
        # similar preambles/beacons repeatedly.
        metrics = sdr_utils.calculate_csi_metrics(captured_burst, args.rate)
        return metrics['cfr_db']
        
    return None


def collect_training_data(usrp, driver, label):
    """
    Captures N frames for a specific label using passive listening.
    """
    print(f"\n  [TRAIN] 📸 Listening for {TRAINING_FRAMES} bursts for '{label}'... (Ctrl+C to cancel)")
    
    rx_streamer = driver.get_rx_streamer()
    buff_len = 4096 # Smaller buffer for faster loops
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    collected_frames = []
    
    try:
        while len(collected_frames) < TRAINING_FRAMES:
            samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                 continue

            if samps > 0:
                data = recv_buffer[0][:samps]
                feature = extract_passive_feature(data)
                
                if feature is not None:
                    collected_frames.append(feature)
                    sys.stdout.write(f"\r  [TRAIN] Progress: {len(collected_frames)}/{TRAINING_FRAMES}")
                    sys.stdout.flush()
                    # Sleep slightly to avoid capturing the exact same packet twice if loop is too fast
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print(f"\n  [TRAIN] ⚠️  Capture cancelled by user.")

    finally:
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

    print("\n  [TRAIN] Done.")
    return collected_frames


def run_inference_loop(usrp, driver, model): 
    """
    The live recognition loop (Passive Mode).
    """
    print(f"\n  [LIVE] 👁️  Starting Passive Recognition. Press Ctrl+C to return to menu.")
    
    rx_streamer = driver.get_rx_streamer()
    buff_len = 4096
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    try:
        while True:
            samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                 continue

            if samps > 0:
                data = recv_buffer[0][:samps]
                feature = extract_passive_feature(data)
                
                if feature is not None:
                    label, dist = model.predict(feature)
                    
                    # ‼️ Visual Feedback for signal strength
                    pwr_db = 10*np.log10(np.mean(np.abs(data)**2) + 1e-12)
                    
                    status_str = f"PREDICTION: {label}"
                    print(f"  [LIVE] {status_str:<20} | Dist: {dist:6.2f} | Burst Pwr: {pwr_db:.1f} dB")

    except KeyboardInterrupt:
        print("\n  [LIVE] 🛑 Stopping recognition loop...")
        pass
    finally:
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))


def main_menu(usrp, driver):
    model = RFModel()
    
    try:
        while True:
            print("\n" + "="*40)
            print("    PASSIVE WIFI SENSING MENU")
            print("="*40)
            print(f" Current Knowledge: {list(model.profiles.keys())}")
            print(" [1] Train 'Empty' (Baseline)")
            print(" [2] Train New Object...")
            print(" [3] Run Live Recognition")
            print(" [q] Quit")
            
            choice = input("\nSelect Option: ").strip().lower()
            
            if choice == 'q':
                break
                
            elif choice == '1':
                print("\nEnsure area is clear.")
                time.sleep(1)
                frames = collect_training_data(usrp, driver, "Empty")
                model.train("Empty", frames)
                
            elif choice == '2':
                label = input("Enter object name (e.g. 'Bottle', 'Hand'): ")
                if label:
                    print(f"\nPlace '{label}' in target zone.")
                    input("Press Enter to start listening...")
                    frames = collect_training_data(usrp, driver, label)
                    model.train(label, frames)
                    
            elif choice == '3':
                if not model.profiles:
                    print("\n⚠️  You must train at least one state (e.g. Empty) first!")
                    continue
                run_inference_loop(usrp, driver, model)
                
    except KeyboardInterrupt:
        print("\n\n--> Application terminated by user.")


if __name__ == "__main__":
    print("--> Initializing Passive Detector...")
    
    print(f"  [CONFIG] Freq: {args.freq/1e9:.3f} GHz (Ensure this matches your router)")
    print(f"  [CONFIG] Rate: {args.rate/1e6:.1f} MHz")
    print(f"  [CONFIG] Gain: {args.gain} dB")
    
    signal.signal(signal.SIGINT, signal.default_int_handler)

    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    
    main_menu(usrp, driver)
