import uhd
import numpy as np
import sys
import time
import signal
import threading

# ‼️ CHANGED: Imported from local modules instead of shared sdr_lib
from usrp_driver import B210UnifiedDriver
import sdr_utils


args = sdr_utils.get_standard_args("Object Detection", default_freq=5.8e9)

CHIRP_LEN = 256        
GAP_LEN = 2000         
THRESHOLD = 0.05    

TRAINING_FRAMES = 30          
CSI_WIN_SIZE = 64              

sig_handler = sdr_utils.SignalHandler()

PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)
# Prepare TX Frame
padding = np.zeros(GAP_LEN, dtype=np.complex64)
TX_FRAME = np.concatenate([padding, PROBE_TX, padding])

class ControllableTransmitter(threading.Thread):
    def __init__(self, driver, frame_data, interval=0.5):
        super().__init__()
        self.driver = driver
        self.frame = frame_data
        self.interval = interval
        self.running = True
        self.daemon = True

    def run(self):
        tx_streamer = self.driver.get_tx_streamer()
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = True
        reshaped_frame = self.frame.reshape(1, -1)
        
        while self.running and sig_handler.running:
            try:
                md.has_time_spec = False
                tx_streamer.send(reshaped_frame, md)
                time.sleep(self.interval)
            except:
                break

    def stop(self):
        self.running = False


class RFModel:
    def __init__(self):
        self.profiles = {} # Dictionary to store 'label' -> 'mean_vector'

    def train(self, label, data_matrix):
        """
        Calculates the centroid (average) of the captured frames 
        to create a reference fingerprint for this object.
        """
        if len(data_matrix) == 0:
            print(f"  [Model] ⚠️ No data captured for '{label}'. Training skipped.")
            return

        # Average across all captured frames to remove noise
        mean_vector = np.mean(np.array(data_matrix), axis=0)
        self.profiles[label] = mean_vector
        print(f"  [Model] Learned '{label}' from {len(data_matrix)} frames.")

    def predict(self, current_vector):
        """
        Finds the nearest known fingerprint using Euclidean distance.
        Returns: (best_label, confidence_score)
        """
        if not self.profiles:
            return "Uncalibrated", 0.0

        best_label = None
        min_dist = float('inf')

        # Compare current signal against all learned profiles
        for label, profile in self.profiles.items():
            # Euclidean distance
            dist = np.linalg.norm(current_vector - profile)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        return best_label, min_dist


def extract_csi_feature(rx_chunk):
    """
    Extracted feature extraction logic for reuse in both training and inference loops.
    """
    res = sdr_utils.correlate_and_detect(rx_chunk, PROBE_TX)
    
    if res['snr_db'] > 10:
        peak_idx = res['peak_idx']
        # Extract CSI Window around the peak
        PRE_CURSOR = 10
        start_idx = peak_idx - PRE_CURSOR
        end_idx = start_idx + CSI_WIN_SIZE
        cir_window = np.zeros(CSI_WIN_SIZE, dtype=np.complex64)
        
        # Safe array slicing
        src_start = max(0, start_idx)
        src_end = min(len(res['correlation']), end_idx)
        dst_start = src_start - start_idx
        dst_end = dst_start + (src_end - src_start)
        
        if src_end > src_start:
             cir_window[dst_start:dst_end] = res['correlation'][src_start:src_end]

        if np.sum(np.abs(cir_window)) < 1e-6:
            return None
        
        # We use the Log-Magnitude CFR (Channel Frequency Response) as the feature
        metrics = sdr_utils.calculate_csi_metrics(cir_window, args.rate)
        return metrics['cfr_db']
        
    return None


def collect_training_data(usrp, driver, label):
    """
    Specialized loop to capture N frames for a specific label.
    """
    print(f"\n  [TRAIN] 📸 capturing {TRAINING_FRAMES} frames for '{label}'... (Ctrl+C to cancel)")
    
    tx = ControllableTransmitter(driver, TX_FRAME, interval=0.5)
    tx.start()
    
    rx_streamer = driver.get_rx_streamer()
    buff_len = 10000 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    collected_frames = []
    
    try:
        # Loop uses True instead of sig_handler.running to allow interrupt
        while len(collected_frames) < TRAINING_FRAMES:
            samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                 continue

            if samps > 0:
                data = recv_buffer[0][:samps]
                if np.max(np.abs(data)) > THRESHOLD:
                    feature = extract_csi_feature(data)
                    if feature is not None:
                        collected_frames.append(feature)
                        sys.stdout.write(f"\r  [TRAIN] Progress: {len(collected_frames)}/{TRAINING_FRAMES}")
                        sys.stdout.flush()

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print(f"\n  [TRAIN] ⚠️  Capture cancelled by user.")
        # Don't return yet, ensure cleanup runs

    finally:
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))
        tx.stop()
        tx.join(timeout=1.0)

    print("\n  [TRAIN] Done.")
    return collected_frames


def run_inference_loop(usrp, driver, model): 
    """
    The live recognition loop.
    """
    print(f"\n  [LIVE] 👁️  Starting Recognition. Press Ctrl+C to return to menu.")
    
    tx = ControllableTransmitter(driver, TX_FRAME, interval=0.5)
    tx.start()

    rx_streamer = driver.get_rx_streamer()
    buff_len = 10000 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    try:
        # Loop uses True to rely on KeyboardInterrupt for exit
        while True:
            samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                 continue

            if samps > 0:
                data = recv_buffer[0][:samps]
                if np.max(np.abs(data)) > THRESHOLD:
                    feature = extract_csi_feature(data)
                    
                    if feature is not None:
                        label, dist = model.predict(feature)
                        status_str = f"PREDICTION: {label}"
                        print(f"  [LIVE] {status_str:<20} | Dist: {dist:6.2f} | Input Pwr: {np.mean(np.abs(data)):.3f}")

    except KeyboardInterrupt:
        # Catch interruption and return to menu
        print("\n  [LIVE] 🛑 Stopping recognition loop...")
        pass
    finally:
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))
        tx.stop()
        tx.join(timeout=1.0)


def main_menu(usrp, driver):
    model = RFModel()
    
    # Outer loop catches KeyboardInterrupt for clean exit
    try:
        while True:
            print("\n" + "="*40)
            print("   RF OBJECT RECOGNITION MENU")
            print("="*40)
            print(f" Current Knowledge: {list(model.profiles.keys())}")
            print(" [1] Train 'Empty' (Baseline)")
            print(" [2] Train New Object...")
            print(" [3] Run Live Recognition")
            print(" [q] Quit")
            
            # Ctrl+C here will raise KeyboardInterrupt and go to 'except' block below
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
                    input("Press Enter to start capturing...")
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
    print("--> Initializing Object Detector (Trainable)...")
    
    # Force restoration of default SIGINT handler.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    
    main_menu(usrp, driver)
