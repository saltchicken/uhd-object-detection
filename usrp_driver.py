import uhd
import sys
import time



class B210UnifiedDriver:
    """
    Supports both Single Channel (SISO) and Dual Channel (MIMO) setups.
    Configures both TX and RX chains.
    """
    
    STREAM_MODE_START = uhd.types.StreamMode.start_cont
    STREAM_MODE_STOP = uhd.types.StreamMode.stop_cont
    MODE_NAME = "Native Continuous"

    def __init__(self, freq, rate, gain, num_channels=1, device_args="type=b200"):
        self.freq = freq
        self.rate = rate
        self.gain = gain
        self.num_channels = num_channels
        self.device_args = device_args
        self.usrp = None

    def initialize(self):
        """
        Returns the configured USRP object.
        """
        print(f"--> Scanning for B210 Device (Mode: {self.num_channels}ch)...")
        try:
            self.usrp = uhd.usrp.MultiUSRP(self.device_args)
        except RuntimeError:
            print("No USRP found. Ensure a B210 is connected.")
            sys.exit(1)

        if self.usrp.get_rx_num_channels() < self.num_channels:
            print(f"Error: Device has {self.usrp.get_rx_num_channels()} channels. App requires {self.num_channels}.")
            sys.exit(1)

        if self.num_channels == 2:
            print("--> Applying MIMO Subdev Spec (A:A A:B)...")
            self.usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A A:B"))
            self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:A A:B"))

        for ch in range(self.num_channels):
            # RX Setup
            self.usrp.set_rx_rate(self.rate, ch)
            treq_rx = uhd.types.TuneRequest(self.freq)
            treq_rx.args = uhd.types.DeviceAddr("mode_n=integer") 
            self.usrp.set_rx_freq(treq_rx, ch)
            self.usrp.set_rx_gain(self.gain, ch)
            self.usrp.set_rx_antenna("RX2", ch)
            
            # TX Setup
            self.usrp.set_tx_rate(self.rate, ch)
            treq_tx = uhd.types.TuneRequest(self.freq)
            treq_tx.args = uhd.types.DeviceAddr("mode_n=integer")
            self.usrp.set_tx_freq(treq_tx, ch)
            self.usrp.set_tx_gain(self.gain, ch)
            self.usrp.set_tx_antenna("TX/RX", ch)

        self.usrp.set_time_now(uhd.types.TimeSpec(0.0))
        
        print("--> Waiting for LO Lock...")
        time.sleep(1.0) 
        
        return self.usrp



    def get_rx_streamer(self):
        """Helper to get the RX streamer for active channels."""
        if not self.usrp:
            raise RuntimeError("USRP not initialized.")
            
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = list(range(self.num_channels))
        return self.usrp.get_rx_stream(st_args)

    def get_tx_streamer(self):
        """Helper to get the TX streamer for active channels."""
        if not self.usrp:
            raise RuntimeError("USRP not initialized.")
            
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = list(range(self.num_channels))
        return self.usrp.get_tx_stream(st_args)


