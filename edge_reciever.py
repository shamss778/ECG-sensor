# edge_receiver_advanced_dual.py
# Advanced Receiver - Handles both 5-point and 20-point tests with separate visualizations
import paho.mqtt.client as mqtt
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "ecg/data"

# Store data for both tests
test1_data = {"timestamps": deque(maxlen=500), "values": deque(maxlen=500), "test_type": "TEST 1"}
test2_data = {"timestamps": deque(maxlen=500), "values": deque(maxlen=500), "test_type": "TEST 2"}

current_test = test1_data
data_complete = False
test_switch_flag = False

class ECGReceiver:
    def __init__(self, buffer_size=500):
        self.timestamps = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.original_signal = None
        
    def add_sample(self, timestamp, value):
        """Add received sample to buffer"""
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def get_data(self):
        """Get buffered data"""
        return np.array(self.timestamps), np.array(self.values)
    
    def interpolate_signal(self):
        """Linear interpolation to reconstruct full signal"""
        if len(self.timestamps) < 2:
            return None, None
        
        timestamps = np.array(self.timestamps)
        values = np.array(self.values)
        
        # Create dense time grid
        t_dense = np.linspace(timestamps[0], timestamps[-1], 
                             int((timestamps[-1] - timestamps[0]) * 500))
        
        # Linear interpolation
        values_interp = np.interp(t_dense, timestamps, values)
        
        return t_dense, values_interp
    
    def plot_signal_advanced(self, title="ECG Signal at Edge", sampling_type="Coarse"):
        """
        ADVANCED visualization with 3-step explanation
        
        Args:
            title: Plot title
            sampling_type: "Coarse (5 points)" or "Fine (20 points)"
        """
        timestamps, values = self.get_data()
        t_dense, values_interp = self.interpolate_signal()
        
        if len(values) == 0:
            print("No data to plot!")
            return
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(16, 11))
        
        # Define grid for subplots
        gs = fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[1, 1, 1.2])
        ax1 = fig.add_subplot(gs[0])  # Top
        ax2 = fig.add_subplot(gs[1])  # Middle
        ax3 = fig.add_subplot(gs[2])  # Bottom (larger)
        
        # ========== PLOT 1: Received Samples ONLY ==========
        ax1.scatter(timestamps, values, color='red', s=150, zorder=5, 
                   edgecolors='darkred', linewidth=2.5, label='Received Samples')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.set_ylabel("Amplitude (mV)", fontsize=11, fontweight='bold')
        ax1.set_title(f"STEP 1: {sampling_type} - What Was Actually Transmitted via MQTT", 
                     fontsize=12, fontweight='bold', color='darkred', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax1.set_xlim(timestamps[0] - 0.1, timestamps[-1] + 0.1)
        
        # Add sample count
        ax1.text(0.02, 0.95, f'Samples: {len(values)}/500', transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # ========== PLOT 2: Interpolation Process ==========
        # Show the straight lines being drawn
        for i in range(len(timestamps) - 1):
            ax2.plot([timestamps[i], timestamps[i+1]], 
                    [values[i], values[i+1]], 
                    color='blue', linewidth=2, alpha=0.6, linestyle='--')
        
        ax2.scatter(timestamps, values, color='red', s=150, zorder=5, 
                   edgecolors='darkred', linewidth=2.5, label='Received Samples')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.set_ylabel("Amplitude (mV)", fontsize=11, fontweight='bold')
        ax2.set_title("STEP 2: Linear Interpolation - Connecting the Dots", 
                     fontsize=12, fontweight='bold', color='darkblue', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax2.set_xlim(timestamps[0] - 0.1, timestamps[-1] + 0.1)
        
        ax2.text(0.02, 0.95, 'Straight lines = Linear interpolation', transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', style='italic',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # ========== PLOT 3: Final Reconstructed Signal ==========
        if values_interp is not None:
            ax3.plot(t_dense, values_interp, color='blue', linewidth=2.5, 
                    label='Reconstructed Signal (Interpolated)', alpha=0.8, zorder=2)
        
        ax3.scatter(timestamps, values, color='red', s=150, zorder=4, 
                   edgecolors='darkred', linewidth=2, label='Received Samples', alpha=0.9)
        
        # If we have original signal, plot it
        if self.original_signal is not None:
            orig_t, orig_v = self.original_signal
            ax3.plot(orig_t, orig_v, color='green', linewidth=1.5, 
                    label='Original Signal (Ground Truth)', alpha=0.6, linestyle=':', zorder=1)
        
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax3.set_xlabel("Time (s)", fontsize=11, fontweight='bold')
        ax3.set_ylabel("Amplitude (mV)", fontsize=11, fontweight='bold')
        ax3.set_title("STEP 3: Edge Device - Final Reconstructed ECG Signal", 
                     fontsize=12, fontweight='bold', color='green', pad=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=10, loc='upper right', framealpha=0.9, ncol=3)
        ax3.set_xlim(timestamps[0] - 0.1, timestamps[-1] + 0.1)
        
        # Add quality metrics
        if values_interp is not None:
            num_samples = len(values)
            data_reduction = 100 - (num_samples / 500 * 100)
            time_span = timestamps[-1] - timestamps[0]
            
            info_text = (
                f"📊 METRICS:\n"
                f"  • Samples: {num_samples}/500\n"
                f"  • Reduction: {data_reduction:.0f}%\n"
                f"  • Time: {time_span:.2f}s\n"
                f"  • Gap: {(time_span/num_samples):.3f}s avg"
            )
            ax3.text(0.02, 0.97, info_text, transform=ax3.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=0.8))
        
        plt.suptitle(f'ECG Signal Processing at Edge - {sampling_type}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.show()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✓ Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
        print(f"✓ Subscribed to {MQTT_TOPIC}")
    else:
        print(f"✗ Connection failed: {rc}")


def on_message(client, userdata, msg):
    global current_test, data_complete, test_switch_flag
    try:
        payload = json.loads(msg.payload.decode())
        timestamp = payload["timestamp"]
        value = payload["value"]
        index = payload["index"]
        test_type = payload.get("test_type", "UNKNOWN")
        
        # Check if we need to switch to test 2
        if test_type == "TEST 2" and not test_switch_flag:
            print(f"\n{'='*70}")
            print("SWITCHING TO TEST 2: FINE SAMPLING")
            print(f"{'='*70}\n")
            test_switch_flag = True
            current_test = test2_data
        
        # Add to current test buffer
        current_test["timestamps"].append(timestamp)
        current_test["values"].append(value)
        
        print(f"[{index:2d}] {test_type} → t={timestamp:.3f}s, val={value:+.4f} mV")
        
        # Check if transmission complete
        if index >= 40:
            data_complete = True
            
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing message: {e}")


def start_receiver():
    """Start MQTT receiver"""
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        print("=" * 70)
        print("ECG EDGE RECEIVER - ADVANCED DUAL TEST MODE")
        print("=" * 70)
        print("Waiting for TEST 1 (5 points) and TEST 2 (20 points)...\n")
        
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        # Wait for all data to arrive
        print("Listening for MQTT data...")
        while not data_complete:
            time.sleep(0.1)
        
        client.loop_stop()
        time.sleep(0.5)
        
        # ============================================================
        # TEST 1: COARSE SAMPLING (5 points)
        # ============================================================
        print("\n" + "="*70)
        print("VISUALIZING TEST 1: COARSE SAMPLING (5 points)")
        print("="*70)
        
        receiver1 = ECGReceiver()
        timestamps1 = np.array(list(test1_data["timestamps"]))
        values1 = np.array(list(test1_data["values"]))
        
        for t, v in zip(timestamps1, values1):
            receiver1.add_sample(t, v)
        
        if len(values1) > 0:
            print(f"✓ Received {len(values1)} samples")
            print(f"✓ Time range: {timestamps1[0]:.3f}s to {timestamps1[-1]:.3f}s")
            print(f"✓ Data reduction: {100 - (len(values1) / 500 * 100):.0f}%")
            receiver1.plot_signal_advanced(sampling_type="Coarse (5 points)")
        else:
            print("✗ No data received for TEST 1")
        
        # ============================================================
        # TEST 2: FINE SAMPLING (20 points)
        # ============================================================
        print("\n" + "="*70)
        print("VISUALIZING TEST 2: FINE SAMPLING (20 points)")
        print("="*70)
        
        receiver2 = ECGReceiver()
        timestamps2 = np.array(list(test2_data["timestamps"]))
        values2 = np.array(list(test2_data["values"]))
        
        for t, v in zip(timestamps2, values2):
            receiver2.add_sample(t, v)
        
        if len(values2) > 0:
            print(f"✓ Received {len(values2)} samples")
            print(f"✓ Time range: {timestamps2[0]:.3f}s to {timestamps2[-1]:.3f}s")
            print(f"✓ Data reduction: {100 - (len(values2) / 500 * 100):.0f}%")
            receiver2.plot_signal_advanced(sampling_type="Fine (20 points)")
        else:
            print("✗ No data received for TEST 2")
        
        # ============================================================
        # Quality Comparison
        # ============================================================
        if len(values1) > 0 and len(values2) > 0:
            print("\n" + "="*70)
            print("QUALITY COMPARISON")
            print("="*70)
            
            # Generate ground truth for comparison
            def ecg_cycle(t):
                return (
                    0.1 * np.sin(2 * np.pi * t * 1) +
                    -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
                    1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +
                    -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
                    0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
                )
            
            t_full = np.linspace(0, 1, 500)
            full_signal = ecg_cycle(t_full)
            full_signal = np.tile(full_signal, 5)  # 5 cycles
            
            # Interpolate both
            t_interp = np.linspace(min(timestamps1[0], timestamps2[0]), 
                                  max(timestamps1[-1], timestamps2[-1]), len(full_signal))
            
            recon1 = np.interp(t_interp, timestamps1, values1)
            recon2 = np.interp(t_interp, timestamps2, values2)
            
            from sklearn.metrics import mean_squared_error
            
            mse1 = mean_squared_error(full_signal[:len(recon1)], recon1)
            mse2 = mean_squared_error(full_signal[:len(recon2)], recon2)
            
            print(f"Coarse (5 pts):  MSE = {mse1:.6f}")
            print(f"Fine (20 pts):   MSE = {mse2:.6f}")
            print(f"\nFine sampling is {mse1/mse2:.1f}x better!")
            print("="*70)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    start_receiver()