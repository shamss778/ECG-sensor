import numpy as np
import paho.mqtt.client as mqtt
import time
import json

# Configuration MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "ecg/data"

# Configuration ECG
SAMPLING_RATE = 500  # points/sec
CYCLE_DURATION = 1.0  # 1 second per cycle
NOISE_LEVEL = 0.05

class ECGSensor:
    def __init__(self, sampling_rate=500, noise_level=0.05):
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.t = np.linspace(0, 1, sampling_rate)
        
    def ecg_cycle(self, t):
        """Generate a single ECG cycle (1 heartbeat)"""
        return (
            0.1 * np.sin(2 * np.pi * t * 1) +                      # Onde P
            -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +           # Onde Q
            1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +             # Pic R
            -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +            # Onde S
            0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)                 # Onde T
        )
    
    def generate_signal(self, num_cycles=5):
        """Generate full ECG signal with noise"""
        cycle = self.ecg_cycle(self.t)
        noise = self.noise_level * np.random.normal(size=cycle.shape)
        signal = np.tile(cycle + noise, num_cycles)
        return signal
    
    def get_sampled_points(self, sampling_factor=10):
        """Sample ECG signal at reduced frequency"""
        signal = self.generate_signal(num_cycles=5)
        sampled_indices = np.arange(0, len(signal), sampling_factor)
        sampled_signal = signal[sampled_indices]
        time_points = sampled_indices / self.sampling_rate
        return time_points, sampled_signal

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✓ Connected to MQTT broker")
    else:
        print(f"✗ Connection failed with code {rc}")

def publish_ecg_data(client, sampling_factor, test_name):
    """Publish ECG samples via MQTT"""
    print(f"\n{'='*70}")
    print(f"Publishing: {test_name}")
    print(f"{'='*70}")
    
    sensor = ECGSensor(sampling_rate=SAMPLING_RATE, noise_level=NOISE_LEVEL)
    time_points, sampled_signal = sensor.get_sampled_points(sampling_factor)
    
    print(f"Generated {len(sampled_signal)} samples (sampling factor: {sampling_factor})")
    print(f"Data reduction: {100 - (len(sampled_signal) / 500 * 100):.0f}%")
    print(f"\nTransmitting data via MQTT...\n")
    
    for i, (t, value) in enumerate(zip(time_points, sampled_signal)):
        payload = {
            "timestamp": float(t),
            "value": float(value),
            "index": int(i),
            "test_type": test_name
        }
        client.publish(MQTT_TOPIC, json.dumps(payload))
        print(f"[{i:2d}] t={t:.3f}s → {value:+.4f} mV")
        time.sleep(0.1)  # Simulate real-time transmission
    
    print(f"✓ {test_name} transmission completed!")
    time.sleep(2)  # Wait before next test

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    
    try:
        print("\n" + "="*70)
        print("ECG SENSOR - ADVANCED SIMULATOR")
        print("="*70)
        
        print("Connecting to MQTT broker...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        # ============================================================
        # TEST 1: COARSE SAMPLING (5 points per cycle)
        # ============================================================
        # sampling_factor = 100 means every 100th point
        # 500 points / 100 = 5 points per cycle
        publish_ecg_data(client, sampling_factor=100, test_name="TEST 1: COARSE SAMPLING (5 points)")
        
        # ============================================================
        # TEST 2: FINE SAMPLING (20 points per cycle)
        # ============================================================
        # sampling_factor = 25 means every 25th point
        # 500 points / 25 = 20 points per cycle
        publish_ecg_data(client, sampling_factor=25, test_name="TEST 2: FINE SAMPLING (20 points)")
        
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nNow run your receiver to see the visualizations:")
        print("  python edge_receiver_advanced.py")
        print("="*70)
        
        client.loop_stop()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()