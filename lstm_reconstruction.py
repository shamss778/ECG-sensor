# lstm_reconstruction.py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ECGLSTMReconstructor:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def generate_training_data(self, num_cycles=20, sampling_rate=500):
        """Generate synthetic ECG for training"""
        def ecg_cycle(t):
            return (
                0.1 * np.sin(2 * np.pi * t * 1) +
                -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
                1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +
                -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
                0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
            )
        
        t_single = np.linspace(0, 1, sampling_rate)
        cycle = ecg_cycle(t_single)
        signal = np.tile(cycle, num_cycles)
        
        # Add noise
        noise = 0.05 * np.random.normal(size=signal.shape)
        signal = signal + noise
        
        return signal
    
    def create_sequences(self, data):
        """Create training sequences"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.sequence_length, 1), 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(self.model.summary())
    
    def train(self, epochs=30, batch_size=32):
        """Train LSTM model on synthetic data"""
        # Generate training data
        signal = self.generate_training_data(num_cycles=20)
        signal_normalized = self.scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(signal_normalized)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train model
        print("Training LSTM model...")
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                                validation_split=0.2, verbose=1)
        
        return history, signal_normalized
    
    def reconstruct_from_sampled(self, full_signal, sampling_factor=10):
        """Reconstruct full signal from sampled points"""
        # Sample the full signal
        sampled_indices = np.arange(0, len(full_signal), sampling_factor)
        sampled_signal = full_signal[sampled_indices]
        
        # Normalize sampled signal
        sampled_scaled = sampled_signal.reshape(-1, 1)
        
        # Use model to predict missing points
        reconstructed = np.copy(full_signal)
        
        # For each gap between samples, predict intermediate points
        for i in range(len(sampled_indices) - 1):
            start_idx = sampled_indices[i]
            end_idx = sampled_indices[i + 1]
            gap_size = end_idx - start_idx
            
            # Use model prediction for intermediate points
            if i < len(sampled_indices) - self.sequence_length:
                context = full_signal[max(0, start_idx - self.sequence_length):start_idx]
                if len(context) >= self.sequence_length:
                    context = context[-self.sequence_length:].reshape(1, -1, 1)
                    prediction = self.model.predict(context, verbose=0)[0][0]
                    
                    # Linear interpolation + LSTM prediction
                    alpha = 0.7  # Weight for LSTM prediction
                    for j in range(1, gap_size):
                        linear = full_signal[start_idx] + (j / gap_size) * \
                                (full_signal[end_idx] - full_signal[start_idx])
                        reconstructed[start_idx + j] = (1 - alpha) * linear + \
                                                      alpha * prediction
        
        return reconstructed, sampled_signal, sampled_indices
    
    def evaluate(self, full_signal, reconstructed_signal):
        """Evaluate reconstruction quality"""
        mse = mean_squared_error(full_signal, reconstructed_signal)
        mae = mean_absolute_error(full_signal, reconstructed_signal)
        rmse = np.sqrt(mse)
        
        print(f"\n=== Reconstruction Quality ===")
        print(f"MSE:  {mse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return {"mse": mse, "mae": mae, "rmse": rmse}
    
    def plot_comparison(self, full_signal, reconstructed, sampled_signal, 
                       sampled_indices, title="ECG Reconstruction with LSTM"):
        """Visualize original vs reconstructed signal"""
        t_full = np.linspace(0, 1, len(full_signal))
        t_sampled = sampled_indices / len(full_signal)
        
        plt.figure(figsize=(14, 8))
        
        # Plot original signal
        plt.plot(t_full, full_signal, 'g-', linewidth=2, label='Original Signal', alpha=0.7)
        
        # Plot sampled points
        plt.plot(t_sampled, sampled_signal, 'ro', markersize=8, label='Sampled Points')
        
        # Plot reconstructed signal
        plt.plot(t_full, reconstructed, 'b--', linewidth=2, label='LSTM Reconstructed', alpha=0.8)
        
        plt.title(title, fontsize=14)
        plt.xlabel("Normalized Time")
        plt.ylabel("Amplitude (mV)")
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    print("=" * 60)
    print("ECG LSTM Reconstruction Lab")
    print("=" * 60)
    
    # Initialize reconstructor
    reconstructor = ECGLSTMReconstructor(sequence_length=50)
    
    # Build and train model
    reconstructor.build_model()
    history, signal_normalized = reconstructor.train(epochs=30, batch_size=32)
    
    print("\n" + "=" * 60)
    print("Testing Reconstruction")
    print("=" * 60)
    
    # Test Case 1: Coarse Sampling (5 points per cycle)
    print("\n[Test 1] Coarse Sampling (5 points per cycle)")
    reconstructed_coarse, sampled_coarse, indices_coarse = \
        reconstructor.reconstruct_from_sampled(signal_normalized, sampling_factor=100)
    metrics_coarse = reconstructor.evaluate(signal_normalized, reconstructed_coarse)
    reconstructor.plot_comparison(signal_normalized, reconstructed_coarse, 
                                 sampled_coarse, indices_coarse,
                                 "ECG Reconstruction: Coarse Sampling (5 points/cycle)")
    
    # Test Case 2: Fine Sampling (20 points per cycle)
    print("\n[Test 2] Fine Sampling (20 points per cycle)")
    reconstructed_fine, sampled_fine, indices_fine = \
        reconstructor.reconstruct_from_sampled(signal_normalized, sampling_factor=25)
    metrics_fine = reconstructor.evaluate(signal_normalized, reconstructed_fine)
    reconstructor.plot_comparison(signal_normalized, reconstructed_fine, 
                                 sampled_fine, indices_fine,
                                 "ECG Reconstruction: Fine Sampling (20 points/cycle)")
    
    # Comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"\nCoarse (5 pts): MSE={metrics_coarse['mse']:.6f}, RMSE={metrics_coarse['rmse']:.6f}")
    print(f"Fine (20 pts):  MSE={metrics_fine['mse']:.6f}, RMSE={metrics_fine['rmse']:.6f}")
    print(f"\nImprovement: {((metrics_coarse['mse'] - metrics_fine['mse']) / metrics_coarse['mse'] * 100):.1f}%")

if __name__ == "__main__":
    main()
