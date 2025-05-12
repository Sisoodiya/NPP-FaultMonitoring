"""
Adaptive Real-Time Learning for NPP Fault Monitoring.

This module implements the adaptive real-time learning methods described in the research paper:
"Advanced Online Fault Monitoring in Nuclear Power Plants"

It includes:
- Enhanced sliding window technique for real-time model updating
- Adaptive learning rate adjustment
- Online model parameter updates
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from feature_extraction_new import extract_all_features
from dynamic_reliability import analyze_reliability, plot_reliability_curve


class AdaptiveModelUpdater:
    """
    Class for implementing adaptive real-time learning with sliding windows.
    """
    
    def __init__(self, model_path, window_size=100, step_size=10, 
                 learning_rate=0.001, min_lr=0.0001, max_lr=0.01, 
                 lr_decay_factor=0.95, performance_threshold=0.85):
        """
        Initialize the adaptive model updater.
        
        Args:
            model_path: Path to the pre-trained model
            window_size: Size of the sliding window
            step_size: Step size for the sliding window
            learning_rate: Initial learning rate for model updates
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            lr_decay_factor: Factor to decay learning rate when performance is good
            performance_threshold: Threshold for good performance
        """
        self.model_path = model_path
        self.window_size = window_size
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay_factor = lr_decay_factor
        self.performance_threshold = performance_threshold
        
        # Load the model
        self.model = load_model(model_path)
        
        # Initialize performance history
        self.performance_history = []
        self.lr_history = []
        
        # Initialize data buffer for sliding window
        self.data_buffer = []
        self.label_buffer = []
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize time tracking
        self.last_update_time = time.time()
        self.update_frequency = 60  # Update model every 60 seconds by default
        
        print(f"Adaptive model updater initialized with window size {window_size} and step size {step_size}")
    
    def add_data_point(self, data_point, label=None):
        """
        Add a new data point to the buffer.
        
        Args:
            data_point: New data point (features)
            label: Label for the data point (if available)
        """
        # Add to buffer
        self.data_buffer.append(data_point)
        if label is not None:
            self.label_buffer.append(label)
        
        # Keep buffer size limited to window_size
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
            if self.label_buffer:
                self.label_buffer.pop(0)
        
        # Check if it's time to update the model
        current_time = time.time()
        if current_time - self.last_update_time > self.update_frequency:
            self.update_model()
            self.last_update_time = current_time
    
    def add_batch_data(self, data_batch, labels=None):
        """
        Add a batch of data to the buffer.
        
        Args:
            data_batch: Batch of data points
            labels: Labels for the data points (if available)
        """
        for i, data_point in enumerate(data_batch):
            label = labels[i] if labels is not None and i < len(labels) else None
            self.add_data_point(data_point, label)
    
    def update_model(self, force=False):
        """
        Update the model using the current data buffer.
        
        Args:
            force: Force update regardless of buffer size
        """
        # Check if we have enough data
        if len(self.data_buffer) < self.window_size and not force:
            print(f"Not enough data for model update. Have {len(self.data_buffer)}/{self.window_size} points.")
            return
        
        # Convert buffer to numpy arrays
        X = np.array(self.data_buffer)
        
        # If we have labels, use them for supervised update
        if len(self.label_buffer) == len(self.data_buffer):
            y = np.array(self.label_buffer)
            self._supervised_update(X, y)
        else:
            # Otherwise do self-supervised update
            self._self_supervised_update(X)
        
        print(f"Model updated with learning rate {self.learning_rate:.6f}")
    
    def _supervised_update(self, X, y):
        """
        Update the model with supervised learning.
        
        Args:
            X: Features
            y: Labels
        """
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create a temporary optimizer with the current learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile the model with the new optimizer
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fit the model for one epoch
        history = self.model.fit(
            X_scaled, y,
            epochs=1,
            batch_size=min(32, len(X)),
            verbose=0
        )
        
        # Get the performance
        performance = history.history['accuracy'][0]
        self.performance_history.append(performance)
        self.lr_history.append(self.learning_rate)
        
        # Adjust learning rate based on performance
        self._adjust_learning_rate(performance)
        
        print(f"Supervised update completed. Performance: {performance:.4f}")
    
    def _self_supervised_update(self, X):
        """
        Update the model with self-supervised learning.
        
        Args:
            X: Features
        """
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Use high confidence predictions as pseudo-labels
        confidence = np.max(predictions, axis=1)
        high_confidence_mask = confidence > 0.9
        
        if np.sum(high_confidence_mask) > 0:
            # Use only high confidence samples
            X_high_conf = X_scaled[high_confidence_mask]
            pseudo_labels = predictions[high_confidence_mask]
            
            # Create a temporary optimizer with the current learning rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            # Compile the model with the new optimizer
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Fit the model for one epoch
            history = self.model.fit(
                X_high_conf, pseudo_labels,
                epochs=1,
                batch_size=min(32, len(X_high_conf)),
                verbose=0
            )
            
            # Get the performance
            performance = history.history['accuracy'][0]
            self.performance_history.append(performance)
            self.lr_history.append(self.learning_rate)
            
            # Adjust learning rate based on performance
            self._adjust_learning_rate(performance)
            
            print(f"Self-supervised update completed with {np.sum(high_confidence_mask)} high-confidence samples. Performance: {performance:.4f}")
        else:
            print("No high confidence predictions for self-supervised update.")
    
    def _adjust_learning_rate(self, performance):
        """
        Adjust learning rate based on performance.
        
        Args:
            performance: Current performance metric
        """
        # If performance is good, reduce learning rate
        if performance > self.performance_threshold:
            self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay_factor)
        else:
            # If performance is poor, increase learning rate
            self.learning_rate = min(self.max_lr, self.learning_rate / self.lr_decay_factor)
    
    def save_model(self, path=None):
        """
        Save the updated model.
        
        Args:
            path: Path to save the model (if None, use original path)
        """
        if path is None:
            path = self.model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def plot_performance_history(self, save_path=None):
        """
        Plot the performance history.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.performance_history:
            print("No performance history to plot.")
            return
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot performance
        ax1.set_xlabel('Update Step')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        ax1.plot(self.performance_history, 'b-', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Plot learning rate on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', color='tab:red')
        ax2.plot(self.lr_history, 'r-', label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Add horizontal line for performance threshold
        ax1.axhline(y=self.performance_threshold, color='g', linestyle='--', 
                   label=f'Performance Threshold ({self.performance_threshold})')
        
        fig.tight_layout()
        plt.title('Model Performance and Learning Rate History')
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance history plot saved to {save_path}")
        
        plt.show()


class AdaptiveOnlineFaultMonitor:
    """
    Class for implementing adaptive online fault monitoring.
    """
    
    def __init__(self, model_path, feature_extractor=None, window_size=100, 
                 step_size=10, update_interval=100, reliability_analysis=True):
        """
        Initialize the adaptive online fault monitor.
        
        Args:
            model_path: Path to the pre-trained model
            feature_extractor: Function to extract features from raw data
            window_size: Size of the sliding window
            step_size: Step size for the sliding window
            update_interval: Number of samples before model update
            reliability_analysis: Whether to perform reliability analysis
        """
        self.model_updater = AdaptiveModelUpdater(
            model_path=model_path,
            window_size=window_size,
            step_size=step_size
        )
        
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.step_size = step_size
        self.update_interval = update_interval
        self.reliability_analysis = reliability_analysis
        
        # Initialize buffers
        self.raw_data_buffer = []
        self.prediction_buffer = []
        self.sample_count = 0
        
        # Initialize reliability analysis results
        self.reliability_results = {}
        
        print("Adaptive online fault monitor initialized")
    
    def process_data_stream(self, data_stream, process_count=None):
        """
        Process a stream of data for online fault monitoring.
        
        Args:
            data_stream: Iterator or generator yielding data points
            process_count: Number of data points to process (None for all)
        
        Returns:
            List of predictions
        """
        predictions = []
        count = 0
        
        for data_point in tqdm(data_stream, desc="Processing data stream", total=process_count):
            # Process the data point
            prediction = self.process_data_point(data_point)
            predictions.append(prediction)
            
            count += 1
            if process_count is not None and count >= process_count:
                break
        
        # Final model update
        if self.sample_count > 0:
            self.model_updater.update_model(force=True)
        
        # Perform reliability analysis if enabled
        if self.reliability_analysis and self.prediction_buffer:
            self._perform_reliability_analysis()
        
        return predictions
    
    def process_data_point(self, data_point):
        """
        Process a single data point for fault detection.
        
        Args:
            data_point: Raw data point
        
        Returns:
            Prediction for the data point
        """
        # Add to raw data buffer
        self.raw_data_buffer.append(data_point)
        
        # Keep buffer size limited
        if len(self.raw_data_buffer) > self.window_size:
            self.raw_data_buffer.pop(0)
        
        # Check if we have enough data for feature extraction
        if len(self.raw_data_buffer) < self.window_size:
            return None
        
        # Extract features if we have a feature extractor
        if self.feature_extractor is not None:
            features = self.feature_extractor(self.raw_data_buffer)
        else:
            # Otherwise use the raw data as features
            features = np.array(self.raw_data_buffer)
        
        # Get prediction from model
        prediction = self._get_prediction(features)
        
        # Add to prediction buffer
        self.prediction_buffer.append(prediction)
        
        # Increment sample count
        self.sample_count += 1
        
        # Check if it's time to update the model
        if self.sample_count % self.update_interval == 0:
            self.model_updater.update_model()
        
        return prediction
    
    def _get_prediction(self, features):
        """
        Get prediction from the model.
        
        Args:
            features: Extracted features
        
        Returns:
            Prediction
        """
        # Reshape features if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize features
        features_scaled = self.model_updater.scaler.transform(features)
        
        # Get prediction
        prediction = self.model_updater.model.predict(features_scaled, verbose=0)
        
        # Return the class with highest probability
        return np.argmax(prediction, axis=1)[0]
    
    def _perform_reliability_analysis(self):
        """
        Perform reliability analysis on the predictions.
        """
        # Convert predictions to a format suitable for reliability analysis
        predictions_df = pd.DataFrame({
            'time000000000': np.arange(len(self.prediction_buffer)),
            'predicted': self.prediction_buffer
        })
        
        # Get unique fault types
        fault_types = np.unique(self.prediction_buffer)
        
        # Analyze reliability for each fault type
        for fault_type in fault_types:
            results = analyze_reliability(predictions_df, fault_value=fault_type)
            self.reliability_results[fault_type] = results
            
            # Plot reliability curve
            plot_reliability_curve(
                results, 
                title=f"Reliability Curve for Fault Type {fault_type}",
                save_path=f"reliability_fault_{fault_type}.png"
            )
        
        print(f"Reliability analysis completed for {len(fault_types)} fault types")
    
    def save_model(self, path=None):
        """
        Save the updated model.
        
        Args:
            path: Path to save the model
        """
        self.model_updater.save_model(path)
    
    def plot_performance_history(self, save_path=None):
        """
        Plot the performance history.
        
        Args:
            save_path: Path to save the plot
        """
        self.model_updater.plot_performance_history(save_path)


# Example usage
if __name__ == "__main__":
    from data_preprocessing import process_pipeline
    
    # Define a simple feature extractor
    def extract_features(data_window):
        # Convert to DataFrame if it's a list
        if isinstance(data_window, list):
            data_window = pd.DataFrame(data_window)
        
        # Extract features
        features, _, _ = extract_all_features(
            data_window, window_size=len(data_window), include_wks=True
        )
        
        return features.values
    
    # Create adaptive online fault monitor
    monitor = AdaptiveOnlineFaultMonitor(
        model_path="models/saved/best_model.h5",
        feature_extractor=extract_features,
        window_size=100,
        step_size=10,
        update_interval=50
    )
    
    # Get preprocessed data
    processed_data, _ = process_pipeline()
    
    # Simulate a data stream
    def data_stream_simulator(data, chunk_size=1):
        for i in range(0, len(data), chunk_size):
            yield data.iloc[i:i+chunk_size]
    
    # Process the data stream
    data_stream = data_stream_simulator(processed_data)
    predictions = monitor.process_data_stream(data_stream, process_count=1000)
    
    # Save the updated model
    monitor.save_model("models/saved/updated_model.h5")
    
    # Plot performance history
    monitor.plot_performance_history("adaptive_performance.png")
