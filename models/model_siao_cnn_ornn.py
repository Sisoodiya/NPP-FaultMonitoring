"""SIAO-CNN-ORNN model with standard and enhanced versions
Provides both the original SIAO model and an enhanced version with additional features:
- Focal loss for handling class imbalance
- Weighted Kurtosis and Skewness (WKS) feature extraction
- Multi-head attention mechanism
- Custom class weights for underrepresented classes
- Higher default dropout and L2 regularization
- Optimized for M1/M2 Mac performance
"""
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.optimizers import Adam, Optimizer
# Import legacy optimizers for better M1/M2 Mac performance
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
import tensorflow as tf
import numpy as np
import os
import time
import math
import datetime
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Enable Metal plugin for better performance on M1/M2 Macs
os.environ['TF_METAL_ENABLED'] = '1'

# Check if running on M1/M2 Mac
import platform

# Most reliable way to detect Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'

print(f"Running on {'Apple Silicon (M1/M2)' if IS_APPLE_SILICON else 'Standard hardware'}")

# Import SIAO optimizer if available
try:
    from siao_optimizer import aquila_optimizer
    SIAO_AVAILABLE = True
except ImportError:
    SIAO_AVAILABLE = False
    print("SIAO optimizer not available. Using custom implementation.")


class AquilaOptimizer(Optimizer):
    """Aquila Optimizer implementation for Keras/TensorFlow.
    
    This optimizer implements the four-stage Aquila Optimizer algorithm as described in the research paper.
    """
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name="AquilaOptimizer", **kwargs):
        super(AquilaOptimizer, self).__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon
        self._iterations = 0
        self._max_iterations = 1000  # Default value, will be updated during training
        
    def _create_slots(self, var_list):
        # Create slots for the first and second moments
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "prev_best")
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        prev_best = self.get_slot(var, "prev_best")
        
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        
        # Increment iterations
        self._iterations += 1
        
        # Calculate phase based on current iteration
        if self._iterations < self._max_iterations * 0.25:
            phase = "expanded_exploration"
        elif self._iterations < self._max_iterations * 0.5:
            phase = "narrowed_exploration"
        elif self._iterations < self._max_iterations * 0.75:
            phase = "expanded_exploitation"
        else:
            phase = "narrowed_exploitation"
        
        # Update biased first moment estimate
        m.assign(beta_1_t * m + (1 - beta_1_t) * grad)
        
        # Update biased second raw moment estimate
        v.assign(beta_2_t * v + (1 - beta_2_t) * tf.square(grad))
        
        # Compute bias-corrected first and second moment estimates
        m_hat = m / (1 - tf.pow(beta_1_t, self._iterations))
        v_hat = v / (1 - tf.pow(beta_2_t, self._iterations))
        
        # Calculate alpha and beta parameters
        alpha = 2.0 * (1.0 - (self._iterations / self._max_iterations))
        beta = 2.0 * tf.pow(self._iterations / self._max_iterations, 2)
        
        # Generate random values
        r1 = tf.random.uniform(tf.shape(var), minval=0, maxval=1, dtype=var_dtype)
        r2 = tf.random.uniform(tf.shape(var), minval=0, maxval=1, dtype=var_dtype)
        
        # Update based on the current phase
        if phase == "expanded_exploration":
            # Expanded exploration: Wide search with random walks
            levy = self._levy_flight(tf.shape(var), var_dtype)
            update = -lr_t * (alpha * r1 * m_hat + r2 * levy)
            
        elif phase == "narrowed_exploration":
            # Narrowed exploration: More focused around promising areas
            update = -lr_t * alpha * r1 * m_hat
            
        elif phase == "expanded_exploitation":
            # Expanded exploitation: Exploit the best solution with some randomness
            chaotic = self._chaotic_map(var_dtype)
            update = -lr_t * (beta * r1 * m_hat + r2 * chaotic)
            
        else:  # narrowed_exploitation
            # Narrowed exploitation: Fine-tune around the best solution
            update = -lr_t * beta * r1 * m_hat * 0.5
        
        # Apply update
        new_var = var + update
        
        # Save current value as previous best if it improves
        condition = tf.reduce_sum(tf.square(grad)) < tf.reduce_sum(tf.square(tf.gradients(var, [prev_best])[0]))
        prev_best.assign(tf.where(condition, var, prev_best))
        
        # Apply the update to the variable
        var.assign(new_var)
        
        return tf.group(*[var, m, v, prev_best])
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # For sparse updates, we convert to dense for now (not ideal but functional)
        return self._resource_apply_dense(
            tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(var))),
            var,
            apply_state=apply_state)
    
    def _levy_flight(self, shape, dtype):
        """Generate Levy flight for exploration."""
        beta = 1.5
        sigma = tf.pow(
            (tf.exp(tf.math.lgamma(1 + beta)) * tf.sin(math.pi * beta / 2)) / 
            (tf.exp(tf.math.lgamma((1 + beta) / 2)) * beta * tf.pow(2, (beta - 1) / 2)),
            1 / beta
        )
        u = tf.random.normal(shape, dtype=dtype) * sigma
        v = tf.random.normal(shape, dtype=dtype)
        step = u / tf.pow(tf.abs(v), 1 / beta)
        return step * 0.01  # Scale down to avoid too large steps
    
    def _chaotic_map(self, dtype):
        """Generate chaotic values using logistic map."""
        x = tf.constant(0.7, dtype=dtype)  # Initial value
        iterations = tf.cast(10 * self._iterations / self._max_iterations, tf.int32) + 1
        
        for _ in range(iterations):
            x = 4 * x * (1 - x)
        
        return x
    
    def set_max_iterations(self, max_iterations):
        """Set the maximum number of iterations for phase calculation."""
        self._max_iterations = max_iterations
    
    def get_config(self):
        config = super(AquilaOptimizer, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
        })
        return config


class AquilaOptimizerCallback(Callback):
    """Callback to update the Aquila Optimizer's max iterations."""
    
    def on_train_begin(self, logs=None):
        if isinstance(self.model.optimizer, AquilaOptimizer):
            # Set max iterations to the total number of batches * epochs
            max_iterations = self.params['epochs'] * self.params['steps']
            self.model.optimizer.set_max_iterations(max_iterations)
            print(f"Set Aquila Optimizer max iterations to {max_iterations}")


# Focal Loss implementation for imbalanced classes
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss implementation to focus more on hard-to-classify examples.
    
    Args:
        gamma: Focusing parameter. Higher values give more weight to hard examples.
        alpha: Weighting factor for the positive class.
        
    Returns:
        Focal loss function compatible with Keras.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip prediction values to avoid log(0) error
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * K.log(y_pred)
        
        # Apply class weights using alpha
        if alpha is not None:
            # For multi-class, alpha should be a tensor of shape (num_classes,)
            cross_entropy = alpha * cross_entropy
        
        # Apply focusing parameter using gamma
        if gamma is not None:
            # Calculate focusing factor: (1 - p_t)^gamma
            # where p_t is the probability of the true class
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focusing_factor = K.pow(1.0 - p_t, gamma)
            cross_entropy = focusing_factor * cross_entropy
        
        # Sum over classes and average over samples
        return K.mean(K.sum(cross_entropy, axis=-1))
    
    return focal_loss_fixed


# Enhanced Weighted Kurtosis and Skewness (WKS) feature extraction
def calculate_weighted_kurtosis(data, weights=None):
    """
    Calculate weighted kurtosis for time series data with improved stability.
    
    Args:
        data: Numpy array of time series data
        weights: Optional weights for each time step
        
    Returns:
        Weighted kurtosis value
    """
    if weights is None:
        # Use uniform weights if none provided
        weights = np.ones_like(data) / len(data)
    else:
        # Normalize weights
        weights = weights / np.sum(weights)
    
    # Calculate weighted mean
    weighted_mean = np.sum(data * weights)
    
    # Calculate weighted variance with stability check
    weighted_var = np.sum(weights * (data - weighted_mean)**2)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    weighted_var = max(weighted_var, epsilon)
    
    # Calculate weighted kurtosis with improved numerical stability
    weighted_kurtosis = np.sum(weights * (data - weighted_mean)**4) / (weighted_var**2)
    
    # Subtract 3 to make normal distribution have kurtosis of 0
    # Apply clipping to avoid extreme values that could destabilize training
    return np.clip(weighted_kurtosis - 3, -100, 100)  # Excess kurtosis (normal distribution has kurtosis=3)


def calculate_weighted_skewness(data, weights=None):
    """
    Calculate weighted skewness for time series data with improved stability.
    
    Args:
        data: Numpy array of time series data
        weights: Optional weights for each time step
        
    Returns:
        Weighted skewness value with enhanced stability
    """
    if weights is None:
        # Use uniform weights if none provided
        weights = np.ones_like(data) / len(data)
    else:
        # Normalize weights
        weights = weights / np.sum(weights)
    
    # Calculate weighted mean
    weighted_mean = np.sum(weights * data)
    
    # Calculate weighted variance with stability check
    weighted_variance = np.sum(weights * ((data - weighted_mean) ** 2))
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    weighted_variance = max(weighted_variance, epsilon)
    weighted_std = np.sqrt(weighted_variance)
    
    # Calculate weighted skewness with improved numerical stability
    weighted_skewness = np.sum(weights * ((data - weighted_mean) ** 3)) / (weighted_std ** 3)
    
    # Apply clipping to avoid extreme values that could destabilize training
    return np.clip(weighted_skewness, -100, 100)


def extract_wks_features(data, window_size=20, step_size=10):
    """
    Extract Weighted Kurtosis and Skewness (WKS) features from time series data.
    Enhanced version with improved stability and feature extraction.
    
    Args:
        data: Numpy array of time series data (samples, features)
        window_size: Size of sliding window
        step_size: Step size for sliding window
        
    Returns:
        WKS features with enhanced discriminative power
    """
    if len(data.shape) == 1:
        # Handle 1D data
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    n_windows = max(1, (n_samples - window_size) // step_size + 1)
    
    # Initialize feature arrays
    kurtosis_features = np.zeros((n_windows, n_features))
    skewness_features = np.zeros((n_windows, n_features))
    
    # Create exponential weights that give more importance to recent data points
    # This helps better detect sudden changes in the signal (important for fault detection)
    exp_weights = np.exp(np.linspace(-1, 0, window_size))
    exp_weights = exp_weights / np.sum(exp_weights)  # Normalize
    
    # Linear weights that increase importance toward the middle of the window
    # This helps detect sustained anomalies in the middle of the window
    mid_point = window_size // 2
    lin_weights = np.ones(window_size)
    lin_weights[:mid_point] = np.linspace(0.7, 1.0, mid_point)
    lin_weights[mid_point:] = np.linspace(1.0, 0.7, window_size - mid_point)
    lin_weights = lin_weights / np.sum(lin_weights)  # Normalize
    
    # Extract features for each window with multiple weighting schemes
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, n_samples)
        window_data = data[start_idx:end_idx, :]
        
        # Calculate weighted kurtosis and skewness for each feature
        for j in range(n_features):
            feature_data = window_data[:, j]
            
            # Use exponential weights for kurtosis (better for detecting sudden changes)
            kurtosis_features[i, j] = calculate_weighted_kurtosis(feature_data, 
                                                                exp_weights[:len(feature_data)])
            
            # Use linear weights for skewness (better for detecting sustained anomalies)
            skewness_features[i, j] = calculate_weighted_skewness(feature_data, 
                                                                lin_weights[:len(feature_data)])
    
    # Combine features
    wks_features = np.hstack((kurtosis_features.flatten(), skewness_features.flatten()))
    
    # Apply feature normalization to prevent any single feature from dominating
    # Use robust scaling to handle outliers
    wks_mean = np.mean(wks_features)
    wks_std = np.std(wks_features) + 1e-10  # Add epsilon to avoid division by zero
    wks_features = (wks_features - wks_mean) / wks_std
    
    return wks_features


class WKSLayer(layers.Layer):
    """
    Enhanced custom layer for Weighted Kurtosis and Skewness (WKS) feature extraction.
    Implemented using TensorFlow operations for graph compatibility with improved
    stability and feature extraction capabilities for underrepresented classes.
    """
    def __init__(self, window_size=20, step_size=10, **kwargs):
        super(WKSLayer, self).__init__(**kwargs)
        self.window_size = window_size
        self.step_size = step_size
    
    def build(self, input_shape):
        # No trainable weights needed
        super(WKSLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Extract shape information
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        n_features = tf.shape(inputs)[2]
        
        # Calculate number of windows
        n_windows = (seq_length - self.window_size) // self.step_size + 1
        
        # Create exponential weights that give more importance to recent data points
        # This helps better detect sudden changes in the signal (important for fault detection)
        exp_weights = tf.exp(tf.linspace(-1.0, 0.0, self.window_size))
        exp_weights = exp_weights / tf.reduce_sum(exp_weights)  # Normalize
        exp_weights = tf.reshape(exp_weights, [1, self.window_size, 1])  # Reshape for broadcasting
        
        # Linear weights that increase importance toward the middle of the window
        # This helps detect sustained anomalies in the middle of the window
        mid_point = self.window_size // 2
        lin_weights_first_half = tf.linspace(0.7, 1.0, mid_point)
        lin_weights_second_half = tf.linspace(1.0, 0.7, self.window_size - mid_point)
        lin_weights = tf.concat([lin_weights_first_half, lin_weights_second_half], axis=0)
        lin_weights = lin_weights / tf.reduce_sum(lin_weights)  # Normalize
        lin_weights = tf.reshape(lin_weights, [1, self.window_size, 1])  # Reshape for broadcasting
        
        # Initialize output tensors for kurtosis and skewness
        kurtosis_features = []
        skewness_features = []
        
        # Loop over windows
        for i in range(n_windows):
            window_start = i * self.step_size
            window_end = window_start + self.window_size
            window_data = inputs[:, window_start:window_end, :]
            
            # Calculate weighted mean using exponential weights for kurtosis
            weighted_data_exp = window_data * exp_weights
            exp_mean = tf.reduce_sum(weighted_data_exp, axis=1, keepdims=True)
            
            # Calculate weighted mean using linear weights for skewness
            weighted_data_lin = window_data * lin_weights
            lin_mean = tf.reduce_sum(weighted_data_lin, axis=1, keepdims=True)
            
            # Calculate centered data
            exp_centered_data = window_data - exp_mean
            lin_centered_data = window_data - lin_mean
            
            # Calculate weighted variance
            exp_variance = tf.reduce_sum(exp_weights * tf.square(exp_centered_data), axis=1)
            lin_variance = tf.reduce_sum(lin_weights * tf.square(lin_centered_data), axis=1)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            exp_variance = tf.maximum(exp_variance, epsilon)
            lin_variance = tf.maximum(lin_variance, epsilon)
            
            # Calculate weighted kurtosis with exponential weights
            exp_fourth_moment = tf.reduce_sum(exp_weights * tf.pow(exp_centered_data, 4), axis=1)
            kurtosis = exp_fourth_moment / tf.square(exp_variance)
            kurtosis = kurtosis - 3.0  # Excess kurtosis
            
            # Calculate weighted skewness with linear weights
            lin_third_moment = tf.reduce_sum(lin_weights * tf.pow(lin_centered_data, 3), axis=1)
            skewness = lin_third_moment / tf.pow(tf.sqrt(lin_variance), 3)
            
            # Apply clipping to avoid extreme values
            kurtosis = tf.clip_by_value(kurtosis, -100.0, 100.0)
            skewness = tf.clip_by_value(skewness, -100.0, 100.0)
            
            # Add to feature lists
            kurtosis_features.append(kurtosis)
            skewness_features.append(skewness)
        
        # Stack all window features
        kurtosis_features = tf.stack(kurtosis_features, axis=1)
        skewness_features = tf.stack(skewness_features, axis=1)
        
        # Reshape to 2D tensor [batch_size, n_windows*n_features]
        kurtosis_features = tf.reshape(kurtosis_features, [batch_size, n_windows * n_features])
        skewness_features = tf.reshape(skewness_features, [batch_size, n_windows * n_features])
        
        # Concatenate kurtosis and skewness features
        wks_features = tf.concat([kurtosis_features, skewness_features], axis=1)
        
        # Apply feature normalization to prevent any single feature from dominating
        wks_mean = tf.reduce_mean(wks_features, axis=1, keepdims=True)
        wks_std = tf.math.reduce_std(wks_features, axis=1, keepdims=True) + epsilon
        wks_features = (wks_features - wks_mean) / wks_std
        
        return wks_features
    
    def compute_output_shape(self, input_shape):
        # Calculate number of windows
        n_windows = (input_shape[1] - self.window_size) // self.step_size + 1
        
        # Calculate output shape: batch_size, n_windows * n_features * 2 (kurtosis and skewness)
        output_shape = (input_shape[0], n_windows * input_shape[2] * 2)
        
        return output_shape
    
    def get_config(self):
        config = super(WKSLayer, self).get_config()
        config.update({
            'window_size': self.window_size,
            'step_size': self.step_size
        })
        return config


class SIAOAttention(layers.Layer):
    """
    Self-attention mechanism for time series data.
    Enhanced with multi-head attention and improved scaling.
    """
    def __init__(self, units=128, num_heads=4, **kwargs):
        super(SIAOAttention, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads
        
    def build(self, input_shape):
        # Query, Key, Value projections for each head
        self.query_dense = layers.Dense(self.units, use_bias=False)
        self.key_dense = layers.Dense(self.units, use_bias=False)
        self.value_dense = layers.Dense(self.units, use_bias=False)
        
        # Output projection
        self.output_dense = layers.Dense(input_shape[-1])
        
        # Layer normalization for better training stability
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        super(SIAOAttention, self).build(input_shape)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Apply layer normalization
        x = self.layernorm(inputs)
        
        # Linear projections
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention weights to values
        output = tf.matmul(attention_weights, v)
        
        # Reshape back to original dimensions
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.units))
        
        # Final projection
        output = self.output_dense(output)
        
        # Add residual connection
        output = output + inputs
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(SIAOAttention, self).get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config


# Legacy SIAOAttention for backward compatibility
class SIAOAttentionLegacy(layers.Layer):
    """Self-Improved Attention mechanism for the SIAO-CNN-ORNN model (original version)."""
    
    def __init__(self, units, **kwargs):
        super(SIAOAttentionLegacy, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.b = None
        self.V = None
    
    def build(self, input_shape):
        # Create weights for attention mechanism
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.V = self.add_weight(
            name="attention_context_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super(SIAOAttentionLegacy, self).build(input_shape)
    
    def call(self, inputs):
        # Apply attention mechanism
        # inputs shape: (batch_size, time_steps, features)
        
        # Calculate attention scores
        # ui = tanh(W * hi + b)
        ui = K.tanh(K.dot(inputs, self.W) + self.b)
        
        # Calculate attention weights
        # ai = softmax(V^T * ui)
        ai = K.softmax(K.dot(ui, self.V))
        
        # Apply attention weights to input sequence
        # z = sum(ai * hi)
        z = inputs * ai
        z = K.sum(z, axis=1)
        
        return z
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def get_config(self):
        config = super(SIAOAttentionLegacy, self).get_config()
        config.update({"units": self.units})
        return config


def build_siao_enhanced(input_shape, num_classes, dropout_rate=0.5, recurrent_dropout=0.0, 
                       l2_reg=0.002, learning_rate=0.001, use_gru=None, attention_units=256, 
                       filters_multiplier=1.5, dense_units_multiplier=1.5, num_heads=4,
                       use_aquila_optimizer=False, use_wks=True, wks_window_size=20, wks_step_size=10,
                       use_focal_loss=True, focal_gamma=2.0, focal_alpha=0.25):
    """
    Build an enhanced SIAO (Self-Improving Architecture Optimization) CNN-RNN hybrid model
    with increased parameters for higher accuracy.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization (default: 0.5)
        recurrent_dropout: Dropout rate for recurrent layers
        l2_reg: L2 regularization factor (default: 0.002)
        learning_rate: Learning rate for optimizer
        use_gru: Whether to use GRU cells (True), LSTM cells (False), or auto-detect (None)
        attention_units: Number of units in attention layer
        filters_multiplier: Multiplier for number of filters in CNN layers
        dense_units_multiplier: Multiplier for number of units in dense layers
        num_heads: Number of attention heads
        use_aquila_optimizer: Whether to use the Aquila Optimizer
        use_wks: Whether to use Weighted Kurtosis and Skewness features (default: True)
        wks_window_size: Window size for WKS feature extraction
        wks_step_size: Step size for WKS feature extraction
        use_focal_loss: Whether to use focal loss for imbalanced classes (default: True)
        focal_gamma: Focusing parameter for focal loss
        focal_alpha: Weighting factor for focal loss
        
    Returns:
        Compiled Keras SIAO enhanced model
    """
    # Auto-detect GRU vs LSTM based on hardware if not specified
    if use_gru is None:
        # Use GRU on Apple Silicon for better performance
        use_gru = IS_APPLE_SILICON
        print(f"Auto-detected {'GRU' if use_gru else 'LSTM'} for recurrent layer based on hardware")
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Add WKS feature extraction if enabled
    if use_wks:
        print(f"Using WKS feature extraction with window_size={wks_window_size}, step_size={wks_step_size}")
        wks_features = WKSLayer(window_size=wks_window_size, step_size=wks_step_size)(inputs)
        wks_features = layers.Reshape((1, -1))(wks_features)  # Reshape for concatenation later
    
    # CNN branch for spatial feature extraction
    # Use smaller filters (3, 5, 7) with more filters for better feature extraction
    filter_sizes = [3, 5, 7]
    conv_blocks = []
    
    for filter_size in filter_sizes:
        # Calculate number of filters based on multiplier
        num_filters = int(32 * filters_multiplier)
        
        # 1D convolution with L2 regularization
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(inputs)
        
        # Batch normalization for better training stability
        conv = layers.BatchNormalization()(conv)
        
        # Max pooling to reduce dimensionality
        conv = layers.MaxPooling1D(pool_size=2)(conv)
        
        # Dropout for regularization
        conv = layers.Dropout(dropout_rate)(conv)
        
        # Add to list of conv blocks
        conv_blocks.append(conv)
    
    # Concatenate CNN branches
    x = layers.Concatenate(axis=-1)(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    
    # Add another convolutional layer to combine features
    x = layers.Conv1D(
        filters=int(64 * filters_multiplier),
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Recurrent layer (GRU or LSTM)
    if use_gru:
        # GRU layer
        rnn = layers.GRU(
            units=int(128 * dense_units_multiplier),
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
    else:
        # LSTM layer
        rnn = layers.LSTM(
            units=int(128 * dense_units_multiplier),
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
    
    # Add batch normalization after recurrent layer
    rnn = layers.BatchNormalization()(rnn)
    
    # Apply multi-head self-attention
    attention = SIAOAttention(units=attention_units, num_heads=num_heads)(rnn)
    
    # Combine with WKS features if enabled
    if use_wks:
        # Flatten attention output
        attention_flat = layers.Flatten()(attention)
        
        # Flatten WKS features
        wks_flat = layers.Flatten()(wks_features)
        
        # Concatenate attention output with WKS features
        x = layers.Concatenate()([attention_flat, wks_flat])
    else:
        # Just flatten attention output
        x = layers.Flatten()(attention)
    
    # Dense layers with increasing dropout for better regularization
    x = layers.Dense(
        units=int(256 * dense_units_multiplier),
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        units=int(128 * dense_units_multiplier),
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Define loss function
    if use_focal_loss:
        # Enhanced focal loss with higher gamma for harder examples
        loss_function = focal_loss(gamma=focal_gamma, alpha=focal_alpha)
        print(f"Using focal loss with gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        # Apply label smoothing to categorical crossentropy for better generalization
        loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        print("Using categorical crossentropy with label smoothing=0.1")
        
    # Select optimizer
    if use_aquila_optimizer:
        optimizer = AquilaOptimizer(learning_rate=learning_rate)
        print("Using Aquila Optimizer")
    else:
        if IS_APPLE_SILICON:
            optimizer = LegacyAdam(learning_rate=learning_rate)
            print("Using Legacy Adam optimizer for better M1/M2 Mac performance")
        else:
            optimizer = Adam(learning_rate=learning_rate)
            print("Using standard Adam optimizer")
    
    # Add more comprehensive metrics for better evaluation
    metrics = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )
    
    return model


def build_siao_cnn_ornn(input_shape, num_classes, use_gru=None, dropout_rate=0.3, recurrent_dropout=0.0, l2_reg=0.001, learning_rate=0.001, use_aquila_optimizer=False):
    """
    Build an optimized SIAO (Self-Improving Architecture Optimization) CNN-RNN hybrid model.
    
    This model combines CNN for spatial feature extraction with RNN for temporal dependencies,
    and adds a self-attention mechanism to focus on the most important features.
    
    Args:
        input_shape: Shape of input data (time_steps, features)
        num_classes: Number of output classes
        use_gru: Whether to use GRU cells instead of LSTM. If None, auto-detects based on hardware.
        dropout_rate: Dropout rate for regularization
        recurrent_dropout: Dropout rate for recurrent layers
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for optimizer
        use_aquila_optimizer: Whether to use the Aquila Optimizer
        
    Returns:
        Compiled Keras SIAO CNN-RNN hybrid model
    """
    # If use_gru is None, determine based on hardware
    if use_gru is None:
        use_gru = not IS_APPLE_SILICON  # Use LSTM on Apple Silicon, GRU otherwise
        
    # Set recurrent_dropout to 0.0 for Apple Silicon to enable cuDNN acceleration
    if IS_APPLE_SILICON:
        recurrent_dropout = 0.0
    
    # Choose RNN layer type - Use LSTM by default for better performance on sequential data
    if use_gru:
        rnn_layer = layers.GRU
    else:
        rnn_layer = layers.LSTM  # Changed from SimpleRNN to LSTM for better performance
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Normalize input data
    x = layers.BatchNormalization()(inputs)
    
    # First CNN block with residual connection
    conv1 = layers.Conv1D(
        filters=128,  # Increased from 64 to 128
        kernel_size=7,  # Increased from 5 to 7 to capture longer patterns
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'  # Better initialization for deep networks
    )(x)
    x = layers.BatchNormalization()(conv1)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)  # Spatial dropout works better for CNNs
    
    # Store for residual connection
    res1 = x
    
    # Second CNN block with increased filters
    conv2 = layers.Conv1D(
        filters=256,  # Increased from 128 to 256
        kernel_size=5,  # Increased from 3 to 5
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(conv2)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(dropout_rate/2)(x)
    
    # Third CNN block with dilation for capturing longer patterns
    conv3 = layers.Conv1D(
        filters=256,  # Increased from 128 to 256
        kernel_size=5,  # Increased from 3 to 5
        padding='same',
        dilation_rate=2,  # Dilated convolution to capture longer patterns
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(conv3)
    x = layers.Activation('relu')(x)
    
    # Fourth CNN block with larger dilation
    conv4 = layers.Conv1D(
        filters=256,  # New layer
        kernel_size=3,
        padding='same',
        dilation_rate=4,  # Larger dilation to capture even longer patterns
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(conv4)
    x = layers.Activation('relu')(x)
    
    # Add residual connection from first CNN block - always use 1x1 conv to match dimensions
    # This ensures the channel dimensions match (128 vs 256)
    res1_matched = layers.Conv1D(filters=256, kernel_size=1, padding='same')(res1)
    x = layers.Add()([x, res1_matched])
    
    # Apply SIAO attention mechanism with increased units
    attention_units = 256  # Increased from 128 to 256
    # Reshape for attention mechanism
    x_reshaped = layers.Reshape((x.shape[1], x.shape[2]))(x)
    attention_output = SIAOAttention(attention_units)(x_reshaped)
    
    # Reshape back and continue with the model
    # Reshape to (batch_size, 1, attention_units)
    attention_output = layers.Reshape((1, attention_units))(attention_output)
    x = layers.Concatenate(axis=1)([x, attention_output])
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(dropout_rate/2)(x)
    
    # Multi-layer bidirectional RNN for better sequence modeling
    # First RNN layer with increased units
    rnn1 = layers.Bidirectional(
        rnn_layer(128,  # Increased from 64 to 128
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2),  # Added recurrent regularization
                kernel_initializer='glorot_uniform')
    )(x)
    
    # Add a residual connection - always use 1x1 conv to match dimensions
    # This ensures the channel dimensions match correctly
    x_matched = layers.Conv1D(filters=rnn1.shape[2], kernel_size=1, padding='same')(x)
    x = layers.Add()([x_matched, rnn1])
    
    x = layers.LayerNormalization()(x)  # Normalize for stable training
    
    # Second RNN layer with increased units
    if use_gru and not IS_APPLE_SILICON:
        x = layers.Bidirectional(
            layers.GRU(
                256, 
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg)
            )
        )(x)
    else:
        x = layers.Bidirectional(
            layers.LSTM(
                256, 
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=0.0,  # Set to 0 for better M1/M2 Mac performance
                kernel_regularizer=regularizers.l2(l2_reg)
            )
        )(x)
    
    # Third RNN layer (new)
    rnn3 = layers.Bidirectional(
        rnn_layer(256,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2),
                kernel_initializer='glorot_uniform')
    )(x)
    
    # Multiple pooling strategies for better feature extraction
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    attention_pool = SIAOAttention(256)(x)  # Additional attention-based pooling
    
    # Concatenate different pooling results
    x = layers.Concatenate()([avg_pool, max_pool, attention_pool])
    
    # Dense layers with increased capacity
    x = layers.Dense(
        512,  # Increased from 256 to 512
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second dense layer with increased capacity
    x = layers.Dense(
        256,  # Increased from 128 to 256
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Third dense layer (new)
    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Choose optimizer based on parameter
    if use_aquila_optimizer and SIAO_AVAILABLE:
        optimizer = aquila_optimizer.AquilaOptimizer(learning_rate=learning_rate)
    elif use_aquila_optimizer:
        optimizer = AquilaOptimizer(learning_rate=learning_rate)
    elif IS_APPLE_SILICON:
        # Use legacy Adam for better M1/M2 Mac performance
        optimizer = LegacyAdam(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model


def train_siao_enhanced(X_train, y_train, X_val=None, y_val=None, input_shape=None, num_classes=None, 
                       epochs=100, batch_size=16, early_stopping=True, class_weights=None,
                       dropout_rate=0.5, l2_reg=0.002, learning_rate=0.001, use_gru=None,
                       attention_units=256, filters_multiplier=1.5, dense_units_multiplier=1.5, 
                       num_heads=4, use_aquila_optimizer=False, use_wks=True, 
                       wks_window_size=20, wks_step_size=10, use_focal_loss=True,
                       focal_gamma=2.0, focal_alpha=0.25, custom_class_weights=True):
    """
    Train the enhanced SIAO CNN-RNN model with advanced training techniques.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        input_shape: Shape of input data
        num_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size for training
        early_stopping: Whether to use early stopping
        class_weights: Class weights for imbalanced data
        dropout_rate: Dropout rate for regularization (default: 0.5)
        l2_reg: L2 regularization factor (default: 0.002)
        learning_rate: Learning rate for optimizer
        use_gru: Whether to use GRU cells
        attention_units: Number of units in attention layer
        filters_multiplier: Multiplier for number of filters in CNN layers
        dense_units_multiplier: Multiplier for number of units in dense layers
        num_heads: Number of attention heads
        use_aquila_optimizer: Whether to use the Aquila Optimizer
        use_wks: Whether to use Weighted Kurtosis and Skewness features (default: True)
        wks_window_size: Window size for WKS feature extraction
        wks_step_size: Step size for WKS feature extraction
        use_focal_loss: Whether to use focal loss for imbalanced classes (default: True)
        focal_gamma: Focusing parameter for focal loss
        focal_alpha: Weighting factor for focal loss
        custom_class_weights: Whether to use custom class weights for underrepresented classes
        
    Returns:
        tuple: (model, history)
    """
    # Determine input shape and number of classes if not provided
    if input_shape is None:
        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"Using input shape: {input_shape}")
    
    if num_classes is None:
        num_classes = y_train.shape[1]
        print(f"Using num_classes: {num_classes}")
    
    # Prepare class weights if needed
    if custom_class_weights and class_weights is None:
        # Enhanced custom class weights for better detection of underrepresented classes
        # Based on analysis of the dataset and focusing on the classes that need improvement
        # 0: pressurizer_porv, 1: power_change, 2: steady_state, 3: pump_failure, 4: sg_tube_rupture, 5: feedwater_break
        custom_weights = {
            0: 1.5,   # pressurizer_porv (common class)
            1: 8.0,   # power_change (highly underrepresented, needs more weight)
            2: 2.5,   # steady_state (needs better detection)
            3: 1.0,   # pump_failure (well-detected class)
            4: 1.0,   # sg_tube_rupture (well-detected class)
            5: 5.0    # feedwater_break (underrepresented, needs more weight)
        }
        print(f"Using enhanced class weights for better detection of underrepresented classes:")
        print(f"  - pressurizer_porv: {custom_weights[0]}")
        print(f"  - power_change: {custom_weights[1]}")
        print(f"  - steady_state: {custom_weights[2]}")
        print(f"  - pump_failure: {custom_weights[3]}")
        print(f"  - sg_tube_rupture: {custom_weights[4]}")
        print(f"  - feedwater_break: {custom_weights[5]}")
        
        # Analyze training data class distribution if available
        if y_train is not None and len(y_train.shape) > 1:
            train_classes = np.argmax(y_train, axis=1)
            class_counts = np.bincount(train_classes, minlength=num_classes if num_classes else 6)
            print("\nTraining class distribution:")
            for i, count in enumerate(class_counts):
                print(f"  Class {i}: {count} samples ({count/len(train_classes)*100:.2f}%)")
            
            # Calculate inverse frequency class weights for comparison
            total_samples = len(train_classes)
            inv_freq_weights = {}
            for i, count in enumerate(class_counts):
                if count > 0:
                    inv_freq_weights[i] = total_samples / (len(class_counts) * count)
                else:
                    inv_freq_weights[i] = 1.0
            print(f"\nInverse frequency class weights: {inv_freq_weights}")
            print("Using custom weights instead of inverse frequency weights for better control.")
        
        class_weights = custom_weights
        print(f"\nFinal class weights being used: {class_weights}")
    elif class_weights is not None:
        print(f"Using provided class weights: {class_weights}")
    else:
        print("No class weights being used.")
        
    # Print hardware and optimizer information
    if IS_APPLE_SILICON:
        print("Running on Apple Silicon")
    else:
        print("Running on standard hardware")
    
    # Build the enhanced SIAO model
    model = build_siao_enhanced(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        learning_rate=learning_rate,
        use_gru=use_gru,
        attention_units=attention_units,
        filters_multiplier=filters_multiplier,
        dense_units_multiplier=dense_units_multiplier,
        num_heads=num_heads,
        use_aquila_optimizer=use_aquila_optimizer,
        use_wks=use_wks,
        wks_window_size=wks_window_size,
        wks_step_size=wks_step_size,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha
    )
    
    # Print model summary and parameter count
    model.summary()
    trainable_count = sum(K.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(K.count_params(w) for w in model.non_trainable_weights)
    print(f"Total parameters: {trainable_count + non_trainable_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    
    # Setup callbacks with improved monitoring strategy
    callbacks = []
    
    # Create checkpoints directory
    os.makedirs('models/checkpoints', exist_ok=True)
    
    # Save models based on different metrics to ensure we capture the best model
    # Monitor val_loss (primary metric when accuracy is unstable)
    callbacks.append(ModelCheckpoint(
        filepath=f'models/checkpoints/siao_enhanced_best_loss.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ))
    
    # Monitor val_auc (better than accuracy for imbalanced datasets)
    callbacks.append(ModelCheckpoint(
        filepath=f'models/checkpoints/siao_enhanced_best_auc.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',  # AUC should be maximized
        verbose=1
    ))
    
    # Monitor val_accuracy (traditional metric)
    callbacks.append(ModelCheckpoint(
        filepath=f'models/checkpoints/siao_enhanced_best_acc.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ))
    
    # Early stopping based on validation loss with increased patience
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',  # More reliable than accuracy for imbalanced data
            patience=20,  # Increased patience to allow for more exploration
            restore_best_weights=True,
            verbose=1
        ))
    
    # Learning rate scheduler with more gradual reduction
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # More gradual reduction (was 0.2)
        patience=10,  # Increased patience
        min_lr=1e-7,  # Lower minimum learning rate
        verbose=1
    ))
    
    # Add custom callback to monitor composite metrics
    class CompositeMetricCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Calculate composite metric (weighted combination of multiple metrics)
            val_loss = logs.get('val_loss', 0)
            val_acc = logs.get('val_accuracy', 0)
            val_auc = logs.get('val_auc', 0)
            val_precision = logs.get('val_precision', 0)
            val_recall = logs.get('val_recall', 0)
            
            # Weighted combination favoring AUC when accuracy is low
            if val_acc < 0.1:
                composite = (0.6 * val_auc) + (0.2 * val_precision) + (0.2 * val_recall) - (0.2 * val_loss)
            else:
                composite = (0.4 * val_acc) + (0.3 * val_auc) + (0.15 * val_precision) + (0.15 * val_recall) - (0.2 * val_loss)
            
            print(f"Epoch {epoch+1}: Composite metric = {composite:.4f}")
            logs['composite_metric'] = composite
    
    callbacks.append(CompositeMetricCallback())
    
    # Add AquilaOptimizerCallback if using Aquila Optimizer
    if use_aquila_optimizer:
        callbacks.append(AquilaOptimizerCallback())
    
    # Add a custom callback to monitor metrics during training
    class MetricsMonitor(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            metrics = ["loss", "accuracy", "auc", "precision", "recall", 
                      "val_loss", "val_accuracy", "val_auc", "val_precision", "val_recall"]
            
            print(f"\nEpoch {epoch+1} metrics:")
            for metric in metrics:
                if metric in logs:
                    print(f"  {metric}: {logs[metric]:.4f}")
    
    callbacks.append(MetricsMonitor())
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Print training time
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    return model, history


def train_siao_cnn_ornn(X_train, y_train, X_val=None, y_val=None, input_shape=None, num_classes=None, 
                      epochs=100, batch_size=16, early_stopping=True, class_weights=None,
                      use_aquila_optimizer=True, use_gru=False, dropout_rate=0.4, l2_reg=0.0005):
    """
    Train the SIAO CNN-RNN model with advanced training techniques.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        input_shape: Shape of input data
        num_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size for training
        early_stopping: Whether to use early stopping
        class_weights: Weights for imbalanced classes
        use_aquila_optimizer: Whether to use the Aquila Optimizer
        use_gru: Whether to use GRU cells instead of SimpleRNN
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        
    Returns:
        Trained model and training history
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
    from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
    import os
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Infer input shape and number of classes if not provided
    if input_shape is None and X_train is not None:
        input_shape = X_train.shape[1:]
    
    if num_classes is None and y_train is not None:
        if len(y_train.shape) > 1:
            num_classes = y_train.shape[1]  # One-hot encoded
        else:
            num_classes = len(np.unique(y_train))  # Class indices
    
    # Advanced data augmentation for time series to improve model generalization
    print("Applying advanced data augmentation to increase training samples...")
    augmented_X = []
    augmented_y = []
    
    # Original data
    augmented_X.extend(X_train)
    augmented_y.extend(y_train)
    
    # 1. Add Gaussian noise (multiple noise levels)
    for noise_level in [0.02, 0.05, 0.08]:  # Multiple noise levels
        for i in range(len(X_train)):
            noisy_sample = X_train[i] + np.random.normal(0, noise_level, X_train[i].shape)
            augmented_X.append(noisy_sample)
            augmented_y.append(y_train[i])
    
    # 2. Time warping (both stretch and compress)
    stretch_factors = [0.9, 1.1]  # Compress by 10% and stretch by 10%
    for stretch_factor in stretch_factors:
        for i in range(len(X_train)):
            if X_train[i].shape[0] > 10:  # Only if enough time steps
                time_steps = X_train[i].shape[0]
                new_time_steps = int(time_steps * stretch_factor)
                
                # Handle both stretching and compressing
                if new_time_steps <= time_steps:
                    # Compressing
                    indices = np.linspace(0, time_steps-1, new_time_steps)
                    warped_sample = np.zeros((time_steps, X_train[i].shape[1]))
                    
                    # Interpolate values and pad
                    for j in range(X_train[i].shape[1]):
                        warped_channel = np.interp(indices, np.arange(time_steps), X_train[i][:, j])
                        warped_sample[:new_time_steps, j] = warped_channel
                        warped_sample[new_time_steps:, j] = X_train[i][-(time_steps-new_time_steps):, j]
                else:
                    # Stretching
                    indices = np.linspace(0, time_steps-1, new_time_steps)
                    warped_sample = np.zeros((time_steps, X_train[i].shape[1]))
                    
                    # Interpolate values and truncate
                    for j in range(X_train[i].shape[1]):
                        warped_channel = np.interp(indices, np.arange(time_steps), X_train[i][:, j])
                        warped_sample[:, j] = warped_channel[:time_steps]
                
                augmented_X.append(warped_sample)
                augmented_y.append(y_train[i])
    
    # 3. Magnitude scaling (amplitude variation)
    for scale_factor in [0.9, 1.1]:  # Scale down by 10% and up by 10%
        for i in range(len(X_train)):
            scaled_sample = X_train[i] * scale_factor
            augmented_X.append(scaled_sample)
            augmented_y.append(y_train[i])
    
    # 4. Channel dropout (randomly zero out some channels to improve robustness)
    for i in range(len(X_train)):
        if X_train[i].shape[1] > 5:  # Only if enough channels
            channel_dropout_sample = X_train[i].copy()
            # Randomly select 10% of channels to zero out
            num_channels_to_drop = max(1, int(0.1 * X_train[i].shape[1]))
            channels_to_drop = np.random.choice(X_train[i].shape[1], num_channels_to_drop, replace=False)
            channel_dropout_sample[:, channels_to_drop] = 0
            augmented_X.append(channel_dropout_sample)
            augmented_y.append(y_train[i])
    
    # 5. Time masking (mask out random time segments)
    for i in range(len(X_train)):
        if X_train[i].shape[0] > 20:  # Only if enough time steps
            time_mask_sample = X_train[i].copy()
            # Mask out a random segment (10-20% of time steps)
            mask_length = np.random.randint(int(0.1 * X_train[i].shape[0]), int(0.2 * X_train[i].shape[0]))
            mask_start = np.random.randint(0, X_train[i].shape[0] - mask_length)
            time_mask_sample[mask_start:mask_start+mask_length, :] = 0
            augmented_X.append(time_mask_sample)
            augmented_y.append(y_train[i])
    
    # 6. Combine augmentations (noise + scaling)
    for i in range(len(X_train)):
        combined_sample = X_train[i] * 1.05 + np.random.normal(0, 0.03, X_train[i].shape)
        augmented_X.append(combined_sample)
        augmented_y.append(y_train[i])
    
    # Convert to numpy arrays
    X_train = np.array(augmented_X)
    y_train = np.array(augmented_y)
    
    print(f"Advanced data augmentation complete. New training set size: {len(X_train)}")
    
    # Balance classes if needed
    if class_weights is None and num_classes > 1:
        print("Checking class distribution...")
        if len(y_train.shape) > 1:  # One-hot encoded
            class_counts = np.sum(y_train, axis=0)
        else:  # Class indices
            class_counts = np.bincount(y_train)
        
        min_class = np.min(class_counts)
        max_class = np.max(class_counts)
        imbalance_ratio = max_class / min_class
        
        if imbalance_ratio > 1.5:  # If classes are imbalanced
            print(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Computing class weights...")
            if len(y_train.shape) > 1:  # One-hot encoded
                class_indices = np.argmax(y_train, axis=1)
            else:  # Class indices
                class_indices = y_train
            
            # Compute class weights
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
            class_weights = dict(enumerate(class_weights))
            print("Class weights:", class_weights)
    
    # Build model with specified parameters
    model = build_siao_cnn_ornn(
        input_shape=input_shape, 
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        recurrent_dropout=dropout_rate/2,
        l2_reg=l2_reg,
        learning_rate=0.001,
        use_gru=use_gru,
        use_aquila_optimizer=use_aquila_optimizer
    )
    
    # Print model summary
    model.summary()
    
    # Prepare callbacks
    callbacks = []
    
    # Add Aquila Optimizer callback if using that optimizer
    if use_aquila_optimizer:
        try:
            callbacks.append(AquilaOptimizerCallback())
        except Exception as e:
            print(f"Error adding Aquila Optimizer callback: {e}")
    
    # Advanced learning rate schedule with warm-up, plateau, and cosine decay
    def lr_schedule(epoch, lr):
        initial_lr = 0.001  # Starting learning rate
        min_lr = 1e-6      # Minimum learning rate
        warmup_epochs = 5   # Number of warmup epochs
        plateau_epochs = 30 # Number of plateau epochs
        decay_epochs = epochs - warmup_epochs - plateau_epochs
        
        if epoch < warmup_epochs:  # Warm-up phase: linear increase
            return initial_lr * (epoch + 1) / warmup_epochs
        elif epoch < warmup_epochs + plateau_epochs:  # Plateau phase: constant learning rate
            return initial_lr
        else:  # Cosine decay phase: gradual decrease
            progress = (epoch - warmup_epochs - plateau_epochs) / decay_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return max(min_lr, initial_lr * cosine_decay)
    
    callbacks.append(LearningRateScheduler(lr_schedule))
    
    # Try to use SWA if available and not on Apple Silicon
    try:
        from tensorflow_addons.optimizers import SWA
        use_swa = not IS_APPLE_SILICON  # Skip SWA on M1/M2 Macs for better performance
    except ImportError:
        print("SWA not available. Skipping.")
        use_swa = False
    
    if use_swa:
        # Stochastic Weight Averaging for better generalization
        swa_start = int(epochs * 0.75)  # Start SWA at 75% of training
        swa_freq = 5  # Average weights every 5 epochs
        callbacks.append(SWA(start_epoch=swa_start, swa_freq=swa_freq, swa_lr=0.0001))
        print(f"Using Stochastic Weight Averaging starting at epoch {swa_start}")
    
    # Gradient clipping to prevent exploding gradients
    from tensorflow.keras.callbacks import Callback
    class GradientClippingCallback(Callback):
        def on_train_begin(self, logs=None):
            self.model.optimizer.clipnorm = 1.0  # Clip gradients to prevent explosion
            self.model.optimizer.clipvalue = 0.5  # Clip gradient values
            
    callbacks.append(GradientClippingCallback())
    
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=20,  # Increased patience for better convergence
            restore_best_weights=True,
            verbose=1
        ))
    
    # Model checkpoint with multiple metrics
    os.makedirs('models/checkpoints', exist_ok=True)
    
    # Checkpoint for best accuracy
    callbacks.append(ModelCheckpoint(
        'models/checkpoints/siao_cnn_ornn_best_acc.h5',
        monitor='val_accuracy' if X_val is not None else 'accuracy',
        save_best_only=True,
        verbose=1
    ))
    
    # Checkpoint for best loss
    callbacks.append(ModelCheckpoint(
        'models/checkpoints/siao_cnn_ornn_best_loss.h5',
        monitor='val_loss' if X_val is not None else 'loss',
        save_best_only=True,
        verbose=1
    ))
    
    # TensorBoard logging
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    callbacks.append(TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    ))
    
    # Train model
    optimizer_name = 'Aquila' if use_aquila_optimizer else ('Legacy Adam' if IS_APPLE_SILICON else 'Adam')
    print(f"\nTraining SIAO-CNN-ORNN model with {optimizer_name} optimizer")
    print(f"Using {'GRU' if use_gru and not IS_APPLE_SILICON else 'LSTM'} cells for recurrent layers")
    print(f"Running on {'Apple Silicon' if IS_APPLE_SILICON else 'standard hardware'}")
    print(f"Training on {len(X_train)} samples, validating on {len(X_val) if X_val is not None else 0} samples")
    
    start_time = time.time()
    
    # Use a higher initial learning rate with the cosine schedule
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
        shuffle=True  # Ensure data is shuffled each epoch
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate model on validation data if available
    if X_val is not None and y_val is not None:
        val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nValidation Results:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"AUC: {val_auc:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        
        # Plot training history
        os.makedirs('analysis/plots', exist_ok=True)
        
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('analysis/plots/siao_cnn_ornn_training_history.png')
        print("Training history plot saved to 'analysis/plots/siao_cnn_ornn_training_history.png'")
    
    # Save final model
    os.makedirs('models/saved', exist_ok=True)
    # Create trained_models directory if it doesn't exist
    os.makedirs('trained_models', exist_ok=True)
    model.save('trained_models/siao_cnn_ornn_final.h5')
    print("Model saved to 'trained_models/siao_cnn_ornn_final.h5'")
    
    return model, history