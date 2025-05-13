"""
Enhanced SIAO-CNN-ORNN model with increased parameters for higher accuracy
Optimized for M1/M2 Mac performance
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
            # Narrowed exploration: More focused search around promising regions
            update = -lr_t * (alpha * r1 * m_hat / (tf.sqrt(v_hat) + self.epsilon))
            
        elif phase == "expanded_exploitation":
            # Expanded exploitation: Exploit promising regions with chaotic maps
            chaotic = self._chaotic_map(tf.shape(var), var_dtype)
            update = -lr_t * (beta * chaotic * m_hat / (tf.sqrt(v_hat) + self.epsilon))
            
        else:  # narrowed_exploitation
            # Narrowed exploitation: Fine-tune the solution
            update = -lr_t * (beta * m_hat / (tf.sqrt(v_hat) + self.epsilon))
        
        # Apply update
        var.assign_add(update)
        
        # Update prev_best if current solution is better (assuming minimization)
        # This would require loss value, which we don't have here
        # For simplicity, we'll just store the current variable value
        prev_best.assign(var)
        
        return var
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Fallback to dense implementation
        return self._resource_apply_dense(
            tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(var))),
            var,
            apply_state=apply_state)
    
    def _levy_flight(self, shape, dtype):
        """Generate Lévy flight random values for expanded exploration."""
        # Lévy flight parameters
        beta = 1.5
        sigma_u = tf.pow((tf.math.gamma(1 + beta) * tf.sin(np.pi * beta / 2)) / 
                         (tf.math.gamma((1 + beta) / 2) * beta * tf.pow(2, (beta - 1) / 2)), 1 / beta)
        sigma_v = 1.0
        
        # Generate Lévy flight step
        u = tf.random.normal(shape, mean=0.0, stddev=sigma_u, dtype=dtype)
        v = tf.random.normal(shape, mean=0.0, stddev=sigma_v, dtype=dtype)
        step = u / tf.pow(tf.abs(v), 1 / beta)
        
        # Normalize step
        return step / tf.reduce_max(tf.abs(step))
    
    def _chaotic_map(self, shape, dtype):
        """Generate chaotic map values for expanded exploitation."""
        # Use logistic map: x_{n+1} = r * x_n * (1 - x_n)
        r = 3.9  # Chaotic behavior for r > 3.57
        x = tf.random.uniform(shape, minval=0.1, maxval=0.9, dtype=dtype)  # Initial value
        
        # Apply logistic map
        for _ in range(10):  # Iterate to get chaotic behavior
            x = r * x * (1 - x)
        
        return x
    
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
        if hasattr(self.model.optimizer, '_max_iterations'):
            # Set max iterations to match total training iterations
            self.model.optimizer._max_iterations = self.params['epochs'] * self.params['steps']

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
        # Clip predictions to avoid numerical instability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate focal loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        
        # Sum over classes
        return K.sum(loss, axis=-1)
    
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
        
        # Residual connection
        return output + inputs
    
    def get_config(self):
        config = super(SIAOAttention, self).get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
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
    # If use_gru is None, determine based on hardware
    if use_gru is None:
        use_gru = not IS_APPLE_SILICON  # Use LSTM on Apple Silicon, GRU otherwise
        
    # Set recurrent_dropout to 0.0 for Apple Silicon to enable cuDNN acceleration
    if IS_APPLE_SILICON:
        recurrent_dropout = 0.0
    
    # Calculate enhanced filter and unit counts
    cnn_filters1 = int(64 * filters_multiplier)
    cnn_filters2 = int(128 * filters_multiplier)
    cnn_filters3 = int(256 * filters_multiplier)
    dense_units1 = int(512 * dense_units_multiplier)
    dense_units2 = int(256 * dense_units_multiplier)
    dense_units3 = int(128 * dense_units_multiplier)
    
    # Choose RNN layer type - Use LSTM by default for better performance on sequential data
    if use_gru:
        rnn_layer = layers.GRU
        print("Using GRU cells for recurrent layers")
    else:
        rnn_layer = layers.LSTM
        print("Using LSTM cells for recurrent layers")
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Apply WKS feature extraction if requested
    if use_wks:
        print(f"Using Weighted Kurtosis and Skewness (WKS) feature extraction")
        wks_features = WKSLayer(window_size=wks_window_size, step_size=wks_step_size, name='wks_layer')(inputs)
        # Reshape WKS features to match input shape for concatenation later
        wks_features = layers.Dense(input_shape[-1], kernel_regularizer=regularizers.l2(l2_reg))(wks_features)
    
    # CNN branch for spatial feature extraction
    # First convolutional block with residual connection
    x_cnn = layers.Conv1D(cnn_filters1, 3, padding='same', kernel_initializer='he_uniform',
                         kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.Activation('relu')(x_cnn)
    x_cnn = layers.SpatialDropout1D(dropout_rate * 0.5)(x_cnn)  # Use SpatialDropout1D instead of regular Dropout
    x_cnn_res = x_cnn  # Save for residual connection
    
    # Second convolutional block
    x_cnn = layers.Conv1D(cnn_filters2, 3, padding='same', kernel_initializer='he_uniform',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_cnn)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.Activation('relu')(x_cnn)
    x_cnn = layers.MaxPooling1D(2, padding='same')(x_cnn)
    
    # Third convolutional block with residual connection
    x_cnn_res = layers.Conv1D(cnn_filters2, 1, padding='same')(x_cnn_res)  # Match dimensions
    x_cnn_res = layers.MaxPooling1D(2, padding='same')(x_cnn_res)  # Match dimensions
    x_cnn = layers.add([x_cnn, x_cnn_res])  # Residual connection
    
    x_cnn = layers.Conv1D(cnn_filters3, 3, padding='same', kernel_initializer='he_uniform',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_cnn)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.Activation('relu')(x_cnn)
    
    # Global pooling for CNN features
    x_cnn_max = layers.GlobalMaxPooling1D()(x_cnn)
    x_cnn_avg = layers.GlobalAveragePooling1D()(x_cnn)
    
    # RNN branch for temporal feature extraction
    # First RNN layer with bidirectional wrapper
    x_rnn = layers.Bidirectional(
        rnn_layer(cnn_filters2, return_sequences=True, 
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2))
    )(inputs)
    x_rnn = layers.BatchNormalization()(x_rnn)
    
    # Second RNN layer with bidirectional wrapper
    x_rnn = layers.Bidirectional(
        rnn_layer(cnn_filters2, return_sequences=True, 
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2))
    )(x_rnn)
    x_rnn = layers.BatchNormalization()(x_rnn)
    
    # Third RNN layer with bidirectional wrapper
    x_rnn = layers.Bidirectional(
        rnn_layer(cnn_filters3, return_sequences=True, 
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2))
    )(x_rnn)
    x_rnn = layers.BatchNormalization()(x_rnn)
    
    # Apply self-attention mechanism to RNN features
    x_att = SIAOAttention(units=attention_units, num_heads=num_heads, name='siao_attention_1')(x_rnn)
    
    # Global pooling for RNN features
    x_rnn_max = layers.GlobalMaxPooling1D()(x_att)
    x_rnn_avg = layers.GlobalAveragePooling1D()(x_rnn)
    
    # Concatenate CNN and RNN features (and WKS features if used)
    if use_wks:
        # Get global pooling for WKS features
        wks_max = layers.GlobalMaxPooling1D()(wks_features)
        wks_avg = layers.GlobalAveragePooling1D()(wks_features)
        x = layers.concatenate([x_cnn_max, x_cnn_avg, x_rnn_max, x_rnn_avg, wks_max, wks_avg])
    else:
        x = layers.concatenate([x_cnn_max, x_cnn_avg, x_rnn_max, x_rnn_avg])
    
    # Fully connected layers with residual connections
    x = layers.Dense(dense_units1, kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x_res = x  # Save for residual connection
    
    x = layers.Dense(dense_units2, kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Residual connection with projection
    x_res = layers.Dense(dense_units2)(x_res)
    x = layers.add([x, x_res])
    
    x = layers.Dense(dense_units3, kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
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
        from model_siao_cnn_ornn import AquilaOptimizer
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
    print("\nTraining enhanced SIAO model with the following configuration:")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"L2 regularization: {l2_reg}")
    print(f"Learning rate: {learning_rate}")
    print(f"Attention units: {attention_units}")
    print(f"Filters multiplier: {filters_multiplier}")
    print(f"Dense units multiplier: {dense_units_multiplier}")
    print(f"Number of attention heads: {num_heads}")
    
    # Set input shape if not provided
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
    
    # 3. Magnitude scaling (both up and down)
    scale_factors = [0.9, 1.1]  # Scale down by 10% and up by 10%
    for scale_factor in scale_factors:
        for i in range(len(X_train)):
            scaled_sample = X_train[i] * scale_factor
            augmented_X.append(scaled_sample)
            augmented_y.append(y_train[i])
    
    # Convert augmented data to numpy arrays
    X_train_aug = np.array(augmented_X)
    y_train_aug = np.array(augmented_y)
    
    print(f"Data augmentation: {len(X_train)} original samples -> {len(X_train_aug)} augmented samples")
    
    # Ensure validation data is properly formatted and balanced
    if X_val is not None and y_val is not None:
        print(f"Validation data shapes before processing: X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        # Check if validation labels need to be converted to one-hot encoding
        if len(y_val.shape) == 1 or y_val.shape[1] == 1:
            print("Converting validation labels to one-hot encoding")
            from tensorflow.keras.utils import to_categorical
            y_val = to_categorical(y_val, num_classes=num_classes)
        
        # Analyze class distribution in validation set
        if len(y_val.shape) > 1:
            val_classes = np.argmax(y_val, axis=1)
            val_class_counts = np.bincount(val_classes, minlength=num_classes)
            print("Validation class distribution:")
            for i, count in enumerate(val_class_counts):
                print(f"  Class {i}: {count} samples ({count/len(val_classes)*100:.2f}%)")
            
            # Check for severe class imbalance in validation set
            min_class_count = np.min(val_class_counts)
            if min_class_count < 5:
                print("WARNING: Some classes have very few samples in validation set!")
                print("This can cause validation accuracy to be unstable or zero.")
                
                # Option 1: Duplicate samples from underrepresented classes
                # This helps ensure the model sees examples from all classes during validation
                print("Balancing validation set by duplicating samples from underrepresented classes...")
                target_count = max(20, np.max(val_class_counts) // 5)  # Aim for at least 20 samples per class
                
                balanced_X_val = []
                balanced_y_val = []
                
                for class_idx in range(num_classes):
                    class_mask = val_classes == class_idx
                    class_X = X_val[class_mask]
                    class_y = y_val[class_mask]
                    
                    if len(class_X) == 0:
                        continue  # Skip if no samples for this class
                    
                    # Calculate how many times to duplicate
                    repeat_factor = max(1, target_count // len(class_X))
                    remainder = target_count % len(class_X)
                    
                    # Add repeated samples
                    for _ in range(repeat_factor):
                        balanced_X_val.append(class_X)
                        balanced_y_val.append(class_y)
                    
                    # Add remainder samples
                    if remainder > 0:
                        balanced_X_val.append(class_X[:remainder])
                        balanced_y_val.append(class_y[:remainder])
                
                # Concatenate and shuffle
                X_val = np.vstack(balanced_X_val)
                y_val = np.vstack(balanced_y_val)
                
                # Shuffle the balanced validation set
                indices = np.arange(len(X_val))
                np.random.shuffle(indices)
                X_val = X_val[indices]
                y_val = y_val[indices]
                
                print(f"Balanced validation data shape: X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        print(f"Validation data shapes after processing: X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Build the model
    model = build_siao_enhanced(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        recurrent_dropout=0.0 if IS_APPLE_SILICON else 0.2,  # Use 0.0 on Apple Silicon for cuDNN acceleration
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
    
    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    
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
    
    start_time = time.time()
    
    # Print model summary and parameter count before training
    model.summary()
    trainable_count = sum(K.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(K.count_params(w) for w in model.non_trainable_weights)
    print(f"Total parameters: {trainable_count + non_trainable_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    
    # Print class weights if using them
    if class_weights:
        print(f"Using class weights: {class_weights}")
    
    # Add a custom callback to monitor metrics during training
    class MetricsMonitor(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            metrics = ["loss", "accuracy", "auc", "precision", "recall", 
                      "val_loss", "val_accuracy", "val_auc", "val_precision", "val_recall"]
            metrics_str = ", ".join([f"{m}: {logs.get(m, 0):.4f}" for m in metrics if m in logs])
            print(f"Epoch {epoch+1} metrics: {metrics_str}")
    
    callbacks.append(MetricsMonitor())
    
    # Train the model
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history
