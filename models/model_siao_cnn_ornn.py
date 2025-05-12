from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.optimizers import Adam, Optimizer
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
import math
import datetime

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


class SIAOAttention(layers.Layer):
    """Self-Improved Attention mechanism for the SIAO-CNN-ORNN model."""
    
    def __init__(self, units, **kwargs):
        super(SIAOAttention, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.b = None
        self.V = None
    
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='attention_W',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='attention_b',
            trainable=True
        )
        self.V = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            name='attention_V',
            trainable=True
        )
        super(SIAOAttention, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        
        # Score function: tanh(W·x + b)
        score = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        
        # Attention weights: softmax(V·score)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)
        
        # Context vector: sum(attention_weights * inputs)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
    
    def get_config(self):
        config = super(SIAOAttention, self).get_config()
        config.update({
            'units': self.units
        })
        return config


def build_siao_cnn_ornn(input_shape, num_classes, dropout_rate=0.3, recurrent_dropout=0.2, l2_reg=0.001, learning_rate=0.001, use_gru=False, use_aquila_optimizer=True):
    """
    Build an optimized SIAO (Self-Improving Architecture Optimization) CNN-RNN hybrid model.
    
    This model combines CNN for spatial feature extraction with RNN for temporal dependencies,
    and adds a self-attention mechanism to focus on the most important features.
    
    Args:
        input_shape: Shape of input data (time_steps, features)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        recurrent_dropout: Dropout rate for recurrent layers
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for optimizer
        use_gru: Whether to use GRU cells instead of SimpleRNN
        use_aquila_optimizer: Whether to use the Aquila Optimizer
        
    Returns:
        Compiled Keras SIAO CNN-RNN hybrid model
    """
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
    rnn2 = layers.Bidirectional(
        rnn_layer(256,  # Increased from 128 to 256
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2),
                kernel_initializer='glorot_uniform')
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
    )(rnn2)
    
    # Multiple pooling strategies for better feature extraction
    avg_pool = layers.GlobalAveragePooling1D()(rnn3)
    max_pool = layers.GlobalMaxPooling1D()(rnn3)
    attention_pool = SIAOAttention(256)(rnn3)  # Additional attention-based pooling
    
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
    if use_aquila_optimizer:
        try:
            optimizer = AquilaOptimizer(learning_rate=learning_rate)
            print("Using Aquila Optimizer for SIAO-CNN-ORNN model")
        except Exception as e:
            print(f"Error initializing Aquila Optimizer: {e}. Falling back to Adam.")
            optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model


def train_siao_cnn_ornn(X_train, y_train, X_val=None, y_val=None, input_shape=None, num_classes=None, 
                      epochs=100, batch_size=16, early_stopping=True, class_weights=None,
                      use_aquila_optimizer=True, use_gru=True, dropout_rate=0.4, l2_reg=0.0005):
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
    
    # Stochastic Weight Averaging for better generalization
    try:
        from tensorflow.keras.experimental import SWA
        # Apply SWA in the later part of training
        swa_start = int(epochs * 0.75)  # Start SWA at 75% of training
        swa_freq = 5  # Average weights every 5 epochs
        callbacks.append(SWA(start_epoch=swa_start, swa_freq=swa_freq, swa_lr=0.0001))
        print(f"Using Stochastic Weight Averaging starting at epoch {swa_start}")
    except ImportError:
        print("SWA not available in this TensorFlow version. Skipping.")
        
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
    print(f"\nTraining SIAO-CNN-ORNN model with {'Aquila' if use_aquila_optimizer else 'Adam'} optimizer")
    print(f"Using {'GRU' if use_gru else 'LSTM'} cells for recurrent layers")
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
    model.save('models/saved/siao_cnn_ornn_final.h5')
    print("Model saved to 'models/saved/siao_cnn_ornn_final.h5'")
    
    return model, history