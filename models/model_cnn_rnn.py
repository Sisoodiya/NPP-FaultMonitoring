from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.optimizers import Adam
# Import legacy optimizers for better M1/M2 Mac performance
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
import tensorflow as tf
import numpy as np
import os
import time
import datetime

# Import focal loss and WKS features from the SIAO model
from model_siao_cnn_ornn import focal_loss, extract_wks_features, WKSLayer

# Enable Metal plugin for better performance on M1/M2 Macs
os.environ['TF_METAL_ENABLED'] = '1'

# Check if running on M1/M2 Mac
import platform

# Most reliable way to detect Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'

print(f"Running on {'Apple Silicon (M1/M2)' if IS_APPLE_SILICON else 'Standard hardware'}")

def build_cnn_rnn(input_shape, num_classes, dropout_rate=0.4, recurrent_dropout=0.2, l2_reg=0.001, 
            learning_rate=0.001, use_gru=None, use_bidirectional=True, units_multiplier=1.0,
            filters_multiplier=1.0, kernel_size=3, pool_size=2,
            use_focal_loss=False, focal_gamma=2.0, focal_alpha=0.25, use_wks=False,
            wks_window_size=20, wks_step_size=10):
    """
    Build an optimized CNN-RNN hybrid model with improved architecture and regularization.
    Enhanced with focal loss and WKS features from the SIAO model.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization (default: 0.4)
        recurrent_dropout: Dropout rate for recurrent connections
        l2_reg: L2 regularization factor (default: 0.001)
        learning_rate: Learning rate for optimizer (default: 0.001)
        use_gru: Whether to use GRU cells (True), LSTM cells (False), or auto-detect (None)
        use_bidirectional: Whether to use bidirectional recurrent layers (default: True)
        units_multiplier: Multiplier for number of units in recurrent layers (default: 1.0)
        filters_multiplier: Multiplier for number of filters in convolutional layers (default: 1.0)
        kernel_size: Size of convolutional kernels (default: 3)
        pool_size: Size of pooling windows (default: 2)
        use_focal_loss: Whether to use focal loss for imbalanced classes (default: False)
        focal_gamma: Focusing parameter for focal loss (default: 2.0)
        focal_alpha: Weighting factor for focal loss (default: 0.25)
        use_wks: Whether to use Weighted Kurtosis and Skewness features (default: False)
        wks_window_size: Window size for WKS feature extraction (default: 20)
        wks_step_size: Step size for WKS feature extraction (default: 10)
        
    Returns:
        Compiled Keras CNN-RNN hybrid model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Add WKS feature extraction if enabled
    if use_wks:
        print(f"Using WKS feature extraction with window_size={wks_window_size}, step_size={wks_step_size}")
        wks_features = WKSLayer(window_size=wks_window_size, step_size=wks_step_size)(inputs)
        wks_features = layers.Reshape((1, -1))(wks_features)  # Reshape for concatenation later
    
    # On M1/M2 Macs, use LSTM instead of GRU for better performance with cuDNN
    if use_gru is None:
        use_gru = not IS_APPLE_SILICON  # Use LSTM on Apple Silicon, GRU otherwise
        
    # Set recurrent_dropout to 0 for Apple Silicon to enable cuDNN acceleration
    if IS_APPLE_SILICON:
        recurrent_dropout = 0.0
        print("Using LSTM with recurrent_dropout=0.0 for better M1/M2 Mac performance")
        rnn_layer = layers.LSTM
    else:
        # Choose between GRU (better performance) or SimpleRNN
        rnn_layer = layers.GRU if use_gru else layers.LSTM
    
    # Calculate filter and unit sizes based on multipliers
    filters1 = int(64 * filters_multiplier)
    filters2 = int(128 * filters_multiplier)
    units_rnn = int(64 * units_multiplier)
    units_dense = int(128 * units_multiplier)
    
    # CNN part: First convolutional block
    x = layers.Conv1D(
        filters=filters1, 
        kernel_size=kernel_size, 
        padding='same', 
        kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size)(x)
    x = layers.Dropout(dropout_rate/2)(x)  # Lower dropout after pooling
    
    # CNN part: Second convolutional block
    x = layers.Conv1D(
        filters=filters2, 
        kernel_size=kernel_size, 
        padding='same', 
        kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size)(x)
    x = layers.Dropout(dropout_rate/2)(x)
    
    # RNN part: Recurrent layer (bidirectional if enabled)
    if use_bidirectional:
        rnn_output = layers.Bidirectional(
            rnn_layer(
                units=units_rnn,
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg)
            )
        )(x)
    else:
        rnn_output = rnn_layer(
            units=units_rnn,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
    
    # Add batch normalization for better training stability
    rnn_output = layers.BatchNormalization()(rnn_output)
    
    # Combine with WKS features if enabled
    if use_wks:
        # Flatten WKS features
        wks_flat = layers.Flatten()(wks_features)
        
        # Concatenate RNN output with WKS features
        x = layers.Concatenate()([rnn_output, wks_flat])
    else:
        x = rnn_output
    
    # Fully connected layers
    x = layers.Dense(
        units=units_dense,
        kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
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
    
    # Choose optimizer based on hardware
    if IS_APPLE_SILICON:
        # Use legacy Adam for better M1/M2 Mac performance
        print("Using Legacy Adam optimizer for better M1/M2 Mac performance")
        optimizer = LegacyAdam(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    
    # Add more comprehensive metrics for better evaluation
    metrics = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )
    
    return model


def train_cnn_rnn(X_train, y_train, X_val=None, y_val=None, input_shape=None, num_classes=None, 
                epochs=50, batch_size=32, early_stopping=True, class_weights=None,
                dropout_rate=0.4, recurrent_dropout=0.2, l2_reg=0.001, learning_rate=0.001, 
                use_gru=None, use_bidirectional=True, units_multiplier=1.0, filters_multiplier=1.0,
                kernel_size=3, pool_size=2, use_focal_loss=False, focal_gamma=2.0, focal_alpha=0.25,
                use_wks=False, wks_window_size=20, wks_step_size=10, custom_class_weights=False,
                checkpoint_path=None, use_adaptive_learning=True, patience=10,
                min_delta=0.001, monitor='val_loss'):
    """
    Train the CNN-RNN hybrid model with advanced training techniques from the SIAO model.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        input_shape: Shape of input data
        num_classes: Number of output classes
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        early_stopping: Whether to use early stopping
        class_weights: Class weights for imbalanced data
        dropout_rate: Dropout rate for regularization (default: 0.4)
        recurrent_dropout: Dropout rate for recurrent connections (default: 0.2)
        l2_reg: L2 regularization factor (default: 0.001)
        learning_rate: Learning rate for optimizer (default: 0.001)
        use_gru: Whether to use GRU cells (True), LSTM cells (False), or auto-detect (None)
        use_bidirectional: Whether to use bidirectional recurrent layers (default: True)
        units_multiplier: Multiplier for number of units in recurrent layers (default: 1.0)
        filters_multiplier: Multiplier for number of filters in convolutional layers (default: 1.0)
        kernel_size: Size of convolutional kernels (default: 3)
        pool_size: Size of pooling windows (default: 2)
        use_focal_loss: Whether to use focal loss for imbalanced classes (default: False)
        focal_gamma: Focusing parameter for focal loss (default: 2.0)
        focal_alpha: Weighting factor for focal loss (default: 0.25)
        use_wks: Whether to use Weighted Kurtosis and Skewness features (default: False)
        wks_window_size: Window size for WKS feature extraction (default: 20)
        wks_step_size: Step size for WKS feature extraction (default: 10)
        custom_class_weights: Whether to use custom class weights based on class distribution (default: False)
        checkpoint_path: Path to save model checkpoints (default: None)
        use_adaptive_learning: Whether to use adaptive learning rate (default: True)
        patience: Patience for early stopping and learning rate reduction (default: 10)
        min_delta: Minimum change to qualify as improvement (default: 0.001)
        monitor: Metric to monitor for early stopping and checkpoints (default: 'val_loss')
        
    Returns:
        Trained model and training history
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    # Infer input shape and number of classes if not provided
    if input_shape is None and X_train is not None:
        input_shape = X_train.shape[1:]
    
    if num_classes is None and y_train is not None:
        if len(y_train.shape) > 1:
            num_classes = y_train.shape[1]  # One-hot encoded
        else:
            num_classes = len(np.unique(y_train))  # Class indices
    
    # Calculate custom class weights if enabled
    if custom_class_weights and class_weights is None:
        print("Calculating custom class weights based on class distribution...")
        if len(y_train.shape) > 1:  # One-hot encoded
            y_integers = np.argmax(y_train, axis=1)
        else:
            y_integers = y_train
            
        # Count class frequencies
        class_counts = np.bincount(y_integers.astype(int))
        total_samples = np.sum(class_counts)
        n_classes = len(class_counts)
        
        # Calculate class weights with smoothing factor
        weights = {}
        max_weight = 10.0  # Cap maximum weight to prevent extreme values
        
        for i in range(n_classes):
            # Add smoothing factor to prevent extreme weights for very rare classes
            # More aggressive weighting for underrepresented classes
            weight = (total_samples / (n_classes * class_counts[i])) ** 0.75
            weights[i] = min(weight, max_weight)  # Cap the weight
        
        print(f"Custom class weights: {weights}")
        class_weights = weights
    
    # Create checkpoint path if not provided
    if checkpoint_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = f"models/checkpoints/cnn_rnn_{timestamp}/best_model.h5"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Build model with specified hyperparameters
    model = build_cnn_rnn(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        l2_reg=l2_reg,
        learning_rate=learning_rate,
        use_gru=use_gru,
        use_bidirectional=use_bidirectional,
        units_multiplier=units_multiplier,
        filters_multiplier=filters_multiplier,
        kernel_size=kernel_size,
        pool_size=pool_size,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        use_wks=use_wks,
        wks_window_size=wks_window_size,
        wks_step_size=wks_step_size
    )
    
    # Define callbacks
    callbacks = []
    
    # Add early stopping if enabled
    if early_stopping and X_val is not None:
        callbacks.append(EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            min_delta=min_delta
        ))
    
    # Add learning rate scheduler if adaptive learning is enabled
    if use_adaptive_learning:
        callbacks.append(ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,  # Reduce patience for LR scheduler
            min_lr=1e-6,
            verbose=1,
            min_delta=min_delta
        ))
    
    # Add model checkpoint to save best model
    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    ))
    
    # Add composite metric callback to monitor multiple metrics
    class CompositeMetricCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Calculate composite score from multiple metrics
            accuracy = logs.get('val_accuracy', 0)
            auc = logs.get('val_auc', 0)
            precision = logs.get('val_precision', 0)
            recall = logs.get('val_recall', 0)
            
            # Weighted combination of metrics
            composite_score = (0.4 * accuracy) + (0.3 * auc) + (0.15 * precision) + (0.15 * recall)
            logs['val_composite_score'] = composite_score
            
            print(f"\nEpoch {epoch+1}: val_composite_score: {composite_score:.4f}")
    
    # Add composite metric callback if validation data is available
    if X_val is not None:
        callbacks.append(CompositeMetricCallback())
    
    # Add timing callback to measure training time
    class TimingCallback(Callback):
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            elapsed_time = time.time() - self.start_time
            logs['time_elapsed'] = elapsed_time
            print(f"Time elapsed: {elapsed_time:.2f}s")
    
    callbacks.append(TimingCallback())
    
    # Train model
    if X_val is not None:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    else:
        # Use validation split if validation data not provided
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    
    # Print final evaluation
    if X_val is not None:
        print("\nFinal evaluation on validation data:")
        evaluation = model.evaluate(X_val, y_val, verbose=0)
        metrics = model.metrics_names
        
        for metric, value in zip(metrics, evaluation):
            print(f"{metric}: {value:.4f}")
    
    return model, history