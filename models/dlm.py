"""
Deep Learning Module for NPP Fault Monitoring.
This module provides functions for creating and training deep learning models.
"""

import numpy as np
import warnings

# Check NumPy version for compatibility with TensorFlow
TENSORFLOW_AVAILABLE = False
try:
    np_version = np.__version__
    if np_version.startswith('2'):  # NumPy 2.x is not compatible with older TensorFlow
        warnings.warn(f"NumPy version {np_version} detected. This may not be compatible with TensorFlow. Consider downgrading to NumPy 1.x for TensorFlow compatibility.")
        # Don't even try to import TensorFlow with NumPy 2.x
    else:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
            TENSORFLOW_AVAILABLE = True
        except (ImportError, TypeError, AttributeError) as e:
            warnings.warn(f"TensorFlow import failed: {e}. Deep learning models will not be available.")
            TENSORFLOW_AVAILABLE = False
except Exception as e:
    warnings.warn(f"Error checking NumPy version: {e}. Deep learning models will not be available.")
    TENSORFLOW_AVAILABLE = False

def check_tensorflow():
    """Check if TensorFlow is available"""
    return TENSORFLOW_AVAILABLE


def prepare_sliding_window_data(data, window_size=100, step_size=1, time_col=None, target_col=None):
    """
    Prepare sliding window data for deep learning models.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Size of the sliding window
        step_size (int): Step size for the sliding window
        time_col (str): Name of the time column to exclude
        target_col (str): Name of the target column
        
    Returns:
        tuple: (X, y) where X is the windowed data and y is the target
    """
    if not isinstance(data, np.ndarray):
        # Convert DataFrame to numpy array, excluding time and target columns
        cols_to_exclude = []
        if time_col is not None:
            cols_to_exclude.append(time_col)
        if target_col is not None:
            cols_to_exclude.append(target_col)
            
        feature_cols = [col for col in data.columns if col not in cols_to_exclude]
        X_data = data[feature_cols].values
        
        if target_col is not None:
            y_data = data[target_col].values
        else:
            y_data = None
    else:
        X_data = data
        y_data = None
    
    # Create sliding windows
    n_samples = len(X_data)
    n_windows = (n_samples - window_size) // step_size + 1
    
    if n_windows <= 0:
        warnings.warn(f"Window size {window_size} is too large for data with {n_samples} samples.")
        return None, None
    
    n_features = X_data.shape[1]
    X_windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        X_windows[i] = X_data[start_idx:end_idx]
    
    # Prepare target data if available
    if y_data is not None:
        y_windows = np.zeros(n_windows)
        for i in range(n_windows):
            # Use the label at the end of each window
            end_idx = i * step_size + window_size - 1
            y_windows[i] = y_data[end_idx]
    else:
        y_windows = None
    
    return X_windows, y_windows

def create_cnn_lstm_model(input_shape, num_classes):
    """
    Create a CNN-LSTM model for time series classification.
    
    Args:
        input_shape (tuple): Shape of input data (window_size, num_features)
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: CNN-LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Cannot create CNN-LSTM model.")
    
    model = Sequential([
        # CNN layers
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM layer
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_cnn_lstm_model(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs=50, batch_size=32, patience=10):
    """
    Train a CNN-LSTM model for time series classification.
    
    Args:
        X_train (np.array): Training data with shape (samples, time_steps, features)
        y_train (np.array): Training labels
        X_val (np.array): Validation data
        y_val (np.array): Validation labels
        input_shape (tuple): Shape of input data (time_steps, features)
        num_classes (int): Number of classes
        epochs (int): Number of epochs
        batch_size (int): Batch size
        patience (int): Patience for early stopping
        
    Returns:
        model: Trained model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Cannot train deep learning model.")
    
    # Create model
    model = create_cnn_lstm_model(input_shape, num_classes)
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def tune_hyperparameters(X_train, y_train, X_val, y_val, model_type='cnn_lstm', n_trials=10, verbose=1):
    """
    Tune hyperparameters for deep learning models using random search.
    
    Args:
        X_train (np.array): Training data
        y_train (np.array): Training labels
        X_val (np.array): Validation data
        y_val (np.array): Validation labels
        model_type (str): Type of model to tune ('cnn_lstm', 'lstm', 'cnn')
        n_trials (int): Number of hyperparameter combinations to try
        verbose (int): Verbosity level
        
    Returns:
        dict: Best hyperparameters and model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Cannot tune hyperparameters.")
    
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    # Get input shape and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    
    # Define hyperparameter search space
    if model_type == 'cnn_lstm':
        param_space = {
            'conv_filters': [16, 32, 64, 128],
            'conv_kernel_size': [3, 5, 7],
            'lstm_units': [32, 64, 128, 256],
            'dense_units': [32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'activation': ['relu', 'elu']
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are: 'cnn_lstm'")
    
    # Initialize best parameters and score
    best_params = None
    best_score = 0
    best_model = None
    best_history = None
    
    print(f"Starting hyperparameter tuning for {model_type} model with {n_trials} trials...")
    
    # Run random search
    for trial in range(n_trials):
        # Sample random hyperparameters
        params = {}
        for param_name, param_values in param_space.items():
            params[param_name] = np.random.choice(param_values)
        
        if verbose > 0:
            print(f"\nTrial {trial+1}/{n_trials}:")
            print(f"Parameters: {params}")
        
        try:
            # Create and compile model with sampled hyperparameters
            if model_type == 'cnn_lstm':
                model = Sequential()
                
                # CNN layers
                model.add(Conv1D(filters=params['conv_filters'],
                                kernel_size=params['conv_kernel_size'],
                                activation=params['activation'],
                                input_shape=input_shape))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Dropout(params['dropout_rate']))
                
                # LSTM layers
                model.add(LSTM(units=params['lstm_units'], return_sequences=False))
                model.add(Dropout(params['dropout_rate']))
                
                # Dense layers
                model.add(Dense(params['dense_units'], activation=params['activation']))
                model.add(Dropout(params['dropout_rate']))
                model.add(Dense(num_classes, activation='softmax'))
                
                # Compile model
                from tensorflow.keras.optimizers import Adam
                optimizer = Adam(learning_rate=params['learning_rate'])
                model.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])
            
            # Train model
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(X_train, y_train,
                              validation_data=(X_val, y_val),
                              epochs=30,
                              batch_size=params['batch_size'],
                              callbacks=[early_stopping],
                              verbose=0 if verbose <= 1 else 1)
            
            # Evaluate model
            y_pred = model.predict(X_val)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_val, axis=1)
            accuracy = accuracy_score(y_true_classes, y_pred_classes)
            
            if verbose > 0:
                print(f"Validation accuracy: {accuracy:.4f}")
            
            # Update best parameters if current model is better
            if accuracy > best_score:
                best_score = accuracy
                best_params = params.copy()
                best_model = model
                best_history = history
                
                if verbose > 0:
                    print(f"New best model found! Accuracy: {best_score:.4f}")
        
        except Exception as e:
            if verbose > 0:
                print(f"Error during trial {trial+1}: {e}")
    
    print(f"\nHyperparameter tuning completed.")
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_model': best_model,
        'best_history': best_history,
        'best_score': best_score
    }


def train_deep_learning_model(X_train, y_train, X_test=None, y_test=None, model_type='cnn_lstm', **kwargs):
    """
    Train a deep learning model for time series classification.
    
    Args:
        X_train (np.array): Training data with shape (samples, time_steps, features)
        y_train (np.array): Training labels with shape (samples, num_classes) - one-hot encoded
        X_test (np.array): Test data
        y_test (np.array): Test labels
        model_type (str): Type of model to train ('cnn_lstm', 'lstm', 'cnn')
        **kwargs: Additional arguments for model training
        
    Returns:
        dict: Dictionary containing the trained model and evaluation metrics
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Cannot train deep learning model.")
    
    # Check input shapes
    if len(X_train.shape) != 3:
        raise ValueError(f"Expected X_train to have shape (samples, time_steps, features), got {X_train.shape}")
    
    if len(y_train.shape) != 2:
        raise ValueError(f"Expected y_train to be one-hot encoded with shape (samples, num_classes), got {y_train.shape}")
    
    # Get input shape and number of classes
    samples, time_steps, features = X_train.shape
    input_shape = (time_steps, features)
    num_classes = y_train.shape[1]
    
    # Split training data into training and validation sets if test data is not provided
    if X_test is None or y_test is None:
        # Use 20% of training data for validation
        val_split = int(samples * 0.2)
        X_val = X_train[-val_split:]
        y_val = y_train[-val_split:]
        X_train = X_train[:-val_split]
        y_train = y_train[:-val_split]
    else:
        X_val = X_test
        y_val = y_test
    
    # Set default hyperparameters
    epochs = kwargs.get('epochs', 50)
    batch_size = kwargs.get('batch_size', 32)
    patience = kwargs.get('patience', 10)
    
    print(f"Training {model_type} model with {samples} samples, {time_steps} time steps, {features} features, {num_classes} classes")
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    
    # Train model based on type
    if model_type == 'cnn_lstm':
        model, history = train_cnn_lstm_model(
            X_train, y_train, X_val, y_val, input_shape, num_classes,
            epochs=epochs, batch_size=batch_size, patience=patience
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are: 'cnn_lstm'")
    
    # Evaluate model on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    # Calculate metrics
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    except ImportError:
        # If sklearn is not available, use TensorFlow metrics
        accuracy = val_accuracy
        precision = recall = f1 = None
    
    # Return results
    results = {
        'model': model,
        'history': history,
        'metrics': {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    return results

def prepare_data_for_deep_learning(X, y, window_size=100, step_size=None):
    """
    Prepare data for deep learning models by reshaping into 3D format.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector (one-hot encoded)
        window_size (int): Size of the sliding window
        step_size (int): Step size for the sliding window
        
    Returns:
        tuple: (X_reshaped, y_reshaped)
    """
    if step_size is None:
        step_size = window_size // 2
    
    # Reshape X to (samples, time_steps, features)
    samples = (X.shape[0] - window_size) // step_size + 1
    X_reshaped = np.zeros((samples, window_size, X.shape[1]))
    y_reshaped = np.zeros((samples, y.shape[1]))
    
    for i in range(samples):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        X_reshaped[i] = X[start_idx:end_idx]
        y_reshaped[i] = y[start_idx]  # Use the label of the first sample in the window
    
    return X_reshaped, y_reshaped

def evaluate_deep_learning_model(model, X_test, y_test):
    """
    Evaluate a deep learning model.
    
    Args:
        model (tf.keras.Model): Trained model
        X_test (np.array): Test data
        y_test (np.array): Test labels (one-hot encoded)
        
    Returns:
        dict: Evaluation metrics
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Cannot evaluate deep learning model.")
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return metrics
