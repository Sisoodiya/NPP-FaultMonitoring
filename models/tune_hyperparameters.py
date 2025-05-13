"""
Comprehensive Hyperparameter Tuning Script for NPP Fault Monitoring Deep Learning Models.

This script performs hyperparameter tuning for various deep learning models to optimize their performance
for fault detection in nuclear power plants. It combines the functionality of tune_hyperparameters.py
and hyperparameter_tuning.py into a single, comprehensive solution.

Supported models include:
- CNN
- RNN
- LSTM
- CNN-RNN
- CNN-LSTM (primary focus)
- SIAO-CNN-ORNN (primary focus)

Features:
- Multiple tuning methods: grid search, random search, Bayesian optimization
- Standardized hyperparameter search spaces for all models
- Advanced features: focal loss, WKS feature extraction, custom class weights
- Adaptive learning rate and Aquila optimizer support
- Comprehensive evaluation metrics and visualization
- Support for model comparison
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import datetime
import json
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

# Try to import Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Bayesian optimization not available. Install scikit-optimize for this feature.")

# Import custom modules
from data_preprocessing import process_pipeline

# Import feature extraction modules
from feature_extraction_new import extract_statistical_features, extract_all_features

# Import model modules
from model_cnn import build_cnn, train_cnn
from model_rnn import build_rnn, train_rnn
from model_lstm import build_lstm, train_lstm
from model_cnn_rnn import build_cnn_rnn, train_cnn_rnn
from model_cnn_lstm import build_cnn_lstm, train_cnn_lstm
# Import both standard and enhanced SIAO models from the combined file
from model_siao_cnn_ornn import build_siao_cnn_ornn, train_siao_cnn_ornn, build_siao_enhanced, train_siao_enhanced


def apply_sliding_window(X, y, window_size, step_size):
    """
    Apply sliding window technique to time series data.
    
    Args:
        X (np.array): Input features with shape (samples, features)
        y (np.array): Labels with shape (samples, classes) for one-hot encoded labels
                      or (samples,) for integer labels
        window_size (int): Size of the sliding window
        step_size (int): Step size for the sliding window
        
    Returns:
        tuple: (X_windows, y_windows) where X_windows has shape (n_windows, window_size, features)
               and y_windows has the same shape as y but with n_windows samples
    """
    n_samples, n_features = X.shape
    
    # Calculate number of windows
    n_windows = (n_samples - window_size) // step_size + 1
    
    if n_windows <= 0:
        print(f"Warning: window_size {window_size} is too large for data with {n_samples} samples.")
        # Adjust window size to ensure at least one window
        window_size = n_samples // 2
        step_size = window_size // 2
        n_windows = (n_samples - window_size) // step_size + 1
        print(f"Adjusted to window_size={window_size}, step_size={step_size}, n_windows={n_windows}")
    
    # Create windows
    X_windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        X_windows[i] = X[start_idx:end_idx]
    
    # Handle labels - repeat each label for each window
    if len(y.shape) == 1:  # Integer labels
        y_windows = np.repeat(y, n_windows // len(y) + 1)[:n_windows]
    else:  # One-hot encoded labels
        n_classes = y.shape[1]
        y_windows = np.zeros((n_windows, n_classes))
        for i in range(n_windows):
            start_idx = i * step_size
            # Use the label of the last timestep in the window
            idx = min(start_idx + window_size - 1, len(y) - 1)
            y_windows[i] = y[idx // len(y)]
    
    print(f"Applied sliding window: {X.shape} -> {X_windows.shape}")
    return X_windows, y_windows

# Import reliability and adaptive learning modules
from dynamic_reliability import analyze_reliability, plot_reliability_curve, generate_reliability_report
from adaptive_learning import AdaptiveModelUpdater

# Try to import SIAO optimizer
try:
    from siao_optimizer import aquila_optimizer, optimize_wks_weights
    SIAO_AVAILABLE = True
except ImportError:
    SIAO_AVAILABLE = False
    print("SIAO optimizer not available. Using default optimization.")

# Import TensorFlow and Keras
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Please install TensorFlow to use this script.")

def create_windows(data, window_size, step_size=1):
    """Create sliding windows from data."""
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def train_model(X_train, y_train, X_val, y_val, model_type='cnn', model_name=None, 
                epochs=50, batch_size=32, class_weights=None, use_adaptive_learning=False,
                use_aquila_optimizer=False, use_focal_loss=False, use_wks=False,
                custom_class_weights=False, checkpoint_path=None, **model_params):
    """
    Train a model with the given parameters.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_type: Type of model ('cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced')
        model_name: Name for the model (used for saving)
        epochs: Number of training epochs
        batch_size: Batch size for training
        class_weights: Class weights for imbalanced datasets
        use_adaptive_learning: Whether to use adaptive learning rate
        use_aquila_optimizer: Whether to use Aquila optimizer
        use_focal_loss: Whether to use focal loss
        use_wks: Whether to use WKS feature extraction
        custom_class_weights: Whether to use custom class weights
        checkpoint_path: Path to save model checkpoints (if None, will be created automatically)
        **model_params: Additional model parameters
        
    Returns:
        Trained model and training history
    """
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    # Generate a timestamp-based model name if none provided
    if model_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{timestamp}"
        
    # Create checkpoint path if not provided
    if checkpoint_path is None:
        # Ensure the checkpoint directory exists
        os.makedirs(f"models/checkpoints/{model_type}", exist_ok=True)
        checkpoint_path = f"models/checkpoints/{model_type}/{model_name}_best_model.h5"
    
    # Build model based on type
    if model_type == 'cnn':
        model, history = train_cnn(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            use_adaptive_learning=use_adaptive_learning,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=checkpoint_path,
            filters_multiplier=model_params.get('filters_multiplier', 1.0),
            kernel_size=model_params.get('kernel_size', 3),
            pool_size=model_params.get('pool_size', 2),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'rnn':
        model, history = train_rnn(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            use_adaptive_learning=use_adaptive_learning,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=checkpoint_path,
            units_multiplier=model_params.get('units_multiplier', 1.0),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            use_bidirectional=model_params.get('use_bidirectional', False),
            use_gru=model_params.get('use_gru', False),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'lstm':
        model, history = train_lstm(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            use_adaptive_learning=use_adaptive_learning,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=checkpoint_path,
            units_multiplier=model_params.get('units_multiplier', 1.0),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            use_bidirectional=model_params.get('use_bidirectional', False),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'cnn_rnn':
        model, history = train_cnn_rnn(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            use_adaptive_learning=use_adaptive_learning,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=checkpoint_path,
            filters_multiplier=model_params.get('filters_multiplier', 1.0),
            kernel_size=model_params.get('kernel_size', 3),
            pool_size=model_params.get('pool_size', 2),
            units_multiplier=model_params.get('units_multiplier', 1.0),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            use_bidirectional=model_params.get('use_bidirectional', False),
            use_gru=model_params.get('use_gru', False),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'cnn_lstm':
        model, history = train_cnn_lstm(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            use_adaptive_learning=use_adaptive_learning,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=checkpoint_path,
            filters_multiplier=model_params.get('filters_multiplier', 1.0),
            kernel_size=model_params.get('kernel_size', 3),
            pool_size=model_params.get('pool_size', 2),
            units_multiplier=model_params.get('units_multiplier', 1.0),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            use_bidirectional=model_params.get('use_bidirectional', False),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'siao':
        model, history = train_siao_cnn_ornn(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=checkpoint_path,
            filters_multiplier=model_params.get('filters_multiplier', 1.0),
            attention_units=model_params.get('attention_units', 64),
            num_heads=model_params.get('num_heads', 4),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'siao_enhanced':
        model, history = train_siao_enhanced(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=checkpoint_path,
            filters_multiplier=model_params.get('filters_multiplier', 1.0),
            kernel_size=model_params.get('kernel_size', 3),
            pool_size=model_params.get('pool_size', 2),
            units_multiplier=model_params.get('units_multiplier', 1.0),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            attention_units=model_params.get('attention_units', 64),
            num_heads=model_params.get('num_heads', 4),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, history

# Define standardized hyperparameter search spaces
def filter_hyperparameters(param_grid, model_type, use_focal_loss=False, use_wks=False, 
                          custom_class_weights=False, use_aquila_optimizer=False,
                          use_adaptive_learning=True, use_bidirectional=True, use_gru=None):
    """
    Filter hyperparameters based on enabled features.
    
    Args:
        param_grid: Hyperparameter grid
        model_type: Type of model
        use_focal_loss: Whether to use focal loss
        use_wks: Whether to use WKS features
        custom_class_weights: Whether to use custom class weights
        use_aquila_optimizer: Whether to use Aquila optimizer
        use_adaptive_learning: Whether to use adaptive learning rate
        use_bidirectional: Whether to use bidirectional recurrent layers
        use_gru: Whether to use GRU cells (True), LSTM cells (False), or auto-detect (None)
        
    Returns:
        Filtered hyperparameter grid
    """
    filtered_grid = {}
    
    # Copy all parameters except those being filtered
    for param, values in param_grid.items():
        # Skip parameters based on feature flags
        if param == 'use_focal_loss' and not use_focal_loss:
            filtered_grid[param] = [False]
            continue
        elif param in ['focal_gamma', 'focal_alpha'] and not use_focal_loss:
            continue
        elif param == 'use_wks' and not use_wks:
            filtered_grid[param] = [False]
            continue
        elif param in ['wks_window_size', 'wks_step_size'] and not use_wks:
            continue
        elif param == 'custom_class_weights' and not custom_class_weights:
            filtered_grid[param] = [False]
            continue
        elif param == 'use_aquila_optimizer' and not use_aquila_optimizer:
            filtered_grid[param] = [False]
            continue
        elif param == 'use_adaptive_learning' and not use_adaptive_learning:
            filtered_grid[param] = [False]
            continue
        elif param == 'use_bidirectional' and use_bidirectional is not None:
            filtered_grid[param] = [use_bidirectional]
            continue
        elif param == 'use_gru' and use_gru is not None:
            filtered_grid[param] = [use_gru]
            continue
            
        # Include the parameter with its values
        filtered_grid[param] = values
    
    return filtered_grid


def evaluate_model(X_train, y_train, X_val, y_val, model_type, params, train_model_func,
                  use_adaptive_learning=True, use_aquila_optimizer=False, 
                  use_focal_loss=False, use_wks=False, custom_class_weights=False,
                  use_bidirectional=True, use_gru=None, results_dir=None, epochs=50):
    """
    Evaluate a model with the given parameters.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_type: Type of model
        params: Model parameters
        train_model_func: Function to train the model
        use_adaptive_learning: Whether to use adaptive learning rate (default: True)
        use_aquila_optimizer: Whether to use Aquila optimizer (default: False)
        use_focal_loss: Whether to use focal loss (default: False)
        use_wks: Whether to use WKS features (default: False)
        custom_class_weights: Whether to use custom class weights (default: False)
        use_bidirectional: Whether to use bidirectional recurrent layers (default: True)
        use_gru: Whether to use GRU cells (True), LSTM cells (False), or auto-detect (None)
        results_dir: Directory to save results (models will be saved to trained_models, 
                    checkpoints to models/checkpoints, and analysis to analysis directory)
        epochs: Number of training epochs (default: 50)
        
    Returns:
        Composite score (higher is better)
    """
    try:
        # Create a unique model name for this evaluation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_eval_{timestamp}"
        
        # Create checkpoint path for this evaluation
        eval_checkpoint_path = f"models/checkpoints/{model_type}/{model_name}_eval_best.h5"
        os.makedirs(os.path.dirname(eval_checkpoint_path), exist_ok=True)
        
        # Train the model
        model, history = train_model(
            X_train, y_train,
            X_val, y_val,
            model_type=model_type,
            epochs=epochs,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            checkpoint_path=eval_checkpoint_path,
            **params
        )
        
        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Get predictions
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)
        
        # Calculate metrics
        precision = precision_score(y_val_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_val_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_val_classes, y_pred_classes, average='weighted')
        
        # Calculate a composite score (higher is better)
        composite_score = (accuracy_score(y_val_classes, y_pred_classes) * 0.4 + 
                          precision * 0.2 + 
                          recall * 0.2 + 
                          f1 * 0.2) * (1.0 / (val_loss + 1e-10))
        
        # Save results if directory is provided
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            
            # Save model summary
            with open(f"{results_dir}/{model_name}_summary.txt", 'w') as f:
                f.write(f"Model: {model_type}\n")
                f.write(f"Parameters: {params}\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Composite Score: {composite_score:.4f}\n")
            
            # Save training history plot
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{model_name}_history.png")
            plt.close()
        
        return composite_score
    
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return -1.0  # Return a negative score for failed evaluations


def grid_search(X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
               use_adaptive_learning=False, use_aquila_optimizer=False, 
               use_focal_loss=False, use_wks=False, custom_class_weights=False,
               results_dir=None, n_jobs=1, epochs=50):
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_type: Type of model
        param_grid: Hyperparameter grid
        train_model_func: Function to train the model
        use_adaptive_learning: Whether to use adaptive learning
        use_aquila_optimizer: Whether to use Aquila optimizer
        use_focal_loss: Whether to use focal loss
        use_wks: Whether to use WKS features
        custom_class_weights: Whether to use custom class weights
        results_dir: Directory to save results
        n_jobs: Number of parallel jobs
        
    Returns:
        Best parameters and best score
    """
    # Create results directory if it doesn't exist
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total hyperparameter combinations to try: {len(param_combinations)}")
    
    # Initialize best parameters and score
    best_params = None
    best_score = -float('inf')
    
    # Evaluate each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"\nTrying combination {i+1}/{len(param_combinations)}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Evaluate the model with the current parameters
        score = evaluate_model(
            X_train, y_train, X_val, y_val,
            model_type, params, train_model_func,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            results_dir=results_dir
        )
        
        print(f"  Score: {score:.4f}")
        
        # Update best parameters if the score is better
        if score > best_score:
            best_score = score
            best_params = params
            print(f"  New best score: {best_score:.4f}")
    
    return best_params, best_score


def random_search(X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
                 use_adaptive_learning=False, use_aquila_optimizer=False, 
                 use_focal_loss=False, use_wks=False, custom_class_weights=False,
                 results_dir=None, n_trials=20, n_jobs=1, epochs=50):
    """
    Perform random search for hyperparameter tuning.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_type: Type of model
        param_grid: Hyperparameter grid
        train_model_func: Function to train the model
        use_adaptive_learning: Whether to use adaptive learning
        use_aquila_optimizer: Whether to use Aquila optimizer
        use_focal_loss: Whether to use focal loss
        use_wks: Whether to use WKS features
        custom_class_weights: Whether to use custom class weights
        results_dir: Directory to save results
        n_trials: Number of random trials
        n_jobs: Number of parallel jobs
        
    Returns:
        Best parameters and best score
    """
    # Create results directory if it doesn't exist
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    # Generate random parameter combinations
    param_combinations = list(ParameterSampler(param_grid, n_iter=n_trials, random_state=42))
    print(f"Total hyperparameter combinations to try: {len(param_combinations)}")
    
    # Initialize best parameters and score
    best_params = None
    best_score = -float('inf')
    
    # Evaluate each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"\nTrying combination {i+1}/{len(param_combinations)}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Evaluate the model with the current parameters
        score = evaluate_model(
            X_train, y_train, X_val, y_val,
            model_type, params, train_model_func,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            results_dir=results_dir
        )
        
        print(f"  Score: {score:.4f}")
        
        # Update best parameters if the score is better
        if score > best_score:
            best_score = score
            best_params = params
            print(f"  New best score: {best_score:.4f}")
    
    return best_params, best_score


def bayesian_search(X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
                   use_adaptive_learning=False, use_aquila_optimizer=False, 
                   use_focal_loss=False, use_wks=False, custom_class_weights=False,
                   results_dir=None, n_trials=20, epochs=50):
    """
    Perform Bayesian optimization for hyperparameter tuning.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_type: Type of model
        param_grid: Hyperparameter grid
        train_model_func: Function to train the model
        use_adaptive_learning: Whether to use adaptive learning
        use_aquila_optimizer: Whether to use Aquila optimizer
        use_focal_loss: Whether to use focal loss
        use_wks: Whether to use WKS features
        custom_class_weights: Whether to use custom class weights
        results_dir: Directory to save results
        n_trials: Number of trials
        
    Returns:
        Best parameters and best score
    """
    if not BAYESIAN_AVAILABLE:
        print("Bayesian optimization is not available. Please install scikit-optimize.")
        print("Falling back to random search.")
        return random_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            results_dir=results_dir,
            n_trials=n_trials
        )
    
    # Create results directory if it doesn't exist
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    # Convert parameter grid to skopt space
    space = []
    param_names = []
    
    for param_name, param_values in param_grid.items():
        param_names.append(param_name)
        
        if isinstance(param_values[0], bool):
            space.append(Categorical([True, False], name=param_name))
        elif isinstance(param_values[0], int):
            space.append(Integer(min(param_values), max(param_values), name=param_name))
        elif isinstance(param_values[0], float):
            space.append(Real(min(param_values), max(param_values), name=param_name))
        else:
            space.append(Categorical(param_values, name=param_name))
    
    # Define the objective function
    def objective(x):
        # Convert the parameter values to a dictionary
        params = {param_names[i]: x[i] for i in range(len(param_names))}
        
        # Evaluate the model with the current parameters
        score = evaluate_model(
            X_train, y_train, X_val, y_val,
            model_type, params, train_model_func,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            results_dir=results_dir
        )
        
        # Return negative score for minimization
        return -score
    
    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=n_trials, random_state=42, verbose=True)
    
    # Get the best parameters
    best_params = {param_names[i]: result.x[i] for i in range(len(param_names))}
    best_score = -result.fun
    
    return best_params, best_score


def get_hyperparameter_grid(model_type):
    """
    Get standardized hyperparameter search grid for the specified model type.
    
    Args:
        model_type: Type of model ('cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced')
        
    Returns:
        Dictionary of hyperparameter search spaces
    """
    # Common parameters for all models
    common_params = {
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'l2_reg': [0.0001, 0.001, 0.002, 0.005],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [16, 32, 64, 128],
        'use_focal_loss': [True, False],
        'focal_gamma': [2.0, 3.0, 4.0],
        'focal_alpha': [0.25, 0.5, 0.75],
        'use_wks': [True, False],
        'wks_window_size': [10, 20, 30],
        'wks_step_size': [5, 10, 15],
        'custom_class_weights': [True, False],
    }
    
    # Model-specific parameters
    if model_type == 'cnn':
        specific_params = {
            'filters_multiplier': [0.5, 1.0, 1.5, 2.0],
            'kernel_size': [3, 5, 7],
            'pool_size': [2, 3],
        }
    elif model_type in ['rnn', 'lstm']:
        specific_params = {
            'recurrent_dropout': [0.0, 0.1, 0.2],
            'use_bidirectional': [True, False],
            'units_multiplier': [0.5, 1.0, 1.5, 2.0],
            'use_gru': [True, False, None],
        }
    elif model_type in ['cnn_rnn', 'cnn_lstm']:
        specific_params = {
            'filters_multiplier': [0.5, 1.0, 1.5, 2.0],
            'kernel_size': [3, 5, 7],
            'pool_size': [2, 3],
            'recurrent_dropout': [0.0, 0.1, 0.2],
            'use_bidirectional': [True, False],
            'units_multiplier': [0.5, 1.0, 1.5, 2.0],
            'use_gru': [True, False, None],  # Only relevant for CNN-RNN
        }
    elif model_type in ['siao', 'siao_enhanced']:
        specific_params = {
            'filters_multiplier': [0.5, 1.0, 1.5, 2.0],
            'units_multiplier': [0.5, 1.0, 1.5, 2.0],
            'kernel_size': [3, 5, 7],
            'pool_size': [2, 3],
            'attention_units': [32, 64, 128],
            'num_heads': [1, 2, 4, 8],
            'use_aquila_optimizer': [True, False],
        }
    else:
        specific_params = {}
    
    # Combine common and specific parameters
    return {**common_params, **specific_params}


def tune_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test, model_type, train_model_func,
                        use_adaptive_learning=True, use_aquila_optimizer=False, 
                        use_focal_loss=False, use_wks=False, custom_class_weights=False,
                        use_bidirectional=True, use_gru=None, tuning_method='grid', 
                        n_trials=20, n_jobs=1, epochs=50):
    """
    Tune hyperparameters for the specified model type.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        X_test: Test data
        y_test: Test labels
        model_type: Type of model
        epochs: Number of training epochs (default: 50)
        train_model_func: Function to train the model
        use_adaptive_learning: Whether to use adaptive learning rate (default: True)
        use_aquila_optimizer: Whether to use Aquila optimizer (default: False)
        use_focal_loss: Whether to use focal loss (default: False)
        use_wks: Whether to use WKS features (default: False)
        custom_class_weights: Whether to use custom class weights (default: False)
        use_bidirectional: Whether to use bidirectional recurrent layers (default: True)
        use_gru: Whether to use GRU cells (True), LSTM cells (False), or auto-detect (None)
        tuning_method: Tuning method ('grid', 'random', or 'bayesian')
        n_trials: Number of trials for random or Bayesian search
        n_jobs: Number of parallel jobs
        
    Returns:
        Final model, best parameters, and test metrics
    """
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{model_type}_tuning_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get hyperparameter grid
    param_grid = get_hyperparameter_grid(model_type)
    
    # Filter hyperparameters based on enabled features
    param_grid = filter_hyperparameters(
        param_grid, model_type,
        use_focal_loss=use_focal_loss,
        use_wks=use_wks,
        custom_class_weights=custom_class_weights,
        use_aquila_optimizer=use_aquila_optimizer,
        use_adaptive_learning=use_adaptive_learning,
        use_bidirectional=use_bidirectional,
        use_gru=use_gru
    )
    
    # Remove use_wks, use_focal_loss, and custom_class_weights from param_grid to avoid conflicts
    if 'use_wks' in param_grid:
        del param_grid['use_wks']
    if 'use_focal_loss' in param_grid:
        del param_grid['use_focal_loss']
    if 'custom_class_weights' in param_grid:
        del param_grid['custom_class_weights']
    
    # Perform hyperparameter tuning
    print(f"\nPerforming hyperparameter tuning using {tuning_method} search...")
    if tuning_method == 'grid':
        best_params, best_score = grid_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            results_dir=results_dir,
            n_jobs=n_jobs,
            epochs=epochs
        )
    elif tuning_method == 'random':
        best_params, best_score = random_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            results_dir=results_dir,
            n_trials=n_trials,
            n_jobs=n_jobs,
            epochs=epochs
        )
    elif tuning_method == 'bayesian':
        best_params, best_score = bayesian_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            results_dir=results_dir,
            n_trials=n_trials,
            epochs=epochs
        )
    else:
        raise ValueError(f"Unknown tuning method: {tuning_method}")
    
    # Print best parameters and score
    print(f"\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best score: {best_score:.4f}")
    
    # Train final model with best parameters
    print(f"\nTraining final model with best parameters...")
    # Filter out parameters that are already passed directly
    filtered_params = {k: v for k, v in best_params.items() 
                     if k not in ['use_adaptive_learning', 'use_aquila_optimizer', 
                                  'use_focal_loss', 'use_wks', 'custom_class_weights']}
    
    final_model, history = train_model(
        X_train, y_train, X_val, y_val,
        model_type=model_type,
        epochs=epochs,
        use_adaptive_learning=use_adaptive_learning,
        use_aquila_optimizer=use_aquila_optimizer,
        use_focal_loss=use_focal_loss,
        use_wks=use_wks,
        custom_class_weights=custom_class_weights,
        **filtered_params
    )
    
    # Evaluate final model on test set
    print(f"\nEvaluating final model on test set...")
    test_metrics = final_model.evaluate(X_test, y_test, verbose=0)
    
    # Handle different return formats (single value or multiple metrics)
    if isinstance(test_metrics, list):
        test_loss = test_metrics[0]
        test_accuracy = test_metrics[1] if len(test_metrics) > 1 else 0.0
    else:
        test_loss = test_metrics
        test_accuracy = 0.0
    
    # Get predictions
    y_pred = final_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    
    # Print test metrics
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    
    # Create directories if they don't exist
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('analysis/plots', exist_ok=True)
    os.makedirs('analysis/metrics', exist_ok=True)
    
    # Save final model to trained_models directory
    model_save_path = f"trained_models/{model_type}_final_model.h5"
    final_model.save(model_save_path)
    print(f"Final model saved to {model_save_path}")
    
    # Save best parameters to analysis directory
    params_save_path = f"analysis/metrics/{model_type}_best_params.json"
    with open(params_save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to {params_save_path}")
    
    # Save test metrics to analysis directory
    test_metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    metrics_save_path = f"analysis/metrics/{model_type}_test_metrics.json"
    with open(metrics_save_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Test metrics saved to {metrics_save_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plot_save_path = f"analysis/plots/{model_type}_training_history.png"
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Training history plot saved to {plot_save_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    cm_save_path = f"analysis/plots/{model_type}_confusion_matrix.png"
    plt.savefig(cm_save_path)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_save_path}")
    
    return final_model, best_params, test_metrics


def main(model_type='cnn', use_advanced_features=False, include_wks=False, optimize_wks=False,
         feature_window=100, feature_step=50, window_size=10, step_size=5,
         use_adaptive_learning=False, use_aquila_optimizer=False, reliability_analysis=False,
         use_focal_loss=False, use_wks=False, custom_class_weights=False, 
         tuning_method='grid', n_trials=20, n_jobs=-1, epochs=50):
    """

    Main function to run hyperparameter tuning for NPP Fault Monitoring models.
    
    Args:
        model_type: Type of model to tune ('cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced')
        use_advanced_features: Whether to use advanced features
        include_wks: Whether to include WKS features in the data
        optimize_wks: Whether to optimize WKS parameters
        feature_window: Window size for feature extraction
        feature_step: Step size for feature extraction
        window_size: Size of sliding window
        step_size: Step size for sliding window
        use_adaptive_learning: Whether to use adaptive learning rate
        use_aquila_optimizer: Whether to use Aquila optimizer
        reliability_analysis: Whether to perform reliability analysis
        use_focal_loss: Whether to use focal loss
        use_wks: Whether to use WKS feature extraction
        custom_class_weights: Whether to use custom class weights
        tuning_method: Tuning method ('grid', 'random', or 'bayesian')
        n_trials: Number of trials for random or Bayesian search
        n_jobs: Number of parallel jobs
    """
    print("NPP Fault Monitoring - Hyperparameter Tuning")
    print("=" * 50)
    
    # Validate model type
    valid_models = ['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced']
    if model_type not in valid_models:
        print(f"Error: Invalid model type '{model_type}'. Valid options are: {valid_models}")
        return
        
    # Create all necessary directories for saving models, checkpoints, and analysis results
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs(f'models/checkpoints/{model_type}', exist_ok=True)
    os.makedirs('analysis/plots', exist_ok=True)
    os.makedirs('analysis/metrics', exist_ok=True)
    print(f"Created directories for saving models, checkpoints, and analysis results.")
    print(f"Selected model type: {model_type}\n")
    
    # Print configuration
    print("Configuration:")
    print(f"  Advanced features: {'Enabled' if use_advanced_features else 'Disabled'}")
    if use_advanced_features:
        print(f"  Feature window size: {feature_window}")
        print(f"  Feature step size: {feature_step}")
    else:
        print(f"  Window size: {window_size}")
        print(f"  Step size: {step_size}")
    
    print(f"  Adaptive learning: {'Enabled' if use_adaptive_learning else 'Disabled'}")
    print(f"  Aquila optimizer: {'Enabled' if use_aquila_optimizer else 'Disabled'}")
    print(f"  Reliability analysis: {'Enabled' if reliability_analysis else 'Disabled'}")
    print(f"  Focal loss: {'Enabled' if use_focal_loss else 'Disabled'}")
    print(f"  WKS features: {'Enabled' if use_wks else 'Disabled'}")
    print(f"  Custom class weights: {'Enabled' if custom_class_weights else 'Disabled'}")
    print(f"  Tuning method: {tuning_method}")
    print(f"  Number of trials: {n_trials}")
    print(f"  Number of jobs: {n_jobs}")
    print()
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not available. Cannot perform hyperparameter tuning.")
        return
    
    # 1. Load and preprocess data
    print("1. Loading and preprocessing data...")
    os.makedirs('analysis/text', exist_ok=True)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_dir = 'data'
    processed_data_dir = 'processed_data'
    os.makedirs(processed_data_dir, exist_ok=True)
    
    X_train_file = f"{processed_data_dir}/X_train.npy"
    y_train_file = f"{processed_data_dir}/y_train.npy"
    X_val_file = f"{processed_data_dir}/X_val.npy"
    y_val_file = f"{processed_data_dir}/y_val.npy"
    X_test_file = f"{processed_data_dir}/X_test.npy"
    y_test_file = f"{processed_data_dir}/y_test.npy"
    
    try:
        if (os.path.exists(X_train_file) and os.path.exists(y_train_file) and
            os.path.exists(X_val_file) and os.path.exists(y_val_file) and
            os.path.exists(X_test_file) and os.path.exists(y_test_file)):
            # Load existing processed data files
            print("Loading existing processed data files...")
            X_train = np.load(X_train_file)
            y_train = np.load(y_train_file)
            X_val = np.load(X_val_file)
            y_val = np.load(y_val_file)
            X_test = np.load(X_test_file)
            y_test = np.load(y_test_file)
        else:
            # Process all data files in the data directory
            print("Processing raw data files...")
            processed_data, file_names = process_pipeline(data_dir)
            
            if processed_data is None or processed_data.empty:
                print("Error: No data processed.")
                return
            
            print(f"Processed data shape: {processed_data.shape}")
            
            # Get the fault type column
            if 'fault_type' in processed_data.columns:
                fault_col = 'fault_type'
            elif 'condition' in processed_data.columns:
                fault_col = 'condition'
            elif 'fault' in processed_data.columns:
                fault_col = 'fault'
            else:
                print("Error: Could not find fault type column in the data.")
                return
            
            # Extract features and labels
            features = processed_data.drop([fault_col], axis=1)
            labels = processed_data[fault_col]
            
            # Convert labels to one-hot encoding
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
            num_classes = len(label_encoder.classes_)
            labels_onehot = to_categorical(labels_encoded, num_classes=num_classes)
            
            # Split data into train, validation, and test sets
            X_train, X_temp, y_train, y_temp = train_test_split(
                features, labels_onehot, test_size=0.4, random_state=42, stratify=labels_encoded
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            
            # Save processed data files
            np.save(X_train_file, X_train)
            np.save(y_train_file, y_train)
            np.save(X_val_file, X_val)
            np.save(y_val_file, y_val)
            np.save(X_test_file, X_test)
            np.save(y_test_file, y_test)
        
        # Print data shapes
        print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"           X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"           X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Get number of classes
        num_classes = y_train.shape[1]
        print(f"Number of classes: {num_classes}")
        
    except Exception as e:
        print(f"Error during data loading and preprocessing: {e}")
        return
            
    # 3. Applying sliding window technique
    print("\n3. Applying sliding window technique...")
    try:
        # Apply sliding window to create sequences
        X_train_win, y_train_win = apply_sliding_window(X_train, y_train, window_size, step_size)
        X_val_win, y_val_win = apply_sliding_window(X_val, y_val, window_size, step_size)
        X_test_win, y_test_win = apply_sliding_window(X_test, y_test, window_size, step_size)
        
        # Print sliding window shapes
        print(f"Sliding window shapes: X_train_win: {X_train_win.shape}, y_train_win: {y_train_win.shape}")
        print(f"                      X_val_win: {X_val_win.shape}, y_val_win: {y_val_win.shape}")
        print(f"                      X_test_win: {X_test_win.shape}, y_test_win: {y_test_win.shape}")
        
        # Check if shapes are compatible with the model
        print(f"Original data shapes: X_train_win: {X_train_win.shape}, X_val_win: {X_val_win.shape}")
        
        # Prepare data for the model
        # For CNN models, reshape to [samples, timesteps, features]
        if model_type == 'cnn':
            # Already in the right shape: [samples, timesteps, features]
            pass
        # For RNN and LSTM models, reshape to [samples, timesteps, features]
        elif model_type in ['rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced']:
            # Already in the right shape: [samples, timesteps, features]
            pass
        
        print(f"Prepared training data shape: {X_train_win.shape}")
        print(f"Prepared validation data shape: {X_val_win.shape}")
        print(f"Prepared test data shape: {X_test_win.shape}")
        
    except Exception as e:
        print(f"Error during sliding window application: {e}")
        return
        
    # 4. Defining hyperparameters to tune
    print("\n4. Defining hyperparameters to tune...")
    
    # Get hyperparameter grid
    param_grid = get_hyperparameter_grid(model_type)
    
    # Filter parameters based on command line arguments
    param_grid = filter_hyperparameters(
        param_grid, model_type,
        use_focal_loss=use_focal_loss,
        use_wks=use_wks,
        custom_class_weights=custom_class_weights,
        use_aquila_optimizer=use_aquila_optimizer,
        use_adaptive_learning=use_adaptive_learning
    )
    
    # Print hyperparameter grid
    print("\nHyperparameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Print tuning method
    print(f"\nUsing {tuning_method} search with {n_jobs} parallel jobs")
    
    # Calculate total number of combinations
    total_combinations = 1
    for param, values in param_grid.items():
        total_combinations *= len(values)
    print(f"Total hyperparameter combinations to try: {total_combinations}")
        
    # 5. Performing hyperparameter tuning
    print("\n5. Performing hyperparameter tuning...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Perform hyperparameter tuning
    try:
        final_model, best_params, test_metrics = tune_hyperparameters(
            X_train_win, y_train_win, X_val_win, y_val_win, X_test_win, y_test_win,
            model_type=model_type,
            train_model_func=train_model,
            use_adaptive_learning=use_adaptive_learning,
            use_aquila_optimizer=use_aquila_optimizer,
            use_focal_loss=use_focal_loss,
            use_wks=use_wks,
            custom_class_weights=custom_class_weights,
            tuning_method=tuning_method,
            n_trials=n_trials,
            n_jobs=n_jobs,
            epochs=epochs
        )
        
        # Print final results
        print("\n6. Final Results:")
        print(f"Best parameters: {best_params}")
        print(f"Test metrics: {test_metrics}")
        
        # If reliability analysis is enabled, perform it
        if reliability_analysis:
            print("\n7. Performing reliability analysis...")
            try:
                from dynamic_reliability import analyze_reliability
                
                reliability_results = analyze_reliability(
                    final_model, X_test_win, y_test_win,
                    model_type=model_type,
                    threshold=0.8
                )
                
                print(f"Reliability analysis results: {reliability_results}")
                
            except Exception as e:
                print(f"Error during reliability analysis: {e}")
        
        print("\nHyperparameter tuning completed successfully!")
        
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for NPP fault monitoring models')
    parser.add_argument('--model_type', type=str, default='cnn', 
                        choices=['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced'],
                        help='Type of model to tune')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for sliding window')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for sliding window')
    parser.add_argument('--tuning_method', type=str, default='grid', choices=['grid', 'random', 'bayesian'],
                        help='Method for hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials for random or Bayesian search')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss for training')
    parser.add_argument('--use_wks', action='store_true', help='Use WKS features for training')
    parser.add_argument('--custom_class_weights', action='store_true', help='Use custom class weights for training')
    parser.add_argument('--use_adaptive_learning', action='store_true', help='Use adaptive learning rate')
    parser.add_argument('--use_aquila_optimizer', action='store_true',
                        help='Use Aquila optimizer (only for SIAO models)')
    parser.add_argument('--reliability_analysis', action='store_true', 
                        help='Perform reliability analysis after tuning')

    # Parse arguments and call main function
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(
        model_type=args.model_type,
        window_size=args.window_size,
        step_size=args.step_size,
        tuning_method=args.tuning_method,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        epochs=args.epochs,
        use_focal_loss=args.use_focal_loss,
        use_wks=args.use_wks,
        custom_class_weights=args.custom_class_weights,
        use_adaptive_learning=args.use_adaptive_learning,
        use_aquila_optimizer=args.use_aquila_optimizer,
        reliability_analysis=args.reliability_analysis
    )
