"""
Hyperparameter Tuning Script for NPP Fault Monitoring Deep Learning Models.

This script performs hyperparameter tuning for various deep learning models to optimize their performance
for fault detection in nuclear power plants. Supported models include:
- CNN
- RNN
- LSTM
- CNN-RNN
- CNN-LSTM
- SIAO-CNN-ORNN
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import custom modules
from data_preprocessing import process_pipeline

# Import feature extraction modules
from feature_extraction_new import extract_statistical_features, extract_all_features

# Import model modules
from model_cnn import build_cnn
from model_rnn import build_rnn
from model_lstm import build_lstm
from model_cnn_lstm import build_cnn_lstm as build_hybrid_model
from model_cnn_rnn import train_cnn_rnn
from model_cnn_lstm import train_cnn_lstm
from model_siao_cnn_ornn import build_siao_cnn_ornn, train_siao_cnn_ornn, AquilaOptimizerCallback

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
                use_aquila_optimizer=False, **model_params):
    """Train a model with the given parameters."""
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    # Build model based on type
    if model_type == 'cnn':
        model = build_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            filters=model_params.get('filters', 32),
            kernel_size=model_params.get('kernel_size', 3),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'rnn':
        model = build_rnn(
            input_shape=input_shape,
            num_classes=num_classes,
            units=model_params.get('units', 64),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'lstm':
        model = build_lstm(
            input_shape=input_shape,
            num_classes=num_classes,
            units=model_params.get('units', 64),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'cnn_rnn':
        model = build_hybrid_model(
            input_shape=input_shape,
            num_classes=num_classes,
            model_type='cnn_rnn',
            cnn_filters=model_params.get('cnn_filters', 32),
            cnn_kernel_size=model_params.get('cnn_kernel_size', 3),
            rnn_units=model_params.get('rnn_units', 64),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'cnn_lstm':
        model = build_hybrid_model(
            input_shape=input_shape,
            num_classes=num_classes,
            model_type='cnn_lstm',
            cnn_filters=model_params.get('cnn_filters', 32),
            cnn_kernel_size=model_params.get('cnn_kernel_size', 3),
            rnn_units=model_params.get('lstm_units', 64),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    elif model_type == 'siao':
        model = build_siao_cnn_ornn(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=model_params.get('dropout_rate', 0.3),
            recurrent_dropout=model_params.get('recurrent_dropout', 0.2),
            l2_reg=model_params.get('l2_reg', 0.001),
            learning_rate=model_params.get('learning_rate', 0.001)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    # Add model checkpoint if a name is provided
    if model_name:
        os.makedirs('checkpoints', exist_ok=True)
        callbacks.append(ModelCheckpoint(
            f'checkpoints/{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))
    
    # Add Aquila optimizer callback if applicable
    if use_aquila_optimizer and SIAO_AVAILABLE and model_type == 'siao':
        callbacks.append(AquilaOptimizerCallback())
    
    # Add adaptive learning if requested
    if use_adaptive_learning:
        adaptive_updater = AdaptiveModelUpdater(model, learning_rate=model_params.get('learning_rate', 0.001))
        callbacks.append(adaptive_updater.get_callback())
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history

def main(model_type='cnn_lstm', use_advanced_features=False, include_wks=False, optimize_wks=False,
         feature_window=100, feature_step=50, window_size=10, step_size=5,
         use_adaptive_learning=False, use_aquila_optimizer=False, reliability_analysis=False):
    """
    Main function to run hyperparameter tuning.
    """
    print("NPP Fault Monitoring - Hyperparameter Tuning")
    print("=" * 50)
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not available. Cannot perform hyperparameter tuning.")
        return
    
    # Validate model type
    valid_models = ['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao']
    if model_type not in valid_models:
        print(f"Error: Invalid model type '{model_type}'. Valid options are: {valid_models}")
        return
        
    print(f"Selected model type: {model_type}")
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Advanced features: {'Enabled' if use_advanced_features else 'Disabled'}")
    if use_advanced_features:
        print(f"  WKS features: {'Included' if include_wks else 'Not included'}")
        if include_wks:
            print(f"  WKS optimization: {'Enabled' if optimize_wks else 'Disabled'}")
        print(f"  Feature window size: {feature_window}")
        print(f"  Feature step size: {feature_step}")
    else:
        print(f"  Window size: {window_size}")
        print(f"  Step size: {step_size}")
    
    print(f"  Adaptive learning: {'Enabled' if use_adaptive_learning else 'Disabled'}")
    print(f"  Aquila optimizer: {'Enabled' if use_aquila_optimizer else 'Disabled'}")
    print(f"  Reliability analysis: {'Enabled' if reliability_analysis else 'Disabled'}")
    
    # Check for SIAO components if requested
    if (optimize_wks or use_aquila_optimizer) and not SIAO_AVAILABLE:
        print("\nWARNING: SIAO components were requested but are not available.")
        print("         Falling back to standard methods.")
        optimize_wks = False
        use_aquila_optimizer = False
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('analysis/plots', exist_ok=True)
    os.makedirs('analysis/text', exist_ok=True)
    
    try:
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        processed_data, scaler = process_pipeline()
        
        if processed_data is None or processed_data.empty:
            print("Error: No data processed.")
            return
        
        print(f"Processed data shape: {processed_data.shape}")
        
        # 2. Prepare data for deep learning
        print("\n2. Preparing data for deep learning...")
        
        # Get the fault type column
        if 'fault_type' in processed_data.columns:
            target_col = 'fault_type'
        elif 'condition' in processed_data.columns:
            target_col = 'condition'
        else:
            print("Error: No fault_type or condition column found in the data.")
            return
        
        # Split data into features and labels
        X = processed_data.drop(target_col, axis=1)
        y = processed_data[target_col]
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        
        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Convert labels to one-hot encoding for deep learning models
        y_train_onehot = np.eye(num_classes)[y_train]
        y_val_onehot = np.eye(num_classes)[y_val]
        y_test_onehot = np.eye(num_classes)[y_test]
        
        # Prepare data based on feature extraction choice
        if use_advanced_features:
            print("Using advanced feature extraction...")
            
            # Extract features from each dataset using the new feature extraction module
            print(f"Extracting features with window size {feature_window} and step size {feature_step}...")
            X_train_features, y_train_features, _ = extract_all_features(
                X_train.copy(), window_size=feature_window, step_size=feature_step,
                include_wks=include_wks, optimize_wks=optimize_wks
            )
            X_val_features, y_val_features, _ = extract_all_features(
                X_val.copy(), window_size=feature_window, step_size=feature_step,
                include_wks=include_wks, optimize_wks=optimize_wks
            )
            X_test_features, y_test_features, _ = extract_all_features(
                X_test.copy(), window_size=feature_window, step_size=feature_step,
                include_wks=include_wks, optimize_wks=optimize_wks
            )
            
            # Print feature information
            print(f"Extracted feature dimensions: {X_train_features.shape[1]}")
            if include_wks:
                print("WKS features included in the feature set")
                if optimize_wks and SIAO_AVAILABLE:
                    print("WKS weights optimized using SIAO")
            
            # Use extracted features
            X_train_win = X_train_features.values
            X_val_win = X_val_features.values
            X_test_win = X_test_features.values
            
            # Adjust labels to match feature extraction
            y_train_win = y_train_onehot
            y_val_win = y_val_onehot
            y_test_win = y_test_onehot
            
            # Ensure labels match the number of feature samples
            if len(X_train_win) != len(y_train_win):
                print(f"Adjusting training labels: {len(y_train_win)} -> {len(X_train_win)}")
                # Repeat labels to match feature samples
                repeat_factor = len(X_train_win) // len(y_train_win) + 1
                y_train_win = np.repeat(y_train_win, repeat_factor, axis=0)[:len(X_train_win)]
            
            if len(X_val_win) != len(y_val_win):
                print(f"Adjusting validation labels: {len(y_val_win)} -> {len(X_val_win)}")
                repeat_factor = len(X_val_win) // len(y_val_win) + 1
                y_val_win = np.repeat(y_val_win, repeat_factor, axis=0)[:len(X_val_win)]
            
            if len(X_test_win) != len(y_test_win):
                print(f"Adjusting test labels: {len(y_test_win)} -> {len(X_test_win)}")
                repeat_factor = len(X_test_win) // len(y_test_win) + 1
                y_test_win = np.repeat(y_test_win, repeat_factor, axis=0)[:len(X_test_win)]
            
        else:
            print("Using raw data with sliding windows...")
            
            # Create sliding windows for raw data
            X_train_windows = create_windows(X_train.values, window_size, step_size)
            X_val_windows = create_windows(X_val.values, window_size, step_size)
            X_test_windows = create_windows(X_test.values, window_size, step_size)
            
            # Adjust labels to match windowing
            y_train_windows = np.repeat(y_train_onehot, (len(X_train.values) - window_size) // step_size + 1, axis=0)
            y_val_windows = np.repeat(y_val_onehot, (len(X_val.values) - window_size) // step_size + 1, axis=0)
            y_test_windows = np.repeat(y_test_onehot, (len(X_test.values) - window_size) // step_size + 1, axis=0)
            
            # Truncate labels if necessary
            y_train_windows = y_train_windows[:len(X_train_windows)]
            y_val_windows = y_val_windows[:len(X_val_windows)]
            y_test_windows = y_test_windows[:len(X_test_windows)]
            
            # Use windowed data
            X_train_win = X_train_windows
            X_val_win = X_val_windows
            X_test_win = X_test_windows
            
            y_train_win = y_train_windows
            y_val_win = y_val_windows
            y_test_win = y_test_windows
        
        # Reshape data for CNN models if needed
        if model_type in ['cnn', 'cnn_rnn', 'cnn_lstm', 'siao']:
            X_train_win = X_train_win.reshape(X_train_win.shape[0], X_train_win.shape[1], 1)
            X_val_win = X_val_win.reshape(X_val_win.shape[0], X_val_win.shape[1], 1)
            X_test_win = X_test_win.reshape(X_test_win.shape[0], X_test_win.shape[1], 1)
        
        print(f"Prepared training data shape: {X_train_win.shape}")
        print(f"Prepared validation data shape: {X_val_win.shape}")
        print(f"Prepared test data shape: {X_test_win.shape}")
        
        # 3. Define hyperparameter search space
        print("\n3. Defining hyperparameter search space...")
        if model_type == 'cnn':
            param_grid = {
                'filters': [32, 64],
                'kernel_size': [3, 5],
                'dropout_rate': [0.3, 0.4],
                'learning_rate': [0.001, 0.0005],
                'l2_reg': [0.001, 0.0005]
            }
        elif model_type == 'rnn':
            param_grid = {
                'units': [64, 128],
                'dropout_rate': [0.3, 0.4],
                'recurrent_dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'l2_reg': [0.001, 0.0005]
            }
        elif model_type == 'lstm':
            param_grid = {
                'units': [64, 128],
                'dropout_rate': [0.3, 0.4],
                'recurrent_dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'l2_reg': [0.001, 0.0005]
            }
        elif model_type == 'cnn_rnn':
            param_grid = {
                'cnn_filters': [32, 64],
                'cnn_kernel_size': [3, 5],
                'rnn_units': [64, 128],
                'dropout_rate': [0.3, 0.4],
                'recurrent_dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'l2_reg': [0.001, 0.0005]
            }
        elif model_type == 'cnn_lstm':
            param_grid = {
                'cnn_filters': [32, 64],
                'cnn_kernel_size': [3, 5],
                'lstm_units': [64, 128],
                'dropout_rate': [0.3, 0.4],
                'recurrent_dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'l2_reg': [0.001, 0.0005]
            }
        elif model_type == 'siao':
            param_grid = {
                'cnn_filters': [32, 64],
                'cnn_kernel_size': [3, 5],
                'rnn_units': [64, 128],
                'dropout_rate': [0.3, 0.4],
                'recurrent_dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'l2_reg': [0.001, 0.0005]
            }
            # Add SIAO-specific parameters if available
            if use_aquila_optimizer and SIAO_AVAILABLE:
                param_grid['use_aquila_optimizer'] = [True]
            if use_adaptive_learning:
                param_grid['use_adaptive_learning'] = [True]
        
        # Generate all combinations of hyperparameters
        import itertools
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        print(f"Total hyperparameter combinations to try: {len(param_combinations)}")
        
        # 4. Perform hyperparameter tuning
        print("\n4. Performing hyperparameter tuning...")
        
        # Initialize tracking variables for best model
        best_val_accuracy = 0.0
        best_model = None
        best_params = None
        best_history = None
        
        # Try each combination
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            print(f"\nTrying combination {i+1}/{len(param_combinations)}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            
            # Train model with current hyperparameters
            model, history = train_model(
                X_train_win, y_train_win,
                X_val_win, y_val_win,
                model_type=model_type,
                model_name=f"{model_type}_tune_{i}",
                epochs=30,
                batch_size=32,
                use_adaptive_learning=use_adaptive_learning,
                use_aquila_optimizer=use_aquila_optimizer,
                **params
            )
            
            # Evaluate model on validation set
            val_loss, val_accuracy = model.evaluate(X_val_win, y_val_win, verbose=0)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Check if this is the best model so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model
                best_params = params
                best_history = history
                print(f"New best model found! Validation accuracy: {val_accuracy:.4f}")
        
        print("\nHyperparameter tuning completed.")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print("Best hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # 5. Evaluate the best model on test data
        if best_model is not None:
            print("\n5. Evaluating best model on test data...")
            test_loss, test_accuracy = best_model.evaluate(X_test_win, y_test_win, verbose=1)
            print(f"Test accuracy: {test_accuracy:.4f}")
            
            # Get predictions
            y_pred = np.argmax(best_model.predict(X_test_win), axis=1)
            y_true = np.argmax(y_test_win, axis=1)
            
            # Calculate additional metrics
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Perform reliability analysis if requested
            if reliability_analysis:
                print("\nPerforming reliability analysis on best model...")
                # Create a dataframe for reliability analysis
                reliability_df = pd.DataFrame({
                    'true': encoder.inverse_transform(y_true),
                    'predicted': encoder.inverse_transform(y_pred)
                })
                
                # Add a dummy time column if needed
                if 'time' not in reliability_df.columns:
                    reliability_df['time'] = np.arange(len(reliability_df))
                
                # Perform reliability analysis for each fault type
                for fault_type in np.unique(reliability_df['predicted']):
                    if fault_type == 'normal' or fault_type == 'steady_state':
                        continue  # Skip normal operation states
                        
                    print(f"Analyzing reliability for fault type: {fault_type}")
                    reliability_results = analyze_reliability(
                        reliability_df,
                        fault_column='predicted',
                        fault_value=fault_type
                    )
                    
                    # Generate reliability report
                    report = generate_reliability_report(reliability_results, fault_type=fault_type)
                    print(f"MTTF: {reliability_results['mttf']:.2f}")
                    print(f"Failure Rate: {reliability_results['failure_rate']:.6f}")
                    
                    # Save report to file
                    os.makedirs('analysis/reliability', exist_ok=True)
                    with open(f"analysis/reliability/{model_type}_{fault_type}_reliability.txt", "w") as f:
                        f.write(report)
        
            # 6. Save best model and hyperparameters
            # Create directories if they don't exist
            os.makedirs('trained_models', exist_ok=True)
            
            # Save best model
            best_model.save(f'trained_models/best_{model_type}_model')
            
            # Save best hyperparameters
            joblib.dump(best_params, f'trained_models/best_{model_type}_hyperparameters.pkl')
            
            print(f"\nBest hyperparameters saved to trained_models/best_{model_type}_hyperparameters.pkl")
            print(f"Best model saved to trained_models/best_{model_type}_model")
            
            # Plot training history
            if best_history is not None:
                plt.figure(figsize=(12, 5))
                
                # Plot accuracy
                plt.subplot(1, 2, 1)
                plt.plot(best_history.history['accuracy'], label='Training Accuracy')
                plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
                plt.title(f'{model_type.upper()} Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                # Plot loss
                plt.subplot(1, 2, 2)
                plt.plot(best_history.history['loss'], label='Training Loss')
                plt.plot(best_history.history['val_loss'], label='Validation Loss')
                plt.title(f'{model_type.upper()} Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'analysis/plots/{model_type}_tuning_history.png')
                plt.close()
                
                print(f"Training history plot saved to analysis/plots/{model_type}_tuning_history.png")
        else:
            print("No successful model was found during hyperparameter tuning.")
    
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nHyperparameter tuning completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for NPP fault monitoring models')
    parser.add_argument('--model', type=str, default='cnn_lstm',
                        choices=['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao'],
                        help='Model type to tune')
    
    # Feature extraction options
    parser.add_argument('--use_advanced_features', action='store_true',
                        help='Use advanced feature extraction instead of raw data')
    parser.add_argument('--include_wks', action='store_true',
                        help='Include Weighted Kurtosis and Skewness (WKS) features')
    parser.add_argument('--optimize_wks', action='store_true',
                        help='Optimize WKS weights using SIAO (requires SIAO optimizer)')
    parser.add_argument('--feature_window', type=int, default=100,
                        help='Window size for feature extraction')
    parser.add_argument('--feature_step', type=int, default=50,
                        help='Step size for feature extraction')
    
    # Window options for raw data
    parser.add_argument('--window_size', type=int, default=10,
                        help='Size of sliding window for time series data (when not using advanced features)')
    parser.add_argument('--step_size', type=int, default=5,
                        help='Step size for sliding window (when not using advanced features)')
    
    # Advanced model options
    parser.add_argument('--use_aquila_optimizer', action='store_true',
                        help='Use the Self-Improved Aquila Optimizer (SIAO) for training')
    parser.add_argument('--use_adaptive_learning', action='store_true',
                        help='Use adaptive learning techniques for model training')
    parser.add_argument('--reliability_analysis', action='store_true',
                        help='Perform detailed reliability analysis on the best model')
    
    args = parser.parse_args()
    
    main(model_type=args.model, 
         use_advanced_features=args.use_advanced_features,
         include_wks=args.include_wks,
         optimize_wks=args.optimize_wks,
         feature_window=args.feature_window,
         feature_step=args.feature_step,
         window_size=args.window_size,
         step_size=args.step_size,
         use_adaptive_learning=args.use_adaptive_learning,
         use_aquila_optimizer=args.use_aquila_optimizer,
         reliability_analysis=args.reliability_analysis)
