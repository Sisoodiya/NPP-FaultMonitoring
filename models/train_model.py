# models/train_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
import json
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Import local modules
from data_preprocessing import process_pipeline
from model_cnn import build_cnn
from model_rnn import build_rnn
from model_lstm import build_lstm
from model_cnn_lstm import build_cnn_lstm as build_hybrid_model

# Import new components
from feature_extraction_new import extract_statistical_features, extract_all_features
from model_siao_cnn_ornn import build_siao_cnn_ornn, train_siao_cnn_ornn, AquilaOptimizerCallback
from dynamic_reliability import analyze_reliability, plot_reliability_curve, generate_reliability_report
from adaptive_learning import AdaptiveModelUpdater, AdaptiveOnlineFaultMonitor

# Try to import SIAO optimizer
try:
    from siao_optimizer import aquila_optimizer, optimize_wks_weights
    SIAO_AVAILABLE = True
except ImportError:
    SIAO_AVAILABLE = False
    print("SIAO optimizer not available. Using default optimization.")

# Import optimized models
from model_cnn import train_cnn
from model_rnn import train_rnn
from model_lstm import train_lstm
from model_cnn_rnn import train_cnn_rnn
from model_cnn_lstm import train_cnn_lstm
from model_siao_cnn_ornn import train_siao_cnn_ornn


def train_model(X_train, y_train, X_val, y_val, model_type='cnn', model_name=None, 
                epochs=50, batch_size=32, class_weights=None, use_adaptive_learning=False,
                use_aquila_optimizer=True, use_advanced_features=False, **model_params):
    """
    Train a model using the optimized model training functions.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_type: Type of model to train ('cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao')
        model_name: Name for saving the model (defaults to model_type if None)
        epochs: Number of training epochs
        batch_size: Batch size for training
        class_weights: Class weights for imbalanced data
        use_adaptive_learning: Whether to use adaptive learning techniques
        use_aquila_optimizer: Whether to use the Aquila Optimizer (for SIAO model)
        use_advanced_features: Whether advanced features were used
        **model_params: Additional parameters for the specific model
        
    Returns:
        tuple: (trained_model, history)
    """
    if model_name is None:
        model_name = model_type
    
    # Create directories for saving models and logs
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('analysis/plots', exist_ok=True)
    os.makedirs('analysis/text', exist_ok=True)
    
    # Get input shape from training data
    input_shape = X_train.shape[1:]
    
    # Get number of classes from training labels
    if len(y_train.shape) > 1:
        num_classes = y_train.shape[1]
    else:
        num_classes = len(np.unique(y_train))
    
    print(f"\nTraining {model_type.upper()} model with the following configuration:")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Using adaptive learning: {use_adaptive_learning}")
    print(f"Using advanced features: {use_advanced_features}")
    if model_type == 'siao':
        print(f"Using Aquila Optimizer: {use_aquila_optimizer}")
    
    # Adjust model_params to match the expected parameter names and filter out unexpected parameters
    adjusted_params = model_params.copy()
    
    # Rename 'dropout' to 'dropout_rate' for models that use it
    if 'dropout' in adjusted_params and model_type in ['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm']:
        adjusted_params['dropout_rate'] = adjusted_params.pop('dropout')
    
    # Define allowed parameters for each model type
    allowed_params = {
        'cnn': ['dropout_rate', 'l2_reg', 'learning_rate'],
        'rnn': ['dropout_rate', 'recurrent_dropout', 'l2_reg', 'learning_rate', 'use_gru'],
        'lstm': ['dropout_rate', 'recurrent_dropout', 'l2_reg', 'learning_rate'],
        'cnn_rnn': ['dropout_rate', 'recurrent_dropout', 'l2_reg', 'learning_rate', 'use_gru'],
        'cnn_lstm': ['dropout_rate', 'recurrent_dropout', 'l2_reg', 'learning_rate'],
        'siao': ['dropout_rate', 'l2_reg', 'use_gru']
    }
    
    # Filter out parameters that aren't allowed for the specific model type
    if model_type in allowed_params:
        filtered_params = {k: v for k, v in adjusted_params.items() if k in allowed_params[model_type]}
    else:
        filtered_params = adjusted_params  # Keep all params for unknown model types
    
    # Select the appropriate training function based on model type
    if model_type == 'cnn':
        model, history = train_cnn(
            X_train, y_train, X_val, y_val, 
            input_shape=input_shape, num_classes=num_classes,
            epochs=epochs, batch_size=batch_size, class_weights=class_weights,
            **filtered_params
        )
    elif model_type == 'rnn':
        model, history = train_rnn(
            X_train, y_train, X_val, y_val, 
            input_shape=input_shape, num_classes=num_classes,
            epochs=epochs, batch_size=batch_size, class_weights=class_weights,
            **filtered_params
        )
    elif model_type == 'lstm':
        model, history = train_lstm(
            X_train, y_train, X_val, y_val, 
            input_shape=input_shape, num_classes=num_classes,
            epochs=epochs, batch_size=batch_size, class_weights=class_weights,
            **filtered_params
        )
    elif model_type == 'cnn_rnn':
        model, history = train_cnn_rnn(
            X_train, y_train, X_val, y_val, 
            input_shape=input_shape, num_classes=num_classes,
            epochs=epochs, batch_size=batch_size, class_weights=class_weights,
            **filtered_params
        )
    elif model_type == 'cnn_lstm':
        model, history = train_cnn_lstm(
            X_train, y_train, X_val, y_val, 
            input_shape=input_shape, num_classes=num_classes,
            epochs=epochs, batch_size=batch_size, class_weights=class_weights,
            **filtered_params
        )
    elif model_type == 'siao':
        model, history = train_siao_cnn_ornn(
            X_train, y_train, X_val, y_val, 
            input_shape=input_shape, num_classes=num_classes,
            epochs=epochs, batch_size=batch_size, class_weights=class_weights,
            use_aquila_optimizer=use_aquila_optimizer,
            use_gru=True,  # Use GRU for better performance
            **filtered_params
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Apply adaptive learning if requested
    if use_adaptive_learning:
        print("\nApplying adaptive learning to improve model...")
        try:
            # Create adaptive model updater
            adaptive_updater = AdaptiveModelUpdater(
                model_path=f'trained_models/{model_name}_initial.h5',
                window_size=model_params.get('window_size', 100),
                step_size=model_params.get('step_size', 10),
                learning_rate=0.0005  # Lower learning rate for fine-tuning
            )
            
            # Save initial model before adaptive learning
            model.save(f'trained_models/{model_name}_initial.h5')
            
            # Add validation data to adaptive updater for supervised updates
            adaptive_updater.add_batch_data(X_val, y_val)
            
            # Update model
            adaptive_updater.update_model(force=True)
            
            # Save updated model
            adaptive_updater.save_model(f'trained_models/{model_name}_adaptive.h5')
            
            # Plot performance history
            adaptive_updater.plot_performance_history(f'analysis/plots/{model_name}_adaptive_performance.png')
            
            # Load the updated model
            model = load_model(f'trained_models/{model_name}_adaptive.h5')
            print("Adaptive learning completed successfully.")
        except Exception as e:
            print(f"Error during adaptive learning: {e}. Using original model.")
    
    # Save the final model
    model.save(f'trained_models/{model_name}.h5')
    print(f"Model saved to trained_models/{model_name}.h5")
    
    # Plot training history
    plot_training_history(history, model_name)
    
    return model, history


def plot_training_history(history, model_name):
    """
    Plot and save the training history.
    
    Args:
        history: Training history object
        model_name: Name of the model for saving plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs('analysis/plots', exist_ok=True)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'analysis/plots/{model_name}_training_history.png')
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for NPP fault monitoring')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao'],
                        help='Model type to train')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for sliding window')
    parser.add_argument('--step_size', type=int, default=None, help='Step size for sliding window')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of training data to use for validation')
    parser.add_argument('--balance', action='store_true', help='Whether to balance classes using SMOTE')
    parser.add_argument('--use_advanced_features', action='store_true', help='Whether to use advanced feature extraction')
    parser.add_argument('--include_wks', action='store_true', help='Whether to include WKS features (only used with advanced features)')
    parser.add_argument('--optimize_wks', action='store_true', help='Whether to optimize WKS weights using SIAO')
    parser.add_argument('--use_adaptive_learning', action='store_true', help='Whether to use adaptive learning techniques')
    parser.add_argument('--use_aquila_optimizer', action='store_true', help='Whether to use the Aquila Optimizer (for SIAO model)')
    parser.add_argument('--reliability_analysis', action='store_true', help='Whether to perform detailed reliability analysis')
    parser.add_argument('--feature_window', type=int, default=100, help='Window size for feature extraction')
    parser.add_argument('--feature_step', type=int, default=50, help='Step size for feature extraction')
    
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Training {args.model.upper()} model")
    print(f"{'='*50}\n")
    
    # 1. Load and preprocess data from the data directory
    print("Loading and preprocessing data...")
    processed_data, scaler = process_pipeline()

    # Determine the fault/condition column
    if 'fault_type' in processed_data.columns:
        fault_col = 'fault_type'
    elif 'condition' in processed_data.columns:
        fault_col = 'condition'
    elif 'fault' in processed_data.columns:
        fault_col = 'fault'
    else:
        print("Error: Could not find fault type column in the data.")
        exit(1)
    
    # Check if we should use advanced feature extraction
    if args.use_advanced_features:
        print("\nUsing advanced feature extraction as described in the research paper...")
        
        # Extract features using the methods from the research paper
        if args.include_wks:
            print("Including Weighted Kurtosis and Skewness (WKS) features...")
            print("This may take some time as it involves optimization...")
            features, y_onehot, encoder = extract_all_features(
                processed_data, 
                window_size=args.feature_window, 
                step_size=args.feature_step,
                time_col='time000000000' if 'time000000000' in processed_data.columns else None,
                include_wks=True
            )
        else:
            print("Extracting statistical features...")
            features, labels = extract_statistical_features(
                processed_data, 
                window_size=args.feature_window, 
                step_size=args.feature_step,
                time_col='time000000000' if 'time000000000' in processed_data.columns else None
            )
            # Encode labels
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(labels)
            num_classes = len(encoder.classes_)
            y_onehot = np.zeros((len(y_encoded), num_classes))
            y_onehot[np.arange(len(y_encoded)), y_encoded] = 1
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Number of classes: {len(encoder.classes_)}")
        
        # Split data into training, validation, and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_onehot, test_size=0.2, random_state=42, stratify=np.argmax(y_onehot, axis=1)
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
        )
        
        # We already have one-hot encoded labels from extract_all_features
        y_train_onehot = y_train
        y_val_onehot = y_val
        y_test_onehot = y_test
        
    else:
        # Use raw features as before
        print("\nUsing raw sensor data with sliding windows...")
        
        # Get features and labels
        feature_cols = [col for col in processed_data.columns 
                      if col != fault_col and 'time' not in col.lower()]
        X = processed_data[feature_cols]
        y = processed_data[fault_col]
        
        print(f"Data shape: {X.shape}")
        print(f"Number of classes: {len(y.unique())}")
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Split data into training, validation, and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
    # Balance classes if requested
    if args.balance and not args.use_advanced_features:  # Only apply SMOTE to raw features
        print("\nBalancing classes using SMOTE...")
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame to maintain column names
        if isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        else:
            X_train = X_train_resampled
        y_train = y_train_resampled
        
        print(f"Balanced training data shape: {X_train.shape}")
    
    # Convert labels to one-hot encoding if not already done
    if not args.use_advanced_features:
        num_classes = len(encoder.classes_)
        y_train_onehot = np.zeros((len(y_train), num_classes))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        y_val_onehot = np.zeros((len(y_val), num_classes))
        y_val_onehot[np.arange(len(y_val)), y_val] = 1
        
        y_test_onehot = np.zeros((len(y_test), num_classes))
        y_test_onehot[np.arange(len(y_test)), y_test] = 1
        
        print("Class distribution after SMOTE:")
        unique, counts = np.unique(y_train, return_counts=True)
        for i, (cls, count) in enumerate(zip(encoder.classes_, counts)):
            print(f"{cls}: {count} samples ({count/len(y_train)*100:.2f}%)")

    # Create sliding windows for DL models
    window_size = args.window_size
    step_size = args.step_size
    print(f"\nCreating sliding windows (size={window_size}, step={step_size})...")

    def prepare_sliding_window_data(data, window_size, step_size):
        """Create sliding windows from time series data.
        
        Args:
            data: Input data (samples, features)
            window_size: Size of each window
            step_size: Step size between windows
            
        Returns:
            tuple: (windowed_data, window_indices)
        """
        n_samples, n_features = data.shape
        n_windows = (n_samples - window_size) // step_size + 1
        
        # Initialize arrays
        windows = np.zeros((n_windows, window_size, n_features))
        indices = np.zeros((n_windows, window_size), dtype=int)
        
        # Create windows
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            windows[i] = data[start_idx:end_idx]
            indices[i] = np.arange(start_idx, end_idx)
        
        return windows, indices
    
    def make_windows(X, y):
        """Create sliding windows for time series data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_windows, _ = prepare_sliding_window_data(X, window_size, step_size)
        
        # Create one-hot encoded labels for each window
        if len(y.shape) == 1:  # If y is not one-hot encoded
            n_classes = len(np.unique(y))
            y_windows = np.zeros((X_windows.shape[0], n_classes))
            for i in range(X_windows.shape[0]):
                start_idx = i * step_size
                if start_idx < len(y):
                    y_windows[i] = np.eye(n_classes)[y[start_idx]]
        else:  # If y is already one-hot encoded
            y_windows = np.zeros((X_windows.shape[0], y.shape[1]))
            for i in range(X_windows.shape[0]):
                start_idx = i * step_size
                if start_idx < len(y):
                    y_windows[i] = y[start_idx]
                    
        return X_windows, y_windows

    # Create windowed data for training, validation, and testing
    if not args.use_advanced_features:  # Only create windows for raw data
        print(f"\nCreating sliding windows with window_size={args.window_size}, step_size={args.step_size}...")
        
        if isinstance(X_train, pd.DataFrame):
            X_train_values = X_train.values
        else:
            X_train_values = X_train
            
        if isinstance(X_val, pd.DataFrame):
            X_val_values = X_val.values
        else:
            X_val_values = X_val
            
        if isinstance(X_test, pd.DataFrame):
            X_test_values = X_test.values
        else:
            X_test_values = X_test
        
        # Create sliding windows
        X_train_win, y_train_win = make_windows(X_train_values, y_train_onehot)
        X_val_win, y_val_win = make_windows(X_val_values, y_val_onehot)
        X_test_win, y_test_win = make_windows(X_test_values, y_test_onehot)
    else:
        # For advanced features, we don't need to create windows as they're already extracted
        # using the sliding window approach in the feature extraction process
        print("\nUsing pre-extracted features (no additional windowing needed)...")
        
        # Reshape for CNN models if needed
        if args.model in ['cnn', 'cnn_rnn', 'cnn_lstm', 'siao']:
            # Add a channel dimension for CNN models (samples, features) -> (samples, features, 1)
            X_train_win = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_win = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)
            X_test_win = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        else:
            # For RNN/LSTM models, reshape to (samples, timesteps, features)
            # Since our features are already extracted with temporal information,
            # we'll use a dummy timestep of 1
            X_train_win = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val_win = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])
            X_test_win = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        y_train_win = y_train_onehot
        y_val_win = y_val_onehot
        y_test_win = y_test_onehot
        
    print(f"Training data shape: {X_train_win.shape}")
    print(f"Validation data shape: {X_val_win.shape}")
    print(f"Test data shape: {X_test_win.shape}")

    # Calculate class weights if needed
    class_weights = None
    if args.balance and False:  # Disabled as we're using SMOTE
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))
    
    # 2. Train the model
    print("\nTraining model...")
    model_params = {
        'dropout': args.dropout if hasattr(args, 'dropout') else 0.3,
        'l2_reg': args.l2_reg if hasattr(args, 'l2_reg') else 0.001,
        'learning_rate': args.learning_rate if hasattr(args, 'learning_rate') else 0.001,
        'window_size': args.window_size,
        'step_size': args.step_size
    }
    
    start_time = time.time()
    trained_model, history = train_model(
        X_train_win, y_train_win, 
        X_val_win, y_val_win, 
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=class_weights,
        use_adaptive_learning=args.use_adaptive_learning,
        use_aquila_optimizer=args.use_aquila_optimizer,
        use_advanced_features=args.use_advanced_features,
        **model_params
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # 2. Evaluate the model
    print("\nEvaluating model on test set...")
    predictions = trained_model.predict(X_test_win)
    y_true = np.argmax(y_test_win, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {args.model.upper()}')
    plt.tight_layout()
    plt.savefig(f'analysis/plots/{args.model}_confusion_matrix.png')
    plt.close()

    # Save evaluation report
    os.makedirs('analysis/text', exist_ok=True)
    with open(f"analysis/text/{args.model}_evaluation.txt", "w") as f:
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=encoder.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nModel Parameters:\n")
        for param, value in model_params.items():
            f.write(f"{param}: {value}\n")

    # 3. Perform reliability analysis
    if args.reliability_analysis:
        print("\nPerforming detailed reliability analysis...")
        fault_data = processed_data.copy()
        fault_data['predicted'] = -1
        
        # Map predictions back to original indices
        pred_indices = X_test.index[:len(y_pred)]
        if len(pred_indices) > 0:
            fault_data.loc[pred_indices, 'predicted'] = encoder.inverse_transform(y_pred)
            
            # Perform reliability analysis for each fault type
            for fault_idx, fault_type in enumerate(encoder.classes_):
                if fault_type == 'steady_state' or fault_type == 'normal':
                    continue  # Skip normal operation states
                    
                print(f"Analyzing reliability for fault type: {fault_type}")
                reliability_results = analyze_reliability(
                    fault_data, 
                    fault_column='predicted', 
                    fault_value=fault_type,
                    time_col='time000000000' if 'time000000000' in fault_data.columns else None
                )
                
                # Generate comprehensive reliability report
                report = generate_reliability_report(reliability_results, fault_type=fault_type)
                
                # Save report to file
                with open(f"analysis/text/{args.model}_{fault_type}_reliability.txt", "w") as f:
                    f.write(report)
                
                # Plot reliability curve
                plot_reliability_curve(
                    reliability_results, 
                    title=f"Reliability Curve - {fault_type}",
                    save_path=f"analysis/plots/{args.model}_{fault_type}_reliability_curve.png"
                )
                
                # Print key reliability metrics
                print(f"  MTTF: {reliability_results['mttf']:.2f} time units")
                print(f"  Failure Rate: {reliability_results['failure_rate']:.6f} failures per time unit")
                if reliability_results['weibull_shape'] is not None:
                    print(f"  Weibull Shape Parameter: {reliability_results['weibull_shape']:.4f}")
                    print(f"  Weibull Scale Parameter: {reliability_results['weibull_scale']:.4f}")
                
                print(f"  Reliability analysis for {fault_type} completed")
        else:
            print("No test predictions available for reliability analysis")
    else:
        print("\nSkipping detailed reliability analysis (use --reliability_analysis to enable)")
        
        # Perform basic reliability metrics calculation
        print("Calculating basic reliability metrics...")
        # Calculate accuracy over time to estimate reliability
        window_size = 20  # Use a sliding window to calculate accuracy over time
        accuracies = []
        
        for i in range(0, len(y_pred) - window_size, window_size // 2):
            window_true = y_true[i:i+window_size]
            window_pred = y_pred[i:i+window_size]
            acc = accuracy_score(window_true, window_pred)
            accuracies.append(acc)
        
        if accuracies:
            print(f"Mean accuracy over time: {np.mean(accuracies):.4f}")
            print(f"Accuracy stability (std dev): {np.std(accuracies):.4f}")
            
            # Plot accuracy over time as a simple reliability indicator
            plt.figure(figsize=(10, 6))
            plt.plot(accuracies)
            plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.4f}')
            plt.xlabel('Time Window')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy Over Time - {args.model.upper()}')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'analysis/plots/{args.model}_accuracy_over_time.png')
            plt.close()
    
    print(f"\n{'='*50}")
    print(f"Training and evaluation of {args.model.upper()} model completed")
    print(f"Results saved to analysis/text/{args.model}_evaluation.txt")
    print(f"{'='*50}\n")
