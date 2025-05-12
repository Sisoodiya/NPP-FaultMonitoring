"""
Apply Tuned Hyperparameters to Deep Learning Models for NPP Fault Monitoring

This script loads the best hyperparameters found during tuning and applies them
to train a selected model on the full dataset, then evaluates its performance.
Supported models include:
- CNN
- RNN
- LSTM
- CNN-RNN
- CNN-LSTM
- SIAO-CNN-RNN
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import TensorFlow and Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Please install TensorFlow to use this script.")

# Import custom modules
from data_preprocessing import process_pipeline

# Import model modules
from model_cnn import build_cnn_model, train_cnn_model
from model_rnn import build_rnn_model, train_rnn_model
from model_lstm import build_lstm_model, train_lstm_model
from model_cnn_rnn import build_cnn_rnn_model, train_cnn_rnn_model
from model_cnn_lstm import build_cnn_lstm_model, train_cnn_lstm_model
from model_siao_cnn_ornn import build_siao_cnn_ornn_model, train_siao_cnn_ornn_model

def main(model_type='cnn_lstm'):
    """
    Main function to apply tuned hyperparameters to a CNN-LSTM model.
    """
    print("NPP Fault Monitoring - Applying Tuned Hyperparameters")
    print("=" * 50)
    
    # Validate model type
    valid_models = ['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao']
    if model_type not in valid_models:
        print(f"Error: Invalid model type '{model_type}'. Valid options are: {valid_models}")
        return
        
    print(f"Selected model type: {model_type}")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not available. Cannot train the model.")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    try:
        # Process all data files in the data directory
        data_dir = 'data'
        processed_data, file_names = process_pipeline(data_dir)
        
        if processed_data is None or processed_data.empty:
            print("Error: No data processed.")
            return
        
        print(f"Processed data shape: {processed_data.shape}")
        
    except Exception as e:
        print(f"Error during data loading and preprocessing: {e}")
        return
    
    # 2. Extract features and prepare data
    print("\n2. Extracting features and preparing data...")
    try:
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
        
        # Extract features directly from the processed data
        # Remove the fault column and any time columns for feature extraction
        feature_cols = [col for col in processed_data.columns 
                      if col != fault_col and 'time' not in col.lower()]
        
        # Get features and labels
        X = processed_data[feature_cols]
        y = processed_data[fault_col]
        
        print(f"Raw features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Check for NaN values in features
        if X.isna().any().any():
            print("Warning: NaN values found in features. Imputing with mean...")
            X = X.fillna(X.mean())
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        print(f"Number of unique classes: {len(encoder.classes_)}")
        print(f"Classes: {encoder.classes_}")
        
        # Save the encoder for future use
        joblib.dump(encoder, 'models/label_encoder.pkl')
        print("Label encoder saved to models/label_encoder.pkl")
        
        # One-hot encode labels
        y_onehot = pd.get_dummies(y_encoded).values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save the scaler for future use
        joblib.dump(scaler, 'models/feature_scaler.pkl')
        print("Feature scaler saved to models/feature_scaler.pkl")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Reshape data for CNN-LSTM (samples, timesteps, features)
        window_size = 10  # Number of timesteps in each window
        
        # Function to create sliding windows
        def create_windows(data, window_size):
            windows = []
            for i in range(len(data) - window_size + 1):
                windows.append(data[i:i+window_size])
            return np.array(windows)
        
        # Create sliding windows for training and test sets
        X_train_windows = create_windows(X_train, window_size)
        X_test_windows = create_windows(X_test, window_size)
        
        # For labels, we'll use the label of the last timestep in each window
        y_train_windows = y_train[window_size-1:]
        y_test_windows = y_test[window_size-1:]
        
        print(f"Training data shape: {X_train_windows.shape}")
        print(f"Test data shape: {X_test_windows.shape}")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Load best hyperparameters
    print("\n3. Loading best hyperparameters...")
    try:
        # Load best hyperparameters for the selected model
        best_params_file = f'trained_models/best_{model_type}_hyperparameters.pkl'
        if not os.path.exists(best_params_file):
            print(f"Error: Best hyperparameters file '{best_params_file}' not found.")
            print(f"Please run hyperparameter tuning for {model_type} model first.")
            return
        
        best_params = joblib.load(best_params_file)
        print("Best hyperparameters loaded:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        return
    
    # 4. Train model with best hyperparameters
    print("\n4. Training model with best hyperparameters...")
    try:
        # Get number of classes
        num_classes = y_train_windows.shape[1]
        
        # Build model based on model type with best hyperparameters
        if model_type == 'cnn':
            model = build_cnn_model(
                input_shape=(X_train_windows.shape[1], X_train_windows.shape[2]),
                num_classes=num_classes,
                filters=best_params['conv_filters'],
                kernel_size=best_params['kernel_size'],
                num_conv_layers=best_params['conv_layers'],
                dense_units=best_params['dense_units'],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation'],
                l2_reg=best_params['l2_reg']
            )
            
        elif model_type == 'rnn':
            model = build_rnn_model(
                input_shape=(X_train_windows.shape[1], X_train_windows.shape[2]),
                num_classes=num_classes,
                rnn_units=best_params['rnn_units'],
                num_rnn_layers=best_params['rnn_layers'],
                bidirectional=best_params['bidirectional'],
                use_gru=best_params['use_gru'],
                dense_units=best_params['dense_units'],
                dropout_rate=best_params['dropout_rate'],
                recurrent_dropout=best_params['recurrent_dropout'],
                activation=best_params['activation'],
                l2_reg=best_params['l2_reg']
            )
            
        elif model_type == 'lstm':
            model = build_lstm_model(
                input_shape=(X_train_windows.shape[1], X_train_windows.shape[2]),
                num_classes=num_classes,
                lstm_units=best_params['lstm_units'],
                num_lstm_layers=best_params['lstm_layers'],
                bidirectional=best_params['bidirectional'],
                dense_units=best_params['dense_units'],
                dropout_rate=best_params['dropout_rate'],
                recurrent_dropout=best_params['recurrent_dropout'],
                activation=best_params['activation'],
                l2_reg=best_params['l2_reg']
            )
            
        elif model_type == 'cnn_rnn':
            model = build_cnn_rnn_model(
                input_shape=(X_train_windows.shape[1], X_train_windows.shape[2]),
                num_classes=num_classes,
                conv_filters=best_params['conv_filters'],
                kernel_size=best_params['kernel_size'],
                rnn_units=best_params['rnn_units'],
                bidirectional=best_params['bidirectional'],
                use_gru=best_params['use_gru'],
                dense_units=best_params['dense_units'],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation'],
                l2_reg=best_params['l2_reg']
            )
            
        elif model_type == 'cnn_lstm':
            model = build_cnn_lstm_model(
                input_shape=(X_train_windows.shape[1], X_train_windows.shape[2]),
                num_classes=num_classes,
                conv_filters=best_params['conv_filters'],
                kernel_size=best_params['kernel_size'],
                lstm_units=best_params['lstm_units'],
                bidirectional=best_params['bidirectional'],
                dense_units=best_params['dense_units'],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation'],
                l2_reg=best_params['l2_reg']
            )
            
        elif model_type == 'siao':
            model = build_siao_cnn_ornn_model(
                input_shape=(X_train_windows.shape[1], X_train_windows.shape[2]),
                num_classes=num_classes,
                conv_filters=best_params['conv_filters'],
                kernel_size=best_params['kernel_size'],
                rnn_units=best_params['rnn_units'],
                attention_units=best_params['attention_units'],
                dense_units=best_params['dense_units'],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation'],
                l2_reg=best_params['l2_reg']
            )
        
        # Compile model
        optimizer = Adam(learning_rate=best_params['learning_rate'])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # Create directories if they don't exist
        os.makedirs('trained_models', exist_ok=True)
        os.makedirs('analysis/plots', exist_ok=True)
        os.makedirs('analysis/text', exist_ok=True)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            # Save model checkpoint
            ModelCheckpoint(f'trained_models/best_{model_type}_checkpoint.h5', 
                           monitor='val_accuracy', 
                           save_best_only=True, 
                           mode='max'),
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train_windows, y_train_windows,
            validation_data=(X_test_windows, y_test_windows),
            epochs=50,  # Train for longer with early stopping
            batch_size=best_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model.save(f'trained_models/optimized_{model_type}_model')
        print(f"Optimized model saved to trained_models/optimized_{model_type}_model")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'analysis/plots/optimized_{model_type}_history.png')
        plt.close()
        
        print(f"Training history plot saved to analysis/plots/optimized_{model_type}_history.png")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Evaluate model on test data
    print("\n5. Evaluating model on test data...")
    try:
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test_windows, y_test_windows, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Predict on test data
        y_pred = model.predict(X_test_windows)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_windows, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(encoder.classes_))
        plt.xticks(tick_marks, encoder.classes_, rotation=90)
        plt.yticks(tick_marks, encoder.classes_)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'analysis/plots/{model_type}_confusion_matrix.png')
        plt.close()
        
        print(f"Confusion matrix saved to analysis/plots/{model_type}_confusion_matrix.png")
        
        # Show final model performance
        print(f"\nOptimized {model_type.upper()} model performance:")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Save evaluation results
        with open(f'analysis/text/{model_type}_evaluation_results.txt', 'w') as f:
            f.write(f"Optimized {model_type.upper()} Model Evaluation Results\n")
            f.write(f"========================================\n\n")
            f.write(f"Test accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test loss: {test_loss:.4f}\n\n")
            f.write(f"Classification Report:\n")
            f.write(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))
            f.write(f"\nBest hyperparameters:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
        
        print(f"Evaluation results saved to analysis/text/{model_type}_evaluation_results.txt")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nOptimized {model_type.upper()} model training and evaluation completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply tuned hyperparameters to NPP fault monitoring models')
    parser.add_argument('--model', type=str, default='cnn_lstm',
                        choices=['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao'],
                        help='Model type to apply tuned hyperparameters to')
    args = parser.parse_args()
    
    main(model_type=args.model)
