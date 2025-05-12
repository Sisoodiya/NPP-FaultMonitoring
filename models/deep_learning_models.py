import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, LSTM, GRU, BatchNormalization, Activation, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import warnings

def build_cnn_lstm_model(input_shape, num_classes, lstm_units=64, dropout_rate=0.3):
    """
    Build a hybrid CNN-LSTM model for fault detection.
    
    Args:
        input_shape (tuple): Shape of input data (time_steps, features)
        num_classes (int): Number of fault classes
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled CNN-LSTM model
    """
    # Reshape input for CNN
    input_layer = Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = Reshape((input_shape[0], input_shape[1], 1))(input_layer)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Reshape for LSTM
    x = Reshape((-1, x.shape[3] * x.shape[2]))(x)
    
    # LSTM layers for temporal analysis
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_dcnn_model(input_shape, num_classes, dropout_rate=0.3):
    """
    Build a Deep Convolutional Neural Network for fault detection.
    
    Args:
        input_shape (tuple): Shape of input data (time_steps, features)
        num_classes (int): Number of fault classes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled DCNN model
    """
    # Reshape input for 2D CNN
    input_layer = Input(shape=input_shape)
    x = Reshape((input_shape[0], input_shape[1], 1))(input_layer)
    
    # First convolutional block
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Second convolutional block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Third convolutional block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_lstm_severity_model(input_shape, dropout_rate=0.3):
    """
    Build an LSTM model for fault severity estimation.
    
    Args:
        input_shape (tuple): Shape of input data (time_steps, features)
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled LSTM regression model
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(32))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Single output for severity percentage
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_deep_learning_model(X, y, model_type='cnn_lstm', validation_split=0.2, epochs=50, batch_size=32):
    """
    Train a deep learning model for fault detection.
    
    Args:
        X (np.array): Input features with shape (samples, time_steps, features)
        y (np.array): Target labels (one-hot encoded)
        model_type (str): Type of model to build ('cnn_lstm', 'dcnn', 'lstm_severity')
        validation_split (float): Fraction of data to use for validation
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (trained_model, history)
    """
    # Check input shape
    if len(X.shape) != 3:
        raise ValueError(f"Expected 3D input (samples, time_steps, features), got shape {X.shape}")
    
    # Get input shape and number of classes
    input_shape = (X.shape[1], X.shape[2])
    
    if model_type in ['cnn_lstm', 'dcnn']:
        num_classes = y.shape[1] if len(y.shape) > 1 else len(np.unique(y))
        
        # Build model based on type
        if model_type == 'cnn_lstm':
            model = build_cnn_lstm_model(input_shape, num_classes)
        elif model_type == 'dcnn':
            model = build_dcnn_model(input_shape, num_classes)
    
    elif model_type == 'lstm_severity':
        # For severity estimation (regression)
        model = build_lstm_severity_model(input_shape)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=f'models/{model_type}_best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X, y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def prepare_sliding_window_data(data, window_size, step_size=1):
    """
    Prepare data using sliding window approach for deep learning models.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Size of the sliding window
        step_size (int): Step size for the sliding window
        
    Returns:
        np.array: Windowed data with shape (n_windows, window_size, n_features)
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // step_size + 1
    
    windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = data.iloc[start_idx:end_idx].values
    
    return windows

def plot_training_history(history, model_type):
    """
    Plot training history for a deep learning model.
    
    Args:
        history: Training history object
        model_type (str): Type of model
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_type} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_type} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'models/{model_type}_training_history.png')
    plt.close()

def evaluate_deep_learning_model(model, X_test, y_test):
    """
    Evaluate a deep learning model.
    
    Args:
        model: Trained deep learning model
        X_test (np.array): Test features
        y_test (np.array): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Evaluate model
    evaluation = model.evaluate(X_test, y_test, verbose=0)
    
    # Get metrics
    metrics = {
        'loss': evaluation[0],
        'accuracy': evaluation[1]
    }
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # For classification models
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_true = y_test
        y_pred_classes = np.round(y_pred).astype(int)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred_classes)
    report = classification_report(y_true, y_pred_classes)
    
    return metrics, cm, report
