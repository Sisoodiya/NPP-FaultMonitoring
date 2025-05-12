from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

def build_lstm(input_shape, num_classes, dropout_rate=0.3, recurrent_dropout=0.2, l2_reg=0.001, learning_rate=0.001):
    """
    Build an optimized LSTM model with bidirectional layers, dropout, and regularization.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        recurrent_dropout: Dropout rate for recurrent connections
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras LSTM model
    """
    model = models.Sequential([
        # First bidirectional LSTM layer
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, 
                      dropout=dropout_rate,
                      recurrent_dropout=recurrent_dropout,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      recurrent_regularizer=regularizers.l2(l2_reg/2),
                      kernel_initializer='glorot_uniform'),
            input_shape=input_shape
        ),
        layers.BatchNormalization(),
        
        # Second bidirectional LSTM layer
        layers.Bidirectional(
            layers.LSTM(64, 
                      dropout=dropout_rate,
                      recurrent_dropout=recurrent_dropout,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      recurrent_regularizer=regularizers.l2(l2_reg/2),
                      kernel_initializer='glorot_uniform')
        ),
        layers.BatchNormalization(),
        
        # Fully connected layers
        layers.Dense(128, kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with Adam optimizer and learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None, input_shape=None, num_classes=None, 
             epochs=50, batch_size=32, early_stopping=True, class_weights=None,
             dropout_rate=0.3, recurrent_dropout=0.2, l2_reg=0.001, learning_rate=0.001):
    """
    Train the LSTM model with advanced training techniques from model.py and enhanced_model.py.
    
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
        dropout_rate: Dropout rate for regularization
        recurrent_dropout: Dropout rate for recurrent connections
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for Adam optimizer
        
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
    
    # Build model with specified hyperparameters
    model = build_lstm(input_shape, num_classes, dropout_rate, recurrent_dropout, l2_reg, learning_rate)
    
    # Prepare callbacks
    callbacks = []
    
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        ))
    
    # Learning rate scheduler
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss' if X_val is not None else 'loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ))
    
    # Model checkpoint
    os.makedirs('models/checkpoints', exist_ok=True)
    callbacks.append(ModelCheckpoint(
        'models/checkpoints/lstm_best.h5',
        monitor='val_loss' if X_val is not None else 'loss',
        save_best_only=True
    ))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return model, history
