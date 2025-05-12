from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

def build_rnn(input_shape, num_classes, dropout_rate=0.3, recurrent_dropout=0.2, l2_reg=0.001, learning_rate=0.001, use_gru=True):
    """
    Build an optimized RNN model with GRU cells, bidirectional layers, dropout, and regularization.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        recurrent_dropout: Dropout rate for recurrent connections
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for Adam optimizer
        use_gru: Whether to use GRU cells (True) or SimpleRNN cells (False)
        
    Returns:
        Compiled Keras RNN model
    """
    model = models.Sequential()
    
    # Choose between GRU (better performance) or SimpleRNN
    rnn_layer = layers.GRU if use_gru else layers.SimpleRNN
    
    # First bidirectional RNN layer
    model.add(layers.Bidirectional(
        rnn_layer(64, return_sequences=True, 
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2),
                kernel_initializer='glorot_uniform'),
        input_shape=input_shape
    ))
    model.add(layers.BatchNormalization())
    
    # Second bidirectional RNN layer
    model.add(layers.Bidirectional(
        rnn_layer(64, 
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg/2),
                kernel_initializer='glorot_uniform')
    ))
    model.add(layers.BatchNormalization())
    
    # Fully connected layers
    model.add(layers.Dense(128, kernel_initializer='he_uniform',
                        kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model with Adam optimizer and learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model


def train_rnn(X_train, y_train, X_val=None, y_val=None, input_shape=None, num_classes=None, 
             epochs=50, batch_size=32, early_stopping=True, class_weights=None,
             dropout_rate=0.3, recurrent_dropout=0.2, l2_reg=0.001, learning_rate=0.001, use_gru=True):
    """
    Train the RNN model with advanced training techniques.
    
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
        use_gru: Whether to use GRU cells (True) or SimpleRNN cells (False)
        
    Returns:
        Trained model and training history
    """
    # Infer input shape and number of classes if not provided
    if input_shape is None and X_train is not None:
        input_shape = X_train.shape[1:]
    
    if num_classes is None and y_train is not None:
        if len(y_train.shape) > 1:
            num_classes = y_train.shape[1]  # One-hot encoded
        else:
            num_classes = len(np.unique(y_train))  # Class indices
    
    # Build model with specified hyperparameters
    model = build_rnn(input_shape, num_classes, dropout_rate, recurrent_dropout, l2_reg, learning_rate, use_gru)
    
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
        'models/checkpoints/rnn_best.h5',
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
