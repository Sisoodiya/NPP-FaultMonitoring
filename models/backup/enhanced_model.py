"""
Enhanced Model for NPP Fault Monitoring

This script implements three key enhancements to the CNN-LSTM model:
1. Address Class Imbalance: Using SMOTE and class weights
2. Feature Importance Analysis: Identifying the most important features
3. Ensemble Methods: Combining CNN-LSTM with other models for better performance
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
import warnings

# Import for addressing class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imblearn not available. Install with 'pip install imbalanced-learn' for SMOTE.")

# Define a fallback for to_categorical function in case TensorFlow is not available
def fallback_to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    This is a fallback implementation when TensorFlow is not available.
    
    Args:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes. If None, this is inferred from y.
    
    Returns:
        A binary matrix representation of the input.
    """
    import numpy as np
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

# Import TensorFlow and Keras - with more robust error handling
try:
    # First try direct import
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow successfully imported, version:", tf.__version__)
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    try:
        # Try importing just keras as a fallback
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
        from keras.callbacks import EarlyStopping
        from keras.optimizers import Adam
        from keras.utils import to_categorical
        TENSORFLOW_AVAILABLE = True
        print("Keras successfully imported as fallback, version:", keras.__version__)
    except ImportError as e2:
        print(f"Keras import error: {e2}")
        TENSORFLOW_AVAILABLE = False
        # Use our fallback implementation of to_categorical
        to_categorical = fallback_to_categorical
        warnings.warn("Neither TensorFlow nor Keras available. Will focus on traditional ML models.")
except Exception as e:
    print(f"Unexpected error importing TensorFlow: {e}")
    TENSORFLOW_AVAILABLE = False
    # Use our fallback implementation of to_categorical
    to_categorical = fallback_to_categorical
    warnings.warn("Error loading TensorFlow. Will focus on traditional ML models.")

# Import custom modules
from data_preprocessing import process_pipeline
import dlm

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def create_cnn_lstm_model(input_shape, num_classes, params=None):
    """
    Create a CNN-LSTM model with specified hyperparameters.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        num_classes: Number of output classes
        params: Dictionary of hyperparameters
        
    Returns:
        Compiled Keras model
    """
    if params is None:
        # Default parameters if none provided
        params = {
            'conv_filters': 64,
            'conv_kernel_size': 3,
            'lstm_units': 128,
            'dense_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'activation': 'relu'
        }
    
    model = Sequential()
    
    # CNN layers
    model.add(Conv1D(filters=params['conv_filters'],
                    kernel_size=(params['conv_kernel_size'],),
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
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    
    return model

def analyze_feature_importance(X, y, feature_names):
    """
    Analyze feature importance using Random Forest and return top features.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
    """
    print("Analyzing feature importance...")
    
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importances = rf.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()
    
    print(f"Top 20 important features saved to models/feature_importance.png")
    
    # Save feature importance to file
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    print(f"Feature importance saved to models/feature_importance.csv")
    
    return feature_importance

def create_ensemble_model(X_train, y_train, X_val, y_val, input_shape=None, num_classes=None, best_params=None):
    """
    Create an ensemble model combining CNN-LSTM with traditional ML models.
    If TensorFlow is not available, only traditional ML models will be used.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        input_shape: Shape for CNN-LSTM input (optional if TensorFlow not available)
        num_classes: Number of classes (optional if TensorFlow not available)
        best_params: Best hyperparameters for CNN-LSTM (optional if TensorFlow not available)
        
    Returns:
        Dictionary with trained models
    """
    print("Creating ensemble model...")
    
    # Flatten the 3D data for traditional ML models if it's not already flat
    if len(X_train.shape) > 2:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
    else:
        X_train_flat = X_train
        X_val_flat = X_val
    
    # Convert one-hot encoded labels to class indices for traditional ML models if needed
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:  # One-hot encoded
        y_train_class = np.argmax(y_train, axis=1)
        y_val_class = np.argmax(y_val, axis=1)
    else:  # Already class indices
        y_train_class = y_train
        y_val_class = y_val
    
    # Initialize results dictionary
    ensemble_models = {}
    
    # 1. Train CNN-LSTM model if TensorFlow is available
    if TENSORFLOW_AVAILABLE and input_shape is not None and num_classes is not None and best_params is not None:
        try:
            print("Training CNN-LSTM model...")
            cnn_lstm = create_cnn_lstm_model(input_shape, num_classes, best_params)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            cnn_lstm_history = cnn_lstm.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=best_params['batch_size'],
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate CNN-LSTM model
            cnn_lstm_val_loss, cnn_lstm_val_acc = cnn_lstm.evaluate(X_val, y_val, verbose=0)
            print(f"CNN-LSTM validation accuracy: {cnn_lstm_val_acc:.4f}")
            
            # Save CNN-LSTM model
            cnn_lstm.save('models/ensemble_cnn_lstm')
            
            # Add to ensemble models dictionary
            ensemble_models['cnn_lstm'] = cnn_lstm
            ensemble_models['cnn_lstm_history'] = cnn_lstm_history
        except Exception as e:
            print(f"Error training CNN-LSTM model: {e}")
            print("Continuing with traditional ML models only.")
    else:
        print("Skipping CNN-LSTM model (TensorFlow not available or parameters missing).")
    
    # 2. Train Random Forest
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_flat, y_train_class)
    
    # 3. Train Gradient Boosting
    print("Training Gradient Boosting model...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_flat, y_train_class)
    
    # 4. Train Extra Trees (additional model for better ensemble)
    print("Training Extra Trees model...")
    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    et.fit(X_train_flat, y_train_class)
    
    # Evaluate traditional ML models
    rf_val_acc = rf.score(X_val_flat, y_val_class)
    gb_val_acc = gb.score(X_val_flat, y_val_class)
    et_val_acc = et.score(X_val_flat, y_val_class)
    
    print(f"Random Forest validation accuracy: {rf_val_acc:.4f}")
    print(f"Gradient Boosting validation accuracy: {gb_val_acc:.4f}")
    print(f"Extra Trees validation accuracy: {et_val_acc:.4f}")
    
    # Save traditional ML models
    joblib.dump(rf, 'models/ensemble_random_forest.pkl')
    joblib.dump(gb, 'models/ensemble_gradient_boosting.pkl')
    joblib.dump(et, 'models/ensemble_extra_trees.pkl')
    
    # Add traditional ML models to ensemble dictionary
    ensemble_models['random_forest'] = rf
    ensemble_models['gradient_boosting'] = gb
    ensemble_models['extra_trees'] = et
    
    return ensemble_models

def ensemble_predict(models, X_test, X_test_flat):
    """
    Make predictions using the ensemble model with weighted voting.
    Works with or without CNN-LSTM model.
    
    Args:
        models: Dictionary of trained models
        X_test: Test data for CNN-LSTM (3D) if available
        X_test_flat: Flattened test data for traditional ML models
        
    Returns:
        Ensemble predictions and probabilities
    """
    # Initialize weights for each model type
    weights = {}
    pred_probas = {}
    
    # Get predictions from CNN-LSTM if available
    if 'cnn_lstm' in models:
        try:
            cnn_lstm_pred_proba = models['cnn_lstm'].predict(X_test)
            pred_probas['cnn_lstm'] = cnn_lstm_pred_proba
            weights['cnn_lstm'] = 0.5  # Higher weight for deep learning model
        except Exception as e:
            print(f"Error getting CNN-LSTM predictions: {e}")
    
    # Get predictions from Random Forest
    rf_pred_proba = models['random_forest'].predict_proba(X_test_flat)
    pred_probas['random_forest'] = rf_pred_proba
    weights['random_forest'] = 0.3 if 'cnn_lstm' in weights else 0.5
    
    # Get predictions from Gradient Boosting
    gb_pred_proba = models['gradient_boosting'].predict_proba(X_test_flat)
    pred_probas['gradient_boosting'] = gb_pred_proba
    weights['gradient_boosting'] = 0.2 if 'cnn_lstm' in weights else 0.3
    
    # Get predictions from Extra Trees if available
    if 'extra_trees' in models:
        et_pred_proba = models['extra_trees'].predict_proba(X_test_flat)
        pred_probas['extra_trees'] = et_pred_proba
        weights['extra_trees'] = 0.1 if 'cnn_lstm' in weights else 0.2
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        for key in weights:
            weights[key] /= weight_sum
    
    # Weighted ensemble
    ensemble_pred_proba = np.zeros_like(list(pred_probas.values())[0])
    for model_name, pred_proba in pred_probas.items():
        ensemble_pred_proba += weights[model_name] * pred_proba
    
    ensemble_pred_classes = np.argmax(ensemble_pred_proba, axis=1)
    
    return ensemble_pred_classes, ensemble_pred_proba

def main():
    """
    Main function to run the enhanced model pipeline.
    """
    print("NPP Fault Monitoring - Enhanced Model Pipeline")
    print("=" * 50)
    
    # Continue even if TensorFlow is not available
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not available. Will focus on traditional ML models and feature importance.")
        # We'll still proceed with the pipeline but skip the CNN-LSTM parts
    
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
        
        # Class distribution before balancing
        class_counts = pd.Series(y_encoded).value_counts().sort_index()
        print("\nClass distribution before balancing:")
        for i, count in enumerate(class_counts):
            print(f"  {encoder.classes_[i]}: {count} samples")
        
        # Save the encoder for future use
        joblib.dump(encoder, 'models/enhanced_label_encoder.pkl')
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save the scaler for future use
        joblib.dump(scaler, 'models/enhanced_feature_scaler.pkl')
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Feature Importance Analysis
    print("\n3. Performing feature importance analysis...")
    try:
        # Analyze feature importance
        feature_importance = analyze_feature_importance(X_scaled, y_encoded, feature_cols)
        
        # Select top features (e.g., top 80% of cumulative importance)
        cumulative_importance = 0
        top_features_idx = []
        
        for i, row in feature_importance.iterrows():
            cumulative_importance += row['Importance']
            top_features_idx.append(feature_cols.index(row['Feature']))
            if cumulative_importance >= 0.8:  # Select features that contribute to 80% importance
                break
        
        top_feature_names = feature_importance.iloc[:len(top_features_idx)]['Feature'].tolist()
        print(f"Selected {len(top_features_idx)} top features out of {len(feature_cols)}")
        print(f"Top 10 features: {top_feature_names[:10]}")
        
        # Use selected features
        X_selected = X_scaled[:, top_features_idx]
        
    except Exception as e:
        print(f"Error during feature importance analysis: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to using all features
        X_selected = X_scaled
        top_feature_names = feature_cols
    
    # 4. Address Class Imbalance
    print("\n4. Addressing class imbalance...")
    try:
        if IMBLEARN_AVAILABLE:
            # Apply SMOTE to balance classes
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_selected, y_encoded)
            
            # Class distribution after balancing
            balanced_counts = pd.Series(y_balanced).value_counts().sort_index()
            print("\nClass distribution after SMOTE balancing:")
            for i, count in enumerate(balanced_counts):
                print(f"  {encoder.classes_[i]}: {count} samples")
        else:
            print("imblearn not available. Proceeding without SMOTE.")
            X_balanced = X_selected
            y_balanced = y_encoded
        
        # One-hot encode labels for deep learning
        y_onehot = to_categorical(y_balanced)
        
        # Split data into training, validation, and testing sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_balanced, y_onehot, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        # Reshape data for CNN-LSTM (samples, timesteps, features)
        window_size = 10  # Number of timesteps in each window
        
        # Function to create sliding windows
        def create_windows(data, window_size):
            windows = []
            for i in range(len(data) - window_size + 1):
                windows.append(data[i:i+window_size])
            return np.array(windows)
        
        # Create sliding windows for training, validation, and test sets
        X_train_windows = create_windows(X_train, window_size)
        X_val_windows = create_windows(X_val, window_size)
        X_test_windows = create_windows(X_test, window_size)
        
        # For labels, we'll use the label of the last timestep in each window
        y_train_windows = y_train[window_size-1:]
        y_val_windows = y_val[window_size-1:]
        y_test_windows = y_test[window_size-1:]
        
        print(f"Training data shape: {X_train_windows.shape}")
        print(f"Validation data shape: {X_val_windows.shape}")
        print(f"Test data shape: {X_test_windows.shape}")
        
        # Flatten data for traditional ML models
        X_train_flat = X_train_windows.reshape(X_train_windows.shape[0], -1)
        X_val_flat = X_val_windows.reshape(X_val_windows.shape[0], -1)
        X_test_flat = X_test_windows.reshape(X_test_windows.shape[0], -1)
        
    except Exception as e:
        print(f"Error during class balancing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Load best hyperparameters
    print("\n5. Loading best hyperparameters...")
    try:
        best_params = joblib.load('models/best_hyperparameters.pkl')
        print(f"Best hyperparameters: {best_params}")
        
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        print("Using default hyperparameters.")
        best_params = {
            'conv_filters': 64,
            'conv_kernel_size': 3,
            'lstm_units': 128,
            'dense_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'activation': 'relu'
        }
    
    # 6. Create and train ensemble model
    print("\n6. Creating and training ensemble model...")
    try:
        # Prepare parameters for ensemble model
        if TENSORFLOW_AVAILABLE:
            input_shape = X_train_windows.shape[1:]
            num_classes = y_train_windows.shape[1]
        else:
            input_shape = None
            num_classes = None
        
        # Create and train ensemble model
        ensemble_models = create_ensemble_model(
            X_train_windows, y_train_windows,
            X_val_windows, y_val_windows,
            input_shape, num_classes, best_params
        )
        
        # Plot training history for CNN-LSTM if available
        if 'cnn_lstm_history' in ensemble_models:
            history = ensemble_models['cnn_lstm_history']
            plt.figure(figsize=(12, 5))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('CNN-LSTM Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('CNN-LSTM Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('models/ensemble_training_history.png')
            plt.close()
            
            print("CNN-LSTM training history saved to models/ensemble_training_history.png")
        
    except Exception as e:
        print(f"Error during ensemble model creation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. Evaluate ensemble model
    print("\n7. Evaluating ensemble model...")
    try:
        # Get predictions from individual models
        rf = ensemble_models['random_forest']
        gb = ensemble_models['gradient_boosting']
        et = ensemble_models['extra_trees']
        
        # Make predictions for traditional ML models
        rf_pred_classes = rf.predict(X_test_flat)
        gb_pred_classes = gb.predict(X_test_flat)
        et_pred_classes = et.predict(X_test_flat)
        
        # True classes - handle both one-hot encoded and class indices
        if len(y_test_windows.shape) > 1 and y_test_windows.shape[1] > 1:  # One-hot encoded
            y_true_classes = np.argmax(y_test_windows, axis=1)
        else:  # Already class indices
            y_true_classes = y_test_windows
        
        # Initialize results dictionary
        model_results = {}
        
        # Add CNN-LSTM results if available
        if 'cnn_lstm' in ensemble_models:
            cnn_lstm = ensemble_models['cnn_lstm']
            cnn_lstm_pred = cnn_lstm.predict(X_test_windows)
            cnn_lstm_pred_classes = np.argmax(cnn_lstm_pred, axis=1)
            cnn_lstm_acc = accuracy_score(y_true_classes, cnn_lstm_pred_classes)
            cnn_lstm_f1 = f1_score(y_true_classes, cnn_lstm_pred_classes, average='weighted')
            model_results['CNN-LSTM'] = {'accuracy': cnn_lstm_acc, 'f1': cnn_lstm_f1}
        
        # Calculate metrics for traditional ML models
        rf_acc = accuracy_score(y_true_classes, rf_pred_classes)
        rf_f1 = f1_score(y_true_classes, rf_pred_classes, average='weighted')
        model_results['Random Forest'] = {'accuracy': rf_acc, 'f1': rf_f1}
        
        gb_acc = accuracy_score(y_true_classes, gb_pred_classes)
        gb_f1 = f1_score(y_true_classes, gb_pred_classes, average='weighted')
        model_results['Gradient Boosting'] = {'accuracy': gb_acc, 'f1': gb_f1}
        
        et_acc = accuracy_score(y_true_classes, et_pred_classes)
        et_f1 = f1_score(y_true_classes, et_pred_classes, average='weighted')
        model_results['Extra Trees'] = {'accuracy': et_acc, 'f1': et_f1}
        
        # Get ensemble predictions
        ensemble_pred_classes, _ = ensemble_predict(
            ensemble_models, X_test_windows, X_test_flat
        )
        
        # Calculate ensemble metrics
        ensemble_acc = accuracy_score(y_true_classes, ensemble_pred_classes)
        ensemble_f1 = f1_score(y_true_classes, ensemble_pred_classes, average='weighted')
        model_results['Ensemble'] = {'accuracy': ensemble_acc, 'f1': ensemble_f1}
        
        # Print model comparison
        print("\nModel Comparison:")
        for model_name, metrics in model_results.items():
            print(f"{model_name} Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1']:.4f}")

        
        # Classification report for ensemble
        print("\nEnsemble Classification Report:")
        print(classification_report(y_true_classes, ensemble_pred_classes, 
                                   target_names=encoder.classes_))
        
        # Confusion matrix for ensemble
        cm = confusion_matrix(y_true_classes, ensemble_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Ensemble Model Confusion Matrix')
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
        plt.savefig('models/ensemble_confusion_matrix.png')
        plt.close()
        
        print("Ensemble confusion matrix saved to models/ensemble_confusion_matrix.png")
        
        # Save evaluation results
        with open('models/ensemble_evaluation_results.txt', 'w') as f:
            f.write(f"Enhanced NPP Fault Monitoring Model Evaluation\n")
            f.write(f"==========================================\n\n")
            f.write(f"Model Comparison:\n")
            f.write(f"CNN-LSTM Accuracy: {cnn_lstm_acc:.4f}, F1 Score: {cnn_lstm_f1:.4f}\n")
            f.write(f"Random Forest Accuracy: {rf_acc:.4f}, F1 Score: {rf_f1:.4f}\n")
            f.write(f"Gradient Boosting Accuracy: {gb_acc:.4f}, F1 Score: {gb_f1:.4f}\n")
            f.write(f"Ensemble Accuracy: {ensemble_acc:.4f}, F1 Score: {ensemble_f1:.4f}\n\n")
            f.write(f"Ensemble Classification Report:\n")
            f.write(classification_report(y_true_classes, ensemble_pred_classes, 
                                         target_names=encoder.classes_))
            f.write(f"\nTop 20 Important Features:\n")
            for i, row in feature_importance.head(20).iterrows():
                f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
        
        print("Ensemble evaluation results saved to models/ensemble_evaluation_results.txt")
        
    except Exception as e:
        print(f"Error during ensemble model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nEnhanced model pipeline completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
