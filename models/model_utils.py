"""
Utility functions for model training, evaluation, and ensemble creation.
This file contains the essential utility functions extracted from model.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from imblearn.over_sampling import SMOTE

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning

# Cache for storing previously built models with the same parameters
_model_cache = {}

def build_model(model_type='rf', use_cache=True, **kwargs):
    """
    Build a machine learning model for fault detection.
    
    Args:
        model_type (str): Type of model to build ('rf', 'gb', 'mlp', 'svm')
        use_cache (bool): Whether to use cached models for faster initialization
        **kwargs: Additional arguments for the specific model
        
    Returns:
        sklearn model: Initialized model
    """
    # Create a cache key from model_type and sorted kwargs
    if use_cache:
        cache_key = (model_type, tuple(sorted(kwargs.items())))
        if cache_key in _model_cache:
            return _model_cache[cache_key]
    
    # Default parameters for each model type
    default_params = {
        'rf': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'n_jobs': -1,
            'random_state': 42
        },
        'gb': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        },
        'mlp': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': 42,
            'early_stopping': True,
            'n_iter_no_change': 10
        },
        'svm': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
    }
    
    # Merge default parameters with provided kwargs
    if model_type in default_params:
        params = default_params[model_type].copy()
        params.update(kwargs)
    else:
        params = kwargs
    
    # Build the model based on type
    if model_type == 'rf':
        model = RandomForestClassifier(**params)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(**params)
    elif model_type == 'mlp':
        model = MLPClassifier(**params)
    elif model_type == 'svm':
        model = SVC(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cache the model if caching is enabled
    if use_cache:
        _model_cache[cache_key] = model
    
    return model


def train_model(X, y, model_type='rf', test_size=0.2, validation_split=0.0, handle_imbalance=False, **kwargs):
    """
    Train a machine learning model for fault detection.
    
    Args:
        X (np.array): Input features
        y (np.array): Target labels (can be one-hot encoded or class indices)
        model_type (str): Type of model to build ('rf', 'gb', 'mlp', 'svm')
        test_size (float): Fraction of data to use for testing
        validation_split (float): Fraction of training data to use for validation
        handle_imbalance (bool): Whether to handle class imbalance using SMOTE
        **kwargs: Additional arguments for the specific model
        
    Returns:
        tuple: (trained_model, X_train, X_test, y_train, y_test)
    """
    try:
        # Check for NaN values in input features
        if isinstance(X, pd.DataFrame):
            nan_count = X.isna().sum().sum()
            if nan_count > 0:
                warnings.warn(f"Input features contain {nan_count} NaN values. These will be imputed.")
                # Impute NaN values
                imputer = SimpleImputer(strategy='mean')
                X = pd.DataFrame(
                    imputer.fit_transform(X),
                    columns=X.columns
                )
        
        # Convert one-hot encoded y to class indices if needed
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Handle class imbalance if requested
        if handle_imbalance:
            X_train, y_train = balance_data_with_smote(X_train, y_train)
            print(f"Applied SMOTE oversampling. New training set shape: {X_train.shape}")
        
        # Further split training data into training and validation if requested
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
            )
        
        # Build and train the model
        model = build_model(model_type=model_type, **kwargs)
        
        # Suppress convergence warnings for MLPClassifier
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        
        # Print model accuracy on training and test sets
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"Model: {model_type.upper()}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        return model, X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error training model: {e}")
        raise


def optimize_hyperparameters(X, y, model_type='rf', param_grid=None, cv=5, n_jobs=-1, scoring='accuracy', verbose=1):
    """
    Optimize hyperparameters using GridSearchCV with parallel processing.
    
    Args:
        X (np.array): Input features
        y (np.array): Target labels
        model_type (str): Type of model to optimize
        param_grid (dict): Parameter grid to search
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs (-1 for all processors)
        scoring (str): Scoring metric
        verbose (int): Verbosity level
        
    Returns:
        tuple: (optimized_model, best_params)
    """
    # Preprocess data
    X, y = _preprocess_data(X, y)
    
    # Default parameter grids for each model type
    default_param_grids = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gb': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        },
        'mlp': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
    }
    
    # Use default parameter grid if none provided
    if param_grid is None:
        if model_type in default_param_grids:
            param_grid = default_param_grids[model_type]
        else:
            raise ValueError(f"No default parameter grid for model type: {model_type}")
    
    # Build base model
    model = build_model(model_type=model_type, use_cache=False)
    
    # Create grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, n_jobs=n_jobs,
        scoring=scoring, verbose=verbose
    )
    
    # Suppress convergence warnings for MLPClassifier
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        grid_search.fit(X, y)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best {scoring}: {best_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Build optimized model with best parameters
    optimized_model = build_model(model_type=model_type, use_cache=False, **best_params)
    optimized_model.fit(X, y)
    
    return optimized_model, best_params


def _preprocess_data(X, y):
    """Helper function to preprocess data for hyperparameter optimization."""
    # Check for NaN values in input features
    if isinstance(X, pd.DataFrame):
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            warnings.warn(f"Input features contain {nan_count} NaN values. These will be imputed.")
            # Impute NaN values
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns
            )
    
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    # Convert pandas DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Convert pandas Series to numpy array if needed
    if isinstance(y, pd.Series):
        y = y.values
    
    return X, y


def calculate_metrics(cm):
    """
    Calculate various performance metrics from a confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix
        
    Returns:
        dict: Dictionary of metrics
    """
    # Ensure confusion matrix is numpy array
    cm = np.array(cm)
    
    # Extract true positives, false positives, false negatives, true negatives
    if cm.shape == (2, 2):  # Binary classification
        tn, fp, fn, tp = cm.ravel()
    else:  # Multi-class classification
        # Calculate metrics for each class and average
        n_classes = cm.shape[0]
        tp = np.zeros(n_classes)
        fp = np.zeros(n_classes)
        fn = np.zeros(n_classes)
        tn = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp[i] = cm[i, i]
            fp[i] = np.sum(cm[:, i]) - cm[i, i]
            fn[i] = np.sum(cm[i, :]) - cm[i, i]
            tn[i] = np.sum(cm) - (tp[i] + fp[i] + fn[i])
        
        # Average metrics across all classes
        tp = np.mean(tp)
        fp = np.mean(fp)
        fn = np.mean(fn)
        tn = np.mean(tn)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity
    }


def evaluate_model(model, X, y, encoder=None):
    """
    Evaluate model performance on the given data.
    
    Args:
        model: Trained scikit-learn model
        X (np.array): Input features
        y (np.array): Target labels (can be one-hot encoded or class indices)
        encoder (LabelEncoder, optional): Label encoder for class names
        
    Returns:
        tuple: (metrics_dict, classification_report_str, confusion_matrix)
    """
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Calculate metrics
    metrics = calculate_metrics(cm)
    
    # Generate classification report
    if encoder is not None:
        target_names = encoder.classes_
        report = classification_report(y, y_pred, target_names=target_names)
    else:
        report = classification_report(y, y_pred)
    
    return metrics, report, cm


def cross_validate_models(X, y, model_types=None, n_splits=5):
    """
    Perform k-fold cross-validation on multiple model types.
    
    Args:
        X (np.array): Input features
        y (np.array): Target labels (can be one-hot encoded or class indices)
        model_types (list): List of model types to evaluate
        n_splits (int): Number of folds for cross-validation
        
    Returns:
        dict: Dictionary of average metrics for each model type
    """
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    # Default model types if not provided
    if model_types is None:
        model_types = ['rf', 'gb', 'mlp', 'svm']
    
    # Create KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Dictionary to store results
    results = {}
    
    # Evaluate each model type
    for model_type in model_types:
        print(f"Cross-validating {model_type.upper()} model...")
        
        # Build model
        model = build_model(model_type=model_type)
        
        # Lists to store metrics for each fold
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # Perform k-fold cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # Calculate average metrics
        results[model_type] = {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores)
        }
        
        # Print results
        print(f"  Accuracy: {results[model_type]['accuracy']:.4f}")
        print(f"  Precision: {results[model_type]['precision']:.4f}")
        print(f"  Recall: {results[model_type]['recall']:.4f}")
        print(f"  F1 Score: {results[model_type]['f1_score']:.4f}")
    
    return results


def predict_fault_condition(model, X, encoder=None):
    """
    Predict fault conditions using the trained model.
    
    Args:
        model: Trained model
        X (np.array): Input features
        encoder (LabelEncoder, optional): Label encoder for class names
        
    Returns:
        np.array: Predicted conditions
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Convert predictions to class names if encoder is provided
    if encoder is not None:
        predictions = encoder.inverse_transform(predictions)
    
    return predictions


def balance_data_with_smote(X, y):
    """
    Balance the dataset using SMOTE oversampling.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    try:
        # Initialize SMOTE
        smote = SMOTE(random_state=42)
        
        # Apply SMOTE
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Print class distribution before and after
        unique_before, counts_before = np.unique(y, return_counts=True)
        unique_after, counts_after = np.unique(y_resampled, return_counts=True)
        
        print("Class distribution before SMOTE:")
        for cls, count in zip(unique_before, counts_before):
            print(f"  Class {cls}: {count}")
        
        print("Class distribution after SMOTE:")
        for cls, count in zip(unique_after, counts_after):
            print(f"  Class {cls}: {count}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"Error applying SMOTE: {e}")
        return X, y


def ensemble_models(X, y, model_types=None, weights=None):
    """
    Create an ensemble of different model types.
    
    Args:
        X (np.array): Input features
        y (np.array): Target labels
        model_types (list): List of model types to include in ensemble
        weights (list): Weights for each model in the ensemble
        
    Returns:
        tuple: (trained_models, X_train, X_test, y_train, y_test)
    """
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    # Reshape X if needed (for 3D data)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    # Default model types if not provided
    if model_types is None:
        model_types = ['rf', 'gb', 'mlp']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = {}
    for model_type in model_types:
        print(f"Training {model_type.upper()} model for ensemble...")
        model = build_model(model_type=model_type)
        model.fit(X_train, y_train)
        models[model_type] = model
    
    # Evaluate individual models
    for model_type, model in models.items():
        accuracy = model.score(X_test, y_test)
        print(f"{model_type.upper()} accuracy: {accuracy:.4f}")
    
    # Evaluate ensemble (majority voting)
    def ensemble_predict(X, models, weights=None):
        predictions = np.array([model.predict(X) for model in models.values()])
        if weights is not None:
            # Weighted voting
            weighted_preds = np.zeros_like(predictions[0], dtype=float)
            for i, weight in enumerate(weights):
                weighted_preds += weight * predictions[i]
            return np.round(weighted_preds).astype(int)
        else:
            # Majority voting
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
    
    # Evaluate ensemble
    ensemble_preds = ensemble_predict(X_test, models, weights)
    ensemble_accuracy = np.mean(ensemble_preds == y_test)
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    return models, X_train, X_test, y_train, y_test


def save_model_and_preprocessors(model, scaler, encoder, base_path='models'):
    """
    Save the trained model and preprocessors for later use.
    
    Args:
        model: Trained scikit-learn model
        scaler (StandardScaler): Fitted scaler
        encoder (LabelEncoder): Fitted encoder
        base_path (str): Directory to save files
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(base_path, 'model.joblib'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(base_path, 'scaler.joblib'))
    
    # Save encoder
    joblib.dump(encoder, os.path.join(base_path, 'encoder.joblib'))
    
    print(f"Model and preprocessors saved to {base_path}")


def load_model_and_preprocessors(base_path='models'):
    """
    Load the saved model and preprocessors.
    
    Args:
        base_path (str): Directory where files are saved
        
    Returns:
        tuple: (model, scaler, encoder)
    """
    # Load model
    model = joblib.load(os.path.join(base_path, 'model.joblib'))
    
    # Load scaler
    scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
    
    # Load encoder
    encoder = joblib.load(os.path.join(base_path, 'encoder.joblib'))
    
    return model, scaler, encoder
