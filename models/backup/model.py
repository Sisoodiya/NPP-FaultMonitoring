import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from imblearn.over_sampling import SMOTE

# Use scikit-learn models instead of TensorFlow for compatibility
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning

# Import custom modules for deep learning and reliability analysis
import dlm  # Deep Learning Module
import ra   # Reliability Analysis Module


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
                # Impute NaN values using SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X = pd.DataFrame(
                    imputer.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
            # Save feature names for later use
            feature_names = X.columns.tolist()
            X_values = X.values
        elif isinstance(X, np.ndarray):
            if np.isnan(X).any():
                warnings.warn(f"Input features contain NaN values. These will be imputed.")
                # Impute NaN values using SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X_values = imputer.fit_transform(X)
            else:
                X_values = X
            feature_names = None
        else:
            X_values = X
            feature_names = None
        
        # Convert one-hot encoded y to class indices if needed
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            y_values = np.argmax(y, axis=1)
        elif isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_values, y_values, test_size=test_size, random_state=42, stratify=y_values
        )
        
        # Further split training data into training and validation if requested
        if validation_split > 0:
            val_size = validation_split / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
            )
        
        # Reshape X if needed (for 3D data from CNN)
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            if validation_split > 0:
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_test_flat = X_test
            if validation_split > 0:
                X_val_flat = X_val
        
        # Handle class imbalance if requested
        if handle_imbalance:
            try:
                # Check class distribution before SMOTE
                class_counts = np.bincount(y_train)
                print(f"Class distribution before SMOTE: {class_counts}")
                
                # Apply SMOTE to balance the training data
                smote = SMOTE(random_state=42)
                X_train_flat, y_train = smote.fit_resample(X_train_flat, y_train)
                
                # Check class distribution after SMOTE
                class_counts = np.bincount(y_train)
                print(f"Class distribution after SMOTE: {class_counts}")
                
                # If original data was 3D, reshape back to 3D
                if len(X_train.shape) > 2:
                    # This is a simplification - in practice, SMOTE on 3D data requires more care
                    # as it flattens the data and the original shape information is lost
                    warnings.warn("SMOTE applied to flattened 3D data. Original shape information may be lost.")
            except Exception as e:
                warnings.warn(f"SMOTE failed: {e}. Proceeding with original data.")
        
        # Build model
        model = build_model(model_type=model_type, **kwargs)
        
        # Train model with error handling
        print(f"Training {model_type.upper()} model...")
        try:
            # Suppress convergence warnings for iterative models like MLP
            with warnings.catch_warnings():
                if model_type == 'mlp':
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                if validation_split > 0 and hasattr(model, 'validation_fraction'):
                    # For models that support validation data
                    model.validation_fraction = validation_split
                    model.fit(X_train_flat, y_train)
                else:
                    model.fit(X_train_flat, y_train)
        except Exception as e:
            print(f"Error during model training: {e}")
            # If training fails, try with more robust parameters
            if model_type == 'gb':
                print("Trying GradientBoostingClassifier with more robust parameters...")
                model = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=3, 
                    subsample=0.8,  # Add subsample to make more robust
                    random_state=42
                )
            elif model_type == 'mlp':
                print("Trying MLPClassifier with more robust parameters...")
                model = MLPClassifier(
                    hidden_layer_sizes=(50,),  # Simpler architecture
                    max_iter=300,  # More iterations
                    alpha=0.01,  # Stronger regularization
                    random_state=42
                )
            # Try again with the more robust model
            model.fit(X_train_flat, y_train)
        
        # Store feature names in the model if available
        if feature_names is not None:
            model.feature_names_ = feature_names
        
        # Print training and testing accuracy
        train_accuracy = model.score(X_train_flat, y_train)
        test_accuracy = model.score(X_test_flat, y_test)
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        
        return model, X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error in train_model: {e}")
        import traceback
        traceback.print_exc()
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
        n_jobs (int): Number of jobs to run in parallel (-1 for all cores)
        scoring (str): Scoring metric to use
        verbose (int): Verbosity level
        
    Returns:
        tuple: (optimized_model, best_params)
    """
    try:
        # Process input data
        X_processed, y_processed = _preprocess_data(X, y)
        
        # Define default parameter grids if none provided
        if param_grid is None:
            param_grid = _get_default_param_grid(model_type)
        
        # Build the base model
        model = build_model(model_type, use_cache=False)  # Don't use cache for grid search
        
        # Determine if we should use RandomizedSearchCV instead of GridSearchCV
        # for large parameter spaces
        param_combinations = 1
        for param_values in param_grid.values():
            param_combinations *= len(param_values)
        
        use_randomized = param_combinations > 100
        
        if use_randomized:
            from sklearn.model_selection import RandomizedSearchCV
            search_cv = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=min(100, param_combinations),  # Cap at 100 iterations
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=42
            )
            print(f"Using RandomizedSearchCV with {min(100, param_combinations)} iterations due to large parameter space")
        else:
            search_cv = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True  # Return training scores for analysis
            )
        
        # Suppress convergence warnings during search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Perform grid search
            print(f"Optimizing {model_type.upper()} hyperparameters with {cv}-fold cross-validation...")
            search_cv.fit(X_processed, y_processed)
        
        # Get the best model and parameters
        best_model = search_cv.best_estimator_
        best_params = search_cv.best_params_
        best_score = search_cv.best_score_
        
        print(f"Best {cv}-fold CV score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Print top 3 models if available
        if hasattr(search_cv, 'cv_results_'):
            results = pd.DataFrame(search_cv.cv_results_)
            top_results = results.sort_values('rank_test_score').head(3)
            print("\nTop 3 models:")
            for i, (_, row) in enumerate(top_results.iterrows()):
                params_str = ', '.join(f"{k}={v}" for k, v in {
                    k.replace('param_', ''): v for k, v in row.items() 
                    if k.startswith('param_') and not pd.isna(v)
                }.items())
                print(f"{i+1}. Test score: {row['mean_test_score']:.4f}, Train score: {row.get('mean_train_score', 'N/A')}, Params: {params_str}")
        
        return best_model, best_params
        
    except Exception as e:
        print(f"Error in optimize_hyperparameters: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to a simple model with default parameters
        print("Falling back to default model...")
        model = build_model(model_type)
        return model, {}


def _preprocess_data(X, y):
    """Helper function to preprocess data for hyperparameter optimization."""
    # Check for NaN values in input features
    if isinstance(X, pd.DataFrame):
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            warnings.warn(f"Input features contain {nan_count} NaN values. These will be imputed.")
            # Impute NaN values using SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns
            )
        X_processed = X.values
    elif isinstance(X, np.ndarray):
        if np.isnan(X).any():
            warnings.warn(f"Input features contain NaN values. These will be imputed.")
            # Impute NaN values using SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_processed = imputer.fit_transform(X)
        else:
            X_processed = X
    else:
        X_processed = X
        # Build model with default parameters
        optimized_model = build_model(model_type=model_type, **best_params)
        
        # Try to fit the model with default parameters
        try:
            optimized_model.fit(X, y)
        except Exception as e2:
            print(f"Error fitting model with default parameters: {e2}")
            print("Trying with more robust model configuration...")
            
            # Fall back to the most robust model type (Random Forest)
            if model_type != 'rf':
                print("Switching to Random Forest as a fallback...")
                model_type = 'rf'
                best_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
                optimized_model = build_model(model_type=model_type, **best_params)
                optimized_model.fit(X, y)
        
        return optimized_model, best_params
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Build optimized model
    optimized_model = build_model(model_type=model_type, **best_params)
    
    # Train optimized model on all data with error handling
    try:
        optimized_model.fit(X, y)
    except Exception as e:
        print(f"Error training optimized model: {e}")
        print("Trying with more robust parameters...")
        
        # Modify parameters to be more robust
        if model_type == 'gb':
            best_params['subsample'] = 0.8  # Add subsample for robustness if not already present
        
        # Rebuild and train model
        optimized_model = build_model(model_type=model_type, **best_params)
        optimized_model.fit(X, y)
    
    return optimized_model, best_params


def calculate_metrics(cm):
    """
    Calculate various performance metrics from a confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix
        
    Returns:
        dict: Dictionary of metrics
    """
    # Total samples
    total = np.sum(cm)
    
    # True positives, false positives, false negatives, true negatives
    if cm.shape[0] == 2:  # Binary classification
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]
        
        # Calculate metrics
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mcc = matthews_corrcoef(np.repeat([0, 1], [tn + fp, tp + fn]), np.repeat([0, 1], [tn + fn, tp + fp]))
    else:  # Multi-class classification
        # Calculate metrics using weighted averages
        n_classes = cm.shape[0]
        class_metrics = []
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = total - (tp + fp + fn)
            
            class_acc = (tp + tn) / total
            class_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * class_prec * class_rec / (class_prec + class_rec) if (class_prec + class_rec) > 0 else 0
            
            class_metrics.append({
                'accuracy': class_acc,
                'precision': class_prec,
                'recall': class_rec,
                'f1': class_f1
            })
        
        # Weighted averages
        class_counts = np.sum(cm, axis=1)
        weights = class_counts / total
        
        accuracy = np.mean([m['accuracy'] for m in class_metrics])
        precision = np.sum([m['precision'] * w for m, w in zip(class_metrics, weights)])
        recall = np.sum([m['recall'] * w for m, w in zip(class_metrics, weights)])
        f1 = np.sum([m['f1'] * w for m, w in zip(class_metrics, weights)])
        mcc = 0  # MCC for multiclass is complex, set to 0 for now
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc
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
        y_true_classes = np.argmax(y, axis=1)
    else:
        y_true_classes = y
    
    # Reshape X if needed (for 3D data)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    # Generate predictions
    y_pred_classes = model.predict(X)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Calculate metrics
    metrics = calculate_metrics(cm)
    
    # Generate classification report
    if encoder is not None and hasattr(encoder, 'classes_'):
        # Get unique classes in the test data
        unique_classes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
        
        # Filter target names to only include classes present in the data
        target_names = [encoder.classes_[i] for i in unique_classes if i < len(encoder.classes_)]
        
        # If there's still a mismatch, use generic class names
        if len(target_names) != len(unique_classes):
            target_names = [f"Class {i}" for i in range(len(unique_classes))]
    else:
        target_names = [f"Class {i}" for i in range(len(np.unique(y_true_classes)))]
    
    try:
        report = classification_report(y_true_classes, y_pred_classes, target_names=target_names)
    except ValueError as e:
        # Fallback if there's still an issue with target_names
        print(f"Warning during classification report generation: {e}")
        report = classification_report(y_true_classes, y_pred_classes, target_names=None)
    
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
    
    # Reshape X if needed (for 3D data)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    # Default model types if not provided
    if model_types is None:
        model_types = ['rf', 'gb', 'mlp']
    
    # Dictionary to store results
    results = {}
    
    # Evaluate each model type
    for model_type in model_types:
        print(f"\nCross-validating {model_type.upper()} model...")
        
        # Build model
        model = build_model(model_type=model_type)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted'
        }
        
        # Perform cross-validation
        cv_results = {}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X, y, cv=n_splits, scoring=scorer, n_jobs=-1)
            cv_results[metric_name] = scores
            print(f"{metric_name}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Store results
        results[model_type] = {
            metric: {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            } for metric, scores in cv_results.items()
        }
    
    # Find best model type based on accuracy
    best_model_type = max(results.keys(), key=lambda k: results[k]['accuracy']['mean'])
    print(f"\nBest model type: {best_model_type.upper()} with accuracy: {results[best_model_type]['accuracy']['mean']:.4f}")
    
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
    # Reshape X if needed (for 3D data)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    # Generate predictions
    predictions = model.predict(X)
    
    # Convert to condition names if encoder is provided
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
    # Convert one-hot encoded y to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    # Reshape X if needed (for 3D data)
    if len(X.shape) > 2:
        X_reshaped = X.reshape(X.shape[0], -1)
    else:
        X_reshaped = X
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y_indices)
    
    return X_resampled, y_resampled


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
                lambda x: np.argmax(np.bincount(x)), 
                axis=0, 
                arr=predictions
            )
    
    # Predict with ensemble
    ensemble_preds = ensemble_predict(X_test, models, weights)
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
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
    model_path = os.path.join(base_path, 'model.pkl')
    joblib.dump(model, model_path)
    
    # Save preprocessors
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    encoder_path = os.path.join(base_path, 'encoder.pkl')
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    
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
    model_path = os.path.join(base_path, 'model.pkl')
    model = joblib.load(model_path)
    
    # Load preprocessors
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    encoder_path = os.path.join(base_path, 'encoder.pkl')
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    return model, scaler, encoder


def main():
    """
    Main function for direct execution.
    """
    # Check if data directory exists
    if not os.path.exists('data'):
        print("Error: 'data' directory not found.")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        # Import necessary modules
        from data_preprocessing import process_pipeline
        from feature_extraction import extract_all_features
        
        # Process data with error handling
        print("\n1. Loading and preprocessing data...")
        try:
            processed_data, scaler = process_pipeline('data')
            
            # Check for NaN values after preprocessing
            nan_count = processed_data.isna().sum().sum()
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            raise ValueError("Cannot continue without preprocessed data.")
        
        # Step 2: Extract features
        print("\n2. Extracting features...")
        try:
            # Enable WKS features for better performance
            features, y, encoder = extract_all_features(processed_data, include_wks=True)
            
            # Check for NaN values in features
            if isinstance(features, pd.DataFrame):
                nan_count = features.isna().sum().sum()
                if nan_count > 0:
                    print(f"Warning: Extracted features contain {nan_count} NaN values.")
                    print("These will be imputed during model training.")
                    
                    # Impute NaN values in features
                    imputer = SimpleImputer(strategy='mean')
                    features = pd.DataFrame(
                        imputer.fit_transform(features),
                        columns=features.columns
                    )
                    print("NaN values have been imputed.")
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise ValueError("Cannot continue without proper feature extraction.")
        
        # Reshape features if needed
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        
        # Final check for NaN values in X
        if np.isnan(X).any():
            print("Warning: NaN values still present in feature matrix.")
            print("Applying final imputation...")
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            print("Final imputation complete.")
        
        # Step 3: Address class imbalance using SMOTE
        print("\n3. Addressing class imbalance with SMOTE...")
        try:
            X_resampled, y_resampled = balance_data_with_smote(X, y)
            print(f"Original data shape: {X.shape}, Resampled data shape: {X_resampled.shape}")
            # Use resampled data for further processing
            X = X_resampled
            y = y_resampled
        except Exception as e:
            print(f"Warning: Could not apply SMOTE: {e}")
            print("Proceeding with original imbalanced data.")
            
        # Step 4: Cross-validate traditional ML models
        print("\n4. Cross-validating traditional models...")
        try:
            cv_results = cross_validate_models(X, y)
            # Find best model type
            best_model_type = max(cv_results.keys(), key=lambda k: cv_results[k]['accuracy']['mean'])
            print(f"Best traditional model: {best_model_type.upper()} with accuracy: {cv_results[best_model_type]['accuracy']['mean']:.4f}")
        except Exception as e:
            print(f"Error during cross-validation: {e}")
            print("Falling back to Random Forest as default model.")
            best_model_type = 'rf'  # Use Random Forest as fallback
        
        # Step 5: Train traditional model with best type
        print(f"\n5. Training {best_model_type.upper()} model...")
        try:
            model, X_train, X_test, y_train, y_test = train_model(X, y, model_type=best_model_type, test_size=0.3)
        except Exception as e:
            print(f"Error during model training: {e}")
            print("Trying with Random Forest as fallback...")
            best_model_type = 'rf'  # Use Random Forest as fallback
            model, X_train, X_test, y_train, y_test = train_model(X, y, model_type=best_model_type, test_size=0.3)
        
        # Step 6: Optimize hyperparameters with regularization to prevent overfitting
        print("\n6. Optimizing hyperparameters with regularization...")
        try:
            # Define parameter grid with regularization parameters
            if best_model_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif best_model_type == 'gb':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9],
                    'min_samples_split': [2, 5, 10]
                }
            elif best_model_type == 'mlp':
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
                    'learning_rate_init': [0.001, 0.01],
                    'early_stopping': [True],
                    'max_iter': [300]
                }
            else:
                param_grid = None
                
            optimized_model, best_params = optimize_hyperparameters(X, y, model_type=best_model_type, param_grid=param_grid)
            print(f"Best parameters: {best_params}")
        except Exception as e:
            print(f"Error during hyperparameter optimization: {e}")
            print("Using default model without optimization...")
            optimized_model = model  # Use the already trained model
            best_params = {}
        
        # Step 7: Evaluate optimized traditional model
        print("\n7. Evaluating optimized traditional model...")
        try:
            metrics, report, cm = evaluate_model(optimized_model, X_test, y_test, encoder)
            
            print("\nTraditional Model Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            print("\nClassification Report:")
            print(report)
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            print("Skipping detailed evaluation.")
        
        # Step 8: Prepare data for deep learning models
        print("\n8. Preparing data for deep learning models...")
        try:
            # Create sliding windows for deep learning
            window_size = 100  # Adjust based on your data characteristics
            step_size = 10
            
            # Reshape the original processed data into sliding windows
            numerical_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
            X_windows, _ = dlm.prepare_sliding_window_data(
                processed_data[numerical_cols], 
                window_size=window_size, 
                step_size=step_size
            )
            
            # Check if windowed data was created successfully
            if X_windows is None:
                raise ValueError("Failed to create windowed data. Window size may be too large for the data.")
                
            # Create labels for each window (use the most frequent label in the window)
            windowed_labels = []
            for i in range(X_windows.shape[0]):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                if end_idx > len(processed_data):
                    end_idx = len(processed_data)
                
                # Get the most frequent fault type in this window
                # Check which column contains the fault type information
                fault_col = None
                if 'fault_type' in processed_data.columns:
                    fault_col = 'fault_type'
                elif 'condition' in processed_data.columns:
                    fault_col = 'condition'
                elif 'fault' in processed_data.columns:
                    fault_col = 'fault'
                
                if fault_col is None:
                    raise ValueError(f"Could not find fault type column in the data. Available columns: {processed_data.columns.tolist()}")
                    
                window_fault_types = processed_data[fault_col].iloc[start_idx:end_idx]
                most_frequent = window_fault_types.mode()[0]
                windowed_labels.append(most_frequent)
            
            # Encode labels
            dl_encoder = encoder  # Use the same encoder for consistency
            windowed_y = pd.get_dummies(dl_encoder.transform(windowed_labels)).values
            
            print(f"Deep learning data shape: {X_windows.shape}")
            print(f"Deep learning labels shape: {windowed_y.shape}")
            
            # Split data for deep learning
            X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
                X_windows, windowed_y, test_size=0.3, random_state=42
            )
        except Exception as e:
            print(f"Error preparing deep learning data: {e}")
            print("Skipping deep learning models.")
            X_train_dl, X_test_dl, y_train_dl, y_test_dl = None, None, None, None
        
        # Step 9: Train CNN-LSTM hybrid model if data is prepared
        if X_train_dl is not None:
            print("\n9. Training CNN-LSTM hybrid model...")
            try:
                # Train CNN-LSTM model using the new function
                dl_results = dlm.train_deep_learning_model(
                    X_train_dl, y_train_dl, X_test_dl, y_test_dl,
                    model_type='cnn_lstm',
                    epochs=30,
                    batch_size=16,
                    patience=5
                )
                
                # Extract model and metrics
                dl_model = dl_results['model']
                dl_metrics = dl_results['metrics']
                dl_history = dl_results['history']
                
                # Print evaluation metrics
                print("\nDeep Learning Model Evaluation Metrics:")
                print(f"accuracy: {dl_metrics['accuracy']:.4f}")
                print(f"precision: {dl_metrics['precision']:.4f}")
                print(f"recall: {dl_metrics['recall']:.4f}")
                print(f"f1_score: {dl_metrics['f1']:.4f}")
                
                # Save the model
                if dlm.TENSORFLOW_AVAILABLE:
                    model_save_path = 'models/cnn_lstm_model'
                    dl_model.save(model_save_path)
                    print(f"CNN-LSTM model saved to {model_save_path}")
            except Exception as e:
                print(f"Error training CNN-LSTM model: {e}")
                print("Skipping CNN-LSTM model.")
        else:
            print("\n9. Training CNN-LSTM hybrid model...")
            print("Deep learning data not prepared. Skipping CNN-LSTM model.")
            
        # Step 10: Perform reliability analysis
        print("\n10. Performing reliability analysis...")
        try:
            fault_types = encoder.classes_
            
            for fault_type in fault_types:
                if fault_type == 'steady_state':
                    continue  # Skip steady state as it's not a fault
            
                # Simulate fault occurrences (in a real scenario, use actual data)
                # For demonstration, we'll use random values
                np.random.seed(42)  # For reproducibility
                num_failures = np.random.randint(1, 10)  # Random number of failures
                operating_hours = 8760  # One year in hours
                
                # Perform reliability analysis
                reliability_results = ra.analyze_reliability(
                    pd.DataFrame({'fault': [fault_type] * num_failures}),
                    time_column=None,
                    fault_column='fault',
                    fault_value=fault_type,
                    operating_hours=operating_hours
                )
                
                # Generate reliability report
                report = ra.generate_reliability_report(reliability_results, fault_type=fault_type)
                print(f"\nReliability Analysis for {fault_type}:")
                print(f"MTTF: {reliability_results['mttf']:.2f} hours")
                print(f"Reliability: {reliability_results['reliability']:.4f}")
                
                # Save the report to a file
                with open(f"models/{fault_type}_reliability_report.txt", 'w') as f:
                    f.write(report)
                
                # Plot reliability curve
                ra.plot_reliability_curve(
                    reliability_results,
                    time_range=(0, 2 * reliability_results['mttf'], 100),
                    title=f"{fault_type} Reliability Over Time"
                )
                print(f"Reliability curve saved to reliability_curve.png")
        
        # Handle any exceptions during the entire process
        except Exception as e:
            print(f"Error during analysis: {e}")
            print("Skipping remaining analysis.")
        
        # Step 11: Save all models and preprocessors
        print("\n11. Saving all models and preprocessors...")
        try:
            # Save traditional model
            save_model_and_preprocessors(optimized_model, scaler, encoder)
            print("Traditional model and preprocessors saved to models directory")
            
            # Save additional metadata
            metadata = {
                'traditional_model': {
                    'type': best_model_type,
                    'best_params': best_params,
                    'metrics': metrics
                },
                'deep_learning': {
                    'window_size': window_size,
                    'step_size': step_size
                },
                'features': {
                    'total': X.shape[1],
                    'wks_enabled': True
                }
            }
            
            import json
            with open('models/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print("\nNPP Fault Monitoring System setup complete!")
            print("All models and analysis results saved to the models directory.")
        except Exception as e:
            print(f"Error saving models: {e}")
            print("Model training completed but could not save all models.")
        
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        print("NPP Fault Monitoring System setup failed.")
        import traceback
        traceback.print_exc()


# Execute main function if script is run directly
if __name__ == "__main__":
    main()