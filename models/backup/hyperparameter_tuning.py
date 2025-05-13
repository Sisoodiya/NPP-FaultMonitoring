"""
Standardized Hyperparameter Tuning for NPP Fault Monitoring Models

This module provides a standardized approach to hyperparameter tuning for all models
in the NPP Fault Monitoring project, including:
- CNN
- RNN
- LSTM
- CNN-RNN
- CNN-LSTM
- SIAO-CNN-ORNN (standard and enhanced)

Features:
- Standardized hyperparameter search spaces for all models
- Multiple tuning methods: grid search, random search, Bayesian optimization
- Comprehensive evaluation metrics
- Automatic results logging and visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json
import time
from sklearn.model_selection import ParameterGrid, ParameterSampler

# Try to import Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Define standardized hyperparameter search spaces
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
        'use_adaptive_learning': [True, False],
        'patience': [5, 10, 15],
        'min_delta': [0.0001, 0.001, 0.01],
        'monitor': ['val_loss', 'val_accuracy', 'val_auc', 'val_composite_score'],
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
            'attention_units': [32, 64, 128],
            'num_heads': [1, 2, 4, 8],
            'use_aquila_optimizer': [True, False],
            'filters_multiplier': [0.5, 1.0, 1.5, 2.0],
            'units_multiplier': [0.5, 1.0, 1.5, 2.0],
            'kernel_size': [3, 5, 7],
            'pool_size': [2, 3],
        }
    else:
        specific_params = {}
    
    # Combine common and specific parameters
    return {**common_params, **specific_params}

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
    filtered_grid = param_grid.copy()
    
    # Remove focal loss parameters if not using focal loss
    if not use_focal_loss and 'use_focal_loss' in filtered_grid:
        filtered_grid.pop('use_focal_loss')
        filtered_grid.pop('focal_gamma', None)
        filtered_grid.pop('focal_alpha', None)
    
    # Remove WKS parameters if not using WKS
    if not use_wks and 'use_wks' in filtered_grid:
        filtered_grid.pop('use_wks')
        filtered_grid.pop('wks_window_size', None)
        filtered_grid.pop('wks_step_size', None)
    
    # Remove custom class weights parameter if not using custom class weights
    if not custom_class_weights and 'custom_class_weights' in filtered_grid:
        filtered_grid.pop('custom_class_weights')
    
    # Remove Aquila optimizer parameter if not using Aquila optimizer
    if not use_aquila_optimizer and 'use_aquila_optimizer' in filtered_grid:
        filtered_grid.pop('use_aquila_optimizer')
    
    # Remove adaptive learning rate parameter if not using adaptive learning
    if not use_adaptive_learning and 'use_adaptive_learning' in filtered_grid:
        filtered_grid.pop('use_adaptive_learning')
    
    # Remove bidirectional parameter if not using bidirectional layers
    if use_bidirectional is not None and 'use_bidirectional' in filtered_grid:
        filtered_grid.pop('use_bidirectional')
    
    # Remove GRU parameter if GRU usage is fixed
    if use_gru is not None and 'use_gru' in filtered_grid:
        filtered_grid.pop('use_gru')
    
    # Remove model-specific parameters
    if model_type not in ['cnn', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced']:
        filtered_grid.pop('filters_multiplier', None)
        filtered_grid.pop('kernel_size', None)
        filtered_grid.pop('pool_size', None)
    
    if model_type not in ['rnn', 'lstm', 'cnn_rnn', 'cnn_lstm', 'siao', 'siao_enhanced']:
        filtered_grid.pop('recurrent_dropout', None)
        filtered_grid.pop('use_bidirectional', None)
        filtered_grid.pop('units_multiplier', None)
    
    if model_type not in ['rnn', 'lstm', 'cnn_rnn']:
        filtered_grid.pop('use_gru', None)
    
    if model_type not in ['siao', 'siao_enhanced']:
        filtered_grid.pop('attention_units', None)
        filtered_grid.pop('num_heads', None)
    
    return filtered_grid

def evaluate_model(X_train, y_train, X_val, y_val, model_type, params, train_model_func,
                  use_adaptive_learning=True, use_aquila_optimizer=False, 
                  use_focal_loss=False, use_wks=False, custom_class_weights=False,
                  use_bidirectional=True, use_gru=None, results_dir=None):
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
        results_dir: Directory to save results
        
    Returns:
        Composite score (higher is better)
    """
    print(f"\nEvaluating parameters: {params}")
    
    # Create a copy of the parameters to avoid modifying the original
    model_params = params.copy()
    
    # Extract parameters that are not model-specific
    batch_size = model_params.pop('batch_size', 32)
    
    # Train the model with the current parameters
    model, history = train_model_func(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_type=model_type,
        model_name=f"{model_type}_tuning",
        epochs=30,  # Use a fixed number of epochs for tuning
        batch_size=batch_size,
        use_adaptive_learning=use_adaptive_learning,
        use_aquila_optimizer=use_aquila_optimizer,
        use_focal_loss=use_focal_loss,
        use_wks=use_wks,
        custom_class_weights=custom_class_weights,
        **model_params
    )
    
    # Evaluate on validation set
    val_loss, val_accuracy, val_auc, val_precision, val_recall = model.evaluate(
        X_val, y_val, verbose=0
    )
    
    # Calculate F1 score
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)
    
    # Calculate composite score (weighted combination of metrics)
    composite_score = (
        0.3 * val_accuracy + 
        0.3 * val_auc + 
        0.2 * val_f1 + 
        0.2 * (1.0 - val_loss)  # Convert loss to a score (higher is better)
    )
    
    # Print evaluation results
    print(f"Validation results - Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, "
          f"F1: {val_f1:.4f}, Loss: {val_loss:.4f}, Composite: {composite_score:.4f}")
    
    # Save results to a file if results_dir is provided
    if results_dir:
        results_file = f"{results_dir}/results.csv"
        
        # Create header if file doesn't exist
        if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
            with open(results_file, "w") as f:
                f.write("params,val_loss,val_accuracy,val_auc,val_precision,val_recall,val_f1,composite_score\n")
        
        # Append results
        with open(results_file, "a") as f:
            f.write(f"{params},{val_loss},{val_accuracy},{val_auc},{val_precision},{val_recall},{val_f1},{composite_score}\n")
    
    return composite_score

def grid_search(X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
               use_adaptive_learning=False, use_aquila_optimizer=False, 
               use_focal_loss=False, use_wks=False, custom_class_weights=False,
               results_dir=None, n_jobs=1):
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
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total parameter combinations: {len(param_combinations)}")
    
    # Evaluate each combination
    results = []
    for params in param_combinations:
        score = evaluate_model(
            X_train, y_train, X_val, y_val, model_type, params, train_model_func,
            use_adaptive_learning, use_aquila_optimizer, use_focal_loss, use_wks,
            custom_class_weights, results_dir
        )
        results.append((params, score))
    
    # Sort results by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Get best parameters
    best_params = results[0][0]
    best_score = results[0][1]
    
    return best_params, best_score

def random_search(X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
                 use_adaptive_learning=False, use_aquila_optimizer=False, 
                 use_focal_loss=False, use_wks=False, custom_class_weights=False,
                 results_dir=None, n_trials=20, n_jobs=1):
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
    # Generate random parameter combinations
    param_combinations = list(ParameterSampler(param_grid, n_iter=n_trials, random_state=42))
    print(f"Random parameter combinations: {len(param_combinations)}")
    
    # Evaluate each combination
    results = []
    for params in param_combinations:
        score = evaluate_model(
            X_train, y_train, X_val, y_val, model_type, params, train_model_func,
            use_adaptive_learning, use_aquila_optimizer, use_focal_loss, use_wks,
            custom_class_weights, results_dir
        )
        results.append((params, score))
    
    # Sort results by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Get best parameters
    best_params = results[0][0]
    best_score = results[0][1]
    
    return best_params, best_score

def bayesian_search(X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
                   use_adaptive_learning=False, use_aquila_optimizer=False, 
                   use_focal_loss=False, use_wks=False, custom_class_weights=False,
                   results_dir=None, n_trials=20):
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
        print("Bayesian optimization not available. Falling back to random search.")
        return random_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            use_adaptive_learning, use_aquila_optimizer, use_focal_loss, use_wks,
            custom_class_weights, results_dir, n_trials
        )
    
    # Convert parameter grid to skopt space
    space = []
    param_names = []
    
    for param, values in param_grid.items():
        param_names.append(param)
        
        if all(isinstance(v, bool) for v in values):
            space.append(Categorical(values))
        elif all(isinstance(v, int) for v in values):
            space.append(Integer(min(values), max(values)))
        elif all(isinstance(v, float) for v in values):
            space.append(Real(min(values), max(values)))
        else:
            space.append(Categorical(values))
    
    # Define objective function for minimization (negative score)
    def objective(x):
        # Convert list of parameters to dictionary
        params = {param_names[i]: x[i] for i in range(len(param_names))}
        
        score = evaluate_model(
            X_train, y_train, X_val, y_val, model_type, params, train_model_func,
            use_adaptive_learning, use_aquila_optimizer, use_focal_loss, use_wks,
            custom_class_weights, results_dir
        )
        
        return -score  # Negative because we want to maximize score
    
    # Run Bayesian optimization
    res = gp_minimize(objective, space, n_calls=n_trials, random_state=42)
    
    # Get best parameters
    best_params = {param_names[i]: res.x[i] for i in range(len(param_names))}
    best_score = -res.fun  # Convert back to positive score
    
    return best_params, best_score

def tune_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test, model_type, train_model_func,
                        use_adaptive_learning=True, use_aquila_optimizer=False, 
                        use_focal_loss=False, use_wks=False, custom_class_weights=False,
                        use_bidirectional=True, use_gru=None, tuning_method='grid', 
                        n_trials=20, n_jobs=1):
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
    print(f"Hyperparameter Tuning for {model_type.upper()} Model")
    print("=" * 50)
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"./tuning_results/{model_type}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get hyperparameter grid for the specified model type
    param_grid = get_hyperparameter_grid(model_type)
    
    # Filter hyperparameters based on command line arguments
    param_grid = filter_hyperparameters(
        param_grid, model_type, use_focal_loss, use_wks, custom_class_weights, 
        use_aquila_optimizer, use_adaptive_learning, use_bidirectional, use_gru
    )
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"results/{model_type}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save hyperparameter grid to file
    with open(f"{results_dir}/param_grid.json", "w") as f:
        # Convert numpy types to Python native types for JSON serialization
        serializable_grid = {}
        for k, v in param_grid.items():
            if isinstance(v, list):
                serializable_grid[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else 
                                       int(x) if isinstance(x, (np.int32, np.int64)) else x 
                                       for x in v]
            else:
                serializable_grid[k] = v
        json.dump(serializable_grid, f, indent=2)
    
    # Perform hyperparameter tuning
    print(f"\nPerforming {tuning_method} search for {model_type} model...")
    print(f"Number of hyperparameter combinations: {len(list(ParameterGrid(param_grid)))}")
    
    # Create a dictionary of feature flags to pass to search functions
    feature_flags = {
        'use_adaptive_learning': use_adaptive_learning,
        'use_aquila_optimizer': use_aquila_optimizer,
        'use_focal_loss': use_focal_loss,
        'use_wks': use_wks,
        'custom_class_weights': custom_class_weights,
        'use_bidirectional': use_bidirectional,
        'use_gru': use_gru
    }
    
    if tuning_method == 'grid':
        best_params, best_score = grid_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            **feature_flags, results_dir=results_dir, n_jobs=n_jobs
        )
    elif tuning_method == 'random':
        best_params, best_score = random_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            **feature_flags, results_dir=results_dir, n_trials=n_trials, n_jobs=n_jobs
        )
    elif tuning_method == 'bayesian':
        best_params, best_score = bayesian_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            **feature_flags, results_dir=results_dir, n_trials=n_trials
        )
    else:
        print(f"Invalid tuning method: {tuning_method}. Falling back to grid search.")
        best_params, best_score = grid_search(
            X_train, y_train, X_val, y_val, model_type, param_grid, train_model_func,
            **feature_flags, results_dir=results_dir, n_jobs=n_jobs
        )
    
    # Print best parameters and score
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best score: {best_score:.4f}")
    
    # Save best parameters to a file
    with open(f"{results_dir}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    
    # Extract parameters that are not model-specific
    batch_size = best_params.pop('batch_size', 32) if 'batch_size' in best_params else 32
    
    # Train the final model
    final_model, final_history = train_model_func(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,  # Use more epochs for final model
        batch_size=batch_size,
        use_adaptive_learning=use_adaptive_learning,
        use_aquila_optimizer=use_aquila_optimizer,
        use_focal_loss=use_focal_loss,
        use_wks=use_wks,
        custom_class_weights=custom_class_weights,
        **best_params
    )
    
    # Evaluate final model
    print("\nEvaluating final model...")
    
    # Evaluate on test set
    test_metrics = final_model.evaluate(X_test, y_test, verbose=1)
    test_metric_names = final_model.metrics_names
    
    # Create dictionary of test metrics
    test_results = {name: float(value) for name, value in zip(test_metric_names, test_metrics)}
    
    # Calculate F1 score if precision and recall are available
    if 'precision' in test_results and 'recall' in test_results:
        precision = test_results['precision']
        recall = test_results['recall']
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        test_results['f1'] = f1
    
    # Print test results
    print("\nTest results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model and results
    print("\nSaving model and results...")
    
    # Save model architecture and weights
    final_model.save(f"{results_dir}/final_model.h5")
    
    # Save model parameters and test results
    with open(f"{results_dir}/model_results.json", "w") as f:
        results = {
            'model_type': model_type,
            'hyperparameters': {k: (float(v) if isinstance(v, (int, float)) else v) 
                               for k, v in best_params.items()},
            'test_metrics': test_results,
            'tuning_method': tuning_method,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        }
        json.dump(results, f, indent=2)
    
    # Plot training history
    print("\nPlotting training history...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(final_history.history['accuracy'], label='Train')
    plt.plot(final_history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(final_history.history['loss'], label='Train')
    plt.plot(final_history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot AUC if available
    if 'auc' in final_history.history:
        plt.subplot(2, 2, 3)
        plt.plot(final_history.history['auc'], label='Train')
        plt.plot(final_history.history['val_auc'], label='Validation')
        plt.title('AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
    
    # Plot learning rate if available
    if 'lr' in final_history.history:
        plt.subplot(2, 2, 4)
        plt.plot(final_history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_history.png")
    
    print(f"\nHyperparameter tuning completed successfully!")
    print(f"Results saved to {results_dir}")
    
    return final_model, best_params, test_results
