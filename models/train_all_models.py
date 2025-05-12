#!/usr/bin/env python3
"""
Script to train all models sequentially for NPP Fault Monitoring.
This script trains CNN, RNN, LSTM, CNN-RNN, and CNN-LSTM models with the same parameters
to allow for easy comparison of their performance.
"""

import os
import sys
import subprocess
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_model(model_type, epochs=5, batch_size=32, window_size=100, step_size=50, balance=True):
    """
    Train a specific model type using the train_model.py script.
    
    Args:
        model_type (str): Type of model to train ('cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm')
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        window_size (int): Window size for sliding window
        step_size (int): Step size for sliding window
        balance (bool): Whether to balance the classes
        
    Returns:
        tuple: (exit_code, output)
    """
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} model with {epochs} epochs")
    print(f"{'='*80}")
    
    # Build command
    cmd = [
        "python", "models/train_model.py",
        f"--model={model_type}",
        f"--epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--window_size={window_size}",
        f"--step_size={step_size}"
    ]
    
    if balance:
        cmd.append("--balance")
    
    # Run command
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Capture output in real-time
    output = []
    for line in process.stdout:
        print(line, end='')
        output.append(line)
    
    # Wait for process to complete
    process.wait()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining time for {model_type.upper()}: {training_time:.2f} seconds")
    
    return process.returncode, ''.join(output)

def compare_models(models_data):
    """
    Compare the performance of different models.
    
    Args:
        models_data (dict): Dictionary containing model performance data
    """
    # Create directory for comparison results
    os.makedirs('analysis/comparison', exist_ok=True)
    
    # Extract metrics for comparison
    model_names = list(models_data.keys())
    accuracies = [data['accuracy'] for data in models_data.values()]
    precisions = [data['precision'] for data in models_data.values()]
    recalls = [data['recall'] for data in models_data.values()]
    f1_scores = [data['f1_score'] for data in models_data.values()]
    training_times = [data['training_time'] for data in models_data.values()]
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'Training Time (s)': training_times
    })
    
    # Save comparison to CSV
    comparison_df.to_csv('analysis/comparison/model_comparison.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    bar_width = 0.15
    index = np.arange(len(model_names))
    
    plt.bar(index, accuracies, bar_width, label='Accuracy')
    plt.bar(index + bar_width, precisions, bar_width, label='Precision')
    plt.bar(index + 2*bar_width, recalls, bar_width, label='Recall')
    plt.bar(index + 3*bar_width, f1_scores, bar_width, label='F1 Score')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(index + 1.5*bar_width, model_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis/comparison/model_performance.png')
    
    # Plot training times
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, training_times)
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.tight_layout()
    plt.savefig('analysis/comparison/training_times.png')
    
    print("\nModel comparison complete!")
    print(f"Results saved to analysis/comparison/model_comparison.csv")
    print(f"Performance plot saved to analysis/comparison/model_performance.png")
    print(f"Training time plot saved to analysis/comparison/training_times.png")
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Identify best model based on F1 score
    best_model_idx = f1_scores.index(max(f1_scores))
    best_model = model_names[best_model_idx]
    print(f"\nBest performing model based on F1 score: {best_model}")
    print(f"  Accuracy: {accuracies[best_model_idx]:.4f}")
    print(f"  Precision: {precisions[best_model_idx]:.4f}")
    print(f"  Recall: {recalls[best_model_idx]:.4f}")
    print(f"  F1 Score: {f1_scores[best_model_idx]:.4f}")
    print(f"  Training Time: {training_times[best_model_idx]:.2f} seconds")

def extract_metrics_from_output(output):
    """
    Extract performance metrics from the training output.
    
    Args:
        output (str): Output from the training process
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'training_time': 0.0
    }
    
    # Extract metrics from output
    for line in output.split('\n'):
        if 'Accuracy:' in line and 'Test Metrics:' in output.split('\n')[output.split('\n').index(line) - 1]:
            metrics['accuracy'] = float(line.split(':')[1].strip())
        elif 'Precision:' in line and not 'macro' in line and not 'weighted' in line:
            metrics['precision'] = float(line.split(':')[1].strip())
        elif 'Recall:' in line and not 'macro' in line and not 'weighted' in line:
            metrics['recall'] = float(line.split(':')[1].strip())
        elif 'F1 Score:' in line and not 'macro' in line and not 'weighted' in line:
            metrics['f1_score'] = float(line.split(':')[1].strip())
        elif 'Training completed in' in line:
            metrics['training_time'] = float(line.split('in')[1].split('seconds')[0].strip())
    
    return metrics

def main():
    """Main function to train all models and compare their performance."""
    parser = argparse.ArgumentParser(description='Train all models for NPP Fault Monitoring')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for sliding window')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for sliding window')
    parser.add_argument('--no_balance', action='store_true', help='Disable class balancing')
    parser.add_argument('--models', type=str, nargs='+', default=['cnn', 'rnn', 'lstm', 'cnn_rnn', 'cnn_lstm'],
                        help='Models to train (default: all models)')
    
    args = parser.parse_args()
    
    # Create directories for results
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('analysis/plots', exist_ok=True)
    os.makedirs('analysis/text', exist_ok=True)
    
    # Train all models
    models_data = {}
    for model_type in args.models:
        exit_code, output = train_model(
            model_type, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            window_size=args.window_size,
            step_size=args.step_size,
            balance=not args.no_balance
        )
        
        if exit_code == 0:
            print(f"\n{model_type.upper()} model training completed successfully!")
            # Extract metrics from output
            metrics = extract_metrics_from_output(output)
            models_data[model_type] = metrics
        else:
            print(f"\n{model_type.upper()} model training failed with exit code {exit_code}")
    
    # Compare models
    if models_data:
        compare_models(models_data)
    else:
        print("\nNo models were trained successfully. Cannot compare models.")

if __name__ == "__main__":
    main()
