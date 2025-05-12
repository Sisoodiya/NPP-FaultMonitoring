"""
Self-Improved Aquila Optimizer (SIAO) implementation for NPP Fault Monitoring.

This module implements the Aquila Optimizer algorithm as described in the research paper:
"Intelligent Fault Monitoring and Reliability Analysis in Safety-Critical Systems of 
Nuclear Power Plants Using SIAO-CNN-ORNN"

The optimizer has four stages:
1. Expanded exploration
2. Narrowed exploration
3. Expanded exploitation
4. Narrowed exploitation
"""

import numpy as np
import random
import math
from tqdm import tqdm

def aquila_optimizer(objective_function, dimensions, lb, ub, max_iter=100, pop_size=30):
    """
    Implement the four-stage Aquila Optimizer for parameter optimization.
    
    Args:
        objective_function: Function to optimize (minimize)
        dimensions: Number of dimensions in the search space
        lb: Lower bounds for each dimension
        ub: Upper bounds for each dimension
        max_iter: Maximum number of iterations
        pop_size: Population size
        
    Returns:
        tuple: (best_position, best_fitness, convergence_curve)
    """
    # Initialize population
    population = initialize_population(pop_size, dimensions, lb, ub)
    
    # Initialize fitness
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = objective_function(population[i])
    
    # Initialize best solution
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Initialize convergence curve
    convergence_curve = np.zeros(max_iter)
    
    # Main loop
    for t in tqdm(range(max_iter), desc="SIAO Optimization"):
        # Calculate alpha and beta parameters
        alpha = 2 * (1 - (t / max_iter))
        beta = 2 * ((t / max_iter) ** 2)
        
        # Determine the phase based on current iteration
        if t < max_iter * 0.25:
            phase = "expanded_exploration"
        elif t < max_iter * 0.5:
            phase = "narrowed_exploration"
        elif t < max_iter * 0.75:
            phase = "expanded_exploitation"
        else:
            phase = "narrowed_exploitation"
        
        # Update each solution
        for i in range(pop_size):
            # Skip the best solution
            if i == best_idx:
                continue
            
            # Update position based on the current phase
            if phase == "expanded_exploration":
                # Expanded exploration: Wide search with random walks
                r1 = np.random.rand(dimensions)
                r2 = np.random.rand(dimensions)
                
                # Levy flight component for exploration
                levy = levy_flight(dimensions)
                
                # Update position with expanded exploration
                population[i] = population[i] + alpha * r1 * (best_position - population[i]) + r2 * levy
                
            elif phase == "narrowed_exploration":
                # Narrowed exploration: More focused around promising areas
                r1 = np.random.rand(dimensions)
                
                # Calculate mean position of top solutions
                top_indices = np.argsort(fitness)[:max(3, pop_size//5)]
                mean_position = np.mean(population[top_indices], axis=0)
                
                # Update position with narrowed exploration
                population[i] = population[i] + alpha * r1 * (mean_position - population[i])
                
            elif phase == "expanded_exploitation":
                # Expanded exploitation: Exploit the best solution with some randomness
                r1 = np.random.rand(dimensions)
                r2 = np.random.rand(dimensions)
                
                # Chaotic component for diversity
                chaotic = chaotic_map(t, max_iter)
                
                # Update position with expanded exploitation
                population[i] = best_position + beta * r1 * (best_position - population[i]) + r2 * chaotic
                
            else:  # narrowed_exploitation
                # Narrowed exploitation: Fine-tune around the best solution
                r1 = np.random.rand(dimensions)
                
                # Update position with narrowed exploitation (more aggressive)
                population[i] = best_position + beta * r1 * (best_position - population[i]) * 0.5
            
            # Apply bounds
            population[i] = np.clip(population[i], lb, ub)
            
            # Evaluate new solution
            new_fitness = objective_function(population[i])
            
            # Update if better
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
                
                # Update global best if needed
                if new_fitness < best_fitness:
                    best_position = population[i].copy()
                    best_fitness = new_fitness
                    best_idx = i
        
        # Record best fitness for this iteration
        convergence_curve[t] = best_fitness
        
        # Optional: Print progress
        if (t+1) % 10 == 0 or t == 0 or t == max_iter-1:
            print(f"Iteration {t+1}/{max_iter}, Best fitness: {best_fitness:.6f}, Phase: {phase}")
    
    return best_position, best_fitness, convergence_curve

def initialize_population(pop_size, dimensions, lb, ub):
    """Initialize population with random values within bounds."""
    population = np.zeros((pop_size, dimensions))
    for i in range(pop_size):
        for j in range(dimensions):
            population[i, j] = lb[j] + (ub[j] - lb[j]) * np.random.rand()
    return population

def levy_flight(dimensions):
    """Generate Levy flight for exploration."""
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dimensions) * sigma
    v = np.random.randn(dimensions)
    step = u / np.abs(v) ** (1 / beta)
    return step * 0.01  # Scale down to avoid too large steps

def chaotic_map(t, max_iter):
    """Generate chaotic values using logistic map."""
    x = 0.7  # Initial value
    for _ in range(int(10 * t / max_iter) + 1):
        x = 4 * x * (1 - x)
    return x

def optimize_wks_weights(data, window_size=100, max_iter=50, pop_size=20):
    """
    Use Aquila Optimizer to find optimal weight parameter for WKS.
    
    Args:
        data: Input data for WKS calculation
        window_size: Window size for WKS calculation
        max_iter: Maximum iterations for optimization
        pop_size: Population size for optimization
        
    Returns:
        float: Optimized omega parameter for WKS
    """
    from feature_extraction import weighted_kurtosis_skewness
    
    # Define objective function (minimize negative WKS to maximize WKS)
    def objective_function(params):
        omega = params[0]  # Only one parameter to optimize
        
        # Calculate WKS for each window
        wks_values = []
        for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i + window_size]
            try:
                wks = weighted_kurtosis_skewness(window, omega)
                wks_values.append(wks)
            except:
                wks_values.append(0)
        
        # Calculate mean WKS
        mean_wks = np.mean(wks_values) if wks_values else 0
        
        # Return negative WKS (since we want to maximize WKS)
        return -mean_wks
    
    # Define bounds for omega parameter (typically between 0 and 1)
    lb = np.array([0.01])
    ub = np.array([0.99])
    
    # Run optimization
    best_params, best_fitness, _ = aquila_optimizer(
        objective_function, 
        dimensions=1, 
        lb=lb, 
        ub=ub, 
        max_iter=max_iter, 
        pop_size=pop_size
    )
    
    # Return optimized omega parameter
    return best_params[0]

# Example usage
if __name__ == "__main__":
    # Test the optimizer with a simple function
    def sphere(x):
        return np.sum(x**2)
    
    # Define bounds
    dimensions = 5
    lb = np.ones(dimensions) * -10
    ub = np.ones(dimensions) * 10
    
    # Run optimization
    best_position, best_fitness, convergence = aquila_optimizer(
        sphere, dimensions, lb, ub, max_iter=100, pop_size=30
    )
    
    print(f"Best position: {best_position}")
    print(f"Best fitness: {best_fitness}")
    
    # Plot convergence
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(convergence)
    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig('siao_convergence.png')
    plt.close()
