import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import warnings

def calculate_failure_rate(failures, operating_time):
    """
    Calculate the failure rate (λ).
    
    Args:
        failures (int): Number of failures
        operating_time (float): Total operating time
        
    Returns:
        float: Failure rate
    """
    if operating_time <= 0:
        raise ValueError("Operating time must be positive")
    
    return failures / operating_time

def calculate_reliability(failure_rate):
    """
    Calculate reliability (1 - λ).
    
    Args:
        failure_rate (float): Failure rate
        
    Returns:
        float: Reliability
    """
    return 1 - failure_rate

def calculate_mttf(failure_rate):
    """
    Calculate Mean Time To Failure (MTTF = 1/λ).
    
    Args:
        failure_rate (float): Failure rate
        
    Returns:
        float: MTTF
    """
    if failure_rate <= 0:
        warnings.warn("Failure rate is zero or negative, MTTF will be infinite or negative")
        return float('inf') if failure_rate == 0 else float('-inf')
    
    return 1 / failure_rate

def calculate_dynamic_reliability(time, mttf):
    """
    Calculate dynamic reliability over time (e^(-T/MTTF)).
    
    Args:
        time (float or np.array): Time or array of times
        mttf (float): Mean Time To Failure
        
    Returns:
        float or np.array: Reliability at specified time(s)
    """
    if mttf <= 0:
        warnings.warn("MTTF is zero or negative, reliability calculation may be incorrect")
        return 0.0
    
    return np.exp(-time / mttf)

def estimate_weibull_parameters(failure_times):
    """
    Estimate Weibull distribution parameters from failure times.
    
    Args:
        failure_times (list or np.array): Times at which failures occurred
        
    Returns:
        tuple: (shape, scale) parameters of the Weibull distribution
    """
    shape, loc, scale = weibull_min.fit(failure_times, floc=0)
    return shape, scale

def weibull_reliability(time, shape, scale):
    """
    Calculate reliability using Weibull distribution.
    
    Args:
        time (float or np.array): Time or array of times
        shape (float): Weibull shape parameter
        scale (float): Weibull scale parameter
        
    Returns:
        float or np.array: Reliability at specified time(s)
    """
    return np.exp(-((time / scale) ** shape))

def analyze_reliability(fault_data, time_column, fault_column, fault_value, operating_hours=8760):
    """
    Perform reliability analysis on fault data.
    
    Args:
        fault_data (pd.DataFrame): Fault data
        time_column (str): Name of the time column
        fault_column (str): Name of the fault indicator column
        fault_value: Value in fault_column that indicates a fault
        operating_hours (float): Total operating hours (default: 1 year = 8760 hours)
        
    Returns:
        dict: Reliability metrics
    """
    # Count failures
    failures = fault_data[fault_data[fault_column] == fault_value].shape[0]
    
    # Calculate basic metrics
    failure_rate = calculate_failure_rate(failures, operating_hours)
    reliability = calculate_reliability(failure_rate)
    mttf = calculate_mttf(failure_rate)
    
    # Extract failure times if available
    if time_column in fault_data.columns:
        failure_times = fault_data[fault_data[fault_column] == fault_value][time_column].values
        
        # If we have enough failure times, estimate Weibull parameters
        if len(failure_times) >= 3:
            try:
                shape, scale = estimate_weibull_parameters(failure_times)
                weibull_params = {'shape': shape, 'scale': scale}
            except Exception as e:
                warnings.warn(f"Could not estimate Weibull parameters: {e}")
                weibull_params = None
        else:
            weibull_params = None
    else:
        failure_times = None
        weibull_params = None
    
    # Compile results
    results = {
        'failures': failures,
        'operating_hours': operating_hours,
        'failure_rate': failure_rate,
        'reliability': reliability,
        'mttf': mttf,
        'failure_times': failure_times,
        'weibull_params': weibull_params
    }
    
    return results

def plot_reliability_curve(results, time_range=None, title="System Reliability Over Time"):
    """
    Plot reliability curve over time.
    
    Args:
        results (dict): Results from analyze_reliability
        time_range (tuple): (start_time, end_time, num_points) for time axis
        title (str): Plot title
    """
    if time_range is None:
        # Default: Plot from 0 to 2*MTTF with 100 points
        time_range = (0, 2 * results['mttf'], 100)
    
    start_time, end_time, num_points = time_range
    times = np.linspace(start_time, end_time, num_points)
    
    plt.figure(figsize=(10, 6))
    
    # Plot exponential reliability
    exp_reliability = calculate_dynamic_reliability(times, results['mttf'])
    plt.plot(times, exp_reliability, 'b-', label='Exponential Model')
    
    # Plot Weibull reliability if parameters are available
    if results['weibull_params'] is not None:
        shape = results['weibull_params']['shape']
        scale = results['weibull_params']['scale']
        weib_reliability = weibull_reliability(times, shape, scale)
        plt.plot(times, weib_reliability, 'r--', label=f'Weibull Model (shape={shape:.2f}, scale={scale:.2f})')
    
    # Add MTTF vertical line
    plt.axvline(x=results['mttf'], color='g', linestyle=':', label=f'MTTF = {results["mttf"]:.2f}')
    
    # Add horizontal line at reliability = 0.5
    plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('Time (hours)')
    plt.ylabel('Reliability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    
    # Save the plot
    plt.savefig('reliability_curve.png')
    plt.close()
    
    return 'reliability_curve.png'

def generate_reliability_report(results, fault_type="System"):
    """
    Generate a text report of reliability analysis results.
    
    Args:
        results (dict): Results from analyze_reliability
        fault_type (str): Type of fault or system being analyzed
        
    Returns:
        str: Formatted report
    """
    report = f"Reliability Analysis Report for {fault_type}\n"
    report += "=" * 50 + "\n\n"
    
    report += f"Number of Failures: {results['failures']}\n"
    report += f"Operating Hours: {results['operating_hours']}\n\n"
    
    report += f"Failure Rate (λ): {results['failure_rate']:.6f} failures/hour\n"
    report += f"Reliability: {results['reliability']:.4f}\n"
    report += f"Mean Time To Failure (MTTF): {results['mttf']:.2f} hours\n\n"
    
    # Add Weibull information if available
    if results['weibull_params'] is not None:
        shape = results['weibull_params']['shape']
        scale = results['weibull_params']['scale']
        report += "Weibull Distribution Parameters:\n"
        report += f"  Shape (β): {shape:.4f}\n"
        report += f"  Scale (η): {scale:.4f}\n\n"
        
        # Interpret shape parameter
        if shape < 1:
            report += "Shape parameter < 1 indicates early life failures (infant mortality).\n"
        elif shape == 1:
            report += "Shape parameter = 1 indicates random failures (constant failure rate).\n"
        else:
            report += "Shape parameter > 1 indicates wear-out failures (increasing failure rate).\n"
    
    # Add reliability at different time points
    report += "\nReliability at Different Time Points:\n"
    time_points = [results['mttf']/4, results['mttf']/2, results['mttf'], results['mttf']*2]
    
    for t in time_points:
        rel = calculate_dynamic_reliability(t, results['mttf'])
        report += f"  At t = {t:.2f} hours: {rel:.4f}\n"
    
    return report
