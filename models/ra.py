"""
Reliability Analysis Module for NPP Fault Monitoring.
This module provides functions for calculating reliability metrics and generating reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def calculate_failure_rate(num_failures, operating_hours):
    """
    Calculate the failure rate (lambda).
    
    Args:
        num_failures (int): Number of failures
        operating_hours (float): Total operating hours
        
    Returns:
        float: Failure rate (failures per hour)
    """
    if operating_hours <= 0:
        return 0.0
    return num_failures / operating_hours

def calculate_reliability(failure_rate, time=1.0):
    """
    Calculate reliability using the exponential model.
    
    Args:
        failure_rate (float): Failure rate (lambda)
        time (float): Time point at which to calculate reliability
        
    Returns:
        float: Reliability at the specified time
    """
    return np.exp(-failure_rate * time)

def calculate_mttf(failure_rate):
    """
    Calculate Mean Time To Failure (MTTF).
    
    Args:
        failure_rate (float): Failure rate (lambda)
        
    Returns:
        float: MTTF (hours)
    """
    if failure_rate <= 0:
        return float('inf')
    return 1.0 / failure_rate

def calculate_availability(mttf, mttr):
    """
    Calculate availability.
    
    Args:
        mttf (float): Mean Time To Failure
        mttr (float): Mean Time To Repair
        
    Returns:
        float: Availability
    """
    if mttf <= 0 or mttr < 0:
        return 0.0
    return mttf / (mttf + mttr)

def analyze_reliability(fault_data, time_column=None, fault_column='fault', fault_value=None, operating_hours=8760):
    """
    Analyze reliability metrics from fault data.
    
    Args:
        fault_data (pd.DataFrame): DataFrame containing fault data
        time_column (str): Name of the column containing time information
        fault_column (str): Name of the column containing fault information
        fault_value (str): Value in fault_column that indicates a fault
        operating_hours (float): Total operating hours
        
    Returns:
        dict: Dictionary of reliability metrics
    """
    # Count failures
    if fault_value is not None:
        num_failures = fault_data[fault_data[fault_column] == fault_value].shape[0]
    else:
        num_failures = fault_data.shape[0]
    
    # Calculate failure rate
    failure_rate = calculate_failure_rate(num_failures, operating_hours)
    
    # Calculate MTTF
    mttf = calculate_mttf(failure_rate)
    
    # Calculate reliability at different time points
    reliability_1h = calculate_reliability(failure_rate, 1)
    reliability_24h = calculate_reliability(failure_rate, 24)
    reliability_168h = calculate_reliability(failure_rate, 168)  # 1 week
    reliability_720h = calculate_reliability(failure_rate, 720)  # 30 days
    reliability_8760h = calculate_reliability(failure_rate, 8760)  # 1 year
    
    # Return metrics
    return {
        'num_failures': num_failures,
        'operating_hours': operating_hours,
        'failure_rate': failure_rate,
        'mttf': mttf,
        'reliability': reliability_24h,  # Default to 24-hour reliability
        'reliability_1h': reliability_1h,
        'reliability_24h': reliability_24h,
        'reliability_168h': reliability_168h,
        'reliability_720h': reliability_720h,
        'reliability_8760h': reliability_8760h
    }

def generate_reliability_report(reliability_results, fault_type=None):
    """
    Generate a text report of reliability analysis results.
    
    Args:
        reliability_results (dict): Dictionary of reliability metrics
        fault_type (str): Type of fault being analyzed
        
    Returns:
        str: Text report
    """
    report = []
    report.append("=" * 50)
    report.append(f"RELIABILITY ANALYSIS REPORT")
    if fault_type:
        report.append(f"Fault Type: {fault_type}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 50)
    report.append("")
    
    report.append("SUMMARY METRICS:")
    report.append(f"Number of Failures: {reliability_results['num_failures']}")
    report.append(f"Operating Hours: {reliability_results['operating_hours']}")
    report.append(f"Failure Rate (λ): {reliability_results['failure_rate']:.6f} failures/hour")
    report.append(f"Mean Time To Failure (MTTF): {reliability_results['mttf']:.2f} hours")
    report.append("")
    
    report.append("RELIABILITY ESTIMATES:")
    report.append(f"Reliability (1 hour): {reliability_results['reliability_1h']:.6f}")
    report.append(f"Reliability (24 hours): {reliability_results['reliability_24h']:.6f}")
    report.append(f"Reliability (1 week): {reliability_results['reliability_168h']:.6f}")
    report.append(f"Reliability (30 days): {reliability_results['reliability_720h']:.6f}")
    report.append(f"Reliability (1 year): {reliability_results['reliability_8760h']:.6f}")
    report.append("")
    
    report.append("INTERPRETATION:")
    if reliability_results['mttf'] > 8760:
        report.append("The system shows excellent reliability with MTTF exceeding one year.")
    elif reliability_results['mttf'] > 720:
        report.append("The system shows good reliability with MTTF exceeding one month.")
    elif reliability_results['mttf'] > 168:
        report.append("The system shows moderate reliability with MTTF exceeding one week.")
    else:
        report.append("The system shows poor reliability with MTTF less than one week.")
    
    if reliability_results['reliability_720h'] > 0.95:
        report.append("30-day reliability is excellent (>95%).")
    elif reliability_results['reliability_720h'] > 0.9:
        report.append("30-day reliability is good (>90%).")
    elif reliability_results['reliability_720h'] > 0.8:
        report.append("30-day reliability is acceptable (>80%).")
    else:
        report.append("30-day reliability is poor (<80%).")
    
    return "\n".join(report)

def plot_reliability_curve(reliability_results, time_range=(0, 8760, 100), title=None, save_path=None):
    """
    Plot the reliability curve.
    
    Args:
        reliability_results (dict): Dictionary of reliability metrics
        time_range (tuple): (start, end, num_points) for time axis
        title (str): Plot title
        save_path (str): Path to save the plot
        
    Returns:
        None
    """
    failure_rate = reliability_results['failure_rate']
    
    # Generate time points
    start_time, end_time, num_points = time_range
    time_points = np.linspace(start_time, end_time, num_points)
    
    # Calculate reliability at each time point
    reliability_values = np.exp(-failure_rate * time_points)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, reliability_values, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (hours)')
    plt.ylabel('Reliability')
    plt.title(title or 'Reliability Curve')
    
    # Add MTTF line
    mttf = reliability_results['mttf']
    if mttf < end_time:
        plt.axvline(x=mttf, color='r', linestyle='--', alpha=0.7)
        plt.text(mttf * 1.05, 0.5, f'MTTF = {mttf:.2f} hours', color='r')
    
    # Add reliability at specific points
    plt.axhline(y=0.9, color='g', linestyle=':', alpha=0.7)
    plt.axhline(y=0.8, color='y', linestyle=':', alpha=0.7)
    plt.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7)
    
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig('reliability_curve.png')
    
    plt.close()

def calculate_dynamic_reliability(time_points, mttf):
    """
    Calculate reliability over time using the exponential model.
    
    Args:
        time_points (np.array): Array of time points
        mttf (float): Mean Time To Failure
        
    Returns:
        np.array: Reliability values at each time point
    """
    if mttf <= 0:
        return np.ones_like(time_points)
    
    failure_rate = 1.0 / mttf
    return np.exp(-failure_rate * time_points)

def weibull_reliability(time_points, shape, scale):
    """
    Calculate reliability using the Weibull distribution.
    
    Args:
        time_points (np.array): Array of time points
        shape (float): Weibull shape parameter (beta)
        scale (float): Weibull scale parameter (eta)
        
    Returns:
        np.array: Reliability values at each time point
    """
    return np.exp(-((time_points / scale) ** shape))

def estimate_weibull_parameters(failure_times):
    """
    Estimate Weibull parameters from failure times.
    
    Args:
        failure_times (list): List of failure times
        
    Returns:
        tuple: (shape, scale) parameters
    """
    # Simple estimation method
    if not failure_times or len(failure_times) < 2:
        return 1.0, 1000.0  # Default values
    
    # Sort failure times
    failure_times = sorted(failure_times)
    
    # Calculate mean and standard deviation
    mean = np.mean(failure_times)
    std = np.std(failure_times)
    
    # Estimate shape parameter (beta)
    if std == 0:
        shape = 1.0  # Default to exponential
    else:
        shape = (mean / std) ** 1.2
    
    # Estimate scale parameter (eta)
    scale = mean / np.exp(np.log(0.5) / shape)
    
    return shape, scale
