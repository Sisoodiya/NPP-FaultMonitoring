"""
Dynamic Reliability Analysis for NPP Fault Monitoring.

This module implements the dynamic reliability analysis methods described in the research paper:
"Intelligent Fault Monitoring and Reliability Analysis in Safety-Critical Systems of 
Nuclear Power Plants Using SIAO-CNN-ORNN"

It includes functions for calculating:
- Failure rates
- Mean Time To Failure (MTTF)
- Dynamic reliability using e^(-T/MTTF) formula
- Reliability curves over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import os

def calculate_failure_rate(failure_count, operating_time):
    """
    Calculate failure rate based on number of failures and operating time.
    
    Args:
        failure_count: Number of failures observed
        operating_time: Total operating time
        
    Returns:
        float: Failure rate (failures per unit time)
    """
    if operating_time <= 0:
        return np.nan
    
    return failure_count / operating_time

def calculate_mttf(failure_rate):
    """
    Calculate Mean Time To Failure (MTTF) from failure rate.
    
    Args:
        failure_rate: Failure rate (failures per unit time)
        
    Returns:
        float: MTTF (mean time to failure)
    """
    if failure_rate <= 0:
        return np.inf
    
    return 1.0 / failure_rate

def calculate_dynamic_reliability(time, mttf):
    """
    Calculate dynamic reliability using e^(-T/MTTF) formula.
    
    Args:
        time: Time point for reliability calculation
        mttf: Mean Time To Failure
        
    Returns:
        float: Reliability at the specified time
    """
    if mttf <= 0:
        return 0.0
    
    return np.exp(-time / mttf)

def calculate_reliability_curve(mttf, time_points=None):
    """
    Calculate reliability curve over a range of time points.
    
    Args:
        mttf: Mean Time To Failure
        time_points: Array of time points for calculation (if None, generates 100 points from 0 to 3*MTTF)
        
    Returns:
        tuple: (time_points, reliability_values)
    """
    if time_points is None:
        # Generate time points from 0 to 3*MTTF (or 100 if MTTF is infinite)
        max_time = 100 if np.isinf(mttf) else 3 * mttf
        time_points = np.linspace(0, max_time, 100)
    
    reliability_values = np.array([calculate_dynamic_reliability(t, mttf) for t in time_points])
    
    return time_points, reliability_values

def extract_fault_events(data, fault_column, fault_value):
    """
    Extract fault events from data.
    
    Args:
        data: DataFrame containing fault data
        fault_column: Column name containing fault information
        fault_value: Value indicating the fault of interest
        
    Returns:
        DataFrame: Filtered data containing only the specified fault events
    """
    return data[data[fault_column] == fault_value]

def analyze_fault_sequence(fault_data, time_col='time000000000'):
    """
    Analyze sequence of fault occurrences to calculate failure statistics.
    
    Args:
        fault_data: DataFrame containing fault events
        time_col: Column name containing time information
        
    Returns:
        dict: Dictionary containing failure statistics
    """
    if fault_data.empty:
        return {
            'failure_count': 0,
            'total_time': 0,
            'failure_rate': 0,
            'mttf': np.inf,
            'failure_times': [],
            'time_between_failures': []
        }
    
    # Sort by time
    fault_data = fault_data.sort_values(by=time_col)
    
    # Extract time values
    times = fault_data[time_col].values
    
    # Calculate time between failures
    time_between_failures = np.diff(times)
    
    # Calculate total operating time
    total_time = times[-1] - times[0] if len(times) > 1 else 0
    
    # Calculate failure count and rate
    failure_count = len(fault_data)
    failure_rate = calculate_failure_rate(failure_count, total_time) if total_time > 0 else 0
    
    # Calculate MTTF
    mttf = calculate_mttf(failure_rate)
    
    return {
        'failure_count': failure_count,
        'total_time': total_time,
        'failure_rate': failure_rate,
        'mttf': mttf,
        'failure_times': times,
        'time_between_failures': time_between_failures
    }

def fit_weibull_distribution(time_between_failures):
    """
    Fit Weibull distribution to time between failures data.
    
    Args:
        time_between_failures: Array of time intervals between failures
        
    Returns:
        tuple: (shape, scale) parameters of the Weibull distribution
    """
    if len(time_between_failures) < 2:
        return None, None
    
    try:
        # Fit Weibull distribution
        shape, loc, scale = weibull_min.fit(time_between_failures, floc=0)
        return shape, scale
    except:
        return None, None

def analyze_reliability(data, fault_column='predicted', fault_value=None, time_col='time000000000'):
    """
    Perform comprehensive reliability analysis for a specific fault type.
    
    Args:
        data: DataFrame containing fault data
        fault_column: Column name containing fault information
        fault_value: Value indicating the fault of interest
        time_col: Column name containing time information
        
    Returns:
        dict: Dictionary containing reliability analysis results
    """
    # Extract fault events
    fault_data = extract_fault_events(data, fault_column, fault_value) if fault_value else data
    
    # Analyze fault sequence
    fault_stats = analyze_fault_sequence(fault_data, time_col)
    
    # Fit Weibull distribution
    shape, scale = fit_weibull_distribution(fault_stats['time_between_failures'])
    
    # Calculate reliability curve
    time_points, reliability_values = calculate_reliability_curve(fault_stats['mttf'])
    
    # Return comprehensive results
    results = {
        'fault_type': fault_value,
        'failure_count': fault_stats['failure_count'],
        'total_time': fault_stats['total_time'],
        'failure_rate': fault_stats['failure_rate'],
        'mttf': fault_stats['mttf'],
        'weibull_shape': shape,
        'weibull_scale': scale,
        'time_points': time_points,
        'reliability_values': reliability_values
    }
    
    return results

def generate_reliability_report(reliability_results, fault_type=None):
    """
    Generate a text report of reliability analysis results.
    
    Args:
        reliability_results: Dictionary containing reliability analysis results
        fault_type: Optional fault type name for the report title
        
    Returns:
        str: Formatted text report
    """
    title = f"Reliability Analysis Report for {fault_type}" if fault_type else "Reliability Analysis Report"
    
    report = [
        title,
        "=" * len(title),
        "",
        f"Failure Count: {reliability_results['failure_count']}",
        f"Total Operating Time: {reliability_results['total_time']:.2f} units",
        f"Failure Rate: {reliability_results['failure_rate']:.6f} failures per time unit",
        f"Mean Time To Failure (MTTF): {reliability_results['mttf']:.2f} time units",
        ""
    ]
    
    # Add Weibull distribution parameters if available
    if reliability_results['weibull_shape'] is not None:
        report.extend([
            "Weibull Distribution Parameters:",
            f"  Shape (β): {reliability_results['weibull_shape']:.4f}",
            f"  Scale (η): {reliability_results['weibull_scale']:.4f}",
            ""
        ])
    
    # Add reliability values at specific time points
    report.append("Reliability at Key Time Points:")
    
    # Select a few representative time points
    time_indices = [0, 24, 49, 74, 99]  # 0%, 25%, 50%, 75%, 100% of the time range
    for idx in time_indices:
        if idx < len(reliability_results['time_points']):
            t = reliability_results['time_points'][idx]
            r = reliability_results['reliability_values'][idx]
            report.append(f"  Time {t:.2f}: {r:.4f}")
    
    return "\n".join(report)

def plot_reliability_curve(reliability_results, title=None, save_path=None):
    """
    Plot reliability curve from reliability analysis results.
    
    Args:
        reliability_results: Dictionary containing reliability analysis results
        title: Optional title for the plot
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot reliability curve
    ax.plot(
        reliability_results['time_points'],
        reliability_results['reliability_values'],
        'b-',
        linewidth=2
    )
    
    # Add MTTF vertical line
    if not np.isinf(reliability_results['mttf']):
        ax.axvline(
            x=reliability_results['mttf'],
            color='r',
            linestyle='--',
            label=f"MTTF = {reliability_results['mttf']:.2f}"
        )
        
        # Add point at (MTTF, e^-1)
        ax.plot(
            [reliability_results['mttf']],
            [np.exp(-1)],
            'ro',
            markersize=8
        )
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Reliability R(t)')
    
    if title:
        ax.set_title(title)
    else:
        fault_type = reliability_results.get('fault_type', 'Unknown')
        ax.set_title(f"Reliability Curve for {fault_type}")
    
    # Set axis limits
    ax.set_xlim(0, max(reliability_results['time_points']))
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Example usage
if __name__ == "__main__":
    # Generate sample fault data
    np.random.seed(42)
    times = np.sort(np.random.uniform(0, 100, 10))
    fault_data = pd.DataFrame({
        'time000000000': times,
        'predicted': ['fault_A'] * len(times)
    })
    
    # Perform reliability analysis
    results = analyze_reliability(fault_data, fault_value='fault_A')
    
    # Generate report
    report = generate_reliability_report(results, fault_type='fault_A')
    print(report)
    
    # Plot reliability curve
    plot_reliability_curve(results, save_path='reliability_curve.png')
    
    print("Example completed. Check 'reliability_curve.png' for the plot.")
