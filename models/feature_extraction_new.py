"""
Advanced Feature Extraction for NPP Fault Monitoring.

This module implements advanced feature extraction methods as described in the research papers:
1. "Intelligent Fault Monitoring and Reliability Analysis in Safety-Critical Systems of 
   Nuclear Power Plants Using SIAO-CNN-ORNN"
2. "Advanced Online Fault Monitoring in Nuclear Power Plants"

It includes:
- Statistical feature extraction (time-domain and frequency-domain)
- Weighted Kurtosis and Skewness (WKS) features with SIAO optimization
- Frequency domain features using FFT and spectral analysis
- Time-frequency domain features using wavelet transforms
- Comprehensive feature extraction pipeline with optimization
- Adaptive feature selection based on signal characteristics
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy import signal
from scipy.fftpack import fft, ifft
import pywt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import SIAO optimizer for WKS optimization
try:
    from siao_optimizer import optimize_wks_weights
    SIAO_AVAILABLE = True
except ImportError:
    SIAO_AVAILABLE = False
    warnings.warn("SIAO optimizer not available. Using default weights for WKS.")


def extract_statistical_features(data, window_size=100, step_size=None, time_col='time000000000'):
    """
    Extract statistical features over a sliding window of the data.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Size of the sliding window
        step_size (int, optional): Step size for the sliding window. If None, uses window_size (non-overlapping)
        time_col (str): Name of the time column to exclude from features
        
    Returns:
        tuple: (features_df, labels)
    """
    if step_size is None:
        step_size = window_size
    
    features = []
    labels = []
    
    # Select numerical columns, excluding time column
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if time_col in numerical_cols:
        numerical_cols = numerical_cols.drop(time_col)
    
    # Handle NaN values in the input data
    data_clean = data.copy()
    
    # Check for NaN values and warn if found
    nan_cols = data_clean[numerical_cols].columns[data_clean[numerical_cols].isna().any()].tolist()
    if nan_cols:
        warnings.warn(f"NaN values found in columns: {nan_cols}. These will be imputed.")
        
    # Impute NaN values using forward fill, backward fill, and then mean imputation for any remaining NaNs
    data_clean[numerical_cols] = data_clean[numerical_cols].ffill().bfill()
    
    # Use SimpleImputer for any remaining NaNs (with mean strategy)
    imputer = SimpleImputer(strategy='mean')
    data_clean[numerical_cols] = pd.DataFrame(
        imputer.fit_transform(data_clean[numerical_cols]),
        columns=numerical_cols,
        index=data_clean.index
    )
    
    # Calculate features for each window
    for i in tqdm(range(0, len(data_clean) - window_size + 1, step_size), desc="Extracting features"):
        window = data_clean[numerical_cols].iloc[i:i + window_size]
        feature_vector = {}
        
        for col in numerical_cols:
            try:
                # Basic statistical features
                feature_vector[f'{col}_mean'] = np.mean(window[col])
                feature_vector[f'{col}_median'] = np.median(window[col])
                feature_vector[f'{col}_std'] = np.std(window[col])
                feature_vector[f'{col}_variance'] = np.var(window[col])
                feature_vector[f'{col}_min'] = np.min(window[col])
                feature_vector[f'{col}_max'] = np.max(window[col])
                feature_vector[f'{col}_range'] = np.max(window[col]) - np.min(window[col])
                
                # Higher-order statistics with error handling
                try:
                    feature_vector[f'{col}_kurtosis'] = kurtosis(window[col])
                except Exception as e:
                    warnings.warn(f"Error calculating kurtosis for {col}: {e}. Using 0 instead.")
                    feature_vector[f'{col}_kurtosis'] = 0
                    
                try:
                    feature_vector[f'{col}_skewness'] = skew(window[col])
                except Exception as e:
                    warnings.warn(f"Error calculating skewness for {col}: {e}. Using 0 instead.")
                    feature_vector[f'{col}_skewness'] = 0
                
                # Information theory features
                try:
                    hist, _ = np.histogram(window[col], bins=10)
                    feature_vector[f'{col}_entropy'] = entropy(hist) if np.sum(hist) > 0 else 0
                except Exception as e:
                    warnings.warn(f"Error calculating entropy for {col}: {e}. Using 0 instead.")
                    feature_vector[f'{col}_entropy'] = 0
                
                # Frequency domain features
                try:
                    # Calculate power spectral density
                    f, psd = signal.welch(window[col], fs=1.0, nperseg=min(256, len(window[col])))
                    
                    # Extract frequency domain features
                    feature_vector[f'{col}_psd_mean'] = np.mean(psd)
                    feature_vector[f'{col}_psd_max'] = np.max(psd)
                    feature_vector[f'{col}_psd_std'] = np.std(psd)
                    
                    # Calculate dominant frequency
                    dominant_freq_idx = np.argmax(psd)
                    feature_vector[f'{col}_dominant_freq'] = f[dominant_freq_idx] if dominant_freq_idx < len(f) else 0
                except Exception as e:
                    warnings.warn(f"Error calculating frequency domain features for {col}: {e}. Using 0 instead.")
                    feature_vector[f'{col}_psd_mean'] = 0
                    feature_vector[f'{col}_psd_max'] = 0
                    feature_vector[f'{col}_psd_std'] = 0
                    feature_vector[f'{col}_dominant_freq'] = 0
            
            except Exception as e:
                warnings.warn(f"Error processing column {col}: {e}. Using zeros for all features.")
                # Set all features for this column to zero
                feature_vector[f'{col}_mean'] = 0
                feature_vector[f'{col}_median'] = 0
                feature_vector[f'{col}_std'] = 0
                feature_vector[f'{col}_variance'] = 0
                feature_vector[f'{col}_min'] = 0
                feature_vector[f'{col}_max'] = 0
                feature_vector[f'{col}_range'] = 0
                feature_vector[f'{col}_kurtosis'] = 0
                feature_vector[f'{col}_skewness'] = 0
                feature_vector[f'{col}_entropy'] = 0
                feature_vector[f'{col}_psd_mean'] = 0
                feature_vector[f'{col}_psd_max'] = 0
                feature_vector[f'{col}_psd_std'] = 0
                feature_vector[f'{col}_dominant_freq'] = 0
        
        # Add feature vector to features list
        features.append(feature_vector)
        
        # Extract label if 'condition' column exists
        if 'condition' in data_clean.columns:
            # Use the most frequent condition in the window
            label = data_clean['condition'].iloc[i:i + window_size].mode().iloc[0]
            labels.append(label)
        else:
            # If no condition column, use a default label
            labels.append('normal')
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features)
    
    return features_df, labels


def weighted_kurtosis_skewness(data, omega=None):
    """
    Calculate weighted kurtosis and skewness (WKS) for a data series.
    
    Args:
        data (np.array): Input data array
        omega (float, optional): Weight parameter for kurtosis. If None, uses default value.
        
    Returns:
        float: Weighted kurtosis and skewness value
    """
    try:
        # Handle NaN values in the input data
        data_clean = np.copy(data)
        mask = np.isnan(data_clean)
        if np.any(mask):
            data_clean[mask] = np.nanmean(data_clean)
            data = data_clean
            warnings.warn("NaN values found in WKS calculation. Replaced with mean.")
            
        # Calculate kurtosis and skewness
        k = kurtosis(data)
        s = skew(data)
        
        # Handle potential NaN values
        if np.isnan(k) or np.isnan(s):
            return 0.0
        
        # If omega is not provided, use default value
        if omega is None:
            omega = 0.5  # Default balanced weight
        
        # Calculate weighted kurtosis and skewness
        wks = omega * k + (1 - omega) * s
        
        return wks
    except Exception as e:
        warnings.warn(f"Error in WKS calculation: {e}. Returning 0.")
        return 0.0


def optimize_wks_parameter(data, max_iter=50, pop_size=20, window_size=100):
    """
    Find optimal weight parameter for WKS using optimization.
    
    Args:
        data (np.array): Input data array
        max_iter (int): Maximum number of iterations
        pop_size (int): Population size
        window_size (int): Window size for WKS calculation
        
    Returns:
        float: Optimized omega parameter
    """
    if SIAO_AVAILABLE:
        # Use the imported SIAO optimizer
        try:
            return optimize_wks_weights(data, window_size, max_iter, pop_size)
        except Exception as e:
            warnings.warn(f"Error using SIAO optimizer: {e}. Using default approach.")
    
    # Fallback to a simpler optimization approach
    best_omega = 0.5
    best_wks = evaluate_wks_effectiveness(data, best_omega, window_size)
    
    # Try a few values and pick the best one
    for omega in np.linspace(0.1, 0.9, 9):
        wks = evaluate_wks_effectiveness(data, omega, window_size)
        if wks > best_wks:
            best_wks = wks
            best_omega = omega
    
    return best_omega


def evaluate_wks_effectiveness(data, omega, window_size):
    """
    Evaluate the effectiveness of an omega value for WKS calculation.
    
    Args:
        data (np.array): Input data array
        omega (float): Weight parameter to evaluate
        window_size (int): Window size for WKS calculation
        
    Returns:
        float: Mean WKS value across all windows
    """
    wks_values = []
    
    # Calculate WKS for each window
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size]
        try:
            wks = weighted_kurtosis_skewness(window, omega)
            wks_values.append(wks)
        except Exception as e:
            pass  # Skip windows that cause errors
    
    # Return mean WKS (or 0 if no valid windows)
    return np.mean(wks_values) if wks_values else 0


def extract_wks_features(data, window_size=100, step_size=None, time_col='time000000000', optimize_weights=True):
    """
    Extract Weighted Kurtosis and Skewness (WKS) features from data.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Size of the sliding window
        step_size (int, optional): Step size for the sliding window
        time_col (str): Name of the time column
        optimize_weights (bool): Whether to optimize the omega parameter for each column
        
    Returns:
        pd.DataFrame: DataFrame with WKS features
    """
    if step_size is None:
        step_size = window_size
    
    # Create a copy of the data
    data_copy = data.copy()
    
    # Get numerical columns (excluding time column)
    numerical_cols = [col for col in data_copy.columns if col != time_col and pd.api.types.is_numeric_dtype(data_copy[col])]
    
    # Initialize features DataFrame
    features = []
    timestamps = []
    
    # Optimize omega parameters for each column if requested
    optimal_omegas = {}
    if optimize_weights and SIAO_AVAILABLE:
        print("Optimizing WKS weights for each column...")
        for col in tqdm(numerical_cols, desc="Optimizing WKS weights"):
            try:
                # Use the first window to optimize omega
                first_window = data_copy[col].iloc[:min(len(data_copy), window_size * 5)].values
                optimal_omegas[col] = optimize_wks_parameter(first_window, window_size=window_size)
                print(f"Optimized omega for {col}: {optimal_omegas[col]:.4f}")
            except Exception as e:
                optimal_omegas[col] = 0.5  # Default value
                print(f"Error optimizing omega for {col}: {e}. Using default value.")
    
    # Extract WKS features using sliding window
    for i in range(0, len(data_copy) - window_size + 1, step_size):
        # Get window
        window = data_copy.iloc[i:i + window_size]
        
        # Extract timestamp (use the last timestamp in the window)
        timestamp = window[time_col].iloc[-1] if time_col in window.columns else i
        timestamps.append(timestamp)
        
        # Calculate WKS for each numerical column
        window_features = {}
        for col in numerical_cols:
            # Use optimized omega if available, otherwise use default
            omega = optimal_omegas.get(col, 0.5) if optimize_weights else 0.5
            try:
                wks = weighted_kurtosis_skewness(window[col].values, omega)
                window_features[f"{col}_wks"] = wks
            except Exception as e:
                window_features[f"{col}_wks"] = 0.0
        
        features.append(window_features)
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features)
    
    # Add timestamps
    features_df[time_col] = timestamps
    
    return features_df


def extract_all_features(data, window_size=100, step_size=None, time_col='time000000000', 
                      include_wks=False, optimize_wks=True, include_frequency=False, 
                      include_wavelet=False, wavelet_type='db4', wavelet_level=3,
                      select_best=False, n_best_features=None):
    """
    Extract all features from the data including time-domain, frequency-domain, and time-frequency domain features.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Size of the window
        step_size (int, optional): Step size for the sliding window
        time_col (str): Name of the time column to exclude
        include_wks (bool): Whether to include WKS features
        optimize_wks (bool): Whether to optimize WKS weights
        include_frequency (bool): Whether to include frequency domain features
        include_wavelet (bool): Whether to include wavelet transform features
        wavelet_type (str): Type of wavelet to use for wavelet transform
        wavelet_level (int): Decomposition level for wavelet transform
        select_best (bool): Whether to select the best features
        n_best_features (int): Number of best features to select if select_best is True
        
    Returns:
        tuple: (features_df, labels, encoder)
    """
    # Check for NaN values in the input data
    nan_count = data.isna().sum().sum()
    if nan_count > 0:
        warnings.warn(f"Input data contains {nan_count} NaN values. These will be handled during feature extraction.")
    
    # Extract statistical features (time domain)
    print("Extracting statistical features...")
    stat_features, labels = extract_statistical_features(
        data, window_size=window_size, step_size=step_size, time_col=time_col
    )
    
    # Initialize all_features with statistical features
    all_features = stat_features
    min_len = len(stat_features)
    
    # Extract WKS features if requested
    if include_wks:
        print("Extracting WKS features (this may take a while)...")
        try:
            wks_features = extract_wks_features(
                data, window_size=window_size, step_size=step_size, 
                time_col=time_col, optimize_weights=optimize_wks
            )
            
            # Ensure both feature sets have the same length
            min_len = min(min_len, len(wks_features))
            if min_len == 0:
                raise ValueError("No features were extracted. Check your input data and parameters.")
                
            # Combine features
            all_features = pd.concat([all_features.iloc[:min_len], wks_features.iloc[:min_len]], axis=1)
        except Exception as e:
            warnings.warn(f"Error extracting WKS features: {e}. Using only previously extracted features.")
    
    # Extract frequency domain features if requested
    if include_frequency:
        print("Extracting frequency domain features...")
        try:
            freq_features = extract_frequency_domain_features(
                data, window_size=window_size, step_size=step_size, time_col=time_col
            )
            
            # Ensure both feature sets have the same length
            min_len = min(min_len, len(freq_features))
            if min_len == 0:
                raise ValueError("No frequency features were extracted. Check your input data and parameters.")
                
            # Combine features
            all_features = pd.concat([all_features.iloc[:min_len], freq_features.iloc[:min_len]], axis=1)
        except Exception as e:
            warnings.warn(f"Error extracting frequency domain features: {e}. Using only previously extracted features.")
    
    # Extract wavelet features if requested
    if include_wavelet:
        print(f"Extracting wavelet features using {wavelet_type} wavelet at level {wavelet_level}...")
        try:
            wavelet_features = extract_wavelet_features(
                data, window_size=window_size, step_size=step_size, 
                time_col=time_col, wavelet=wavelet_type, level=wavelet_level
            )
            
            # Ensure both feature sets have the same length
            min_len = min(min_len, len(wavelet_features))
            if min_len == 0:
                raise ValueError("No wavelet features were extracted. Check your input data and parameters.")
                
            # Combine features
            all_features = pd.concat([all_features.iloc[:min_len], wavelet_features.iloc[:min_len]], axis=1)
        except Exception as e:
            warnings.warn(f"Error extracting wavelet features: {e}. Using only previously extracted features.")
    
    # Adjust labels to match feature length
    labels = labels[:min_len]
    
    # Remove duplicate columns if any
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    # Check for NaN values in the extracted features
    nan_features = all_features.columns[all_features.isna().any()].tolist()
    if nan_features:
        warnings.warn(f"NaN values found in {len(nan_features)} extracted features. These will be imputed.")
        # Impute NaN values in features
        imputer = SimpleImputer(strategy='mean')
        all_features = pd.DataFrame(
            imputer.fit_transform(all_features),
            columns=all_features.columns
        )
    
    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    y_one_hot = pd.get_dummies(y).values  # One-hot encoding
    
    # Select best features if requested
    if select_best and n_best_features is not None and n_best_features < all_features.shape[1]:
        print(f"Selecting top {n_best_features} features out of {all_features.shape[1]} total features...")
        try:
            selected_features, feature_scores = select_best_features(all_features, y, k=n_best_features)
            print("Top 10 selected features:")
            for i, (feature, score) in enumerate(zip(feature_scores['Feature'].iloc[:10], feature_scores['Score'].iloc[:10])):
                print(f"  {i+1}. {feature}: {score:.4f}")
            all_features = selected_features
        except Exception as e:
            warnings.warn(f"Error selecting best features: {e}. Using all features.")
    
    print(f"Extracted {all_features.shape[1]} features with no NaN values from {len(data)} data points")
    print(f"Feature matrix shape: {all_features.shape}")
    print(f"Target matrix shape: {y_one_hot.shape}")
    
    return all_features, y_one_hot, encoder


def extract_frequency_domain_features(data, window_size=100, step_size=None, time_col='time000000000'):
    """
    Extract frequency domain features using FFT over a sliding window of the data.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Size of the sliding window
        step_size (int, optional): Step size for the sliding window. If None, uses window_size
        time_col (str): Name of the time column to exclude from features
        
    Returns:
        pd.DataFrame: DataFrame with frequency domain features
    """
    if step_size is None:
        step_size = window_size
    
    features = []
    timestamps = []
    
    # Select numerical columns, excluding time column
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if time_col in numerical_cols:
        numerical_cols = numerical_cols.drop(time_col)
    
    # Handle NaN values in the input data
    data_clean = data.copy()
    data_clean[numerical_cols] = data_clean[numerical_cols].ffill().bfill()
    imputer = SimpleImputer(strategy='mean')
    data_clean[numerical_cols] = pd.DataFrame(
        imputer.fit_transform(data_clean[numerical_cols]),
        columns=numerical_cols,
        index=data_clean.index
    )
    
    # Calculate features for each window
    for i in tqdm(range(0, len(data_clean) - window_size + 1, step_size), desc="Extracting frequency features"):
        window = data_clean[numerical_cols].iloc[i:i + window_size]
        feature_vector = {}
        
        # Store timestamp (middle of window)
        if time_col in data_clean.columns:
            timestamps.append(data_clean[time_col].iloc[i + window_size // 2])
        else:
            timestamps.append(i + window_size // 2)  # Use index as timestamp
        
        for col in numerical_cols:
            try:
                # Apply FFT
                signal_fft = fft(window[col].values)
                # Get magnitude spectrum
                magnitude = np.abs(signal_fft)
                # Get power spectrum
                power = magnitude ** 2
                # Get phase spectrum
                phase = np.angle(signal_fft)
                
                # Extract frequency domain features
                feature_vector[f'{col}_fft_mean'] = np.mean(magnitude)
                feature_vector[f'{col}_fft_std'] = np.std(magnitude)
                feature_vector[f'{col}_fft_max'] = np.max(magnitude)
                feature_vector[f'{col}_fft_power_mean'] = np.mean(power)
                feature_vector[f'{col}_fft_power_std'] = np.std(power)
                
                # Dominant frequency features
                dominant_freq_idx = np.argmax(magnitude[1:window_size//2]) + 1  # Skip DC component
                feature_vector[f'{col}_dominant_freq'] = dominant_freq_idx / window_size
                feature_vector[f'{col}_dominant_magnitude'] = magnitude[dominant_freq_idx]
                
                # Spectral energy and entropy
                spectral_energy = np.sum(power) / window_size
                feature_vector[f'{col}_spectral_energy'] = spectral_energy
                
                # Spectral entropy
                normalized_power = power / np.sum(power)
                spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
                feature_vector[f'{col}_spectral_entropy'] = spectral_entropy
                
                # Spectral centroid
                freqs = np.fft.fftfreq(window_size)
                feature_vector[f'{col}_spectral_centroid'] = np.sum(freqs * power) / (np.sum(power) + 1e-10)
                
            except Exception as e:
                warnings.warn(f"Error extracting frequency features for {col}: {e}. Using zeros.")
                feature_vector[f'{col}_fft_mean'] = 0
                feature_vector[f'{col}_fft_std'] = 0
                feature_vector[f'{col}_fft_max'] = 0
                feature_vector[f'{col}_fft_power_mean'] = 0
                feature_vector[f'{col}_fft_power_std'] = 0
                feature_vector[f'{col}_dominant_freq'] = 0
                feature_vector[f'{col}_dominant_magnitude'] = 0
                feature_vector[f'{col}_spectral_energy'] = 0
                feature_vector[f'{col}_spectral_entropy'] = 0
                feature_vector[f'{col}_spectral_centroid'] = 0
        
        features.append(feature_vector)
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features)
    
    # Add timestamps
    features_df[time_col] = timestamps
    
    return features_df


def extract_wavelet_features(data, window_size=100, step_size=None, time_col='time000000000', wavelet='db4', level=3):
    """
    Extract time-frequency domain features using wavelet transform over a sliding window of the data.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Size of the sliding window
        step_size (int, optional): Step size for the sliding window. If None, uses window_size
        time_col (str): Name of the time column to exclude from features
        wavelet (str): Wavelet type to use (e.g., 'db4', 'haar', 'sym4')
        level (int): Decomposition level for wavelet transform
        
    Returns:
        pd.DataFrame: DataFrame with wavelet features
    """
    if step_size is None:
        step_size = window_size
    
    features = []
    timestamps = []
    
    # Select numerical columns, excluding time column
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if time_col in numerical_cols:
        numerical_cols = numerical_cols.drop(time_col)
    
    # Handle NaN values in the input data
    data_clean = data.copy()
    data_clean[numerical_cols] = data_clean[numerical_cols].ffill().bfill()
    imputer = SimpleImputer(strategy='mean')
    data_clean[numerical_cols] = pd.DataFrame(
        imputer.fit_transform(data_clean[numerical_cols]),
        columns=numerical_cols,
        index=data_clean.index
    )
    
    # Calculate features for each window
    for i in tqdm(range(0, len(data_clean) - window_size + 1, step_size), desc="Extracting wavelet features"):
        window = data_clean[numerical_cols].iloc[i:i + window_size]
        feature_vector = {}
        
        # Store timestamp (middle of window)
        if time_col in data_clean.columns:
            timestamps.append(data_clean[time_col].iloc[i + window_size // 2])
        else:
            timestamps.append(i + window_size // 2)  # Use index as timestamp
        
        for col in numerical_cols:
            try:
                # Apply wavelet transform
                coeffs = pywt.wavedec(window[col].values, wavelet, level=level)
                
                # Extract features from each decomposition level
                for j, coef in enumerate(coeffs):
                    if j == 0:
                        # Approximation coefficients
                        feature_vector[f'{col}_wavelet_a{level}_mean'] = np.mean(coef)
                        feature_vector[f'{col}_wavelet_a{level}_std'] = np.std(coef)
                        feature_vector[f'{col}_wavelet_a{level}_energy'] = np.sum(coef**2) / len(coef)
                        feature_vector[f'{col}_wavelet_a{level}_kurtosis'] = kurtosis(coef) if len(coef) > 3 else 0
                        feature_vector[f'{col}_wavelet_a{level}_skewness'] = skew(coef) if len(coef) > 3 else 0
                    else:
                        # Detail coefficients
                        detail_level = level - j + 1
                        feature_vector[f'{col}_wavelet_d{detail_level}_mean'] = np.mean(coef)
                        feature_vector[f'{col}_wavelet_d{detail_level}_std'] = np.std(coef)
                        feature_vector[f'{col}_wavelet_d{detail_level}_energy'] = np.sum(coef**2) / len(coef)
                        feature_vector[f'{col}_wavelet_d{detail_level}_kurtosis'] = kurtosis(coef) if len(coef) > 3 else 0
                        feature_vector[f'{col}_wavelet_d{detail_level}_skewness'] = skew(coef) if len(coef) > 3 else 0
                
            except Exception as e:
                warnings.warn(f"Error extracting wavelet features for {col}: {e}. Using zeros.")
                # Add zero features for this column
                feature_vector[f'{col}_wavelet_a{level}_mean'] = 0
                feature_vector[f'{col}_wavelet_a{level}_std'] = 0
                feature_vector[f'{col}_wavelet_a{level}_energy'] = 0
                feature_vector[f'{col}_wavelet_a{level}_kurtosis'] = 0
                feature_vector[f'{col}_wavelet_a{level}_skewness'] = 0
                
                for detail_level in range(1, level + 1):
                    feature_vector[f'{col}_wavelet_d{detail_level}_mean'] = 0
                    feature_vector[f'{col}_wavelet_d{detail_level}_std'] = 0
                    feature_vector[f'{col}_wavelet_d{detail_level}_energy'] = 0
                    feature_vector[f'{col}_wavelet_d{detail_level}_kurtosis'] = 0
                    feature_vector[f'{col}_wavelet_d{detail_level}_skewness'] = 0
        
        features.append(feature_vector)
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features)
    
    # Add timestamps
    features_df[time_col] = timestamps
    
    return features_df


def select_best_features(X, y, k=20):
    """
    Select the k best features based on ANOVA F-value.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (np.array): Target labels
        k (int): Number of features to select
        
    Returns:
        tuple: (selected_features_df, feature_scores)
    """
    # Create feature selector
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit and transform the data
    X_new = selector.fit_transform(X, y)
    
    # Get selected feature names and scores
    feature_names = X.columns
    feature_scores = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_,
        'P-value': selector.pvalues_,
        'Selected': selector.get_support()
    })
    
    # Sort by score
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    # Get selected feature names
    selected_features = feature_names[selector.get_support()]
    
    # Return selected features and scores
    return X[selected_features], feature_scores


# Example usage
if __name__ == "__main__":
    from data_preprocessing import process_pipeline
    import argparse
    import time
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced Feature Extraction for NPP Fault Monitoring')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for feature extraction')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for sliding window')
    parser.add_argument('--include_wks', action='store_true', help='Include WKS features')
    parser.add_argument('--optimize_wks', action='store_true', help='Optimize WKS weights using SIAO')
    parser.add_argument('--include_frequency', action='store_true', help='Include frequency domain features')
    parser.add_argument('--include_wavelet', action='store_true', help='Include wavelet transform features')
    parser.add_argument('--wavelet_type', type=str, default='db4', help='Wavelet type (db4, haar, sym4, etc.)')
    parser.add_argument('--wavelet_level', type=int, default=3, help='Wavelet decomposition level')
    parser.add_argument('--select_best', action='store_true', help='Select best features')
    parser.add_argument('--n_best_features', type=int, default=100, help='Number of best features to select')
    parser.add_argument('--output_file', type=str, default='data/processed/extracted_features.csv', 
                        help='Output file for extracted features')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Get preprocessed data
    print("Loading and preprocessing data...")
    processed_data, scaler = process_pipeline()
    
    # Start timer
    start_time = time.time()
    
    # Extract features with all options
    print("\nExtracting features with the following options:")
    print(f"  Window size: {args.window_size}")
    print(f"  Step size: {args.step_size}")
    print(f"  Include WKS features: {args.include_wks}")
    print(f"  Optimize WKS weights: {args.optimize_wks}")
    print(f"  Include frequency domain features: {args.include_frequency}")
    print(f"  Include wavelet features: {args.include_wavelet}")
    if args.include_wavelet:
        print(f"  Wavelet type: {args.wavelet_type}")
        print(f"  Wavelet level: {args.wavelet_level}")
    print(f"  Select best features: {args.select_best}")
    if args.select_best:
        print(f"  Number of best features: {args.n_best_features}")
    
    # Extract features
    features, y, encoder = extract_all_features(
        processed_data, 
        window_size=args.window_size,
        step_size=args.step_size,
        include_wks=args.include_wks,
        optimize_wks=args.optimize_wks,
        include_frequency=args.include_frequency,
        include_wavelet=args.include_wavelet,
        wavelet_type=args.wavelet_type,
        wavelet_level=args.wavelet_level,
        select_best=args.select_best,
        n_best_features=args.n_best_features
    )
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nFeature extraction completed in {elapsed_time:.2f} seconds")
    
    # Print class distribution
    class_names = encoder.classes_
    class_counts = np.bincount(encoder.transform(processed_data['fault_type'] if 'fault_type' in processed_data.columns else processed_data['condition']))
    print("\nClass distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {name}: {count} samples")
    
    # Save features
    print(f"\nSaving features to {args.output_file}")
    features.to_csv(args.output_file, index=False)
    
    print("\nFeature extraction complete!")
    print("To use these features for model training, run:")
    print(f"  python tune_hyperparameters.py --use_advanced_features --include_wks --model cnn_lstm")
    
    # Example of how to visualize feature importance
    if args.select_best:
        try:
            import matplotlib.pyplot as plt
            
            # Select top features for visualization
            _, feature_scores = select_best_features(features, encoder.transform(processed_data['fault_type'] 
                                                                               if 'fault_type' in processed_data.columns 
                                                                               else processed_data['condition']))
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = feature_scores.head(20)
            plt.barh(top_features['Feature'], top_features['Score'])
            plt.xlabel('F-score')
            plt.ylabel('Feature')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.savefig('data/processed/feature_importance.png')
            print(f"Feature importance plot saved to data/processed/feature_importance.png")
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")

