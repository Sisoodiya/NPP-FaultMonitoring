import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


def load_data(data_path='data', pattern='**/*.xlsx', recursive=True):
    """
    Load data from Excel files in the specified directory and subdirectories matching the pattern.
    
    Args:
        data_path (str): Path to the directory containing data files
        pattern (str): Pattern to match files (default: '**/*.xlsx' to include subdirectories)
        recursive (bool): Whether to search recursively in subdirectories
        
    Returns:
        dict: Dictionary of dataframes with filenames as keys
    """
    data_files = glob.glob(os.path.join(data_path, pattern), recursive=recursive)
    data_dict = {}
    
    for file in data_files:
        filename = os.path.basename(file)
        try:
            df = pd.read_excel(file)
            data_dict[filename] = df
            print(f"Loaded {filename} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return data_dict


def preprocess_data(data, time_col='time000000000'):
    """
    Preprocess the data by handling missing values, normalizing numerical features.
    
    Args:
        data (pd.DataFrame): Input data to preprocess
        time_col (str): Name of the time column
        
    Returns:
        tuple: (preprocessed_data, scaler)
    """
    # Create a copy to avoid modifying the original
    data = data.copy()
    
    # Handle missing values - first with mean for numerical columns
    data = data.fillna(data.mean(numeric_only=True))
    
    # Then use forward and backward fill for any remaining NaNs (using newer syntax)
    data = data.ffill().bfill()
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Normalize numerical features (exclude 'time' and 'condition')
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if time_col in numerical_cols:
        numerical_cols = numerical_cols.drop(time_col)
    if 'condition' in numerical_cols:
        numerical_cols = numerical_cols.drop('condition')
    
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data, scaler


def label_data(data_dict):
    """
    Add condition labels to data based on filename patterns.
    
    Args:
        data_dict (dict): Dictionary of dataframes with filenames as keys
        
    Returns:
        dict: Dictionary of labeled dataframes
    """
    labeled_data = {}
    
    for filename, df in data_dict.items():
        df_copy = df.copy()
        
        # Extract condition from filename
        if 'Steady State' in filename:
            condition = 'steady_state'
        elif 'FeedWater Break' in filename or 'fwb' in filename.lower():
            condition = 'feedwater_break'
        elif 'Pump Failure' in filename or 'pp' in filename.lower():
            condition = 'pump_failure'
        elif 'Pressurizer PORV' in filename or 'pzrv' in filename.lower():
            condition = 'pressurizer_porv'
        elif 'SG tube rupture' in filename or 'sgtr' in filename.lower():
            condition = 'sg_tube_rupture'
        elif 'Power Change' in filename:
            condition = 'power_change'
        else:
            condition = 'unknown'
        
        df_copy['condition'] = condition
        labeled_data[filename] = df_copy
    
    return labeled_data


def combine_data(labeled_data):
    """
    Combine all labeled dataframes into a single dataframe.
    
    Args:
        labeled_data (dict): Dictionary of labeled dataframes
        
    Returns:
        pd.DataFrame: Combined dataframe
    """
    combined_df = pd.concat(labeled_data.values(), ignore_index=True)
    return combined_df


def balance_data(X, y):
    """
    Balance the dataset using SMOTE.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def prepare_data_for_model(features, labels):
    """
    Prepare data for model training by encoding labels and reshaping features.
    
    Args:
        features (pd.DataFrame): Feature dataframe
        labels (list): List of labels
        
    Returns:
        tuple: (X, y, encoder)
    """
    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    y_one_hot = pd.get_dummies(y).values  # One-hot encoding
    
    # Reshape features for Conv1D
    X = features.values.reshape((features.shape[0], features.shape[1], 1))
    
    return X, y_one_hot, encoder


def process_pipeline(data_path='data', recursive=True):
    """
    Complete data preprocessing pipeline from loading to model-ready data.
    
    Args:
        data_path (str): Path to the directory containing data files
        recursive (bool): Whether to search recursively in subdirectories
        
    Returns:
        tuple: (combined_data, scaler)
    """
    # Load data (including from subdirectories)
    data_dict = load_data(data_path, pattern='**/*.xlsx', recursive=recursive)
    
    # Label data first (before preprocessing to avoid label normalization)
    labeled_data = label_data(data_dict)
    
    # Combine all data
    combined_data = combine_data(labeled_data)
    
    # Preprocess the combined data
    processed_data, scaler = preprocess_data(combined_data)
    
    return processed_data, scaler


# Example usage
if __name__ == "__main__":
    # Run the complete pipeline
    processed_data, scaler = process_pipeline()
    
    # Save preprocessed data (optional)
    processed_data.to_csv('data/processed/preprocessed_data.csv', index=False)
    
    print(f"Preprocessing complete. Data shape: {processed_data.shape}")
    print(f"Condition distribution:\n{processed_data['condition'].value_counts()}")