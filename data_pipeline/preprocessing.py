"""
Data preprocessing module for PINN training.

Handles:
- CSV loading
- Savitzky-Golay smoothing
- Train/Val/Test splitting
- Sequence generation
- DataLoader creation
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DataLoaders:
    """Container for train/val/test data loaders."""
    train: DataLoader
    val: DataLoader
    test: DataLoader
    scaler: StandardScaler
    u_mean: float
    u_scale: float
    feature_cols: List[str]


def apply_savgol_smoothing(
    df: pd.DataFrame,
    column: str = 'u',
    window_length: int = 10,
    poly_order: int = 2
) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to smooth displacement data per location.
    
    Args:
        df: DataFrame with X, Y, and target column
        column: Column to smooth
        window_length: Filter window length
        poly_order: Polynomial order
    
    Returns:
        DataFrame with smoothed column
    """
    df = df.copy()
    
    def smooth_group(x):
        try:
            if len(x) > window_length:
                return savgol_filter(x, window_length=window_length, polyorder=poly_order)
        except Exception:
            pass
        return x
    
    df[column] = df.groupby(['X', 'Y'])[column].transform(smooth_group)
    return df


def create_sequences_with_time(
    df: pd.DataFrame,
    n_steps_in: int,
    n_steps_out: int,
    scaler: StandardScaler,
    feature_cols: List[str],
    target_col: str = 'u'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences respecting location grouping, returning the target time for each sequence.
    
    This allows us to split sequences by time AFTER creation, ensuring we have
    sequences even when individual time splits would be too small.
    
    Args:
        df: Input DataFrame (must have t_days column)
        n_steps_in: Input sequence length
        n_steps_out: Output sequence length (typically 1)
        scaler: Fitted StandardScaler
        feature_cols: Feature column names
        target_col: Target column name
    
    Returns:
        Tuple of (X, y, t_target) numpy arrays where t_target is the day being predicted
    """
    X_list, y_list, t_list = [], [], []
    
    grouped = df.groupby(['X', 'Y'])
    
    for _, group in grouped:
        group = group.sort_values('t_days').reset_index(drop=True)
        group_scaled = scaler.transform(group[feature_cols])
        t_days_values = group['t_days'].values
        
        for i in range(len(group_scaled) - n_steps_in - n_steps_out + 1):
            # Input: [t : t + n_steps_in]
            X_list.append(group_scaled[i : i + n_steps_in, :])
            # Target: [t + n_steps_in : t + n_steps_in + n_steps_out] - only u column (index 0)
            y_list.append(group_scaled[i + n_steps_in : i + n_steps_in + n_steps_out, 0])
            # Target time: the day we're predicting
            t_list.append(t_days_values[i + n_steps_in])
    
    return np.array(X_list), np.array(y_list), np.array(t_list)


def prepare_data(
    csv_path: str,
    config: Dict[str, Any],
    device: str = "cuda"
) -> DataLoaders:
    """
    Full data preparation pipeline.
    
    Strategy: Create ALL sequences first, then split by target time.
    This ensures val/test have data even with small datasets.
    
    Args:
        csv_path: Path to CSV file with rainfall features
        config: Configuration dict
        device: Device for tensors
    
    Returns:
        DataLoaders container with train/val/test loaders and scaler info
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Get config
    data_cfg = config.get('data', {})
    seq_cfg = config.get('sequence', {})
    training_cfg = config.get('training', {})
    model_type = training_cfg.get('model_type', 'gru')
    model_cfg = config.get(model_type, {})
    
    feature_cols = data_cfg.get('feature_cols', 
        ['u', 'X', 'Y', 'p_acum_30d', 'p_acum_60d', 'p_acum_90d', 't_days'])
    target_col = data_cfg.get('target_col', 'u')
    
    n_steps_in = seq_cfg.get('n_steps_in', 10)
    n_steps_out = seq_cfg.get('n_steps_out', 1)
    batch_size = model_cfg.get('batch_size', 64)
    
    # Sort by location and time
    df = df.sort_values(by=['X', 'Y', 't_days']).reset_index(drop=True)
    
    # Apply smoothing
    df = apply_savgol_smoothing(df, column='u')
    
    # Show data info
    unique_days = sorted(df['t_days'].unique())
    n_days = len(unique_days)
    print(f"   📅 Days range: {int(unique_days[0])} - {int(unique_days[-1])} ({n_days} unique days)")
    
    # Calculate split cutoffs based on TEMPORAL order (70% train, 15% val, 15% test)
    # We do this BEFORE creating sequences to fit scaler properly
    train_idx = int(n_days * 0.70)
    val_idx = int(n_days * 0.85)
    
    cutoff_train = unique_days[train_idx]
    cutoff_val = unique_days[val_idx]
    
    print(f"   ✂️  Split cutoffs (days): Train≤{int(cutoff_train)}, Val≤{int(cutoff_val)}")
    
    # Fit scaler ONLY on training data (prevent leakage)
    train_subset = df[df['t_days'] <= cutoff_train]
    scaler = StandardScaler()
    scaler.fit(train_subset[feature_cols])
    print(f"   ⚖️  Scaler fitted on {len(train_subset)} training rows (out of {len(df)})")
    
    # Get u scaling parameters for de-normalization
    u_idx = feature_cols.index('u')
    u_mean = scaler.mean_[u_idx]
    u_scale = scaler.scale_[u_idx]
    
    # Create ALL sequences with their target times
    # Note: create_sequences_with_time uses the fitted scaler to transform all data
    X_all, y_all, t_all = create_sequences_with_time(
        df, n_steps_in, n_steps_out, scaler, feature_cols, target_col
    )
    
    print(f"   📊 Total sequences: {len(X_all)}")
    
    # Split sequences by target time using the SAME cutoffs
    train_mask = t_all <= cutoff_train
    val_mask = (t_all > cutoff_train) & (t_all <= cutoff_val)
    test_mask = t_all > cutoff_val
    
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    
    # Create DataLoaders
    def make_loader(X, y, shuffle):
        if len(X) == 0:
            # Return empty loader
            return DataLoader(TensorDataset(
                torch.zeros(0, n_steps_in, len(feature_cols)),
                torch.zeros(0, n_steps_out)
            ), batch_size=batch_size)
        return DataLoader(
            TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y).float()
            ),
            shuffle=shuffle,
            batch_size=batch_size
        )
    
    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)
    
    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        scaler=scaler,
        u_mean=u_mean,
        u_scale=u_scale,
        feature_cols=feature_cols
    )
