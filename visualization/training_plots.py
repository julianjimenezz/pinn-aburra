"""
Visualization module for training history and results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def plot_training_curves(
    history_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Training Curves"
) -> plt.Figure:
    """
    Plot training history curves.
    
    Creates a 2x2 grid:
        - Total Loss (train)
        - Data Loss (train)
        - Physics Loss (train)
        - Validation Metrics (RMSE, R²)
    
    Args:
        history_df: DataFrame with training history
        output_path: Path to save figure (optional)
        title: Figure title
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    epochs = history_df['epoch']
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history_df['train_loss_total'], 'b-', label='Train', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Data Loss
    ax = axes[0, 1]
    ax.plot(epochs, history_df['train_loss_data'], 'g-', label='Data Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Data Loss (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Physics Loss
    ax = axes[1, 0]
    ax.plot(epochs, history_df['train_loss_physics'], 'r-', label='Physics Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Physics Loss (Perzyna)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Validation Metrics
    ax = axes[1, 1]
    ax2 = ax.twinx()
    
    line1, = ax.plot(epochs, history_df['val_rmse'], 'b-', label='RMSE', linewidth=2)
    line2, = ax2.plot(epochs, history_df['val_r2'], 'orange', label='R²', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (mm)', color='blue')
    ax2.set_ylabel('R²', color='orange')
    ax.set_title('Validation Metrics')
    ax.legend(handles=[line1, line2], loc='center right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📈 Saved: {output_path}")
    
    return fig


def plot_physics_balance(
    history_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Physics Balance"
) -> plt.Figure:
    """
    Plot physics balance over training.
    
    Shows |data_loss - λ*physics_loss| to identify optimal balance point.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = history_df['epoch']
    balance = history_df['physics_balance']
    
    ax.plot(epochs, balance, 'purple', linewidth=2)
    
    # Mark minimum
    min_idx = balance.idxmin()
    min_epoch = epochs[min_idx]
    min_val = balance[min_idx]
    ax.axvline(min_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best: Epoch {min_epoch}')
    ax.scatter([min_epoch], [min_val], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|Data Loss - λ·Physics Loss|')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📈 Saved: {output_path}")
    
    return fig


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Predictions vs Actual"
) -> plt.Figure:
    """
    Create scatter plot and residual histogram.
    """
    from sklearn.metrics import r2_score, mean_squared_error
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(targets, predictions, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    ax.set_xlabel('Actual (mm)')
    ax.set_ylabel('Predicted (mm)')
    ax.set_title(f'{title}\nR² = {r2:.4f}, RMSE = {rmse:.4f} mm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual histogram
    ax = axes[1]
    residuals = predictions - targets
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (mm)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📈 Saved: {output_path}")
    
    return fig


def generate_all_plots(
    history_csv_path: str,
    area_name: str,
    output_dir: str = "outputs/figures"
):
    """
    Generate all training result plots from history CSV.
    
    Args:
        history_csv_path: Path to training history CSV
        area_name: Name for titles and filenames
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history_df = pd.read_csv(history_csv_path)
    
    # Training curves
    plot_training_curves(
        history_df,
        output_path=str(output_dir / f"{area_name}_training_curves.png"),
        title=f"Training Curves - {area_name}"
    )
    
    # Physics balance
    plot_physics_balance(
        history_df,
        output_path=str(output_dir / f"{area_name}_physics_balance.png"),
        title=f"Physics Balance - {area_name}"
    )
    
    plt.close('all')
    print(f"\n✅ All plots generated in {output_dir}")
