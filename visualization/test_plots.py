"""
Visualization functions for test set evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

def plot_test_results(targets: np.ndarray, preds: np.ndarray, output_name: str, figures_dir: Path):
    """
    Generate and save test set visualizations:
    1. Time Series Zoom (first 300 steps)
    2. Scatter Plot (Actual vs Predicted)
    
    Args:
        targets: Actual values
        preds: Predicted values
        output_name: Base name for output files
        figures_dir: Directory to save figures
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. PLOT 1: TIME SERIES (ZOOM) ---
    limit = 300
    plt.figure(figsize=(14, 6))
    plt.plot(targets[:limit], label='Actual', color='black', alpha=0.7, linewidth=1.5)
    plt.plot(preds[:limit], label='PINN Prediction', color='red', linestyle='--', linewidth=1.5)
    plt.title(f"Test Validation: First {limit} time steps")
    plt.xlabel("Time (Steps)")
    plt.ylabel("Displacement (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ts_path = figures_dir / f"{output_name}_test_timeseries.png"
    plt.savefig(ts_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved Time Series: {ts_path}")
    
    # --- 2. PLOT 2: CORRELATION (SCATTER) ---
    plt.figure(figsize=(7, 7))
    plt.scatter(targets, preds, alpha=0.2, color='blue', s=10)
    
    # Identity line (Perfect prediction)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (x=y)')
    
    r2 = r2_score(targets, preds)
    
    plt.xlabel("Actual (m)")
    plt.ylabel("Prediction (m)")
    plt.title(f"Actual vs Pred (R²: {r2:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    scatter_path = figures_dir / f"{output_name}_test_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved Scatter Plot: {scatter_path}")
