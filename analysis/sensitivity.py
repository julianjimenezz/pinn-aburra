"""
Sensitivity analysis for geotechnical parameters in PINN.

Analyzes how model predictions and physics loss change when varying
each learnable parameter in the Perzyna viscoplastic model.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader

from models import BasePINN, SoilPhysics


def load_trained_model(
    checkpoint_path: str,
    model: BasePINN,
    soil_physics: SoilPhysics,
    device: str = "cuda"
) -> Tuple[BasePINN, SoilPhysics]:
    """Load a trained model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    soil_physics.load_state_dict(checkpoint['soil_physics_state_dict'])
    return model, soil_physics


def get_baseline_predictions(
    model: BasePINN,
    soil_physics: SoilPhysics,
    data_loader: DataLoader,
    u_mean: float,
    u_scale: float,
    dt_days: float,
    device: str
) -> Tuple[np.ndarray, float]:
    """Get baseline predictions and physics loss with current parameters."""
    model.eval()
    preds = []
    physics_losses = []
    
    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b = X_b.to(device)
            
            # Compute physics baseline from this soil_physics instance
            u_prev_scaled = X_b[:, -1, 0].view(-1, 1)
            u_prev = (u_prev_scaled * u_scale) + u_mean
            u_physics = soil_physics.compute_displacement(u_prev, dt_days)
            u_physics_scaled = (u_physics - u_mean) / u_scale
            
            # Forward pass with physics baseline
            pred_scaled = model(X_b, u_physics_scaled)
            pred = (pred_scaled * u_scale) + u_mean
            preds.extend(pred.cpu().numpy().flatten())
            
            # Physics loss
            loss_phy = soil_physics.perzyna_loss(pred, u_prev, dt_days)
            physics_losses.append(loss_phy.item())
    
    return np.array(preds), np.mean(physics_losses)


def run_sensitivity_analysis(
    model: BasePINN,
    soil_physics: SoilPhysics,
    data_loader: DataLoader,
    u_mean: float,
    u_scale: float,
    config: Dict[str, Any],
    output_name: str,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Perform sensitivity analysis on geotechnical parameters.
    
    Varies each parameter by ±10%, ±25%, ±50% from learned values
    and measures the impact on predictions and physics loss.
    
    Args:
        model: Trained PINN model
        soil_physics: Trained SoilPhysics (with learned parameters)
        data_loader: Test data loader
        u_mean, u_scale: Scaling parameters
        config: Configuration dict
        output_name: Name for output files
        device: Compute device
    
    Returns:
        DataFrame with sensitivity results
    """
    physics_cfg = config.get('physics', {})
    dt_days = physics_cfg.get('dt_days', 12.0)
    output_cfg = config.get('output', {})
    results_dir = Path(output_cfg.get('results_dir', 'outputs/results'))
    figures_dir = Path(output_cfg.get('figures_dir', 'outputs/figures'))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Get baseline learned parameters
    learned_params = soil_physics.get_params_dict()
    print(f"\n📐 Learned geotechnical parameters:")
    for name, value in learned_params.items():
        print(f"   {name}: {value:.4f}")
    
    # Get baseline predictions
    print("\n⏳ Computing baseline predictions...")
    baseline_preds, baseline_phy_loss = get_baseline_predictions(
        model, soil_physics, data_loader, u_mean, u_scale, dt_days, device
    )
    
    # Parameters to analyze
    param_names = ['c_prime', 'phi', 'n_exp', 'sigma_n', 'eta']
    
    # Variation factors
    variations = [-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50]
    
    results = []
    
    print("\n🔬 Running sensitivity analysis...")
    
    for param_name in param_names:
        if param_name not in learned_params:
            continue
            
        base_value = learned_params[param_name]
        print(f"\n   Analyzing: {param_name} (base: {base_value:.4f})")
        
        for var in variations:
            # Create a copy of soil_physics with modified parameter
            test_physics = SoilPhysics(
                h=soil_physics.h,
                tau_fixed=soil_physics.tau_fixed,
                device=device
            )
            # Copy learned parameters
            test_physics.load_state_dict(soil_physics.state_dict())
            
            # Modify the target parameter
            new_value = base_value * (1 + var)
            
            # Set the parameter (need to set the raw parameter before softplus)
            # Since params are constrained via softplus, we need to work backwards
            with torch.no_grad():
                if param_name == 'c_prime':
                    test_physics.c_prime_raw.fill_(np.log(np.exp(max(new_value, 0.01)) - 1 + 1e-6))
                elif param_name == 'phi':
                    test_physics.phi_raw.fill_(np.log(np.exp(max(new_value, 0.01)) - 1 + 1e-6))
                elif param_name == 'n_exp':
                    test_physics.n_exp_raw.fill_(np.log(np.exp(max(new_value - 1, 0.01)) - 1 + 1e-6))
                elif param_name == 'sigma_n':
                    test_physics.sigma_n_raw.fill_(np.log(np.exp(max(new_value, 0.01)) - 1 + 1e-6))
                elif param_name == 'eta':
                    test_physics.eta_raw.fill_(np.log(np.exp(max(new_value, 0.01)) - 1 + 1e-6))
            
            # Get predictions with modified parameter
            preds, phy_loss = get_baseline_predictions(
                model, test_physics, data_loader, u_mean, u_scale, dt_days, device
            )
            
            # Compute metrics
            pred_diff = np.mean(np.abs(preds - baseline_preds))
            pred_diff_pct = (pred_diff / np.mean(np.abs(baseline_preds))) * 100 if np.mean(np.abs(baseline_preds)) > 0 else 0
            phy_loss_diff_pct = ((phy_loss - baseline_phy_loss) / baseline_phy_loss) * 100 if baseline_phy_loss > 0 else 0
            
            results.append({
                'parameter': param_name,
                'variation_pct': var * 100,
                'original_value': base_value,
                'modified_value': new_value,
                'pred_mae_diff': pred_diff,
                'pred_diff_pct': pred_diff_pct,
                'physics_loss': phy_loss,
                'physics_loss_diff_pct': phy_loss_diff_pct
            })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    results_path = results_dir / f"{output_name}_sensitivity.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n📁 Results saved: {results_path}")
    
    # Create visualization
    plot_sensitivity_results(df_results, output_name, figures_dir)
    
    return df_results


def plot_sensitivity_results(
    df: pd.DataFrame,
    output_name: str,
    figures_dir: Path
):
    """Create sensitivity analysis plots."""
    params = df['parameter'].unique()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Prediction difference
    ax1 = axes[0]
    for param in params:
        param_data = df[df['parameter'] == param]
        ax1.plot(
            param_data['variation_pct'],
            param_data['pred_diff_pct'],
            marker='o',
            label=param,
            linewidth=2
        )
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Parameter Variation (%)')
    ax1.set_ylabel('Prediction Change (%)')
    ax1.set_title('Sensitivity: Impact on Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Physics loss change
    ax2 = axes[1]
    for param in params:
        param_data = df[df['parameter'] == param]
        ax2.plot(
            param_data['variation_pct'],
            param_data['physics_loss_diff_pct'],
            marker='s',
            label=param,
            linewidth=2
        )
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Parameter Variation (%)')
    ax2.set_ylabel('Physics Loss Change (%)')
    ax2.set_title('Sensitivity: Impact on Physics Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = figures_dir / f"{output_name}_sensitivity.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Saved: {plot_path}")


def print_sensitivity_summary(df: pd.DataFrame):
    """Print a summary of sensitivity analysis results."""
    print("\n" + "="*60)
    print("📊 SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n🎯 Parameter Sensitivity Ranking (by MAX prediction impact):")
    
    # Calculate max impact for each parameter
    max_impacts = []
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param]
        # Find row with max pred_diff_pct
        max_row = param_data.loc[param_data['pred_diff_pct'].idxmax()]
        max_impacts.append(max_row)
    
    # Sort by impact
    max_impacts.sort(key=lambda x: x['pred_diff_pct'], reverse=True)
    
    for row in max_impacts:
        print(f"   {row['parameter']:12} → {row['pred_diff_pct']:.2f}% change (at {row['variation_pct']:+.0f}% var)")
    
    print("\n⚡ Physics Loss Sensitivity (MAX impact):")
    
    # Calculate max physics impact
    max_phy_impacts = []
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param]
        # Find row with max abs(physics_loss_diff_pct)
        max_row = param_data.loc[param_data['physics_loss_diff_pct'].abs().idxmax()]
        max_phy_impacts.append(max_row)
        
    max_phy_impacts.sort(key=lambda x: abs(x['physics_loss_diff_pct']), reverse=True)
    
    for row in max_phy_impacts:
        print(f"   {row['parameter']:12} → {row['physics_loss_diff_pct']:+.1f}% change (at {row['variation_pct']:+.0f}% var)")
