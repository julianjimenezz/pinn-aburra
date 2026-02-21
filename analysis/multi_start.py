"""
Multi-start analysis script to evaluate parameter robustness.
"""

import sys
import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import create_model
from training.trainer import PINNTrainer
from torch.utils.data import DataLoader, TensorDataset

def run_multi_start_analysis(
    config_path: str,
    n_runs: int = 10,
    output_dir: str = "outputs/multi_start",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run multiple training sessions with randomized initial parameters.
    
    Args:
        config_path: Path to config.yaml
        n_runs: Number of independent runs
        output_dir: Directory to save results
        device: 'cuda' or 'cpu'
    """
    print(f"🚀 Starting Multi-Start Analysis ({n_runs} runs)")
    print(f"   Config: {config_path}")
    print(f"   Device: {device}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sampling ranges: Read from config or use defaults
    default_ranges = {
        "c_prime": (2.0, 15.0),
        "phi": (10.0, 35.0),
        "n": (1.1, 4.0),
        "sigma_n": (10.0, 50.0),
        "eta": (1e5, 5e6)
    }
    
    # Get dataset name for labeling
    csv_path_conf = config.get('data', {}).get('csv_path', 'unknown')
    dataset_name = Path(csv_path_conf).stem
    print(f"   Dataset: {dataset_name}")
    
    # Get ranges from config if they exist
    config_ranges = config.get('multi_start', {}).get('ranges', {})
    ranges = {**default_ranges, **config_ranges}
    
    # --- Override Config for Multi-Start ---
    # Redirect model checkpoints to multi-start folder
    models_out_dir = out_path / "models"
    models_out_dir.mkdir(parents=True, exist_ok=True)
    config['output']['models_dir'] = str(models_out_dir)
    
    # Also redirect results/history files
    results_out_dir = out_path / "results"
    results_out_dir.mkdir(parents=True, exist_ok=True)
    config['output']['results_dir'] = str(results_out_dir)
    
    # Disable periodic checkpointing to save space (keep only best)
    if 'checkpointing' not in config:
        config['checkpointing'] = {}
    config['checkpointing']['save_every_n_epochs'] = 0  # Disable periodic
    config['checkpointing']['save_best_total'] = False  # Optional: reduce files further
    
    print(f"   Parameter Ranges: {ranges}")
    print(f"   Saving temporary models to: {models_out_dir}")
    
    results = []
    
    # --- Load Data (Shared across runs) ---
    print("\n📦 Loading Data...")
    try:
        # Load CSV
        df = pd.read_csv(config['data']['csv_path'])
        feature_cols = config['data']['feature_cols']
        target_col = config['data']['target_col']
        
        # Preprocessing (Scaling)
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Simple manual scaling for this analysis script
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        y_scaled = (target - target.mean()) / target.std()
        
        # Create windows
        seq_len = config['sequence']['n_steps_in']
        X_list, y_list = [], []
        for i in range(len(X_scaled) - seq_len):
            X_list.append(X_scaled[i:i+seq_len])
            y_list.append(y_scaled[i+seq_len])
        
        X = torch.FloatTensor(np.array(X_list))
        y = torch.FloatTensor(np.array(y_list))
        
        # Split
        train_size = int(len(X) * 0.8)
        train_loader = DataLoader(TensorDataset(X[:train_size], y[:train_size]), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X[train_size:], y[train_size:]), batch_size=64)
        
        input_size = len(feature_cols)
        u_mean = target.mean()
        u_scale = target.std()

        print(f"Data Loaded: {len(X)} samples. Input size: {input_size}")

    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # --- Multi-Start Loop ---
    for i in range(n_runs):
        print(f"\n🔄 Run {i+1}/{n_runs}")
        
        # Sample random initial parameters
        init_params = {
            "c_prime": np.random.uniform(*ranges["c_prime"]),
            "phi": np.random.uniform(*ranges["phi"]),
            "n": np.random.uniform(*ranges["n"]),
            "sigma_n": np.random.uniform(*ranges["sigma_n"]),
            "eta": np.random.uniform(*ranges["eta"])
        }
        
        print(f"   Init: {init_params}")
        
        # Create Model
        model, physics = create_model(config, input_size, device, init_params=init_params)
        
        # Train
        trainer = PINNTrainer(
            model=model,
            soil_physics=physics,
            config=config,
            u_mean=u_mean,
            u_scale=u_scale,
            device=device,
            output_name=f"{dataset_name}_run_{i}"
        )
        
        # Reduce epochs for analysis speed (optional override)
        # trainer.epochs = 20 
        
        history = trainer.train(train_loader, val_loader, verbose=False)
        
        # Collect final params
        final_params = physics.get_params_dict()
        metrics = {
            "run_id": i,
            "best_val_rmse": history.best_val_rmse,
            "final_physics_balance": history.epochs[-1].physics_balance,
            **{f"init_{k}": v for k, v in init_params.items()},
            **{f"final_{k}": v for k, v in final_params.items()}
        }
        results.append(metrics)
        print(f"   ✅ Done. Final: {final_params}")
    
    # Save Results
    results_df = pd.DataFrame(results)
    csv_path = out_path / f"{dataset_name}_multi_start_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📄 Results saved to: {csv_path}")
    
    # --- Cleanup: Keep ONLY the best model ---
    if results:
        # Identify best run by RMSE
        best_run = min(results, key=lambda x: x['best_val_rmse'])
        best_run_id = best_run['run_id']
        print(f"\n🏆 Best Run: {best_run_id} (RMSE: {best_run['best_val_rmse']:.4f})")
        print("   Cleaning up other models...")
        
        # Iterate through all files in models_dir
        for file_path in models_out_dir.glob(f"{dataset_name}_run_*.pt"):
            # specific format: {dataset_name}_run_{i}_{type}.pt
            # Check if this file belongs to the best run
            if f"{dataset_name}_run_{best_run_id}_" not in file_path.name:
                try:
                    file_path.unlink()
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
        
        print(f"   Kept models for run {best_run_id} only.")
    
    # --- Analysis & Visualization ---
    print("\n📊 Statistical Summary:")
    param_names = ["c_prime", "phi", "n", "sigma_n", "eta"]
    
    summary = []
    for p in param_names:
        final_vals = results_df[f"final_{p}"]
        mean_val = final_vals.mean()
        std_val = final_vals.std()
        cv = std_val / mean_val if mean_val != 0 else 0
        
        print(f"   {p:>10}: Mean={mean_val:.4f}, Std={std_val:.4f}, CV={cv:.4f}")
        summary.append({"param": p, "mean": mean_val, "std": std_val, "cv": cv})
    
    # Simple Boxplot
    try:
        plt.figure(figsize=(12, 6))
        # Normalize for visualization comparison
        plot_data = []
        labels = []
        for p in param_names:
            vals = results_df[f"final_{p}"]
            # Normalize by mean to show relative variance
            norm_vals = vals / vals.mean()
            plot_data.append(norm_vals)
            labels.append(p)
            
        plt.boxplot(plot_data, labels=labels)
        plt.boxplot(plot_data, labels=labels)
        plt.title(f"Parameter Stability ({dataset_name}) - {n_runs} runs")
        plt.ylabel("Relative Value (x Mean)")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path / f"{dataset_name}_param_stability_{timestamp}.png")
        print(f"   Plot saved.")
    except Exception as e:
        print(f"   Could not create plot: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--csv", type=str, default=None, help="Override path to CSV dataset")
    args = parser.parse_args()
    
    run_multi_start_analysis(args.config, args.runs, args.csv)
