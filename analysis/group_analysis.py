"""
Group analysis script to evaluate predefined parameter sets.
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

def run_group_analysis(
    config_path: str,
    epochs_override: int = None,
    csv_path: str = None,  # New argument
    output_dir: str = "outputs/group_analysis",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run training sessions with predefined parameter groups (A, B, C...).
    
    Args:
        config_path: Path to config.yaml
        epochs_override: Optional number of epochs to override config
        csv_path: Optional path to override dataset
        output_dir: Directory to save results
        device: 'cuda' or 'cpu'
    """
    print(f"🚀 Starting Group Analysis")
    print(f"   Config: {config_path}")
    print(f"   Device: {device}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override dataset if provided
    if csv_path:
        config['data']['csv_path'] = csv_path
        print(f"   Dataset Override: {csv_path}")
    
    # Get dataset name for labeling
    csv_path_conf = config.get('data', {}).get('csv_path', 'unknown')
    dataset_name = Path(csv_path_conf).stem
    print(f"   Dataset: {dataset_name}")

    # Setup output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    models_out_dir = out_path / "models"
    models_out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get groups from config
    groups = config.get('group_analysis', {})
    if not groups:
        print("❌ No 'group_analysis' section found in config.yaml")
        return

    print(f"   Found {len(groups)} groups: {list(groups.keys())}")
    
    # --- Override Config for Group Analysis ---
    config['output']['models_dir'] = str(models_out_dir)
    
    # Also redirect results/history files
    results_out_dir = out_path / "results"
    results_out_dir.mkdir(parents=True, exist_ok=True)
    config['output']['results_dir'] = str(results_out_dir)
    
    # Disable periodic checkpointing
    if 'checkpointing' not in config:
        config['checkpointing'] = {}
    config['checkpointing']['save_every_n_epochs'] = 0
    config['checkpointing']['save_best_total'] = False
    
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

    # --- Depth Iteration Loop ---
    # Get depths from config, or default to single current h
    depths = groups.pop('depths', [config['physics']['h']])
    if not isinstance(depths, list):
        depths = [depths]
        
    print(f"   Target Depths: {depths}")
    
    original_h = config['physics']['h']
    
    for h_val in depths:
        print(f"\n🌊 Testing Depth h = {h_val} m")
        config['physics']['h'] = float(h_val)
        
        current_depth_results = []
        
        # --- Group Analysis Loop ---
        for group_name, init_params in groups.items():
            print(f"\n   🧪 Testing Group {group_name} (h={h_val})")
            print(f"      Init: {init_params}")
            
            # Create Model with specific init params
            model, physics = create_model(config, input_size, device, init_params=init_params)
            
            # Train
            trainer = PINNTrainer(
                model=model,
                soil_physics=physics,
                config=config,
                u_mean=u_mean,
                u_scale=u_scale,
                device=device,
                output_name=f"{dataset_name}_group_{group_name}_h{h_val}"
            )
            
            # Reduce epochs if override provided
            if epochs_override is not None:
                trainer.epochs = epochs_override
                print(f"      ℹ️  Overriding epochs to: {epochs_override}") 
            
            history = trainer.train(train_loader, val_loader, verbose=False)
            
            # Collect final params
            final_params = physics.get_params_dict()
            metrics = {
                "group": group_name,
                "depth_h": h_val,
                "best_val_rmse": history.best_val_rmse,
                "best_val_r2": history.epochs[history.best_val_epoch - 1].val_r2,
                "final_physics_balance": history.epochs[-1].physics_balance,
                **{f"init_{k}": v for k, v in init_params.items()},
                **{f"final_{k}": v for k, v in final_params.items()}
            }
            current_depth_results.append(metrics)
            print(f"      ✅ Done. Final: {final_params}")
        
        # Save Results for this depth
        results_df = pd.DataFrame(current_depth_results)
        csv_path_out = out_path / "results" / f"{dataset_name}_group_analysis_h{h_val}_{timestamp}.csv"
        results_df.to_csv(csv_path_out, index=False)
        print(f"\n   📄 Results for h={h_val} saved to: {csv_path_out}")
        
        # --- Visualization for this depth ---
        print(f"   📊 Generating comparison plots for h={h_val}...")
        try:
            param_names = ["c_prime", "phi", "n", "sigma_n", "eta"]
            n_grps = len(groups)
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            fig.suptitle(f"Parameter Evolution for Depth h={h_val} m")
            
            for idx, p in enumerate(param_names):
                ax = axes[idx]
                
                # Prepare data
                groups_labels = results_df["group"]
                init_vals = results_df[f"init_{p}"]
                final_vals = results_df[f"final_{p}"]
                
                x = np.arange(n_grps)
                width = 0.35
                
                ax.bar(x - width/2, init_vals, width, label='Init', alpha=0.7)
                ax.bar(x + width/2, final_vals, width, label='Final', alpha=0.7)
                
                ax.set_title(p)
                ax.set_xticks(x)
                ax.set_xticklabels(groups_labels)
                if idx == 0:
                    ax.legend()
                    
            plt.tight_layout()
            plt.savefig(out_path / f"{dataset_name}_group_comparison_h{h_val}_{timestamp}.png")
            print(f"      Plot saved: {out_path / f'{dataset_name}_group_comparison_h{h_val}_{timestamp}.png'}")
            plt.close(fig) # Close to free memory
            
        except Exception as e:
            print(f"      Could not create plot: {e}")

    # Restore original config h
    config['physics']['h'] = original_h

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--csv", type=str, default=None, help="Override path to CSV dataset")
    args = parser.parse_args()
    
    run_group_analysis(args.config, args.epochs, args.csv)
