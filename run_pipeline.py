"""
PINN Automation Pipeline - Main Entry Point

Usage:
    python run_pipeline.py --mode train --csv path/to/data.csv --model gru
    python run_pipeline.py --config config.yaml
"""

import argparse
import yaml
import random
import os
import numpy as np
import torch
from pathlib import Path

from data_pipeline import prepare_data
from models import create_model
from training import PINNTrainer
from training.hyperparameter_tuning import run_hyperparameter_tuning
from visualization import generate_all_plots


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_training(config: dict, csv_path: str = None, model_type: str = None):
    """
    Main training function.
    
    Args:
        config: Configuration dict
        csv_path: Override CSV path from config
        model_type: Override model type from config
    """
    # Override config with CLI args if provided
    if csv_path:
        config['data']['csv_path'] = csv_path
    if model_type:
        config['training']['model_type'] = model_type
    
    # Set seed
    seed = config.get('training', {}).get('seed', 42)
    seed_everything(seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Get area name from CSV path
    csv_path = config['data']['csv_path']
    area_name = Path(csv_path).stem
    model_type = config['training']['model_type']
    output_name = f"{area_name}_{model_type}"
    
    print(f"\n📂 Dataset: {area_name}")
    print(f"🧠 Model: {model_type.upper()}")
    
    # Prepare data
    print("\n⏳ Loading and preprocessing data...")
    data_loaders = prepare_data(csv_path, config, device)
    
    print(f"   Train batches: {len(data_loaders.train)}")
    print(f"   Val batches:   {len(data_loaders.val)}")
    print(f"   Test batches:  {len(data_loaders.test)}")
    print(f"   U mean: {data_loaders.u_mean:.4f}, U scale: {data_loaders.u_scale:.4f}")
    
    # Create model
    input_size = len(data_loaders.feature_cols)
    model, soil_physics = create_model(config, input_size, device)
    
    print(f"\n📐 Model: {model.get_config()}")
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        soil_physics=soil_physics,
        config=config,
        u_mean=data_loaders.u_mean,
        u_scale=data_loaders.u_scale,
        device=device,
        output_name=output_name
    )
    
    # Train
    print("\n🚀 Starting training...\n")
    history = trainer.train(
        train_loader=data_loaders.train,
        val_loader=data_loaders.val,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_results = trainer.evaluate_test(data_loaders.test)
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    results_dir = config.get('output', {}).get('results_dir', 'outputs/results')
    figures_dir = config.get('output', {}).get('figures_dir', 'outputs/figures')
    
    history_path = f"{results_dir}/{output_name}_history.csv"
    generate_all_plots(history_path, output_name, figures_dir)
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE")
    print("="*50)
    print(f"Best Val RMSE: {history.best_val_rmse:.5f} mm (Epoch {history.best_val_epoch})")
    
    if test_results['test_rmse'] is not None:
        print(f"Test RMSE:     {test_results['test_rmse']:.5f} mm")
        print(f"Test R²:       {test_results['test_r2']:.4f}")
    else:
        print("Test:          (skipped - no test data)")
    
    print(f"\nModels saved in: outputs/models/")
    print(f"Results in:      outputs/results/")
    print(f"Figures in:      outputs/figures/")
    
    return history, test_results


def main():
    parser = argparse.ArgumentParser(
        description="PINN Automation Pipeline for Landslide Deformation Prediction"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['train', 'tune', 'analyze'],
        default='train',
        help='Pipeline mode: train, tune (hyperparameter), or analyze (sensitivity)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Override: Path to CSV file for training'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['gru', 'lstm'],
        help='Override: Model type (gru or lstm)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.mode == 'train':
        run_training(config, csv_path=args.csv, model_type=args.model)
    elif args.mode == 'tune':
        # Override config with CLI args
        if args.csv:
            config['data']['csv_path'] = args.csv
        if args.model:
            config['training']['model_type'] = args.model
        
        csv_path = config['data']['csv_path']
        model_type = config['training']['model_type']
        area_name = Path(csv_path).stem
        output_name = f"{area_name}_{model_type}"
        
        seed_everything(config.get('training', {}).get('seed', 42))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🖥️  Device: {device}")
        print(f"\n📂 Dataset: {area_name}")
        print(f"🧠 Model: {model_type.upper()}")
        print("\n⏳ Loading data for tuning...")
        
        data_loaders = prepare_data(csv_path, config, device)
        input_size = len(data_loaders.feature_cols)
        
        print(f"\n🔍 Starting Optuna hyperparameter search...")
        study = run_hyperparameter_tuning(
            model_type=model_type,
            train_loader=data_loaders.train,
            val_loader=data_loaders.val,
            config=config,
            u_mean=data_loaders.u_mean,
            u_scale=data_loaders.u_scale,
            input_size=input_size,
            output_name=output_name,
            device=device
        )
        
        print("\n" + "="*50)
        print("✅ TUNING COMPLETE")
        print("="*50)
        print(f"Best params: {study.best_params}")
        print(f"Best val RMSE: {study.best_value:.5f}")
        print(f"\nResults saved in: outputs/results/")
        
    elif args.mode == 'analyze':
        from analysis import run_sensitivity_analysis, print_sensitivity_summary
        from models import create_model
        
        # Override config with CLI args
        if args.csv:
            config['data']['csv_path'] = args.csv
        if args.model:
            config['training']['model_type'] = args.model
        
        csv_path = config['data']['csv_path']
        model_type = config['training']['model_type']
        area_name = Path(csv_path).stem
        output_name = f"{area_name}_{model_type}"
        
        seed_everything(config.get('training', {}).get('seed', 42))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🖥️  Device: {device}")
        print(f"\n📂 Dataset: {area_name}")
        print(f"🧠 Model: {model_type.upper()}")
        
        # Load data
        print("\n⏳ Loading data...")
        data_loaders = prepare_data(csv_path, config, device)
        input_size = len(data_loaders.feature_cols)
        
        # Create model and load checkpoint
        model, soil_physics = create_model(config, input_size, device)
        
        # Load best checkpoint
        models_dir = Path(config.get('output', {}).get('models_dir', 'outputs/models'))
        checkpoint_path = models_dir / f"{output_name}_best_val.pt"
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("   Run training first with --mode train")
            return
        
        print(f"\n📦 Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        soil_physics.load_state_dict(checkpoint['soil_physics_state_dict'])
        
        # Run sensitivity analysis
        print("\n🔬 Running sensitivity analysis...")
        df_results = run_sensitivity_analysis(
            model=model,
            soil_physics=soil_physics,
            data_loader=data_loaders.test if len(data_loaders.test) > 0 else data_loaders.val,
            u_mean=data_loaders.u_mean,
            u_scale=data_loaders.u_scale,
            config=config,
            output_name=output_name,
            device=device
        )
        
        # Print summary
        print_sensitivity_summary(df_results)
        
        print("\n" + "="*50)
        print("✅ SENSITIVITY ANALYSIS COMPLETE")
        print("="*50)
        print(f"Results: outputs/results/{output_name}_sensitivity.csv")
        print(f"Plot:    outputs/figures/{output_name}_sensitivity.png")


if __name__ == "__main__":
    main()
