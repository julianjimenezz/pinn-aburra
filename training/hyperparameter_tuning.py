"""
Hyperparameter tuning module using Optuna.
"""

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable
import yaml
import json

from models import GRUPINN, LSTMPINN, SoilPhysics


def create_objective(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    u_mean: float,
    u_scale: float,
    input_size: int,
    device: str = "cuda",
    n_epochs: int = 30
) -> Callable:
    """
    Create Optuna objective function for hyperparameter tuning.
    
    Args:
        model_type: "gru" or "lstm"
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Base configuration
        u_mean, u_scale: Scaling parameters
        input_size: Number of input features
        device: Device to train on
        n_epochs: Epochs per trial (fewer for tuning)
    
    Returns:
        Objective function for Optuna
    """
    search_space = config.get('tuning', {}).get(f'{model_type}_search_space', {})
    physics_config = config.get('physics', {})
    
    def objective(trial: Trial) -> float:
        # Sample hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', 
            search_space.get('hidden_size', [64, 128, 256]))
        num_layers = trial.suggest_categorical('num_layers',
            search_space.get('num_layers', [1, 2, 3]))
        dropout = trial.suggest_float('dropout',
            search_space.get('dropout', [0.1, 0.5])[0],
            search_space.get('dropout', [0.1, 0.5])[1])
        learning_rate = trial.suggest_float('learning_rate',
            search_space.get('learning_rate', [0.0001, 0.01])[0],
            search_space.get('learning_rate', [0.0001, 0.01])[1],
            log=True)
        
        # Create model
        if model_type == 'gru':
            model = GRUPINN(input_size, hidden_size, num_layers, dropout)
        else:
            model = LSTMPINN(input_size, hidden_size, num_layers, dropout)
        
        model = model.to(device)
        
        # Create physics
        soil_physics = SoilPhysics(
            h=physics_config.get('h', 15.0),
            tau_fixed=physics_config.get('tau_fixed', 15.0),
            device=device
        )
        
        # Optimizer
        optimizer = optim.Adam(
            list(model.parameters()) + list(soil_physics.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()
        lambda_phy = physics_config.get('lambda_phy', 0.0001)
        dt_days = physics_config.get('dt_days', 12.0)
        
        # Training loop (shortened for tuning)
        best_val_rmse = float('inf')
        
        for epoch in range(n_epochs):
            model.train()
            
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                
                optimizer.zero_grad()
                
                # Compute physics baseline
                u_prev_scaled = X_b[:, -1, 0].view(-1, 1)
                u_prev = (u_prev_scaled * u_scale) + u_mean
                u_physics = soil_physics.compute_displacement(u_prev, dt_days)
                u_physics_scaled = (u_physics - u_mean) / u_scale
                
                # Forward pass: model outputs u_physics + delta_u
                u_pred_scaled = model(X_b, u_physics_scaled)
                loss_data = criterion(u_pred_scaled, y_b.view(-1, 1))
                
                # Physics loss
                u_pred = (u_pred_scaled * u_scale) + u_mean
                loss_physics = soil_physics.perzyna_loss(u_pred, u_prev, dt_days)
                
                loss = loss_data + (lambda_phy * loss_physics)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Validate
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    
                    # Compute physics baseline
                    u_prev_scaled = X_b[:, -1, 0].view(-1, 1)
                    u_prev = (u_prev_scaled * u_scale) + u_mean
                    u_physics = soil_physics.compute_displacement(u_prev, dt_days)
                    u_physics_scaled = (u_physics - u_mean) / u_scale
                    
                    pred = model(X_b, u_physics_scaled)
                    pred_real = (pred * u_scale) + u_mean
                    target_real = (y_b.view(-1, 1) * u_scale) + u_mean
                    preds.extend(pred_real.cpu().numpy().flatten())
                    targets.extend(target_real.cpu().numpy().flatten())
            
            val_rmse = np.sqrt(np.mean((np.array(preds) - np.array(targets))**2))
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
            
            # Pruning
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_rmse
    
    return objective


def run_hyperparameter_tuning(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    u_mean: float,
    u_scale: float,
    input_size: int,
    output_name: str,
    device: str = "cuda"
) -> optuna.Study:
    """
    Run hyperparameter search using Optuna.
    
    Args:
        model_type: "gru" or "lstm"
        train_loader, val_loader: Data loaders
        config: Configuration dict
        u_mean, u_scale: Scaling parameters
        input_size: Number of features
        output_name: Name for saving results
        device: Compute device
    
    Returns:
        Optuna study with results
    """
    tuning_cfg = config.get('tuning', {})
    n_trials = tuning_cfg.get('n_trials', 50)
    
    results_dir = Path(config.get('output', {}).get('results_dir', 'outputs/results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Starting hyperparameter tuning ({n_trials} trials)...")
    print(f"   Model: {model_type.upper()}")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Create objective
    objective = create_objective(
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        u_mean=u_mean,
        u_scale=u_scale,
        input_size=input_size,
        device=device
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n✨ Best trial:")
    print(f"   Val RMSE: {best_value:.5f}")
    print(f"   Params: {best_params}")
    
    # Save to file
    tuning_results = {
        'model_type': model_type,
        'best_params': best_params,
        'best_val_rmse': best_value,
        'n_trials': n_trials,
        'n_completed': len(study.trials)
    }
    
    results_path = results_dir / f"{output_name}_tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    print(f"\n📁 Saved: {results_path}")
    
    return study
