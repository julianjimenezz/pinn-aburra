"""
Training module with unified training loop, checkpointing, and evaluation.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from models import BasePINN, SoilPhysics
from visualization import plot_test_results


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss_total: float
    train_loss_data: float
    train_loss_physics: float
    val_loss_total: float
    val_rmse: float
    val_r2: float
    physics_balance: float  # |data_loss - lambda * physics_loss|


@dataclass
class TrainingHistory:
    """Complete training history."""
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_val_rmse: float = float('inf')
    best_val_epoch: int = 0
    best_physics_balance: float = float('inf')
    best_physics_epoch: int = 0
    best_total_loss: float = float('inf')
    best_total_epoch: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for saving."""
        records = []
        for m in self.epochs:
            records.append({
                'epoch': m.epoch,
                'train_loss_total': m.train_loss_total,
                'train_loss_data': m.train_loss_data,
                'train_loss_physics': m.train_loss_physics,
                'val_loss_total': m.val_loss_total,
                'val_rmse': m.val_rmse,
                'val_r2': m.val_r2,
                'physics_balance': m.physics_balance
            })
        return pd.DataFrame(records)


class PINNTrainer:
    """
    Unified trainer for PINN models with train/val/test splits.
    
    Features:
        - Combined data + physics loss
        - Multi-strategy checkpointing
        - Early stopping on validation loss
        - Comprehensive history tracking
    """
    
    def __init__(
        self,
        model: BasePINN,
        soil_physics: SoilPhysics,
        config: Dict[str, Any],
        u_mean: float,
        u_scale: float,
        device: str = "cuda",
        output_name: str = "model"
    ):
        self.model = model
        self.soil_physics = soil_physics
        self.config = config
        self.u_mean = u_mean
        self.u_scale = u_scale
        self.device = device
        self.output_name = output_name
        
        # Get config sections
        training_cfg = config.get('training', {})
        model_type = training_cfg.get('model_type', 'gru')
        model_cfg = config.get(model_type, {})
        physics_cfg = config.get('physics', {})
        checkpoint_cfg = config.get('checkpointing', {})
        output_cfg = config.get('output', {})
        
        # Training params
        self.epochs = training_cfg.get('epochs', 100)
        self.lr = model_cfg.get('learning_rate', 0.001)
        self.lambda_phy = physics_cfg.get('lambda_phy', 0.0001)
        self.dt_days = physics_cfg.get('dt_days', 12.0)
        self.patience = training_cfg.get('early_stopping_patience', 15)
        
        # Checkpointing
        self.save_best_val = checkpoint_cfg.get('save_best_val', True)
        self.save_best_physics = checkpoint_cfg.get('save_best_physics_balance', True)
        self.save_every_n = checkpoint_cfg.get('save_every_n_epochs', 10)
        self.save_best_total = checkpoint_cfg.get('save_best_total', True)
        
        # Output directories
        self.models_dir = Path(output_cfg.get('models_dir', 'outputs/models'))
        self.results_dir = Path(output_cfg.get('results_dir', 'outputs/results'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter parameters to only those that require gradients (trainable)
        trainable_params = [p for p in list(model.parameters()) + list(soil_physics.parameters()) if p.requires_grad]
        
        # Optimizer with trainable parameters
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.lr
        )
        self.criterion = nn.MSELoss()
        
        # History
        self.history = TrainingHistory()
    
    def _descale_u(self, u_scaled: torch.Tensor) -> torch.Tensor:
        """Convert scaled u back to original units (m)."""
        return (u_scaled * self.u_scale) + self.u_mean
    
    def _scale_u(self, u_m: torch.Tensor) -> torch.Tensor:
        """Convert u in meters to scaled units."""
        return (u_m - self.u_mean) / self.u_scale
    
    def _compute_u_physics_scaled(self, X_b: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-based displacement prediction in scaled units.
        
        Extracts u_prev from the last time step of the input sequence,
        runs Perzyna forward, and re-scales the result.
        
        Args:
            X_b: Input batch (batch, seq_len, input_size)
        
        Returns:
            u_physics_scaled: (batch, 1) in scaled units
        """
        u_prev_scaled = X_b[:, -1, 0].view(-1, 1)  # Last u in sequence
        u_prev = self._descale_u(u_prev_scaled)
        u_physics = self.soil_physics.compute_displacement(u_prev, self.dt_days)
        return self._scale_u(u_physics)
    
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a data loader, return RMSE and R2."""
        if len(loader) == 0:
            return 0.0, 0.0  # Return defaults for empty loader
        
        self.model.eval()
        preds, targets = [], []
        
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                
                # Compute physics baseline
                u_physics_scaled = self._compute_u_physics_scaled(X_b)
                
                pred_scaled = self.model(X_b, u_physics_scaled)
                pred = self._descale_u(pred_scaled)
                target = self._descale_u(y_b.view(-1, 1))
                
                preds.extend(pred.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())
        
        preds = np.array(preds)
        targets = np.array(targets)
        
        if len(preds) == 0:
            return 0.0, 0.0
        
        rmse = np.sqrt(mean_squared_error(targets, preds))
        r2 = r2_score(targets, preds)
        
        return rmse, r2
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch, return (total_loss, data_loss, physics_loss)."""
        self.model.train()
        total_loss_sum = 0.0
        data_loss_sum = 0.0
        physics_loss_sum = 0.0
        n_batches = 0
        
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute physics baseline (in scaled units)
            u_physics_scaled = self._compute_u_physics_scaled(X_b)
            
            # Forward pass: model outputs u_physics + delta_u
            u_pred_scaled = self.model(X_b, u_physics_scaled)
            
            # Data loss (scaled)
            loss_data = self.criterion(u_pred_scaled, y_b.view(-1, 1))
            
            # Physics loss (need real units)
            u_pred = self._descale_u(u_pred_scaled)
            u_prev_scaled = X_b[:, -1, 0].view(-1, 1)  # Last u in sequence
            u_prev = self._descale_u(u_prev_scaled)
            
            loss_physics = self.soil_physics.perzyna_loss(
                u_pred, u_prev, self.dt_days
            )
            
            # Combined loss
            loss_total = loss_data + (self.lambda_phy * loss_physics)
            
            # Backward
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss_sum += loss_total.item()
            data_loss_sum += loss_data.item()
            physics_loss_sum += loss_physics.item()
            n_batches += 1
        
        return (
            total_loss_sum / n_batches,
            data_loss_sum / n_batches,
            physics_loss_sum / n_batches
        )
    
    def _save_checkpoint(self, epoch: int, metrics: EpochMetrics, suffix: str):
        """Save model checkpoint."""
        path = self.models_dir / f"{self.output_name}_{suffix}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'soil_physics_state_dict': self.soil_physics.state_dict(),
            'metrics': {
                'train_loss_total': metrics.train_loss_total,
                'train_loss_data': metrics.train_loss_data,
                'train_loss_physics': metrics.train_loss_physics,
                'val_loss_total': metrics.val_loss_total,
                'val_rmse': metrics.val_rmse,
                'val_r2': metrics.val_r2,
            },
            'physics_params': self.soil_physics.get_params_dict(),
            'model_config': self.model.get_config(),
        }, path)
        print(f"  📁 Saved: {path.name}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> TrainingHistory:
        """
        Full training loop with validation and checkpointing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Print progress
        
        Returns:
            TrainingHistory with all metrics
        """
        patience_counter = 0
        
        if verbose:
            print(f"{'Epoch':<7} | {'Total':<10} | {'Data':<10} | {'Physics':<12} | {'λ×Phy':<10} | {'Val RMSE':<10} | {'Val R²':<8}")
            print("-" * 85)
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_total, train_data, train_physics = self._train_epoch(train_loader)
            
            # Validate
            val_rmse, val_r2 = self._evaluate(val_loader)
            val_total = train_data  # Approximate val loss
            
            # Physics balance metric
            physics_balance = abs(train_data - self.lambda_phy * train_physics)
            
            # Create metrics
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss_total=train_total,
                train_loss_data=train_data,
                train_loss_physics=train_physics,
                val_loss_total=val_total,
                val_rmse=val_rmse,
                val_r2=val_r2,
                physics_balance=physics_balance
            )
            self.history.epochs.append(metrics)
            
            # Check for best models
            improved = False
            
            # Best validation RMSE
            if val_rmse < self.history.best_val_rmse:
                self.history.best_val_rmse = val_rmse
                self.history.best_val_epoch = epoch
                improved = True
                if self.save_best_val:
                    self._save_checkpoint(epoch, metrics, "best_val")
            
            # Best physics balance
            if physics_balance < self.history.best_physics_balance:
                self.history.best_physics_balance = physics_balance
                self.history.best_physics_epoch = epoch
                if self.save_best_physics:
                    self._save_checkpoint(epoch, metrics, "best_physics")
            
            # Best total loss
            if train_total < self.history.best_total_loss:
                self.history.best_total_loss = train_total
                self.history.best_total_epoch = epoch
                if self.save_best_total:
                    self._save_checkpoint(epoch, metrics, "best_total")
            
            # Periodic checkpoints
            if self.save_every_n and epoch % self.save_every_n == 0:
                self._save_checkpoint(epoch, metrics, f"epoch_{epoch}")
            
            # Early stopping
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                if verbose:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"{epoch:<7} | {train_total:<10.5f} | {train_data:<10.5f} | {train_physics:<12.2e} | {self.lambda_phy * train_physics:<10.5f} | {val_rmse:<10.4f} | {val_r2:<8.4f}")
        
        # Save final
        self._save_checkpoint(epoch, metrics, "final")
        
        # Save history
        history_path = self.results_dir / f"{self.output_name}_history.csv"
        self.history.to_dataframe().to_csv(history_path, index=False)
        print(f"\n📊 History saved: {history_path}")
        
        # Save config summary
        config_path = self.results_dir / f"{self.output_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': self.model.model_type,
                'model_config': self.model.get_config(),
                'epochs_trained': epoch,
                'best_val_rmse': self.history.best_val_rmse,
                'best_val_epoch': self.history.best_val_epoch,
                'best_val_r2': max(m.val_r2 for m in self.history.epochs),
                'physics_params_final': self.soil_physics.get_params_dict(),
            }, f, indent=2)
        print(f"⚙️  Config saved: {config_path}")
        
        return self.history
    
    def evaluate_test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Final evaluation on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dict with RMSE, MAE, R2
        """
        if len(test_loader) == 0:
            print("\n⚠️  Test set is empty - skipping test evaluation")
            print("   (Adjust temporal_cutoff_val in config.yaml if needed)")
            return {'test_rmse': None, 'test_mae': None, 'test_r2': None}
        
        self.model.eval()
        preds, targets = [], []
        
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                
                # Compute physics baseline
                u_physics_scaled = self._compute_u_physics_scaled(X_b)
                
                pred_scaled = self.model(X_b, u_physics_scaled)
                pred = self._descale_u(pred_scaled)
                target = self._descale_u(y_b.view(-1, 1))
                
                preds.extend(pred.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())
        
        preds = np.array(preds)
        targets = np.array(targets)
        
        results = {
            'test_rmse': np.sqrt(mean_squared_error(targets, preds)),
            'test_mae': mean_absolute_error(targets, preds),
            'test_r2': r2_score(targets, preds)
        }
        
        print("\n--- 📊 TEST RESULTS ---")
        print(f"RMSE: {results['test_rmse']:.5f} m")
        print(f"MAE:  {results['test_mae']:.5f} m")
        print(f"R²:   {results['test_r2']:.4f}")
        
        # --- Visualizations ---
        if self.results_dir and self.output_name:
            figures_dir = self.results_dir.parent / "figures"
            plot_test_results(targets, preds, self.output_name, figures_dir)
            
        # --- Geotechnical Report ---
        print("\n--- 🌍 GEOTECHNICAL REPORT ---")
        params = self.soil_physics.get_params_dict()
        for k, v in params.items():
            print(f"   {k}: {v:.4f}")
        
        return results

