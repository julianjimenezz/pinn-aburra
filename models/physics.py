"""
Physics module: Perzyna viscoplastic soil model for PINN.

This module implements the Perzyna model with learnable geotechnical parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SoilPhysics(nn.Module):
    """
    Perzyna viscoplastic model for soil behavior.
    
    Learnable parameters (constrained via softplus):
        - c_prime: effective cohesion (kPa)
        - phi: friction angle (degrees)
        - n_exp: flow exponent
        - sigma_n: normal stress (kPa)
        - eta: viscosity (kPa·day)
        - h: depth (m)
        - tau: applied shear stress (kPa)
    
    Fixed parameters:
        - sigma_0: normalization constant (kPa)
    """
    
    def __init__(
        self,
        h: float = 15.0,
        tau: float = 15.0,
        sigma_0: float = 1.0,
        device: str = "cuda",
        init_params: Dict[str, float] = None,
        trainable_params: Dict[str, bool] = None,
        bounds: Dict[str, list] = None
    ):
        super().__init__()
        
        # Fixed parameters
        self.sigma_0 = sigma_0
        self.device = device
        
        # Default initialization values
        defaults = {
            "c_prime": 0.0,
            "phi": 0.0,
            "n": 0.0,       # n_exp in raw is n - 1
            "sigma_n": 0.0,
            "eta": 0.0,
            "h": h,         # Fallback to the h passed in constructor
            "tau": tau      # Fallback to the tau passed in constructor
        }
        
        # Override with provided init_params if any
        if init_params:
            defaults.update(init_params)
            
        # Default trainable values (all true except h and tau which were historically fixed)
        trainable = {
            "c_prime": True,
            "phi": True,
            "n": True,
            "sigma_n": True,
            "eta": True,
            "h": False,
            "tau": False
        }
        
        # Override with provided trainable_params if any
        if trainable_params:
            trainable.update(trainable_params)
            
        # Default bounds [min, max]
        self.bounds = {
            "c_prime": [0.1, 50.0],
            "phi": [15.0, 45.0],
            "n": [1.1, 5.0],
            "sigma_n": [10.0, 200.0],
            "eta": [10000.0, 5000000.0],
            "h": [5.0, 50.0],
            "tau": [10.0, 500.0]
        }
        
        if bounds:
            self.bounds.update(bounds)
            
        # Learnable parameters (raw values, will be transformed via scaled sigmoid)
        self.c_prime_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_scaled(defaults["c_prime"], self.bounds["c_prime"]), device=device), requires_grad=trainable["c_prime"])
        self.phi_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_scaled(defaults["phi"], self.bounds["phi"]), device=device), requires_grad=trainable["phi"])
        self.n_exp_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_scaled(defaults["n"], self.bounds["n"]), device=device), requires_grad=trainable["n"])
        self.sigma_n_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_scaled(defaults["sigma_n"], self.bounds["sigma_n"]), device=device), requires_grad=trainable["sigma_n"])
        self.eta_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_scaled(defaults["eta"], self.bounds["eta"]), device=device), requires_grad=trainable["eta"])
        self.h_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_scaled(defaults["h"], self.bounds["h"]), device=device), requires_grad=trainable["h"])
        self.tau_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_scaled(defaults["tau"], self.bounds["tau"]), device=device), requires_grad=trainable["tau"])
    
    @staticmethod
    def _inv_sigmoid_scaled(y: float, bounds: list) -> float:
        """
        Inverse of scaled sigmoid to initialize raw parameter.
        y = min + (max - min) * sigmoid(x)
        sigmoid(x) = (y - min) / (max - min)
        x = log(sigmoid(x) / (1 - sigmoid(x)))
        """
        import math
        b_min, b_max = bounds
        # Clamp to avoid strictly 0 or 1 which throws math domain error
        clamped_y = max(min(y, b_max - 1e-6), b_min + 1e-6)
        sig_x = (clamped_y - b_min) / (b_max - b_min)
        return math.log(sig_x / (1.0 - sig_x))
        
    def _sigmoid_scaled(self, raw_tensor: torch.Tensor, bounds: list) -> torch.Tensor:
        """
        Forward scaled sigmoid to constrain raw parameters between [min, max].
        """
        b_min, b_max = bounds
        return b_min + (b_max - b_min) * torch.sigmoid(raw_tensor)
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get constrained physics parameters within their bounds."""
        return {
            "c_prime": self._sigmoid_scaled(self.c_prime_raw, self.bounds["c_prime"]),
            "phi": self._sigmoid_scaled(self.phi_raw, self.bounds["phi"]),
            "n": self._sigmoid_scaled(self.n_exp_raw, self.bounds["n"]),
            "sigma_n": self._sigmoid_scaled(self.sigma_n_raw, self.bounds["sigma_n"]),
            "eta": self._sigmoid_scaled(self.eta_raw, self.bounds["eta"]),
            "h": self._sigmoid_scaled(self.h_raw, self.bounds["h"]),
            "tau": self._sigmoid_scaled(self.tau_raw, self.bounds["tau"])
        }
    
    def get_params_dict(self) -> Dict[str, float]:
        """Get parameters as Python floats for logging."""
        params = self.get_params()
        return {k: v.item() for k, v in params.items()}
    
    def compute_displacement(
        self,
        u_prev: torch.Tensor,
        dt_days: float = 12.0
    ) -> torch.Tensor:
        """
        Compute physics-based displacement prediction (forward Perzyna).
        
        Uses the Perzyna constitutive law to predict the next displacement
        from the previous one: u_physics = u_prev + γ̇ · h · dt
        
        Args:
            u_prev: Previous displacement (m), shape (batch, 1)
            dt_days: Time step between observations (days)
        
        Returns:
            u_physics: Physics-predicted displacement (m), shape (batch, 1)
        """
        params = self.get_params()
        
        # Mohr-Coulomb yield strength
        phi_rad = torch.deg2rad(params['phi'])
        yield_strength = params['c_prime'] + params['sigma_n'] * torch.tan(phi_rad)
        
        # Overstress (normalized)
        overstress = (params['tau'] - yield_strength) / self.sigma_0
        
        # Perzyna dynamic strain rate: γ̇ = (1/η) · relu(overstress)^n
        # Using relu enforces strict physics: 0 strain rate if overstress <= 0
        gamma_dot = (1.0 / params['eta']) * torch.pow(
            F.relu(overstress), params['n']
        )
        
        # Strain rate → velocity (m/day): v = γ̇ · h (m)
        velocity_m_per_day = gamma_dot * params['h']
        
        # Clamp velocity to data-appropriate range
        # (for InSAR-scale displacements, max ~0.00001 m/day ≈ 3.6 mm/year)
        velocity_m_per_day = torch.clamp(velocity_m_per_day, min=0.0, max=0.00001)
        
        # Forward Euler integration: u_physics = u_prev + v · dt
        u_physics = u_prev + velocity_m_per_day * dt_days
        
        return u_physics
    
    def perzyna_loss(
        self,
        u_pred: torch.Tensor,
        u_prev: torch.Tensor,
        dt_days: float = 12.0
    ) -> torch.Tensor:
        """
        Compute Perzyna viscoplastic physics-based loss.
        
        Compares kinematic strain rate (from displacement) with
        dynamic strain rate (from Perzyna constitutive law).
        
        Args:
            u_pred: Predicted displacement at time t (m)
            u_prev: Previous displacement at time t-1 (m)
            dt_days: Time step between observations (days)
        
        Returns:
            Physics loss (MSE between kinematic and dynamic strain rates)
        """
        params = self.get_params()
        
        # Kinematic strain rate: du/dt / h
        velocity = (u_pred - u_prev) / dt_days
        gamma_dot_kin = velocity / params['h']
        
        # Mohr-Coulomb yield strength
        phi_rad = torch.deg2rad(params['phi'])
        yield_strength = params['c_prime'] + params['sigma_n'] * torch.tan(phi_rad)
        
        # Overstress (normalized)
        overstress = (params['tau'] - yield_strength) / self.sigma_0
        
        # Perzyna dynamic strain rate: (1/eta) * relu(overstress)^n
        # Using relu enforces strict physics: 0 strain rate if overstress <= 0
        gamma_dot_dyn = (1.0 / params['eta']) * torch.pow(
            F.relu(overstress), params['n']
        )
        
        # Physics loss: difference between kinematic and dynamic
        return torch.mean((gamma_dot_kin - gamma_dot_dyn) ** 2)
    
    def forward(
        self,
        u_pred: torch.Tensor,
        u_prev: torch.Tensor,
        dt_days: float = 12.0
    ) -> torch.Tensor:
        """Alias for perzyna_loss for nn.Module compatibility."""
        return self.perzyna_loss(u_pred, u_prev, dt_days)


def create_soil_physics(config: Dict[str, Any], device: str = "cuda", init_params: Dict[str, float] = None, trainable_params: Dict[str, bool] = None) -> SoilPhysics:
    """Factory function to create SoilPhysics from config dict."""
    physics_config = config.get('physics', {})
    
    # 1. Start with any explicit overrides passed to function (e.g. from Group Analysis)
    final_init_params = init_params.copy() if init_params else {}
    
    # 2. If no explicit overrides for a key, look in config['physics']['initial_params']
    config_init = physics_config.get('initial_params', {})
    for k, v in config_init.items():
        if k not in final_init_params:
            final_init_params[k] = v
            
    # 3. Handle trainable parameters
    final_trainable = trainable_params.copy() if trainable_params else {}
    config_trainable = physics_config.get('trainable', {})
    for k, v in config_trainable.items():
        if k not in final_trainable:
            final_trainable[k] = v
            
    # 4. Handle bounds
    final_bounds = {}
    config_bounds = physics_config.get('bounds', {})
    for k, v in config_bounds.items():
        if isinstance(v, list) and len(v) == 2:
            final_bounds[k] = v
            
    return SoilPhysics(
        h=physics_config.get('h', 15.0),
        tau=physics_config.get('tau_fixed', 100.0), # Fallback to tau_fixed for backwards compatibility
        sigma_0=physics_config.get('sigma_0', 1.0),
        device=device,
        init_params=final_init_params,
        trainable_params=final_trainable,
        bounds=final_bounds if final_bounds else None
    )
