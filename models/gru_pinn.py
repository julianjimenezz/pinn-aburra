"""
GRU-based Physics-Informed Neural Network.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .base_pinn import BasePINN


class GRUPINN(BasePINN):
    """
    GRU-based PINN for landslide deformation prediction.
    
    Architecture:
        GRU layers → FC output layer
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor, u_physics: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_size)
            u_physics: Physics-based prediction (batch, 1), optional
        
        Returns:
            Predictions: (batch, 1)
        """
        # GRU output: (batch, seq_len, hidden_size)
        out, _ = self.gru(x)
        # Take last time step
        out = out[:, -1, :]
        # FC layer → correction term (Δu)
        delta_u = self.fc(out)
        # Add physics baseline if provided
        if u_physics is not None:
            return u_physics + delta_u
        return delta_u
    
    @property
    def model_type(self) -> str:
        return "gru"


def create_gru_pinn(config: Dict[str, Any], input_size: int) -> GRUPINN:
    """Factory function to create GRU PINN from config."""
    gru_config = config.get('gru', {})
    return GRUPINN(
        input_size=input_size,
        hidden_size=gru_config.get('hidden_size', 128),
        num_layers=gru_config.get('num_layers', 2),
        dropout=gru_config.get('dropout', 0.3)
    )
