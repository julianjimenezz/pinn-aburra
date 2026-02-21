"""
LSTM-based Physics-Informed Neural Network.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .base_pinn import BasePINN


class LSTMPINN(BasePINN):
    """
    LSTM-based PINN for landslide deformation prediction.
    
    Architecture:
        LSTM layers → FC output layer
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        
        self.lstm = nn.LSTM(
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
        # LSTM output: (batch, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
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
        return "lstm"


def create_lstm_pinn(config: Dict[str, Any], input_size: int) -> LSTMPINN:
    """Factory function to create LSTM PINN from config."""
    lstm_config = config.get('lstm', {})
    return LSTMPINN(
        input_size=input_size,
        hidden_size=lstm_config.get('hidden_size', 128),
        num_layers=lstm_config.get('num_layers', 2),
        dropout=lstm_config.get('dropout', 0.3)
    )
