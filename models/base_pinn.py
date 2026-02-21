"""
Base PINN module: Abstract base class for LSTM/GRU models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BasePINN(nn.Module, ABC):
    """
    Abstract base class for Physics-Informed Neural Networks.
    
    Subclasses must implement:
        - forward(): prediction from input sequence
        - model_type: string identifier ("lstm" or "gru")
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
    
    @abstractmethod
    def forward(self, x: torch.Tensor, u_physics: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            u_physics: Physics-based displacement prediction (batch, 1).
                       When provided, the model outputs u_physics + delta_u (residual mode).
                       When None, outputs delta_u directly (legacy mode).
        
        Returns:
            Predictions of shape (batch, 1)
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier ('lstm' or 'gru')."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }
