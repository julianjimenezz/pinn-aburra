"""
Models package for PINN implementations.
"""

from .physics import SoilPhysics, create_soil_physics
from .base_pinn import BasePINN
from .gru_pinn import GRUPINN, create_gru_pinn
from .lstm_pinn import LSTMPINN, create_lstm_pinn

__all__ = [
    'SoilPhysics',
    'create_soil_physics',
    'BasePINN',
    'GRUPINN',
    'create_gru_pinn',
    'LSTMPINN',
    'create_lstm_pinn'
]


def create_model(config: dict, input_size: int, device: str = "cuda", init_params: dict = None):
    """
    Factory function to create appropriate model based on config.
    
    Args:
        config: Configuration dict with 'training.model_type'
        input_size: Number of input features
        device: Device to place model on
        init_params: Optional dictionary of initial physics parameters
    
    Returns:
        Tuple of (model, soil_physics)
    """
    model_type = config.get('training', {}).get('model_type', 'gru')
    
    if model_type == 'gru':
        model = create_gru_pinn(config, input_size)
    elif model_type == 'lstm':
        model = create_lstm_pinn(config, input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    soil_physics = create_soil_physics(config, device, init_params=init_params)
    
    return model, soil_physics
