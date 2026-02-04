"""
N2V (Noise2Void) Module - Optimized Implementation

This module implements Noise2Void with the exact "Best Performance" configuration
from the optimization study:
- U-Net: depth=3, start_channels=4, Kaiming initialization
- Masking: 7x7 patches, 10% mask ratio, zero replacement
- Validation: Static 2% validation mask
- Training: Early stopping with patience=20

Usage:
    from ppn2v.n2v import N2VUNet, N2VTrainer, create_n2v_model
    
    # Create model
    model = create_n2v_model(device)
    
    # Create trainer
    trainer = N2VTrainer(model, device)
    
    # Train
    trainer.train(noisy_image, max_epochs=100, patience=20)
    
    # Predict
    denoised = trainer.predict(noisy_image)
"""

from .model import N2VUNet, create_n2v_model
from .training import N2VTrainer, load_model

__all__ = [
    'N2VUNet',
    'N2VTrainer', 
    'create_n2v_model',
    'load_model'
]
