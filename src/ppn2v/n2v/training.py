"""
Noise2Void Training Module - Optimized Configuration

Implements the exact "Best Performance" configuration:
- 7x7 mask patches (larger blind spots)
- 10% mask ratio per iteration
- Zero replacement for masked pixels
- Static validation mask for early stopping
- MSE loss on masked pixels only
"""

import torch
import torch.optim as optim
import numpy as np
import os
import time
from datetime import datetime


class N2VTrainer:
    """
    Noise2Void Trainer with Optimized Configuration
    
    Key features from optimization study:
    - 7x7 mask patches (not 5x5)
    - Zero replacement (not random neighbor)
    - Static validation mask (2% of pixels)
    - Early stopping based on validation mask MSE
    
    Parameters
    ----------
    model : nn.Module
        The N2V U-Net model
    device : torch.device
        GPU or CPU device
    learning_rate : float
        Learning rate for Adam optimizer (default: 0.001)
    mask_ratio : float
        Fraction of pixels to mask per iteration (default: 0.1)
    mask_patch_size : int
        Size of each blind spot patch (default: 7)
    val_mask_ratio : float
        Fraction of pixels for static validation mask (default: 0.02)
    """
    
    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        mask_ratio=0.1,
        mask_patch_size=7,
        val_mask_ratio=0.02
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.val_mask_ratio = val_mask_ratio
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = torch.nn.MSELoss(reduction='none')
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        # Static validation mask (created once before training)
        self.val_mask = None
        
        print(f"N2V Trainer initialized:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Mask ratio: {mask_ratio * 100}%")
        print(f"  Mask patch size: {mask_patch_size}x{mask_patch_size}")
        print(f"  Validation mask ratio: {val_mask_ratio * 100}%")
    
    def _create_static_val_mask(self, image_shape):
        """
        Create a static validation mask (held out throughout training)
        
        Parameters
        ----------
        image_shape : tuple
            Shape of the image (H, W)
        
        Returns
        -------
        numpy.ndarray
            Binary mask with 1s at validation pixel positions
        """
        H, W = image_shape
        total_pixels = H * W
        num_val_pixels = int(total_pixels * self.val_mask_ratio)
        
        # Randomly select validation pixel positions
        mask = np.zeros((H, W), dtype=np.float32)
        indices = np.random.choice(total_pixels, num_val_pixels, replace=False)
        rows = indices // W
        cols = indices % W
        mask[rows, cols] = 1.0
        
        print(f"Created static validation mask: {num_val_pixels} pixels ({self.val_mask_ratio*100}%)")
        return mask
    
    def _create_training_mask(self, image_shape):
        """
        Create a random training mask for one iteration
        
        The mask uses 7x7 patches and replaces with zeros (not neighbors)
        Avoids validation mask pixels
        
        Parameters
        ----------
        image_shape : tuple
            Shape of the image (H, W)
        
        Returns
        -------
        mask : numpy.ndarray
            Binary mask indicating which pixels to predict (1) vs use as input (0)
        masked_image : Not used here, masking done separately
        """
        H, W = image_shape
        total_pixels = H * W
        num_mask_pixels = int(total_pixels * self.mask_ratio)
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.float32)
        
        # Stratified sampling: divide image into grid cells
        patch_size = self.mask_patch_size
        half_patch = patch_size // 2
        
        # Calculate grid
        num_cells_h = max(1, H // patch_size)
        num_cells_w = max(1, W // patch_size)
        pixels_per_cell = max(1, num_mask_pixels // (num_cells_h * num_cells_w))
        
        for i in range(num_cells_h):
            for j in range(num_cells_w):
                # Cell boundaries
                y_start = i * (H // num_cells_h)
                y_end = (i + 1) * (H // num_cells_h) if i < num_cells_h - 1 else H
                x_start = j * (W // num_cells_w)
                x_end = (j + 1) * (W // num_cells_w) if j < num_cells_w - 1 else W
                
                # Sample pixels within this cell
                for _ in range(pixels_per_cell):
                    y = np.random.randint(y_start + half_patch, max(y_start + half_patch + 1, y_end - half_patch))
                    x = np.random.randint(x_start + half_patch, max(x_start + half_patch + 1, x_end - half_patch))
                    
                    # Avoid validation mask pixels
                    if self.val_mask is not None and self.val_mask[y, x] == 1:
                        continue
                    
                    mask[y, x] = 1.0
        
        return mask
    
    def _apply_mask_to_image(self, image, mask):
        """
        Apply blind-spot mask to image: replace masked pixels with ZERO
        
        Parameters
        ----------
        image : numpy.ndarray
            Original noisy image
        mask : numpy.ndarray
            Binary mask (1 = masked, 0 = visible)
        
        Returns
        -------
        numpy.ndarray
            Image with masked pixels set to zero
        """
        masked_image = image.copy()
        masked_image[mask == 1] = 0.0  # Replace with zero (not random neighbor!)
        return masked_image
    
    def _compute_loss(self, prediction, target, mask):
        """
        Compute MSE loss only on masked pixels
        
        Parameters
        ----------
        prediction : torch.Tensor
            Network output
        target : torch.Tensor
            Original noisy image (ground truth for self-supervision)
        mask : torch.Tensor
            Binary mask indicating which pixels to compute loss on
        
        Returns
        -------
        torch.Tensor
            Mean MSE over masked pixels only
        """
        # Element-wise squared error
        mse = self.criterion(prediction, target)
        
        # Apply mask and compute mean over masked pixels only
        masked_mse = mse * mask
        loss = masked_mse.sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def train(
        self,
        image,
        max_epochs=100,
        patience=20,
        save_dir='.',
        model_name='n2v_optimized',
        verbose=True
    ):
        """
        Train N2V on a single noisy image with early stopping
        
        Parameters
        ----------
        image : numpy.ndarray
            Single noisy 2D image to denoise
        max_epochs : int
            Maximum number of training epochs (default: 100)
        patience : int
            Early stopping patience (default: 20)
        save_dir : str
            Directory to save model checkpoints
        model_name : str
            Name for saved model files
        verbose : bool
            Print training progress
        
        Returns
        -------
        train_history : list
            Training loss per epoch
        val_history : list
            Validation loss per epoch
        """
        # Ensure 2D
        image = np.squeeze(image).astype(np.float32)
        H, W = image.shape
        
        # Compute normalization parameters
        self.model.mean = float(np.mean(image))
        self.model.std = float(np.std(image))
        
        # Normalize image
        image_norm = (image - self.model.mean) / (self.model.std + 1e-8)
        
        # Create static validation mask (ONCE, before training)
        self.val_mask = self._create_static_val_mask((H, W))
        val_mask_tensor = torch.from_numpy(self.val_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Prepare target tensor (original normalized image)
        target_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Pad image to be divisible by 2^depth
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_epoch = 0
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        print(f"\n{'='*60}")
        print(f"Starting N2V Training")
        print(f"{'='*60}")
        print(f"Image size: {H}x{W}")
        print(f"Max epochs: {max_epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # ========== TRAINING STEP ==========
            self.model.train()
            
            # Create NEW random training mask for this epoch
            train_mask = self._create_training_mask((H, W))
            
            # Ensure training mask doesn't overlap with validation mask
            train_mask = train_mask * (1 - self.val_mask)
            
            # Apply mask to image (replace masked pixels with zero)
            masked_image = self._apply_mask_to_image(image_norm, train_mask)
            
            # Pad for U-Net
            if pad_h > 0 or pad_w > 0:
                masked_image_padded = np.pad(masked_image, ((0, pad_h), (0, pad_w)), mode='reflect')
                target_padded = np.pad(image_norm, ((0, pad_h), (0, pad_w)), mode='reflect')
                train_mask_padded = np.pad(train_mask, ((0, pad_h), (0, pad_w)), mode='constant')
            else:
                masked_image_padded = masked_image
                target_padded = image_norm
                train_mask_padded = train_mask
            
            # To tensors
            input_tensor = torch.from_numpy(masked_image_padded).unsqueeze(0).unsqueeze(0).to(self.device)
            target_padded_tensor = torch.from_numpy(target_padded).unsqueeze(0).unsqueeze(0).to(self.device)
            train_mask_tensor = torch.from_numpy(train_mask_padded).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            
            # Crop output if padded
            if pad_h > 0 or pad_w > 0:
                output_cropped = output[:, :, :H, :W]
                train_mask_cropped = train_mask_tensor[:, :, :H, :W]
            else:
                output_cropped = output
                train_mask_cropped = train_mask_tensor
            
            # Compute training loss (only on training mask pixels)
            train_loss = self._compute_loss(output_cropped, target_tensor, train_mask_cropped)
            
            # Backward pass
            train_loss.backward()
            self.optimizer.step()
            
            self.train_history.append(train_loss.item())
            
            # ========== VALIDATION STEP ==========
            self.model.eval()
            with torch.no_grad():
                # Use unmasked image for validation prediction (model sees all pixels)
                if pad_h > 0 or pad_w > 0:
                    input_full_padded = np.pad(image_norm, ((0, pad_h), (0, pad_w)), mode='reflect')
                else:
                    input_full_padded = image_norm
                
                input_full_tensor = torch.from_numpy(input_full_padded).unsqueeze(0).unsqueeze(0).to(self.device)
                val_output = self.model(input_full_tensor)
                
                # Crop
                if pad_h > 0 or pad_w > 0:
                    val_output_cropped = val_output[:, :, :H, :W]
                else:
                    val_output_cropped = val_output
                
                # Compute validation loss (only on STATIC validation mask pixels)
                val_loss = self._compute_loss(val_output_cropped, target_tensor, val_mask_tensor)
            
            self.val_history.append(val_loss.item())
            
            # ========== EARLY STOPPING CHECK ==========
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save best model
                torch.save(self.model, os.path.join(save_dir, f'best_{model_name}.pth'))
                
                if verbose:
                    print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                          f"Train Loss: {train_loss.item():.6f} | "
                          f"Val Loss: {val_loss.item():.6f} | "
                          f"✓ New best!")
            else:
                epochs_without_improvement += 1
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                          f"Train Loss: {train_loss.item():.6f} | "
                          f"Val Loss: {val_loss.item():.6f} | "
                          f"No improvement for {epochs_without_improvement} epochs")
            
            # Check early stopping
            if epochs_without_improvement >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}!")
                print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                break
        
        # Save final model
        torch.save(self.model, os.path.join(save_dir, f'last_{model_name}.pth'))
        
        # Save training history
        np.save(os.path.join(save_dir, f'history_{model_name}.npy'), 
                np.array([self.train_history, self.val_history]))
        
        total_time = (time.time() - start_time) / 60
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Total epochs: {len(self.train_history)}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Total time: {total_time:.2f} minutes")
        print(f"Models saved to: {save_dir}")
        print(f"{'='*60}")
        
        return self.train_history, self.val_history
    
    def predict(self, image):
        """
        Denoise an image using the trained model
        
        Parameters
        ----------
        image : numpy.ndarray
            Noisy image to denoise
        
        Returns
        -------
        numpy.ndarray
            Denoised image
        """
        self.model.eval()
        
        # Ensure 2D
        image = np.squeeze(image).astype(np.float32)
        H, W = image.shape
        
        # Normalize
        image_norm = (image - self.model.mean) / (self.model.std + 1e-8)
        
        # Pad
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            image_padded = np.pad(image_norm, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            image_padded = image_norm
        
        # To tensor
        input_tensor = torch.from_numpy(image_padded).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Crop and denormalize
        output_np = output.cpu().numpy()[0, 0]
        if pad_h > 0 or pad_w > 0:
            output_np = output_np[:H, :W]
        
        denoised = output_np * self.model.std + self.model.mean
        
        return denoised


def load_model(path, device):
    """
    Load a saved N2V model
    
    Parameters
    ----------
    path : str
        Path to saved .pth file
    device : torch.device
        Device to load model to
    
    Returns
    -------
    N2VUNet
        Loaded model
    """
    model = torch.load(path, map_location=device)
    model.to(device)
    model.eval()
    return model
