#!/usr/bin/env python3
"""
PPN2V Google Colab Runner
=========================
Complete pipeline for running DATASET_01 denoising on Google Colab with GPU.

This script:
1. Mounts Google Drive for data storage
2. Clones your GitHub repository
3. Installs all dependencies
4. Runs the complete PN2V pipeline (notebooks 0-3)
5. Saves all results to Google Drive
6. Optionally pushes changes back to GitHub

Usage in Colab:
    !git clone https://github.com/ZurvanAkarna/PPN2V.git /content/PPN2V
    %run /content/PPN2V/ppn2v_colab_runner.py

Author: ZurvanAkarna
"""

import os
import sys
import subprocess

# =============================================================================
# SECTION 1: GOOGLE DRIVE SETUP
# =============================================================================
print("=" * 70)
print("STEP 1: MOUNTING GOOGLE DRIVE")
print("=" * 70)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✅ Google Drive mounted successfully!")
except ImportError:
    IN_COLAB = False
    print("⚠️  Not running in Colab - skipping Drive mount")

# Define paths
DRIVE_ROOT = "/content/drive/MyDrive"
DATA_DIR = f"{DRIVE_ROOT}/ppn2v_data/DATASET_01"
MODELS_DIR = f"{DRIVE_ROOT}/ppn2v_models/DATASET_01"
RUNS_DIR = f"{DRIVE_ROOT}/ppn2v_runs/DATASET_01"
REPO_DIR = "/content/PPN2V"

# Create directories
for path in [DATA_DIR, MODELS_DIR, RUNS_DIR]:
    os.makedirs(path, exist_ok=True)
    print(f"📁 {path}")

# =============================================================================
# SECTION 2: CLONE/UPDATE REPOSITORY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: SETTING UP REPOSITORY")
print("=" * 70)

GITHUB_REPO = "https://github.com/ZurvanAkarna/PPN2V.git"

if os.path.exists(REPO_DIR):
    print(f"📂 Repository exists at {REPO_DIR}")
    os.chdir(REPO_DIR)
    subprocess.run(["git", "pull"], check=True)
    print("✅ Repository updated (git pull)")
else:
    subprocess.run(["git", "clone", GITHUB_REPO, REPO_DIR], check=True)
    print(f"✅ Repository cloned to {REPO_DIR}")

os.chdir(REPO_DIR)
print(f"📍 Working directory: {os.getcwd()}")

# =============================================================================
# SECTION 3: INSTALL DEPENDENCIES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: INSTALLING DEPENDENCIES")
print("=" * 70)

# Install the package in editable mode
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "tifffile", "scikit-image"], check=True)

# Add src to path
sys.path.insert(0, f"{REPO_DIR}/src")

# Test imports
import torch
import numpy as np
from tifffile import imread, imwrite
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Import PPN2V modules
from ppn2v.pn2v import gaussianMixtureNoiseModel, histNoiseModel, training, prediction, utils
from ppn2v.unet.model import UNet
print("✅ PPN2V modules imported successfully!")

# =============================================================================
# SECTION 4: CONFIGURATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: CONFIGURATION")
print("=" * 70)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# Dataset configuration
dataName = 'dataset01'
target_noise_level = 0.7

# Training configuration (adjust these for speed vs quality tradeoff)
CONFIG = {
    'n_gaussian': 3,          # GMM components
    'n_coeff': 2,             # Polynomial coefficients
    'n_samples': 800,         # Network output samples
    'depth': 3,               # U-Net depth
    'numOfEpochs': 100,       # Training epochs (reduce for testing)
    'stepsPerEpoch': 50,      # Steps per epoch
    'batchSize': 4,           # Batch size
    'learningRate': 1e-3,     # Learning rate
}

print(f"📊 Dataset: {dataName}")
print(f"📊 Noise level: σ={target_noise_level}")
print(f"📊 Training epochs: {CONFIG['numOfEpochs']}")

# Paths
WORK_DIR = f"{REPO_DIR}/examples/DATASET_01_code"
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f"📍 Working directory: {WORK_DIR}")

# =============================================================================
# SECTION 5: LOAD DATA FROM GOOGLE DRIVE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: LOADING DATA FROM GOOGLE DRIVE")
print("=" * 70)

# Check if data exists on Drive
clean_path = f"{DATA_DIR}/clean_image.tif"
jittered_path = f"{DATA_DIR}/jittered_image.tif"
noisy_path = f"{DATA_DIR}/Noisy images/noisy_image_jitter_skips_0__0_3_flags_0__0_4_Gaussian_{target_noise_level}.tif"

# Alternative: if Noisy images folder doesn't exist, check for flat structure
if not os.path.exists(noisy_path):
    noisy_path_alt = f"{DATA_DIR}/noisy_image_Gaussian_{target_noise_level}.tif"
    if os.path.exists(noisy_path_alt):
        noisy_path = noisy_path_alt

print(f"Looking for data in: {DATA_DIR}")
print(f"  Clean: {os.path.exists(clean_path)}")
print(f"  Jittered: {os.path.exists(jittered_path)}")
print(f"  Noisy: {os.path.exists(noisy_path)}")

if not os.path.exists(clean_path):
    print("\n❌ ERROR: Data not found on Google Drive!")
    print("\nPlease upload your data to Google Drive:")
    print(f"  📁 {DATA_DIR}/")
    print(f"     ├── clean_image.tif")
    print(f"     ├── jittered_image.tif")
    print(f"     └── Noisy images/")
    print(f"         └── noisy_image_...Gaussian_{target_noise_level}.tif")
    raise FileNotFoundError("Data files not found on Google Drive")

# Load images
clean_image = imread(clean_path).astype(np.float32)
jittered_image = imread(jittered_path).astype(np.float32)
noisy_image = imread(noisy_path).astype(np.float32)

print(f"\n✅ Loaded images:")
print(f"   Clean:    {clean_image.shape}, range [{clean_image.min():.2f}, {clean_image.max():.2f}]")
print(f"   Jittered: {jittered_image.shape}, range [{jittered_image.min():.2f}, {jittered_image.max():.2f}]")
print(f"   Noisy:    {noisy_image.shape}, range [{noisy_image.min():.2f}, {noisy_image.max():.2f}]")

# Prepare data for pipeline
noisy_stack = noisy_image[np.newaxis, ...] if len(noisy_image.shape) == 2 else noisy_image

# Save to working directory
imwrite(f'{dataName}_clean.tif', clean_image)
imwrite(f'{dataName}_signal.tif', jittered_image)
imwrite(f'{dataName}_noisy.tif', noisy_stack)
print(f"\n✅ Data prepared in working directory")

# =============================================================================
# SECTION 6: CREATE NOISE MODELS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: CREATING NOISE MODELS")
print("=" * 70)

# Prepare signal/observation for calibration
signal = jittered_image
observation = noisy_image if len(noisy_image.shape) == 2 else noisy_image[0]

signal_for_hist = signal[np.newaxis, ...]
obs_for_hist = observation[np.newaxis, ...]

# Determine intensity range
all_values = np.concatenate([signal.flatten(), observation.flatten()])
minVal = np.percentile(all_values, 0.5)
maxVal = np.percentile(all_values, 99.5)
bins = 256

print(f"📊 Intensity range: [{minVal:.2f}, {maxVal:.2f}]")

# --- Create Histogram Noise Model ---
print("\n📈 Creating Histogram Noise Model...")
nameHistNoiseModel = f'HistNoiseModel_{dataName}_calibration'
histogram = histNoiseModel.createHistogram(bins, minVal, maxVal, obs_for_hist, signal_for_hist)
np.save(nameHistNoiseModel + '.npy', histogram)
print(f"   ✅ Saved: {nameHistNoiseModel}.npy")

# --- Create GMM Noise Model ---
print("\n📈 Creating GMM Noise Model (this may take a few minutes)...")
min_signal = np.percentile(signal, 0.5)
max_signal = np.percentile(signal, 99.5)

nameGMMNoiseModel = f"GMMNoiseModel_{dataName}_{CONFIG['n_gaussian']}_{CONFIG['n_coeff']}_calibration"

gmmNoiseModel = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
    min_signal=min_signal,
    max_signal=max_signal,
    path='./',
    weight=None,
    n_gaussian=CONFIG['n_gaussian'],
    n_coeff=CONFIG['n_coeff'],
    device=device,
    min_sigma=50
)

gmmNoiseModel.train(
    signal_for_hist,
    obs_for_hist,
    batchSize=250000,
    n_epochs=2000,
    learning_rate=0.1,
    name=nameGMMNoiseModel,
    lowerClip=0.5,
    upperClip=99.5
)
print(f"   ✅ Saved: {nameGMMNoiseModel}.npz")

# =============================================================================
# SECTION 7: TRAIN PN2V NETWORK
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: TRAINING PN2V NETWORK")
print("=" * 70)

# Load data for training
data = imread(f'{dataName}_noisy.tif')
print(f"📊 Training data shape: {data.shape}")

# Select noise model (GMM recommended)
nameNoiseModel = nameGMMNoiseModel
print(f"📊 Using noise model: {nameNoiseModel}")

# Load noise model
params = np.load(nameNoiseModel + '.npz')
noiseModel = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(params=params, device=device)

# Create network
net = UNet(CONFIG['n_samples'], depth=CONFIG['depth'])
print(f"📊 Network: {sum(p.numel() for p in net.parameters()):,} parameters")

# Train
print(f"\n🚀 Starting training ({CONFIG['numOfEpochs']} epochs)...")
print(f"   Estimated time: ~{CONFIG['numOfEpochs'] * 0.3:.0f} minutes on GPU")

trainHist, valHist = training.trainNetwork(
    net=net,
    trainData=data.copy(),
    valData=data.copy(),
    postfix=nameNoiseModel,
    directory='./',
    noiseModel=noiseModel,
    device=device,
    numOfEpochs=CONFIG['numOfEpochs'],
    stepsPerEpoch=CONFIG['stepsPerEpoch'],
    virtualBatchSize=20,
    batchSize=CONFIG['batchSize'],
    learningRate=CONFIG['learningRate']
)

print(f"\n✅ Training complete!")
print(f"   Final train loss: {trainHist[-1]:.6f}")
print(f"   Final val loss: {valHist[-1]:.6f}")
print(f"   Best val loss: {min(valHist):.6f}")

# =============================================================================
# SECTION 8: PREDICTION & METRICS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: GENERATING PREDICTIONS")
print("=" * 70)

# Load trained network
net = torch.load(f'best_{nameNoiseModel}.net', weights_only=False)
print(f"✅ Loaded: best_{nameNoiseModel}.net")

# Load noise model
params = np.load(nameNoiseModel + '.npz')
noiseModel = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(params=params, device=device)

# Load images
noisy_for_pred = imread(f'{dataName}_noisy.tif')
noisy_for_pred = np.squeeze(noisy_for_pred)

# Run prediction
print("🔮 Running prediction...")
from ppn2v.pn2v.prediction import predict
means, mseEst = predict(noisy_for_pred, net, noiseModel, device, outScaling=10.0)

print(f"   Prior mean shape: {means.shape}")
print(f"   MMSE estimate shape: {mseEst.shape}")

# Compute uncertainty map
print("📊 Computing uncertainty map...")
net.eval()
img_normalized = (noisy_for_pred - net.mean) / net.std
h, w = img_normalized.shape
pad_h = (16 - h % 16) % 16
pad_w = (16 - w % 16) % 16
img_padded = np.pad(img_normalized, ((0, pad_h), (0, pad_w)), mode='reflect')

with torch.no_grad():
    img_tensor = torch.from_numpy(img_padded[np.newaxis, np.newaxis, ...].astype(np.float32)).to(device)
    output = net(img_tensor)
    samples = output.cpu().numpy()[0] * 10.0 * net.std + net.mean
    uncertainty_map = samples.std(axis=0)[:h, :w]

print(f"   Uncertainty range: [{uncertainty_map.min():.4f}, {uncertainty_map.max():.4f}]")

# Calculate metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def normalize_01(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-10)

clean_norm = normalize_01(clean_image)
noisy_norm = normalize_01(noisy_for_pred)
mmse_norm = normalize_01(np.squeeze(mseEst))

psnr_noisy = psnr(clean_norm, noisy_norm, data_range=1.0)
ssim_noisy = ssim(clean_norm, noisy_norm, data_range=1.0)
psnr_mmse = psnr(clean_norm, mmse_norm, data_range=1.0)
ssim_mmse = ssim(clean_norm, mmse_norm, data_range=1.0)

print("\n" + "=" * 60)
print("QUALITY METRICS")
print("=" * 60)
print(f"{'Method':<25} {'PSNR (dB)':<12} {'SSIM':<10}")
print("-" * 60)
print(f"{'Noisy (baseline)':<25} {psnr_noisy:<12.2f} {ssim_noisy:<10.4f}")
print(f"{'PN2V (MMSE)':<25} {psnr_mmse:<12.2f} {ssim_mmse:<10.4f}")
print("-" * 60)
print(f"{'Supervisor Benchmark':<25} {'28.48':<12} {'0.73':<10}")
print("=" * 60)
print(f"\n📈 PSNR improvement: +{psnr_mmse - psnr_noisy:.2f} dB")

# =============================================================================
# SECTION 9: SAVE RESULTS TO GOOGLE DRIVE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: SAVING RESULTS TO GOOGLE DRIVE")
print("=" * 70)

import shutil

# Files to save
files_to_save = [
    f'{dataName}_denoised_mmse.tif',
    f'{dataName}_denoised_prior_mean.tif',
    f'{dataName}_uncertainty_map.tif',
    f'{nameNoiseModel}.npz',
    f'{nameHistNoiseModel}.npy',
    f'best_{nameNoiseModel}.net',
    f'last_{nameNoiseModel}.net',
    f'history{nameNoiseModel}.npy',
]

# Save denoised images
imwrite(f'{dataName}_denoised_mmse.tif', np.squeeze(mseEst).astype(np.float32))
imwrite(f'{dataName}_denoised_prior_mean.tif', np.squeeze(means).astype(np.float32))
imwrite(f'{dataName}_uncertainty_map.tif', uncertainty_map.astype(np.float32))

# Copy to Drive
for f in files_to_save:
    if os.path.exists(f):
        shutil.copy(f, MODELS_DIR)
        print(f"   ✅ {f} → Drive")

# Save metrics report
metrics_report = f"""
PPN2V Denoising Results - DATASET_01
=====================================
Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Noise Level: σ={target_noise_level}

QUALITY METRICS
---------------
Method                    PSNR (dB)     SSIM
--------------------------------------------------
Noisy (baseline)          {psnr_noisy:.2f}         {ssim_noisy:.4f}
PN2V (MMSE)              {psnr_mmse:.2f}         {ssim_mmse:.4f}
--------------------------------------------------
Supervisor Benchmark      28.48         0.73

PSNR improvement: +{psnr_mmse - psnr_noisy:.2f} dB

CONFIGURATION
-------------
Epochs: {CONFIG['numOfEpochs']}
Network samples: {CONFIG['n_samples']}
GMM components: {CONFIG['n_gaussian']}

FILES SAVED
-----------
{chr(10).join(['- ' + f for f in files_to_save if os.path.exists(f)])}
"""

with open(f'{MODELS_DIR}/RESULTS_SUMMARY.txt', 'w') as f:
    f.write(metrics_report)
print(f"   ✅ RESULTS_SUMMARY.txt → Drive")

# =============================================================================
# SECTION 10: VISUALIZE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: GENERATING VISUALIZATIONS")
print("=" * 70)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Images
axes[0, 0].imshow(clean_image, cmap='gray')
axes[0, 0].set_title('Ground Truth', fontsize=14)
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_for_pred, cmap='gray')
axes[0, 1].set_title(f'Noisy (PSNR: {psnr_noisy:.2f} dB)', fontsize=14)
axes[0, 1].axis('off')

axes[0, 2].imshow(np.squeeze(mseEst), cmap='gray')
axes[0, 2].set_title(f'PN2V Denoised (PSNR: {psnr_mmse:.2f} dB)', fontsize=14)
axes[0, 2].axis('off')

# Row 2: Residuals and Uncertainty
residual_noisy = noisy_for_pred - clean_image
residual_denoised = np.squeeze(mseEst) - clean_image
vmax = np.percentile(np.abs(residual_noisy), 99)

axes[1, 0].imshow(residual_noisy, cmap='RdBu', vmin=-vmax, vmax=vmax)
axes[1, 0].set_title('Residual (Noisy - Clean)', fontsize=14)
axes[1, 0].axis('off')

axes[1, 1].imshow(residual_denoised, cmap='RdBu', vmin=-vmax, vmax=vmax)
axes[1, 1].set_title('Residual (Denoised - Clean)', fontsize=14)
axes[1, 1].axis('off')

im = axes[1, 2].imshow(uncertainty_map, cmap='hot')
axes[1, 2].set_title('UNCERTAINTY MAP', fontsize=14, fontweight='bold')
axes[1, 2].axis('off')
plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

plt.tight_layout()
plt.savefig(f'{MODELS_DIR}/final_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ final_results.png → Drive")

# Training history
plt.figure(figsize=(10, 6))
plt.plot(trainHist, label='Training Loss', alpha=0.7)
plt.plot(valHist, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{MODELS_DIR}/training_history.png', dpi=150)
plt.show()
print(f"   ✅ training_history.png → Drive")

# =============================================================================
# DONE!
# =============================================================================
print("\n" + "=" * 70)
print("🎉 PIPELINE COMPLETE!")
print("=" * 70)
print(f"\n📁 Results saved to: {MODELS_DIR}")
print("\nFiles on Google Drive:")
for f in os.listdir(MODELS_DIR):
    print(f"   📄 {f}")

print("\n" + "-" * 70)
print("NEXT STEPS:")
print("-" * 70)
print("1. Check results in Google Drive: MyDrive/ppn2v_models/DATASET_01/")
print("2. Download denoised images: dataset01_denoised_mmse.tif")
print("3. View uncertainty map: dataset01_uncertainty_map.tif")
print("4. See metrics: RESULTS_SUMMARY.txt")
