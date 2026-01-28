# PPN2V (Parametric Probabilistic Noise2Void) Codebase Guide

## Project Overview
This is a PyTorch implementation of **Fully Unsupervised Probabilistic Noise2Void** for denoising microscopy images. The codebase extends Probabilistic Noise2Void (PN2V) with two key contributions:
1. **PN2V GMM**: Gaussian Mixture Model parameterization of noise models (vs. histogram-based)
2. **Bootstrap Mode**: Fully unsupervised denoising without calibration data

## Architecture & Core Concepts

### Three-Component Pipeline
The denoising workflow consists of three distinct stages executed via Jupyter notebooks:

1. **Noise Model Creation** (`1a_` or `1b_` notebooks)
   - **Calibration Mode** (`1a_CreateNoiseModel_Calibration.ipynb`): Learns p(x_i|s_i) from static noisy calibration images
   - **Bootstrap Mode** (`1b_CreateNoiseModel_Bootstrap.ipynb`): Learns noise model from N2V predictions on the target image itself
   - Outputs: `.npy` (histogram) or `.npz` (GMM parameters)

2. **PN2V Training** (`2_ProbabilisticNoise2VoidTraining.ipynb`)
   - Trains U-Net with N2V-style pixel masking + noise model likelihood
   - Loss function (Eq. 7): `-log(mean(likelihood(observations, samples)))` averaged over masked pixels
   - Outputs: `best_*.net` and `last_*.net` model checkpoints

3. **Prediction** (`3_ProbabilisticNoise2VoidPrediction.ipynb`)
   - Generates two outputs: prior mean (N2V-like) and MMSE estimate (using noise model)
   - MMSE is the weighted sum of samples using likelihood as weights

### Key Components

#### Noise Models (`src/ppn2v/pn2v/`)
- **`gaussianMixtureNoiseModel.py`**: Parametric GMM noise model with polynomial regressors mapping signal intensity to Gaussian parameters (mean, sigma, weight)
  - Initialize with `min_signal`, `max_signal`, `n_gaussian` (typically 3), `n_coeff` (2 for linear)
  - Training: `model.train(signal, observation, batchSize=250000, n_epochs=2000)`
  - Key parameter: `min_sigma=50` prevents degenerate solutions
  
- **`histNoiseModel.py`**: 2D histogram representation of p(x|s)
  - Created via `createHistogram(bins, minVal, maxVal, observation, signal)`
  - Each row normalized to sum to 1

#### Network Architecture (`src/ppn2v/unet/model.py`)
- Standard U-Net with `depth=5`, `start_filts=64`, `merge_mode='add'`
- **Critical**: `num_classes` (output channels) = number of samples predicted (typically 1000-1600)
- Output scaling by factor of 10 (`outScaling=10.0`) speeds up training

#### Training System (`src/ppn2v/pn2v/training.py`)
- **N2V-style masking**: `getStratifiedCoords2D()` selects ~`numPix` pixels per patch; each masked pixel replaced with random neighbor from 5x5 window (excluding center)
- **Virtual batching**: Accumulates gradients over `virtualBatchSize=20` mini-batches before optimizer step
- **Normalization**: All data normalized using `(x - mean) / std` where mean/std computed from combined train+val set
- Loss computed only on masked pixels using `masks` tensor

#### Prediction (`src/ppn2v/pn2v/prediction.py`)
- `tiledPredict()`: Processes large images in overlapping tiles to manage GPU memory
  - `ps`: tile size (power of 2, typically 256-512)
  - `overlap`: pixels overlapping between tiles (typically 48)
- Returns: `means` (prior mean) and `mseEst` (MMSE prediction)

## Development Workflow

### Environment Setup
```bash
conda env create -f torch_ppn2v.yml
# OR manually:
conda create -n ppn2v python=3.9
conda activate ppn2v
conda install pytorch torchvision pytorch-cuda=11.8 'numpy<1.24' scipy matplotlib tifffile jupyter -c pytorch -c nvidia
pip install git+https://github.com/juglab/PPN2V.git
```

**Critical**: `numpy<1.24.0` is required (compatibility constraint)

### Running Examples
Navigate to `examples/{Convallaria|MouseActin|MouseSkullNuclei}/` and choose mode:

**Calibration Mode** (PN2V GMM / PN2V):
```
PN2V/1a_CreateNoiseModel_Calibration.ipynb → 2_ProbabilisticNoise2VoidTraining.ipynb → 3_ProbabilisticNoise2VoidPrediction.ipynb
```

**Bootstrap Mode** (Boot GMM / Boot Hist):
```
N2V/1_N2VTraining.ipynb → N2V/2_N2VPrediction.ipynb → PN2V/1b_CreateNoiseModel_Bootstrap.ipynb → PN2V/2_ProbabilisticNoise2VoidTraining.ipynb → 3_ProbabilisticNoise2VoidPrediction.ipynb
```

### Data Expectations
- **Input format**: TIFF stacks (3D numpy arrays via `tifffile.imread`)
- **Calibration data**: 100+ static noisy images of same sample
- **Training data**: Single noisy image(s) to denoise
- **Naming convention**: Use consistent `dataName` string across notebooks (e.g., `'convallaria'`)

## Project-Specific Conventions

### Noise Model Naming Pattern
```python
nameHistNoiseModel = 'HistNoiseModel_{dataName}_{calibration|bootstrap}'
nameGMMNoiseModel = 'GMMNoiseModel_{dataName}_{n_gaussian}_{n_coeff}_{calibration|bootstrap}'
```

### Hyperparameters by Dataset
From notebooks, typical configurations:
- **Convallaria**: GMM training with `batchSize=250000, n_epochs=2000, learning_rate=0.1`
- **Noise model ranges**: Set `minVal, maxVal` to cover pixel intensity range in images to denoise (not calibration data)
- **N2V/PN2V training**: `numOfEpochs=200, stepsPerEpoch=50, batchSize=4, patchSize=100`

### Training Output Files
- `best_{postfix}.net`: Model with lowest validation loss
- `last_{postfix}.net`: Final model checkpoint
- `history{postfix}.npy`: Training/validation loss history [epochs, trainLoss, valLoss]

### Data Normalization Pattern
```python
# ALL network inputs/outputs normalized using:
normalized = (data - net.mean) / net.std
# where net.mean, net.std computed from np.mean/std(concatenate(trainData, valData))
```

### Device Management
Always use `device = utils.getDevice()` which asserts CUDA availability (no CPU fallback)

## Common Patterns

### Loading Noise Models
```python
if 'HistNoiseModel' in name:
    histogram = np.load(path + name + '.npy')
    noiseModel = histNoiseModel.NoiseModel(histogram, device=device)
elif 'GMMNoiseModel' in name:
    params = np.load(path + name + '.npz')
    noiseModel = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(params=params, device=device)
```

### Network Creation
```python
net = UNet(n_channels=1600, n_depth=5, n_dim_start=64)  # 1600 samples typical
net = torch.load(path + 'best_' + postfix + '.net')  # Loading checkpoint
```

### Data Augmentation
All training uses random 90° rotations and horizontal/vertical flips when `augment=True` (default)

## Critical Implementation Details

1. **Masked pixel replacement**: In N2V masking, replacement value comes from random pixel in 5x5 neighborhood *excluding the center pixel* (see `randomCrop()` loop: `while a_==2 and b_==2`)

2. **Likelihood computation**: Both noise models implement `likelihood(observations, samples)` returning tensor of shape `[n_samples, batch, height, width]`

3. **Output scaling**: Network outputs multiplied by 10.0 during training (`samples = outputs * 10.0`) before denormalization - empirically speeds convergence

4. **Stratified sampling**: Training pixels selected via `getStratifiedCoords2D()` dividing image into grid boxes and sampling one pixel per box (not purely random)

5. **Supervised mode**: Training supports supervised=True with paired data in shape `[n_images, height, width, 2]` where `[:,:,:,0]` is noisy, `[:,:,:,1]` is clean

## Integration Points
- **External dependencies**: Minimal - PyTorch, numpy, scipy, matplotlib, tifffile
- **No external services**: All computation local (GPU required)
- **Data sources**: Zenodo dataset downloads in notebooks (URLs hardcoded)
