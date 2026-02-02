# 🚀 Complete Guide: Running PPN2V on Google Colab

This comprehensive guide teaches you how to run your DATASET_01 denoising pipeline on Google Colab, save results to Google Drive, and sync with GitHub.

---

## 📋 Table of Contents

1. [Prerequisites Setup](#-prerequisites-setup)
2. [Understanding the Workflow](#-understanding-the-workflow)
3. [Quick Start (3 Steps)](#-quick-start-3-steps)
4. [Step-by-Step Tutorial](#-step-by-step-tutorial)
5. [Git Push/Pull from Colab](#-git-pushpull-from-colab)
6. [Running Individual Notebooks](#-running-individual-notebooks)
7. [Customization Options](#️-customization-options)
8. [Troubleshooting](#-troubleshooting)

---

## 🔧 Prerequisites Setup

### 1. Google Drive Structure

Create this folder structure in your Google Drive:

```
My Drive/
└── ppn2v_data/
    └── DATASET_01/
        ├── clean_image.tif          ← Ground truth (for metrics)
        ├── jittered_image.tif       ← Signal for calibration
        └── Noisy images/
            └── noisy_image_jitter_skips_0__0_3_flags_0__0_4_Gaussian_0.7.tif
```

### 2. GitHub Repository

Your repo should be at: `https://github.com/ZurvanAkarna/PPN2V`

### 3. Colab Requirements

- Google account
- GPU runtime enabled (free tier works!)

---

## 🔄 Understanding the Workflow

```
     VS CODE                    GITHUB                      COLAB
    ┌───────┐                 ┌────────┐                  ┌────────┐
    │ Edit  │ ──git push──►  │  Repo  │  ◄──git clone──  │  Run   │
    │ Code  │                 │ Remote │                  │ Code   │
    │       │ ◄──git pull──  │        │  ──git push──►   │  GPU   │
    └───────┘                 └────────┘                  └────────┘
                                                               │
                                                               ▼
                                                        ┌────────────┐
                                                        │   GOOGLE   │
                                                        │   DRIVE    │
                                                        │ Input Data │
                                                        │  + Results │
                                                        └────────────┘
```

### Key Terms

| Term | Meaning |
|------|---------|
| **git clone** | Download repo from GitHub to Colab |
| **git pull** | Update local repo with latest changes from GitHub |
| **git push** | Upload local changes to GitHub |
| **Drive mount** | Connect Google Drive to Colab filesystem |
| **%run** | Colab magic command to execute a Python script |

---

## ⚡ Quick Start (3 Steps)

### Step 1: Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com/) → New Notebook

### Step 2: Enable GPU
`Runtime` → `Change runtime type` → `GPU` → `Save`

### Step 3: Run This Cell
```python
# Clone and run the complete pipeline
!git clone https://github.com/ZurvanAkarna/PPN2V.git /content/PPN2V
%run /content/PPN2V/ppn2v_colab_runner.py
```

**That's it!** The script automatically:
1. ✅ Mounts your Google Drive
2. ✅ Installs all dependencies
3. ✅ Loads your images from Drive
4. ✅ Creates noise models (GMM + Histogram)
5. ✅ Trains PN2V network (~15-30 minutes on GPU)
6. ✅ Generates denoised images + uncertainty maps
7. ✅ Calculates PSNR/SSIM metrics
8. ✅ Saves all results back to your Drive

---

## 📚 Step-by-Step Tutorial

### Part A: Setting Up Colab

#### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Verify your data exists
!ls "/content/drive/MyDrive/ppn2v_data/DATASET_01/"
```

You'll see a popup asking for permission - click "Connect to Google Drive".

#### Cell 2: Clone Your Repository
```python
# Remove old clone if exists
!rm -rf /content/PPN2V

# Clone fresh from GitHub
!git clone https://github.com/ZurvanAkarna/PPN2V.git /content/PPN2V

# Verify
!ls /content/PPN2V
```

#### Cell 3: Install Dependencies
```python
# Change to repo directory
%cd /content/PPN2V

# Install the package
!pip install -e .
!pip install tifffile scikit-image

# Test import
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Part B: Understanding Environments

**Important**: Colab doesn't use `conda activate`. Instead:

| Local (VS Code) | Colab Equivalent |
|-----------------|------------------|
| `conda activate ppn2v` | Already active (Colab's Python) |
| `pip install package` | `!pip install package` |
| `python script.py` | `!python script.py` or `%run script.py` |
| `cd folder` | `%cd folder` |

Colab comes with PyTorch pre-installed! Just add missing packages with `!pip install`.

### Part C: Run the Pipeline

#### Option 1: Automated Runner (Recommended)
```python
%run /content/PPN2V/ppn2v_colab_runner.py
```

#### Option 2: Run Notebooks Step by Step
```python
# Set up paths
import sys
sys.path.insert(0, '/content/PPN2V/src')
%cd /content/PPN2V/examples/DATASET_01_code
```
Then copy cells from each notebook in order: 0 → 1 → 2 → 3

---

## 🔄 Git Push/Pull from Colab

### Pulling Latest Changes (GitHub → Colab)

```python
%cd /content/PPN2V

# Pull latest changes
!git pull origin main

# If you get errors about local changes:
!git stash          # Save local changes temporarily
!git pull origin main
!git stash pop      # Restore local changes
```

### Pushing Changes to GitHub (Colab → GitHub)

#### Step 1: Create GitHub Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. **Name**: "Colab Access"
4. **Expiration**: 30 days (or longer)
5. **Scopes**: Check ✅ `repo` (all sub-items)
6. Click "Generate token"
7. **COPY THE TOKEN NOW** (you won't see it again!)

#### Step 2: Configure Git in Colab
```python
%cd /content/PPN2V

# Configure your identity
!git config user.email "your-email@example.com"
!git config user.name "ZurvanAkarna"

# Set remote with your token (replace YOUR_TOKEN_HERE)
!git remote set-url origin https://YOUR_TOKEN_HERE@github.com/ZurvanAkarna/PPN2V.git
```

#### Step 3: Push Your Changes
```python
# See what changed
!git status

# Add all changes
!git add -A

# Commit with a message
!git commit -m "Update from Colab: training results"

# Push to GitHub
!git push origin main
```

### Secure Token Handling

**Never share your token!** For safer handling:
```python
from getpass import getpass
token = getpass("Enter your GitHub token: ")
!git remote set-url origin https://{token}@github.com/ZurvanAkarna/PPN2V.git
```

---

## 📓 Running Individual Notebooks

### Method 1: Copy-Paste (Most Reliable)

1. Open your notebook in VS Code or on GitHub
2. Copy the code cells one by one
3. Paste into Colab cells
4. Run each cell in order

### Method 2: Using papermill

```python
!pip install papermill

import papermill as pm

# Run notebook 0
pm.execute_notebook(
    '/content/PPN2V/examples/DATASET_01_code/0_PrepareData.ipynb',
    '/content/outputs/0_PrepareData_output.ipynb'
)
```

### Method 3: Convert to Python Script

```python
# Convert notebook to Python script
!jupyter nbconvert --to script "/content/PPN2V/examples/DATASET_01_code/0_PrepareData.ipynb"

# Run the script
%run "/content/PPN2V/examples/DATASET_01_code/0_PrepareData.py"
```

---

## ⚙️ Customization Options

### Adjust Training Time

In `ppn2v_colab_runner.py`, modify the CONFIG:

```python
CONFIG = {
    'numOfEpochs': 50,    # Reduce for faster testing
    'stepsPerEpoch': 25,  # Reduce for faster testing
    ...
}
```

| Epochs | Estimated Time | Quality |
|--------|----------------|---------|
| 20 | ~5 min | Testing only |
| 50 | ~15 min | Quick results |
| 100 | ~30 min | Good quality |
| 200 | ~60 min | Best quality |

### Change Noise Level

```python
target_noise_level = 0.5  # Change from 0.7
```

### Use Histogram Instead of GMM

```python
nameNoiseModel = nameHistNoiseModel  # Use histogram
```

---

## 🔧 Troubleshooting

### "Data not found" Error

```python
# Check exact path
!ls -la "/content/drive/MyDrive/ppn2v_data/DATASET_01/"
```
**Solution**: Folder names are case-sensitive!

### "CUDA out of memory"

```python
CONFIG['batchSize'] = 1       # Reduce from 4
CONFIG['n_samples'] = 400     # Reduce from 800
```

### "Module not found"

```python
%cd /content/PPN2V
!pip install -e .
import sys
sys.path.insert(0, '/content/PPN2V/src')
```

### No GPU Available

**Check GPU status:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```
If False: Runtime → Change runtime type → GPU

### Git Push Rejected

```python
# If you made changes locally that conflict:
!git fetch origin
!git reset --hard origin/main  # WARNING: Discards local changes

# Or force push (careful!):
!git push -f origin main
```

---

## � Expected Output

After running, find results in `MyDrive/ppn2v_models/DATASET_01/`:

| File | Description |
|------|-------------|
| `dataset01_denoised_mmse.tif` | **Main result** - best quality denoised |
| `dataset01_denoised_prior_mean.tif` | N2V-like denoised |
| `dataset01_uncertainty_map.tif` | **Uncertainty map** (key advantage!) |
| `GMMNoiseModel_*.npz` | Trained GMM noise model |
| `best_*.net` | Best trained network |
| `final_results.png` | Visual comparison |
| `RESULTS_SUMMARY.txt` | Metrics report |

---

## 🎓 Complete Example Session

```python
# ============ CELL 1: Setup ============
from google.colab import drive
drive.mount('/content/drive')

# ============ CELL 2: Clone Repo ============
!rm -rf /content/PPN2V
!git clone https://github.com/ZurvanAkarna/PPN2V.git /content/PPN2V
%cd /content/PPN2V

# ============ CELL 3: Install ============
!pip install -e . -q
!pip install tifffile scikit-image -q

# ============ CELL 4: Run Pipeline ============
%run ppn2v_colab_runner.py

# ============ CELL 5: (Optional) Push Results ============
!git config user.email "your-email@example.com"
!git config user.name "ZurvanAkarna"
!git add -A
!git commit -m "Results from Colab run"
# Uncomment after setting up token:
# !git remote set-url origin https://YOUR_TOKEN@github.com/ZurvanAkarna/PPN2V.git
# !git push origin main
```

---

## 📞 Quick Reference Commands

| Action | Command |
|--------|---------|
| Mount Drive | `drive.mount('/content/drive')` |
| Clone repo | `!git clone URL /content/PPN2V` |
| Update repo | `!git pull origin main` |
| Install package | `!pip install -e .` |
| Run script | `%run script.py` |
| Change dir | `%cd /path` |
| List files | `!ls -la` |
| Check GPU | `torch.cuda.is_available()` |
| Add changes | `!git add -A` |
| Commit | `!git commit -m "message"` |
| Push | `!git push origin main` |

---

## �💡 Tips for Best Results

1. **Use GPU Runtime** - Essential for reasonable training time
2. **Verify Data Quality** - Clean image should be noise-free, noisy should have consistent noise
3. **Monitor Training** - Watch the loss curves decrease steadily
4. **Compare Visualizations** - The `final_results_dataset01.png` shows where improvement happened
5. **Check Error Maps** - Green areas in improvement map = successful denoising

## 📞 Need Help?

If you encounter issues:
1. Check the console output for specific error messages
2. Verify all file paths in Google Drive
3. Ensure GPU is enabled in Colab
4. Check that images load correctly (run the data preparation section separately)

## 🎯 Next Steps

After successful denoising:
1. Download `dataset01_denoised_mmse.tif` from Google Drive
2. Compare metrics in `RESULTS_SUMMARY.txt`
3. Analyze the visualization in `final_results_dataset01.png`
4. If metrics are good, use the denoised image for your analysis!
5. If not satisfied, try increasing training epochs or adjusting the noise model

## 🎯 Summary

The workflow in 5 points:

1. **Data** → Upload to Google Drive (`ppn2v_data/DATASET_01/`)
2. **Code** → Clone from GitHub (`git clone`)
3. **Run** → Execute on Colab with GPU
4. **Results** → Auto-saved to Drive (`ppn2v_models/DATASET_01/`)
5. **Sync** → Push changes back to GitHub (`git push`)

---

**Estimated total time:** 20-40 minutes (including setup and training)

**GPU recommended:** Yes (10-20x faster than CPU)

**Internet required:** Only for initial setup and cloning repo

---

Happy denoising! 🔬✨
