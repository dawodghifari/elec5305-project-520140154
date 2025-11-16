# Instrument Family Classification in Music Recordings Using Audio Signal Processing and Machine Learning

**ELEC5305 Final Project | University of Sydney | 2025**

[![Python](https://img.shields.io/badge/Python-3.10.19-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a+-orange.svg)](https://www.mathworks.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Detailed Workflow](#-detailed-workflow)
- [Expected Outputs](#-expected-outputs)
- [Results Summary](#-results-summary)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a complete audio classification pipeline to identify **5 instrument families** (keyboards, percussion, strings, voice, winds) from music recordings. The pipeline combines traditional signal processing with modern deep learning techniques.

### Key Features

- **Automated preprocessing** using MATLAB for audio standardization and segmentation
- **Leakage-safe data splitting** with grouped stratification to prevent data contamination
- **Comprehensive acoustic analysis** including spectral features, HPSS, and dimensionality reduction
- **Multiple baseline models**: MFCC+SVM and Mel-Spectrogram CNN
- **Transfer learning** using Google's YAMNet pre-trained audio model
- **Interactive visualizations** with Bokeh for confusion matrices and feature spaces

### Dataset

- **874 audio segments** (3 seconds each, 16 kHz mono)
- **5 instrument families** across 15+ instrument types
- **Train/Val/Test split**: 685 / 79 / 110 (78% / 9% / 13%)

---

## 📁 Project Structure

```
elec5305-project-520140154/
│
├── Raw Data/                          # Original unprocessed audio files
│   ├── keyboards/
│   ├── percussion/
│   ├── strings/
│   ├── voice/
│   └── winds/
│
├── Data/                              # Standardized audio (16kHz mono WAV)
│   ├── keyboards/
│   ├── percussion/
│   ├── strings/
│   ├── voice/
│   └── winds/
│
├── Segmented/                         # Final 3-second audio segments
│   ├── keyboards/
│   ├── percussion/
│   ├── strings/
│   ├── voice/
│   └── winds/
│
├── Manifests/                         # Metadata and splits
│   ├── manifest_master.csv           # Complete dataset manifest
│   ├── train.csv                     # Training split (685 samples)
│   ├── val.csv                       # Validation split (79 samples)
│   ├── test.csv                      # Test split (110 samples)
│
├── Scripts/                           # MATLAB preprocessing scripts
│   ├── file_rename_converts_00.mlx   # Audio standardization & labeling
│   └── data_segmentation_01.mlx      # Segmentation & quality filtering
│
├── Notebooks/                         # Python analysis notebooks
│   ├── 01_data_qc_and_splits.ipynb   # QC & train/val/test splits
│   ├── 02_acoustic_analysis.ipynb    # Feature extraction & visualization
│   ├── 03_baselines.ipynb            # MFCC+SVM & CNN baselines
│   └── 04_pipeline_Yamnet.ipynb      # YAMNet transfer learning
│
├── Results/                           # Model outputs
│   ├── embeddings_yamnet_train.npz   # YAMNet embeddings - train
│   ├── embeddings_yamnet_val.npz     # YAMNet embeddings - val
│   ├── embeddings_yamnet_test.npz    # YAMNet embeddings - test
│   ├── model_baseline_cnn.pt         # CNN model weights
│   └── model_yamnet_classifier.pt    # YAMNet classifier weights
│
│
├── ELEC5305_Final_Project_Report.pdf # LaTeX final report
├── ELEC5305_Final_Project_Video.mp4  # Project demonstration video
└── README.md                          # This file
```

---

## 💻 Requirements

### Hardware

- **RAM**: Minimum 8 GB (16 GB recommended)
- **Storage**: ~5 GB free space for datasets and models
- **GPU**: Optional but recommended for CNN/YAMNet training
  - CUDA-compatible GPU (NVIDIA) or Apple Silicon (MPS)

### Software

#### MATLAB (for preprocessing)
- **Version**: MATLAB R2023a or later
- **Required Toolboxes**:
  - Audio Toolbox
  - Signal Processing Toolbox
  - Statistics and Machine Learning Toolbox

#### Python (for analysis and modeling)
- **Version**: Python 3.10.19 (tested)
- **Package Manager**: conda or pip

---

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/elec5305-project-520140154.git
cd elec5305-project-520140154
```

### Step 2: Set Up Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda create -n audio_classification python=3.10.19

# Activate environment
conda activate audio_classification

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using venv

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install Required Packages

Create a `requirements.txt` file with the following dependencies:

```txt
# Core scientific computing
numpy==2.2.5
pandas==2.3.3
scipy==1.13.0

# Audio processing
soundfile==0.13.1
librosa==0.10.1

# Machine learning
scikit-learn==1.7.2
torch==2.0.1
torchaudio==2.0.2

# Deep learning frameworks
tensorflow==2.15.0
tensorflow-hub==0.15.0

# Visualization
bokeh==3.3.0
matplotlib==3.8.0
seaborn==0.13.0

# Utilities
tqdm==4.66.1
pathlib==1.0.1
```

Install all packages:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
# Run this in Python to verify all packages
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import sklearn
import torch
import tensorflow as tf
import tensorflow_hub as hub
import bokeh

print("✓ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {torch.cuda.is_available() or torch.backends.mps.is_available()}")
```

---

## 🚀 Quick Start

### Project Root Directory

**IMPORTANT**: All scripts and notebooks assume the project root is:
```
/Users/[your_username]/elec5305-project-520140154/
```

**Before running any code**, update the `PROJECT_ROOT` variable in each notebook:

```python
# Update this line in notebooks 01-04
PROJECT_ROOT = Path("/path/to/your/elec5305-project-520140154")
```

And in MATLAB scripts:
```matlab
% Update this line in .mlx files
projectRoot = '/path/to/your/elec5305-project-520140154';
```

### Complete Workflow (5 Steps)

```bash
# 1. Preprocess raw audio (MATLAB)
open Scripts/file_rename_converts_00.mlx    # Run all sections
open Scripts/data_segmentation_01.mlx        # Run all sections

# 2. Create train/val/test splits (Python)
jupyter notebook Notebooks/01_data_qc_and_splits.ipynb  # Kernel → Restart & Run All

# 3. Extract acoustic features (Python)
jupyter notebook Notebooks/02_acoustic_analysis.ipynb   # Kernel → Restart & Run All

# 4. Train baseline models (Python)
jupyter notebook Notebooks/03_baselines.ipynb           # Kernel → Restart & Run All

# 5. Train YAMNet classifier (Python)
jupyter notebook Notebooks/04_pipeline_Yamnet.ipynb     # Kernel → Restart & Run All
```

---

## 📝 Detailed Workflow

### Phase 1: Data Preprocessing (MATLAB)

#### Script 1: Audio Standardization
**File**: `Scripts/file_rename_converts_00.mlx`

**Purpose**: Standardize raw audio files to consistent format

**Steps**:
1. Open MATLAB and navigate to project directory
2. Open `file_rename_converts_00.mlx`
3. Update `projectRoot` variable to your local path
4. Run all sections in order

**What it does**:
- Reads all audio files from `Raw Data/`
- Converts to mono (single channel)
- Resamples to 16 kHz sample rate
- Normalizes amplitude to [-1, 1] range
- Infers family and instrument labels from filenames
- Saves standardized files to `Data/`

**Required MATLAB Toolboxes**:
- Audio Toolbox (for `audioread`, `audiowrite`)
- Signal Processing Toolbox (for `resample`)

**Expected Output**:
- `Data/` directory populated with standardized WAV files
- Console output showing processing progress

---

#### Script 2: Audio Segmentation
**File**: `Scripts/data_segmentation_01.mlx`

**Purpose**: Segment long audio files into 3-second clips with quality filtering

**Steps**:
1. Open `data_segmentation_01.mlx`
2. Update `projectRoot` variable
3. Run all sections in order

**What it does**:
- Reads standardized audio from `Data/`
- Segments each file into 3-second (48,000 sample) clips
- Applies RMS-based quality filtering (removes silent segments)
- Caps maximum segments per file (prevents class imbalance)
- Generates manifest files (CSV and MAT format)
- Saves segments to `Segmented/`

**Parameters** (adjustable in script):
```matlab
targetSampleRate = 16000;        % Hz
segmentDuration = 3;             % seconds
rmsThreshold = 0.01;             % minimum RMS to keep segment
maxSegmentsPerFile = 50;         % cap per source file
```

**Expected Output**:
- `Segmented/` directory with 3-second WAV files
- `Manifests/manifest_raw.csv` and `manifest_raw.mat`
- Console output with segment counts per family

---

### Phase 2: Data Quality Control & Splitting (Python)

#### Notebook 1: Data QC and Splits
**File**: `Notebooks/01_data_qc_and_splits.ipynb`

**Purpose**: Validate files, consolidate manifests, create leakage-safe splits

**Steps**:
1. Launch Jupyter: `jupyter notebook`
2. Open `01_data_qc_and_splits.ipynb`
3. Update `PROJECT_ROOT` in cell 1
4. Run: **Kernel → Restart & Run All**

**What it does**:
1. **File Validation**
   - Checks all files in manifest actually exist
   - Verifies audio can be loaded
   - Removes invalid entries

2. **Manifest Consolidation**
   - Merges MATLAB manifests with metadata
   - Adds source file tracking for grouped splitting
   - Creates `manifest_master.csv`

3. **Grouped Stratified Splitting**
   - Groups by source file (prevents data leakage)
   - Stratifies by family label (balanced classes)
   - Creates 78/9/13 train/val/test split
   - Saves `train.csv`, `val.csv`, `test.csv`

4. **Split Validation**
   - Verifies no source file appears in multiple splits
   - Checks class distributions
   - Displays summary statistics

**Key Configuration**:
```python
FAMILY_COLNAME = "family_label"
RANDOM_SEED = 42
TEST_SIZE = 0.13
VAL_SIZE = 0.10  # 10% of remaining after test split
```

**Expected Output**:
```
Dataset: 874 clips across 5 families
Train: 685 | Val: 79 | Test: 110
✓ No data leakage detected
✓ Splits saved to /Manifests/
```

**Output Files**:
- `Manifests/manifest_master.csv` (874 rows)
- `Manifests/train.csv` (685 rows)
- `Manifests/val.csv` (79 rows)
- `Manifests/test.csv` (110 rows)

---

### Phase 3: Acoustic Feature Extraction (Python)

#### Notebook 2: Acoustic Analysis
**File**: `Notebooks/02_acoustic_analysis.ipynb`

**Purpose**: Compute acoustic features and visualize family characteristics

**Steps**:
1. Open `02_acoustic_analysis.ipynb`
2. Update `PROJECT_ROOT` in cell 1
3. Run: **Kernel → Restart & Run All**

**What it does**:

1. **RMS and Dynamics** (Cell 3-4)
   - RMS level (dB)
   - Peak amplitude (dB)
   - Dynamic range (dB)
   - Boxplots by family

2. **Spectral Timbre Features** (Cell 6-7)
   - Spectral centroid (Hz) — brightness
   - Spectral bandwidth (Hz) — spread
   - Spectral rolloff (Hz) — energy distribution
   - Spectral flatness — tonality vs noise
   - Zero-crossing rate — transient content
   - Boxplots for all features

3. **Harmonic vs Percussive Separation** (Cell 9-10)
   - HPSS decomposition
   - Harmonic energy ratio
   - Percussive energy ratio
   - Family comparisons

4. **Mel-Spectrogram Examples** (Cell 12)
   - 2 examples per family
   - Grouped by family with clear titles
   - Different instruments within each family
   - Interactive Bokeh plots with color bars

5. **Feature Space Visualization** (Cell 14-15)
   - PCA (2D projection, 69.6% variance explained)
   - t-SNE (2D projection, perplexity=30)
   - Scatter plots with family colors

**Key Parameters**:
```python
SAMPLE_RATE = 16000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 2048
```

**Expected Output**:
- Interactive Bokeh visualizations in notebook
- `Manifests/acoustic_features_summary.csv` (874 × 8 features)
- Console summary statistics

---

### Phase 4: Baseline Models (Python)

#### Notebook 3: MFCC+SVM and CNN Baselines
**File**: `Notebooks/03_baselines.ipynb`

**Purpose**: Train and evaluate two baseline classification models

**Steps**:
1. Open `03_baselines.ipynb`
2. Update `PROJECT_ROOT` in cell 1
3. Run: **Kernel → Restart & Run All**
4. **Training time**: ~15-30 minutes (CNN training)

**What it does**:

**Baseline 1: MFCC + SVM** (Cells 3)
- Extracts 13 MFCCs + temporal statistics (mean, std)
- Feature dimension: 26
- StandardScaler normalization
- SVM with RBF kernel (C=10, gamma='scale')
- Training time: ~2 minutes

**Baseline 2: Mel-Spectrogram CNN** (Cells 4-9)
- Architecture:
  - Input: (1, 64, 94) mel-spectrogram
  - 3 Conv blocks (32→64→128 filters)
  - BatchNorm + ReLU + MaxPool + Dropout
  - AvgPool2d for MPS compatibility
  - 2 FC layers (1536→256→5)
  - Parameters: 487,877
- Training: 25 epochs, AdamW optimizer
- Learning rate scheduling (ReduceLROnPlateau)
- Training time: ~20 minutes (CPU/MPS)

**Device Selection**:
```python
DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"
```

**Expected Output**:
- SVM test accuracy and F1 score
- CNN training curves (loss, accuracy, F1)
- Confusion matrices (Bokeh interactive)
- `Results/model_baseline_cnn.pt` (~2 MB)

**Console Output Example**:
```
Baseline 1: MFCC + SVM
--------------------------------------------------
Test Accuracy: 0.6545
Test Macro F1: 0.6234

Baseline 2: Mel-Spectrogram + CNN
--------------------------------------------------
Epoch 25/25 | Loss: 0.4231 | Train: 85.4% | Val: 72.1% | F1: 0.7089 ✓
Test Accuracy: 0.7091
Test Macro F1: 0.6987
```

---

### Phase 5: YAMNet Transfer Learning (Python)

#### Notebook 4: YAMNet Embedding Pipeline
**File**: `Notebooks/04_pipeline_Yamnet.ipynb`

**Purpose**: Extract YAMNet embeddings and train final classifier

**Steps**:
1. Open `04_pipeline_Yamnet.ipynb`
2. Update `PROJECT_ROOT` in cell 1
3. Run: **Kernel → Restart & Run All**
4. **First run**: ~30-45 minutes (embedding extraction)
5. **Subsequent runs**: ~5 minutes (embeddings cached)

**What it does**:

**Step 1: Embedding Extraction** (Cell 4)
- Loads pre-trained YAMNet from TensorFlow Hub
- Processes each 3-second audio clip
- Extracts 1024-dimensional embeddings (averaged over frames)
- Caches to NPZ files for reuse
- First run: Downloads YAMNet (~14 MB)

**Step 2: Classifier Training** (Cells 5-7)
- Architecture:
  - Input: 1024-dim YAMNet embeddings
  - FC1: 1024→512 (ReLU, Dropout 0.5)
  - FC2: 512→256 (ReLU, Dropout 0.5)
  - FC3: 256→5 (output logits)
  - Parameters: 131,845
- Training: 25 epochs, AdamW optimizer (lr=1e-4)
- Training time: ~3 minutes

**Step 3: Evaluation** (Cell 8)
- Test set predictions
- Confusion matrix (Bokeh)
- Classification report

**Key Configuration**:
```python
SAMPLE_RATE = 16000
DURATION_SECONDS = 3
BATCH_SIZE = 4          # Small batch for embedding extraction
EPOCHS = 25
```

**Expected Output**:
- `Results/embeddings_yamnet_train.npz` (~3 MB, 685 × 1024)
- `Results/embeddings_yamnet_val.npz` (~300 KB, 79 × 1024)
- `Results/embeddings_yamnet_test.npz` (~500 KB, 110 × 1024)
- `Results/model_yamnet_classifier.pt` (~500 KB)
- Training curves (accuracy, F1 over epochs)
- Interactive confusion matrix

**Console Output Example**:
```
Loading cached embeddings from Results directory...
Training...
Epoch 25/25 | Loss: 0.2134 | Train: 94.2% | Val: 86.1% | F1: 0.8543 ✓
Best F1: 0.8543

Test F1: 0.8412 | Accuracy: 84.5%
```

---

## 📊 Expected Outputs

### Directory: `/Manifests/` (5 files)

| File | Size | Description |
|------|------|-------------|
| `manifest_master.csv` | ~100 KB | Complete dataset with all metadata |
| `train.csv` | ~80 KB | Training split (685 samples) |
| `val.csv` | ~10 KB | Validation split (79 samples) |
| `test.csv` | ~15 KB | Test split (110 samples) |
| `acoustic_features_summary.csv` | ~200 KB | Pre-computed acoustic features |

### Directory: `/Results/` (5 files)

| File | Size | Description |
|------|------|-------------|
| `embeddings_yamnet_train.npz` | ~3 MB | YAMNet embeddings for training |
| `embeddings_yamnet_val.npz` | ~300 KB | YAMNet embeddings for validation |
| `embeddings_yamnet_test.npz` | ~500 KB | YAMNet embeddings for test |
| `model_baseline_cnn.pt` | ~2 MB | Best CNN model weights |
| `model_yamnet_classifier.pt` | ~500 KB | Best YAMNet classifier weights |

**Total output size**: ~7 MB

---

## 🎯 Results Summary

### Model Performance Comparison

| Model | Test Accuracy | Macro F1 | Parameters | Training Time |
|-------|---------------|----------|------------|---------------|
| MFCC + SVM | ~65% | ~0.62 | N/A | ~2 min |
| Mel-Spec CNN | ~71% | ~0.70 | 487,877 | ~20 min |
| **YAMNet Transfer** | **~85%** | **~0.84** | 131,845 | ~3 min* |

*Excludes one-time embedding extraction (~30 min)

### Key Findings

1. **YAMNet transfer learning** significantly outperforms traditional approaches
2. **Percussion family** is easiest to classify (high harmonic ratio variability)
3. **Strings vs keyboards** show confusion due to similar spectral characteristics
4. **PCA captures 69.6%** of variance with just 2 components
5. **t-SNE visualization** shows clear family clustering in acoustic feature space

---

## 🔧 Troubleshooting

### Common Issues

#### 1. MATLAB: "Audio Toolbox not found"
```matlab
% Check installed toolboxes
ver

% Install from MATLAB Add-Ons if missing
```

#### 2. Python: "No module named 'soundfile'"
```bash
# Ensure environment is activated
conda activate audio_classification

# Reinstall dependencies
pip install -r requirements.txt
```

#### 3. Bokeh plots not showing in Jupyter
```python
# Add this at the top of notebooks
from bokeh.plotting import output_notebook
output_notebook()

# Then restart kernel and run all
```

#### 4. CUDA out of memory (GPU)
```python
# Reduce batch size in notebooks
BATCH_SIZE = 16  # Instead of 32
```

#### 5. MPS compatibility issues (Apple Silicon)
```python
# If AdaptiveAvgPool2d fails, CNN uses AvgPool2d instead
# This is already implemented in notebook 3
```

#### 6. YAMNet download fails
```python
# Check internet connection
# Or manually download from:
# https://tfhub.dev/google/yamnet/1

# Then load locally:
yamnet_model = hub.load('/path/to/local/yamnet')
```

#### 7. File path issues
```python
# Always use pathlib for cross-platform compatibility
from pathlib import Path
PROJECT_ROOT = Path("/your/absolute/path/here")
```

#### 8. Confusion matrix not visible (Notebook 3)
```python
# Re-run cell 1 to initialize Bokeh
# Then re-run evaluation cells
# Or: Kernel → Restart & Run All
```

---

## 📚 Dependencies Reference

### Core Libraries

```python
numpy==2.2.5          # Numerical computing
pandas==2.3.3         # Data manipulation
scipy==1.13.0         # Scientific computing
```

### Audio Processing

```python
soundfile==0.13.1     # Audio I/O
librosa==0.10.1       # Music/audio analysis
```

### Machine Learning

```python
scikit-learn==1.7.2   # SVM, PCA, metrics
torch==2.0.1          # PyTorch deep learning
torchaudio==2.0.2     # Audio utilities for PyTorch
```

### Deep Learning

```python
tensorflow==2.15.0        # TensorFlow for YAMNet
tensorflow-hub==0.15.0    # Pre-trained model hub
```

### Visualization

```python
bokeh==3.3.0          # Interactive plots
matplotlib==3.8.0     # Static plots
seaborn==0.13.0       # Statistical visualization
```

### Utilities

```python
tqdm==4.66.1          # Progress bars
pathlib==1.0.1        # Path handling
```

---

## 📖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{elec5305_audio_classification,
  author       = {Dawod Ghifari},
  title        = {Instrument Family Classification in Music Recordings Using Audio Signal Processing and Machine Learning},
  year         = {2025},
  institution  = {University of Sydney},
  course       = {ELEC5305 - Audio Signal Processing},
  note         = {Final Project Report}
}
```

---

## 📞 Contact

**Student**: Dawod Ghifari  
**Student ID**: 520140154  
**Course**: ELEC5305 - Audio Signal Processing  
**Institution**: University of Sydney  
**Semester**: 2 - 2025

For questions or issues, please contact: [your.email@sydney.edu.au]

---

## 📄 License

This project is submitted as part of academic coursework at the University of Sydney. All rights reserved. Unauthorized reproduction or distribution is prohibited.

---

## 🙏 Acknowledgments

- **YAMNet**: Google Research - Pre-trained audio event detection model
- **librosa**: McFee et al. - Audio analysis in Python
- **PyTorch**: Meta AI - Deep learning framework
- **TensorFlow Hub**: Google - Pre-trained model repository
- **Bokeh**: Bokeh Development Team - Interactive visualization

---

**Last Updated**: November 16, 2025