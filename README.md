# ASP Project 6: Shazam-style Audio Fingerprinting System

A peak-constellation based audio fingerprinting system similar to Shazam. This system can identify songs from 3-second audio clips.

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Data Files](#-data-files)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Scripts Overview](#-scripts-overview)
- [Usage](#-usage)
- [Important Notes](#-important-notes)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- **Two Fingerprinting Algorithms**:
  - **Design A (Baseline)**: Traditional peak-constellation based method
  - **Design B (Chroma)**: Pitch-robust chroma-based method
  
- **Multi-tempo Search**: Supports audio identification at different playback speeds

- **Robustness Evaluation**:
  - Pitch shift robustness testing
  - Tempo change robustness testing
  - Recognition accuracy under noisy conditions

- **Interactive UI**:
  - Streamlit Web UI for real-time audio identification
  - Fingerprint matching visualization

## 🔧 Requirements

- **Python**: 3.11+
- **Conda**: For environment management
- **Operating System**: macOS / Linux / Windows

## 📦 Installation

### 1. Create and Activate Conda Environment

```bash
# If the asp environment already exists, activate it
conda activate asp

# If not, create the environment (configure according to project requirements)
conda create -n asp python=3.11
conda activate asp
```

### 2. Install Python Dependencies

The project primarily depends on the following Python packages:

```bash
# Core scientific computing libraries
conda install numpy scipy matplotlib

# Audio processing
conda install librosa soundfile

# Web UI
pip install streamlit

# Data visualization (optional, for advanced UI)
pip install plotly pandas

# MATLAB file reading
pip install scipy  # scipy.io for reading .mat files
```

Alternatively, install all dependencies using pip:

```bash
pip install numpy scipy matplotlib librosa soundfile streamlit plotly pandas
```

## 📁 Data Files

### ⚠️ Important: Project6_musicDB.mat File

**`Project6_musicDB.mat` is the core data file of this project. All functionality strongly depends on this file.**

#### File Purpose
- Contains the complete music database (musicDB)
- Stores metadata for each song: title, genre, and audio signal data
- Used for building fingerprint databases and conducting song identification tests

#### File Status
- **Required File**: The project cannot run without this file
- **Added to .gitignore**: Excluded from Git tracking due to large file size (~176 MB)
- **Must be kept locally**: Must remain in the project root directory for normal operation

#### How to Obtain
If the file is missing after cloning the repository from Git:
1. Obtain `Project6_musicDB.mat` separately from the project provider
2. Place the file in the project root directory: `/Project6/Project6_musicDB.mat`
3. Ensure the file path is correct, as all scripts read from this location by default

#### File Structure
The MATLAB file contains a structure array named `musicDB`, where each element contains:
- `title`: Song title
- `genre`: Music genre
- `signal`: Audio signal data (sampling rate 16 kHz)

## 🚀 Quick Start

### 1. Basic System Evaluation

Run the system self-check script to test recognition accuracy and performance:

```bash
conda run -n asp python evaluate_system.py
```

### 2. Design Comparison and Ablation Study

Compare performance of different design approaches:

```bash
conda run -n asp python compare_designs.py
```

### 3. Launch Web UI (Simple Version)

Start the basic Streamlit interface:

```bash
conda run -n asp streamlit run ui_app.py
```

Then open `http://localhost:8501` in your browser.

### 4. Launch Advanced Interactive UI

Start the feature-rich visualization interface:

```bash
conda run -n asp streamlit run ui_demo.py
```

## 📂 Project Structure

```
Project6/
├── shazam_system.py          # Core fingerprinting system
├── ui_app.py                  # Simple Streamlit UI
├── ui_demo.py                 # Advanced interactive UI (with visualization)
├── evaluate_system.py         # System performance evaluation
├── compare_designs.py         # Design comparison and ablation study
├── evaluate_variations.py     # Parameter variation evaluation
├── plot_variation_curves.py  # Generate pitch/tempo curve plots
├── plot_variation_curves_hq.py # High-quality version of curve plots
├── run_heatmap.py            # Generate pitch-tempo heatmap
├── run_pitch_curve.py        # Generate pitch curve
├── run_tempo_curve.py        # Generate tempo curve
├── Project6_musicDB.mat      # ⚠️ Core data file (required)
├── Project6.pdf              # Project documentation
├── README.md                 # This file
├── report_ablation.md        # Ablation study report
├── .gitignore                # Git ignore configuration
├── cache/                    # Cache directory (fingerprint data)
├── fingerprint_cache/        # Fingerprint cache
└── plots/                    # Generated plot output directory
    ├── pitch_curve.png
    ├── tempo_curve.png
    └── pitch_tempo_heatmap.png
```

## 📝 Scripts Overview

### Core Module

#### `shazam_system.py`
Core fingerprinting system providing the following main functions:
- `load_music_db(mat_path)`: Load music database
- `compute_fingerprints(music_db)`: Build baseline fingerprint database
- `compute_fingerprints_chroma(music_db)`: Build chroma-based fingerprint database
- `identify_song(clip, fingerprints)`: Identify the song corresponding to an audio clip
- `identify_song_chroma(clip, fingerprints)`: Identify using chroma features
- `identify_song_multi_tempo(clip, fingerprints)`: Multi-tempo search identification

### Evaluation Scripts

#### `evaluate_system.py`
System self-check script that evaluates:
- Recognition accuracy for clean audio
- Recognition accuracy for noisy audio
- Identification latency

#### `compare_designs.py`
Design comparison including:
- Baseline design vs. chroma-based design
- Ablation study (parameter impact analysis)
- Memory usage and build time comparison

#### `evaluate_variations.py`
Evaluates the impact of different parameter variations on system performance

### Visualization Scripts

#### `plot_variation_curves.py`
Generates curves showing the impact of pitch and tempo changes on recognition accuracy:
- `plots/pitch_curve.png`: Pitch shift accuracy curve
- `plots/tempo_curve.png`: Tempo change accuracy curve
- `plots/pitch_tempo_heatmap.png`: Pitch-tempo joint heatmap

Run with:
```bash
conda run -n asp python plot_variation_curves.py
```

#### `run_heatmap.py`, `run_pitch_curve.py`, `run_tempo_curve.py`
Standalone scripts for generating heatmap, pitch curve, and tempo curve respectively

### UI Applications

#### `ui_app.py`
Simple Streamlit interface supporting:
- Upload audio files for identification
- Select different fingerprinting algorithms (Baseline / Chroma)
- Display identification results

#### `ui_demo.py`
Feature-rich advanced interface supporting:
- Random audio clip selection from database
- Upload local audio files
- Audio augmentation (noise, pitch, tempo)
- Fingerprint visualization
- Matching process analysis

## 💻 Usage

### Method 1: Using Python API

```python
from shazam_system import (
    load_music_db,
    compute_fingerprints,
    identify_song
)
import numpy as np

# Load database
music_db = load_music_db("Project6_musicDB.mat")

# Build fingerprint database
fingerprints = compute_fingerprints(music_db)

# Prepare audio clip (3 seconds, 16 kHz sampling rate)
clip = np.random.randn(48000)  # Example: 3 sec * 16000 Hz

# Identify song
song_id = identify_song(clip, fingerprints, fs=16000)
print(f"Identification result: Song index {song_id}")
```

### Method 2: Using Web UI

1. Launch UI:
   ```bash
   conda run -n asp streamlit run ui_demo.py
   ```

2. Open the displayed URL in your browser (usually `http://localhost:8501`)

3. Select audio source:
   - Random selection from database
   - Upload local audio file

4. Run identification and view results

### Method 3: Running Evaluation Scripts

```bash
# System evaluation
conda run -n asp python evaluate_system.py

# Design comparison
conda run -n asp python compare_designs.py

# Generate curve plots
conda run -n asp python plot_variation_curves.py
```

## ⚠️ Important Notes

### 1. Data File Dependency
- **Ensure `Project6_musicDB.mat` file exists in the project root directory**
- If the file is missing, all scripts will fail to run
- The file has been added to `.gitignore` and will not be synced via Git

### 2. Environment Configuration
- Ensure you use the correct Conda environment (`asp`)
- All commands should be executed from the project root directory
- If you encounter import errors, check Python path and dependency installation

### 3. Cache Files
- The project generates cache files (`cache/`, `fingerprint_cache/`)
- These directories have been added to `.gitignore`
- First run may take longer to build the fingerprint database

### 4. Output Files
- Plot outputs go to the `plots/` directory
- Scripts will automatically create the directory if it doesn't exist

### 5. Performance Considerations
- Initial database loading and fingerprint building may take some time
- Web UI uses Streamlit caching to speed up repeated operations
- Large file processing may require significant memory

## 🔍 Troubleshooting

### Issue: Project6_musicDB.mat not found
**Solution**:
1. Verify the file is in the project root directory
2. Check if the file path is correct
3. If the file is missing, obtain it from the project provider

### Issue: Module import failures
**Solution**:
```bash
# Ensure you're in the correct environment
conda activate asp

# Check if dependencies are installed
pip list | grep streamlit
pip list | grep librosa

# Reinstall dependencies
pip install -r requirements.txt  # If requirements.txt exists
```

### Issue: Streamlit fails to run
**Solution**:
```bash
# Update Streamlit
pip install --upgrade streamlit

# Check if port is occupied
lsof -ti:8501 | xargs kill -9  # macOS/Linux
```

## 📚 References

- Project documentation: `Project6.pdf`
- Ablation study report: `report_ablation.md`
- Core system documentation: See function docstrings in `shazam_system.py`

## 📄 License

Please refer to the license information provided by the project maintainer.

---

**Last Updated**: 2025-12-02
