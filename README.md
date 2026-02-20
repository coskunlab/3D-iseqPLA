# Python based Robotic Imaging and Staining for modular Spatial Omics (PRISMS)

## Project Overview

This is a spatial transcriptomics and single-cell molecular imaging analysis repository supporting the manuscript "Modular, open-sourced multiplexing for democratizing spatial omics". The codebase processes single-cell microscopy data and generates publication-quality figures for multiple manuscripts.

## Data
Example data: https://figshare.com/articles/dataset/Modular_Open-Sourced_Multiplexing_for_Democratizing_Spatial_Multi-Omics/28646996
The data.zip file should be unzipped and placed in the data folder of this repo to run.

## Environment Setup

The repository requires Conda for environment management. Create the environment with:

```bash
conda env create -f environment.yml
```

This creates an environment with extensive scientific Python packages including data analysis (pandas, numpy, scikit-learn), visualization (matplotlib, seaborn, napari), image processing (scikit-image, opencv, cellpose), and deep learning (PyTorch, torch-geometric).

## Running Figure Generation Scripts

All figure scripts are located in `notebooks/` and output PNG files to `figures/`.

**Run a single figure:**
```bash
python notebooks/Fig_5.py
```

**Run all figures:**
```bash
for script in notebooks/Fig_*.py; do python "$script"; done
```

**Performance expectations:**
- Most scripts: 30 seconds to 3 minutes
- `Fig_S6.py` (computationally intensive): 30+ minutes
- Scripts use `joblib` with threading backend for parallelization

## Codebase Structure

### Core Data Analysis Modules

The `notebooks/` directory contains:

- **Figure Scripts** (`Fig_*.py`): Standalone analysis scripts that load pickle files, perform statistical analysis, generate visualizations, and save PNGs
- **GNN Module** (`notebooks/GNN/`): Graph Neural Network implementation for spatial transcriptomics analysis (data.py, model.py, train.py)
- **Custom Packages**:
  - `pixelator/`: Spatial omics pixel-level analysis package (Pixelgen Technologies)
  - `py_sphere/`: Spherical geometry utilities with Voronoi and circumcircle calculations

### Supporting Code

- **Nikon_GUI/** (`03_run_PRISMS.py`): Microscopy automation script for Nikon hardware
- **Cephla_Squid_GUI/**: Related microscopy GUI software

### Data Directory

The `data/` directory contains experiment-specific subdirectories with:
- `.pkl` files (pickled pandas DataFrames with single-molecule spatial coordinates)
- Excel metadata files
- Organized by experiment name and date

## Key Development Patterns

### Figure Script Architecture

All figure scripts follow a consistent pattern:

1. **Relative Path Resolution**: Use `pathlib.Path(__file__).parent` to resolve data paths relative to script location
   ```python
   SCRIPT_DIR = Path(__file__).parent
   pklPath = SCRIPT_DIR / ".." / "data" / "experiment_name" / "subfolder"
   screenshotSavePath = SCRIPT_DIR / ".." / "figures"
   ```

2. **Data Pipeline**:
   - Load `.pkl` files (pickled DataFrames with molecule-level spatial data)
   - Aggregate per-cell using `groupby` operations
   - Apply statistical analysis (correlations, colocalization)
   - Generate matplotlib/seaborn visualizations
   - Save PNG outputs with `saveFigure()` utility function

3. **Parallel Processing**: Use `joblib.Parallel` with threading backend for processing multiple PKL files
   ```python
   dfCell = Parallel(n_jobs=-1, prefer='threads', verbose=10)(
       delayed(processingFunction)(fileName) for fileName in pklFiles
   )
   ```

### Common Data Column Names

Scripts use consistent column naming across figures:
- Spatial: `Z`, `Y`, `X`
- Cell metadata: `CellLabel`, `MaskNucLabel`, `MaskCytoLabel`
- Experiment metadata: `TumorStage`, `TumorType`, `CellRegion`, `Cycle`, `FOV`
- Marker expressions: `[MarkerName] Protein`, `[MarkerName] RNA Dots`

### Image Processing Dependencies

Figure scripts leverage:
- `scikit-image` (segmentation, morphological operations)
- `opencv-python` (image operations)
- `cellpose` (cell segmentation)
- `nd2`/`nd2reader` (Nikon microscopy file reading)
- `napari` (interactive visualization, not saved to figures)
- `torch` (GPU acceleration for cellpose)

## Testing Verification

There is no automated test suite. To verify changes:

1. Run all affected figure generation scripts
2. Inspect PNG outputs in `figures/` directory
3. Visually compare generated figures with expected patterns

## Code Style

Code is not strictly formatted. New code should follow standard Python conventions (PEP 8) with meaningful variable names and comments explaining complex logic.

## Citation

Zhang, N. et al. Modular, open-sourced multiplexing for democratizing spatial multi-omics. Lab Chip 25, 5379–5392 (2025).

