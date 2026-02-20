#!/usr/bin/env python3
"""
Fine-tune scGPT foundation model for binary classification of in vivo datasets
Task: Classify samples/cells as healthy/control vs diseased

Uses in vivo datasets with health/disease annotations
Tracks training with Weights & Biases (wandb)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Import scGPT components
try:
    from scgpt.model import TransformerModel
    from scgpt.utils import set_seed
    print("Successfully imported scGPT components")
except ImportError as e:
    print(f"Error importing scGPT: {e}")
    print("Please ensure scGPT is installed: pip install scgpt")
    sys.exit(1)

# Import wandb
try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Please install: pip install wandb")
    sys.exit(1)

# Set UTF-8 encoding and line buffering
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

# Configuration
WANDB_API_KEY = "261a9172a7233d8b283ce5e9ec99ea601a59bbd3"
WANDB_PROJECT = "scGPT-invivo-health-disease"
WANDB_ENTITY = None  # Will use default

# Paths - Windows compatible
base_dir = Path('Y:/coskun-lab/Nicky/71 CF AI Foundation model')
data_dir = base_dir / 'Data' / '00 In Vivo RAW'
annotations_path = data_dir / 'health_disease_annotations.csv'
prepared_data_dir = base_dir / 'Data' / 'Prepared_for_Training'
save_dir = base_dir / 'Models' / 'scGPT' / 'invivo'
save_dir.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
SEED = 42
BATCH_SIZE = 16  # Reduced from 32 to avoid OOM errors. With 2 GPUs via DataParallel, each GPU gets BATCH_SIZE/2 = 8
MAX_SEQ_LEN = 3000  # Maximum number of genes per cell
USE_ALL_GENES = True  # Use all available genes (no HVG filtering)
USE_BALANCED_SAMPLING = True  # Downsample majority class to match minority class
EPOCHS = 10
LEARNING_RATE = 5e-5  # Reduced for stability with large dataset
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test mode: limit batches for quick evaluation testing
TEST_MODE = False  # Set to False for full training
TEST_MAX_BATCHES = 1000  # Only train on first 1000 batches in test mode

# Pretrained model configuration
USE_PRETRAINED = True
# Use the same pretrained model that Script 29 downloaded
PRETRAINED_MODEL_PATH = base_dir / 'Models' / 'scGPT' / 'multispecies' / 'pretrained' / 'scGPT_human'

# Model configuration
PAD_TOKEN = "<pad>"
MASK_VALUE = -1
PAD_VALUE = -2
N_LAYERS = 4
N_HEADS = 4
EMBSIZE = 512
D_HFF = 512

print("="*100)
print("scGPT Fine-tuning for In Vivo Health/Disease Classification")
print("="*100)
print(f"Device: {DEVICE}")

# Check for multiple GPUs
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    if n_gpus > 1:
        print(f"Multi-GPU training enabled - will use {n_gpus} GPUs with DataParallel")
else:
    print("No GPU available - using CPU")

print(f"Seed: {SEED}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Use pretrained model: {USE_PRETRAINED}")
print(f"Use all genes: {USE_ALL_GENES}")
print(f"Use balanced sampling: {USE_BALANCED_SAMPLING}")
print()

# Set random seed
set_seed(SEED)

def load_h5ad_dataset(file_path: Path) -> Optional[ad.AnnData]:
    """
    Load dataset from h5ad file
    """
    try:
        print(f"    Loading h5ad file: {file_path.name}")
        adata = sc.read_h5ad(file_path)
        print(f"    Shape: {adata.shape}")
        return adata
    except Exception as e:
        print(f"    Error loading h5ad: {e}")
        return None

def load_h5_dataset(file_path: Path) -> Optional[ad.AnnData]:
    """
    Load dataset from h5 file (10X format)
    """
    try:
        print(f"    Loading h5 file: {file_path.name}")
        adata = sc.read_10x_h5(file_path)
        print(f"    Shape: {adata.shape}")
        return adata
    except Exception as e:
        print(f"    Error loading h5: {e}")
        return None

def load_mtx_dataset(dataset_path: Path) -> Optional[ad.AnnData]:
    """
    Load dataset from MTX format (matrix.mtx, genes.tsv, barcodes.tsv)
    """
    try:
        print(f"    Looking for MTX files...")
        mtx_file = None
        genes_file = None
        barcodes_file = None

        for f in dataset_path.rglob('*.mtx*'):
            mtx_file = f
            break

        for f in dataset_path.rglob('*genes*.tsv*'):
            genes_file = f
            break

        for f in dataset_path.rglob('*barcodes*.tsv*'):
            barcodes_file = f
            break

        if mtx_file and genes_file and barcodes_file:
            print(f"    Loading MTX format...")
            adata = sc.read_mtx(mtx_file).T

            # Load genes
            genes = pd.read_csv(genes_file, sep='\t', header=None)
            adata.var_names = genes[1].values if genes.shape[1] > 1 else genes[0].values

            # Load barcodes
            barcodes = pd.read_csv(barcodes_file, sep='\t', header=None)
            adata.obs_names = barcodes[0].values

            print(f"    Shape: {adata.shape}")
            return adata
    except Exception as e:
        print(f"    Error loading MTX: {e}")

    return None

def load_csv_dataset(file_path: Path) -> Optional[ad.AnnData]:
    """
    Load dataset from CSV format
    """
    try:
        print(f"    Loading CSV file: {file_path.name}")
        expr_df = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')

        # Transpose if needed (genes should be columns)
        if expr_df.shape[0] > expr_df.shape[1]:
            print(f"    Transposing matrix (genes as rows -> genes as columns)")
            expr_df = expr_df.T

        adata = ad.AnnData(X=expr_df.values)
        adata.obs_names = expr_df.index
        adata.var_names = expr_df.columns

        print(f"    Shape: {adata.shape}")
        return adata
    except Exception as e:
        print(f"    Error loading CSV: {e}")
        return None

def load_dataset(dataset_num: int, condition: str, has_controls: str) -> Optional[ad.AnnData]:
    """
    Load a single in vivo dataset and create AnnData object

    Args:
        dataset_num: Dataset folder number
        condition: 'diseased', 'healthy', 'mixed', or 'unknown'
        has_controls: 'yes', 'no', 'likely', 'unknown', or 'N/A'

    Returns:
        AnnData object with expression data and labels
    """
    dataset_path = data_dir / str(dataset_num)

    if not dataset_path.exists():
        print(f"  Warning: Dataset {dataset_num} folder not found")
        return None

    # Check if folder is empty
    files = list(dataset_path.iterdir())
    if len(files) == 0:
        print(f"  Warning: Dataset {dataset_num} folder is empty")
        return None

    print(f"  Loading dataset from {dataset_path}...")

    adata = None

    # Try different file formats in order of preference
    # 1. Try h5ad files (AnnData format)
    h5ad_files = list(dataset_path.glob('*.h5ad'))
    if h5ad_files and adata is None:
        adata = load_h5ad_dataset(h5ad_files[0])

    # 2. Try h5 files (10X format)
    if adata is None:
        h5_files = list(dataset_path.glob('*.h5'))
        if h5_files:
            adata = load_h5_dataset(h5_files[0])

    # 3. Try MTX format
    if adata is None:
        adata = load_mtx_dataset(dataset_path)

    # 4. Try CSV files
    if adata is None:
        # Look for expression matrix files
        csv_files = list(dataset_path.glob('*.csv'))
        expr_files = [f for f in csv_files if any(k in f.name.lower()
                      for k in ['count', 'fpkm', 'tpm', 'matrix', 'expression', 'expr'])
                      and 'metadata' not in f.name.lower()]

        if expr_files:
            adata = load_csv_dataset(expr_files[0])

    # 5. Try RDS files (will need R conversion - skip for now)
    # 6. Try other compressed formats

    if adata is None:
        print(f"  Warning: Could not load any data files from dataset {dataset_num}")
        return None

    # Assign labels based on condition
    # If marked as 'diseased' but has controls, treat as mixed for metadata parsing
    if condition == 'diseased' and has_controls == 'yes':
        condition = 'mixed'
        print(f"    Dataset has controls - will parse metadata for labels")

    if condition == 'diseased':
        # All samples are diseased (no controls)
        adata.obs['label'] = 1
        print(f"    Assigned all cells as diseased (label=1)")
    elif condition == 'healthy':
        # All samples are healthy
        adata.obs['label'] = 0
        print(f"    Assigned all cells as healthy (label=0)")
    elif condition == 'mixed':
        # Dataset contains both healthy and diseased
        # Try to infer from metadata
        label_assigned = False

        # Check for common metadata columns
        for col in adata.obs.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ['condition', 'disease', 'phenotype', 'status', 'group']):
                print(f"    Found potential label column: {col}")

                # Map values to binary labels
                unique_vals = adata.obs[col].unique()
                print(f"    Unique values: {unique_vals}")

                # Create label mapping
                label_map = {}
                for val in unique_vals:
                    val_str = str(val).lower()
                    if any(k in val_str for k in ['control', 'healthy', 'normal', 'uninfected', 'mock', 'wt', 'wild']):
                        label_map[val] = 0
                    elif any(k in val_str for k in ['disease', 'infected', 'treatment', 'patient', 'covid', 'ipf', 'copd', 'asthma']):
                        label_map[val] = 1

                if label_map:
                    adata.obs['label'] = adata.obs[col].map(label_map)

                    # Check if any labels assigned
                    n_labeled = adata.obs['label'].notna().sum()
                    if n_labeled > 0:
                        print(f"    Assigned labels using {col}: {n_labeled}/{len(adata)} cells")
                        label_assigned = True
                        break

        if not label_assigned:
            print(f"    Warning: Mixed dataset but could not assign labels from metadata")
            # If has_controls is 'yes', assume we need both classes
            # For now, skip this dataset
            return None
    else:
        # Unknown condition - skip
        print(f"    Warning: Unknown condition, skipping dataset")
        return None

    # Filter cells without labels (if any)
    if 'label' in adata.obs.columns:
        n_before = len(adata)
        adata = adata[adata.obs['label'].notna()].copy()
        n_after = len(adata)
        if n_before != n_after:
            print(f"    Filtered cells without labels: {n_before} -> {n_after}")

    # Add dataset identifier
    adata.obs['dataset'] = str(dataset_num)

    return adata

def load_all_datasets() -> List[Dict]:
    """
    Load metadata for all in vivo datasets (don't load actual data yet)

    Returns:
        List of dataset metadata dictionaries
    """
    print("Loading health/disease annotations...")
    annotations = pd.read_csv(annotations_path)
    print(f"Found {len(annotations)} datasets")
    print()

    print("Scanning individual datasets...")
    dataset_list = []
    total_cells = 0
    total_healthy = 0
    total_diseased = 0

    for idx, row in annotations.iterrows():
        dataset_num = row['Number']
        dataset_name = row['Dataset']
        condition = row['Condition']
        disease = row['Disease']
        has_controls = row['Has_Controls']
        species = row['Species']

        print(f"\nDataset {dataset_num}: {dataset_name}")
        print(f"  Condition: {condition} | Disease: {disease}")
        print(f"  Has controls: {has_controls} | Species: {species}")

        # Skip unknown or reference atlas datasets for now
        if condition in ['unknown', 'mixed'] and has_controls == 'yes':
            print(f"  Skipping reference atlas dataset")
            continue

        # Skip datasets without data
        dataset_path = data_dir / str(dataset_num)
        if not dataset_path.exists() or len(list(dataset_path.iterdir())) == 0:
            print(f"  Skipping: No data available")
            continue

        adata = load_dataset(dataset_num, condition, has_controls)

        if adata is not None and len(adata) > 0:
            # Check if labels are assigned
            if 'label' in adata.obs.columns:
                n_healthy = (adata.obs['label'] == 0).sum()
                n_disease = (adata.obs['label'] == 1).sum()
                n_cells = len(adata)

                print(f"  ✓ Found: {n_cells} cells ({n_healthy} healthy, {n_disease} diseased)")

                # Determine file format and path by re-scanning the dataset directory
                file_format = None
                file_path = None

                # Check for different file formats
                h5ad_files = list(dataset_path.glob('*.h5ad'))
                if h5ad_files:
                    file_format = 'h5ad'
                    file_path = h5ad_files[0]
                else:
                    h5_files = list(dataset_path.glob('*.h5'))
                    if h5_files:
                        file_format = 'h5'
                        file_path = h5_files[0]
                    else:
                        # Check for MTX directory
                        if (dataset_path / 'matrix.mtx').exists() or (dataset_path / 'matrix.mtx.gz').exists():
                            file_format = 'mtx'
                            file_path = dataset_path
                        else:
                            # Check for CSV files
                            csv_files = list(dataset_path.glob('*.csv'))
                            expr_files = [f for f in csv_files if any(k in f.name.lower()
                                          for k in ['count', 'fpkm', 'tpm', 'matrix', 'expression', 'expr'])
                                          and 'metadata' not in f.name.lower()]
                            if expr_files:
                                file_format = 'csv'
                                file_path = expr_files[0]

                # Store dataset info with FILE PATH instead of loaded data
                dataset_info = {
                    'dataset_num': dataset_num,
                    'dataset_name': dataset_name,
                    'condition': condition,
                    'disease': disease,
                    'species': species,
                    'n_cells': n_cells,
                    'n_healthy': n_healthy,
                    'n_diseased': n_disease,
                    'dataset_path': dataset_path,  # Store path, not data
                    'file_format': file_format,  # Store format
                    'file_path': file_path,  # Specific file path
                    'labels': adata.obs['label'].values,  # Only keep labels in RAM
                    'gene_names': adata.var_names.tolist()  # Keep gene names
                }

                dataset_list.append(dataset_info)

                # Free memory immediately
                del adata

                total_cells += n_cells
                total_healthy += n_healthy
                total_diseased += n_disease
            else:
                print(f"  Warning: No labels assigned, skipping")

    if len(dataset_list) == 0:
        raise ValueError("No datasets loaded successfully!")

    print(f"\n{'='*100}")
    print(f"Dataset inventory complete!")
    print(f"  Total datasets: {len(dataset_list)}")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Healthy/Control: {total_healthy:,} ({total_healthy/total_cells*100:.1f}%)")
    print(f"  Diseased: {total_diseased:,} ({total_diseased/total_cells*100:.1f}%)")

    # Check if we have both classes
    if total_healthy == 0 or total_diseased == 0:
        raise ValueError(f"Cannot perform binary classification! Need both healthy and diseased cells. "
                        f"Found {total_healthy} healthy and {total_diseased} diseased cells.")

    return dataset_list

def preprocess_dataset(adata: ad.AnnData) -> ad.AnnData:
    """
    Preprocess a single dataset for scGPT

    Args:
        adata: AnnData object for one dataset

    Returns:
        Preprocessed AnnData
    """
    # Quality control
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata

def preprocess_all_datasets(dataset_list: List[Dict]) -> List[Dict]:
    """
    Skip preprocessing since we'll do it on-demand during batch iteration

    Args:
        dataset_list: List of dataset dictionaries

    Returns:
        Same list (no preprocessing needed)
    """
    print(f"\n{'='*100}")
    print("Skipping preprocessing - will preprocess on-demand during training")
    print(f"{'='*100}")

    return dataset_list

def balance_class_distribution(dataset_list: List[Dict], split_name: str = "train") -> List[Dict]:
    """
    Balance class distribution by downsampling majority class to match minority class

    Args:
        dataset_list: List of dataset dictionaries with indices and labels
        split_name: Name of the split (for logging)

    Returns:
        Balanced dataset list with updated indices
    """
    print(f"\n{'='*100}")
    print(f"Applying balanced class subsampling to {split_name} set...")
    print(f"{'='*100}")

    # Collect all cell indices by class across all datasets
    healthy_cells = []  # List of (dataset_idx, cell_idx) tuples
    diseased_cells = []

    for dataset_idx, dataset_info in enumerate(dataset_list):
        indices = dataset_info['indices']
        labels = dataset_info['labels']

        for cell_idx in indices:
            label = labels[cell_idx]
            if label == 0:
                healthy_cells.append((dataset_idx, cell_idx))
            else:
                diseased_cells.append((dataset_idx, cell_idx))

    n_healthy = len(healthy_cells)
    n_diseased = len(diseased_cells)

    print(f"\nBefore balancing:")
    print(f"  Healthy: {n_healthy:,}")
    print(f"  Diseased: {n_diseased:,}")
    print(f"  Imbalance ratio: {n_diseased/n_healthy:.1f}:1 (diseased:healthy)")

    # Determine minority class size (target size)
    target_size = min(n_healthy, n_diseased)

    print(f"\nDownsampling to balanced set:")
    print(f"  Target size per class: {target_size:,}")
    print(f"  Total cells after balancing: {target_size * 2:,}")

    # Randomly downsample majority class
    np.random.shuffle(healthy_cells)
    np.random.shuffle(diseased_cells)

    balanced_healthy = healthy_cells[:target_size]
    balanced_diseased = diseased_cells[:target_size]

    # Create new dataset list with updated indices
    # Group cells by dataset
    dataset_indices = {i: {'healthy': [], 'diseased': []} for i in range(len(dataset_list))}

    for dataset_idx, cell_idx in balanced_healthy:
        dataset_indices[dataset_idx]['healthy'].append(cell_idx)

    for dataset_idx, cell_idx in balanced_diseased:
        dataset_indices[dataset_idx]['diseased'].append(cell_idx)

    # Update dataset_list with new indices
    balanced_datasets = []
    total_cells = 0
    total_healthy = 0
    total_diseased = 0

    for dataset_idx, dataset_info in enumerate(dataset_list):
        new_indices = dataset_indices[dataset_idx]['healthy'] + dataset_indices[dataset_idx]['diseased']

        if len(new_indices) > 0:  # Only keep datasets that have cells
            new_info = dataset_info.copy()
            new_info['indices'] = np.array(new_indices)
            new_info['n_cells'] = len(new_indices)
            new_info['n_healthy'] = len(dataset_indices[dataset_idx]['healthy'])
            new_info['n_diseased'] = len(dataset_indices[dataset_idx]['diseased'])

            balanced_datasets.append(new_info)

            total_cells += new_info['n_cells']
            total_healthy += new_info['n_healthy']
            total_diseased += new_info['n_diseased']

    print(f"\nAfter balancing:")
    print(f"  Datasets: {len(balanced_datasets)}")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Healthy: {total_healthy:,} ({total_healthy/total_cells*100:.1f}%)")
    print(f"  Diseased: {total_diseased:,} ({total_diseased/total_cells*100:.1f}%)")
    print(f"  Reduction: {len(healthy_cells) + len(diseased_cells):,} → {total_cells:,} ({(1 - total_cells/(len(healthy_cells) + len(diseased_cells)))*100:.1f}% fewer cells)")

    return balanced_datasets

def prepare_data_for_scgpt(dataset_list: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Prepare train/test split for scGPT without loading all data into memory

    Args:
        dataset_list: List of preprocessed datasets

    Returns:
        train_datasets, test_datasets, gene_vocab
    """
    print(f"\n{'='*100}")
    print("Preparing train/test split...")
    print(f"{'='*100}")

    # Build unified gene vocabulary across all datasets
    print("\nBuilding unified gene vocabulary...")
    all_genes = set()
    for dataset_info in dataset_list:
        all_genes.update(dataset_info['gene_names'])

    gene_vocab = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
    print(f"  Total unique genes: {len(gene_vocab):,}")

    # Split each dataset 80/20 for train/test
    train_datasets = []
    test_datasets = []

    total_train_cells = 0
    total_test_cells = 0
    train_healthy = 0
    train_diseased = 0
    test_healthy = 0
    test_diseased = 0

    for dataset_info in dataset_list:
        n_cells = dataset_info['n_cells']
        labels = dataset_info['labels']

        # Create shuffled indices
        indices = np.arange(n_cells)
        np.random.shuffle(indices)

        # 80/20 split
        split_idx = int(0.8 * n_cells)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        # Count labels
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        train_healthy += (train_labels == 0).sum()
        train_diseased += (train_labels == 1).sum()
        test_healthy += (test_labels == 0).sum()
        test_diseased += (test_labels == 1).sum()

        # Create train dataset info
        train_info = dataset_info.copy()
        train_info['indices'] = train_idx  # Store which cells to use
        train_info['n_cells'] = len(train_idx)
        train_info['split'] = 'train'
        train_datasets.append(train_info)
        total_train_cells += len(train_idx)

        # Create test dataset info
        test_info = dataset_info.copy()
        test_info['indices'] = test_idx  # Store which cells to use
        test_info['n_cells'] = len(test_idx)
        test_info['split'] = 'test'
        test_datasets.append(test_info)
        total_test_cells += len(test_idx)

    print(f"\nTrain set (BEFORE balancing): {total_train_cells:,} cells across {len(train_datasets)} datasets")
    print(f"  Healthy: {train_healthy:,}")
    print(f"  Diseased: {train_diseased:,}")

    print(f"\nTest set (BEFORE balancing): {total_test_cells:,} cells across {len(test_datasets)} datasets")
    print(f"  Healthy: {test_healthy:,}")
    print(f"  Diseased: {test_diseased:,}")

    # Apply balanced class subsampling if enabled
    if USE_BALANCED_SAMPLING:
        train_datasets = balance_class_distribution(train_datasets, split_name="train")
        test_datasets = balance_class_distribution(test_datasets, split_name="test")
    else:
        print(f"\nBalanced sampling disabled - using full imbalanced dataset")

    return train_datasets, test_datasets, gene_vocab

def check_pretrained_model():
    """Check if pretrained scGPT model exists (shared with Script 29)"""
    model_file = PRETRAINED_MODEL_PATH / "best_model.pt"

    if model_file.exists():
        print(f"✓ Using pretrained model from: {model_file}")
        print(f"  (Shared with Script 29)")
        return True
    else:
        print(f"✗ Pretrained model not found at: {model_file}")
        print(f"  This model should have been downloaded by Script 29")
        return False

def load_pretrained_model(vocab_size: int, gene_vocab: dict = None) -> TransformerModel:
    """
    Load pretrained scGPT model for fine-tuning

    Args:
        vocab_size: Size of gene vocabulary
        gene_vocab: Dictionary mapping gene names to indices

    Returns:
        Pretrained TransformerModel
    """
    print(f"\n{'='*100}")
    print("Creating scGPT model...")
    print(f"{'='*100}")

    # Create a simple vocab-like object if needed
    if gene_vocab is not None:
        vocab_dict = gene_vocab.copy()
        vocab_dict[PAD_TOKEN] = vocab_size - 2
        vocab_dict["<cls>"] = vocab_size - 1
    else:
        vocab_dict = {PAD_TOKEN: 0, "<cls>": 1}

    # Create model
    model = TransformerModel(
        ntoken=vocab_size,
        d_model=EMBSIZE,
        nhead=N_HEADS,
        d_hid=D_HFF,
        nlayers=N_LAYERS,
        nlayers_cls=2,
        n_cls=2,  # Binary classification
        vocab=vocab_dict,
        dropout=0.2,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        n_input_bins=51,  # CRITICAL: Bin expression values into discrete categories
        ecs_threshold=0.3,  # Elastic cell similarity threshold
        explicit_zero_prob=True,  # CRITICAL: Properly handle zero expression values
        use_fast_transformer=True,
        pre_norm=False,
    )

    # Load pretrained weights if available
    if USE_PRETRAINED:
        model_file = PRETRAINED_MODEL_PATH / "best_model.pt"
        if model_file.exists():
            try:
                print(f"\nLoading pretrained weights from {model_file}...")
                pretrained_dict = torch.load(model_file, map_location=DEVICE)
                model_dict = model.state_dict()

                # Filter out incompatible keys (like embedding size mismatches)
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                 if k in model_dict and v.shape == model_dict[k].shape}

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print(f"✓ Loaded {len(pretrained_dict)} pretrained parameter tensors")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Proceeding with randomly initialized weights")
        else:
            print(f"\nWarning: Pretrained model file not found at {model_file}")
            print("Proceeding with randomly initialized weights")

    # Freeze early layers for parameter-efficient fine-tuning
    if USE_PRETRAINED:
        print("\nFreezing early layers for parameter-efficient fine-tuning...")

        # Freeze embedding layers
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'requires_grad_'):
            model.encoder.requires_grad_(False)
            print("  ✓ Froze embedding layers")

        # Freeze transformer encoder layers
        # Keep last 2 layers trainable (layers 2, 3 out of 4)
        if hasattr(model, 'transformer_encoder'):
            n_layers_to_freeze = N_LAYERS - 2  # Freeze first 2 out of 4 layers

            # Access transformer layers
            if hasattr(model.transformer_encoder, 'layers'):
                for i, layer in enumerate(model.transformer_encoder.layers):
                    if i < n_layers_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"  ✓ Froze first {n_layers_to_freeze}/{N_LAYERS} transformer layers")
                print(f"  ✓ Keeping last {N_LAYERS - n_layers_to_freeze} transformer layers trainable")

        # Keep value encoder trainable if it exists
        # Keep all normalization layers trainable for better adaptation
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True

    # Add classification head (always trainable)
    n_classes = 2  # Binary classification
    if not hasattr(model, 'cls_decoder'):
        print("  ✓ Adding trainable classification head")
        import torch.nn as nn
        model.cls_decoder = nn.Sequential(
            nn.Linear(EMBSIZE, EMBSIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(EMBSIZE, n_classes)
        )

    model.to(DEVICE)

    # Wrap model with DataParallel for multi-GPU training
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"\nWrapping model with DataParallel for {torch.cuda.device_count()} GPUs...")
        model = torch.nn.DataParallel(model)
        print(f"  ✓ Model will use GPUs: {list(range(torch.cuda.device_count()))}")

    # Count parameters
    # If using DataParallel, access the underlying model via model.module
    param_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_params = sum(p.numel() for p in param_model.parameters())
    trainable_params = sum(p.numel() for p in param_model.parameters() if p.requires_grad)

    print(f"\nModel created successfully!")
    print(f"  Use pretrained: {USE_PRETRAINED}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    if USE_PRETRAINED:
        print(f"  Parameters frozen: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params*100:.1f}%)")

    return model

def preload_all_datasets(dataset_list: List[Dict]) -> List[ad.AnnData]:
    """
    Pre-load all preprocessed datasets into RAM for fast training

    Returns:
        List of loaded AnnData objects
    """
    print(f"\n{'='*100}")
    print("Pre-loading all datasets into RAM...")
    print(f"{'='*100}\n")

    loaded_datasets = []
    total_cells = 0

    for dataset_info in dataset_list:
        dataset_num = dataset_info['dataset_num']
        file_path = Path(dataset_info['file_path'])

        print(f"Loading dataset {dataset_num}: {file_path.name}")
        adata = sc.read_h5ad(file_path)

        # Add labels back
        adata.obs['label'] = dataset_info['labels']
        adata.obs['dataset'] = str(dataset_num)

        loaded_datasets.append(adata)
        total_cells += len(adata)
        print(f"  ✓ Loaded {len(adata):,} cells")

    print(f"\n✓ All datasets loaded into RAM: {total_cells:,} total cells")
    print(f"{'='*100}\n")

    return loaded_datasets

class FastBatchIterator:
    """Fast iterator using pre-loaded data in RAM"""

    def __init__(self, dataset_list: List[Dict], loaded_datasets: List[ad.AnnData],
                 batch_size: int, gene_vocab: Dict[str, int], max_seq_len: int, shuffle: bool = True):
        self.dataset_list = dataset_list
        self.loaded_datasets = loaded_datasets
        self.batch_size = batch_size
        self.gene_vocab = gene_vocab
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle

        # Pre-compute vocabulary indices for faster lookup
        self.gene_to_idx = gene_vocab

        # Create index mapping: (dataset_idx, cell_idx)
        self.cell_indices = []
        for dataset_idx, dataset_info in enumerate(dataset_list):
            indices = dataset_info['indices']  # Train or test indices
            for cell_idx in indices:
                self.cell_indices.append((dataset_idx, cell_idx))

        self.total_cells = len(self.cell_indices)
        self.n_batches = self.total_cells // batch_size

        if self.shuffle:
            np.random.shuffle(self.cell_indices)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        """Iterate over batches using pre-loaded data"""
        batch_data = []

        for dataset_idx, cell_idx in self.cell_indices:
            # Get data from pre-loaded dataset (already in RAM!)
            adata = self.loaded_datasets[dataset_idx]
            dataset_info = self.dataset_list[dataset_idx]

            # Get cell data
            cell_expr = adata.X[cell_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[cell_idx].flatten()
            gene_names = adata.var_names
            label = int(dataset_info['labels'][cell_idx])

            # Select top expressed genes
            top_indices = np.argsort(cell_expr)[-self.max_seq_len:]
            genes = gene_names[top_indices].tolist()
            values = cell_expr[top_indices].tolist()

            batch_data.append({
                'genes': genes,
                'values': values,
                'label': label
            })

            # Yield batch when full
            if len(batch_data) == self.batch_size:
                yield self._prepare_batch(batch_data)
                batch_data = []

        # Yield remaining data
        if len(batch_data) > 0:
            yield self._prepare_batch(batch_data)

    def _prepare_batch(self, batch_data):
        """Convert batch data to tensors"""
        batch_genes = [item['genes'] for item in batch_data]
        batch_values = [item['values'] for item in batch_data]
        batch_labels = [item['label'] for item in batch_data]

        # Convert to tensors
        max_len = max(len(g) for g in batch_genes)
        gene_ids = torch.zeros(len(batch_genes), max_len, dtype=torch.long)
        expr_values = torch.zeros(len(batch_genes), max_len, dtype=torch.float)
        padding_mask = torch.ones(len(batch_genes), max_len, dtype=torch.bool)  # True = padding

        for i, (genes, values) in enumerate(zip(batch_genes, batch_values)):
            seq_len = len(genes)
            for j, (gene, val) in enumerate(zip(genes, values)):
                if gene in self.gene_to_idx:
                    gene_ids[i, j] = self.gene_to_idx[gene]
                expr_values[i, j] = val
            # Mark actual tokens as not padding (False)
            padding_mask[i, :seq_len] = False

        labels = torch.tensor(batch_labels, dtype=torch.long)

        return gene_ids, expr_values, labels, padding_mask

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }

    # Add AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['auc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics

def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    checkpoint_dir = save_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if not checkpoint_files:
        return None

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('epoch')[1].replace('.pt', '')))
    return checkpoint_files[-1]

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load checkpoint and return starting epoch"""
    print(f"\n{'='*100}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*100}\n")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Handle DataParallel wrapper when loading state dict
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    best_test_f1 = checkpoint.get('best_test_f1', 0.0)

    print(f"✓ Resuming from epoch {start_epoch + 1}/{EPOCHS}")
    print(f"✓ Best test F1 so far: {best_test_f1:.4f}\n")

    return start_epoch, best_test_f1

def save_checkpoint(epoch, model, optimizer, scheduler, best_test_f1):
    """Save training checkpoint at epoch boundaries"""
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"

    # Handle DataParallel wrapper when saving state dict
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_test_f1': best_test_f1,
    }

    torch.save(checkpoint_data, checkpoint_path)
    print(f"  ✓ Saved checkpoint to: {checkpoint_path}")

def train_model(model, train_datasets, test_datasets, gene_vocab):
    """
    Train the scGPT model with wandb tracking using batch iteration
    """
    print(f"\n{'='*100}")
    print("Starting training...")
    print(f"{'='*100}\n")

    # Pre-load all datasets into RAM for fast training
    print("Pre-loading training datasets...")
    loaded_train_datasets = preload_all_datasets(train_datasets)
    print("Pre-loading test datasets...")
    loaded_test_datasets = preload_all_datasets(test_datasets)

    # Calculate total cells
    total_train_cells = sum(d['n_cells'] for d in train_datasets)
    total_test_cells = sum(d['n_cells'] for d in test_datasets)

    # Initialize wandb
    # Set API key as environment variable for wandb
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "architecture": "scGPT",
            "task": "health_disease_classification",
            "dataset": "in_vivo_lung",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "embsize": EMBSIZE,
            "d_hff": D_HFF,
            "use_all_genes": USE_ALL_GENES,
            "use_pretrained": USE_PRETRAINED,
            "use_balanced_sampling": USE_BALANCED_SAMPLING,
            "max_seq_len": MAX_SEQ_LEN,
            "seed": SEED,
            "train_samples": total_train_cells,
            "test_samples": total_test_cells,
            "n_genes": len(gene_vocab),
            "n_datasets": len(train_datasets),
        }
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    total_steps = (total_train_cells // BATCH_SIZE) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=warmup_steps,
        T_mult=1,
        eta_min=1e-6
    )

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Check for existing checkpoint
    checkpoint_path = find_latest_checkpoint()
    start_epoch = 0
    best_test_f1 = 0.0
    checkpoint_loaded = False

    if checkpoint_path:
        start_epoch, best_test_f1 = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        checkpoint_loaded = True
    else:
        print("\nNo checkpoint found - starting training from scratch\n")

    # Training loop - if resuming from checkpoint, start from next epoch
    resume_from_epoch = start_epoch + 1 if checkpoint_loaded else start_epoch
    for epoch in range(resume_from_epoch, EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*80}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Create batch iterator for training
        train_iterator = FastBatchIterator(
            train_datasets,
            loaded_train_datasets,
            BATCH_SIZE,
            gene_vocab,
            MAX_SEQ_LEN,
            shuffle=True
        )

        n_batches = len(train_iterator)
        print(f"Total training batches: {n_batches}")
        if TEST_MODE:
            print(f"TEST MODE: Will only train on first {TEST_MAX_BATCHES} batches")

        for batch_idx, (gene_ids, expr_values, labels, padding_mask) in enumerate(train_iterator):
            # Test mode: break after reaching max batches
            if TEST_MODE and batch_idx >= TEST_MAX_BATCHES:
                print(f"\nTEST MODE: Reached {TEST_MAX_BATCHES} batches, stopping training phase...")
                break

            # Move to device
            gene_ids = gene_ids.to(DEVICE)
            expr_values = expr_values.to(DEVICE)
            labels = labels.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()

            try:
                output = model(gene_ids, expr_values, padding_mask)

                # Get cell embeddings and pass through classification head
                if isinstance(output, dict):
                    # Use cell embeddings for classification
                    if 'cell_emb' in output:
                        cell_emb = output['cell_emb']
                    elif 'cls_output' in output:
                        cell_emb = output['cls_output']
                    else:
                        # Print available keys for debugging on first batch
                        if batch_idx == 0:
                            print(f"  Model output keys: {output.keys()}")
                        # Use first available output
                        cell_emb = list(output.values())[0]
                elif isinstance(output, tuple):
                    # Model returns tuple, embeddings are usually first
                    cell_emb = output[0]
                else:
                    # Direct tensor output
                    cell_emb = output

                # Pass through classification head
                # Handle DataParallel wrapper
                cls_decoder = model.module.cls_decoder if isinstance(model, torch.nn.DataParallel) else model.cls_decoder
                logits = cls_decoder(cell_emb)

                loss = criterion(logits, labels)

                # Check for NaN/inf early
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/inf loss detected at batch {batch_idx}")
                    print(f"    Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                    print(f"    Expr values stats: min={expr_values.min().item():.4f}, max={expr_values.max().item():.4f}, mean={expr_values.mean().item():.4f}")
                    print(f"  Skipping this batch...")
                    continue

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Track metrics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                if batch_idx % 50 == 0:
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    print(f"  Batch {batch_idx}/{n_batches} - Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")

            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                import traceback
                import sys
                traceback.print_exc()
                sys.stdout.flush()  # Ensure error is written to log
                # Continue training despite error
                continue

        # Training metrics
        train_loss = train_loss / n_batches if n_batches > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0

        print(f"\nTraining - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Save checkpoint after training (before evaluation)
        print(f"\nSaving checkpoint after epoch {epoch + 1}...")
        save_checkpoint(epoch, model, optimizer, scheduler, best_test_f1)

        # Evaluation phase
        print(f"\nStarting evaluation on test set...")
        model.eval()
        test_loss = 0.0
        all_test_preds = []
        all_test_labels = []
        all_test_probs = []

        # Create batch iterator for testing
        test_iterator = FastBatchIterator(
            test_datasets,
            loaded_test_datasets,
            BATCH_SIZE,
            gene_vocab,
            MAX_SEQ_LEN,
            shuffle=False
        )

        n_test_batches = len(test_iterator)
        print(f"Total test batches: {n_test_batches}")

        with torch.no_grad():
            for batch_idx, (gene_ids, expr_values, labels, padding_mask) in enumerate(test_iterator):
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f"  Evaluating batch {batch_idx}/{n_test_batches}...")
                # Move to device
                gene_ids = gene_ids.to(DEVICE)
                expr_values = expr_values.to(DEVICE)
                labels = labels.to(DEVICE)
                padding_mask = padding_mask.to(DEVICE)

                try:
                    output = model(gene_ids, expr_values, padding_mask)

                    # Get cell embeddings and pass through classification head
                    if isinstance(output, dict):
                        # Use cell embeddings for classification
                        if 'cell_emb' in output:
                            cell_emb = output['cell_emb']
                        elif 'cls_output' in output:
                            cell_emb = output['cls_output']
                        else:
                            cell_emb = list(output.values())[0]
                    elif isinstance(output, tuple):
                        cell_emb = output[0]
                    else:
                        cell_emb = output

                    # Pass through classification head
                    # Handle DataParallel wrapper
                    cls_decoder = model.module.cls_decoder if isinstance(model, torch.nn.DataParallel) else model.cls_decoder
                    logits = cls_decoder(cell_emb)

                    loss = criterion(logits, labels)

                    # Check for NaN/inf loss during evaluation
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  WARNING: NaN/inf loss in evaluation batch {batch_idx}, skipping...")
                        continue

                    test_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)
                    _, predicted = torch.max(logits, 1)

                    all_test_preds.extend(predicted.cpu().numpy())
                    all_test_labels.extend(labels.cpu().numpy())
                    all_test_probs.extend(probs.cpu().numpy())

                except Exception as e:
                    print(f"  Error in test batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        test_loss = test_loss / n_test_batches if n_test_batches > 0 else 0

        # Calculate comprehensive metrics
        test_metrics = calculate_metrics(
            np.array(all_test_labels),
            np.array(all_test_preds),
            np.array(all_test_probs)
        )

        print(f"Testing  - Loss: {test_loss:.4f}")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC:       {test_metrics['auc']:.4f}")

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_metrics['accuracy'],
            "test_precision": test_metrics['precision'],
            "test_recall": test_metrics['recall'],
            "test_f1": test_metrics['f1'],
            "test_auc": test_metrics['auc'],
            "test_specificity": test_metrics.get('specificity', 0),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Save best model
        if test_metrics['f1'] > best_test_f1:
            best_test_f1 = test_metrics['f1']
            model_save_path = save_dir / f"best_model_epoch{epoch+1}_f1{test_metrics['f1']:.4f}.pt"

            # Handle DataParallel wrapper when saving state dict
            model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'test_metrics': test_metrics,
                'test_loss': test_loss,
            }, model_save_path)
            print(f"\n  ✓ Saved best model to: {model_save_path}")

    wandb.finish()

    print(f"\n{'='*100}")
    print(f"Training complete!")
    print(f"Best test F1 score: {best_test_f1:.4f}")
    print(f"{'='*100}\n")

def load_prepared_data():
    """Load prepared data from Script 32"""
    print(f"\n{'='*100}")
    print("Loading prepared data from Script 32...")
    print(f"{'='*100}\n")

    # Load train datasets
    train_path = prepared_data_dir / 'train_datasets.pkl'
    print(f"Loading train datasets from: {train_path}")
    with open(train_path, 'rb') as f:
        train_datasets = pickle.load(f)
    print(f"  ✓ Loaded {len(train_datasets)} training dataset splits")

    # Load test datasets
    test_path = prepared_data_dir / 'test_datasets.pkl'
    print(f"Loading test datasets from: {test_path}")
    with open(test_path, 'rb') as f:
        test_datasets = pickle.load(f)
    print(f"  ✓ Loaded {len(test_datasets)} test dataset splits")

    # Load gene vocabulary
    vocab_path = prepared_data_dir / 'gene_vocab.pkl'
    print(f"Loading gene vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        gene_vocab = pickle.load(f)
    print(f"  ✓ Loaded gene vocabulary with {len(gene_vocab):,} genes")

    # Calculate statistics
    total_train_cells = sum(d['n_cells'] for d in train_datasets)
    total_test_cells = sum(d['n_cells'] for d in test_datasets)
    train_healthy = sum(d['n_healthy'] for d in train_datasets)
    train_diseased = sum(d['n_diseased'] for d in train_datasets)
    test_healthy = sum(d['n_healthy'] for d in test_datasets)
    test_diseased = sum(d['n_diseased'] for d in test_datasets)

    print(f"\nDataset Summary:")
    print(f"  Train: {total_train_cells:,} cells ({train_healthy:,} healthy, {train_diseased:,} diseased)")
    print(f"  Test:  {total_test_cells:,} cells ({test_healthy:,} healthy, {test_diseased:,} diseased)")
    print(f"  Genes: {len(gene_vocab):,}")

    return train_datasets, test_datasets, gene_vocab

def main():
    """Main training pipeline"""

    try:
        # Load prepared data from Script 32
        train_datasets, test_datasets, gene_vocab = load_prepared_data()

        # Apply balanced class subsampling if enabled
        if USE_BALANCED_SAMPLING:
            train_datasets = balance_class_distribution(train_datasets, split_name="train")
            test_datasets = balance_class_distribution(test_datasets, split_name="test")
        else:
            print(f"\nBalanced sampling disabled - using full imbalanced dataset")

        # Create vocabulary size
        vocab_size = len(gene_vocab) + 2  # +2 for special tokens
        print(f"\nVocabulary size (with special tokens): {vocab_size:,}")

        # Check for pretrained model (shared with Script 29)
        if USE_PRETRAINED:
            model_exists = check_pretrained_model()
            if not model_exists:
                print("Warning: Proceeding without pretrained weights")

        # Create model
        model = load_pretrained_model(vocab_size, gene_vocab)

        # Train using batch iteration
        train_model(model, train_datasets, test_datasets, gene_vocab)

    except Exception as e:
        print(f"\nError in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
