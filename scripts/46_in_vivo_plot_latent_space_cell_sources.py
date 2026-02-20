#!/usr/bin/env python3
"""
Visualize Cell Latent Space of In Vivo scGPT Model
Extracts cell embeddings from the trained model and creates UMAP/t-SNE visualizations
colored by phenotype (control vs diseased) and dataset source

Designed for the in vivo model trained with script 35
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import scGPT components
try:
    from scgpt.model import TransformerModel
    from scgpt.utils import set_seed
    print("[OK] Successfully imported scGPT components")
except ImportError as e:
    print(f"[ERROR] Error importing scGPT: {e}")
    sys.exit(1)

# Import dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: UMAP not installed. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

from sklearn.manifold import TSNE

# Configuration
SEED = 42
set_seed(SEED)

# Paths
base_dir = Path('Y:/coskun-lab/Nicky/71 CF AI Foundation model')
model_dir = base_dir / 'Models' / 'scGPT' / 'invivo'
prepared_data_dir = base_dir / 'Data' / 'Prepared_for_Training'
output_dir = base_dir / 'Figures' / 'invivo_cell_latent_space'
output_dir.mkdir(parents=True, exist_ok=True)

# Model parameters (must match script 35)
BATCH_SIZE = 32
MAX_SEQ_LEN = 3000
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
PAD_VALUE = -2
N_LAYERS = 4
N_HEADS = 4
EMBSIZE = 512
D_HID = 512
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visualization parameters
N_CELLS_TO_SAMPLE = 10000  # Sample cells for faster visualization
METHODS_TO_RUN = ['TSNE']
if UMAP_AVAILABLE:
    METHODS_TO_RUN.append('UMAP')

print("="*100)
print("scGPT In Vivo Model - Cell Latent Space Visualization")
print("="*100)
print(f"Device: {DEVICE}")
print(f"Model directory: {model_dir}")
print(f"Output directory: {output_dir}")
print(f"Cells to sample: {N_CELLS_TO_SAMPLE:,}")
print(f"Methods: {', '.join(METHODS_TO_RUN)}")
print()

# Set plot style
sns.set(font_scale=1.5)
sns.set_style("white")

# ============================================================================
# Load Gene Vocabulary
# ============================================================================
print("="*100)
print("Loading Gene Vocabulary")
print("="*100)

vocab_file = prepared_data_dir / 'gene_vocab.pkl'
print(f"Loading vocabulary from: {vocab_file}")

with open(vocab_file, 'rb') as f:
    vocab_data = pickle.load(f)

# Build vocabulary
if isinstance(vocab_data, dict) and PAD_TOKEN in vocab_data:
    vocab = vocab_data
    gene_names = [g for g in vocab.keys() if g not in [PAD_TOKEN, CLS_TOKEN]]
else:
    gene_names = vocab_data if isinstance(vocab_data, list) else list(vocab_data.keys())
    vocab = {gene: idx for idx, gene in enumerate(gene_names)}
    vocab[PAD_TOKEN] = len(gene_names)
    vocab[CLS_TOKEN] = len(gene_names) + 1

VOCAB_SIZE = len(vocab)
print(f"[OK] Vocabulary size: {VOCAB_SIZE:,}")
print(f"  Genes: {len(gene_names):,}")

# ============================================================================
# Load Test Data
# ============================================================================
print("\n" + "="*100)
print("Loading Test Data")
print("="*100)

test_file = prepared_data_dir / 'test_datasets.pkl'
print(f"Loading test data from: {test_file}")

with open(test_file, 'rb') as f:
    test_datasets = pickle.load(f)

print(f"Loading {len(test_datasets)} test datasets...")

all_test_cells = []
for dataset_info in test_datasets:
    file_path = Path(dataset_info['file_path'])
    dataset_num = dataset_info['dataset_num']
    indices = dataset_info['indices']
    labels = dataset_info['labels']

    print(f"  Loading dataset {dataset_num}: {file_path.name}")
    adata = sc.read_h5ad(file_path)
    adata = adata[indices].copy()

    # Add labels and dataset info
    adata.obs['label'] = [labels[i] for i in range(len(indices))]
    adata.obs['dataset'] = str(dataset_num)

    all_test_cells.append(adata)

# Make gene names unique before concatenating
for adata in all_test_cells:
    adata.var_names_make_unique()

# Concatenate all test datasets
print("\nConcatenating all test datasets...")
adata_test = ad.concat(all_test_cells, axis=0, join='outer', fill_value=0)

print(f"\n[OK] Loaded {adata_test.n_obs:,} test cells")
print(f"  Control: {(adata_test.obs['label'] == 0).sum():,}")
print(f"  Diseased: {(adata_test.obs['label'] == 1).sum():,}")

# Align to vocabulary
print("\nAligning test data to model vocabulary...")
common_genes = set(gene_names).intersection(set(adata_test.var_names))
print(f"  Common genes: {len(common_genes):,} / {len(gene_names):,}")

if len(common_genes) < len(gene_names):
    missing_genes = set(gene_names) - common_genes
    missing_adata = ad.AnnData(
        X=np.zeros((adata_test.n_obs, len(missing_genes))),
        obs=adata_test.obs,
        var=pd.DataFrame(index=list(missing_genes))
    )
    adata_test = ad.concat([adata_test[:, list(common_genes)], missing_adata], axis=1)

adata_test = adata_test[:, gene_names].copy()
print(f"[OK] Final test dataset: {adata_test.n_obs:,} cells × {adata_test.n_vars:,} genes")

# Sample cells if dataset is too large
if adata_test.n_obs > N_CELLS_TO_SAMPLE:
    print(f"\nSubsampling {N_CELLS_TO_SAMPLE:,} cells from {adata_test.n_obs:,} for visualization...")
    np.random.seed(SEED)
    sample_indices = np.random.choice(adata_test.n_obs, N_CELLS_TO_SAMPLE, replace=False)
    adata_sample = adata_test[sample_indices].copy()
    print(f"[OK] Sampled {adata_sample.n_obs:,} cells")
else:
    adata_sample = adata_test

# ============================================================================
# Load Trained Model
# ============================================================================
print("\n" + "="*100)
print("Loading Trained Model")
print("="*100)

# Find the best model checkpoint
checkpoints = list(model_dir.glob('best_model_epoch*.pt'))
if not checkpoints:
    print("Error: No model checkpoints found!")
    sys.exit(1)

checkpoint_path = sorted(checkpoints)[-1]
print(f"Loading checkpoint: {checkpoint_path}")

# Initialize model
model = TransformerModel(
    ntoken=VOCAB_SIZE,
    d_model=EMBSIZE,
    nhead=N_HEADS,
    d_hid=D_HID,
    nlayers=N_LAYERS,
    nlayers_cls=2,
    n_cls=2,
    vocab=vocab,
    dropout=DROPOUT,
    pad_token=PAD_TOKEN,
    pad_value=PAD_VALUE,
    do_mvc=False,
    do_dab=False,
    use_batch_labels=False,
    domain_spec_batchnorm=False,
    n_input_bins=51,
    ecs_threshold=0.3,
    explicit_zero_prob=True,
    use_fast_transformer=True,
    pre_norm=False,
)

# Add manual cls_decoder if not present
if not hasattr(model, 'cls_decoder'):
    import torch.nn as nn
    model.cls_decoder = nn.Sequential(
        nn.Linear(EMBSIZE, EMBSIZE),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(EMBSIZE, 2)
    )

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"[OK] Model loaded successfully")
print(f"  Checkpoint epoch: {checkpoint['epoch']}")

# ============================================================================
# Extract Cell Embeddings
# ============================================================================
print("\n" + "="*100)
print("Extracting Cell Embeddings")
print("="*100)

def prepare_batch(adata_batch):
    """Prepare a batch of cells for scGPT input"""
    batch_size = adata_batch.n_obs
    gene_ids_batch = []
    values_batch = []
    padding_masks = []

    for i in range(batch_size):
        # Get expression for this cell
        if hasattr(adata_batch.X, 'toarray'):
            expr = adata_batch.X[i].toarray().flatten()
        else:
            expr = adata_batch.X[i].flatten()

        # Get non-zero genes
        nonzero_idx = np.nonzero(expr)[0]

        # Limit to max_len genes
        if len(nonzero_idx) > MAX_SEQ_LEN:
            top_idx = np.argsort(expr[nonzero_idx])[-MAX_SEQ_LEN:]
            nonzero_idx = nonzero_idx[top_idx]

        # Get gene IDs and values
        genes = nonzero_idx
        vals = expr[nonzero_idx]

        # Create padding mask
        mask = np.zeros(MAX_SEQ_LEN, dtype=bool)

        # Pad if needed
        if len(genes) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(genes)
            genes = np.concatenate([genes, np.full(pad_len, vocab[PAD_TOKEN])])
            vals = np.concatenate([vals, np.full(pad_len, PAD_VALUE)])
            mask[len(nonzero_idx):] = True

        gene_ids_batch.append(genes)
        values_batch.append(vals)
        padding_masks.append(mask)

    gene_ids_batch = torch.LongTensor(gene_ids_batch).to(DEVICE)
    values_batch = torch.FloatTensor(values_batch).to(DEVICE)
    padding_mask = torch.BoolTensor(padding_masks).to(DEVICE)

    return gene_ids_batch, values_batch, padding_mask

# Extract embeddings in batches
print(f"Extracting embeddings for {adata_sample.n_obs:,} cells...")
all_embeddings = []
all_labels = []
all_datasets = []

n_batches = adata_sample.n_obs // BATCH_SIZE + (1 if adata_sample.n_obs % BATCH_SIZE != 0 else 0)

with torch.no_grad():
    for batch_idx in range(n_batches):
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{n_batches} batches...")

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, adata_sample.n_obs)

        # Get batch
        adata_batch = adata_sample[start_idx:end_idx]

        # Prepare batch
        try:
            gene_ids_batch, values_batch, padding_mask = prepare_batch(adata_batch)

            # Forward pass
            output = model(gene_ids_batch, values_batch, src_key_padding_mask=padding_mask)

            # Get CLS token embedding
            if isinstance(output, dict):
                cls_output = output.get('cls_output', output.get('cell_emb'))
            else:
                cls_output = output[:, 0, :]

            # Store embeddings
            all_embeddings.append(cls_output.cpu().numpy())
            all_labels.extend(adata_batch.obs['label'].values)
            all_datasets.extend(adata_batch.obs['dataset'].values)
        except Exception as e:
            print(f"  Warning: Skipping batch {batch_idx} due to error: {e}")
            continue

# Concatenate all embeddings
embeddings = np.vstack(all_embeddings)
labels = np.array(all_labels)
datasets = np.array(all_datasets)

print(f"\n[OK] Extracted embeddings: {embeddings.shape}")
print(f"  Shape: {embeddings.shape[0]:,} cells × {embeddings.shape[1]} dimensions")

# ============================================================================
# Dimensionality Reduction
# ============================================================================
print("\n" + "="*100)
print("Dimensionality Reduction")
print("="*100)

# Store results
reductions = {}

# t-SNE
if 'TSNE' in METHODS_TO_RUN:
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    coords_tsne = tsne.fit_transform(embeddings)
    reductions['TSNE'] = coords_tsne
    print("  t-SNE complete")

# UMAP
if 'UMAP' in METHODS_TO_RUN:
    print("\nComputing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    coords_umap = reducer.fit_transform(embeddings)
    reductions['UMAP'] = coords_umap
    print("  UMAP complete")

# ============================================================================
# Create Visualizations
# ============================================================================
print("\n" + "="*100)
print("Creating Visualizations")
print("="*100)

# Create color maps
phenotype_colors = {0: '#4ECDC4', 1: '#FF6B6B'}  # Control: turquoise, Diseased: red
phenotype_labels = {0: 'Control', 1: 'Diseased'}

for method_name, coords in reductions.items():
    print(f"\nCreating {method_name} plots...")

    # Create DataFrame
    plot_df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'label': labels,
        'dataset': datasets,
        'phenotype': [phenotype_labels[l] for l in labels]
    })

    # Figure 1: Colored by phenotype
    fig, ax = plt.subplots()

    for label_val, phenotype_name in phenotype_labels.items():
        subset = plot_df[plot_df['label'] == label_val]
        ax.scatter(subset['x'], subset['y'],
                  c=phenotype_colors[label_val],
                  s=20, alpha=0.6,
                  label=f'{phenotype_name} (n={len(subset):,})',
                  edgecolors='none')

    ax.set_xlabel(f'{method_name} 1', fontweight='bold', fontsize=14)
    ax.set_ylabel(f'{method_name} 2', fontweight='bold', fontsize=14)
    ax.set_title(f'Cell Latent Space ({method_name})\nColored by Phenotype - In Vivo Model',
                fontweight='bold', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f'cell_latent_space_{method_name.lower()}_phenotype.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    # Figure 2: Colored by dataset source
    fig, ax = plt.subplots()

    # Get unique datasets and assign colors
    unique_datasets = sorted(plot_df['dataset'].unique())
    dataset_colors = sns.color_palette("husl", len(unique_datasets))
    dataset_color_map = {ds: dataset_colors[i] for i, ds in enumerate(unique_datasets)}

    for dataset_id in unique_datasets:
        subset = plot_df[plot_df['dataset'] == dataset_id]
        ax.scatter(subset['x'], subset['y'],
                  c=[dataset_color_map[dataset_id]],
                  s=20, alpha=0.5,
                  label=f'Dataset {dataset_id} (n={len(subset):,})',
                  edgecolors='none')

    ax.set_xlabel(f'{method_name} 1', fontweight='bold', fontsize=14)
    ax.set_ylabel(f'{method_name} 2', fontweight='bold', fontsize=14)
    ax.set_title(f'Cell Latent Space ({method_name})\nColored by Dataset Source - In Vivo Model',
                fontweight='bold', fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f'cell_latent_space_{method_name.lower()}_dataset.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    # Save coordinates to CSV
    save_df = plot_df[['x', 'y', 'phenotype', 'dataset']].copy()
    csv_path = output_dir / f'cell_coords_{method_name.lower()}.csv'
    save_df.to_csv(csv_path, index=False)
    print(f"  Saved coordinates: {csv_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*100)
print("CELL LATENT SPACE ANALYSIS COMPLETE")
print("="*100)
print(f"\nAll plots saved to: {output_dir}")
print(f"\nGenerated visualizations:")
for method in reductions.keys():
    print(f"  - cell_latent_space_{method.lower()}_phenotype.png")
    print(f"  - cell_latent_space_{method.lower()}_dataset.png")
    print(f"  - cell_coords_{method.lower()}.csv")

print(f"\nCells analyzed: {embeddings.shape[0]:,}")
print(f"  Control: {(labels == 0).sum():,}")
print(f"  Diseased: {(labels == 1).sum():,}")
print(f"  Unique datasets: {len(unique_datasets)}")
print(f"{'='*100}\n")
