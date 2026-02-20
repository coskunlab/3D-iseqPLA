#!/usr/bin/env python3
"""
Plot Gene Latent Space with Highlighted Important Genes
Visualizes gene embeddings from trained scGPT model using UMAP
Highlights important genes and user-selected genes of interest
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import dimensionality reduction
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: UMAP not available, will use t-SNE instead")
    UMAP_AVAILABLE = False

# Import scGPT components
try:
    from scgpt.model import TransformerModel
    from scgpt.utils import set_seed
    print("[OK] Successfully imported scGPT components")
except ImportError as e:
    print(f"Error importing scGPT: {e}")
    sys.exit(1)

# Configuration
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_dir = Path('Y:/coskun-lab/Nicky/71 CF AI Foundation model/Models/scGPT/multispecies')
data_dir = Path('Y:/coskun-lab/Nicky/71 CF AI Foundation model/Data')
select_genes_file = data_dir / '01_select_genes.txt'
plot_dir = Path('Y:/coskun-lab/Nicky/71 CF AI Foundation model/Figures/gene_latent_space')
plot_dir.mkdir(parents=True, exist_ok=True)

# Model configuration (must match training)
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
PAD_VALUE = -2
N_LAYERS = 12
N_HEADS = 8
EMBSIZE = 512
D_HID = 512
DROPOUT = 0.2

# Visualization parameters
N_TOP_GENES = 50  # Number of top important genes to highlight
USE_UMAP = UMAP_AVAILABLE  # Use UMAP if available, otherwise t-SNE

print("="*100)
print("Gene Latent Space Visualization with Highlighted Genes")
print("="*100)
print(f"Device: {DEVICE}")
print(f"Model directory: {model_dir}")
print(f"Plot directory: {plot_dir}")
print(f"Dimensionality reduction: {'UMAP' if USE_UMAP else 't-SNE'}")
print()

set_seed(SEED)

# ============================================================================
# Load Data
# ============================================================================
print("="*100)
print("Loading Training Data")
print("="*100)

data_file = Path('/coskun-lab/Nicky/71 CF AI Foundation model/Data/00 In Vitro RAW/converted_anndata/scGPT_multispecies_training_corpus.h5ad')

if not data_file.exists():
    print(f"Error: Data file not found: {data_file}")
    sys.exit(1)

print(f"Loading from: {data_file.name}")
adata = sc.read_h5ad(data_file)

# Filter genes with zero expression
gene_nonzero = (adata.X != 0).sum(axis=0)
if hasattr(gene_nonzero, 'A1'):
    gene_nonzero = gene_nonzero.A1
genes_to_keep = gene_nonzero > 0
adata = adata[:, genes_to_keep].copy()

print(f"  Cells: {adata.n_obs:,}")
print(f"  Genes: {adata.n_vars:,}")

# ============================================================================
# Reconstruct Vocabulary
# ============================================================================
print("\n" + "="*100)
print("Reconstructing Vocabulary")
print("="*100)

gene_names = adata.var_names.tolist()
vocab = {gene: idx for idx, gene in enumerate(gene_names)}
vocab[PAD_TOKEN] = len(vocab)
vocab[CLS_TOKEN] = len(vocab)
VOCAB_SIZE = len(vocab)

print(f"[OK] Vocabulary size: {VOCAB_SIZE:,}")

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
    fast_transformer_backend="flash"
)

# Add classification head
n_classes = 2
model.classifier = torch.nn.Linear(EMBSIZE, n_classes)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"[OK] Model loaded successfully")

# ============================================================================
# Extract Gene Embeddings
# ============================================================================
print("\n" + "="*100)
print("Extracting Gene Embeddings")
print("="*100)

# Get gene embeddings from the model
# The encoder.embedding is the gene embedding layer
if hasattr(model, 'encoder') and hasattr(model.encoder, 'embedding'):
    gene_embeddings = model.encoder.embedding.weight.data.cpu().numpy()
    # Remove special tokens
    gene_embeddings = gene_embeddings[:len(gene_names)]
else:
    print("Error: Could not find gene embedding layer")
    sys.exit(1)

print(f"Gene embeddings shape: {gene_embeddings.shape}")
print(f"  Number of genes: {gene_embeddings.shape[0]}")
print(f"  Embedding dimension: {gene_embeddings.shape[1]}")

# ============================================================================
# Identify Important Genes
# ============================================================================
print("\n" + "="*100)
print("Identifying Important Genes")
print("="*100)

# Calculate gene importance based on:
# 1. Expression variance across cells
# 2. Differential expression between control and diseased

# Get expression matrix
if hasattr(adata.X, 'toarray'):
    expr_matrix = adata.X.toarray()
else:
    expr_matrix = adata.X

# Calculate variance for each gene
gene_var = np.var(expr_matrix, axis=0)

# Calculate differential expression (mean difference between control and diseased)
control_mask = adata.obs['label'] == 0
diseased_mask = adata.obs['label'] == 1

mean_control = expr_matrix[control_mask].mean(axis=0)
mean_diseased = expr_matrix[diseased_mask].mean(axis=0)
gene_diff = np.abs(mean_diseased - mean_control)

# Combine metrics (weighted average)
gene_importance = 0.5 * (gene_var / gene_var.max()) + 0.5 * (gene_diff / gene_diff.max())

# Get top N important genes
top_gene_indices = np.argsort(gene_importance)[-N_TOP_GENES:]
top_gene_names = [gene_names[i] for i in top_gene_indices]

print(f"Top {N_TOP_GENES} important genes identified")
print(f"Top 10: {top_gene_names[-10:]}")

# ============================================================================
# Load Select Genes
# ============================================================================
print("\n" + "="*100)
print("Loading Select Genes")
print("="*100)

if not select_genes_file.exists():
    print(f"Warning: Select genes file not found: {select_genes_file}")
    select_genes = []
else:
    with open(select_genes_file, 'r') as f:
        select_genes = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(select_genes)} select genes:")
    print(f"  {select_genes}")

# Known gene name mappings (common aliases to standard nomenclature)
GENE_ALIASES = {
    # NF-κB pathway
    'p65': ['RELA', 'Rela'],
    'p105': ['NFKB1', 'Nfkb1'],
    'p50': ['NFKB1', 'Nfkb1'],  # Processed form of p105
    'a20': ['TNFAIP3', 'Tnfaip3'],
    'ikkb': ['IKBKB', 'Ikbkb'],
    'ikkg': ['IKBKG', 'Ikbkg', 'NEMO'],
    'traf-5': ['TRAF5', 'Traf5'],
    'traf-2': ['TRAF2', 'Traf2'],
    'tradd': ['TRADD', 'Tradd'],
    'gapdh': ['GAPDH', 'Gapdh']
}

# Map select genes to dataset gene names (with known aliases)
def find_matching_gene(query_gene, gene_list):
    """Find best matching gene name using known aliases and fuzzy matching"""
    query_lower = query_gene.lower()

    # Check known aliases first
    if query_lower in GENE_ALIASES:
        for alias in GENE_ALIASES[query_lower]:
            if alias in gene_list:
                return alias

    # Exact match
    for gene in gene_list:
        if gene.lower() == query_lower:
            return gene

    # Partial match (query in gene name) - but longer than 3 chars to avoid spurious matches
    if len(query_lower) > 3:
        for gene in gene_list:
            if query_lower in gene.lower():
                return gene

    # Reverse partial match (gene name in query)
    for gene in gene_list:
        if len(gene) > 3 and gene.lower() in query_lower:
            return gene

    return None

matched_select_genes = []
unmatched_genes = []
# Create mapping from matched name back to original name for labeling
matched_to_original = {}

for select_gene in select_genes:
    matched = find_matching_gene(select_gene, gene_names)
    if matched:
        matched_select_genes.append(matched)
        matched_to_original[matched] = select_gene
        print(f"  '{select_gene}' -> '{matched}'")
    else:
        unmatched_genes.append(select_gene)
        print(f"  '{select_gene}' -> NOT FOUND")

print(f"\nMatched {len(matched_select_genes)}/{len(select_genes)} select genes")
if unmatched_genes:
    print(f"Unmatched genes: {unmatched_genes}")

# ============================================================================
# Dimensionality Reduction
# ============================================================================
print("\n" + "="*100)
print(f"Performing Dimensionality Reduction ({('UMAP' if USE_UMAP else 't-SNE')})")
print("="*100)

# Standardize embeddings
scaler = StandardScaler()
gene_embeddings_scaled = scaler.fit_transform(gene_embeddings)

# Reduce to 2D
if USE_UMAP:
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=SEED)
    gene_coords_2d = reducer.fit_transform(gene_embeddings_scaled)
else:
    print("Running t-SNE...")
    reducer = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000)
    gene_coords_2d = reducer.fit_transform(gene_embeddings_scaled)

print(f"[OK] Reduced to 2D: {gene_coords_2d.shape}")

# ============================================================================
# Create DataFrame for Plotting
# ============================================================================
print("\n" + "="*100)
print("Preparing Visualization Data")
print("="*100)

plot_df = pd.DataFrame({
    'gene': gene_names,
    'x': gene_coords_2d[:, 0],
    'y': gene_coords_2d[:, 1],
    'importance': gene_importance,
    'is_important': [gene in top_gene_names for gene in gene_names],
    'is_select': [gene in matched_select_genes for gene in gene_names]
})

# Create category for coloring
def assign_category(row):
    if row['is_select']:
        return 'Select Genes'
    elif row['is_important']:
        return 'Important Genes'
    else:
        return 'Other Genes'

plot_df['category'] = plot_df.apply(assign_category, axis=1)

print(f"Gene categories:")
print(f"  Other genes: {(plot_df['category'] == 'Other Genes').sum()}")
print(f"  Important genes: {(plot_df['category'] == 'Important Genes').sum()}")
print(f"  Select genes: {(plot_df['category'] == 'Select Genes').sum()}")

# ============================================================================
# Create Visualization
# ============================================================================
print("\n" + "="*100)
print("Creating Visualization")
print("="*100)

# Set style with larger fonts
sns.set(font_scale=1.5)
sns.set_style("white")

# Define colors
colors = {
    'Other Genes': '#CCCCCC',      # Light gray
    'Important Genes': '#3498db',  # Blue
    'Select Genes': '#e74c3c'      # Red
}

# Create figure (smaller size for better font proportions)
fig, ax = plt.subplots(figsize=(10, 8))

# Plot in order: other genes first, then important, then select (so they're on top)
for category in ['Other Genes', 'Important Genes', 'Select Genes']:
    subset = plot_df[plot_df['category'] == category]

    if category == 'Other Genes':
        # Other genes: small, transparent
        ax.scatter(subset['x'], subset['y'],
                  c=colors[category], s=10, alpha=0.3,
                  label=f'{category} (n={len(subset)})',
                  edgecolors='none')
    else:
        # Important/Select genes: larger, opaque, with edge
        ax.scatter(subset['x'], subset['y'],
                  c=colors[category], s=100, alpha=0.8,
                  label=f'{category} (n={len(subset)})',
                  edgecolors='black', linewidths=0.5)

        # Add gene labels for select genes (using original names)
        if category == 'Select Genes':
            for idx, row in subset.iterrows():
                # Use original name if available, otherwise use matched name
                label = matched_to_original.get(row['gene'], row['gene'])
                ax.annotate(label,
                           (row['x'], row['y']),
                           xytext=(5, 5), textcoords='offset points',
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   edgecolor='black',
                                   alpha=0.7))

# Customize plot
method_name = 'UMAP' if USE_UMAP else 't-SNE'
ax.set_xlabel(f'{method_name} 1', fontweight='bold')
ax.set_ylabel(f'{method_name} 2', fontweight='bold')
ax.set_title(f'Gene Latent Space Visualization\nscGPT Multi-Species Model',
             fontweight='bold', pad=20)

# Legend outside plot
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plot_path = plot_dir / f'gene_latent_space_{method_name.lower()}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {plot_path}")
plt.close()

# ============================================================================
# Create Additional Plots
# ============================================================================

# Plot 2: Colored by gene importance (continuous scale)
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(plot_df['x'], plot_df['y'],
                    c=plot_df['importance'],
                    cmap='viridis', s=20, alpha=0.6)

# Add colorbar (positioned to not overlap with legend)
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Gene Importance', rotation=270, labelpad=25, fontweight='bold')

# Highlight select genes
if len(matched_select_genes) > 0:
    select_subset = plot_df[plot_df['is_select']]
    ax.scatter(select_subset['x'], select_subset['y'],
              c='red', s=150, alpha=0.9,
              edgecolors='black', linewidths=1.5,
              marker='*', label='Select Genes')

    # Add labels (using original names)
    for idx, row in select_subset.iterrows():
        # Use original name if available, otherwise use matched name
        label = matched_to_original.get(row['gene'], row['gene'])
        ax.annotate(label,
                   (row['x'], row['y']),
                   xytext=(5, 5), textcoords='offset points',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white',
                           edgecolor='black',
                           alpha=0.7))

ax.set_xlabel(f'{method_name} 1', fontweight='bold')
ax.set_ylabel(f'{method_name} 2', fontweight='bold')
ax.set_title(f'Gene Importance in Latent Space\nscGPT Multi-Species Model',
             fontweight='bold', pad=20)

if len(matched_select_genes) > 0:
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), framealpha=0.9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plot_path = plot_dir / f'gene_importance_{method_name.lower()}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {plot_path}")
plt.close()

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*100)
print("Saving Results")
print("="*100)

# Save coordinates and metadata
results_df = plot_df.copy()
results_df[f'{method_name.lower()}_1'] = results_df['x']
results_df[f'{method_name.lower()}_2'] = results_df['y']
results_df = results_df[['gene', f'{method_name.lower()}_1', f'{method_name.lower()}_2',
                         'importance', 'category']]

csv_path = plot_dir / f'gene_latent_space_{method_name.lower()}_coordinates.csv'
results_df.to_csv(csv_path, index=False)
print(f"[OK] Saved coordinates: {csv_path}")

# Save top important genes
top_genes_df = pd.DataFrame({
    'gene': top_gene_names,
    'importance': [gene_importance[gene_names.index(g)] for g in top_gene_names]
}).sort_values('importance', ascending=False)

top_genes_path = plot_dir / 'top_important_genes.csv'
top_genes_df.to_csv(top_genes_path, index=False)
print(f"[OK] Saved top genes: {top_genes_path}")

# Save matched select genes
if matched_select_genes:
    select_df = pd.DataFrame({
        'original_name': [sg for sg in select_genes if find_matching_gene(sg, gene_names)],
        'matched_name': matched_select_genes
    })
    select_path = plot_dir / 'matched_select_genes.csv'
    select_df.to_csv(select_path, index=False)
    print(f"[OK] Saved matched genes: {select_path}")

print("\n" + "="*100)
print("VISUALIZATION COMPLETE")
print("="*100)
print(f"\nAll plots saved to: {plot_dir}")
print(f"\nSummary:")
print(f"  Total genes visualized: {len(gene_names):,}")
print(f"  Important genes highlighted: {N_TOP_GENES}")
print(f"  Select genes matched: {len(matched_select_genes)}")
print(f"  Dimensionality reduction: {method_name}")
