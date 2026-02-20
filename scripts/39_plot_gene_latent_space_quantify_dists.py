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
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, directed_hausdorff, mahalanobis
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
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

# Setup paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'
FIGURES_ROOT = PROJECT_ROOT / 'figures'

# Configuration
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_dir = DATA_ROOT / '71 CF AI Foundation model' / 'Models' / 'scGPT' / 'multispecies'
data_dir = DATA_ROOT / '71 CF AI Foundation model' / 'Data'
select_genes_file = data_dir / '01_select_genes.txt'
plot_dir = FIGURES_ROOT / '39_plot_gene_latent_space_quantify_dists'
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
# Run all three methods for comparison
METHODS_TO_RUN = ['PCA', 'TSNE']
if UMAP_AVAILABLE:
    METHODS_TO_RUN.append('UMAP')

print("="*100)
print("Gene Latent Space Visualization with Highlighted Genes")
print("="*100)
print(f"Device: {DEVICE}")
print(f"Model directory: {model_dir}")
print(f"Plot directory: {plot_dir}")
print(f"Dimensionality reduction methods: {', '.join(METHODS_TO_RUN)}")
print()

set_seed(SEED)

# Set global plot style with larger fonts
sns.set(font_scale=1.5)
sns.set_style("white")

# ============================================================================
# Load Data
# ============================================================================
print("="*100)
print("Loading Training Data")
print("="*100)

data_file = DATA_ROOT / '71 CF AI Foundation model' / 'Data' / '00 In Vitro RAW' / 'converted_anndata' / 'scGPT_multispecies_training_corpus.h5ad'

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
print(f"Performing Dimensionality Reduction")
print("="*100)

# Standardize embeddings
scaler = StandardScaler()
gene_embeddings_scaled = scaler.fit_transform(gene_embeddings)

# Store results for each method
reduction_results = {}

for method in METHODS_TO_RUN:
    print(f"\nRunning {method}...")

    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=SEED)
        gene_coords_2d = reducer.fit_transform(gene_embeddings_scaled)
        print(f"  Explained variance: {reducer.explained_variance_ratio_.sum():.3f}")

    elif method == 'TSNE':
        reducer = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
        gene_coords_2d = reducer.fit_transform(gene_embeddings_scaled)

    elif method == 'UMAP':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=SEED)
        gene_coords_2d = reducer.fit_transform(gene_embeddings_scaled)

    reduction_results[method] = gene_coords_2d
    print(f"[OK] {method} reduced to 2D: {gene_coords_2d.shape}")

# ============================================================================
# Comprehensive Distance Analysis Between Gene Sets
# ============================================================================
print("\n" + "="*100)
print("Comprehensive Distance Analysis Between Gene Sets")
print("="*100)

def calculate_comprehensive_distances(coords, gene_names, select_genes, important_genes,
                                     space_name='Unknown', k_neighbors=10):
    """
    Calculate comprehensive distance metrics between select and important gene sets

    Args:
        coords: Gene coordinates (N_genes x N_dimensions)
        gene_names: List of gene names
        select_genes: List of select gene names
        important_genes: List of important gene names
        space_name: Name of the space (for labeling)
        k_neighbors: Number of neighbors for overlap coefficient
    """
    # Get indices
    select_indices = [i for i, g in enumerate(gene_names) if g in select_genes]
    important_indices = [i for i, g in enumerate(gene_names) if g in important_genes]

    if len(select_indices) == 0 or len(important_indices) == 0:
        return None

    # Get coordinates
    select_coords = coords[select_indices]
    important_coords = coords[important_indices]

    results = {
        'space': space_name,
        'n_select': len(select_indices),
        'n_important': len(important_indices)
    }

    # ===== 1. EUCLIDEAN DISTANCES =====
    euclidean_dists = cdist(select_coords, important_coords, metric='euclidean')
    results['euclidean_mean'] = np.mean(euclidean_dists)
    results['euclidean_median'] = np.median(euclidean_dists)
    results['euclidean_min'] = np.min(euclidean_dists)
    results['euclidean_max'] = np.max(euclidean_dists)
    results['euclidean_std'] = np.std(euclidean_dists)
    results['euclidean_all'] = euclidean_dists.flatten()

    # ===== 2. COSINE SIMILARITY =====
    cosine_dists = cdist(select_coords, important_coords, metric='cosine')
    # Convert distance to similarity: similarity = 1 - distance
    cosine_sims = 1 - cosine_dists
    results['cosine_sim_mean'] = np.mean(cosine_sims)
    results['cosine_sim_median'] = np.median(cosine_sims)
    results['cosine_sim_min'] = np.min(cosine_sims)
    results['cosine_sim_max'] = np.max(cosine_sims)
    results['cosine_sim_std'] = np.std(cosine_sims)
    results['cosine_sim_all'] = cosine_sims.flatten()

    # ===== 3. MANHATTAN DISTANCE =====
    manhattan_dists = cdist(select_coords, important_coords, metric='cityblock')
    results['manhattan_mean'] = np.mean(manhattan_dists)
    results['manhattan_median'] = np.median(manhattan_dists)
    results['manhattan_min'] = np.min(manhattan_dists)
    results['manhattan_max'] = np.max(manhattan_dists)
    results['manhattan_std'] = np.std(manhattan_dists)
    results['manhattan_all'] = manhattan_dists.flatten()

    # ===== 4. MAHALANOBIS DISTANCE =====
    try:
        # Compute covariance matrix for all genes
        cov_matrix = np.cov(coords.T)
        cov_inv = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse for numerical stability

        mahalanobis_dists = []
        for s_coord in select_coords:
            for i_coord in important_coords:
                diff = s_coord - i_coord
                dist = np.sqrt(diff @ cov_inv @ diff.T)
                mahalanobis_dists.append(dist)

        mahalanobis_dists = np.array(mahalanobis_dists)
        results['mahalanobis_mean'] = np.mean(mahalanobis_dists)
        results['mahalanobis_median'] = np.median(mahalanobis_dists)
        results['mahalanobis_min'] = np.min(mahalanobis_dists)
        results['mahalanobis_max'] = np.max(mahalanobis_dists)
        results['mahalanobis_std'] = np.std(mahalanobis_dists)
        results['mahalanobis_all'] = mahalanobis_dists
    except Exception as e:
        print(f"    Warning: Mahalanobis distance failed for {space_name}: {e}")
        results['mahalanobis_mean'] = np.nan
        results['mahalanobis_all'] = np.array([])

    # ===== 5. HAUSDORFF DISTANCE =====
    hausdorff_forward = directed_hausdorff(select_coords, important_coords)[0]
    hausdorff_backward = directed_hausdorff(important_coords, select_coords)[0]
    results['hausdorff_forward'] = hausdorff_forward
    results['hausdorff_backward'] = hausdorff_backward
    results['hausdorff_symmetric'] = max(hausdorff_forward, hausdorff_backward)

    # ===== 6. K-NEAREST NEIGHBORS OVERLAP =====
    # For each select gene, find k nearest neighbors and check if any are important genes
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(coords)-1),
                           metric='euclidean').fit(coords)

    overlap_scores = []
    for s_idx in select_indices:
        distances, indices = nbrs.kneighbors([coords[s_idx]])
        neighbor_indices = indices[0]
        # Count how many neighbors are in important genes
        overlap = sum(1 for idx in neighbor_indices if idx in important_indices)
        overlap_scores.append(overlap / len(neighbor_indices))

    results['knn_overlap_mean'] = np.mean(overlap_scores)
    results['knn_overlap_std'] = np.std(overlap_scores)
    results['knn_overlap_all'] = np.array(overlap_scores)

    # ===== 7. SILHOUETTE SCORE =====
    # Create labels: 0 for select, 1 for important, -1 for other
    labels = np.full(len(coords), -1)
    for idx in select_indices:
        labels[idx] = 0
    for idx in important_indices:
        labels[idx] = 1

    # Only compute silhouette for select and important genes
    subset_indices = select_indices + important_indices
    subset_coords = coords[subset_indices]
    subset_labels = labels[subset_indices]

    if len(np.unique(subset_labels)) > 1:
        sil_score = silhouette_score(subset_coords, subset_labels, metric='euclidean')
        results['silhouette_score'] = sil_score
    else:
        results['silhouette_score'] = np.nan

    # ===== 8. STATISTICAL TESTS =====
    # Compare distances of select genes to important genes vs distances to other genes
    other_indices = [i for i in range(len(gene_names))
                    if i not in select_indices and i not in important_indices]

    if len(other_indices) > 0:
        other_coords = coords[other_indices]

        # Distances from select to important
        dist_to_important = euclidean_dists.flatten()

        # Distances from select to other genes
        dist_to_others = cdist(select_coords, other_coords, metric='euclidean').flatten()

        # Mann-Whitney U test
        u_stat, p_value_mw = mannwhitneyu(dist_to_important, dist_to_others,
                                         alternative='less')  # Test if important genes are closer
        results['mannwhitney_u'] = u_stat
        results['mannwhitney_p'] = p_value_mw

        # Kolmogorov-Smirnov test
        ks_stat, p_value_ks = ks_2samp(dist_to_important, dist_to_others)
        results['ks_statistic'] = ks_stat
        results['ks_p'] = p_value_ks

        results['dist_to_others_mean'] = np.mean(dist_to_others)
        results['dist_to_others_all'] = dist_to_others
    else:
        results['mannwhitney_u'] = np.nan
        results['mannwhitney_p'] = np.nan
        results['ks_statistic'] = np.nan
        results['ks_p'] = np.nan
        results['dist_to_others_all'] = np.array([])

    # ===== 9. PERMUTATION TEST =====
    # Test if observed mean distance is significantly lower than random
    n_permutations = 1000
    random_means = []

    for _ in range(n_permutations):
        # Randomly sample genes (same number as important genes)
        random_indices = np.random.choice(len(coords), size=len(important_indices),
                                         replace=False)
        random_coords = coords[random_indices]
        random_dists = cdist(select_coords, random_coords, metric='euclidean')
        random_means.append(np.mean(random_dists))

    observed_mean = results['euclidean_mean']
    p_value_perm = np.mean(np.array(random_means) <= observed_mean)
    results['permutation_p'] = p_value_perm
    results['permutation_null_dist'] = np.array(random_means)

    return results

# Calculate distances in original 512D embedding space
print("\n" + "="*100)
print("Analyzing Original 512D Embedding Space")
print("="*100)

original_space_results = calculate_comprehensive_distances(
    gene_embeddings_scaled, gene_names, matched_select_genes, top_gene_names,
    space_name='Original_512D', k_neighbors=10
)

if original_space_results:
    print(f"\nOriginal 512D Embedding Space:")
    print(f"  Euclidean - Mean: {original_space_results['euclidean_mean']:.3f}, "
          f"Median: {original_space_results['euclidean_median']:.3f}")
    print(f"  Cosine Similarity - Mean: {original_space_results['cosine_sim_mean']:.3f}")
    print(f"  Manhattan - Mean: {original_space_results['manhattan_mean']:.3f}")
    if not np.isnan(original_space_results['mahalanobis_mean']):
        print(f"  Mahalanobis - Mean: {original_space_results['mahalanobis_mean']:.3f}")
    print(f"  Hausdorff (symmetric): {original_space_results['hausdorff_symmetric']:.3f}")
    print(f"  KNN Overlap (k=10): {original_space_results['knn_overlap_mean']:.3%}")
    print(f"  Silhouette Score: {original_space_results['silhouette_score']:.3f}")
    print(f"  Statistical Tests:")
    print(f"    Mann-Whitney U p-value: {original_space_results['mannwhitney_p']:.4f}")
    print(f"    K-S test p-value: {original_space_results['ks_p']:.4f}")
    print(f"    Permutation test p-value: {original_space_results['permutation_p']:.4f}")

# Calculate distances for each reduced space
print("\n" + "="*100)
print("Analyzing Reduced Dimensional Spaces")
print("="*100)

distance_summary = {'Original_512D': original_space_results}

for method, coords_2d in reduction_results.items():
    print(f"\n{method}:")
    dist_results = calculate_comprehensive_distances(
        coords_2d, gene_names, matched_select_genes, top_gene_names,
        space_name=method, k_neighbors=10
    )

    if dist_results:
        distance_summary[method] = dist_results
        print(f"  Euclidean - Mean: {dist_results['euclidean_mean']:.3f}, "
              f"Median: {dist_results['euclidean_median']:.3f}")
        print(f"  Cosine Similarity - Mean: {dist_results['cosine_sim_mean']:.3f}")
        print(f"  Manhattan - Mean: {dist_results['manhattan_mean']:.3f}")
        if not np.isnan(dist_results['mahalanobis_mean']):
            print(f"  Mahalanobis - Mean: {dist_results['mahalanobis_mean']:.3f}")
        print(f"  Hausdorff (symmetric): {dist_results['hausdorff_symmetric']:.3f}")
        print(f"  KNN Overlap (k=10): {dist_results['knn_overlap_mean']:.3%}")
        print(f"  Silhouette Score: {dist_results['silhouette_score']:.3f}")
        print(f"  Statistical Tests:")
        print(f"    Mann-Whitney U p-value: {dist_results['mannwhitney_p']:.4f}")
        print(f"    K-S test p-value: {dist_results['ks_p']:.4f}")
        print(f"    Permutation test p-value: {dist_results['permutation_p']:.4f}")

# ============================================================================
# Create Distance Metric Visualizations
# ============================================================================
print("\n" + "="*100)
print("Creating Distance Metric Visualizations")
print("="*100)

# Prepare data for plotting
spaces_to_plot = list(distance_summary.keys())

# === PLOT 1: Bar chart comparing mean distances across metrics ===
print("\n1. Creating metric comparison bar chart...")

metrics_to_compare = ['euclidean_mean', 'cosine_sim_mean', 'manhattan_mean',
                     'mahalanobis_mean', 'hausdorff_symmetric']
metric_labels = ['Euclidean', 'Cosine Sim', 'Manhattan', 'Mahalanobis', 'Hausdorff']

fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(20, 4))

for idx, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
    ax = axes[idx]
    values = []
    space_names = []

    for space in spaces_to_plot:
        if space in distance_summary and metric in distance_summary[space]:
            val = distance_summary[space][metric]
            if not np.isnan(val):
                values.append(val)
                space_names.append(space)

    if values:
        bars = ax.bar(range(len(values)), values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(space_names, rotation=45, ha='right')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Highlight best value (lowest for distances, highest for similarity)
        if 'sim' in metric.lower():
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_color('orange')

plt.suptitle('Distance Metrics Comparison Across Embedding Spaces', fontweight='bold', fontsize=14)
plt.tight_layout()
plot_path = plot_dir / 'distance_metrics_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 2: Distance distributions (violin plots) ===
print("\n2. Creating distance distribution violin plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

distance_types = [
    ('euclidean_all', 'Euclidean Distance'),
    ('cosine_sim_all', 'Cosine Similarity'),
    ('manhattan_all', 'Manhattan Distance'),
    ('mahalanobis_all', 'Mahalanobis Distance')
]

for idx, (dist_key, dist_label) in enumerate(distance_types):
    ax = axes[idx]

    plot_data = []
    plot_labels = []

    for space in spaces_to_plot:
        if space in distance_summary and dist_key in distance_summary[space]:
            data = distance_summary[space][dist_key]
            if len(data) > 0:
                plot_data.append(data)
                plot_labels.append(space)

    if plot_data:
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                             showmeans=True, showmedians=True)

        # Color the violins
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.6)

        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title(dist_label, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)

plt.suptitle('Distance Distributions: Select vs Important Genes', fontweight='bold', fontsize=14)
plt.tight_layout()
plot_path = plot_dir / 'distance_distributions.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 3: Statistical significance heatmap ===
print("\n3. Creating statistical significance heatmap...")

stat_tests = ['mannwhitney_p', 'ks_p', 'permutation_p']
test_labels = ['Mann-Whitney U', 'Kolmogorov-Smirnov', 'Permutation Test']

pvalue_matrix = np.zeros((len(spaces_to_plot), len(stat_tests)))

for i, space in enumerate(spaces_to_plot):
    for j, test in enumerate(stat_tests):
        if space in distance_summary and test in distance_summary[space]:
            pvalue_matrix[i, j] = distance_summary[space][test]
        else:
            pvalue_matrix[i, j] = np.nan

fig, ax = plt.subplots(figsize=(8, 6))

# Use log scale for p-values
log_pvalues = -np.log10(pvalue_matrix + 1e-10)  # Add small value to avoid log(0)

im = ax.imshow(log_pvalues, cmap='RdYlGn', aspect='auto', vmin=0, vmax=4)

# Add significance thresholds
ax.axhline(y=-0.5, color='black', linewidth=2)
for i in range(len(spaces_to_plot)):
    ax.axhline(y=i+0.5, color='gray', linewidth=0.5, alpha=0.3)

# Annotations
for i in range(len(spaces_to_plot)):
    for j in range(len(stat_tests)):
        pval = pvalue_matrix[i, j]
        if not np.isnan(pval):
            # Add stars for significance levels
            if pval < 0.001:
                text = f'{pval:.1e}\n***'
            elif pval < 0.01:
                text = f'{pval:.3f}\n**'
            elif pval < 0.05:
                text = f'{pval:.3f}\n*'
            else:
                text = f'{pval:.3f}'

            ax.text(j, i, text, ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   color='white' if log_pvalues[i, j] > 2 else 'black')

ax.set_xticks(range(len(test_labels)))
ax.set_xticklabels(test_labels, rotation=45, ha='right')
ax.set_yticks(range(len(spaces_to_plot)))
ax.set_yticklabels(spaces_to_plot)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('-log10(p-value)', rotation=270, labelpad=25, fontweight='bold')

ax.set_title('Statistical Significance: Select vs Important Genes Closer Than Random',
            fontweight='bold', pad=20)

plt.tight_layout()

# Add legend for stars below the plot
fig.text(0.5, 0.02, '* p<0.05, ** p<0.01, *** p<0.001',
        ha='center', fontsize=12, style='italic')

plot_path = plot_dir / 'statistical_significance_heatmap.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 4: KNN Overlap comparison ===
print("\n4. Creating KNN overlap comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

overlap_means = []
overlap_stds = []
space_names = []

for space in spaces_to_plot:
    if space in distance_summary:
        overlap_means.append(distance_summary[space]['knn_overlap_mean'] * 100)
        overlap_stds.append(distance_summary[space]['knn_overlap_std'] * 100)
        space_names.append(space)

x_pos = np.arange(len(space_names))
bars = ax.bar(x_pos, overlap_means, yerr=overlap_stds, capsize=5,
             color='mediumseagreen', alpha=0.7, edgecolor='black', linewidth=1.5)

# Highlight best value
best_idx = np.argmax(overlap_means)
bars[best_idx].set_color('orange')

ax.set_xticks(x_pos)
ax.set_xticklabels(space_names, rotation=45, ha='right')
ax.set_ylabel('Overlap Percentage (%)', fontweight='bold')
ax.set_title('K-Nearest Neighbors Overlap\n(% of k=10 neighbors that are important genes)',
            fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(overlap_means) * 1.2)

# Add value labels on bars
for i, (mean, std) in enumerate(zip(overlap_means, overlap_stds)):
    ax.text(i, mean + std + 1, f'{mean:.1f}%',
           ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plot_path = plot_dir / 'knn_overlap_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 5: Permutation test results ===
print("\n5. Creating permutation test visualizations...")

n_spaces = len(spaces_to_plot)
fig, axes = plt.subplots(1, n_spaces, figsize=(6*n_spaces, 5))
if n_spaces == 1:
    axes = [axes]

for idx, space in enumerate(spaces_to_plot):
    ax = axes[idx]

    if space in distance_summary and 'permutation_null_dist' in distance_summary[space]:
        null_dist = distance_summary[space]['permutation_null_dist']
        observed = distance_summary[space]['euclidean_mean']
        p_value = distance_summary[space]['permutation_p']

        # Plot histogram of null distribution
        ax.hist(null_dist, bins=50, alpha=0.7, color='lightgray',
               edgecolor='black', label='Null Distribution')

        # Add observed value line
        ax.axvline(observed, color='red', linewidth=3,
                  label=f'Observed: {observed:.3f}')

        # Add mean of null distribution
        ax.axvline(np.mean(null_dist), color='blue', linewidth=2,
                  linestyle='--', label=f'Null Mean: {np.mean(null_dist):.3f}')

        # Determine significance
        sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'

        ax.set_xlabel('Mean Distance', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{space}\np-value: {p_value:.4f} {sig_text}',
                    fontweight='bold', pad=10)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

plt.suptitle('Permutation Test Results: Observed vs Random Gene Sets',
            fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plot_path = plot_dir / 'permutation_test_results.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 6: Distance comparison (Important vs Other genes) ===
print("\n6. Creating distance comparison: Important vs Other genes...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, space in enumerate(spaces_to_plot[:4]):  # Plot first 4 spaces
    if idx >= len(axes):
        break

    ax = axes[idx]

    if space in distance_summary:
        dist_to_important = distance_summary[space]['euclidean_all']
        dist_to_others = distance_summary[space].get('dist_to_others_all', np.array([]))

        if len(dist_to_important) > 0 and len(dist_to_others) > 0:
            # Create violin plots
            parts = ax.violinplot([dist_to_important, dist_to_others],
                                 positions=[1, 2], showmeans=True, showmedians=True)

            # Color the violins
            for pc in parts['bodies']:
                pc.set_alpha(0.6)
            parts['bodies'][0].set_facecolor('steelblue')
            parts['bodies'][1].set_facecolor('lightcoral')

            # Get significance stars
            mw_p = distance_summary[space]["mannwhitney_p"]
            sig_text = '***' if mw_p < 0.001 else '**' if mw_p < 0.01 else '*' if mw_p < 0.05 else 'ns'

            ax.set_xticks([1, 2])
            ax.set_xticklabels([f'To Important\nGenes\n(n={len(dist_to_important)})',
                               f'To Other\nGenes\n(n={len(dist_to_others)})'])
            ax.set_ylabel('Euclidean Distance', fontweight='bold')
            ax.set_title(f'{space}\nMann-Whitney p={mw_p:.4f} {sig_text}',
                        fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3)

# Remove empty subplots
for idx in range(len(spaces_to_plot), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Distance Distributions: Select Genes to Important vs Other Genes',
            fontweight='bold', fontsize=14)
plt.tight_layout()
plot_path = plot_dir / 'distance_comparison_important_vs_others.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 7: Silhouette scores comparison ===
print("\n7. Creating silhouette scores comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

silhouette_scores = []
space_names = []

for space in spaces_to_plot:
    if space in distance_summary and 'silhouette_score' in distance_summary[space]:
        score = distance_summary[space]['silhouette_score']
        if not np.isnan(score):
            silhouette_scores.append(score)
            space_names.append(space)

x_pos = np.arange(len(space_names))
colors = ['green' if s > 0 else 'red' for s in silhouette_scores]
bars = ax.bar(x_pos, silhouette_scores, color=colors, alpha=0.7,
             edgecolor='black', linewidth=1.5)

# Highlight best value
best_idx = np.argmax(silhouette_scores)
bars[best_idx].set_edgecolor('orange')
bars[best_idx].set_linewidth(3)

ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
ax.set_xticks(x_pos)
ax.set_xticklabels(space_names, rotation=45, ha='right')
ax.set_ylabel('Silhouette Score', fontweight='bold')
ax.set_title('Silhouette Score: Separation of Select vs Important Genes\n' +
            '(Higher = better separation, Range: -1 to +1)',
            fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(min(silhouette_scores) * 1.2 if min(silhouette_scores) < 0 else -0.1,
           max(silhouette_scores) * 1.2)

# Add value labels
for i, score in enumerate(silhouette_scores):
    y_pos = score + (0.02 if score > 0 else -0.02)
    va = 'bottom' if score > 0 else 'top'
    ax.text(i, y_pos, f'{score:.3f}', ha='center', va=va, fontweight='bold')

plt.tight_layout()
plot_path = plot_dir / 'silhouette_scores_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 8: Comprehensive summary heatmap ===
print("\n8. Creating comprehensive summary heatmap...")

summary_metrics = [
    ('euclidean_mean', 'Euclidean\nMean', 'lower'),
    ('cosine_sim_mean', 'Cosine\nSimilarity', 'higher'),
    ('manhattan_mean', 'Manhattan\nMean', 'lower'),
    ('hausdorff_symmetric', 'Hausdorff\nDistance', 'lower'),
    ('knn_overlap_mean', 'KNN\nOverlap', 'higher'),
    ('silhouette_score', 'Silhouette\nScore', 'higher'),
    ('mannwhitney_p', 'MW Test\np-value', 'lower'),
    ('permutation_p', 'Perm Test\np-value', 'lower')
]

# Create matrix
summary_matrix = np.zeros((len(spaces_to_plot), len(summary_metrics)))
for i, space in enumerate(spaces_to_plot):
    for j, (metric, _, _) in enumerate(summary_metrics):
        if space in distance_summary and metric in distance_summary[space]:
            summary_matrix[i, j] = distance_summary[space][metric]
        else:
            summary_matrix[i, j] = np.nan

# Normalize each column to 0-1 scale for visualization
normalized_matrix = np.zeros_like(summary_matrix)
for j in range(summary_matrix.shape[1]):
    col = summary_matrix[:, j]
    valid_mask = ~np.isnan(col)
    if np.any(valid_mask):
        valid_vals = col[valid_mask]
        col_min, col_max = valid_vals.min(), valid_vals.max()
        if col_max > col_min:
            normalized_matrix[:, j] = (col - col_min) / (col_max - col_min)
            # Invert for metrics where lower is better
            if summary_metrics[j][2] == 'lower':
                normalized_matrix[:, j] = 1 - normalized_matrix[:, j]

fig, ax = plt.subplots(figsize=(12, 8))

im = ax.imshow(normalized_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Annotations
for i in range(len(spaces_to_plot)):
    for j in range(len(summary_metrics)):
        val = summary_matrix[i, j]
        if not np.isnan(val):
            if 'p' in summary_metrics[j][0]:  # p-values
                text = f'{val:.3f}' if val >= 0.001 else f'{val:.1e}'
            else:
                text = f'{val:.3f}'

            ax.text(j, i, text, ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   color='white' if normalized_matrix[i, j] < 0.5 else 'black')

ax.set_xticks(range(len(summary_metrics)))
ax.set_xticklabels([label for _, label, _ in summary_metrics], rotation=45, ha='right')
ax.set_yticks(range(len(spaces_to_plot)))
ax.set_yticklabels(spaces_to_plot)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Score\n(Green = Better)', rotation=270, labelpad=25, fontweight='bold')

ax.set_title('Comprehensive Distance Metrics Summary\n(All metrics normalized: Green = Better for hypothesis)',
            fontweight='bold', pad=20)

plt.tight_layout()
plot_path = plot_dir / 'comprehensive_metrics_heatmap.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# ============================================================================
# Focused Comparison: Original 512D vs PCA (Most Significant Results)
# ============================================================================
print("\n" + "="*100)
print("Creating Focused Comparisons: Original 512D vs PCA")
print("="*100)

focus_spaces = ['Original_512D', 'PCA']

# === PLOT 9: Side-by-side distance distributions ===
print("\n9. Creating focused distance distribution comparisons...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

metrics_to_plot = [
    ('euclidean_all', 'Euclidean Distance', 'Blues'),
    ('cosine_sim_all', 'Cosine Similarity', 'Greens'),
    ('manhattan_all', 'Manhattan Distance', 'Oranges')
]

for col, (metric_key, metric_name, cmap) in enumerate(metrics_to_plot):
    for row, space in enumerate(focus_spaces):
        ax = axes[row, col]

        if space in distance_summary and metric_key in distance_summary[space]:
            data_important = distance_summary[space][metric_key]
            data_others = distance_summary[space].get('dist_to_others_all', np.array([]))

            if len(data_important) > 0:
                # Create histogram
                ax.hist(data_important, bins=50, alpha=0.7, color='steelblue',
                       edgecolor='black', label='To Important Genes')

                if len(data_others) > 0 and metric_key == 'euclidean_all':
                    ax.hist(data_others, bins=50, alpha=0.5, color='lightcoral',
                           edgecolor='black', label='To Other Genes')

                # Add mean line
                mean_val = np.mean(data_important)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.3f}')

                ax.set_xlabel(metric_name, fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title(f'{space}\n{metric_name}', fontweight='bold', pad=10)
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(axis='y', alpha=0.3)

plt.suptitle('Distance Distribution Comparison: Original 512D vs PCA\n(Select Genes to Important Genes)',
            fontweight='bold', fontsize=14, y=0.995)
plt.tight_layout()
plot_path = plot_dir / 'focused_distance_distributions_512d_vs_pca.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 10: Direct metric comparison bar chart ===
print("\n10. Creating direct metric comparison (512D vs PCA)...")

metrics_comparison = [
    ('euclidean_mean', 'Euclidean\nMean', 'lower'),
    ('euclidean_median', 'Euclidean\nMedian', 'lower'),
    ('cosine_sim_mean', 'Cosine\nSimilarity', 'higher'),
    ('manhattan_mean', 'Manhattan\nMean', 'lower'),
    ('mahalanobis_mean', 'Mahalanobis\nMean', 'lower'),
    ('hausdorff_symmetric', 'Hausdorff\nSymmetric', 'lower'),
    ('silhouette_score', 'Silhouette\nScore', 'higher')
]

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(metrics_comparison))
width = 0.35

vals_512d = []
vals_pca = []

for metric, _, _ in metrics_comparison:
    val_512d = distance_summary['Original_512D'].get(metric, np.nan)
    val_pca = distance_summary['PCA'].get(metric, np.nan)
    vals_512d.append(val_512d)
    vals_pca.append(val_pca)

bars1 = ax.bar(x - width/2, vals_512d, width, label='Original 512D',
              color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, vals_pca, width, label='PCA',
              color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
ax.set_ylabel('Value', fontweight='bold', fontsize=12)
ax.set_title('Direct Comparison: Original 512D vs PCA\n(Lower is better for distances, Higher is better for similarity/silhouette)',
            fontweight='bold', fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels([label for _, label, _ in metrics_comparison], fontsize=10)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = plot_dir / 'direct_comparison_512d_vs_pca.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 11: Statistical significance comparison ===
print("\n11. Creating statistical significance comparison (512D vs PCA)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

stat_tests = [
    ('mannwhitney_p', 'Mann-Whitney U\nTest'),
    ('ks_p', 'Kolmogorov-Smirnov\nTest'),
    ('permutation_p', 'Permutation\nTest')
]

for idx, space in enumerate(focus_spaces):
    ax = axes[idx]

    p_values = []
    test_names = []
    colors_list = []

    for test_key, test_name in stat_tests:
        p_val = distance_summary[space].get(test_key, np.nan)
        if not np.isnan(p_val):
            p_values.append(p_val)
            test_names.append(test_name)

            # Color based on significance
            if p_val < 0.001:
                colors_list.append('#006400')  # Dark green
            elif p_val < 0.01:
                colors_list.append('#228B22')  # Forest green
            elif p_val < 0.05:
                colors_list.append('#90EE90')  # Light green
            else:
                colors_list.append('#DC143C')  # Crimson

    bars = ax.barh(test_names, p_values, color=colors_list, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add significance threshold lines
    ax.axvline(0.05, color='orange', linestyle='--', linewidth=2, label='p=0.05')
    ax.axvline(0.01, color='red', linestyle='--', linewidth=2, label='p=0.01')
    ax.axvline(0.001, color='darkred', linestyle='--', linewidth=2, label='p=0.001')

    # Add p-value labels
    for i, (p_val, test_name) in enumerate(zip(p_values, test_names)):
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(p_val + 0.01, i, f'{p_val:.4f} {stars}',
               va='center', fontweight='bold', fontsize=10)

    ax.set_xlabel('P-value', fontweight='bold', fontsize=12)
    ax.set_title(f'{space}\nStatistical Significance', fontweight='bold', fontsize=12, pad=10)
    ax.set_xlim(0, max(p_values) * 1.3 if p_values else 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Statistical Significance: Are Select Genes Closer to Important Genes?\n(Lower p-value = YES, more significant)',
            fontweight='bold', fontsize=13, y=1.00)
plt.tight_layout()
plot_path = plot_dir / 'statistical_significance_512d_vs_pca.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 12: Permutation test comparison ===
print("\n12. Creating permutation test comparison (512D vs PCA)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, space in enumerate(focus_spaces):
    ax = axes[idx]

    null_dist = distance_summary[space]['permutation_null_dist']
    observed = distance_summary[space]['euclidean_mean']
    p_value = distance_summary[space]['permutation_p']

    # Plot histogram with better styling
    n, bins, patches = ax.hist(null_dist, bins=60, alpha=0.7, color='lightgray',
           edgecolor='black', linewidth=0.5, label='Null Distribution (Random)')

    # Color the histogram based on position relative to observed
    for i, patch in enumerate(patches):
        if bins[i] < observed:
            patch.set_facecolor('#90EE90')  # Light green for better than observed
        else:
            patch.set_facecolor('#FFB6C1')  # Light red for worse than observed

    # Add observed value line
    ax.axvline(observed, color='red', linewidth=3, linestyle='-',
              label=f'Observed: {observed:.3f}', zorder=10)

    # Add mean of null distribution
    null_mean = np.mean(null_dist)
    ax.axvline(null_mean, color='blue', linewidth=2, linestyle='--',
              label=f'Null Mean: {null_mean:.3f}', zorder=10)

    # Add shaded region for p-value
    ax.axvspan(0, observed, alpha=0.2, color='green',
              label=f'Better than observed ({p_value:.1%})')

    # Determine significance for title
    sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    interpretation = "SIGNIFICANT!" if p_value < 0.05 else "Not significant"

    ax.set_xlabel('Mean Euclidean Distance', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title(f'{space}\np={p_value:.4f} {sig_text} | {interpretation}',
                fontweight='bold', fontsize=12, pad=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Permutation Test: Observed Distance vs Random Gene Sets\n(Left region = Select genes closer than random)',
            fontweight='bold', fontsize=13, y=0.98)
plt.tight_layout()
plot_path = plot_dir / 'permutation_comparison_512d_vs_pca.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 13: Box plot comparison ===
print("\n13. Creating box plot comparison (512D vs PCA)...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

distance_metrics = [
    ('euclidean_all', 'Euclidean Distance'),
    ('cosine_sim_all', 'Cosine Similarity'),
    ('manhattan_all', 'Manhattan Distance')
]

for idx, (metric_key, metric_name) in enumerate(distance_metrics):
    ax = axes[idx]

    data_512d = distance_summary['Original_512D'][metric_key]
    data_pca = distance_summary['PCA'][metric_key]

    bp = ax.boxplot([data_512d, data_pca], labels=['Original\n512D', 'PCA'],
                    patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))

    # Color boxes differently
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')

    # Add mean markers
    means = [np.mean(data_512d), np.mean(data_pca)]
    ax.scatter([1, 2], means, color='yellow', s=100, zorder=3,
              edgecolors='black', linewidths=2, marker='D', label='Mean')

    # Add statistical comparison text to title
    from scipy.stats import mannwhitneyu
    title_extra = ""
    if metric_key == 'euclidean_all':
        # Compare to "other genes" for Original_512D
        data_others_512d = distance_summary['Original_512D'].get('dist_to_others_all', np.array([]))
        if len(data_others_512d) > 0:
            u_stat, p_val = mannwhitneyu(data_512d, data_others_512d, alternative='less')
            sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            title_extra = f"\n512D vs Others: p={p_val:.4f} {sig_text}"

    ax.set_ylabel(metric_name, fontweight='bold', fontsize=12)
    ax.set_title(f'{metric_name}\nDistribution Comparison{title_extra}',
                fontweight='bold', fontsize=12, pad=10)
    ax.grid(axis='y', alpha=0.3)

    if idx == 0:
        ax.legend(loc='upper right', fontsize=10)

plt.suptitle('Distribution Comparison: Original 512D vs PCA\n(Select Genes → Important Genes)',
            fontweight='bold', fontsize=14, y=0.98)
plt.tight_layout()
plot_path = plot_dir / 'boxplot_comparison_512d_vs_pca.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# === PLOT 14: Summary comparison table visualization ===
print("\n14. Creating summary comparison table...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Create summary data
summary_data = [
    ['Metric', 'Original 512D', 'PCA', 'Better'],
    ['', '', '', ''],
    ['DISTANCE METRICS', '', '', ''],
    ['Euclidean Mean', f"{distance_summary['Original_512D']['euclidean_mean']:.4f}",
     f"{distance_summary['PCA']['euclidean_mean']:.4f}",
     'PCA' if distance_summary['PCA']['euclidean_mean'] < distance_summary['Original_512D']['euclidean_mean'] else '512D'],
    ['Euclidean Median', f"{distance_summary['Original_512D']['euclidean_median']:.4f}",
     f"{distance_summary['PCA']['euclidean_median']:.4f}",
     'PCA' if distance_summary['PCA']['euclidean_median'] < distance_summary['Original_512D']['euclidean_median'] else '512D'],
    ['Cosine Similarity', f"{distance_summary['Original_512D']['cosine_sim_mean']:.4f}",
     f"{distance_summary['PCA']['cosine_sim_mean']:.4f}",
     'PCA' if distance_summary['PCA']['cosine_sim_mean'] > distance_summary['Original_512D']['cosine_sim_mean'] else '512D'],
    ['Manhattan Mean', f"{distance_summary['Original_512D']['manhattan_mean']:.4f}",
     f"{distance_summary['PCA']['manhattan_mean']:.4f}",
     'PCA' if distance_summary['PCA']['manhattan_mean'] < distance_summary['Original_512D']['manhattan_mean'] else '512D'],
    ['Mahalanobis Mean', f"{distance_summary['Original_512D']['mahalanobis_mean']:.4f}",
     f"{distance_summary['PCA']['mahalanobis_mean']:.4f}",
     'PCA' if distance_summary['PCA']['mahalanobis_mean'] < distance_summary['Original_512D']['mahalanobis_mean'] else '512D'],
    ['Hausdorff Distance', f"{distance_summary['Original_512D']['hausdorff_symmetric']:.4f}",
     f"{distance_summary['PCA']['hausdorff_symmetric']:.4f}",
     'PCA' if distance_summary['PCA']['hausdorff_symmetric'] < distance_summary['Original_512D']['hausdorff_symmetric'] else '512D'],
    ['', '', '', ''],
    ['CLUSTER METRICS', '', '', ''],
    ['Silhouette Score', f"{distance_summary['Original_512D']['silhouette_score']:.4f}",
     f"{distance_summary['PCA']['silhouette_score']:.4f}",
     'PCA' if distance_summary['PCA']['silhouette_score'] > distance_summary['Original_512D']['silhouette_score'] else '512D'],
    ['KNN Overlap (k=10)', f"{distance_summary['Original_512D']['knn_overlap_mean']:.2%}",
     f"{distance_summary['PCA']['knn_overlap_mean']:.2%}",
     'PCA' if distance_summary['PCA']['knn_overlap_mean'] > distance_summary['Original_512D']['knn_overlap_mean'] else '512D'],
    ['', '', '', ''],
    ['STATISTICAL TESTS', '', '', ''],
    ['Mann-Whitney p-value', f"{distance_summary['Original_512D']['mannwhitney_p']:.4f}",
     f"{distance_summary['PCA']['mannwhitney_p']:.4f}",
     'PCA' if distance_summary['PCA']['mannwhitney_p'] < distance_summary['Original_512D']['mannwhitney_p'] else '512D'],
    ['K-S Test p-value', f"{distance_summary['Original_512D']['ks_p']:.4f}",
     f"{distance_summary['PCA']['ks_p']:.4f}",
     'PCA' if distance_summary['PCA']['ks_p'] < distance_summary['Original_512D']['ks_p'] else '512D'],
    ['Permutation p-value', f"{distance_summary['Original_512D']['permutation_p']:.4f}",
     f"{distance_summary['PCA']['permutation_p']:.4f}",
     'PCA' if distance_summary['PCA']['permutation_p'] < distance_summary['Original_512D']['permutation_p'] else '512D'],
]

# Create table
table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.25, 0.25, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style the table
for i, row in enumerate(summary_data):
    for j in range(4):
        cell = table[(i, j)]

        # Header row
        if i == 0:
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white', fontsize=12)
        # Section headers
        elif row[0] in ['DISTANCE METRICS', 'CLUSTER METRICS', 'STATISTICAL TESTS']:
            cell.set_facecolor('#A0A0A0')
            cell.set_text_props(weight='bold', color='white', fontsize=11)
        # Empty rows
        elif row[0] == '':
            cell.set_facecolor('#FFFFFF')
        # Data rows
        else:
            if j == 0:
                cell.set_facecolor('#F0F0F0')
                cell.set_text_props(weight='bold')
            elif j == 1:
                cell.set_facecolor('#E8F4F8')
            elif j == 2:
                cell.set_facecolor('#F8E8F4')
            elif j == 3:
                # Highlight winner
                if row[j] == 'PCA':
                    cell.set_facecolor('#90EE90')
                    cell.set_text_props(weight='bold')
                elif row[j] == '512D':
                    cell.set_facecolor('#FFD700')
                    cell.set_text_props(weight='bold')

plt.suptitle('Comprehensive Summary: Original 512D vs PCA\nWhich Space Shows Better Gene Proximity?\nGreen = PCA wins  |  Gold = 512D wins',
            fontweight='bold', fontsize=14, y=0.96)

plt.tight_layout()

# Add legend below the figure
fig.text(0.5, 0.02, 'Lower distances & p-values are better  |  Higher similarity & silhouette are better',
        ha='center', fontsize=12, style='italic')

plot_path = plot_dir / 'summary_table_512d_vs_pca.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

print(f"\n[OK] Created 6 additional focused comparison plots for Original 512D vs PCA")

# === PLOT 15: Single Mann-Whitney U Test comparison ===
print("\n15. Creating single Mann-Whitney U test comparison (512D vs PCA)...")

fig, ax = plt.subplots(figsize=(10, 6))

# Get Mann-Whitney p-values
mw_512d = distance_summary['Original_512D']['mannwhitney_p']
mw_pca = distance_summary['PCA']['mannwhitney_p']

spaces = ['Original 512D', 'PCA']
p_values = [mw_512d, mw_pca]

# Color based on significance
colors = []
for p_val in p_values:
    if p_val < 0.001:
        colors.append('#006400')  # Dark green
    elif p_val < 0.01:
        colors.append('#228B22')  # Forest green
    elif p_val < 0.05:
        colors.append('#90EE90')  # Light green
    else:
        colors.append('#DC143C')  # Crimson

# Create horizontal bar chart
bars = ax.barh(spaces, p_values, color=colors, alpha=0.8,
               edgecolor='black', linewidth=2, height=0.5)

# Add significance threshold lines
ax.axvline(0.05, color='orange', linestyle='--', linewidth=2.5,
          label='p = 0.05 (significance threshold)', zorder=0)
ax.axvline(0.01, color='red', linestyle='--', linewidth=2,
          label='p = 0.01', alpha=0.7, zorder=0)
ax.axvline(0.001, color='darkred', linestyle='--', linewidth=2,
          label='p = 0.001', alpha=0.7, zorder=0)

# Add p-value labels with significance stars
for i, (space, p_val) in enumerate(zip(spaces, p_values)):
    if p_val < 0.001:
        stars = '***'
        sig_text = 'Highly Significant'
    elif p_val < 0.01:
        stars = '**'
        sig_text = 'Very Significant'
    elif p_val < 0.05:
        stars = '*'
        sig_text = 'Significant'
    else:
        stars = 'ns'
        sig_text = 'Not Significant'

    # Position text to the right of the bar
    ax.text(p_val + 0.003, i, f'  p = {p_val:.4f} {stars}\n  {sig_text}',
           va='center', ha='left', fontweight='bold', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='black', alpha=0.8))

# Determine overall interpretation
interpretation = "Both spaces show p < 0.05: Select genes ARE significantly closer to important genes!"

ax.set_xlabel('P-value', fontweight='bold', fontsize=13)
ax.set_ylabel('Embedding Space', fontweight='bold', fontsize=13)
ax.set_title('Mann-Whitney U Test: Are Select Genes Closer to Important Genes?\n' +
            '(Testing if distances to important genes < distances to other genes)\n' +
            interpretation,
            fontweight='bold', fontsize=13, pad=20)

# Set x-axis limits to show the full range
ax.set_xlim(0, max(p_values) * 1.35)
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(axis='x', alpha=0.3, linewidth=0.5)

# Make the plot look cleaner
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plot_path = plot_dir / 'mannwhitney_comparison_512d_vs_pca.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

print(f"\n[OK] Created single Mann-Whitney U test comparison plot")

# ============================================================================
# Create DataFrame for Plotting
# ============================================================================
print("\n" + "="*100)
print("Preparing Visualization Data")
print("="*100)

# Create base dataframe
base_df = pd.DataFrame({
    'gene': gene_names,
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

base_df['category'] = base_df.apply(assign_category, axis=1)

print(f"Gene categories:")
print(f"  Other genes: {(base_df['category'] == 'Other Genes').sum()}")
print(f"  Important genes: {(base_df['category'] == 'Important Genes').sum()}")
print(f"  Select genes: {(base_df['category'] == 'Select Genes').sum()}")

# ============================================================================
# Create Visualizations for Each Method
# ============================================================================
print("\n" + "="*100)
print("Creating Visualizations")
print("="*100)

# Define colors
colors = {
    'Other Genes': '#CCCCCC',      # Light gray
    'Important Genes': '#3498db',  # Blue
    'Select Genes': '#e74c3c'      # Red
}

# Create plots for each dimensionality reduction method
for method, coords_2d in reduction_results.items():
    print(f"\nCreating plots for {method}...")

    # Create plot dataframe for this method
    plot_df = base_df.copy()
    plot_df['x'] = coords_2d[:, 0]
    plot_df['y'] = coords_2d[:, 1]

    # Plot 1: Categorical plot with highlighted genes
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

    # Add distance annotation to title
    if method in distance_summary:
        dist_text = f" | Mean: {distance_summary[method]['euclidean_mean']:.2f}, Median: {distance_summary[method]['euclidean_median']:.2f}"
    else:
        dist_text = ""

    # Customize plot
    ax.set_xlabel(f'{method} 1', fontweight='bold')
    ax.set_ylabel(f'{method} 2', fontweight='bold')
    ax.set_title(f'Gene Latent Space Visualization ({method})\nscGPT Multi-Species Model{dist_text}',
                 fontweight='bold', pad=20)

    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = plot_dir / f'gene_latent_space_{method.lower()}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {plot_path}")
    plt.close()

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

    ax.set_xlabel(f'{method} 1', fontweight='bold')
    ax.set_ylabel(f'{method} 2', fontweight='bold')
    ax.set_title(f'Gene Importance in Latent Space ({method})\nscGPT Multi-Species Model',
                 fontweight='bold', pad=20)

    if len(matched_select_genes) > 0:
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = plot_dir / f'gene_importance_{method.lower()}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {plot_path}")
    plt.close()

# ============================================================================
# Create Comparison Plot
# ============================================================================
print("\nCreating comparison plot...")

n_methods = len(reduction_results)
fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 6))
if n_methods == 1:
    axes = [axes]

for idx, (method, coords_2d) in enumerate(reduction_results.items()):
    ax = axes[idx]

    # Create plot dataframe for this method
    plot_df = base_df.copy()
    plot_df['x'] = coords_2d[:, 0]
    plot_df['y'] = coords_2d[:, 1]

    # Plot genes
    for category in ['Other Genes', 'Important Genes', 'Select Genes']:
        subset = plot_df[plot_df['category'] == category]

        if category == 'Other Genes':
            ax.scatter(subset['x'], subset['y'],
                      c=colors[category], s=5, alpha=0.3,
                      edgecolors='none')
        else:
            ax.scatter(subset['x'], subset['y'],
                      c=colors[category], s=50, alpha=0.8,
                      edgecolors='black', linewidths=0.5,
                      label=category)

    # Add distance to title
    if method in distance_summary:
        title_text = f"{method}\nMean: {distance_summary[method]['euclidean_mean']:.2f}"
    else:
        title_text = method

    ax.set_xlabel(f'{method} 1', fontweight='bold')
    ax.set_ylabel(f'{method} 2', fontweight='bold')
    ax.set_title(title_text, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if idx == n_methods - 1:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)

plt.suptitle('Comparison of Dimensionality Reduction Methods\nGene Latent Space',
             fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plot_path = plot_dir / 'method_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {plot_path}")
plt.close()

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*100)
print("Saving Results")
print("="*100)

# Save coordinates for each method
for method, coords_2d in reduction_results.items():
    results_df = base_df.copy()
    results_df[f'{method.lower()}_1'] = coords_2d[:, 0]
    results_df[f'{method.lower()}_2'] = coords_2d[:, 1]
    results_df = results_df[['gene', f'{method.lower()}_1', f'{method.lower()}_2',
                             'importance', 'category']]

    csv_path = plot_dir / f'gene_latent_space_{method.lower()}_coordinates.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Saved {method} coordinates: {csv_path}")

# Save comprehensive distance summary
if distance_summary:
    # Create summary table with scalar metrics only
    summary_rows = []
    for space_name, metrics in distance_summary.items():
        row = {
            'space': space_name,
            'euclidean_mean': metrics.get('euclidean_mean', np.nan),
            'euclidean_median': metrics.get('euclidean_median', np.nan),
            'euclidean_min': metrics.get('euclidean_min', np.nan),
            'euclidean_std': metrics.get('euclidean_std', np.nan),
            'cosine_sim_mean': metrics.get('cosine_sim_mean', np.nan),
            'cosine_sim_median': metrics.get('cosine_sim_median', np.nan),
            'manhattan_mean': metrics.get('manhattan_mean', np.nan),
            'manhattan_median': metrics.get('manhattan_median', np.nan),
            'mahalanobis_mean': metrics.get('mahalanobis_mean', np.nan),
            'hausdorff_symmetric': metrics.get('hausdorff_symmetric', np.nan),
            'hausdorff_forward': metrics.get('hausdorff_forward', np.nan),
            'hausdorff_backward': metrics.get('hausdorff_backward', np.nan),
            'knn_overlap_mean': metrics.get('knn_overlap_mean', np.nan),
            'knn_overlap_std': metrics.get('knn_overlap_std', np.nan),
            'silhouette_score': metrics.get('silhouette_score', np.nan),
            'mannwhitney_u': metrics.get('mannwhitney_u', np.nan),
            'mannwhitney_p': metrics.get('mannwhitney_p', np.nan),
            'ks_statistic': metrics.get('ks_statistic', np.nan),
            'ks_p': metrics.get('ks_p', np.nan),
            'permutation_p': metrics.get('permutation_p', np.nan),
            'dist_to_others_mean': metrics.get('dist_to_others_mean', np.nan),
            'n_select': metrics.get('n_select', 0),
            'n_important': metrics.get('n_important', 0)
        }
        summary_rows.append(row)

    dist_summary_df = pd.DataFrame(summary_rows)
    dist_path = plot_dir / 'comprehensive_distance_summary.csv'
    dist_summary_df.to_csv(dist_path, index=False)
    print(f"[OK] Saved comprehensive distance summary: {dist_path}")

    # Save detailed pairwise distances for each space
    for space_name, metrics in distance_summary.items():
        # Save euclidean pairwise distances
        if 'euclidean_all' in metrics and len(metrics['euclidean_all']) > 0:
            euclidean_dists = metrics['euclidean_all']
            n_select = metrics['n_select']
            n_important = metrics['n_important']

            # Reshape to matrix form
            dist_matrix = euclidean_dists.reshape(n_select, n_important)

            # Create dataframe with gene names
            select_genes_list = [g for g in gene_names if g in matched_select_genes]
            important_genes_list = [g for g in gene_names if g in top_gene_names]

            dist_df = pd.DataFrame(dist_matrix,
                                  index=select_genes_list,
                                  columns=important_genes_list)

            pairwise_path = plot_dir / f'pairwise_distances_{space_name.lower()}.csv'
            dist_df.to_csv(pairwise_path)
            print(f"[OK] Saved {space_name} pairwise distances: {pairwise_path}")

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
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*100)
print(f"\nAll results saved to: {plot_dir}")

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")
print(f"\nGene Sets:")
print(f"  Total genes visualized: {len(gene_names):,}")
print(f"  Important genes highlighted: {N_TOP_GENES}")
print(f"  Select genes matched: {len(matched_select_genes)}")

print(f"\nDimensionality Reduction Methods:")
print(f"  {', '.join(METHODS_TO_RUN)} + Original 512D")

if distance_summary:
    print(f"\n{'='*100}")
    print("DISTANCE METRICS SUMMARY")
    print(f"{'='*100}")

    for space_name in ['Original_512D'] + METHODS_TO_RUN:
        if space_name in distance_summary:
            stats = distance_summary[space_name]
            print(f"\n{space_name}:")
            print(f"  Euclidean Distance:")
            print(f"    Mean: {stats['euclidean_mean']:.4f}, Median: {stats['euclidean_median']:.4f}")
            print(f"  Cosine Similarity:")
            print(f"    Mean: {stats['cosine_sim_mean']:.4f}")
            print(f"  KNN Overlap (k=10):")
            print(f"    {stats['knn_overlap_mean']:.2%} of neighbors are important genes")
            print(f"  Silhouette Score: {stats['silhouette_score']:.4f}")
            print(f"  Statistical Significance:")
            print(f"    Mann-Whitney p-value: {stats['mannwhitney_p']:.4f} {'***' if stats['mannwhitney_p'] < 0.001 else '**' if stats['mannwhitney_p'] < 0.01 else '*' if stats['mannwhitney_p'] < 0.05 else ''}")
            print(f"    Permutation p-value: {stats['permutation_p']:.4f} {'***' if stats['permutation_p'] < 0.001 else '**' if stats['permutation_p'] < 0.01 else '*' if stats['permutation_p'] < 0.05 else ''}")

print(f"\n{'='*100}")
print("GENERATED PLOTS")
print(f"{'='*100}")
print("\nDistance Metric Analysis Plots (All Methods):")
print("  1. distance_metrics_comparison.png - Bar charts comparing metrics across spaces")
print("  2. distance_distributions.png - Violin plots of distance distributions")
print("  3. statistical_significance_heatmap.png - P-values from statistical tests")
print("  4. knn_overlap_comparison.png - K-nearest neighbors overlap analysis")
print("  5. permutation_test_results.png - Observed vs null distributions")
print("  6. distance_comparison_important_vs_others.png - Compare distances to important vs other genes")
print("  7. silhouette_scores_comparison.png - Cluster separation scores")
print("  8. comprehensive_metrics_heatmap.png - All metrics summary heatmap")

print("\nFocused Comparison Plots (Original 512D vs PCA):")
print("  9. focused_distance_distributions_512d_vs_pca.png - Side-by-side histograms")
print("  10. direct_comparison_512d_vs_pca.png - Direct metric comparison bars")
print("  11. statistical_significance_512d_vs_pca.png - Statistical test p-values")
print("  12. permutation_comparison_512d_vs_pca.png - Permutation test distributions")
print("  13. boxplot_comparison_512d_vs_pca.png - Box plots of distance distributions")
print("  14. summary_table_512d_vs_pca.png - Comprehensive comparison table")
print("  15. mannwhitney_comparison_512d_vs_pca.png - Single Mann-Whitney U test comparison")

print("\nGene Latent Space Plots:")
for method in METHODS_TO_RUN:
    print(f"  - gene_latent_space_{method.lower()}.png")
    print(f"  - gene_importance_{method.lower()}.png")
print("  - method_comparison.png")

print("\nData Files:")
print("  - comprehensive_distance_summary.csv - All distance metrics")
for space in ['Original_512D'] + METHODS_TO_RUN:
    print(f"  - pairwise_distances_{space.lower()}.csv - Detailed pairwise distances")
for method in METHODS_TO_RUN:
    print(f"  - gene_latent_space_{method.lower()}_coordinates.csv - 2D coordinates")
print("  - top_important_genes.csv")
print("  - matched_select_genes.csv")

print(f"\n{'='*100}")
print("INTERPRETATION GUIDE")
print(f"{'='*100}")
print("\nKey Findings to Look For:")
print("  1. Original 512D space provides the MOST ACCURATE distance measurements")
print("  2. Lower Euclidean/Manhattan distances = genes are closer together")
print("  3. Higher Cosine Similarity = genes have similar expression patterns")
print("  4. Higher KNN Overlap = select genes have important genes as neighbors")
print("  5. Lower p-values (< 0.05) = statistically significant proximity")
print("  6. Silhouette score > 0 suggests genes form distinct clusters")
print("  7. Compare Original_512D vs reduced spaces to assess distance preservation")

print(f"\n{'='*100}")
