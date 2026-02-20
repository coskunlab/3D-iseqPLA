#!/usr/bin/env python3
"""
Plot scGPT Multi-Species Model Performance
Evaluates the trained model on train/val/test splits and creates bar plots
showing classification metrics (AUC, F1, Accuracy, Precision, Recall)
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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Import scGPT components
try:
    from scgpt.model import TransformerModel
    from scgpt.utils import set_seed
    print("[OK] Successfully imported scGPT components")
except ImportError as e:
    print(f"✗ Error importing scGPT: {e}")
    sys.exit(1)

# Setup paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'
FIGURES_ROOT = PROJECT_ROOT / 'figures'

# Configuration
SEED = 42
BATCH_SIZE = 8
MAX_SEQ_LEN = 1200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_dir = DATA_ROOT / '71 CF AI Foundation model' / 'Models' / 'scGPT' / 'multispecies'
plot_dir = FIGURES_ROOT / '37_plot_in_vitro_model_performance'
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

print("="*100)
print("scGPT Multi-Species Model Performance Evaluation")
print("="*100)
print(f"Device: {DEVICE}")
print(f"Model directory: {model_dir}")
print(f"Plot directory: {plot_dir}")
print()

set_seed(SEED)

# ============================================================================
# Load Data and Create Splits
# ============================================================================
print("="*100)
print("Loading Data and Creating Splits")
print("="*100)

from sklearn.model_selection import train_test_split

# Load original data
data_file = DATA_ROOT / '71 CF AI Foundation model' / 'Data' / '00 In Vitro RAW' / 'converted_anndata' / 'scGPT_multispecies_training_corpus.h5ad'

if not data_file.exists():
    print(f"Error: Data file not found: {data_file}")
    sys.exit(1)

print(f"Loading data from: {data_file.name}")
adata = sc.read_h5ad(data_file)
print(f"  Cells: {adata.n_obs:,}")
print(f"  Genes: {adata.n_vars:,}")

# Filter genes with zero expression
gene_nonzero = (adata.X != 0).sum(axis=0)
if hasattr(gene_nonzero, 'A1'):
    gene_nonzero = gene_nonzero.A1
genes_to_keep = gene_nonzero > 0
print(f"Filtering genes: keeping {genes_to_keep.sum():,} / {adata.n_vars:,}")
adata = adata[:, genes_to_keep].copy()

# Log-normalize if needed
if adata.X.max() > 100:
    print("Log-normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# Split data (same as training script)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

labels = adata.obs['label'].values
if 'species' in adata.obs.columns:
    strat_key = adata.obs['species'].astype(str) + "_" + adata.obs['label'].astype(str)
else:
    strat_key = labels

print("\nCreating train/val/test splits...")
# First split: train vs (val + test)
train_idx, temp_idx = train_test_split(
    np.arange(len(labels)),
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=SEED,
    stratify=strat_key
)

# Second split: val vs test
val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=(1 - val_size),
    random_state=SEED,
    stratify=strat_key[temp_idx]
)

# Create splits
adata_train = adata[train_idx].copy()
adata_val = adata[val_idx].copy()
adata_test = adata[test_idx].copy()

print(f"[OK] Train: {adata_train.n_obs:,} cells")
print(f"[OK] Val:   {adata_val.n_obs:,} cells")
print(f"[OK] Test:  {adata_test.n_obs:,} cells")

# ============================================================================
# Reconstruct Vocabulary
# ============================================================================
print("\n" + "="*100)
print("Reconstructing Vocabulary")
print("="*100)

gene_names = adata_train.var_names.tolist()
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

# Use the most recent checkpoint
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
print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
print(f"  Validation accuracy: {checkpoint['val_accuracy']:.4f}")

# ============================================================================
# Data Preparation Function
# ============================================================================
def prepare_batch(adata_batch, gene_ids, max_len=MAX_SEQ_LEN):
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
        if len(nonzero_idx) > max_len:
            # Sample most highly expressed genes
            top_idx = np.argsort(expr[nonzero_idx])[-max_len:]
            nonzero_idx = nonzero_idx[top_idx]

        # Get gene IDs and values
        genes = nonzero_idx
        vals = expr[nonzero_idx]

        # Create padding mask
        mask = np.zeros(max_len, dtype=bool)

        # Pad if needed
        if len(genes) < max_len:
            pad_len = max_len - len(genes)
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

# ============================================================================
# Evaluation Function
# ============================================================================
def evaluate_split(model, adata, split_name):
    """Evaluate model on a data split and return metrics"""
    print(f"\nEvaluating {split_name} split...")

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    n_batches = len(adata) // BATCH_SIZE

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(adata))

            # Get batch
            adata_batch = adata[start_idx:end_idx]
            labels = adata_batch.obs['label'].values

            # Prepare batch
            gene_ids_batch, values_batch, padding_mask = prepare_batch(
                adata_batch, gene_names
            )

            try:
                # Forward pass
                output = model(gene_ids_batch, values_batch,
                             src_key_padding_mask=padding_mask)

                # Get CLS token representation
                if isinstance(output, dict):
                    cls_output = output.get('cls_output', output.get('cell_emb'))
                else:
                    cls_output = output[:, 0, :]

                # Get classification logits
                logits = model.classifier(cls_output)

                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
                all_labels.extend(labels)

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUC:       {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"  Confusion Matrix:")
    print(f"    {cm}")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'confusion_matrix': cm
    }

# ============================================================================
# Evaluate All Splits
# ============================================================================
print("\n" + "="*100)
print("Evaluating Model Performance")
print("="*100)

results = {}
results['Train'] = evaluate_split(model, adata_train, "Train")
results['Validation'] = evaluate_split(model, adata_val, "Validation")
results['Test'] = evaluate_split(model, adata_test, "Test")

# ============================================================================
# Create Performance Bar Plots
# ============================================================================
print("\n" + "="*100)
print("Creating Performance Plots")
print("="*100)

# Set style
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Prepare data for plotting
metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC']
splits = ['Train', 'Validation', 'Test']
colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Create DataFrame for easier plotting
df_data = []
for split in splits:
    for metric, metric_name in zip(metrics, metric_names):
        df_data.append({
            'Split': split,
            'Metric': metric_name,
            'Value': results[split][metric]
        })

df = pd.DataFrame(df_data)

# Plot 1: Grouped bar chart - All metrics
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metric_names))
width = 0.25

for i, split in enumerate(splits):
    values = [results[split][metric] for metric in metrics]
    offset = width * (i - 1)
    bars = ax.bar(x + offset, values, width, label=split, color=colors[i], alpha=0.8)

ax.set_xlabel('Metrics', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('scGPT Multi-Species Model Performance\nAcross Train/Validation/Test Splits',
             fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.legend(title='Data Split', loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = plot_dir / 'model_performance_all_metrics.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {plot_path}")
plt.close()

# Plot 2: Individual metric comparisons (separate figures)
for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    fig, ax = plt.subplots(figsize=(8, 6))
    values = [results[split][metric] for split in splits]

    bars = ax.bar(splits, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'{metric_name} - scGPT Multi-Species Model', fontweight='bold', pad=15)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plot_path = plot_dir / f'metric_{metric}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {plot_path}")
    plt.close()

# Plot 3: Confusion Matrices (separate figures)
for idx, split in enumerate(splits):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = results[split]['confusion_matrix']

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Confusion Matrix: {split} Split\nscGPT Multi-Species Classification',
                 fontweight='bold', pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Count', rotation=270, labelpad=20)

    # Add labels
    classes = ['Control', 'Diseased']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add text annotations with smaller font
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2%})',
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black",
                   fontweight='bold', fontsize=10)

    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')

    plt.tight_layout()
    plot_path = plot_dir / f'confusion_matrix_{split.lower()}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {plot_path}")
    plt.close()

# Plot 4: Heatmap of all metrics
fig, ax = plt.subplots(figsize=(8, 4))

# Create matrix for heatmap
heatmap_data = np.array([[results[split][metric] for metric in metrics]
                         for split in splits])

im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(metric_names)))
ax.set_yticks(np.arange(len(splits)))
ax.set_xticklabels(metric_names)
ax.set_yticklabels(splits)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations
for i in range(len(splits)):
    for j in range(len(metrics)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                      ha="center", va="center", color="black", fontweight='bold')

ax.set_title('Performance Heatmap: scGPT Multi-Species Model',
             fontweight='bold', pad=20)
plt.tight_layout()
plot_path = plot_dir / 'performance_heatmap.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {plot_path}")
plt.close()

# ============================================================================
# Save Results to CSV
# ============================================================================
print("\n" + "="*100)
print("Saving Results")
print("="*100)

# Save summary metrics
summary_df = pd.DataFrame({
    'Split': splits,
    'Accuracy': [results[split]['accuracy'] for split in splits],
    'F1_Score': [results[split]['f1'] for split in splits],
    'Precision': [results[split]['precision'] for split in splits],
    'Recall': [results[split]['recall'] for split in splits],
    'AUC_ROC': [results[split]['auc'] for split in splits]
})

csv_path = plot_dir / 'model_performance_summary.csv'
summary_df.to_csv(csv_path, index=False)
print(f"[OK] Saved summary: {csv_path}")

# Save detailed classification reports
for split in splits:
    report = classification_report(
        results[split]['labels'],
        results[split]['predictions'],
        target_names=['Control', 'Diseased'],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = plot_dir / f'classification_report_{split.lower()}.csv'
    report_df.to_csv(report_path)
    print(f"[OK] Saved {split} report: {report_path}")

print("\n" + "="*100)
print("EVALUATION COMPLETE")
print("="*100)
print(f"\nAll plots saved to: {plot_dir}")
print("\nSummary:")
for split in splits:
    print(f"\n{split} Split:")
    print(f"  Accuracy:  {results[split]['accuracy']:.4f}")
    print(f"  F1 Score:  {results[split]['f1']:.4f}")
    print(f"  AUC-ROC:   {results[split]['auc']:.4f}")
