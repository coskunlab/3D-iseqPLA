#!/usr/bin/env python3
"""
Fine-tune scGPT for Binary Classification - Multi-Species Approach
Task: Classify cells as control vs diseased/treated
Data: 31,915 cells (mouse + human) with unified gene vocabulary

Training strategy:
- Mixed batches containing both mouse and human samples
- Species metadata for model to learn species-specific patterns
- Binary classification: 0=control, 1=diseased
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# Import scGPT components
try:
    from scgpt.model import TransformerModel
    from scgpt.utils import set_seed
    from scgpt.tokenizer import tokenize_and_pad_batch
    from scgpt.preprocess import Preprocessor
    print("✓ Successfully imported scGPT components")
except ImportError as e:
    print(f"✗ Error importing scGPT: {e}")
    print("Please ensure scGPT is installed: pip install scgpt")
    sys.exit(1)

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Training will proceed without logging.")
    WANDB_AVAILABLE = False

# Configuration
WANDB_API_KEY = "261a9172a7233d8b283ce5e9ec99ea601a59bbd3"
WANDB_PROJECT = "scGPT-multispecies-classification"
WANDB_ENTITY = None

# Paths
data_file = Path('/coskun-lab/Nicky/71 CF AI Foundation model/Data/00 In Vitro RAW/converted_anndata/scGPT_multispecies_training_corpus_v2.h5ad')
save_dir = Path('/coskun-lab/Nicky/71 CF AI Foundation model/Models/scGPT/multispecies')
save_dir.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
SEED = 42
BATCH_SIZE = 8  # Reduced for GPU memory
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32
MAX_SEQ_LEN = 1200  # Reduced from 3000 for GPU memory
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRADIENT_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model configuration
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
MASK_VALUE = -1
PAD_VALUE = -2
N_LAYERS = 12  # Pretrained model layers
N_HEADS = 8
EMBSIZE = 512
D_HID = 512
DROPOUT = 0.2

# Pretrained model configuration
USE_PRETRAINED = True
PRETRAINED_MODEL_PATH = save_dir / "pretrained" / "scGPT_human"

print("="*100)
print("scGPT Multi-Species Fine-Tuning for Binary Classification")
print("="*100)
print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Data file: {data_file.name}")
print()

# Set random seed
set_seed(SEED)

# ============================================================================
# Load Multi-Species Data
# ============================================================================
print("="*100)
print("Loading Multi-Species Training Corpus")
print("="*100)

if not data_file.exists():
    print(f"Error: Data file not found: {data_file}")
    sys.exit(1)

print(f"Loading from: {data_file}")
adata = sc.read_h5ad(data_file)

print(f"\nDataset loaded:")
print(f"  Cells: {adata.n_obs:,}")
print(f"  Genes: {adata.n_vars:,}")
print(f"  Control: {(adata.obs['label'] == 0).sum():,}")
print(f"  Diseased: {(adata.obs['label'] == 1).sum():,}")

# Check species distribution
if 'species' in adata.obs.columns:
    print(f"\nSpecies distribution:")
    for species in adata.obs['species'].unique():
        species_mask = adata.obs['species'] == species
        n_total = species_mask.sum()
        n_control = ((adata.obs['species'] == species) & (adata.obs['label'] == 0)).sum()
        n_diseased = ((adata.obs['species'] == species) & (adata.obs['label'] == 1)).sum()
        print(f"  {species.capitalize()}: {n_total:,} cells ({n_control:,} control, {n_diseased:,} diseased)")

# ============================================================================
# Preprocess Data
# ============================================================================
print("\n" + "="*100)
print("Preprocessing Data")
print("="*100)

# Filter genes with zero expression across all cells
gene_nonzero = (adata.X != 0).sum(axis=0)
if hasattr(gene_nonzero, 'A1'):
    gene_nonzero = gene_nonzero.A1
genes_to_keep = gene_nonzero > 0
print(f"Filtering genes: keeping {genes_to_keep.sum():,} / {adata.n_vars:,} with non-zero expression")
adata = adata[:, genes_to_keep].copy()

# Log-normalize if not already done
if adata.X.max() > 100:
    print("Log-normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

print(f"\nPreprocessed shape: {adata.shape}")

# ============================================================================
# Split Data
# ============================================================================
print("\n" + "="*100)
print("Splitting Data (Train/Val/Test)")
print("="*100)

# Stratified split by label and species
labels = adata.obs['label'].values
if 'species' in adata.obs.columns:
    # Create stratification key combining species and label
    strat_key = adata.obs['species'].astype(str) + "_" + adata.obs['label'].astype(str)
else:
    strat_key = labels

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

print(f"\nSplit sizes:")
print(f"  Train: {len(train_idx):,} ({len(train_idx)/len(labels)*100:.1f}%)")
print(f"    Control: {(adata_train.obs['label'] == 0).sum():,}")
print(f"    Diseased: {(adata_train.obs['label'] == 1).sum():,}")
print(f"  Val:   {len(val_idx):,} ({len(val_idx)/len(labels)*100:.1f}%)")
print(f"    Control: {(adata_val.obs['label'] == 0).sum():,}")
print(f"    Diseased: {(adata_val.obs['label'] == 1).sum():,}")
print(f"  Test:  {len(test_idx):,} ({len(test_idx)/len(labels)*100:.1f}%)")
print(f"    Control: {(adata_test.obs['label'] == 0).sum():,}")
print(f"    Diseased: {(adata_test.obs['label'] == 1).sum():,}")

# ============================================================================
# Initialize Weights & Biases
# ============================================================================
if WANDB_AVAILABLE:
    print("\n" + "="*100)
    print("Initializing Weights & Biases")
    print("="*100)

    os.environ['WANDB_API_KEY'] = WANDB_API_KEY

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"multispecies_scGPT_{EPOCHS}epochs",
        config={
            "architecture": "scGPT",
            "dataset": "Multi-species (mouse + human)",
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_seq_len": MAX_SEQ_LEN,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "embsize": EMBSIZE,
            "d_hid": D_HID,
            "dropout": DROPOUT,
            "seed": SEED
        }
    )
    print("✓ W&B initialized")

# ============================================================================
# Create Gene Vocabulary
# ============================================================================
print("\n" + "="*100)
print("Creating Gene Vocabulary")
print("="*100)

gene_names = adata.var_names.tolist()
vocab = {gene: idx for idx, gene in enumerate(gene_names)}
vocab[PAD_TOKEN] = len(vocab)
vocab[CLS_TOKEN] = len(vocab)

VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE:,}")
print(f"  Genes: {len(gene_names):,}")
print(f"  Special tokens: {CLS_TOKEN}, {PAD_TOKEN}")

# ============================================================================
# Download and Load Pretrained Model
# ============================================================================
def download_pretrained_scgpt():
    """Download pretrained scGPT model from HuggingFace"""
    pretrained_dir = PRETRAINED_MODEL_PATH.parent
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    model_file = PRETRAINED_MODEL_PATH / "best_model.pt"

    if model_file.exists():
        print(f"✓ Pretrained model already exists at {model_file}")
        return True

    print(f"Downloading pretrained scGPT model to {PRETRAINED_MODEL_PATH}...")
    PRETRAINED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace using huggingface_hub
    try:
        from huggingface_hub import hf_hub_download

        print("  Downloading model file from HuggingFace...")
        # Try multiple potential repositories
        repo_options = [
            ("bo wang-lab/scGPT", "scGPT_human"),
            ("tdc/scGPT", "whole_human"),
            ("agemagician/scgpt", "model")
        ]

        downloaded_file = None
        for repo_id, filename_base in repo_options:
            try:
                print(f"    Trying {repo_id}...")
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{filename_base}.pt" if not filename_base.endswith('.pt') else filename_base,
                    cache_dir=str(pretrained_dir),
                    local_dir=str(PRETRAINED_MODEL_PATH),
                    local_dir_use_symlinks=False
                )
                print(f"    ✓ Successfully downloaded from {repo_id}")
                break
            except Exception as e:
                print(f"    Failed: {str(e)[:100]}")
                continue

        if not downloaded_file:
            raise Exception("Could not download from any repository")
        print(f"✓ Pretrained model downloaded successfully to {downloaded_file}")
        return True
    except ImportError:
        print("  huggingface_hub not installed, trying direct download...")
        return False
    except Exception as e:
        print(f"  All HuggingFace attempts failed. Trying direct download from official source...")
        # Fallback to direct download from official scGPT repository
        try:
            import subprocess
            # Official scGPT pretrained model URL (from their documentation)
            urls = [
                "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/scGPT/scGPT_human/model.pt",
                "https://huggingface.co/tdc/scGPT/resolve/main/whole_human.pt"
            ]

            for url in urls:
                print(f"  Trying direct download from: {url}")
                result = subprocess.run(
                    ["wget", "-O", str(model_file), url, "-q", "--show-progress"],
                    timeout=1200
                )
                if result.returncode == 0 and model_file.exists():
                    print(f"✓ Pretrained model downloaded successfully")
                    return True

            print(f"Warning: All download attempts failed")
            return False
        except Exception as e2:
            print(f"Warning: Could not download pretrained model: {e2}")
            return False

# ============================================================================
# Initialize Model
# ============================================================================
print("\n" + "="*100)
print("Initializing Model")
print("="*100)

# Download pretrained model if needed
if USE_PRETRAINED:
    downloaded = download_pretrained_scgpt()
    if not downloaded:
        print("Warning: Proceeding with model from scratch")
        USE_PRETRAINED = False

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

# Load pretrained weights if available
if USE_PRETRAINED:
    model_file = PRETRAINED_MODEL_PATH / "best_model.pt"
    if model_file.exists():
        try:
            print(f"Loading pretrained weights from {model_file}...")
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

# Freeze early layers for fine-tuning
# Only train the last few transformer layers + classifier
if USE_PRETRAINED:
    print("\nFreezing early layers for parameter-efficient fine-tuning...")

    # Freeze embedding layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'requires_grad_'):
        model.encoder.requires_grad_(False)
        print("  ✓ Froze embedding layers")

    # Freeze transformer encoder layers
    # Keep last 3 layers trainable (layers 9, 10, 11 out of 12)
    if hasattr(model, 'transformer_encoder'):
        n_layers_to_freeze = N_LAYERS - 3  # Freeze first 9 out of 12 layers

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
model.classifier = torch.nn.Linear(EMBSIZE, n_classes)
print("  ✓ Added trainable classification head")

model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Model initialized")
print(f"  Use pretrained: {USE_PRETRAINED}")
print(f"  Total parameters: {n_params:,}")
print(f"  Trainable parameters: {n_trainable:,}")

# ============================================================================
# Training Setup
# ============================================================================
print("\n" + "="*100)
print("Setting Up Training")
print("="*100)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Loss function with class weights for imbalance
n_control = (adata_train.obs['label'] == 0).sum()
n_diseased = (adata_train.obs['label'] == 1).sum()
class_weights = torch.tensor([n_diseased / n_control, 1.0], dtype=torch.float32).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

print(f"✓ Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"✓ Loss: CrossEntropyLoss with class weights [{class_weights[0]:.2f}, {class_weights[1]:.2f}]")
print(f"✓ Gradient clipping: {GRADIENT_CLIP}")

# ============================================================================
# Data Preparation Functions
# ============================================================================
def prepare_batch(adata_batch, gene_ids, max_len=MAX_SEQ_LEN):
    """
    Prepare a batch of cells for scGPT input

    Returns:
        gene_ids: Tensor of gene indices [batch_size, seq_len]
        values: Tensor of expression values [batch_size, seq_len]
        padding_mask: Boolean tensor indicating padding positions [batch_size, seq_len]
    """
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

        # Create padding mask (True for padding positions)
        mask = np.zeros(max_len, dtype=bool)

        # Pad if needed
        if len(genes) < max_len:
            pad_len = max_len - len(genes)
            genes = np.concatenate([genes, np.full(pad_len, vocab[PAD_TOKEN])])
            vals = np.concatenate([vals, np.full(pad_len, PAD_VALUE)])
            mask[len(nonzero_idx):] = True  # Mark padded positions as True

        gene_ids_batch.append(genes)
        values_batch.append(vals)
        padding_masks.append(mask)

    gene_ids_batch = torch.LongTensor(gene_ids_batch).to(DEVICE)
    values_batch = torch.FloatTensor(values_batch).to(DEVICE)
    padding_mask = torch.BoolTensor(padding_masks).to(DEVICE)

    return gene_ids_batch, values_batch, padding_mask

# ============================================================================
# Training and Validation Functions
# ============================================================================
def train_epoch(model, adata, optimizer, criterion, gene_ids):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    n_batches = len(adata) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(adata))

        # Get batch
        adata_batch = adata[start_idx:end_idx]
        labels = torch.LongTensor(adata_batch.obs['label'].values).to(DEVICE)

        # Prepare batch
        gene_ids_batch, values_batch, padding_mask = prepare_batch(adata_batch, gene_ids)

        try:
            # Get model output
            output = model(gene_ids_batch, values_batch, src_key_padding_mask=padding_mask)

            # Get CLS token representation (first token)
            if isinstance(output, dict):
                cls_output = output['cls_output'] if 'cls_output' in output else output['cell_emb']
            else:
                # Output is a tensor [batch_size, seq_len, d_model]
                cls_output = output[:, 0, :]  # Use first token (CLS)

            # Get classification logits
            logits = model.classifier(cls_output)

            # Compute loss (scale by gradient accumulation steps)
            loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS

            # Backward pass
            loss.backward()

            # Update weights every GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy

def validate(model, adata, criterion, gene_ids):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    n_batches = len(adata) // BATCH_SIZE

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(adata))

            # Get batch
            adata_batch = adata[start_idx:end_idx]
            labels = torch.LongTensor(adata_batch.obs['label'].values).to(DEVICE)

            # Prepare batch
            gene_ids_batch, values_batch, padding_mask = prepare_batch(adata_batch, gene_ids)

            try:
                # Forward pass
                output = model(gene_ids_batch, values_batch, src_key_padding_mask=padding_mask)

                # Get CLS token representation (first token)
                if isinstance(output, dict):
                    cls_output = output['cls_output'] if 'cls_output' in output else output['cell_emb']
                else:
                    # Output is a tensor [batch_size, seq_len, d_model]
                    cls_output = output[:, 0, :]  # Use first token (CLS)

                # Get classification logits
                logits = model.classifier(cls_output)

                # Compute loss
                loss = criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    accuracy = correct / total if total > 0 else 0

    # Calculate additional metrics
    if len(all_preds) > 0 and len(all_labels) > 0:
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    else:
        f1 = precision = recall = 0

    return avg_loss, accuracy, f1, precision, recall

# ============================================================================
# Training Loop
# ============================================================================
print("\n" + "="*100)
print("BEGINNING TRAINING")
print("="*100)

print(f"\nTraining configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches per epoch: {len(train_idx) // BATCH_SIZE}")
print(f"  Val batches per epoch: {len(val_idx) // BATCH_SIZE}")
print(f"  Device: {DEVICE}")
print(f"  Model save dir: {save_dir}")

print(f"\nDataset ready:")
print(f"  ✓ {adata_train.n_obs:,} training samples")
print(f"  ✓ {adata_val.n_obs:,} validation samples")
print(f"  ✓ {adata_test.n_obs:,} test samples")
print(f"  ✓ Species metadata included for mixed batches")

# Training loop
best_val_loss = float('inf')
best_val_acc = 0
patience_counter = 0
max_patience = 5

for epoch in range(EPOCHS):
    print(f"\n{'='*100}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*100}")

    # Train
    train_loss, train_acc = train_epoch(model, adata_train, optimizer, criterion, gene_names)
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # Validate
    val_loss, val_acc, val_f1, val_prec, val_rec = validate(model, adata_val, criterion, gene_names)
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"  Val F1: {val_f1:.4f} | Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f}")

    # Log to W&B
    if WANDB_AVAILABLE:
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_precision': val_prec,
            'val_recall': val_rec
        })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        patience_counter = 0

        # Save checkpoint
        checkpoint_path = save_dir / f'best_model_epoch{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }, checkpoint_path)
        print(f"  ✓ Saved best model to {checkpoint_path}")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print(f"\n{'='*100}")
print("Training Complete!")
print(f"{'='*100}")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best validation loss: {best_val_loss:.4f}")

# Save splits for later use
print(f"\nSaving dataset splits...")
adata_train.write_h5ad(save_dir / 'train_split.h5ad')
adata_val.write_h5ad(save_dir / 'val_split.h5ad')
adata_test.write_h5ad(save_dir / 'test_split.h5ad')
print(f"✓ Splits saved to {save_dir}")

print("\n" + "="*100)
print("SETUP COMPLETE - Ready for Training")
print("="*100)
