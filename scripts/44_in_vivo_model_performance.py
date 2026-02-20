#!/usr/bin/env python3
"""
scGPT In Vivo Model Performance Visualization
Loads metrics from trained model checkpoint and creates performance visualizations

Designed for the in vivo model trained with script 35
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEED = 42

# Paths
base_dir = Path('Y:/coskun-lab/Nicky/71 CF AI Foundation model')
model_dir = base_dir / 'Models' / 'scGPT' / 'invivo'
plot_dir = base_dir / 'Figures' / 'in_vivo_model_performance'
plot_dir.mkdir(parents=True, exist_ok=True)

print("="*100)
print("scGPT In Vivo Model Performance Visualization")
print("="*100)
print(f"Model directory: {model_dir}")
print(f"Plot directory: {plot_dir}")
print()

# ============================================================================
# Load Metrics from Best Checkpoint
# ============================================================================
print("="*100)
print("Loading Performance Metrics from Checkpoint")
print("="*100)

# Find the best model checkpoint
checkpoints = list(model_dir.glob('best_model_epoch*.pt'))
if not checkpoints:
    print("Error: No model checkpoints found!")
    sys.exit(1)

# Get the checkpoint with highest F1 (from filename)
checkpoint_path = sorted(checkpoints, key=lambda x: float(str(x.stem).split('_f1')[-1]))[-1]
print(f"\nLoading checkpoint: {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"Checkpoint epoch: {checkpoint['epoch']}")

# Extract metrics
if 'test_metrics' in checkpoint:
    metrics = checkpoint['test_metrics']
    print("\nTest Set Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:       {metrics['auc']:.4f}")
else:
    print("\nError: No test_metrics found in checkpoint")
    sys.exit(1)

# ============================================================================
# Create Performance Plots
# ============================================================================
print("\n" + "="*100)
print("Creating Performance Plots")
print("="*100)

# Set style
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Figure 1: Bar plot of all metrics
fig, ax = plt.subplots()

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [
    metrics['accuracy'],
    metrics['precision'],
    metrics['recall'],
    metrics['f1']
]

if 'auc' in metrics:
    metric_names.append('AUC')
    metric_values.append(metrics['auc'])

colors = sns.color_palette("husl", len(metric_names))
bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Score', fontweight='bold', fontsize=16)
ax.set_title('In Vivo Model Performance (Test Set)\nEpoch {}'.format(checkpoint['epoch']),
             fontweight='bold', fontsize=18)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = plot_dir / 'performance_metrics.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {plot_path}")
plt.close()

# Figure 2: Radar/Spider plot
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
values = metric_values + [metric_values[0]]  # Close the plot
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', markersize=8)
ax.fill(angles, values, alpha=0.25, color='#2E86AB')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.grid(True)
ax.set_title('In Vivo Model Performance (Radar Plot)\nEpoch {}'.format(checkpoint['epoch']),
             fontweight='bold', fontsize=16, pad=20)

plt.tight_layout()
plot_path = plot_dir / 'performance_radar.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot_path}")
plt.close()

# Figure 3: Horizontal bar chart (alternative visualization)
fig, ax = plt.subplots()

y_pos = np.arange(len(metric_names))
colors = sns.color_palette("coolwarm", len(metric_names))
bars = ax.barh(y_pos, metric_values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, metric_values)):
    ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
            f'{value:.4f}',
            va='center', fontsize=14, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(metric_names, fontsize=14, fontweight='bold')
ax.set_xlabel('Score', fontweight='bold', fontsize=16)
ax.set_title('In Vivo Model Performance (Test Set)\nEpoch {}'.format(checkpoint['epoch']),
             fontweight='bold', fontsize=18)
ax.set_xlim(0, 1.05)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plot_path = plot_dir / 'performance_horizontal.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot_path}")
plt.close()

# ============================================================================
# Save Metrics Summary
# ============================================================================
print("\n" + "="*100)
print("Saving Metrics Summary")
print("="*100)

import pandas as pd

# Create summary dataframe
summary_df = pd.DataFrame({
    'Metric': metric_names,
    'Value': metric_values,
    'Epoch': checkpoint['epoch']
})

summary_path = plot_dir / 'metrics_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Saved: {summary_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*100)
print("MODEL PERFORMANCE VISUALIZATION COMPLETE")
print("="*100)
print(f"\nAll plots saved to: {plot_dir}")
print(f"\nGenerated visualizations:")
print(f"  - performance_metrics.png (bar chart)")
print(f"  - performance_radar.png (radar plot)")
print(f"  - performance_horizontal.png (horizontal bar chart)")
print(f"  - metrics_summary.csv (metrics table)")

print(f"\n{'='*100}")
print("FINAL PERFORMANCE SUMMARY")
print(f"{'='*100}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics['f1']:.4f}")
if 'auc' in metrics:
    print(f"AUC:       {metrics['auc']:.4f}")
print(f"{'='*100}\n")
