# %% [markdown]
# # Time Series Analysis of PLA and Protein Signals from Script 72 Data
# This script reads the single-cell data processed similarly to Script 72,
# and generates comprehensive time series plots with spline interpolation and boxplots
# for all markers (proteins and PLA signals).

# %%
import numpy as np
import pandas as pd
import time
import os
import sys
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed
import scipy.interpolate
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from statannot import add_stat_annotation
import re

sns.set_style('whitegrid')
np.random.seed(0)

# %% [markdown]
# # directories and inputs

# %%
# Setup paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'
FIGURES_ROOT = PROJECT_ROOT / 'figures'

# --- USER INPUTS ---
# Path to the channels Excel file (used to find the base data path)
channels_path = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P8 24 well plate 015' / '17Oct2025_Plate015_multiplex_cycles1,4,5,6.xlsx'
# Path to the plate layout file
layout_path = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P8 24 well plate 015' / 'Plate015_layout.xlsx'

# --- DERIVED PATHS ---
try:
    channels = pd.read_excel(channels_path)
    channels.dropna(subset=['StitchPath'], inplace=True)
    # Extract cycle folder name from StitchPath
    stitchPath = Path(channels['StitchPath'].iloc[-1])
    cycleFolderName = stitchPath.parent.name
    # Construct path relative to DATA_ROOT
    basePath = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P8 24 well plate 015' / cycleFolderName
    # This is the folder containing the PKL files from Script 71
    pklPath = basePath / '10 PKL single cell'
    # This is the folder where plots will be saved
    screenshotSavePath = FIGURES_ROOT / '77_boxplot_spline_PLA_analysis'
    screenshotSavePath.mkdir(exist_ok=True, parents=True)
    assert pklPath.exists(), f"PKL path not found: {pklPath}"
    print(f"Found PKL data path: {pklPath}")
    print(f"Plots will be saved to: {screenshotSavePath}")
except Exception as e:
    print(f"Error setting up paths: {e}")
    print("Please ensure the channels Excel file path is correct.")
    pklPath = Path('./10_PKL_single_cell_dummy')
    pklPath.mkdir(exist_ok=True)
    screenshotSavePath = FIGURES_ROOT / '77_boxplot_spline_PLA_analysis'
    screenshotSavePath.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load and Process Plate Layout Information

# %%
def parse_layout(layout_file_path):
    """Parses the plate layout Excel file to map wells to conditions."""
    try:
        df_layout = pd.read_excel(layout_file_path)
    except FileNotFoundError:
        print(f"Layout file not found at {layout_file_path}. Please check the path.")
        return pd.DataFrame()

    # The first column is the row identifier (A, B, C, D)
    df_layout = df_layout.set_index(df_layout.columns[0])

    # Melt the dataframe to a long format
    df_melted = df_layout.melt(ignore_index=False, var_name='Column', value_name='ConditionStr').reset_index()
    df_melted.rename(columns={df_melted.columns[0]: 'Row'}, inplace=True)

    # Create the Well ID
    df_melted['Well'] = df_melted['Row'] + df_melted['Column'].astype(str)

    # Extract condition and time using regex
    pattern = re.compile(r'([a-zA-Z0-9\s]+?)\s\d+\s[a-zA-Z\/]+\s(\d+)\s?m')

    # Apply the pattern to the 'ConditionStr' column
    extracted_data = df_melted['ConditionStr'].str.extract(pattern)
    df_melted['Condition'] = extracted_data[0].str.strip()
    df_melted['Timepoint'] = pd.to_numeric(extracted_data[1])

    # Handle DMSO control naming
    df_melted['Condition'] = df_melted['Condition'].replace({'DMSO': 'DMSO Control'})

    df_melted.dropna(subset=['Condition', 'Timepoint'], inplace=True)

    return df_melted[['Well', 'Condition', 'Timepoint']]

layout_df = parse_layout(layout_path)
print("Processed Plate Layout:")
print(layout_df.head())

# %% [markdown]
# # Process PKL Files to Single Cell Format

# %%
def process_pkl_file(file_path, layout_map):
    """
    Reads a single PKL file and computes per-cell metrics.
    Returns a dataframe with CellLabel as index.
    """
    try:
        df = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None

    if df.empty:
        return None

    # Get the well ID from the filename (e.g., 'A1-1' -> 'A1')
    well_id = file_path.stem.split('-')[0]

    # Get protein columns
    protein_cols = [col for col in df.columns if ' Protein' in col]
    # Get PLA/RNA columns (dots)
    pla_cols = [col for col in df.columns if ' Dots' in col]

    if not protein_cols and not pla_cols:
        return None

    # Aggregate by cell and region for proteins (median)
    if protein_cols:
        protein_data = df.groupby(['CellLabel', 'CellRegion'])[protein_cols].median()
    else:
        protein_data = pd.DataFrame()

    # Sum PLA dots per cell (across all regions)
    if pla_cols:
        pla_data = df.groupby('CellLabel')[pla_cols].sum()
    else:
        pla_data = pd.DataFrame()

    # Compute p65 Nuc/Cyto ratio if available
    nuc_cyto_ratio = pd.Series(dtype=float)
    if 'p65 Protein' in protein_cols:
        p65_by_region = df.groupby(['CellLabel', 'CellRegion'])['p65 Protein'].median().unstack()
        if 'Nucleus' in p65_by_region.columns and 'Cytosol' in p65_by_region.columns:
            nuc_cyto_ratio = (p65_by_region['Nucleus'] + 1) / (p65_by_region['Cytosol'] + 1)
            nuc_cyto_ratio.name = 'NucCytoRatio'

    # Combine all cell-level data
    cell_df = pd.DataFrame(index=df['CellLabel'].unique())

    # Add protein data (pivot to have region as part of column name)
    if not protein_data.empty:
        protein_pivot = protein_data.unstack(level='CellRegion')
        protein_pivot.columns = [f"{col[0]} {col[1]}" for col in protein_pivot.columns]
        cell_df = cell_df.join(protein_pivot)

    # Add PLA data
    if not pla_data.empty:
        cell_df = cell_df.join(pla_data)

    # Add NucCytoRatio
    if not nuc_cyto_ratio.empty:
        cell_df = cell_df.join(nuc_cyto_ratio)

    if cell_df.empty:
        return None

    # Add metadata
    condition_info = layout_map.get(well_id)
    if condition_info:
        cell_df['Condition'] = condition_info['Condition']
        cell_df['Timepoint'] = condition_info['Timepoint']
    else:
        cell_df['Condition'] = 'Unknown'
        cell_df['Timepoint'] = -1

    cell_df['Well'] = well_id
    cell_df['FOV'] = file_path.stem  # Use full filename as FOV identifier

    return cell_df

# %% [markdown]
# # Load All PKL Files

# %%
all_pkl_files = list(pklPath.glob("*.pkl"))
print(f"Found {len(all_pkl_files)} PKL files to process.")

# Create a dictionary from the layout dataframe for faster lookups
layout_map = layout_df.set_index('Well').to_dict('index')

# Process files in parallel
results_list = Parallel(n_jobs=10, verbose=10)(
    delayed(process_pkl_file)(f, layout_map) for f in all_pkl_files
)

# Combine all dataframes
dfCell = pd.concat([df for df in results_list if df is not None], ignore_index=False)
dfCell.reset_index(inplace=True)
dfCell.rename(columns={'index': 'CellLabel'}, inplace=True)

print("\nProcessing complete. Shape of the combined dataframe:", dfCell.shape)
print(dfCell.head())

# %%
print("\nCell counts per condition:")
print(dfCell.groupby(['Condition']).agg({'CellLabel': 'nunique'}))

# %%
dfCell.describe()

# %% [markdown]
# # Helper Function to Save Figures

# %%
def saveFigLabelTime(fig, prefix=''):
    """Save figure with current time as label"""
    now = datetime.now()
    now = now.strftime('%d%b%Y_%H%M%S')
    fileOut = f"{prefix}_{now}.png" if prefix else f"{now}.png"
    fileOut = screenshotSavePath / fileOut
    fig.savefig(fileOut, dpi=300, bbox_inches='tight', pad_inches=0)
    time.sleep(0.5)  # Reduced sleep time
    print(f"  Saved: {fileOut.name}")
    return None

# %% [markdown]
# # Plot 1: Line Plots with Spline Interpolation for Each Marker by Region

# %%
print("\nGenerating line plots with spline interpolation...")

# Get all protein markers (by region)
protein_region_cols = [col for col in dfCell.columns if ' Protein ' in col and
                       any(region in col for region in ['Nucleus', 'Cytosol', 'Nuclear Membrane'])]

# Get all PLA markers (whole cell)
pla_cols = [col for col in dfCell.columns if ' Dots' in col]

# Define condition order and colors
condition_order = [c for c in ['TNFa', 'IL1B', 'DMSO Control'] if c in dfCell['Condition'].unique()]
control_condition = 'DMSO Control'

for marker_col in protein_region_cols + pla_cols:

    # Determine marker type and clean name
    if ' Protein ' in marker_col:
        marker_base = marker_col.split(' Protein ')[0]
        region = marker_col.split(' Protein ')[1]
        marker_name = f"{marker_base} Protein - {region}"
        y_label = f"{marker_base} (Median Intensity, AU)"
        data_type = 'Protein'
    elif ' Dots' in marker_col:
        marker_base = marker_col.replace(' Dots', '')
        marker_name = f"{marker_base} Dots"
        y_label = f"{marker_base} (Sum of Dots per Cell)"
        region = 'Whole Cell'
        data_type = 'PLA'
    else:
        continue

    # Create figure for each non-control condition
    for condition in [c for c in condition_order if c != control_condition]:

        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel(y_label)
        ax.set_title(f"{marker_name}\n{condition} vs {control_condition}")

        # Plot treated condition
        dfSub = dfCell[dfCell['Condition'] == condition].copy()
        dfSub = dfSub.dropna(subset=[marker_col])

        if len(dfSub) > 0:
            # Calculate mean and std
            grouped = dfSub.groupby('Timepoint')[marker_col]
            median = grouped.mean().reset_index()
            err = grouped.std().reset_index()

            # Scatter with error bars
            ax.scatter(median['Timepoint'], median[marker_col],
                      color='blue', alpha=1, s=50, zorder=3)
            ax.errorbar(median['Timepoint'], median[marker_col],
                       yerr=err[marker_col], fmt='o', color='blue',
                       alpha=0.5, capsize=5, zorder=2)

            # Spline interpolation
            if len(median) >= 3:
                try:
                    func = scipy.interpolate.interp1d(
                        x=median['Timepoint'],
                        y=median[marker_col],
                        kind='cubic'
                    )
                    timeSmooth = np.linspace(
                        median['Timepoint'].min(),
                        median['Timepoint'].max(),
                        1000
                    )
                    spline = func(timeSmooth)
                    ax.plot(timeSmooth, spline, color='blue',
                           alpha=1, linewidth=2, label=condition, zorder=1)
                except:
                    pass

        # Plot control condition
        dfCtrl = dfCell[dfCell['Condition'] == control_condition].copy()
        dfCtrl = dfCtrl.dropna(subset=[marker_col])

        if len(dfCtrl) > 0:
            # Calculate mean and std
            grouped = dfCtrl.groupby('Timepoint')[marker_col]
            median = grouped.mean().reset_index()
            err = grouped.std().reset_index()

            # Scatter with error bars
            ax.scatter(median['Timepoint'], median[marker_col],
                      color='gray', alpha=0.6, s=50, zorder=3)
            ax.errorbar(median['Timepoint'], median[marker_col],
                       yerr=err[marker_col], fmt='o', color='gray',
                       alpha=0.5, capsize=5, zorder=2)

            # Spline interpolation
            if len(median) >= 3:
                try:
                    func = scipy.interpolate.interp1d(
                        x=median['Timepoint'],
                        y=median[marker_col],
                        kind='cubic'
                    )
                    timeSmooth = np.linspace(
                        median['Timepoint'].min(),
                        median['Timepoint'].max(),
                        1000
                    )
                    spline = func(timeSmooth)
                    ax.plot(timeSmooth, spline, color='gray',
                           linestyle='--', alpha=0.6, linewidth=2,
                           label=control_condition, zorder=1)
                except:
                    pass

        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
        plt.tight_layout()

        # Save figure
        safe_marker_name = marker_name.replace(' ', '_').replace('/', '_')
        safe_condition = condition.replace(' ', '_')
        saveFigLabelTime(fig, prefix=f"spline_{safe_marker_name}_{safe_condition}")
        plt.close(fig)

print("Line plots complete.")

# %% [markdown]
# # Plot 2: Boxplots with Overlaid Spline for Each Marker

# %%
print("\nGenerating boxplots with spline overlays...")

for marker_col in protein_region_cols + pla_cols:

    # Determine marker type and clean name
    if ' Protein ' in marker_col:
        marker_base = marker_col.split(' Protein ')[0]
        region = marker_col.split(' Protein ')[1]
        marker_name = f"{marker_base} Protein - {region}"
        y_label = f"{marker_base} (Median Intensity, AU)"
    elif ' Dots' in marker_col:
        marker_base = marker_col.replace(' Dots', '')
        marker_name = f"{marker_base} Dots"
        y_label = f"{marker_base} (Sum of Dots per Cell)"
        region = 'Whole Cell'
    else:
        continue

    # Create figure for each condition
    for condition in condition_order:

        dfSub = dfCell[dfCell['Condition'] == condition].copy()
        dfSub = dfSub.dropna(subset=[marker_col])

        if len(dfSub) == 0:
            continue

        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel(y_label)
        ax.set_title(f"{marker_name}\n{condition}")

        # Create boxplot
        timepoints = sorted(dfSub['Timepoint'].unique())
        bins, groups = zip(*dfSub.groupby('Timepoint')[marker_col])

        bp = ax.boxplot(
            groups,
            positions=timepoints,
            widths=5,
            patch_artist=True,
            boxprops={'facecolor': [1, 1, 1, 0.5], 'edgecolor': 'blue', 'linewidth': 1.5},
            medianprops={'color': 'red', 'linewidth': 2},
            flierprops={'markersize': 2, 'marker': 'o', 'color': 'blue'},
            showfliers=True
        )

        # Overlay spline of median
        median = dfSub.groupby('Timepoint')[marker_col].median().reset_index()

        if len(median) >= 3:
            try:
                func = scipy.interpolate.interp1d(
                    x=median['Timepoint'],
                    y=median[marker_col],
                    kind='cubic'
                )
                timeSmooth = np.linspace(
                    median['Timepoint'].min(),
                    median['Timepoint'].max(),
                    1000
                )
                spline = func(timeSmooth)
                ax.plot(timeSmooth, spline, color='red',
                       linewidth=2, alpha=0.8, label='Median Spline')
            except:
                pass

        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
        plt.tight_layout()

        # Save figure
        safe_marker_name = marker_name.replace(' ', '_').replace('/', '_')
        safe_condition = condition.replace(' ', '_')
        saveFigLabelTime(fig, prefix=f"boxplot_{safe_marker_name}_{safe_condition}")
        plt.close(fig)

print("Boxplots complete.")

# %% [markdown]
# # Plot 3: Nuclear/Cytosol Ratio with Boxplots and Spline

# %%
print("\nGenerating Nuclear/Cytosol ratio plots...")

if 'NucCytoRatio' in dfCell.columns:

    dfRatio = dfCell.dropna(subset=['NucCytoRatio']).copy()

    for condition in condition_order:

        dfSub = dfRatio[dfRatio['Condition'] == condition].copy()

        if len(dfSub) == 0:
            continue

        # Plot with outliers
        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel('N/C p65 Protein Ratio (AU)')
        ax.set_title(f"{condition} - Nuclear/Cytosol p65 Ratio\n(with outliers)")

        timepoints = sorted(dfSub['Timepoint'].unique())
        bins, groups = zip(*dfSub.groupby('Timepoint')['NucCytoRatio'])

        bp = ax.boxplot(
            groups,
            positions=timepoints,
            widths=5,
            patch_artist=True,
            boxprops={'facecolor': 'magenta', 'alpha': 0.6},
            medianprops={'color': 'black', 'linewidth': 2},
            flierprops={'markersize': 2, 'marker': 'o', 'color': 'black'},
            showfliers=True
        )

        # Overlay spline
        median = dfSub.groupby('Timepoint')['NucCytoRatio'].median().reset_index()

        if len(median) >= 3:
            try:
                func = scipy.interpolate.interp1d(
                    x=median['Timepoint'],
                    y=median['NucCytoRatio'],
                    kind='cubic'
                )
                timeSmooth = np.linspace(
                    median['Timepoint'].min(),
                    median['Timepoint'].max(),
                    1000
                )
                spline = func(timeSmooth)
                ax.plot(timeSmooth, spline, color='magenta',
                       linewidth=2, alpha=0.8)
            except:
                pass

        plt.tight_layout()
        safe_condition = condition.replace(' ', '_')
        saveFigLabelTime(fig, prefix=f"nuc_cyto_ratio_{safe_condition}_with_outliers")
        plt.close(fig)

        # Plot without outliers, with log scale
        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel('N/C p65 Protein Ratio (AU)')
        ax.set_title(f"{condition} - Nuclear/Cytosol p65 Ratio\n(log scale, no outliers)")

        bp = ax.boxplot(
            groups,
            positions=timepoints,
            widths=5,
            patch_artist=True,
            boxprops={'facecolor': 'magenta', 'alpha': 0.6},
            medianprops={'color': 'black', 'linewidth': 2},
            flierprops={'markersize': 2, 'marker': 'o', 'color': 'black'},
            showfliers=False
        )

        # Overlay spline
        if len(median) >= 3:
            try:
                func = scipy.interpolate.interp1d(
                    x=median['Timepoint'],
                    y=median['NucCytoRatio'],
                    kind='cubic'
                )
                timeSmooth = np.linspace(
                    median['Timepoint'].min(),
                    median['Timepoint'].max(),
                    1000
                )
                spline = func(timeSmooth)
                ax.plot(timeSmooth, spline, color='magenta',
                       linewidth=2, alpha=0.8)
            except:
                pass

        ax.set_yscale('log')
        plt.tight_layout()
        saveFigLabelTime(fig, prefix=f"nuc_cyto_ratio_{safe_condition}_log_no_outliers")
        plt.close(fig)

    print("Nuclear/Cytosol ratio plots complete.")

# %% [markdown]
# # Plot 4: Dual-Axis Plots (Protein and PLA on Same Figure)

# %%
print("\nGenerating dual-axis plots (Protein + PLA)...")

# For p65 protein in each region + corresponding PLA markers
p65_regions = [col for col in dfCell.columns if col.startswith('p65 Protein ')]

for p65_col in p65_regions:
    region = p65_col.replace('p65 Protein ', '')

    for condition in [c for c in condition_order if c != control_condition]:

        fig, ax1 = plt.subplots(figsize=(6, 3.5), dpi=300)
        ax1.set_xlabel('Time (mins)')
        ax1.set_title(f"p65 Protein and PLA Signals - {region}\n{condition} vs {control_condition}")

        ax2 = ax1.twinx()
        axes = [ax1, ax2]
        colors = ['blue', 'green']

        # Plot p65 protein on ax1
        dfSub = dfCell[dfCell['Condition'] == condition].dropna(subset=[p65_col])

        if len(dfSub) > 0:
            grouped = dfSub.groupby('Timepoint')[p65_col]
            median = grouped.mean().reset_index()
            err = grouped.std().reset_index()

            ax1.scatter(median['Timepoint'], median[p65_col],
                       color=colors[0], alpha=1, s=50, zorder=3)
            ax1.errorbar(median['Timepoint'], median[p65_col],
                        yerr=err[p65_col], fmt='o', color=colors[0],
                        alpha=0.5, capsize=5, zorder=2)

            if len(median) >= 3:
                try:
                    func = scipy.interpolate.interp1d(
                        x=median['Timepoint'], y=median[p65_col], kind='cubic'
                    )
                    timeSmooth = np.linspace(
                        median['Timepoint'].min(), median['Timepoint'].max(), 1000
                    )
                    spline = func(timeSmooth)
                    ax1.plot(timeSmooth, spline, color=colors[0],
                            linewidth=2, label=f"{condition}", zorder=1)
                except:
                    pass

        # Plot control p65
        dfCtrl = dfCell[dfCell['Condition'] == control_condition].dropna(subset=[p65_col])

        if len(dfCtrl) > 0:
            grouped = dfCtrl.groupby('Timepoint')[p65_col]
            median = grouped.mean().reset_index()
            err = grouped.std().reset_index()

            ax1.scatter(median['Timepoint'], median[p65_col],
                       color=colors[0], alpha=0.5, s=50, zorder=3)
            ax1.errorbar(median['Timepoint'], median[p65_col],
                        yerr=err[p65_col], fmt='o', color=colors[0],
                        alpha=0.3, capsize=5, zorder=2)

            if len(median) >= 3:
                try:
                    func = scipy.interpolate.interp1d(
                        x=median['Timepoint'], y=median[p65_col], kind='cubic'
                    )
                    timeSmooth = np.linspace(
                        median['Timepoint'].min(), median['Timepoint'].max(), 1000
                    )
                    spline = func(timeSmooth)
                    ax1.plot(timeSmooth, spline, color=colors[0],
                            linestyle='--', linewidth=2, alpha=0.5,
                            label=f"{control_condition}", zorder=1)
                except:
                    pass

        ax1.set_ylabel('p65 Protein (AU)', color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])

        # Plot PLA on ax2 (if available)
        if pla_cols:
            pla_col = pla_cols[0]  # Use first PLA marker

            dfSub = dfCell[dfCell['Condition'] == condition].dropna(subset=[pla_col])

            if len(dfSub) > 0:
                grouped = dfSub.groupby('Timepoint')[pla_col]
                median = grouped.mean().reset_index()
                err = grouped.std().reset_index()

                ax2.scatter(median['Timepoint'], median[pla_col],
                           color=colors[1], alpha=1, s=50, zorder=3)
                ax2.errorbar(median['Timepoint'], median[pla_col],
                            yerr=err[pla_col], fmt='o', color=colors[1],
                            alpha=0.5, capsize=5, zorder=2)

                if len(median) >= 3:
                    try:
                        func = scipy.interpolate.interp1d(
                            x=median['Timepoint'], y=median[pla_col], kind='cubic'
                        )
                        timeSmooth = np.linspace(
                            median['Timepoint'].min(), median['Timepoint'].max(), 1000
                        )
                        spline = func(timeSmooth)
                        ax2.plot(timeSmooth, spline, color=colors[1],
                                linewidth=2, zorder=1)
                    except:
                        pass

            # Plot control PLA
            dfCtrl = dfCell[dfCell['Condition'] == control_condition].dropna(subset=[pla_col])

            if len(dfCtrl) > 0:
                grouped = dfCtrl.groupby('Timepoint')[pla_col]
                median = grouped.mean().reset_index()
                err = grouped.std().reset_index()

                ax2.scatter(median['Timepoint'], median[pla_col],
                           color=colors[1], alpha=0.5, s=50, zorder=3)
                ax2.errorbar(median['Timepoint'], median[pla_col],
                            yerr=err[pla_col], fmt='o', color=colors[1],
                            alpha=0.3, capsize=5, zorder=2)

                if len(median) >= 3:
                    try:
                        func = scipy.interpolate.interp1d(
                            x=median['Timepoint'], y=median[pla_col], kind='cubic'
                        )
                        timeSmooth = np.linspace(
                            median['Timepoint'].min(), median['Timepoint'].max(), 1000
                        )
                        spline = func(timeSmooth)
                        ax2.plot(timeSmooth, spline, color=colors[1],
                                linestyle='--', linewidth=2, alpha=0.5, zorder=1)
                    except:
                        pass

            ax2.set_ylabel(f"{pla_col.replace(' Dots', '')} Dots (AU)", color=colors[1])
            ax2.tick_params(axis='y', labelcolor=colors[1])

        ax1.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
        plt.tight_layout()

        safe_region = region.replace(' ', '_')
        safe_condition = condition.replace(' ', '_')
        saveFigLabelTime(fig, prefix=f"dual_axis_{safe_region}_{safe_condition}")
        plt.close(fig)

print("Dual-axis plots complete.")

# %% [markdown]
# # Plot 5: Heatmap - Cell Count by Condition and Time

# %%
print("\nGenerating cell count heatmap...")

pivot_counts = dfCell.pivot_table(
    values='CellLabel',
    index='Condition',
    columns='Timepoint',
    aggfunc='nunique'
)

if not pivot_counts.empty:
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

    sns.heatmap(
        pivot_counts.fillna(0).astype(int),
        annot=True,
        fmt='d',
        cmap='coolwarm',
        cbar_kws={'label': 'Number of Cells'},
        ax=ax
    )

    ax.set_title("Cell Count Heatmap", pad=20)
    ax.set_xlabel("Timepoint (minutes)")
    ax.set_ylabel("Condition")

    plt.tight_layout()
    saveFigLabelTime(fig, prefix="heatmap_cell_count")
    plt.close(fig)

print("Cell count heatmap complete.")

print("\nAll plots generated successfully!")
print(f"Plots saved to: {screenshotSavePath}")
