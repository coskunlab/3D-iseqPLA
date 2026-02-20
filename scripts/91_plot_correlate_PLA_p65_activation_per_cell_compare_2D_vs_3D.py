# %% [markdown]
# # Analysis of Single-Cell p65 Activation and PLA Signals
# This script reads the single-cell data exported from Script 71, calculates p65 nuclear translocation as a measure of activation, and quantifies PLA dot signals. It then correlates these two metrics across different experimental conditions (stimulant and timepoint) and visualizes the results as heatmaps.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import re

# %% [markdown]
# ## 1. Setup Directories and Load Metadata
# First, we'll define the paths to our input files, including the channels file to locate the data and the plate layout file to map wells to experimental conditions.

# %%
# --- USER INPUTS ---
# Path to the channels Excel file (used to find the base data path)
channels_path = r"Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P8 24 well plate 015\17Oct2025_Plate015_multiplex_cycles1,4,5,6.xlsx"
# Path to the plate layout file
layout_path = r"Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P8 24 well plate 015\Plate015_layout.xlsx"

# --- DERIVED PATHS ---
try:
    channels = pd.read_excel(channels_path)
    channels.dropna(subset=['StitchPath'], inplace=True)
    basePath = Path(channels['StitchPath'].iloc[-1]).parent
    # This is the folder containing the PKL files from Script 71
    pklPath = basePath / '10 PKL single cell'
    # This is the folder where plots will be saved
    plotPath = basePath / '11 PNG plots'
    plotPath.mkdir(exist_ok=True)
    assert pklPath.exists(), f"PKL path not found: {pklPath}"
    print(f"Found PKL data path: {pklPath}")
    print(f"Plots will be saved to: {plotPath}")
except Exception as e:
    print(f"Error setting up paths: {e}")
    print("Please ensure the channels Excel file path is correct.")
    pklPath = Path('./10_PKL_single_cell_dummy')
    pklPath.mkdir(exist_ok=True)
    plotPath = Path('./11_PNG_plots_dummy')
    plotPath.mkdir(exist_ok=True)


# %% [markdown]
# ## 2. Load and Process Plate Layout Information
# We need to parse the layout file to create a clean mapping from each well ID (e.g., 'A1') to its corresponding stimulant and timepoint.

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
    # This pattern captures the stimulant name and the time in minutes
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
# ## 3. Define the PKL File Processing Function
# This function will be applied to each `.pkl` file. It computes the p65 activation ratio and sums the PLA dots for every cell in the file.

# %%
def process_pkl_file(file_path, layout_map, processing_type='3D'):
    """
    Reads a single PKL file, computes metrics for each cell, and returns a dataframe.
    
    Parameters:
    -----------
    processing_type : str
        '3D' - Compute median (protein) or sum (PLA/RNA) per cell directly from 3D data (no Z aggregation)
        '2D' - First apply maximum intensity projection across Z, then compute median (protein) or sum (PLA/RNA) per cell
    """
    try:
        df = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None

    # Get the well ID from the filename (e.g., 'A1-1' -> 'A1')
    well_id = file_path.stem.split('-')[0]
    
    # --- 1. Compute p65 Nuc/Cyto Ratio ---
    if 'p65 Protein' not in df.columns:
        return None # Skip files without p65 signal

    p65_median = df.groupby(['CellLabel', 'CellRegion'])['p65 Protein'].median().unstack()
    
    # Ensure both Nucleus and Cytosol columns exist
    if 'Nucleus' not in p65_median.columns or 'Cytosol' not in p65_median.columns:
        return None
        
    # Calculate ratio, replacing division by zero or NaN with NaN
    p65_median['p65 Nuc/Cyto Ratio'] = p65_median['Nucleus'] / p65_median['Cytosol']
    p65_median.replace([np.inf, -np.inf], np.nan, inplace=True)
    p65_results = p65_median[['p65 Nuc/Cyto Ratio']].dropna()

    # --- 2. Get all Protein medians (for bar plots) ---
    prot_cols = [col for col in df.columns if ' Protein' in col]
    if not prot_cols:
        return None # Should be impossible given p65 check, but good practice
    prot_medians = df.groupby('CellLabel')[prot_cols].median()

    # --- 3. Sum all PLA signals ---
    pla_cols = [col for col in df.columns if ' Dots' in col]
    
    # --- 4. Combine results ---
    cell_df = p65_results.join(prot_medians, how='inner')
    
    if pla_cols:
        # User requested SUM of dots per cell
        pla_sums = df.groupby('CellLabel')[pla_cols].sum()
        cell_df = cell_df.join(pla_sums, how='inner') # Use inner join, cells should have all data types
    
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

    return cell_df

# %% [markdown]
# ## 4. Process All PKL Files
# Now we'll find all the `.pkl` files and use `joblib` to process them. The results will be combined into a single master dataframe.

# %%
all_pkl_files = list(pklPath.glob("*.pkl"))
print(f"Found {len(all_pkl_files)} PKL files to process.")

# Create a dictionary from the layout dataframe for faster lookups
layout_map = layout_df.set_index('Well').to_dict('index')

# Process files for 3D data (using median aggregation across Z-slices)
print("\nProcessing 3D data (median across Z-slices)...")
results_list_3d = joblib.Parallel(n_jobs=1)(
    joblib.delayed(process_pkl_file)(f, layout_map, '3D') for f in tqdm(all_pkl_files, desc="Processing PKL files (3D)")
)

# Process files for 2D data (using maximum intensity projection across Z-slices)
print("\nProcessing 2D data (maximum intensity projection across Z-slices)...")
results_list_2d = joblib.Parallel(n_jobs=1)(
    joblib.delayed(process_pkl_file)(f, layout_map, '2D') for f in tqdm(all_pkl_files, desc="Processing PKL files (2D)")
)

# Combine 3D dataframes and add ProcessingType
df_list_3d = [df for df in results_list_3d if df is not None]
if df_list_3d:
    all_data_3d = pd.concat(df_list_3d, ignore_index=False)
    all_data_3d['ProcessingType'] = '3D'
else:
    all_data_3d = pd.DataFrame()

# Combine 2D dataframes and add ProcessingType
df_list_2d = [df for df in results_list_2d if df is not None]
if df_list_2d:
    all_data_2d = pd.concat(df_list_2d, ignore_index=False)
    all_data_2d['ProcessingType'] = '2D'
else:
    all_data_2d = pd.DataFrame()

print(f"\n3D data: {len(all_data_3d)} cells")
print(f"2D data: {len(all_data_2d)} cells")
print("\nProcessing complete.")


# %% [markdown]
# ## 6. Calculate Correlations and Plot Heatmaps (3D and 2D)
# Finally, we'll calculate the Pearson correlation between the p65 activation ratio and each PLA signal, grouped by condition and timepoint. These correlations will be visualized as individual heatmaps for both 3D and 2D data.

# %%
# Get the list of PLA markers to plot (use 3D data to get markers, or 2D if 3D is empty)
if not all_data_3d.empty:
    pla_markers = [col for col in all_data_3d.columns if ' Dots' in col]
elif not all_data_2d.empty:
    pla_markers = [col for col in all_data_2d.columns if ' Dots' in col]
else:
    pla_markers = []

if not pla_markers or (all_data_3d.empty and all_data_2d.empty):
    print("\nNo PLA markers found or data is empty. Cannot generate heatmaps.")
else:
    # Generate heatmaps for both 3D and 2D
    for label, data_df in [('3D', all_data_3d), ('2D', all_data_2d)]:
        if data_df.empty:
            print(f"\n[{label}] No data available. Skipping heatmaps.")
            continue
            
        print(f"\n[{label}] Generating {len(pla_markers)} separate heatmaps...")
        for marker in pla_markers:
            # Create a new figure for each heatmap
            fig, ax = plt.subplots(figsize=(8, 7))
            
            # Calculate the correlation for the current marker
            corr_series = data_df.groupby(['Condition', 'Timepoint']).apply(
                lambda df: df['p65 Nuc/Cyto Ratio'].corr(df[marker])
            )
            
            # Format into a pivot table for the heatmap
            heatmap_data = corr_series.unstack(level='Condition')
            
            # Sort rows and columns for a clean look
            heatmap_data = heatmap_data.sort_index()
            if 'DMSO Control' in heatmap_data.columns:
                # Reorder columns to have control last
                cols = [c for c in heatmap_data.columns if c != 'DMSO Control'] + ['DMSO Control']
                heatmap_data = heatmap_data[cols]

            sns.heatmap(
                heatmap_data,
                ax=ax,
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                fmt=".2f",
                linewidths=.5
            )
            
            marker_name_clean = marker.replace(' Dots', '').replace(' ', '_')
            ax.set_title(f"p65 Activation vs. {marker.replace(' Dots', '')} Signal ({label})\n(Pearson Correlation)")
            ax.set_ylabel("Timepoint (minutes)")
            ax.set_xlabel("Stimulant")
            
            plt.tight_layout()

            # Save the individual figure to the specified folder
            plot_filename = plotPath / f"p65_vs_{marker_name_clean}_correlation_{label}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"  Saved heatmap to: {plot_filename}")

            plt.close(fig) # Close the figure to free memory

# %% [markdown]
# ## 6. Plot Signal Intensity Bar Plots
# Bar plots for each marker (protein and PLA) over time by condition. Separate plots for **3D** and **2D** processing.

# %%
for label, all_data_df in [('3D', all_data_3d), ('2D', all_data_2d)]:
    protein_markers = [col for col in all_data_df.columns if ' Protein' in col]
    pla_markers = [col for col in all_data_df.columns if ' Dots' in col]
    all_markers_to_plot = protein_markers + pla_markers
    condition_order = [c for c in ['TNFa', 'IL1B', 'DMSO Control'] if c in all_data_df['Condition'].unique()]
    palette = sns.color_palette("deep", len(condition_order))

    if not all_markers_to_plot or all_data_df.empty:
        print(f"\n[{label}] No marker data. Skipping bar plots.")
        continue
    print(f"\n[{label}] Generating {len(all_markers_to_plot)} bar plots...")
    for marker in all_markers_to_plot:
        if ' Protein' in marker:
            plot_type = 'Median Signal'
            marker_name_clean = marker.replace(' Protein', '').replace(' ', '_')
            y_label = f"{marker.replace(' Protein', '')} (Median Intensity)"
        elif ' Dots' in marker:
            plot_type = 'Sum per Cell'
            marker_name_clean = marker.replace(' Dots', '').replace(' ', '_')
            y_label = f"{marker.replace(' Dots', '')} (Sum of Dots)"
        else:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=all_data_df, x='Timepoint', y=marker,
            hue='Condition', hue_order=condition_order, palette=palette,
            errorbar='se', capsize=0.1, ax=ax
        )
        ax.set_title(f"{marker.replace(' Protein', '').replace(' Dots', '')} Signal over Time ({label})\n({plot_type})")
        ax.set_ylabel(y_label)
        ax.set_xlabel("Timepoint (minutes)")
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_filename = plotPath / f"barplot_{marker_name_clean}_signal_vs_time_{label}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_filename.name}")
        plt.close(fig)

