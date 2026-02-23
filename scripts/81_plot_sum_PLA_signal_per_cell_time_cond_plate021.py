# %% [markdown]
# # Plot Sum of PLA Dots per Cell Over Time and Treatment Condition
# This script reads single-cell data exported from Script 71, extracts PLA dot counts per cell,
# and visualizes how PLA signals change over time across different treatment conditions.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import re
from datetime import datetime
import scipy.interpolate

sns.set_style('whitegrid')
sns.set(font_scale=1.5)

# %% [markdown]
# ## 1. Setup Directories and Load Metadata

# %%
# --- USER INPUTS ---
# Get the script directory and define project root (parent directory)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Path to the channels Excel file (in the data folder)
channels_path = project_root / "data" / "1Jan2026_Plate021_multiplex.xlsx"
# Path to the plate layout file (in the data folder)
layout_path = project_root / "data" / "Plate021_layout.xlsx"

# --- DERIVED PATHS ---
try:
    channels = pd.read_excel(channels_path)
    channels.dropna(subset=['StitchPath'], inplace=True)
    basePath = Path(channels['StitchPath'].iloc[-1]).parent
    # This is the folder containing the PKL files from Script 71
    pklPath = basePath / '10 PKL single cell'
    # This is the folder where plots will be saved (using figures folder in project root)
    plotPath = project_root / 'figures'
    plotPath.mkdir(exist_ok=True)
    assert pklPath.exists(), f"PKL path not found: {pklPath}"
    print(f"Found PKL data path: {pklPath}")
    print(f"Plots will be saved to: {plotPath}")
except Exception as e:
    print(f"Error setting up paths: {e}")
    print("Please ensure the channels Excel file path is correct.")
    # As a fallback for demonstration, we can create a dummy directory
    pklPath = Path('./10_PKL_single_cell_dummy')
    pklPath.mkdir(exist_ok=True)
    plotPath = Path('./figures_dummy')
    plotPath.mkdir(exist_ok=True)


# %% [markdown]
# ## 2. Load and Process Plate Layout Information

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

# %%
def process_pkl_file(file_path, layout_map):
    """
    Reads a single PKL file, sums PLA dots per cell, and returns a dataframe.
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

    # --- Sum all PLA signals per cell ---
    pla_cols = [col for col in df.columns if ' Dots' in col]

    if not pla_cols:
        return None  # Skip files without PLA signals

    # Sum PLA dots per cell
    pla_sums = df.groupby('CellLabel')[pla_cols].sum()

    if pla_sums.empty:
        return None

    # Add metadata
    condition_info = layout_map.get(well_id)
    if condition_info:
        pla_sums['Condition'] = condition_info['Condition']
        pla_sums['Timepoint'] = condition_info['Timepoint']
    else:
        pla_sums['Condition'] = 'Unknown'
        pla_sums['Timepoint'] = -1

    pla_sums['Well'] = well_id
    pla_sums['FOV'] = file_path.stem
    pla_sums.reset_index(inplace=True)

    return pla_sums


# %% [markdown]
# ## 4. Process All PKL Files in Parallel

# %%
all_pkl_files = list(pklPath.glob("*.pkl"))
print(f"Found {len(all_pkl_files)} PKL files to process.")

# Create a dictionary from the layout dataframe for faster lookups
layout_map = layout_df.set_index('Well').to_dict('index')

# Use joblib to process files in parallel
# Adjust n_jobs based on your system (use -1 for all cores)
results_list = joblib.Parallel(n_jobs=10, verbose=10)(
    joblib.delayed(process_pkl_file)(f, layout_map) for f in tqdm(all_pkl_files, desc="Processing PKL files")
)

# Combine all dataframes into one
# Filter out any None results from failed processing
all_data_df = pd.concat([df for df in results_list if df is not None], ignore_index=True)
print("\nProcessing complete. Shape of the combined dataframe:", all_data_df.shape)
print(all_data_df.head())

# Display summary statistics
print("\nCell counts per condition and timepoint:")
print(all_data_df.groupby(['Condition', 'Timepoint']).agg({'CellLabel': 'nunique'}))


# %% [markdown]
# ## 5. Generate Plots for Each PLA Marker

# %%
# Get the list of PLA markers to plot
pla_markers = [col for col in all_data_df.columns if ' Dots' in col]

if not pla_markers or all_data_df.empty:
    print("\nNo PLA markers found or data is empty. Cannot generate plots.")
else:
    print(f"\nGenerating plots for {len(pla_markers)} PLA markers...")

    # Define condition order and colors
    condition_order = [c for c in ['TNFa', 'IL1B', 'DMSO Control']
                      if c in all_data_df['Condition'].unique()]
    control_condition = 'DMSO Control'
    palette = sns.color_palette("deep", len(condition_order))

    for marker in pla_markers:
        marker_name = marker.replace(' Dots', '')
        marker_name_clean = marker_name.replace(' ', '_')

        # --- PLOT 1: Bar Plot with Error Bars ---
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        sns.barplot(
            data=all_data_df,
            x='Timepoint',
            y=marker,
            hue='Condition',
            hue_order=condition_order,
            palette=palette,
            errorbar='se',  # Show standard error
            capsize=0.1,
            ax=ax
        )

        ax.set_title(f"{marker_name} Signal Over Time\n(Sum of Dots per Cell)")
        ax.set_ylabel(f"{marker_name}\n(Sum of Dots\nper Cell)")
        ax.set_xlabel("Timepoint (minutes)")
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Save the figure
        plot_filename = plotPath / f"barplot_{marker_name_clean}_sum_dots_vs_time.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved bar plot to: {plot_filename}")

        plt.close(fig)


        # --- PLOT 2: Line Plot with Spline for Each Condition vs Control ---
        for condition in [c for c in condition_order if c != control_condition]:

            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(f"{marker_name}\n(Sum of Dots\nper Cell)")
            ax.set_title(f"{marker_name} Signal\n{condition} vs {control_condition}")

            # Plot treated condition
            dfSub = all_data_df[all_data_df['Condition'] == condition].copy()
            dfSub = dfSub.dropna(subset=[marker])

            if len(dfSub) > 0:
                # Calculate mean and std
                grouped = dfSub.groupby('Timepoint')[marker]
                mean_vals = grouped.mean().reset_index()
                std_vals = grouped.std().reset_index()

                # Scatter with error bars
                ax.scatter(mean_vals['Timepoint'], mean_vals[marker],
                          color='blue', alpha=1, s=80, zorder=3, label=f'{condition} (data)')
                ax.errorbar(mean_vals['Timepoint'], mean_vals[marker],
                           yerr=std_vals[marker], fmt='o', color='blue',
                           alpha=0.5, capsize=5, zorder=2)

                # Spline interpolation
                if len(mean_vals) >= 3:
                    try:
                        func = scipy.interpolate.interp1d(
                            x=mean_vals['Timepoint'],
                            y=mean_vals[marker],
                            kind='cubic'
                        )
                        timeSmooth = np.linspace(
                            mean_vals['Timepoint'].min(),
                            mean_vals['Timepoint'].max(),
                            1000
                        )
                        spline = func(timeSmooth)
                        ax.plot(timeSmooth, spline, color='blue',
                               alpha=1, linewidth=2.5, label=f'{condition} (spline)', zorder=1)
                    except Exception as e:
                        print(f"Could not fit spline for {condition}: {e}")

            # Plot control condition
            dfCtrl = all_data_df[all_data_df['Condition'] == control_condition].copy()
            dfCtrl = dfCtrl.dropna(subset=[marker])

            if len(dfCtrl) > 0:
                # Calculate mean and std
                grouped = dfCtrl.groupby('Timepoint')[marker]
                mean_vals = grouped.mean().reset_index()
                std_vals = grouped.std().reset_index()

                # Scatter with error bars
                ax.scatter(mean_vals['Timepoint'], mean_vals[marker],
                          color='gray', alpha=0.7, s=80, zorder=3, label=f'{control_condition} (data)')
                ax.errorbar(mean_vals['Timepoint'], mean_vals[marker],
                           yerr=std_vals[marker], fmt='o', color='gray',
                           alpha=0.5, capsize=5, zorder=2)

                # Spline interpolation
                if len(mean_vals) >= 3:
                    try:
                        func = scipy.interpolate.interp1d(
                            x=mean_vals['Timepoint'],
                            y=mean_vals[marker],
                            kind='cubic'
                        )
                        timeSmooth = np.linspace(
                            mean_vals['Timepoint'].min(),
                            mean_vals['Timepoint'].max(),
                            1000
                        )
                        spline = func(timeSmooth)
                        ax.plot(timeSmooth, spline, color='gray',
                               linestyle='--', alpha=0.7, linewidth=2.5,
                               label=f'{control_condition} (spline)', zorder=1)
                    except Exception as e:
                        print(f"Could not fit spline for {control_condition}: {e}")

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save figure
            safe_condition = condition.replace(' ', '_')
            plot_filename = plotPath / f"spline_{marker_name_clean}_{safe_condition}_vs_control.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved spline plot to: {plot_filename}")

            plt.close(fig)


        # --- PLOT 3: Boxplot with Spline Overlay for Each Condition ---
        for condition in condition_order:

            dfSub = all_data_df[all_data_df['Condition'] == condition].copy()
            dfSub = dfSub.dropna(subset=[marker])

            if len(dfSub) == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(f"{marker_name}\n(Sum of Dots\nper Cell)")
            ax.set_title(f"{marker_name} Signal - {condition}\n(Boxplot with Median Spline)")

            # Create boxplot
            timepoints = sorted(dfSub['Timepoint'].unique())
            bins, groups = zip(*dfSub.groupby('Timepoint')[marker])

            bp = ax.boxplot(
                groups,
                positions=timepoints,
                widths=5,
                patch_artist=True,
                boxprops={'facecolor': [1, 1, 1, 0.5], 'edgecolor': 'blue', 'linewidth': 1.5},
                medianprops={'color': 'red', 'linewidth': 2},
                flierprops={'markersize': 3, 'marker': 'o', 'color': 'blue', 'alpha': 0.5},
                showfliers=True
            )

            # Overlay spline of median
            median = dfSub.groupby('Timepoint')[marker].median().reset_index()

            if len(median) >= 3:
                try:
                    func = scipy.interpolate.interp1d(
                        x=median['Timepoint'],
                        y=median[marker],
                        kind='cubic'
                    )
                    timeSmooth = np.linspace(
                        median['Timepoint'].min(),
                        median['Timepoint'].max(),
                        1000
                    )
                    spline = func(timeSmooth)
                    ax.plot(timeSmooth, spline, color='red',
                           linewidth=2.5, alpha=0.9, label='Median Spline')
                except Exception as e:
                    print(f"Could not fit spline for boxplot {condition}: {e}")

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save figure
            safe_condition = condition.replace(' ', '_')
            plot_filename = plotPath / f"boxplot_{marker_name_clean}_{safe_condition}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved boxplot to: {plot_filename}")

            plt.close(fig)


print("\nAll plots generated successfully!")
print(f"Plots saved to: {plotPath}")
