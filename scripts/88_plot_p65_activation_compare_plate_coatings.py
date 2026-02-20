# %% [markdown]
# # Compare p65 Activation Across Different Plate Coatings
# This script reads plate coating comparison data, calculates p65 activation
# (ratio of median nucleus to median cytosol p65 signal per cell), and
# visualizes how activation changes over time across different treatment
# conditions and plate coatings.

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
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
sns.set(font_scale=1.5)

# %% [markdown]
# ## 1. Setup Directories and Load Comparison Metadata

# %%
# Setup paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'
FIGURES_ROOT = PROJECT_ROOT / 'figures'

# Helper function to strip absolute path prefixes
STRIP_PREFIXES = [
    r'Y:\coskun-lab\Nicky\\',
    r'Y:/coskun-lab/Nicky/',
    r'/coskun-lab/Nicky/',
]

def strip_prefix(path_str):
    """Strip known absolute prefixes to get relative path."""
    path_normalized = str(path_str).replace('\\', '/')
    for prefix in STRIP_PREFIXES:
        prefix_normalized = prefix.replace('\\', '/')
        if path_normalized.startswith(prefix_normalized):
            return path_normalized[len(prefix_normalized):]
    return path_str

# --- USER INPUTS ---
# Path to the comparison Excel file
comparison_file = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '08 compare plate coatings' / '01_compare_plate_coatings.xlsx'

# Output directory for plots
plot_output_dir = FIGURES_ROOT / '88_plot_p65_activation_compare_plate_coatings'
plot_output_dir.mkdir(exist_ok=True, parents=True)

print(f"Plots will be saved to: {plot_output_dir}")

# Load the comparison metadata
comparison_df = pd.read_excel(comparison_file)
print("\nPlate coating comparison data:")
print(comparison_df)


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


# %% [markdown]
# ## 3. Define the PKL File Processing Function

# %%
def calculate_p65_activation_per_cell(df):
    """
    Calculate p65 activation as the ratio of median nucleus to median cytosol p65 signal per cell.

    Parameters:
    -----------
    df : pandas.DataFrame
        Single-cell data with columns: CellLabel, CellRegion, p65

    Returns:
    --------
    pandas.DataFrame
        DataFrame with CellLabel and p65_activation columns
    """
    # Calculate median p65 signal for nucleus and cytosol per cell
    activation_data = []

    for cell_id, cell_df in df.groupby('CellLabel'):
        # Get nucleus median
        nucleus_data = cell_df[cell_df['CellRegion'] == 'Nucleus']['p65']
        cytosol_data = cell_df[cell_df['CellRegion'] == 'Cytosol']['p65']

        # Only calculate if both nucleus and cytosol have data
        if len(nucleus_data) > 0 and len(cytosol_data) > 0:
            nucleus_median = nucleus_data.median()
            cytosol_median = cytosol_data.median()

            # Avoid division by zero
            if cytosol_median > 0:
                activation = nucleus_median / cytosol_median
                activation_data.append({
                    'CellLabel': cell_id,
                    'p65_activation': activation,
                    'nucleus_median': nucleus_median,
                    'cytosol_median': cytosol_median
                })

    return pd.DataFrame(activation_data)


def process_pkl_file(file_path, layout_map, coating, plate_number):
    """
    Reads a single PKL file, calculates p65 activation per cell, and returns a dataframe.
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

    # Calculate p65 activation per cell
    activation_df = calculate_p65_activation_per_cell(df)

    if activation_df.empty:
        return None

    # Add metadata
    condition_info = layout_map.get(well_id)
    if condition_info:
        activation_df['Condition'] = condition_info['Condition']
        activation_df['Timepoint'] = condition_info['Timepoint']
    else:
        activation_df['Condition'] = 'Unknown'
        activation_df['Timepoint'] = -1

    activation_df['Well'] = well_id
    activation_df['FOV'] = file_path.stem
    activation_df['Coating'] = coating
    activation_df['Plate'] = plate_number

    return activation_df


# %% [markdown]
# ## 4. Process All Plates and PKL Files

# %%
all_plates_data = []

for idx, row in comparison_df.iterrows():
    plate_number = row['Plate number']
    coating = row['Coating']
    # Transform absolute paths to relative paths
    pkl_path_rel = strip_prefix(row['PklPath'])
    pkl_path = DATA_ROOT / pkl_path_rel
    layout_file_rel = strip_prefix(row['LayoutFile'])
    layout_file = DATA_ROOT / layout_file_rel

    print(f"\n{'='*60}")
    print(f"Processing Plate {plate_number} - Coating: {coating}")
    print(f"{'='*60}")

    # Parse layout
    layout_df = parse_layout(layout_file)
    if layout_df.empty:
        print(f"  Skipping plate {plate_number} due to layout parsing error.")
        continue

    print(f"  Layout parsed successfully.")

    # Get PKL files
    all_pkl_files = list(pkl_path.glob("*.pkl"))
    print(f"  Found {len(all_pkl_files)} PKL files.")

    if len(all_pkl_files) == 0:
        print(f"  Skipping plate {plate_number} - no PKL files found.")
        continue

    # Create layout map
    layout_map = layout_df.set_index('Well').to_dict('index')

    # Process files in parallel
    results_list = joblib.Parallel(n_jobs=10, verbose=5)(
        joblib.delayed(process_pkl_file)(f, layout_map, coating, plate_number)
        for f in tqdm(all_pkl_files, desc=f"  Processing Plate {plate_number}")
    )

    # Combine results
    plate_data = pd.concat([df for df in results_list if df is not None], ignore_index=True)

    if not plate_data.empty:
        all_plates_data.append(plate_data)
        print(f"  Processed {len(plate_data)} cells from plate {plate_number}.")
        print(f"  Cell counts per condition:")
        print(plate_data.groupby(['Condition', 'Timepoint']).agg({'CellLabel': 'nunique'}))
    else:
        print(f"  No data collected for plate {plate_number}.")

# Combine all plates
if all_plates_data:
    combined_df = pd.concat(all_plates_data, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"TOTAL DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total cells processed: {len(combined_df)}")
    print(f"\nCells per coating:")
    print(combined_df.groupby('Coating').agg({'CellLabel': 'count'}))
    print(f"\nCells per condition:")
    print(combined_df.groupby('Condition').agg({'CellLabel': 'count'}))
else:
    print("\nNo data was collected. Please check the paths and file formats.")
    combined_df = pd.DataFrame()


# %% [markdown]
# ## 5. Generate Plots Comparing p65 Activation

# %%
if not combined_df.empty:
    # Define condition order and colors
    condition_order = [c for c in ['TNFa', 'IL1B', 'DMSO Control']
                      if c in combined_df['Condition'].unique()]
    control_condition = 'DMSO Control'

    # Get coating order
    coating_order = combined_df['Coating'].unique().tolist()

    # Color palettes
    condition_palette = sns.color_palette("deep", len(condition_order))
    coating_palette = sns.color_palette("Set2", len(coating_order))

    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS")
    print(f"{'='*60}")
    print(f"Conditions: {condition_order}")
    print(f"Coatings: {coating_order}")

    # %% [markdown]
    # ### Plot 1: p65 Activation Over Time - Separate Plots per Coating

    # %%
    for coating in coating_order:
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        coating_data = combined_df[combined_df['Coating'] == coating]

        sns.barplot(
            data=coating_data,
            x='Timepoint',
            y='p65_activation',
            hue='Condition',
            hue_order=condition_order,
            palette=condition_palette,
            errorbar='se',
            capsize=0.1,
            ax=ax
        )

        ax.set_title(f"p65 Activation Over Time\nCoating: {coating}")
        ax.set_ylabel("p65 Activation\n(Nucleus/Cytosol Ratio)")
        ax.set_xlabel("Timepoint (minutes)")
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plot_filename = plot_output_dir / f"barplot_p65_activation_{coating.replace(' ', '_')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 2: p65 Activation Over Time - Compare Coatings per Condition

    # %%
    for condition in [c for c in condition_order if c != control_condition]:
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        condition_data = combined_df[combined_df['Condition'] == condition]

        sns.barplot(
            data=condition_data,
            x='Timepoint',
            y='p65_activation',
            hue='Coating',
            hue_order=coating_order,
            palette=coating_palette,
            errorbar='se',
            capsize=0.1,
            ax=ax
        )

        ax.set_title(f"p65 Activation Over Time - {condition}\nComparing Plate Coatings")
        ax.set_ylabel("p65 Activation\n(Nucleus/Cytosol Ratio)")
        ax.set_xlabel("Timepoint (minutes)")
        ax.legend(title='Coating', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plot_filename = plot_output_dir / f"barplot_p65_activation_{condition.replace(' ', '_')}_compare_coatings.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 3: Spline Plots - Condition vs Control per Coating

    # %%
    for coating in coating_order:
        coating_data = combined_df[combined_df['Coating'] == coating]

        for condition in [c for c in condition_order if c != control_condition]:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('p65 Activation\n(Nucleus/Cytosol Ratio)')
            ax.set_title(f"p65 Activation - {coating}\n{condition} vs {control_condition}")

            # Plot treated condition
            dfSub = coating_data[coating_data['Condition'] == condition].copy()
            dfSub = dfSub.dropna(subset=['p65_activation'])

            if len(dfSub) > 0:
                grouped = dfSub.groupby('Timepoint')['p65_activation']
                mean_vals = grouped.mean().reset_index()
                std_vals = grouped.std().reset_index()

                ax.scatter(mean_vals['Timepoint'], mean_vals['p65_activation'],
                          color='blue', alpha=1, s=80, zorder=3, label=f'{condition} (data)')
                ax.errorbar(mean_vals['Timepoint'], mean_vals['p65_activation'],
                           yerr=std_vals['p65_activation'], fmt='o', color='blue',
                           alpha=0.5, capsize=5, zorder=2)

                if len(mean_vals) >= 3:
                    try:
                        func = scipy.interpolate.interp1d(
                            x=mean_vals['Timepoint'],
                            y=mean_vals['p65_activation'],
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
            dfCtrl = coating_data[coating_data['Condition'] == control_condition].copy()
            dfCtrl = dfCtrl.dropna(subset=['p65_activation'])

            if len(dfCtrl) > 0:
                grouped = dfCtrl.groupby('Timepoint')['p65_activation']
                mean_vals = grouped.mean().reset_index()
                std_vals = grouped.std().reset_index()

                ax.scatter(mean_vals['Timepoint'], mean_vals['p65_activation'],
                          color='gray', alpha=0.7, s=80, zorder=3, label=f'{control_condition} (data)')
                ax.errorbar(mean_vals['Timepoint'], mean_vals['p65_activation'],
                           yerr=std_vals['p65_activation'], fmt='o', color='gray',
                           alpha=0.5, capsize=5, zorder=2)

                if len(mean_vals) >= 3:
                    try:
                        func = scipy.interpolate.interp1d(
                            x=mean_vals['Timepoint'],
                            y=mean_vals['p65_activation'],
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

            safe_coating = coating.replace(' ', '_')
            safe_condition = condition.replace(' ', '_')
            plot_filename = plot_output_dir / f"spline_p65_activation_{safe_coating}_{safe_condition}_vs_control.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 4: Spline Plots - Compare Coatings per Condition

    # %%
    for condition in [c for c in condition_order if c != control_condition]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('p65 Activation\n(Nucleus/Cytosol Ratio)')
        ax.set_title(f"p65 Activation - {condition}\nComparing Plate Coatings")

        colors = sns.color_palette("Set2", len(coating_order))

        for idx, coating in enumerate(coating_order):
            coating_data = combined_df[
                (combined_df['Coating'] == coating) &
                (combined_df['Condition'] == condition)
            ].copy()
            coating_data = coating_data.dropna(subset=['p65_activation'])

            if len(coating_data) > 0:
                grouped = coating_data.groupby('Timepoint')['p65_activation']
                mean_vals = grouped.mean().reset_index()
                std_vals = grouped.std().reset_index()

                ax.scatter(mean_vals['Timepoint'], mean_vals['p65_activation'],
                          color=colors[idx], alpha=1, s=80, zorder=3,
                          label=f'{coating} (data)')
                ax.errorbar(mean_vals['Timepoint'], mean_vals['p65_activation'],
                           yerr=std_vals['p65_activation'], fmt='o',
                           color=colors[idx], alpha=0.5, capsize=5, zorder=2)

                if len(mean_vals) >= 3:
                    try:
                        func = scipy.interpolate.interp1d(
                            x=mean_vals['Timepoint'],
                            y=mean_vals['p65_activation'],
                            kind='cubic'
                        )
                        timeSmooth = np.linspace(
                            mean_vals['Timepoint'].min(),
                            mean_vals['Timepoint'].max(),
                            1000
                        )
                        spline = func(timeSmooth)
                        ax.plot(timeSmooth, spline, color=colors[idx],
                               alpha=1, linewidth=2.5,
                               label=f'{coating} (spline)', zorder=1)
                    except Exception as e:
                        print(f"Could not fit spline for {coating}: {e}")

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        safe_condition = condition.replace(' ', '_')
        plot_filename = plot_output_dir / f"spline_p65_activation_compare_coatings_{safe_condition}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 5: Boxplots per Coating and Condition

    # %%
    for coating in coating_order:
        coating_data = combined_df[combined_df['Coating'] == coating]

        for condition in condition_order:
            dfSub = coating_data[coating_data['Condition'] == condition].copy()
            dfSub = dfSub.dropna(subset=['p65_activation'])

            if len(dfSub) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('p65 Activation\n(Nucleus/Cytosol Ratio)')
            ax.set_title(f"p65 Activation - {coating} - {condition}\n(Boxplot with Median Spline)")

            timepoints = sorted(dfSub['Timepoint'].unique())
            bins, groups = zip(*dfSub.groupby('Timepoint')['p65_activation'])

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
            median = dfSub.groupby('Timepoint')['p65_activation'].median().reset_index()

            if len(median) >= 3:
                try:
                    func = scipy.interpolate.interp1d(
                        x=median['Timepoint'],
                        y=median['p65_activation'],
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
                    print(f"Could not fit spline for boxplot {coating} - {condition}: {e}")

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            safe_coating = coating.replace(' ', '_')
            safe_condition = condition.replace(' ', '_')
            plot_filename = plot_output_dir / f"boxplot_p65_activation_{safe_coating}_{safe_condition}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 6: Heatmap - Mean p65 Activation by Coating, Condition, and Time

    # %%
    # Create a pivot table for heatmap
    for coating in coating_order:
        coating_data = combined_df[combined_df['Coating'] == coating]

        pivot_data = coating_data.pivot_table(
            values='p65_activation',
            index='Condition',
            columns='Timepoint',
            aggfunc='mean'
        )

        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Mean p65 Activation'},
                ax=ax
            )

            ax.set_title(f"Mean p65 Activation Heatmap\nCoating: {coating}")
            ax.set_xlabel("Timepoint (minutes)")
            ax.set_ylabel("Condition")

            plt.tight_layout()
            safe_coating = coating.replace(' ', '_')
            plot_filename = plot_output_dir / f"heatmap_p65_activation_{safe_coating}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 7: Single-Cell p65 Activation at 45 min - Compare Coatings by Condition

    # %%
    # Filter data for 45 min timepoint only
    df_45min = combined_df[combined_df['Timepoint'] == 45].copy()

    if not df_45min.empty:
        # Create a combined plot with all conditions
        fig, axes = plt.subplots(1, len(condition_order), figsize=(18, 6), dpi=300, sharey=True)

        if len(condition_order) == 1:
            axes = [axes]

        for idx, condition in enumerate(condition_order):
            ax = axes[idx]

            # Filter data for this condition
            condition_data = df_45min[df_45min['Condition'] == condition]

            # Create violin plot with individual points
            sns.violinplot(
                data=condition_data,
                x='Coating',
                y='p65_activation',
                order=coating_order,
                palette=coating_palette,
                inner='box',
                ax=ax
            )

            # Overlay strip plot to show individual cells
            sns.stripplot(
                data=condition_data,
                x='Coating',
                y='p65_activation',
                order=coating_order,
                color='black',
                alpha=0.2,
                size=2,
                ax=ax
            )

            # Add statistical annotations (Mann-Whitney U test)
            # Get pairwise comparisons
            y_max = condition_data['p65_activation'].max()
            y_range = condition_data['p65_activation'].max() - condition_data['p65_activation'].min()

            # Perform pairwise Mann-Whitney U tests
            pairs = list(combinations(range(len(coating_order)), 2))
            y_offset = y_max + 0.1 * y_range
            y_step = 0.15 * y_range

            for pair_idx, (i, j) in enumerate(pairs):
                coating1 = coating_order[i]
                coating2 = coating_order[j]

                data1 = condition_data[condition_data['Coating'] == coating1]['p65_activation'].values
                data2 = condition_data[condition_data['Coating'] == coating2]['p65_activation'].values

                if len(data1) > 0 and len(data2) > 0:
                    # Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

                    # Determine significance level
                    if p_value < 0.001:
                        sig_symbol = '***'
                    elif p_value < 0.01:
                        sig_symbol = '**'
                    elif p_value < 0.05:
                        sig_symbol = '*'
                    else:
                        sig_symbol = 'ns'

                    # Draw significance bar
                    y_pos = y_offset + pair_idx * y_step
                    ax.plot([i, i, j, j], [y_pos, y_pos + 0.02*y_range, y_pos + 0.02*y_range, y_pos],
                           'k-', linewidth=1.5)
                    ax.text((i + j) / 2, y_pos + 0.03*y_range, sig_symbol,
                           ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_title(f"{condition}\n(n={len(condition_data)} cells)")
            ax.set_xlabel("Plate Coating")
            if idx == 0:
                ax.set_ylabel("p65 Activation\n(Nucleus/Cytosol Ratio)")
            else:
                ax.set_ylabel("")

            # Rotate x-axis labels if needed
            ax.tick_params(axis='x', rotation=45)

            # Adjust y-axis limits to accommodate annotations
            ax.set_ylim(ax.get_ylim()[0], y_offset + len(pairs) * y_step + 0.1 * y_range)

        plt.suptitle("Single-Cell p65 Activation at 45 Minutes\nComparing Plate Coatings by Condition",
                     fontsize=16, y=1.02)
        plt.tight_layout()

        plot_filename = plot_output_dir / "violin_p65_activation_45min_compare_coatings_by_condition.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Also create a single plot with coating on x-axis and condition as hue
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        sns.violinplot(
            data=df_45min,
            x='Coating',
            y='p65_activation',
            hue='Condition',
            order=coating_order,
            hue_order=condition_order,
            palette=condition_palette,
            split=False,
            inner='box',
            ax=ax
        )

        ax.set_title(f"Single-Cell p65 Activation at 45 Minutes\nComparing Conditions Across Plate Coatings\n(n={len(df_45min)} total cells)")
        ax.set_xlabel("Plate Coating")
        ax.set_ylabel("p65 Activation\n(Nucleus/Cytosol Ratio)")
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        plot_filename = plot_output_dir / "violin_p65_activation_45min_coatings_with_conditions.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Create a statistical summary table for 45 min
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY AT 45 MINUTES")
        print("="*60)
        summary_45min = df_45min.groupby(['Coating', 'Condition'])['p65_activation'].agg([
            ('n_cells', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('sem', 'sem')
        ]).round(3)
        print(summary_45min)


        # %% [markdown]
        # ### Plot 8: Fraction of Activated Cells at 45 Minutes

        # %%
        # Define activated cells: p65_activation > 1.0 (nucleus > cytosol)
        # Define unactivated cells: p65_activation <= 1.0 (cytosol >= nucleus)
        df_45min['activation_status'] = df_45min['p65_activation'].apply(
            lambda x: 'Activated' if x > 1.0 else 'Unactivated'
        )

        # Calculate fraction of activated cells per coating and condition
        activation_summary = df_45min.groupby(['Coating', 'Condition', 'activation_status']).size().reset_index(name='count')

        # Calculate total cells per coating/condition
        total_cells = df_45min.groupby(['Coating', 'Condition']).size().reset_index(name='total')

        # Merge and calculate fraction
        activation_summary = activation_summary.merge(total_cells, on=['Coating', 'Condition'])
        activation_summary['fraction'] = activation_summary['count'] / activation_summary['total']

        # Get only activated cells for simpler plotting
        activated_fraction = activation_summary[activation_summary['activation_status'] == 'Activated'].copy()
        activated_fraction = activated_fraction.pivot(index='Condition', columns='Coating', values='fraction').fillna(0)
        activated_fraction = activated_fraction[coating_order]  # Ensure coating order

        print("\n" + "="*60)
        print("FRACTION OF ACTIVATED CELLS AT 45 MINUTES")
        print("(Activated = nucleus p65 > cytosol p65)")
        print("="*60)
        print(activated_fraction.round(3))

        # Calculate and print statistical comparisons (p-values)
        print("\n" + "="*60)
        print("STATISTICAL COMPARISONS (Mann-Whitney U Test)")
        print("p-values for p65 activation levels")
        print("="*60)

        for condition in condition_order:
            print(f"\n{condition}:")
            condition_data = df_45min[df_45min['Condition'] == condition]

            # Pairwise comparisons
            for i, coating1 in enumerate(coating_order):
                for j, coating2 in enumerate(coating_order):
                    if i < j:
                        data1 = condition_data[condition_data['Coating'] == coating1]['p65_activation'].values
                        data2 = condition_data[condition_data['Coating'] == coating2]['p65_activation'].values

                        if len(data1) > 0 and len(data2) > 0:
                            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                            print(f"  {coating1} vs {coating2}: p = {p_value:.4e} ({sig})")


        # Plot 1: Stacked bar plot showing activated vs unactivated fractions
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        # Prepare data for stacked bar plot
        x_pos = np.arange(len(coating_order))
        width = 0.25

        for idx, condition in enumerate(condition_order):
            activated_fracs = []
            unactivated_fracs = []

            for coating in coating_order:
                subset = activation_summary[
                    (activation_summary['Coating'] == coating) &
                    (activation_summary['Condition'] == condition)
                ]

                activated = subset[subset['activation_status'] == 'Activated']['fraction'].values
                activated_frac = activated[0] if len(activated) > 0 else 0

                activated_fracs.append(activated_frac)
                unactivated_fracs.append(1 - activated_frac)

            # Plot activated bars
            ax.bar(x_pos + idx * width, activated_fracs, width,
                   label=f'{condition}',
                   color=condition_palette[idx],
                   edgecolor='black', linewidth=1)

            # Add text labels on bars
            for i, (x, val) in enumerate(zip(x_pos + idx * width, activated_fracs)):
                if val > 0.05:  # Only show label if fraction > 5%
                    ax.text(x, val/2, f'{val:.2%}',
                           ha='center', va='center',
                           fontsize=10, fontweight='bold', color='white')

        ax.set_xlabel('Plate Coating')
        ax.set_ylabel('Fraction of Activated Cells')
        ax.set_title('Fraction of Activated Cells at 45 Minutes\n(Activated = Nucleus p65 > Cytosol p65)')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(coating_order)
        ax.set_ylim(0, 1)
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_filename = plot_output_dir / "barplot_activated_fraction_45min_by_coating.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Plot 2: Grouped bar plot by condition with statistical annotations
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        x_pos = np.arange(len(condition_order))
        width = 0.25

        for idx, coating in enumerate(coating_order):
            fractions = []

            for condition in condition_order:
                subset = activation_summary[
                    (activation_summary['Coating'] == coating) &
                    (activation_summary['Condition'] == condition) &
                    (activation_summary['activation_status'] == 'Activated')
                ]

                frac = subset['fraction'].values[0] if len(subset) > 0 else 0
                fractions.append(frac)

            ax.bar(x_pos + idx * width, fractions, width,
                   label=coating,
                   color=coating_palette[idx],
                   edgecolor='black', linewidth=1)

            # Add percentage labels
            for i, (x, val) in enumerate(zip(x_pos + idx * width, fractions)):
                ax.text(x, val + 0.02, f'{val:.1%}',
                       ha='center', va='bottom', fontsize=9)

        # Add statistical annotations comparing coatings for each condition
        y_base = 1.0
        y_step = 0.08

        for cond_idx, condition in enumerate(condition_order):
            # Get activation data for each coating for this condition
            coating_pairs = list(combinations(range(len(coating_order)), 2))

            for pair_idx, (i, j) in enumerate(coating_pairs):
                coating1 = coating_order[i]
                coating2 = coating_order[j]

                # Get the binary activation status for each coating
                data1 = df_45min[(df_45min['Condition'] == condition) &
                                 (df_45min['Coating'] == coating1)]['activation_status'].apply(
                    lambda x: 1 if x == 'Activated' else 0
                ).values
                data2 = df_45min[(df_45min['Condition'] == condition) &
                                 (df_45min['Coating'] == coating2)]['activation_status'].apply(
                    lambda x: 1 if x == 'Activated' else 0
                ).values

                if len(data1) > 0 and len(data2) > 0:
                    # Fisher's exact test for proportions (more appropriate than Mann-Whitney for binary data)
                    from scipy.stats import fisher_exact

                    # Create contingency table
                    activated1 = np.sum(data1)
                    total1 = len(data1)
                    activated2 = np.sum(data2)
                    total2 = len(data2)

                    contingency_table = [[activated1, total1 - activated1],
                                        [activated2, total2 - activated2]]

                    _, p_value = fisher_exact(contingency_table)

                    # Determine significance level
                    if p_value < 0.001:
                        sig_symbol = '***'
                    elif p_value < 0.01:
                        sig_symbol = '**'
                    elif p_value < 0.05:
                        sig_symbol = '*'
                    else:
                        sig_symbol = 'ns'

                    # Draw significance bar
                    x1 = cond_idx + i * width
                    x2 = cond_idx + j * width
                    y_pos = y_base + pair_idx * y_step

                    ax.plot([x1, x1, x2, x2],
                           [y_pos - 0.01, y_pos, y_pos, y_pos - 0.01],
                           'k-', linewidth=1.2)
                    ax.text((x1 + x2) / 2, y_pos + 0.01, sig_symbol,
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Condition')
        ax.set_ylabel('Fraction of Activated Cells')
        ax.set_title('Fraction of Activated Cells at 45 Minutes by Condition\n(Activated = Nucleus p65 > Cytosol p65)')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(condition_order)
        ax.set_ylim(0, 1.35)  # Extended to accommodate annotations
        ax.legend(title='Coating', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_filename = plot_output_dir / "barplot_activated_fraction_45min_by_condition.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Plot 3: Heatmap of activation fractions
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Prepare data for heatmap
        heatmap_data = activated_fraction.T  # Transpose so coatings are rows

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Fraction Activated'},
            linewidths=1,
            linecolor='gray',
            ax=ax
        )

        ax.set_title('Fraction of Activated Cells at 45 Minutes\n(Activated = Nucleus p65 > Cytosol p65)')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Plate Coating')

        plt.tight_layout()
        plot_filename = plot_output_dir / "heatmap_activated_fraction_45min.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Plot 4: Stacked bar showing activated vs unactivated proportions
        fig, axes = plt.subplots(1, len(coating_order), figsize=(16, 6), dpi=300, sharey=True)

        for idx, coating in enumerate(coating_order):
            ax = axes[idx]

            coating_data = activation_summary[activation_summary['Coating'] == coating]

            # Prepare data
            activated_vals = []
            unactivated_vals = []

            for condition in condition_order:
                cond_data = coating_data[coating_data['Condition'] == condition]

                activated = cond_data[cond_data['activation_status'] == 'Activated']['fraction'].values
                activated_val = activated[0] if len(activated) > 0 else 0

                activated_vals.append(activated_val)
                unactivated_vals.append(1 - activated_val)

            x_pos = np.arange(len(condition_order))

            # Create stacked bars
            ax.bar(x_pos, activated_vals, label='Activated',
                   color='#d62728', edgecolor='black', linewidth=1)
            ax.bar(x_pos, unactivated_vals, bottom=activated_vals,
                   label='Unactivated', color='#1f77b4',
                   edgecolor='black', linewidth=1)

            # Add percentage labels
            for i, (act, unact) in enumerate(zip(activated_vals, unactivated_vals)):
                if act > 0.05:
                    ax.text(i, act/2, f'{act:.0%}',
                           ha='center', va='center',
                           fontsize=11, fontweight='bold', color='white')
                if unact > 0.05:
                    ax.text(i, act + unact/2, f'{unact:.0%}',
                           ha='center', va='center',
                           fontsize=11, fontweight='bold', color='white')

            ax.set_title(f'{coating}')
            ax.set_xlabel('Condition')
            if idx == 0:
                ax.set_ylabel('Fraction of Cells')
                ax.legend(loc='upper left')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(condition_order, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Activated vs Unactivated Cells at 45 Minutes by Coating\n(Activated = Nucleus p65 > Cytosol p65)',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        plot_filename = plot_output_dir / "stacked_barplot_activation_status_45min.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)

    print(f"\n{'='*60}")
    print(f"ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Plots saved to: {plot_output_dir}")

else:
    print("\nNo data available to generate plots.")

# %%
print("\nScript execution complete!")
