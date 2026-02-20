# %% [markdown]
# # Plot p65 Activation Over Time - Plate 021
# This script reads Plate 021 data, calculates p65 activation
# (ratio of median nucleus to median cytosol p65 signal per cell), and
# visualizes how activation changes over time across different treatment
# conditions.

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
sns.set_context("notebook", font_scale=1.5)

# %% [markdown]
# ## 1. Setup Directories and Load Plate Layout

# %%
# Setup paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'
FIGURES_ROOT = PROJECT_ROOT / 'figures'

# --- USER INPUTS ---
# Path to the plate layout Excel file
layout_file = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P11 24 well plate 021' / 'Plate021_layout.xlsx'

# Path to PKL files directory
pkl_path = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P11 24 well plate 021' / '26Dec2025 cycle 3 PLA' / '10 PKL single cell'

# Output directory for plots
plot_output_dir = FIGURES_ROOT / '90_plot_p65_activation_plate021'
plot_output_dir.mkdir(exist_ok=True, parents=True)

print(f"PKL files directory: {pkl_path}")
print(f"Plots will be saved to: {plot_output_dir}")


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
    Calculate p65 activation as the ratio of mean nucleus to mean cytosol p65 signal per cell.

    Parameters:
    -----------
    df : pandas.DataFrame
        Single-cell data with columns: CellLabel, CellRegion, p65

    Returns:
    --------
    pandas.DataFrame
        DataFrame with CellLabel and p65_activation columns
    """
    # Calculate mean p65 signal for nucleus and cytosol per cell
    activation_data = []

    for cell_id, cell_df in df.groupby('CellLabel'):
        # Get nucleus and cytosol p65
        nucleus_data = cell_df[cell_df['CellRegion'] == 'Nucleus']['p65']
        cytosol_data = cell_df[cell_df['CellRegion'] == 'Cytosol']['p65']

        # Only calculate if both nucleus and cytosol have data
        if len(nucleus_data) > 0 and len(cytosol_data) > 0:
            nucleus_mean = nucleus_data.mean()
            cytosol_mean = cytosol_data.mean()

            # Avoid division by zero
            if cytosol_mean > 0:
                activation = nucleus_mean / cytosol_mean
                activation_data.append({
                    'CellLabel': cell_id,
                    'p65_activation': activation,
                    'nucleus_mean': nucleus_mean,
                    'cytosol_mean': cytosol_mean
                })

    return pd.DataFrame(activation_data)


def process_pkl_file(file_path, layout_map):
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

    return activation_df


# %% [markdown]
# ## 4. Process PKL Files

# %%
print(f"\n{'='*60}")
print(f"Processing Plate 021")
print(f"{'='*60}")

# Parse layout
layout_df = parse_layout(layout_file)
if layout_df.empty:
    print(f"  Error: Could not parse layout file.")
    raise ValueError("Layout file parsing failed.")

print(f"  Layout parsed successfully.")
print(f"  Found {len(layout_df)} wells in layout.")

# Get PKL files
all_pkl_files = list(pkl_path.glob("*.pkl"))
print(f"  Found {len(all_pkl_files)} PKL files.")

if len(all_pkl_files) == 0:
    print(f"  Error: No PKL files found.")
    raise ValueError("No PKL files found in the specified directory.")

# Create layout map
layout_map = layout_df.set_index('Well').to_dict('index')

# Process files in parallel
results_list = joblib.Parallel(n_jobs=4, verbose=5)(
    joblib.delayed(process_pkl_file)(f, layout_map)
    for f in tqdm(all_pkl_files, desc=f"  Processing PKL files")
)

# Combine results
combined_df = pd.concat([df for df in results_list if df is not None], ignore_index=True)

if not combined_df.empty:
    print(f"\n{'='*60}")
    print(f"DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total cells processed: {len(combined_df)}")
    print(f"\nCells per condition:")
    print(combined_df.groupby('Condition').agg({'CellLabel': 'count'}))
    print(f"\nCells per timepoint:")
    print(combined_df.groupby('Timepoint').agg({'CellLabel': 'count'}))
    print(f"\nCells per condition and timepoint:")
    print(combined_df.groupby(['Condition', 'Timepoint']).agg({'CellLabel': 'nunique'}))
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

    # Color palettes
    condition_palette = sns.color_palette("deep", len(condition_order))

    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS")
    print(f"{'='*60}")
    print(f"Conditions: {condition_order}")

    # %% [markdown]
    # ### Plot 1: p65 Activation Over Time - All Conditions

    # %%
    fig, ax = plt.subplots()

    sns.barplot(
        data=combined_df,
        x='Timepoint',
        y='p65_activation',
        hue='Condition',
        hue_order=condition_order,
        palette=condition_palette,
        errorbar='se',
        capsize=0.1,
        ax=ax
    )

    ax.set_title("p65 Activation Over Time - Plate 021")
    ax.set_ylabel("p65 Activation\n(Nucleus/Cytosol Ratio)")
    ax.set_xlabel("Timepoint (minutes)")
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_filename = plot_output_dir / "barplot_p65_activation_over_time.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)


    # %% [markdown]
    # ### Plot 2: Spline Plots - Condition vs Control

    # %%
    for condition in [c for c in condition_order if c != control_condition]:
        fig, ax = plt.subplots()
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('p65 Activation\n(Nucleus/Cytosol Ratio)')
        ax.set_title(f"p65 Activation - Plate 021\n{condition} vs {control_condition}")

        # Plot treated condition
        dfSub = combined_df[combined_df['Condition'] == condition].copy()
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
        dfCtrl = combined_df[combined_df['Condition'] == control_condition].copy()
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

        safe_condition = condition.replace(' ', '_')
        plot_filename = plot_output_dir / f"spline_p65_activation_{safe_condition}_vs_control.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)




    # %% [markdown]
    # ### Plot 3: Boxplots per Condition

    # %%
    for condition in condition_order:
        dfSub = combined_df[combined_df['Condition'] == condition].copy()
        dfSub = dfSub.dropna(subset=['p65_activation'])

        if len(dfSub) == 0:
            continue

        fig, ax = plt.subplots()
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('p65 Activation\n(Nucleus/Cytosol Ratio)')
        ax.set_title(f"p65 Activation - Plate 021 - {condition}\n(Boxplot with Median Spline)")

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
                print(f"Could not fit spline for boxplot {condition}: {e}")

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        safe_condition = condition.replace(' ', '_')
        plot_filename = plot_output_dir / f"boxplot_p65_activation_{safe_condition}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 4: Heatmap - Mean p65 Activation by Condition and Time

    # %%
    # Create a pivot table for heatmap
    pivot_data = combined_df.pivot_table(
        values='p65_activation',
        index='Condition',
        columns='Timepoint',
        aggfunc='mean'
    )

    if not pivot_data.empty:
        fig, ax = plt.subplots()

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Mean p65 Activation'},
            ax=ax
        )

        ax.set_title("Mean p65 Activation Heatmap - Plate 021")
        ax.set_xlabel("Timepoint (minutes)")
        ax.set_ylabel("Condition")

        plt.tight_layout()
        plot_filename = plot_output_dir / "heatmap_p65_activation.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 5: Single-Cell p65 Activation at 45 min by Condition

    # %%
    # Filter data for 45 min timepoint only
    df_45min = combined_df[combined_df['Timepoint'] == 45].copy()

    if not df_45min.empty:
        # Create a combined plot with all conditions
        fig, axes = plt.subplots(1, len(condition_order), sharey=True)

        if len(condition_order) == 1:
            axes = [axes]

        for idx, condition in enumerate(condition_order):
            ax = axes[idx]

            # Filter data for this condition
            condition_data = df_45min[df_45min['Condition'] == condition]

            # Create violin plot with individual points
            sns.violinplot(
                data=condition_data,
                y='p65_activation',
                inner='box',
                ax=ax,
                color=condition_palette[idx]
            )

            # Overlay strip plot to show individual cells
            sns.stripplot(
                data=condition_data,
                y='p65_activation',
                color='black',
                alpha=0.2,
                size=2,
                ax=ax
            )

            ax.set_title(f"{condition}\n(n={len(condition_data)} cells)")
            if idx == 0:
                ax.set_ylabel("p65 Activation\n(Nucleus/Cytosol Ratio)")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

        plt.suptitle("Single-Cell p65 Activation at 45 Minutes - Plate 021",
                     fontsize=16, y=1.02)
        plt.tight_layout()

        plot_filename = plot_output_dir / "violin_p65_activation_45min_by_condition.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Also create a single plot with condition on x-axis
        fig, ax = plt.subplots()

        sns.violinplot(
            data=df_45min,
            x='Condition',
            y='p65_activation',
            hue='Condition',
            hue_order=condition_order,
            palette=condition_palette,
            split=False,
            inner='box',
            ax=ax,
            legend=False
        )

        ax.set_title(f"Single-Cell p65 Activation at 45 Minutes - Plate 021\n(n={len(df_45min)} total cells)")
        ax.set_xlabel("Condition")
        ax.set_ylabel("p65 Activation\n(Nucleus/Cytosol Ratio)")
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        plot_filename = plot_output_dir / "violin_p65_activation_45min_all_conditions.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Create a statistical summary table for 45 min
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY AT 45 MINUTES")
        print("="*60)
        summary_45min = df_45min.groupby('Condition')['p65_activation'].agg([
            ('n_cells', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('sem', 'sem')
        ]).round(3)
        print(summary_45min)


        # %% [markdown]
        # ### Plot 6: Fraction of Activated Cells at 45 Minutes

        # %%
        # Define activated cells: p65_activation > 1.0 (nucleus > cytosol)
        # Define unactivated cells: p65_activation <= 1.0 (cytosol >= nucleus)
        df_45min['activation_status'] = df_45min['p65_activation'].apply(
            lambda x: 'Activated' if x > 1.0 else 'Unactivated'
        )

        # Calculate fraction of activated cells per condition
        activation_summary = df_45min.groupby(['Condition', 'activation_status']).size().reset_index(name='count')

        # Calculate total cells per condition
        total_cells = df_45min.groupby('Condition').size().reset_index(name='total')

        # Merge and calculate fraction
        activation_summary = activation_summary.merge(total_cells, on='Condition')
        activation_summary['fraction'] = activation_summary['count'] / activation_summary['total']

        # Get only activated cells for simpler plotting
        activated_fraction = activation_summary[activation_summary['activation_status'] == 'Activated'].copy()
        activated_fraction = activated_fraction.set_index('Condition')['fraction']

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

        # Pairwise comparisons between conditions
        for i, condition1 in enumerate(condition_order):
            for j, condition2 in enumerate(condition_order):
                if i < j:
                    data1 = df_45min[df_45min['Condition'] == condition1]['p65_activation'].values
                    data2 = df_45min[df_45min['Condition'] == condition2]['p65_activation'].values

                    if len(data1) > 0 and len(data2) > 0:
                        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                        print(f"  {condition1} vs {condition2}: p = {p_value:.4e} ({sig})")


        # Plot 1: Stacked bar plot showing activated vs unactivated fractions
        fig, ax = plt.subplots()

        # Prepare data for stacked bar plot
        x_pos = np.arange(len(condition_order))
        activated_fracs = []
        unactivated_fracs = []

        for condition in condition_order:
            subset = activation_summary[
                (activation_summary['Condition'] == condition) &
                (activation_summary['activation_status'] == 'Activated')
            ]
            activated_frac = subset['fraction'].values[0] if len(subset) > 0 else 0
            activated_fracs.append(activated_frac)
            unactivated_fracs.append(1 - activated_frac)

        # Plot activated bars
        ax.bar(x_pos, activated_fracs, 
               label='Activated',
               color='#d62728', edgecolor='black', linewidth=1)
        # Plot unactivated bars
        ax.bar(x_pos, unactivated_fracs, bottom=activated_fracs,
               label='Unactivated',
               color='#1f77b4', edgecolor='black', linewidth=1)

        # Add text labels on bars
        for i, (act, unact) in enumerate(zip(activated_fracs, unactivated_fracs)):
            if act > 0.05:  # Only show label if fraction > 5%
                ax.text(i, act/2, f'{act:.2%}',
                       ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
            if unact > 0.05:
                ax.text(i, act + unact/2, f'{unact:.2%}',
                       ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

        ax.set_xlabel('Condition')
        ax.set_ylabel('Fraction of Cells')
        ax.set_title('Fraction of Activated Cells at 45 Minutes - Plate 021\n(Activated = Nucleus p65 > Cytosol p65)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_order, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_filename = plot_output_dir / "barplot_activated_fraction_45min.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Plot 2: Grouped bar plot by condition
        fig, ax = plt.subplots()

        x_pos = np.arange(len(condition_order))
        fractions = []

        for condition in condition_order:
            subset = activation_summary[
                (activation_summary['Condition'] == condition) &
                (activation_summary['activation_status'] == 'Activated')
            ]
            frac = subset['fraction'].values[0] if len(subset) > 0 else 0
            fractions.append(frac)

        colors = [condition_palette[condition_order.index(c)] for c in condition_order]
        ax.bar(x_pos, fractions, 
               color=colors, edgecolor='black', linewidth=1)

        # Add percentage labels
        for i, (x, val) in enumerate(zip(x_pos, fractions)):
            ax.text(x, val + 0.02, f'{val:.1%}',
                   ha='center', va='bottom', fontsize=9)

        # Add statistical annotations comparing conditions
        y_base = 1.0
        y_step = 0.08

        condition_pairs = list(combinations(range(len(condition_order)), 2))

        for pair_idx, (i, j) in enumerate(condition_pairs):
            condition1 = condition_order[i]
            condition2 = condition_order[j]

            # Get the binary activation status for each condition
            data1 = df_45min[df_45min['Condition'] == condition1]['activation_status'].apply(
                lambda x: 1 if x == 'Activated' else 0
            ).values
            data2 = df_45min[df_45min['Condition'] == condition2]['activation_status'].apply(
                lambda x: 1 if x == 'Activated' else 0
            ).values

            if len(data1) > 0 and len(data2) > 0:
                # Fisher's exact test for proportions
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
                y_pos = y_base + pair_idx * y_step

                ax.plot([i, i, j, j],
                       [y_pos - 0.01, y_pos, y_pos, y_pos - 0.01],
                       'k-', linewidth=1.2)
                ax.text((i + j) / 2, y_pos + 0.01, sig_symbol,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Condition')
        ax.set_ylabel('Fraction of Activated Cells')
        ax.set_title('Fraction of Activated Cells at 45 Minutes - Plate 021\n(Activated = Nucleus p65 > Cytosol p65)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_order, rotation=45, ha='right')
        ax.set_ylim(0, 1.35)  # Extended to accommodate annotations
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_filename = plot_output_dir / "barplot_activated_fraction_45min_by_condition.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


        # Plot 3: Heatmap of activation fractions
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Prepare data for heatmap - activated_fraction is already a Series indexed by Condition
        heatmap_data = activated_fraction.to_frame().T

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

        ax.set_title('Fraction of Activated Cells at 45 Minutes - Plate 021\n(Activated = Nucleus p65 > Cytosol p65)')
        ax.set_xlabel('Condition')
        ax.set_ylabel('')

        plt.tight_layout()
        plot_filename = plot_output_dir / "heatmap_activated_fraction_45min.png"
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
