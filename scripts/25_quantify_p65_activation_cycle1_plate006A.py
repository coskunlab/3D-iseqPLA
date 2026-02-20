# %% [markdown]
# # Quantify p65 Activation - Plate 006A Cycle 1
# This script reads single-cell PKL files from Plate 006A, calculates p65 activation
# as the ratio of median "S536 p65 Protein (activated)" to median "Total p65 Protein" 
# per cell, and visualizes how activation changes over time across different treatment
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
import time
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
layout_file = DATA_ROOT / '49 Cystic Fibrosis - Rabin' / 'Data' / '24 well plate 006A' / 'Plate006A_layout.xlsx'

# Path to PKL files directory
pkl_path = DATA_ROOT / '49 Cystic Fibrosis - Rabin' / 'Data' / '24 well plate 006A' / '29Dec2025 cycle 1 IF' / '09 PKL single cell'

# Output directory for plots
plot_output_dir = FIGURES_ROOT / '006A_fibroblast_only_coculture'
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
    # Pattern examples: "LPS 10 ng/mL 480 m", "TNFa 10 ng/mL 240 m", "DMSO 0.1% 120 m"
    pattern = re.compile(r'([a-zA-Z0-9]+)\s+[\d\.]+\s*[a-zA-Z/%]+\s+(\d+)\s*m')

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
    Calculate p65 activation as Pearson correlation between S536 p65 Protein 
    and Total p65 Protein signals per cell.

    Parameters:
    -----------
    df : pandas.DataFrame
        Single-cell data with columns: CellLabel, S536 p65 Protein, Total p65 Protein

    Returns:
    --------
    pandas.DataFrame
        DataFrame with CellLabel and p65_activation (Pearson r)
    """
    # Calculate Pearson correlation between S536 p65 and Total p65 for each cell
    activation_data = []
    
    # Check if required columns exist
    s536_col = 'S536 p65 Protein'
    total_col = 'Total p65 Protein'
    
    if s536_col not in df.columns or total_col not in df.columns:
        print(f"Warning: Required columns not found. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    for cell_id, cell_df in df.groupby('CellLabel'):
        # Get S536 p65 (activated) and Total p65 signals for the entire cell
        s536_signal = cell_df[s536_col].values
        total_signal = cell_df[total_col].values
        
        # Only calculate if we have sufficient data points
        if len(s536_signal) > 2 and len(total_signal) > 2:
            # Check if there is variation in the signals (required for correlation)
            if np.std(s536_signal) > 0 and np.std(total_signal) > 0:
                # Calculate Pearson correlation
                pearson_r = np.corrcoef(s536_signal, total_signal)[0, 1]
                
                if not np.isnan(pearson_r):
                    activation_data.append({
                        'CellLabel': cell_id,
                        'p65_activation': pearson_r,
                        's536_mean': s536_signal.mean(),
                        'total_mean': total_signal.mean(),
                        's536_median': np.median(s536_signal),
                        'total_median': np.median(total_signal)
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

    # Get the well ID from the filename (e.g., 'A1-1_Cell0001.pkl' -> 'A1')
    filename_parts = file_path.stem.split('-')
    if len(filename_parts) > 0:
        well_id = filename_parts[0]
    else:
        return None

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
    activation_df['FOV'] = file_path.stem.split('_Cell')[0]

    return activation_df


def add_stat_annotation_barplot(ax, data, x_col, y_col, hue_col, hue_order, control_condition):
    """
    Add statistical significance annotations to grouped barplots comparing each condition to control.
    Uses Mann-Whitney U test with asterisk notation.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis object to add annotations to
    data : pandas.DataFrame
        The dataframe containing the plot data
    x_col : str
        Column name for x-axis (e.g., 'Timepoint')
    y_col : str
        Column name for y-axis (e.g., 'p65_activation')
    hue_col : str
        Column name for hue/grouping (e.g., 'Condition')
    hue_order : list
        Order of hue categories
    control_condition : str
        Name of control condition to compare against
    """
    from scipy import stats
    
    # Get unique x values (timepoints)
    x_values = sorted(data[x_col].unique())
    n_hues = len(hue_order)
    
    # Calculate bar width and positions (matching seaborn's default)
    bar_width = 0.8 / n_hues
    
    # Get y-axis limits to position brackets
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Track maximum y position for stacking brackets
    max_y_per_x = {}
    
    for x_idx, x_val in enumerate(x_values):
        # Get data for this x value
        x_data = data[data[x_col] == x_val]
        
        # Get control data
        control_data = x_data[x_data[hue_col] == control_condition][y_col].values
        
        if len(control_data) == 0:
            continue
        
        # Find control position in hue_order
        try:
            control_idx = hue_order.index(control_condition)
        except ValueError:
            continue
        
        # Calculate control bar x position
        control_x = x_idx + (control_idx - n_hues/2 + 0.5) * bar_width
        
        # Get max y value for this x position (for bracket positioning)
        x_max_y = x_data.groupby(hue_col)[y_col].mean().max()
        x_std = x_data.groupby(hue_col)[y_col].std().max()
        bracket_base = x_max_y + x_std + y_range * 0.02
        
        # Compare each treatment to control
        bracket_offset = 0
        for treat_idx, treatment in enumerate(hue_order):
            if treatment == control_condition:
                continue
            
            # Get treatment data
            treatment_data = x_data[x_data[hue_col] == treatment][y_col].values
            
            if len(treatment_data) == 0:
                continue
            
            # Perform Mann-Whitney U test
            try:
                statistic, p_value = stats.mannwhitneyu(treatment_data, control_data, alternative='two-sided')
            except Exception as e:
                print(f"Could not perform test for {treatment} vs {control_condition} at {x_val}: {e}")
                continue
            
            # Determine significance level
            if p_value < 0.001:
                sig_symbol = '***'
            elif p_value < 0.01:
                sig_symbol = '**'
            elif p_value < 0.05:
                sig_symbol = '*'
            else:
                sig_symbol = 'ns'
            
            # Only draw if significant
            if sig_symbol != 'ns':
                # Calculate treatment bar x position
                treatment_x = x_idx + (treat_idx - n_hues/2 + 0.5) * bar_width
                
                # Calculate bracket position
                bracket_y = bracket_base + bracket_offset * y_range * 0.05
                bracket_h = y_range * 0.01
                
                # Draw bracket
                ax.plot([treatment_x, treatment_x], [bracket_y, bracket_y + bracket_h], 
                       'k-', linewidth=1.5, alpha=0.8)
                ax.plot([treatment_x, control_x], [bracket_y + bracket_h, bracket_y + bracket_h], 
                       'k-', linewidth=1.5, alpha=0.8)
                ax.plot([control_x, control_x], [bracket_y + bracket_h, bracket_y], 
                       'k-', linewidth=1.5, alpha=0.8)
                
                # Add significance symbol
                mid_x = (treatment_x + control_x) / 2
                ax.text(mid_x, bracket_y + bracket_h + y_range * 0.01, sig_symbol,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                bracket_offset += 1


def add_stat_annotation_violin(ax, data, x_col, y_col, x_order, control_condition):
    """
    Add statistical significance annotations to violin plots comparing each condition to control.
    Uses Mann-Whitney U test with asterisk notation.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis object to add annotations to
    data : pandas.DataFrame
        The dataframe containing the plot data
    x_col : str
        Column name for x-axis (e.g., 'Condition')
    y_col : str
        Column name for y-axis (e.g., 'p65_activation')
    x_order : list
        Order of x-axis categories
    control_condition : str
        Name of control condition to compare against
    """
    from scipy import stats
    
    # Get control data
    control_data = data[data[x_col] == control_condition][y_col].values
    
    if len(control_data) == 0:
        return
    
    # Find control position
    try:
        control_idx = x_order.index(control_condition)
    except ValueError:
        return
    
    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Base height for brackets
    bracket_base = y_max - y_range * 0.15
    
    # Compare each treatment to control
    bracket_offset = 0
    for treat_idx, treatment in enumerate(x_order):
        if treatment == control_condition:
            continue
        
        # Get treatment data
        treatment_data = data[data[x_col] == treatment][y_col].values
        
        if len(treatment_data) == 0:
            continue
        
        # Perform Mann-Whitney U test
        try:
            statistic, p_value = stats.mannwhitneyu(treatment_data, control_data, alternative='two-sided')
        except Exception as e:
            print(f"Could not perform test for {treatment} vs {control_condition}: {e}")
            continue
        
        # Determine significance level
        if p_value < 0.001:
            sig_symbol = '***'
        elif p_value < 0.01:
            sig_symbol = '**'
        elif p_value < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'ns'
        
        # Only draw if significant
        if sig_symbol != 'ns':
            # Calculate bracket position
            bracket_y = bracket_base - bracket_offset * y_range * 0.06
            bracket_h = y_range * 0.015
            
            # Draw bracket
            ax.plot([treat_idx, treat_idx], [bracket_y, bracket_y + bracket_h], 
                   'k-', linewidth=1.5, alpha=0.8)
            ax.plot([treat_idx, control_idx], [bracket_y + bracket_h, bracket_y + bracket_h], 
                   'k-', linewidth=1.5, alpha=0.8)
            ax.plot([control_idx, control_idx], [bracket_y + bracket_h, bracket_y], 
                   'k-', linewidth=1.5, alpha=0.8)
            
            # Add significance symbol
            mid_x = (treat_idx + control_idx) / 2
            ax.text(mid_x, bracket_y + bracket_h + y_range * 0.01, sig_symbol,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            bracket_offset += 1


def calculate_effect_size(data1, data2, method='rank_biserial'):
    """
    Calculate effect size for comparing two groups.

    Parameters:
    -----------
    data1 : array-like
        First group data
    data2 : array-like
        Second group data
    method : str
        'rank_biserial' for Mann-Whitney (default) or 'cohens_d' for parametric

    Returns:
    --------
    float
        Effect size value
    """
    if len(data1) == 0 or len(data2) == 0:
        return np.nan

    if method == 'rank_biserial':
        # Rank-biserial correlation (appropriate for Mann-Whitney U test)
        # Formula: r = 1 - (2U)/(n1*n2) where U is the Mann-Whitney statistic
        try:
            u_statistic, _ = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            n1, n2 = len(data1), len(data2)
            rank_biserial = 1 - (2 * u_statistic) / (n1 * n2)
            return rank_biserial
        except Exception:
            return np.nan

    elif method == 'cohens_d':
        # Cohen's d (parametric effect size)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return np.nan

        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d

    else:
        raise ValueError(f"Unknown method: {method}")


def apply_multiple_testing_correction(p_values, method='fdr_bh'):
    """
    Apply multiple testing correction to a list of p-values.

    Parameters:
    -----------
    p_values : array-like
        List of p-values to correct
    method : str
        'fdr_bh' for Benjamini-Hochberg FDR (default) or 'bonferroni'

    Returns:
    --------
    numpy.ndarray
        Corrected p-values
    """
    p_array = np.array(p_values)
    n = len(p_array)

    if method == 'bonferroni':
        # Bonferroni correction: multiply by number of tests
        corrected = np.minimum(p_array * n, 1.0)
        return corrected

    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        # Sort p-values and their indices
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # Calculate correction
        corrected_sorted = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                corrected_sorted[i] = sorted_p[i]
            else:
                corrected_sorted[i] = min(sorted_p[i] * n / (i + 1), corrected_sorted[i + 1])

        # Restore original order
        corrected = np.zeros(n)
        corrected[sorted_indices] = corrected_sorted

        # Cap at 1.0
        corrected = np.minimum(corrected, 1.0)
        return corrected

    else:
        raise ValueError(f"Unknown method: {method}")


def generate_compact_letters(p_values_matrix, condition_names, alpha=0.05):
    """
    Generate compact letter display from pairwise p-values.
    Groups that share letters are not significantly different.

    Parameters:
    -----------
    p_values_matrix : pandas.DataFrame or numpy.ndarray
        Square matrix of p-values (can have NaN on diagonal)
    condition_names : list
        Names of conditions (must match order in matrix)
    alpha : float
        Significance threshold (default 0.05)

    Returns:
    --------
    dict
        Dictionary mapping condition names to letter groups (e.g., {'LPS': 'a', 'TNFa': 'ab', 'DMSO': 'b'})
    """
    n = len(condition_names)

    # Create adjacency matrix (True if NOT significantly different)
    not_different = np.ones((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                if isinstance(p_values_matrix, pd.DataFrame):
                    p_val = p_values_matrix.iloc[i, j]
                else:
                    p_val = p_values_matrix[i, j]

                if not np.isnan(p_val) and p_val < alpha:
                    not_different[i, j] = False

    # Find groups using a greedy algorithm
    # Start by assigning letters to each condition
    letters = []
    available_letters = 'abcdefghijklmnopqrstuvwxyz'
    letter_idx = 0

    # Track which conditions have been assigned which letters
    condition_letters = {name: [] for name in condition_names}

    # For each pair of conditions that are not significantly different,
    # they should share at least one letter
    for i in range(n):
        # Find all conditions not significantly different from condition i
        group = [j for j in range(n) if not_different[i, j]]

        # Check if this group already has a common letter
        if len(condition_letters[condition_names[i]]) == 0:
            # Assign new letter to this group
            letter = available_letters[letter_idx % len(available_letters)]
            letter_idx += 1

            for j in group:
                if letter not in condition_letters[condition_names[j]]:
                    condition_letters[condition_names[j]].append(letter)

    # Sort letters for each condition and join
    compact_letters = {}
    for name in condition_names:
        letters_list = sorted(condition_letters[name])
        compact_letters[name] = ''.join(letters_list) if letters_list else '-'

    return compact_letters


def interpret_effect_size(effect_size, method='rank_biserial'):
    """
    Provide interpretation of effect size magnitude.

    Parameters:
    -----------
    effect_size : float
        The calculated effect size
    method : str
        'rank_biserial' or 'cohens_d'

    Returns:
    --------
    str
        Interpretation string (e.g., 'small', 'medium', 'large')
    """
    abs_effect = abs(effect_size)

    if np.isnan(abs_effect):
        return 'undefined'

    if method == 'rank_biserial':
        # Interpretation for rank-biserial correlation
        # Similar to correlation coefficient interpretation
        if abs_effect < 0.1:
            return 'negligible'
        elif abs_effect < 0.3:
            return 'small'
        elif abs_effect < 0.5:
            return 'medium'
        else:
            return 'large'

    elif method == 'cohens_d':
        # Cohen's guidelines for effect size
        if abs_effect < 0.2:
            return 'negligible'
        elif abs_effect < 0.5:
            return 'small'
        elif abs_effect < 0.8:
            return 'medium'
        else:
            return 'large'

    return 'unknown'


# %% [markdown]
# ## 4. Process PKL Files

# %%
print(f"\n{'='*60}")
print(f"Processing Plate 006A")
print(f"{'='*60}")

# Parse layout
layout_df = parse_layout(layout_file)
if layout_df.empty:
    print(f"  Error: Could not parse layout file.")
    raise ValueError("Layout file parsing failed.")

print(f"  Layout parsed successfully.")
print(f"  Found {len(layout_df)} wells in layout.")
print(f"\n  Layout preview:")
print(layout_df.head(10))

# Get PKL files
all_pkl_files = list(pkl_path.glob("*.pkl"))
print(f"\n  Found {len(all_pkl_files)} PKL files.")

if len(all_pkl_files) == 0:
    print(f"  Error: No PKL files found.")
    raise ValueError("No PKL files found in the specified directory.")

# Create layout map
layout_map = layout_df.set_index('Well').to_dict('index')

# Process files in parallel
print(f"\n  Processing PKL files...")
results_list = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(process_pkl_file)(f, layout_map)
    for f in tqdm(all_pkl_files, desc="  Processing PKL files")
)

# Combine results
valid_results = [df for df in results_list if df is not None]
if len(valid_results) == 0:
    print("\n" + "="*60)
    print("ERROR: No valid data was extracted from any PKL files.")
    print("Please check the column names in your PKL files.")
    print("="*60)
    combined_df = pd.DataFrame()
else:
    combined_df = pd.concat(valid_results, ignore_index=True)

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
    print(combined_df.groupby(['Condition', 'Timepoint']).agg({'CellLabel': 'count'}))
    
    # Print activation statistics
    print(f"\n{'='*60}")
    print(f"P65 ACTIVATION STATISTICS")
    print(f"{'='*60}")
    print(f"Mean activation ratio: {combined_df['p65_activation'].mean():.3f}")
    print(f"Median activation ratio: {combined_df['p65_activation'].median():.3f}")
    print(f"Min activation ratio: {combined_df['p65_activation'].min():.3f}")
    print(f"Max activation ratio: {combined_df['p65_activation'].max():.3f}")
else:
    print("\nNo data was collected. Please check the paths and file formats.")
    combined_df = pd.DataFrame()


# %% [markdown]
# ## 5. Generate Plots Comparing p65 Activation

# %%
if not combined_df.empty:
    # Define condition order and colors
    condition_order = [c for c in ['LPS', 'TNFa', 'IL1B', 'DMSO Control']
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
    fig, ax = plt.subplots(figsize=(12, 6))

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

    ax.set_title("p65 Activation Over Time - Plate 006A")
    ax.set_ylabel("p65 Activation\n(Pearson r: S536 p65 vs Total p65)")
    ax.set_xlabel("Timepoint (minutes)")
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No correlation (r=0)')
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plot_filename = plot_output_dir / "01_barplot_p65_activation_over_time.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)


    # %% [markdown]
    # ### Plot 1B: p65 Activation Over Time - All Conditions (with Statistical Significance)

    # %%
    fig, ax = plt.subplots(figsize=(14, 7))

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

    ax.set_title("p65 Activation Over Time - Plate 006A\n(with Statistical Significance)", fontsize=14)
    ax.set_ylabel("p65 Activation\n(Pearson r: S536 p65 vs Total p65)")
    ax.set_xlabel("Timepoint (minutes)")
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No correlation (r=0)')
    ax.set_ylim(0, 1.2)  # Extended to accommodate significance brackets

    # Add statistical annotations
    add_stat_annotation_barplot(
        ax=ax,
        data=combined_df,
        x_col='Timepoint',
        y_col='p65_activation',
        hue_col='Condition',
        hue_order=condition_order,
        control_condition=control_condition
    )

    plt.tight_layout()
    plot_filename = plot_output_dir / "01_barplot_p65_activation_over_time_with_stats.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)


    # %% [markdown]
    # ### Plot 2: Spline Plots - Condition vs Control

    # %%
    for condition in [c for c in condition_order if c != control_condition]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('p65 Activation\n(Pearson r: S536 p65 vs Total p65)')
        ax.set_title(f"p65 Activation - Plate 006A\n{condition} vs {control_condition}")

        # Plot treated condition
        dfSub = combined_df[combined_df['Condition'] == condition].copy()
        dfSub = dfSub.dropna(subset=['p65_activation'])

        if len(dfSub) > 0:
            grouped = dfSub.groupby('Timepoint')['p65_activation']
            mean_vals = grouped.mean().reset_index()
            std_vals = grouped.std().reset_index()

            ax.scatter(mean_vals['Timepoint'], mean_vals['p65_activation'],
                      color='blue', alpha=1, s=80, zorder=3, label=f'{condition} (mean)')
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
        if control_condition in combined_df['Condition'].unique():
            dfCtrl = combined_df[combined_df['Condition'] == control_condition].copy()
            dfCtrl = dfCtrl.dropna(subset=['p65_activation'])

            if len(dfCtrl) > 0:
                grouped = dfCtrl.groupby('Timepoint')['p65_activation']
                mean_vals = grouped.mean().reset_index()
                std_vals = grouped.std().reset_index()

                ax.scatter(mean_vals['Timepoint'], mean_vals['p65_activation'],
                          color='gray', alpha=0.7, s=80, zorder=3, label=f'{control_condition} (mean)')
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

        ax.axhline(y=0, color='red', linestyle=':', alpha=0.3, label='No correlation (r=0)')
        ax.set_ylim(-1, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        safe_condition = condition.replace(' ', '_')
        plot_filename = plot_output_dir / f"02_spline_p65_activation_{safe_condition}_vs_control.png"
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

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('p65 Activation\n(Pearson r: S536 p65 vs Total p65)')
        ax.set_title(f"p65 Activation - Plate 006A - {condition}\n(Boxplot with Median Spline)")

        timepoints = sorted(dfSub['Timepoint'].unique())
        bins, groups = zip(*dfSub.groupby('Timepoint')['p65_activation'])

        bp = ax.boxplot(
            groups,
            positions=timepoints,
            widths=15,
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

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3, label='No correlation (r=0)')
        ax.set_ylim(-1, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        safe_condition = condition.replace(' ', '_')
        plot_filename = plot_output_dir / f"03_boxplot_p65_activation_{safe_condition}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 4: Heatmap - Mean and Median p65 Activation by Condition and Time

    # %%
    # Create heatmaps for both mean and median
    for aggregation_method in ['mean', 'median']:
        # Create a pivot table for heatmap
        pivot_data = combined_df.pivot_table(
            values='p65_activation',
            index='Condition',
            columns='Timepoint',
            aggfunc=aggregation_method
        )

        # Reorder rows to match condition_order
        pivot_data = pivot_data.reindex([c for c in condition_order if c in pivot_data.index])

        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                cbar_kws={'label': f'{aggregation_method.capitalize()} p65 Activation (Pearson r)'},
                ax=ax
            )

            ax.set_title(f"{aggregation_method.capitalize()} p65 Activation Heatmap - Plate 006A\n(Pearson r: S536 p65 vs Total p65)", pad=20)
            ax.set_xlabel("Timepoint (minutes)")
            ax.set_ylabel("Condition")

            plt.tight_layout()
            plot_filename = plot_output_dir / f"04_heatmap_p65_activation_{aggregation_method}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 4B: Heatmap - Number of Cells per Condition and Timepoint

    # %%
    # Create a pivot table counting cells per condition and timepoint
    cell_count_pivot = combined_df.pivot_table(
        values='CellLabel',
        index='Condition',
        columns='Timepoint',
        aggfunc='count'
    )

    # Reorder rows to match condition_order
    cell_count_pivot = cell_count_pivot.reindex([c for c in condition_order if c in cell_count_pivot.index])

    if not cell_count_pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            cell_count_pivot,
            annot=True,
            fmt='d',
            cmap='coolwarm',
            cbar_kws={'label': 'Number of Cells'},
            ax=ax
        )

        ax.set_title("Number of Cells per Condition and Timepoint - Plate 006A", pad=20)
        ax.set_xlabel("Timepoint (minutes)")
        ax.set_ylabel("Condition")

        plt.tight_layout()
        plot_filename = plot_output_dir / "04B_heatmap_cell_counts.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 5: Single-Cell p65 Activation - Violin Plots by Timepoint

    # %%
    # Create separate plots for each timepoint
    for timepoint in sorted(combined_df['Timepoint'].unique()):
        df_timepoint = combined_df[combined_df['Timepoint'] == timepoint].copy()

        if not df_timepoint.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.violinplot(
                data=df_timepoint,
                x='Condition',
                y='p65_activation',
                order=condition_order,
                palette=condition_palette,
                inner='box',
                ax=ax
            )

            # Overlay strip plot to show individual cells
            sns.stripplot(
                data=df_timepoint,
                x='Condition',
                y='p65_activation',
                order=condition_order,
                color='black',
                alpha=0.2,
                size=2,
                ax=ax
            )

            ax.set_title(f"Single-Cell p65 Activation at {timepoint} Minutes - Plate 006A\n(n={len(df_timepoint)} total cells)")
            ax.set_xlabel("Condition")
            ax.set_ylabel("p65 Activation\n(Pearson r: S536 p65 vs Total p65)")
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=0, color='red', linestyle=':', alpha=0.3, label='No correlation (r=0)')
            ax.set_ylim(-1, 1)

            plt.tight_layout()

            plot_filename = plot_output_dir / f"05_violin_p65_activation_{timepoint}min_all_conditions.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 5B: Single-Cell p65 Activation - Violin Plots by Timepoint (with Statistical Significance)

    # %%
    # Create separate plots for each timepoint with statistical annotations
    for timepoint in sorted(combined_df['Timepoint'].unique()):
        df_timepoint = combined_df[combined_df['Timepoint'] == timepoint].copy()

        if not df_timepoint.empty:
            fig, ax = plt.subplots(figsize=(10, 7))

            sns.violinplot(
                data=df_timepoint,
                x='Condition',
                y='p65_activation',
                order=condition_order,
                palette=condition_palette,
                inner='box',
                ax=ax
            )

            # Overlay strip plot to show individual cells
            sns.stripplot(
                data=df_timepoint,
                x='Condition',
                y='p65_activation',
                order=condition_order,
                color='black',
                alpha=0.2,
                size=2,
                ax=ax
            )

            ax.set_title(f"Single-Cell p65 Activation at {timepoint} Minutes - Plate 006A\n(n={len(df_timepoint)} total cells, with Statistical Significance)", fontsize=12)
            ax.set_xlabel("Condition")
            ax.set_ylabel("p65 Activation\n(Pearson r: S536 p65 vs Total p65)")
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=0, color='red', linestyle=':', alpha=0.3, label='No correlation (r=0)')
            ax.set_ylim(-1, 1.2)  # Extended to accommodate significance brackets

            # Add statistical annotations
            add_stat_annotation_violin(
                ax=ax,
                data=df_timepoint,
                x_col='Condition',
                y_col='p65_activation',
                x_order=condition_order,
                control_condition=control_condition
            )

            plt.tight_layout()

            plot_filename = plot_output_dir / f"05_violin_p65_activation_{timepoint}min_all_conditions_with_stats.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 6: Statistical Summary Tables and Comparisons

    # %%
    # Create statistical summary for each timepoint
    for timepoint in sorted(combined_df['Timepoint'].unique()):
        df_timepoint = combined_df[combined_df['Timepoint'] == timepoint].copy()
        
        if not df_timepoint.empty:
            print("\n" + "="*60)
            print(f"STATISTICAL SUMMARY AT {timepoint} MINUTES")
            print("="*60)
            summary = df_timepoint.groupby('Condition')['p65_activation'].agg([
                ('n_cells', 'count'),
                ('mean', 'mean'),
                ('median', 'median'),
                ('std', 'std'),
                ('sem', 'sem'),
                ('min', 'min'),
                ('max', 'max')
            ]).round(3)
            print(summary)

            # Calculate and print statistical comparisons with effect sizes
            print("\nSTATISTICAL COMPARISONS (Mann-Whitney U Test)")
            print("Pairwise comparisons with effect sizes and corrected p-values")
            print("-"*80)

            # Store all pairwise comparisons for correction
            comparison_results = []

            # Pairwise comparisons between conditions
            for i, condition1 in enumerate(condition_order):
                for j, condition2 in enumerate(condition_order):
                    if i < j:
                        data1 = df_timepoint[df_timepoint['Condition'] == condition1]['p65_activation'].values
                        data2 = df_timepoint[df_timepoint['Condition'] == condition2]['p65_activation'].values

                        if len(data1) > 0 and len(data2) > 0:
                            # Calculate p-value
                            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

                            # Calculate effect size
                            effect_size = calculate_effect_size(data1, data2, method='rank_biserial')
                            effect_interpretation = interpret_effect_size(effect_size, method='rank_biserial')

                            comparison_results.append({
                                'comparison': f"{condition1} vs {condition2}",
                                'p_value': p_value,
                                'effect_size': effect_size,
                                'interpretation': effect_interpretation,
                                'n1': len(data1),
                                'n2': len(data2)
                            })

            # Apply multiple testing correction
            if len(comparison_results) > 0:
                raw_p_values = [r['p_value'] for r in comparison_results]
                corrected_p_values = apply_multiple_testing_correction(raw_p_values, method='fdr_bh')

                for idx, result in enumerate(comparison_results):
                    result['p_corrected'] = corrected_p_values[idx]

                # Print results with both raw and corrected p-values
                print(f"{'Comparison':<25} {'Raw p-value':<12} {'FDR p-value':<12} {'Effect Size':<12} {'Magnitude':<12}")
                print("-"*80)

                for result in comparison_results:
                    # Determine significance for raw p-value
                    raw_sig = ('***' if result['p_value'] < 0.001 else
                              '**' if result['p_value'] < 0.01 else
                              '*' if result['p_value'] < 0.05 else 'ns')

                    # Determine significance for corrected p-value
                    corr_sig = ('***' if result['p_corrected'] < 0.001 else
                               '**' if result['p_corrected'] < 0.01 else
                               '*' if result['p_corrected'] < 0.05 else 'ns')

                    effect_str = f"{result['effect_size']:.3f}" if not np.isnan(result['effect_size']) else "N/A"

                    print(f"{result['comparison']:<25} "
                          f"{result['p_value']:.4e} {raw_sig:<4} "
                          f"{result['p_corrected']:.4e} {corr_sig:<4} "
                          f"{effect_str:<12} "
                          f"{result['interpretation']:<12}")

                print("\nNote: Effect size is rank-biserial correlation (appropriate for Mann-Whitney U test)")
                print("      FDR correction: Benjamini-Hochberg method")
                print("      Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")


    # %% [markdown]
    # ### NEW PLOTS: Comprehensive Statistical Comparisons Among All Conditions

    # %% [markdown]
    # ### Plot 11: P-value Heatmap Matrix - All Pairwise Comparisons

    # %%
    # Create p-value heatmaps for each timepoint showing ALL pairwise comparisons
    print(f"\n{'='*60}")
    print(f"GENERATING PAIRWISE COMPARISON HEATMAPS")
    print(f"{'='*60}")

    for timepoint in sorted(combined_df['Timepoint'].unique()):
        df_timepoint = combined_df[combined_df['Timepoint'] == timepoint].copy()

        if not df_timepoint.empty:
            # Calculate all pairwise comparisons with p-values and effect sizes
            n_conditions = len(condition_order)
            p_value_matrix = np.ones((n_conditions, n_conditions))
            effect_size_matrix = np.zeros((n_conditions, n_conditions))

            # Store raw p-values for correction
            raw_p_values = []
            comparison_indices = []

            for i, condition1 in enumerate(condition_order):
                for j, condition2 in enumerate(condition_order):
                    if i != j:
                        data1 = df_timepoint[df_timepoint['Condition'] == condition1]['p65_activation'].values
                        data2 = df_timepoint[df_timepoint['Condition'] == condition2]['p65_activation'].values

                        if len(data1) > 0 and len(data2) > 0:
                            # Calculate p-value
                            try:
                                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                p_value_matrix[i, j] = p_value
                                raw_p_values.append(p_value)
                                comparison_indices.append((i, j))

                                # Calculate effect size
                                effect_size = calculate_effect_size(data1, data2, method='rank_biserial')
                                effect_size_matrix[i, j] = effect_size
                            except Exception as e:
                                p_value_matrix[i, j] = np.nan
                                effect_size_matrix[i, j] = np.nan
                        else:
                            p_value_matrix[i, j] = np.nan
                            effect_size_matrix[i, j] = np.nan
                    else:
                        # Diagonal (self-comparison)
                        p_value_matrix[i, j] = np.nan
                        effect_size_matrix[i, j] = np.nan

            # Apply multiple testing correction
            if len(raw_p_values) > 0:
                corrected_p_values = apply_multiple_testing_correction(raw_p_values, method='fdr_bh')

                # Update matrix with corrected p-values
                corrected_p_matrix = p_value_matrix.copy()
                for idx, (i, j) in enumerate(comparison_indices):
                    corrected_p_matrix[i, j] = corrected_p_values[idx]
            else:
                corrected_p_matrix = p_value_matrix.copy()

            # Create DataFrame for easier plotting
            p_value_df = pd.DataFrame(p_value_matrix, index=condition_order, columns=condition_order)
            corrected_p_df = pd.DataFrame(corrected_p_matrix, index=condition_order, columns=condition_order)
            effect_size_df = pd.DataFrame(effect_size_matrix, index=condition_order, columns=condition_order)

            # Plot 1: Raw p-values heatmap with effect sizes as annotations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            # Left panel: P-values with significance stars
            # Create annotations combining p-values and significance
            annot_matrix = np.empty((n_conditions, n_conditions), dtype=object)
            for i in range(n_conditions):
                for j in range(n_conditions):
                    if i == j:
                        annot_matrix[i, j] = '-'
                    else:
                        p_val = p_value_df.iloc[i, j]
                        if np.isnan(p_val):
                            annot_matrix[i, j] = 'N/A'
                        else:
                            # Determine significance
                            if p_val < 0.001:
                                sig = '***'
                            elif p_val < 0.01:
                                sig = '**'
                            elif p_val < 0.05:
                                sig = '*'
                            else:
                                sig = 'ns'
                            annot_matrix[i, j] = f'{p_val:.3f}\n{sig}'

            # Plot p-values with discrete colormap
            # Create custom colormap with thresholds
            colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
            n_bins = 6
            cmap = sns.color_palette(colors, as_cmap=True)

            # Mask diagonal
            mask = np.eye(n_conditions, dtype=bool)

            sns.heatmap(
                p_value_df,
                annot=annot_matrix,
                fmt='',
                cmap=cmap,
                vmin=0,
                vmax=0.1,
                mask=mask,
                cbar_kws={'label': 'P-value (raw)', 'extend': 'max'},
                linewidths=1,
                linecolor='white',
                square=True,
                ax=ax1
            )
            ax1.set_title(f'Raw P-values at {timepoint} Minutes\n(Mann-Whitney U Test, Uncorrected)', fontsize=12)
            ax1.set_xlabel('Condition')
            ax1.set_ylabel('Condition')

            # Right panel: Corrected p-values with significance stars
            annot_matrix_corrected = np.empty((n_conditions, n_conditions), dtype=object)
            for i in range(n_conditions):
                for j in range(n_conditions):
                    if i == j:
                        annot_matrix_corrected[i, j] = '-'
                    else:
                        p_val = corrected_p_df.iloc[i, j]
                        if np.isnan(p_val):
                            annot_matrix_corrected[i, j] = 'N/A'
                        else:
                            # Determine significance
                            if p_val < 0.001:
                                sig = '***'
                            elif p_val < 0.01:
                                sig = '**'
                            elif p_val < 0.05:
                                sig = '*'
                            else:
                                sig = 'ns'
                            annot_matrix_corrected[i, j] = f'{p_val:.3f}\n{sig}'

            sns.heatmap(
                corrected_p_df,
                annot=annot_matrix_corrected,
                fmt='',
                cmap=cmap,
                vmin=0,
                vmax=0.1,
                mask=mask,
                cbar_kws={'label': 'P-value (FDR corrected)', 'extend': 'max'},
                linewidths=1,
                linecolor='white',
                square=True,
                ax=ax2
            )
            ax2.set_title(f'FDR-Corrected P-values at {timepoint} Minutes\n(Benjamini-Hochberg Correction)', fontsize=12)
            ax2.set_xlabel('Condition')
            ax2.set_ylabel('Condition')

            plt.tight_layout()
            plot_filename = plot_output_dir / f"11_heatmap_pairwise_pvalues_{timepoint}min.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 12: Compact Letter Display - Violin Plots with Statistical Groupings

    # %%
    # Create violin plots with compact letter display for each timepoint
    print(f"\n{'='*60}")
    print(f"GENERATING COMPACT LETTER DISPLAY PLOTS")
    print(f"{'='*60}")

    for timepoint in sorted(combined_df['Timepoint'].unique()):
        df_timepoint = combined_df[combined_df['Timepoint'] == timepoint].copy()

        if not df_timepoint.empty:
            # Calculate all pairwise comparisons for this timepoint
            n_conditions = len(condition_order)
            p_value_matrix = np.ones((n_conditions, n_conditions))
            raw_p_values = []
            comparison_indices = []

            for i, condition1 in enumerate(condition_order):
                for j, condition2 in enumerate(condition_order):
                    if i < j:  # Only upper triangle for compact letters
                        data1 = df_timepoint[df_timepoint['Condition'] == condition1]['p65_activation'].values
                        data2 = df_timepoint[df_timepoint['Condition'] == condition2]['p65_activation'].values

                        if len(data1) > 0 and len(data2) > 0:
                            try:
                                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                p_value_matrix[i, j] = p_value
                                p_value_matrix[j, i] = p_value  # Symmetric
                                raw_p_values.append(p_value)
                                comparison_indices.append((i, j))
                            except Exception:
                                p_value_matrix[i, j] = np.nan
                                p_value_matrix[j, i] = np.nan
                        else:
                            p_value_matrix[i, j] = np.nan
                            p_value_matrix[j, i] = np.nan

            # Apply multiple testing correction
            if len(raw_p_values) > 0:
                corrected_p_values = apply_multiple_testing_correction(raw_p_values, method='fdr_bh')

                # Update matrix with corrected p-values (symmetric)
                corrected_p_matrix = p_value_matrix.copy()
                for idx, (i, j) in enumerate(comparison_indices):
                    corrected_p_matrix[i, j] = corrected_p_values[idx]
                    corrected_p_matrix[j, i] = corrected_p_values[idx]

                # Generate compact letter display
                compact_letters = generate_compact_letters(
                    corrected_p_matrix,
                    condition_order,
                    alpha=0.05
                )
            else:
                compact_letters = {cond: '-' for cond in condition_order}

            # Create violin plot with compact letters
            fig, ax = plt.subplots(figsize=(10, 7))

            # Plot violin
            sns.violinplot(
                data=df_timepoint,
                x='Condition',
                y='p65_activation',
                order=condition_order,
                palette=condition_palette,
                inner='box',
                ax=ax
            )

            # Overlay strip plot
            sns.stripplot(
                data=df_timepoint,
                x='Condition',
                y='p65_activation',
                order=condition_order,
                color='black',
                alpha=0.2,
                size=2,
                ax=ax
            )

            # Add compact letters above each violin
            y_max = ax.get_ylim()[1]
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05

            for idx, condition in enumerate(condition_order):
                letter = compact_letters.get(condition, '-')
                ax.text(idx, y_max - y_offset, letter,
                       ha='center', va='top', fontsize=14,
                       fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            ax.set_title(f'Single-Cell p65 Activation at {timepoint} Minutes - Plate 006A\n' +
                        f'(n={len(df_timepoint)} cells, Compact Letter Display)', fontsize=12)
            ax.set_xlabel('Condition', fontsize=11)
            ax.set_ylabel('p65 Activation\n(Pearson r: S536 p65 vs Total p65)', fontsize=11)
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=0, color='red', linestyle=':', alpha=0.3, label='No correlation (r=0)')
            ax.set_ylim(-1, 1.15)

            # Add legend explaining compact letters
            legend_text = 'Groups sharing letters are not\nsignificantly different (α=0.05, FDR-corrected)'
            ax.text(0.02, 0.98, legend_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plot_filename = plot_output_dir / f"12_violin_compact_letters_{timepoint}min.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 13: Effect Size Heatmap - Magnitude of Differences

    # %%
    # Create effect size heatmaps for each timepoint
    print(f"\n{'='*60}")
    print(f"GENERATING EFFECT SIZE HEATMAPS")
    print(f"{'='*60}")

    for timepoint in sorted(combined_df['Timepoint'].unique()):
        df_timepoint = combined_df[combined_df['Timepoint'] == timepoint].copy()

        if not df_timepoint.empty:
            # Calculate effect sizes and p-values
            n_conditions = len(condition_order)
            effect_size_matrix = np.zeros((n_conditions, n_conditions))
            p_value_matrix = np.ones((n_conditions, n_conditions))
            raw_p_values = []
            comparison_indices = []

            for i, condition1 in enumerate(condition_order):
                for j, condition2 in enumerate(condition_order):
                    if i != j:
                        data1 = df_timepoint[df_timepoint['Condition'] == condition1]['p65_activation'].values
                        data2 = df_timepoint[df_timepoint['Condition'] == condition2]['p65_activation'].values

                        if len(data1) > 0 and len(data2) > 0:
                            try:
                                # Calculate effect size
                                effect_size = calculate_effect_size(data1, data2, method='rank_biserial')
                                effect_size_matrix[i, j] = effect_size

                                # Calculate p-value for significance
                                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                p_value_matrix[i, j] = p_value
                                raw_p_values.append(p_value)
                                comparison_indices.append((i, j))
                            except Exception:
                                effect_size_matrix[i, j] = np.nan
                                p_value_matrix[i, j] = np.nan
                        else:
                            effect_size_matrix[i, j] = np.nan
                            p_value_matrix[i, j] = np.nan
                    else:
                        effect_size_matrix[i, j] = np.nan
                        p_value_matrix[i, j] = np.nan

            # Apply multiple testing correction
            if len(raw_p_values) > 0:
                corrected_p_values = apply_multiple_testing_correction(raw_p_values, method='fdr_bh')
                corrected_p_matrix = p_value_matrix.copy()
                for idx, (i, j) in enumerate(comparison_indices):
                    corrected_p_matrix[i, j] = corrected_p_values[idx]
            else:
                corrected_p_matrix = p_value_matrix.copy()

            # Create DataFrames
            effect_size_df = pd.DataFrame(effect_size_matrix, index=condition_order, columns=condition_order)
            corrected_p_df = pd.DataFrame(corrected_p_matrix, index=condition_order, columns=condition_order)

            # Create annotations with effect size and significance
            annot_matrix = np.empty((n_conditions, n_conditions), dtype=object)
            for i in range(n_conditions):
                for j in range(n_conditions):
                    if i == j:
                        annot_matrix[i, j] = '-'
                    else:
                        effect_size = effect_size_df.iloc[i, j]
                        p_val = corrected_p_df.iloc[i, j]

                        if np.isnan(effect_size) or np.isnan(p_val):
                            annot_matrix[i, j] = 'N/A'
                        else:
                            # Determine significance
                            if p_val < 0.001:
                                sig = '***'
                            elif p_val < 0.01:
                                sig = '**'
                            elif p_val < 0.05:
                                sig = '*'
                            else:
                                sig = 'ns'

                            # Get effect size interpretation
                            interpretation = interpret_effect_size(effect_size, method='rank_biserial')
                            annot_matrix[i, j] = f'{effect_size:.2f}\n{sig}'

            # Plot effect size heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Mask diagonal
            mask = np.eye(n_conditions, dtype=bool)

            # Use diverging colormap centered at 0
            sns.heatmap(
                effect_size_df,
                annot=annot_matrix,
                fmt='',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                mask=mask,
                cbar_kws={'label': 'Rank-Biserial Correlation (Effect Size)'},
                linewidths=1,
                linecolor='white',
                square=True,
                ax=ax
            )

            ax.set_title(f'Effect Size Matrix at {timepoint} Minutes - Plate 006A\n' +
                        f'(Rank-Biserial Correlation with FDR-Corrected Significance)', fontsize=12)
            ax.set_xlabel('Condition', fontsize=11)
            ax.set_ylabel('Condition', fontsize=11)

            # Add legend for interpretation
            legend_text = ('Effect Size Interpretation:\n'
                          '|r| < 0.1: negligible\n'
                          '|r| < 0.3: small\n'
                          '|r| < 0.5: medium\n'
                          '|r| ≥ 0.5: large')
            ax.text(1.15, 0.5, legend_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plot_filename = plot_output_dir / f"13_heatmap_effect_sizes_{timepoint}min.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {plot_filename.name}")
            plt.close(fig)


    # %% [markdown]
    # ### Plot 7: Activation Fraction Over Time

    # %%
    # Define "activated" cells as those with positive correlation
    activation_threshold = 0.5  # Pearson r > 0.5 indicates strong positive correlation
    
    combined_df['activation_status'] = combined_df['p65_activation'].apply(
        lambda x: 'Activated' if x > activation_threshold else 'Not Activated'
    )

    # Calculate fraction of activated cells per condition and timepoint
    activation_summary = combined_df.groupby(['Condition', 'Timepoint', 'activation_status']).size().reset_index(name='count')
    
    # Calculate total cells per condition and timepoint
    total_cells = combined_df.groupby(['Condition', 'Timepoint']).size().reset_index(name='total')
    
    # Merge and calculate fraction
    activation_summary = activation_summary.merge(total_cells, on=['Condition', 'Timepoint'])
    activation_summary['fraction'] = activation_summary['count'] / activation_summary['total']

    # Get only activated cells for plotting
    activated_fraction = activation_summary[activation_summary['activation_status'] == 'Activated'].copy()

    # Plot 1: Line plot of activation fraction over time
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, condition in enumerate(condition_order):
        condition_data = activated_fraction[activated_fraction['Condition'] == condition]
        
        if not condition_data.empty:
            ax.plot(condition_data['Timepoint'], condition_data['fraction'],
                   marker='o', linewidth=2.5, markersize=8,
                   color=condition_palette[idx], label=condition)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Fraction of Activated Cells')
    ax.set_title(f'Fraction of Activated Cells Over Time - Plate 006A\n(Activated = Pearson r > {activation_threshold})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_filename = plot_output_dir / "06_lineplot_activated_fraction_over_time.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)

    # Plot 2: Heatmap of activation fraction
    pivot_fraction = activated_fraction.pivot_table(
        values='fraction',
        index='Condition',
        columns='Timepoint',
        aggfunc='mean'
    )

    # Reorder rows to match condition_order
    pivot_fraction = pivot_fraction.reindex([c for c in condition_order if c in pivot_fraction.index])

    if not pivot_fraction.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            pivot_fraction,
            annot=True,
            fmt='.2%',
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Fraction Activated'},
            linewidths=1,
            linecolor='gray',
            ax=ax
        )

        ax.set_title(f'Fraction of Activated Cells Heatmap - Plate 006A\n(Activated = Pearson r > {activation_threshold})')
        ax.set_xlabel('Timepoint (minutes)')
        ax.set_ylabel('Condition')

        plt.tight_layout()
        plot_filename = plot_output_dir / "07_heatmap_activated_fraction.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename.name}")
        plt.close(fig)


    # %% [markdown]
    # ### Plot 8: S536 and Total p65 Signals Separately

    # %%
    # Plot S536 p65 over time
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=combined_df,
        x='Timepoint',
        y='s536_mean',
        hue='Condition',
        hue_order=condition_order,
        palette=condition_palette,
        errorbar='se',
        capsize=0.1,
        ax=ax
    )

    ax.set_title("S536 p65 Protein (Activated) Over Time - Plate 006A")
    ax.set_ylabel("S536 p65 Protein\n(Mean Intensity)")
    ax.set_xlabel("Timepoint (minutes)")
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_filename = plot_output_dir / "08_barplot_s536_p65_over_time.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)

    # Plot Total p65 over time
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=combined_df,
        x='Timepoint',
        y='total_mean',
        hue='Condition',
        hue_order=condition_order,
        palette=condition_palette,
        errorbar='se',
        capsize=0.1,
        ax=ax
    )

    ax.set_title("Total p65 Protein Over Time - Plate 006A")
    ax.set_ylabel("Total p65 Protein\n(Mean Intensity)")
    ax.set_xlabel("Timepoint (minutes)")
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_filename = plot_output_dir / "09_barplot_total_p65_over_time.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)


    # %% [markdown]
    # ### Plot 8B: S536 and Total p65 Signals Separately (with Statistical Significance)

    # %%
    # Plot S536 p65 over time with statistical significance
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(
        data=combined_df,
        x='Timepoint',
        y='s536_mean',
        hue='Condition',
        hue_order=condition_order,
        palette=condition_palette,
        errorbar='se',
        capsize=0.1,
        ax=ax
    )

    ax.set_title("S536 p65 Protein (Activated) Over Time - Plate 006A\n(with Statistical Significance)", fontsize=14)
    ax.set_ylabel("S536 p65 Protein\n(Mean Intensity)")
    ax.set_xlabel("Timepoint (minutes)")
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add statistical annotations
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.15)  # Extend to accommodate significance brackets
    
    add_stat_annotation_barplot(
        ax=ax,
        data=combined_df,
        x_col='Timepoint',
        y_col='s536_mean',
        hue_col='Condition',
        hue_order=condition_order,
        control_condition=control_condition
    )

    plt.tight_layout()
    plot_filename = plot_output_dir / "08_barplot_s536_p65_over_time_with_stats.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)

    # Plot Total p65 over time with statistical significance
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(
        data=combined_df,
        x='Timepoint',
        y='total_mean',
        hue='Condition',
        hue_order=condition_order,
        palette=condition_palette,
        errorbar='se',
        capsize=0.1,
        ax=ax
    )

    ax.set_title("Total p65 Protein Over Time - Plate 006A\n(with Statistical Significance)", fontsize=14)
    ax.set_ylabel("Total p65 Protein\n(Mean Intensity)")
    ax.set_xlabel("Timepoint (minutes)")
    ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add statistical annotations
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.15)  # Extend to accommodate significance brackets
    
    add_stat_annotation_barplot(
        ax=ax,
        data=combined_df,
        x_col='Timepoint',
        y_col='total_mean',
        hue_col='Condition',
        hue_order=condition_order,
        control_condition=control_condition
    )

    plt.tight_layout()
    plot_filename = plot_output_dir / "09_barplot_total_p65_over_time_with_stats.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_filename.name}")
    plt.close(fig)


    # %% [markdown]
    # ### Plot 9: Scatter plots of S536 vs Total p65

    # %%
    # Create scatter plots for each timepoint
    for timepoint in sorted(combined_df['Timepoint'].unique()):
        df_timepoint = combined_df[combined_df['Timepoint'] == timepoint].copy()

        if not df_timepoint.empty:
            fig, ax = plt.subplots(figsize=(10, 8))

            for idx, condition in enumerate(condition_order):
                condition_data = df_timepoint[df_timepoint['Condition'] == condition]
                
                if not condition_data.empty:
                    ax.scatter(condition_data['total_mean'], condition_data['s536_mean'],
                             alpha=0.5, s=20, color=condition_palette[idx], label=condition)

            # Add diagonal line (perfect correlation)
            max_val = max(df_timepoint['total_mean'].max(), df_timepoint['s536_mean'].max())
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect correlation')

            ax.set_xlabel('Total p65 Protein (Mean Intensity)')
            ax.set_ylabel('S536 p65 Protein (Mean Intensity)')
            ax.set_title(f'S536 p65 vs Total p65 at {timepoint} Minutes - Plate 006A')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)

            plt.tight_layout()
            plot_filename = plot_output_dir / f"10_scatter_s536_vs_total_p65_{timepoint}min.png"
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
