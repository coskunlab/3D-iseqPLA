"""
Gather all data files referenced by analysis scripts into ../data/ directory.

This script copies all input data from absolute paths on the Y: drive into a local
data directory with preserved folder hierarchy, making the repo portable.

Usage:
    python 00_gather_figures_data.py          # Interactive with size confirmation
    python 00_gather_figures_data.py --yes    # Skip confirmation
    python 00_gather_figures_data.py --dry-run # List files without copying
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import warnings

from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import pandas as pd
except ImportError:
    pd = None
    warnings.warn("pandas not available - Excel-derived paths will be skipped")


# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'

# Prefixes to strip from absolute paths to create relative paths
STRIP_PREFIXES = [
    r'Y:\coskun-lab\Nicky\\',
    r'Y:/coskun-lab/Nicky/',
    r'/coskun-lab/Nicky/',
]

# Hardcoded registry of (source_path, type) tuples
# Type is 'file' or 'dir'
INPUT_PATHS: List[Tuple[str, str]] = [
    # Script 25
    (r'Y:\coskun-lab\Nicky\49 Cystic Fibrosis - Rabin\Data\24 well plate 006C\Plate006C_layout.xlsx', 'file'),
    (r'Y:\coskun-lab\Nicky\49 Cystic Fibrosis - Rabin\Data\24 well plate 006C\29Dec2025 cycle 1 IF\09 PKL single cell', 'dir'),

    # Scripts 29, 37
    (r'Y:\coskun-lab\Nicky\71 CF AI Foundation model\Data\00 In Vitro RAW\converted_anndata', 'dir'),

    # Script 35
    (r'Y:\coskun-lab\Nicky\71 CF AI Foundation model\Data\00 In Vivo RAW', 'dir'),
    (r'Y:\coskun-lab\Nicky\71 CF AI Foundation model\Data\Prepared_for_Training', 'dir'),

    # Scripts 38, 39, 42
    (r'Y:\coskun-lab\Nicky\71 CF AI Foundation model\Data\01_select_genes.txt', 'file'),

    # Scripts 29, 37, 38, 39 - multispecies model
    (r'Y:\coskun-lab\Nicky\71 CF AI Foundation model\Models\scGPT\multispecies', 'dir'),

    # Scripts 35, 42, 44, 46 - in vivo model
    (r'Y:\coskun-lab\Nicky\71 CF AI Foundation model\Models\scGPT\invivo', 'dir'),

    # Scripts 77, 91 - Plate015
    (r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P8 24 well plate 015\17Oct2025_Plate015_multiplex_cycles1,4,5,6.xlsx', 'file'),
    (r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P8 24 well plate 015\Plate015_layout.xlsx', 'file'),

    # Script 88
    (r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\08 compare plate coatings\01_compare_plate_coatings.xlsx', 'file'),

    # Scripts 90, 94 - Plate021
    (r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P11 24 well plate 021\Plate021_layout.xlsx', 'file'),
    (r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P11 24 well plate 021\26Dec2025 cycle 3 PLA\10 PKL single cell', 'dir'),

    # Script 100
    (r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P11 24 well plate 021\1Jan2026_Plate021_multiplex.xlsx', 'file'),
]


# ==============================================================================
# Helper Functions
# ==============================================================================

def strip_prefix(path_str: str) -> str:
    """
    Strip known absolute prefixes to get relative path.

    Args:
        path_str: Absolute path string

    Returns:
        Relative path string with prefix removed
    """
    path_normalized = path_str.replace('\\', '/')

    for prefix in STRIP_PREFIXES:
        prefix_normalized = prefix.replace('\\', '/')
        if path_normalized.startswith(prefix_normalized):
            return path_normalized[len(prefix_normalized):]

    # If no prefix matches, return as-is
    return path_str


def resolve_derived_paths() -> List[Tuple[str, str]]:
    """
    Resolve paths derived from Excel files at runtime.

    Returns:
        List of (source_path, type) tuples for derived paths
    """
    if pd is None:
        warnings.warn("pandas not available - skipping Excel-derived paths")
        return []

    derived_paths = []

    # Scripts 77, 91: Read Plate015 Excel StitchPath -> parent / '10 PKL single cell'
    try:
        plate015_excel = Path(r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P8 24 well plate 015\17Oct2025_Plate015_multiplex_cycles1,4,5,6.xlsx')
        if plate015_excel.exists():
            df = pd.read_excel(plate015_excel)
            if 'StitchPath' in df.columns:
                stitch_paths = df['StitchPath'].dropna().unique()
                for sp in stitch_paths:
                    parent = Path(sp).parent
                    pkl_dir = parent / '10 PKL single cell'
                    if pkl_dir.exists():
                        derived_paths.append((str(pkl_dir), 'dir'))
    except Exception as e:
        warnings.warn(f"Could not read Plate015 Excel: {e}")

    # Script 88: Read comparison Excel PklPath and LayoutFile columns
    try:
        comparison_excel = Path(r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\08 compare plate coatings\01_compare_plate_coatings.xlsx')
        if comparison_excel.exists():
            df = pd.read_excel(comparison_excel)

            if 'PklPath' in df.columns:
                for pkl_path in df['PklPath'].dropna().unique():
                    pkl_path_obj = Path(pkl_path)
                    if pkl_path_obj.exists():
                        derived_paths.append((str(pkl_path_obj), 'dir'))

            if 'LayoutFile' in df.columns:
                for layout_file in df['LayoutFile'].dropna().unique():
                    layout_file_obj = Path(layout_file)
                    if layout_file_obj.exists():
                        derived_paths.append((str(layout_file_obj), 'file'))
    except Exception as e:
        warnings.warn(f"Could not read comparison Excel: {e}")

    # Script 100: Read Plate021 Excel StitchPath -> parent contains multiple dirs
    try:
        plate021_excel = Path(r'Y:\coskun-lab\Nicky\48 NFkB gradient on chip\Data\01-3T3 P11 24 well plate 021\1Jan2026_Plate021_multiplex.xlsx')
        if plate021_excel.exists():
            df = pd.read_excel(plate021_excel)
            if 'StitchPath' in df.columns:
                stitch_paths = df['StitchPath'].dropna().unique()
                for sp in stitch_paths:
                    parent = Path(sp).parent

                    # Add subdirectories
                    for subdir in ['08 TIF registered 3D volumes',
                                   '09 TIF masks 2D MIP tiles registered on p65',
                                   '10 PKL single cell']:
                        subdir_path = parent / subdir
                        if subdir_path.exists():
                            derived_paths.append((str(subdir_path), 'dir'))

                    # Add CSV files in parent
                    if parent.exists():
                        for csv_file in parent.glob('*.csv'):
                            derived_paths.append((str(csv_file), 'file'))
    except Exception as e:
        warnings.warn(f"Could not read Plate021 Excel: {e}")

    return derived_paths


def get_size(path: Path) -> int:
    """
    Get size of file or directory in bytes.

    Args:
        path: Path to file or directory

    Returns:
        Size in bytes
    """
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass
        return total
    return 0


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., '1.5 GB')
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def copy_path(source: Path, path_type: str, data_root: Path, dry_run: bool = False) -> bool:
    """
    Copy a file or directory to data_root, preserving hierarchy.

    Args:
        source: Source path to copy
        path_type: 'file' or 'dir'
        data_root: Root directory for data
        dry_run: If True, don't actually copy

    Returns:
        True if successful, False otherwise
    """
    if not source.exists():
        warnings.warn(f"Source does not exist: {source}")
        return False

    # Get relative path
    rel_path = strip_prefix(str(source))
    dest = data_root / rel_path

    # Check if already exists
    if dest.exists():
        if dry_run:
            print(f"  [EXISTS] {rel_path}")
        return True

    if dry_run:
        size = get_size(source)
        print(f"  [COPY] {rel_path} ({format_size(size)})")
        return True

    try:
        # Create parent directories
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy
        if path_type == 'file':
            shutil.copy2(source, dest)
        elif path_type == 'dir':
            shutil.copytree(source, dest, dirs_exist_ok=True)

        return True
    except Exception as e:
        warnings.warn(f"Failed to copy {source}: {e}")
        return False


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gather all data files into ../data/ directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='List files without copying')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip size confirmation prompt')

    args = parser.parse_args()

    print("=" * 80)
    print("Data Gathering Script")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    print()

    # Collect all paths
    print("Collecting paths...")
    all_paths = INPUT_PATHS.copy()

    # Add derived paths from Excel files
    derived = resolve_derived_paths()
    if derived:
        print(f"  Found {len(derived)} derived paths from Excel files")
        all_paths.extend(derived)

    # Remove duplicates
    all_paths = list(set(all_paths))

    print(f"  Total paths: {len(all_paths)}")
    print()

    # Estimate total size
    print("Estimating sizes...")
    total_size = 0
    valid_paths = []

    for path_str, path_type in all_paths:
        source = Path(path_str)
        if source.exists():
            size = get_size(source)
            total_size += size
            valid_paths.append((source, path_type))
        else:
            warnings.warn(f"Path does not exist: {path_str}")

    print(f"  Valid paths: {len(valid_paths)}/{len(all_paths)}")
    print(f"  Total size: {format_size(total_size)}")
    print()

    # Confirm unless --yes or --dry-run
    if not args.yes and not args.dry_run:
        response = input(f"Copy {format_size(total_size)} to {DATA_ROOT}? [y/N] ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return

    # Copy files
    if args.dry_run:
        print("DRY RUN - files that would be copied:")
        print("-" * 80)
        success_count = 0
        for source, path_type in tqdm(valid_paths, desc="Scanning"):
            if copy_path(source, path_type, DATA_ROOT, dry_run=True):
                success_count += 1
    else:
        print("Copying files in parallel (using all CPU cores)...")
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

        # Parallel copy using all available cores (n_jobs=-1)
        # Note: For network I/O, fewer jobs (4-8) is often optimal, but using all cores as requested
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(copy_path)(source, path_type, DATA_ROOT, dry_run=False)
            for source, path_type in valid_paths
        )
        success_count = sum(results)

    print()
    print("=" * 80)
    print(f"Complete: {success_count}/{len(valid_paths)} paths copied")

    if args.dry_run:
        print("(Dry run - no files were actually copied)")

    print("=" * 80)


if __name__ == '__main__':
    main()
