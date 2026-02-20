# %% [markdown]
# # Measure PLA Supercomplex Sizes
# This script reads existing PKL files (with cell segmentation and dot locations filtered to cells),
# applies rolling ball background subtraction to MIP images, maps dots to connected regions,
# measures supercomplex areas, and generates distribution plots.
#
# Prerequisites: The main processing script has already been run, generating:
# - PKL files with single-cell data including dot locations (filtered to cells only)
# - 3D TIF images for creating MIPs

# %%
import numpy as np
import pandas as pd
import time
import os
import sys
import re
from tqdm import tqdm, trange
from datetime import date
import nd2reader
from joblib import Parallel, delayed
import tifffile as tf
import xml.etree.ElementTree as ET
import napari
import skimage
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.filters import try_all_threshold
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask
import matplotlib
import matplotlib.pyplot as plt
import nd2
from skimage import exposure, restoration
from joblib import Parallel, delayed
import torch
import cellpose
import cv2
import scipy
from scipy import ndimage as ndi
from skimage.morphology import closing, square
from skimage.measure import label
from skimage.segmentation import watershed
import seaborn as sns
import math
import sklearn
from sklearn.neighbors import KDTree
import networkx
from networkx.algorithms.components.connected import connected_components
from pathlib import Path
import statistics
import scanpy as sc
from anndata import AnnData
import dask_image.imread
import psutil

# %% [markdown]
# # directories and inputs

# %%
# Setup paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'
FIGURES_ROOT = PROJECT_ROOT / 'figures'

#%% directories - USER INPUTS
# read EXCEL file for channel info
channels_excel = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P11 24 well plate 021' / '1Jan2026_Plate021_multiplex.xlsx'
channels = pd.read_excel(channels_excel)
channels.dropna(subset = ['StitchPath'], inplace=True)
print(channels)

# Extract cycle folder name from StitchPath
stitchPath = Path(channels['StitchPath'].iloc[-1])
cycleFolderName = stitchPath.parent.name
# Construct base path relative to DATA_ROOT
basePath = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P11 24 well plate 021' / cycleFolderName

# folder of registered TIFs in ZYX
sigPath = Path(basePath, '08 TIF registered 3D volumes')
assert sigPath.exists()

# folder of registered
maskPath = Path(basePath, '09 TIF masks 2D MIP tiles registered on p65')
assert maskPath.exists()

# folder to export dataframes
dfPath = Path(basePath, '10 PKL single cell')
dfPath.mkdir(exist_ok=True)

# --- NEW CONSOLIDATED FOLDER FOR FIJI WORKFLOW ---
# A single subfolder inside the main output folder for macros and CSV results.
fijiWorkflowPath = Path(dfPath, 'fiji_workflow')
fijiWorkflowPath.mkdir(exist_ok=True, parents=True)
macroPath = fijiWorkflowPath
fijiCsvPath = fijiWorkflowPath
# --- END NEW FOLDER ---

# folder to save supercomplex plots
plotPath = FIGURES_ROOT / '100_measure_supercomplex_sizes'
plotPath.mkdir(exist_ok=True, parents=True)

csvFile = [f for f in basePath.glob('*.csv')]
assert len(csvFile) == 1
coords = pd.read_csv(csvFile[0])

# list of FOVs to skip analysis
skipFovs = []

# --- NEW PARAMETER FOR FIJI ---
# Prominence value for Fiji's "Find Maxima" command for RNA/PLA dot detection.
# This value may need to be adjusted based on image signal-to-noise.
rna_dot_prominence = 1000
# --- END NEW PARAMETER ---

imgRes = 0.112 # resolution of images, um/pixel, 60X water on Squid
# imgRes = 0.1082773 # resolution of images, um/pixel, 60X
# imgRes = 0.18872 # resolution of images, um/pixel, 40X
# imgRes = 0.325 # resolution of images, um/pixel, 20X
# scaleBarWidth = 20 # um, to show on images

# %% [markdown]
# # Helper functions

# %%
# apply threshold
def applyThresholdMethod(img, choice):
    threshVal = None
    if choice == 'Otsu':
        threshVal = skimage.filters.threshold_otsu(img)
    elif choice == 'Triangle':
        threshVal = skimage.filters.threshold_triangle(img)
    # ... (other threshold methods can be added back if needed)
    else:
        threshVal = 0
    return threshVal

# find outline, and blur to thicken outline
def outlineMask(imgMask, ksize):
    # contour masks and add as layer
    maskEdge = skimage.filters.roberts(imgMask)
    maskEdge[maskEdge > 0] = 255
    maskEdge = cv2.blur(maskEdge, ksize = ksize) # blur mask to make edges thicker
    return maskEdge

# discard small mask artifacts. or by circularity
def discardMaskArtifacts(mask):
    global imgRes
    
    area = skimage.measure.regionprops_table(mask, properties = ['label', 'area', 'perimeter'])
    dfArea = pd.DataFrame(area)
    dfArea.rename(columns = {'label': 'CellLabel'}, inplace = True)
    
    # compute circularity score
    dfArea['Circularity'] = np.true_divide(4 * math.pi * dfArea['area'], np.square(dfArea['perimeter']))

    # apply thresh to mask
    minArea = np.pi * (5 / imgRes) ** 2 # assume 5 um radius
    minCirc = 0.6
    for ii, label in enumerate(dfArea['CellLabel']):
        if dfArea['area'].iloc[ii] < minArea and dfArea['Circularity'].iloc[ii] < minCirc: # discard
            mask[mask == label] = 0
            continue
        # also discard circularities greater than 1
        if dfArea['Circularity'].iloc[ii] > 1:
            mask[mask == label] = 0
            continue
            
    return mask

lazy_discard_mask_artifacts = dask.delayed(discardMaskArtifacts)

# read mask for nucleus or cytosol
def readMask(wellLabel, cellRegion, maskPath):
    # Map the cellRegion parameter to the actual filename pattern
    if cellRegion == 'cyto':
        region_pattern = 'whole_cell_mask'
    elif cellRegion == 'nuc':
        region_pattern = 'nuclear_mask'
    else:
        region_pattern = cellRegion

    maskFile = [f for f in maskPath.glob('*.tif') if f.stem.startswith(f'{wellLabel}_') and
                region_pattern in f.name and 'mask' in f.name]
    if len(maskFile) == 0:
        return None, None

    print('Reading mask', maskFile[0].stem)
    imgMask = dask_image.imread.imread(maskFile[0])
 
    # discard invalid masks
    if imgMask.ndim == 2:
        cleaned = lazy_discard_mask_artifacts(imgMask)
        imgMask = da.from_delayed(cleaned, shape = imgMask.shape, dtype = imgMask.dtype)
    else: # 3D
        zStack = []
        for ii in range(imgMask.shape[0]):
            plane = imgMask[ii, ...]
            cleaned = lazy_discard_mask_artifacts(plane)
            cleaned = da.from_delayed(cleaned, shape = plane.shape, dtype = plane.dtype)
            zStack.append(cleaned)
        imgMask = da.stack(zStack, axis = 0)

    print('Computing cleaned', maskFile[0].stem, 'into RAM')
    with ProgressBar():
        imgMask = imgMask.compute()

    # Remove singleton dimensions (e.g., shape (1, Y, X) -> (Y, X))
    imgMask = np.squeeze(imgMask)

    # convert mask to dataframe
    if imgMask.ndim == 2:
        y, x = np.where(imgMask > 0)
        z = np.zeros_like(y) # Add Z for consistency if mask is 2D
    else:
        z, y, x = np.where(imgMask > 0)

    gray = imgMask[imgMask > 0]
    df = pd.DataFrame()
    # Explicitly cast to int64 to ensure consistent dtypes across merges
    df['Z'] = z.astype(np.int64)
    df['Y'] = y.astype(np.int64)
    df['X'] = x.astype(np.int64)
    df['CellLabel' + cellRegion.capitalize()] = gray

    return imgMask, df

# %% [markdown]
# # Fiji Macro Generation and Dot Reading
# The following functions handle the new workflow for RNA/PLA dot detection.
# 1. `generate_fiji_macro`: Creates an `.ijm` macro file. You need to run this in Fiji.
# 2. `read_fiji_dots_and_assign_z`: Reads the CSV output from Fiji and maps dots back to the 3D image.

# %%
def generate_fiji_macro(channels, coords, sigPath, macroPath, fijiCsvPath, prominence):
    """
    Generates a single Fiji macro file to process all RNA/PLA images.
    The macro creates a maximum intensity projection, runs Find Maxima, and saves coordinates to a CSV.
    """
    macro_filename = macroPath / 'find_maxima_macro.ijm'
    print(f"Generating Fiji macro at: {macro_filename}")

    with open(macro_filename, 'w') as f:
        f.write('// This macro was generated by a Python script.\n')
        f.write('// It processes all RNA/PLA images to find dots using Find Maxima.\n\n')
        f.write('setBatchMode(true);\n\n') # Run in batch mode for speed

        # Filter for channels that are RNA or PLA
        rna_pla_channels = channels[channels['SignalType'].str.contains('RNA|PLA', na=False)]

        for _, row1 in coords.iterrows(): # each FOV
            for _, row_channel in rna_pla_channels.iterrows(): # each RNA/PLA channel
                cycleNum = row_channel['Cycle']
                markerName = row_channel['Marker']

                # Find the corresponding TIFF file for the current FOV and marker
                fileMarker_list = list(sigPath.glob(f'{row1.ID}_*Cycle{str(cycleNum).zfill(2)}*{markerName}*.tif'))

                if not fileMarker_list:
                    print(f"Warning: TIF file not found for {row1.ID}, {markerName}. Skipping in macro.")
                    continue
                
                inputFile = fileMarker_list[0]
                base_name = inputFile.stem
                outputFile = fijiCsvPath / f'{base_name}_Results.csv'

                # Fiji needs paths with forward slashes, especially on Windows
                inputPath_fiji = str(inputFile).replace('\\', '/')
                outputPath_fiji = str(outputFile).replace('\\', '/')

                f.write(f'// Processing: {base_name}\n')
                f.write(f'run("Bio-Formats Importer", "open=[{inputPath_fiji}] windowless=true");\n')
                f.write(f'rename("{base_name}.tif");\n')
                f.write(f'run("Z Project...", "projection=[Max Intensity]");\n')
                f.write(f'selectWindow("{base_name}.tif");\n')
                f.write('close();\n')
                f.write(f'selectWindow("MAX_{base_name}.tif");\n')
                f.write(f'run("Find Maxima...", "prominence={prominence} output=List");\n')
                f.write(f'saveAs("Results", "{outputPath_fiji}");\n')
                f.write('run("Clear Results");\n')
                f.write('selectWindow("Results");\n')
                f.write('close();\n')
                
                # --- ROBUST WINDOW CLEANUP ---
                # Checks if the MAX projection window is still open before trying to close it.
                f.write(f'if(isOpen("MAX_{base_name}.tif")) {{\n')
                f.write(f'  selectWindow("MAX_{base_name}.tif");\n')
                f.write(f'  close();\n')
                f.write(f'}}\n\n')
        
        f.write('setBatchMode(false);\n')
        f.write('print("Macro finished!");\n')

    print("--- ACTION REQUIRED ---")
    print("Fiji macro generation complete.")
    print("1. Open Fiji/ImageJ.")
    print("2. Run the generated macro script found at:")
    print(f"   {macro_filename}")
    print("3. Once the macro is finished, it will save all result CSVs in:")
    print(f"   {fijiCsvPath}")
    print("4. You can then continue running the rest of this Python script.")
    print("-----------------------")
    return

def read_fiji_dots_and_measure_supercomplexes(fiji_csv_path, imgMarker3D, markerName, cytoMask2D):
    """
    Reads dot coordinates from a Fiji Find Maxima CSV.
    Creates MIP from 3D image, applies rolling ball background subtraction,
    thresholds at 0, and measures the area of connected regions containing dots.
    Returns a dataframe with one row per dot, including the supercomplex area.
    """
    if not os.path.exists(fiji_csv_path):
        print(f"Warning: Fiji results CSV not found at {fiji_csv_path}. Cannot measure supercomplexes for {markerName}.")
        return pd.DataFrame()

    try:
        df_dots_mip = pd.read_csv(fiji_csv_path)
    except Exception as e:
        print(f"Error reading CSV {fiji_csv_path}: {e}")
        return pd.DataFrame()

    if 'X' not in df_dots_mip.columns or 'Y' not in df_dots_mip.columns:
        print(f"Error: CSV {fiji_csv_path} does not contain 'X' and 'Y' columns.")
        return pd.DataFrame()

    if df_dots_mip.empty:
        print(f"CSV {fiji_csv_path} is empty.")
        return pd.DataFrame()

    print(f"Found {len(df_dots_mip)} dots in CSV for {markerName}")

    # Create MIP from 3D image
    print(f"Creating MIP for {markerName}...")
    imgMIP = np.max(imgMarker3D, axis=0)
    
    # Apply rolling ball background subtraction (radius = 10)
    print(f"Applying rolling ball background subtraction (radius=10) to {markerName}...")
    background = skimage.restoration.rolling_ball(imgMIP, radius=10)
    imgMIP_subtracted = imgMIP.astype(np.float32) - background.astype(np.float32)
    
    # Threshold out signal < 0
    imgMIP_subtracted[imgMIP_subtracted < 0] = 0
    
    # Apply cell mask if 2D
    if cytoMask2D is not None and cytoMask2D.ndim == 2:
        imgMIP_subtracted = np.multiply(imgMIP_subtracted, cytoMask2D > 0)
    
    # Apply Triangle thresholding
    print(f"Applying Triangle thresholding to {markerName}...")
    triangle_threshold = skimage.filters.threshold_triangle(imgMIP_subtracted)
    imgMIP_binary = imgMIP_subtracted > triangle_threshold
    
    # Fiji coordinates are 1-based, numpy is 0-based. Subtract 1.
    y_coords = (df_dots_mip['Y'] - 1).astype(int)
    x_coords = (df_dots_mip['X'] - 1).astype(int)

    # Ensure coordinates are within image bounds
    max_y, max_x = imgMIP.shape[0] - 1, imgMIP.shape[1] - 1
    valid_indices = (y_coords >= 0) & (y_coords <= max_y) & (x_coords >= 0) & (x_coords <= max_x)
    y_coords = y_coords[valid_indices].to_numpy()
    x_coords = x_coords[valid_indices].to_numpy()

    if len(y_coords) == 0:
        print(f"No valid coordinates found after bounds checking for {markerName}")
        return pd.DataFrame()

    print(f"{len(y_coords)} valid dots after bounds checking for {markerName}")
    
    # Use watershed to separate touching supercomplexes
    # Create markers from dot locations (each dot is a separate region seed)
    print(f"Applying watershed segmentation to separate touching supercomplexes...")
    markers = np.zeros_like(imgMIP_binary, dtype=np.int32)
    for idx, (y, x) in enumerate(zip(y_coords, x_coords), start=1):
        markers[y, x] = idx
    
    # Compute distance transform for watershed
    distance = ndi.distance_transform_edt(imgMIP_binary)
    
    # Apply watershed segmentation
    labeled_regions = watershed(-distance, markers, mask=imgMIP_binary)
    
    # Get region properties
    region_props = skimage.measure.regionprops(labeled_regions, intensity_image=imgMIP_subtracted)
    region_areas = {prop.label: prop.area for prop in region_props}
    print(f"Watershed segmentation created {len(region_areas)} separate regions for {len(y_coords)} dots")

    # For each dot, find which region it belongs to and record the area
    dot_data = []
    for y, x in zip(y_coords, x_coords):
        region_label = labeled_regions[y, x]
        if region_label > 0:  # Dot is in a labeled region
            area = region_areas.get(region_label, 0)
        else:  # Dot is in background
            area = 0
        
        # Find the z-plane with max intensity in 3D stack for this dot
        z_coord = np.argmax(imgMarker3D[:, y, x])
        
        # Convert area from pixels to um^2 (1 pixel = 0.112 um x 0.112 um)
        area_um2 = area * (0.112 ** 2)
        
        dot_data.append({
            'Y': y,
            'X': x,
            'Z': z_coord,
            markerName + ' Dots': 1,
            markerName + ' Supercomplex Area (pixels)': area,
            markerName + ' Supercomplex Area (um^2)': area_um2,
            markerName + ' Intensity': imgMIP_subtracted[y, x]
        })
    
    dfMarker = pd.DataFrame(dot_data)
    print(f"Measured {len(dfMarker)} supercomplexes for {markerName}")
    
    return dfMarker

# %% [markdown]
# # Load PKL Files and Measure Supercomplex Areas
# Read the previously generated PKL files (containing cell segmentation and dot locations filtered to cells).
# For each dot location in the PKL, process the corresponding MIP image with background subtraction,
# find connected regions, and measure the area of the region containing each dot.

# %%
def process_single_pkl_file(pkl_file, channels, sigPath):
    """
    Process a single PKL file: load it, measure supercomplex areas, and return the result.
    This function is designed to be parallelized with joblib.
    """
    try:
        df_fov = pd.read_pickle(pkl_file)
        
        # Extract FOV ID from filename
        fov_id = pkl_file.stem
        print(f"\nProcessing {fov_id}...")
        
        # Find all dot columns in this dataframe (any marker with " Dots")
        pla_dot_cols = [col for col in df_fov.columns if ' Dots' in col]
        
        if not pla_dot_cols:
            print(f"  No dot markers found in {fov_id}")
            print(f"  Available columns: {df_fov.columns.tolist()}")
            return None
        
        print(f"  Found {len(pla_dot_cols)} dot marker(s): {pla_dot_cols}")
        
        # For each PLA marker, measure supercomplex areas
        for dot_col in pla_dot_cols:
            marker_name = dot_col.replace(' Dots', '').strip()
            area_col_pixels = marker_name + ' Supercomplex Area (pixels)'
            area_col_um2 = marker_name + ' Supercomplex Area (um^2)'
            
            # Skip if area column already exists
            if area_col_pixels in df_fov.columns:
                print(f"  {marker_name}: Area already measured, skipping")
                continue
            
            print(f"  Processing {marker_name}...")
            
            # Get dot locations from the PKL dataframe (only dots inside cells)
            mask_dots = df_fov[dot_col] > 0
            num_dots = mask_dots.sum()
            
            if num_dots == 0:
                print(f"    No dots found for {marker_name}, adding zero-filled columns")
                df_fov[area_col_pixels] = 0
                df_fov[area_col_um2] = 0
                continue
            
            print(f"    Found {num_dots} dots in PKL dataframe")
            
            # Get unique dot coordinates from dataframe
            df_dots_in_cells = df_fov[mask_dots][['Y', 'X']].drop_duplicates()
            print(f"    Found {len(df_dots_in_cells)} unique dot locations")
            
            # Find corresponding image file
            # Search for the marker in channels to get cycle number
            marker_cycle = None
            for _, row in channels.iterrows():
                # Try exact match or partial match
                if row['Marker'] == marker_name or marker_name in str(row['Marker']) or str(row['Marker']) in marker_name:
                    marker_cycle = row['Cycle']
                    break
            
            if marker_cycle is None:
                print(f"    Warning: Could not find cycle number for {marker_name}, skipping")
                print(f"    Available markers: {channels['Marker'].unique().tolist()}")
                df_fov[area_col_pixels] = 0
                df_fov[area_col_um2] = 0
                continue
            
            # Find the 3D TIF file
            img_files = list(sigPath.glob(f'{fov_id}_*Cycle{str(marker_cycle).zfill(2)}*{marker_name}*.tif'))
            if not img_files:
                print(f"    Warning: Image file not found for {marker_name}, skipping")
                df_fov[area_col_pixels] = 0
                df_fov[area_col_um2] = 0
                continue
            
            img_file = img_files[0]
            
            # Load the 3D image
            print(f"    Loading image: {img_file.name}")
            img_3d = tf.imread(img_file)
            
            # Create MIP
            print(f"    Creating MIP...")
            img_mip = np.max(img_3d, axis=0)
            
            # Apply rolling ball background subtraction (radius = 10)
            print(f"    Applying rolling ball background subtraction (radius=10)...")
            background = skimage.restoration.rolling_ball(img_mip, radius=10)
            img_mip_subtracted = img_mip.astype(np.float32) - background.astype(np.float32)
            
            # Threshold out signal < 0
            img_mip_subtracted[img_mip_subtracted < 0] = 0
            
            # Apply Triangle thresholding
            print(f"    Applying Triangle thresholding...")
            triangle_threshold = skimage.filters.threshold_triangle(img_mip_subtracted)
            img_mip_binary = img_mip_subtracted > triangle_threshold
            
            # Use watershed to separate touching supercomplexes
            # Create markers from dot locations (each dot is a separate region seed)
            print(f"    Applying watershed segmentation...")
            markers = np.zeros_like(img_mip_binary, dtype=np.int32)
            for idx, (_, dot_row) in enumerate(df_dots_in_cells.iterrows(), start=1):
                y = int(dot_row['Y'])
                x = int(dot_row['X'])
                if 0 <= y < img_mip.shape[0] and 0 <= x < img_mip.shape[1]:
                    markers[y, x] = idx
            
            # Compute distance transform for watershed
            distance = ndi.distance_transform_edt(img_mip_binary)
            
            # Apply watershed segmentation
            labeled_regions = watershed(-distance, markers, mask=img_mip_binary)
            
            # Get region properties
            region_props = skimage.measure.regionprops(labeled_regions)
            region_areas = {prop.label: prop.area for prop in region_props}
            
            print(f"    Watershed segmentation created {len(region_areas)} separate regions for {len(df_dots_in_cells)} dots")
            
            # Map each dot coordinate to its region area
            try:
                # Create a mapping of (Y, X) -> supercomplex area
                # Use dot coordinates from the PKL dataframe (already filtered to be inside cells)
                dot_area_map = {}
                
                for _, dot_row in df_dots_in_cells.iterrows():
                    y = int(dot_row['Y'])
                    x = int(dot_row['X'])
                    
                    # Check bounds
                    if 0 <= y < img_mip.shape[0] and 0 <= x < img_mip.shape[1]:
                        region_label = labeled_regions[y, x]
                        if region_label > 0:
                            area = region_areas.get(region_label, 0)
                        else:
                            area = 0
                        dot_area_map[(y, x)] = area
                    else:
                        print(f"    Warning: Dot at ({y}, {x}) is out of bounds")
                
                print(f"    Mapped {len(dot_area_map)} dots to regions")
                
                # Add area columns to dataframe by matching coordinates
                def get_area(row):
                    if row[dot_col] > 0:
                        y = int(row['Y'])
                        x = int(row['X'])
                        return dot_area_map.get((y, x), 0)
                    else:
                        return 0
                
                df_fov[area_col_pixels] = df_fov.apply(get_area, axis=1)
                # Convert to um^2 (1 pixel = 0.112 um x 0.112 um)
                df_fov[area_col_um2] = df_fov[area_col_pixels] * (0.112 ** 2)
                
                # Report statistics
                non_zero_areas_pixels = df_fov[df_fov[area_col_pixels] > 0][area_col_pixels]
                non_zero_areas_um2 = df_fov[df_fov[area_col_um2] > 0][area_col_um2]
                if len(non_zero_areas_pixels) > 0:
                    print(f"    Area stats: mean={non_zero_areas_pixels.mean():.1f} pixels ({non_zero_areas_um2.mean():.3f} um^2), median={non_zero_areas_pixels.median():.1f} pixels ({non_zero_areas_um2.median():.3f} um^2), max={non_zero_areas_pixels.max():.0f} pixels ({non_zero_areas_um2.max():.3f} um^2)")
                print(f"    Added area measurements to {mask_dots.sum()} rows")
                
            except Exception as e:
                print(f"    Error processing {marker_name}: {e}")
                import traceback
                traceback.print_exc()
                # Add zero-filled columns if error
                df_fov[area_col_pixels] = 0
                df_fov[area_col_um2] = 0
            
            # Clean up
            del img_3d, img_mip, img_mip_subtracted, labeled_regions
        
        # Create a supercomplex-level dataframe (one row per dot)
        # This will be much smaller than the full pixel-level dataframe
        if 'CellLabel' in df_fov.columns and pla_dot_cols:
            supercomplex_rows = []
            
            for dot_col in pla_dot_cols:
                marker_name = dot_col.replace(' Dots', '').strip()
                area_col_pixels = marker_name + ' Supercomplex Area (pixels)'
                area_col_um2 = marker_name + ' Supercomplex Area (um^2)'
                
                # Get rows where this marker has dots
                mask_dots = df_fov[dot_col] > 0
                if mask_dots.sum() > 0 and area_col_pixels in df_fov.columns:
                    # Extract relevant columns for each dot
                    cols_to_extract = ['CellLabel', 'FOV', area_col_pixels]
                    if area_col_um2 in df_fov.columns:
                        cols_to_extract.append(area_col_um2)
                    
                    df_dots_subset = df_fov[mask_dots][cols_to_extract].copy()
                    df_dots_subset['Marker'] = marker_name
                    df_dots_subset.rename(columns={
                        area_col_pixels: 'Supercomplex_Area_pixels',
                        area_col_um2: 'Supercomplex_Area_um2'
                    }, inplace=True)
                    supercomplex_rows.append(df_dots_subset)
            
            if supercomplex_rows:
                df_supercomplexes = pd.concat(supercomplex_rows, ignore_index=True)
                print(f"  Created supercomplex dataframe: {len(df_supercomplexes)} supercomplexes from {len(df_fov)} pixels")
                return df_supercomplexes
            else:
                print(f"  No supercomplexes with area measurements found")
                return None
        else:
            print(f"  Skipping aggregation (no CellLabel or dot columns)")
            return None
        
    except Exception as e:
        print(f"Warning: Could not process {pkl_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# %%
print("\n" + "="*80)
print("Loading existing PKL files and measuring supercomplex areas...")
print("="*80 + "\n")

# Find all PKL files
pkl_files = list(dfPath.glob('*.pkl'))
print(f"Found {len(pkl_files)} PKL files to load.")

if len(pkl_files) == 0:
    print("ERROR: No PKL files found! Please run the main processing script first.")
    print(f"Expected location: {dfPath}")
    sys.exit(1)

# Process all PKL files in parallel using joblib
print(f"Processing {len(pkl_files)} PKL files in parallel...")
n_jobs = min(10, len(pkl_files))  # Use up to 4 parallel jobs to avoid overwhelming I/O
print(f"Using {n_jobs} parallel workers")

all_fov_data = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_single_pkl_file)(pkl_file, channels, sigPath) 
    for pkl_file in pkl_files
)

# Filter out None results
all_fov_data = [df for df in all_fov_data if df is not None]
print(f"\nSuccessfully processed {len(all_fov_data)} FOV datasets.")

# %% [markdown]
# # Analyze and Plot Supercomplex Distributions

# %%
print("\n" + "="*80)
print("Analyzing Supercomplex Distributions")
print("="*80 + "\n")

if not all_fov_data:
    print("No data collected. Exiting.")
    sys.exit(1)
else:
    # Combine all FOV data
    print("Combining all FOV supercomplex data...")
    df_all = pd.concat(all_fov_data, ignore_index=True)
    print(f"Total combined supercomplex data shape: {df_all.shape}")
    print(f"Columns: {df_all.columns.tolist()}")
    
    if len(df_all) == 0:
        print("No supercomplexes found in any FOV. Exiting.")
        sys.exit(1)
    
    # Load plate layout (should be in the same directory as the channels Excel file)
    channels_excel_dir = DATA_ROOT / '48 NFkB gradient on chip' / 'Data' / '01-3T3 P11 24 well plate 021'
    layout_path = channels_excel_dir / 'Plate021_layout.xlsx'
    
    def parse_layout(layout_file_path):
        """Parses the plate layout Excel file to map wells to conditions."""
        try:
            df_layout = pd.read_excel(layout_file_path)
        except FileNotFoundError:
            print(f"Layout file not found at {layout_file_path}. Skipping layout mapping.")
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
    
    # Parse layout and add condition info
    if layout_path.exists():
        layout_df = parse_layout(layout_path)
        print(f"Loaded layout with {len(layout_df)} well-condition mappings")
        
        # Extract well from FOV (e.g., 'A1-1' -> 'A1')
        df_all['Well'] = df_all['FOV'].str.split('-').str[0]
        
        # Merge with layout
        df_all = df_all.merge(layout_df, on='Well', how='left')
        df_all['Condition'].fillna('Unknown', inplace=True)
        df_all['Timepoint'].fillna(-1, inplace=True)
    else:
        print("Layout file not found. Proceeding without condition mapping.")
        df_all['Condition'] = 'Unknown'
        df_all['Timepoint'] = -1
    
    # Check if we have the expected columns
    if 'Marker' not in df_all.columns:
        print("ERROR: Expected columns not found in combined dataframe")
        print(f"Available columns: {df_all.columns.tolist()}")
        sys.exit(1)
    
    # Use um^2 if available, otherwise fall back to pixels
    if 'Supercomplex_Area_um2' in df_all.columns:
        area_column = 'Supercomplex_Area_um2'
        area_unit = 'μm²'
    elif 'Supercomplex_Area_pixels' in df_all.columns:
        area_column = 'Supercomplex_Area_pixels'
        area_unit = 'pixels'
    else:
        print("ERROR: No area column found in combined dataframe")
        print(f"Available columns: {df_all.columns.tolist()}")
        sys.exit(1)
    
    # Get list of unique markers
    markers = df_all['Marker'].unique()
    print(f"\nFound {len(markers)} marker(s): {markers.tolist()}")
    
    if len(markers) == 0:
        print("No markers found. Cannot generate plots.")
        sys.exit(1)
    
    # Set plot style
    sns.set_style('whitegrid')
    sns.set(font_scale=1.5)
    plt.rcParams['figure.figsize'] = (10, 7)  # Set default figure size (width, height) in inches
    
    for marker_name in markers:
        print(f"\nProcessing {marker_name}...")
        
        # Filter to this marker
        df_marker = df_all[df_all['Marker'] == marker_name].copy()
        
        if df_marker.empty:
            print(f"  No data for {marker_name}. Skipping.")
            continue
        
        print(f"  Found {len(df_marker)} supercomplexes")
        
        # Calculate per-cell statistics
        # Group by CellLabel to get: number of supercomplexes, mean area
        cell_stats = df_marker.groupby(['CellLabel', 'Condition', 'Timepoint', 'Well']).agg({
            area_column: ['count', 'mean', 'sum', 'std']
        }).reset_index()
        
        # Flatten column names
        cell_stats.columns = ['CellLabel', 'Condition', 'Timepoint', 'Well',
                             'Num_Supercomplexes', 'Mean_Area', 'Total_Area', 'Std_Area']
            
        print(f"  Cell statistics shape: {cell_stats.shape}")
        print(f"  Mean supercomplexes per cell: {cell_stats['Num_Supercomplexes'].mean():.2f}")
        print(f"  Mean supercomplex area: {cell_stats['Mean_Area'].mean():.3f} {area_unit}")
        
        # Define condition order and colors
        condition_order = [c for c in ['TNFa', 'IL1B', 'DMSO Control']
                          if c in cell_stats['Condition'].unique()]
        if not condition_order:
            condition_order = sorted(cell_stats['Condition'].unique())
        
        palette = sns.color_palette("deep", len(condition_order))
        
        marker_name_clean = marker_name.replace(' ', '_')
        
        # --- PLOT 1: Number of Supercomplexes per Cell (Bar Plot) ---
        fig, ax = plt.subplots(dpi=300)
        
        sns.barplot(
            data=cell_stats,
            x='Timepoint',
            y='Num_Supercomplexes',
            hue='Condition',
            hue_order=condition_order,
            palette=palette,
            errorbar='se',
            capsize=0.1,
            ax=ax
        )
        
        ax.set_title(f"{marker_name}\nNumber of Supercomplexes per Cell")
        ax.set_ylabel("Number of\nSupercomplexes\nper Cell")
        ax.set_xlabel("Timepoint (minutes)")
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plot_filename = plotPath / f"barplot_{marker_name_clean}_num_supercomplexes_per_cell.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_filename.name}")
        plt.close(fig)
        
        # --- PLOT 2: Mean Supercomplex Area per Cell (Bar Plot) ---
        fig, ax = plt.subplots(dpi=300)
        
        sns.barplot(
            data=cell_stats,
            x='Timepoint',
            y='Mean_Area',
            hue='Condition',
            hue_order=condition_order,
            palette=palette,
            errorbar='se',
            capsize=0.1,
            ax=ax
        )
        
        ax.set_title(f"{marker_name}\nMean Supercomplex Area per Cell")
        ax.set_ylabel(f"Mean Supercomplex\nArea ({area_unit})")
        ax.set_xlabel("Timepoint (minutes)")
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plot_filename = plotPath / f"barplot_{marker_name_clean}_mean_area_per_cell.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_filename.name}")
        plt.close(fig)
        
        # --- PLOT 3: Distribution of Supercomplex Areas (Histogram per Condition) ---
        for condition in condition_order:
            df_cond = df_marker[df_marker['Condition'] == condition].copy()
            
            if df_cond.empty or df_cond[area_column].sum() == 0:
                continue
            
            fig, ax = plt.subplots(dpi=300)
            
            # Create histogram with multiple timepoints
            timepoints = sorted(df_cond['Timepoint'].unique())
            for tp in timepoints:
                df_tp = df_cond[df_cond['Timepoint'] == tp]
                areas = df_tp[area_column].values
                areas = areas[areas > 0]  # Remove zero areas
                
                if len(areas) > 0:
                    ax.hist(areas, bins=30, alpha=0.6, label=f'{tp} min', edgecolor='black')
            
            ax.set_title(f"{marker_name}\nSupercomplex Area Distribution - {condition}")
            ax.set_xlabel(f"Supercomplex Area ({area_unit})")
            ax.set_ylabel("Frequency")
            ax.legend(title='Timepoint', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            safe_condition = condition.replace(' ', '_')
            plot_filename = plotPath / f"histogram_{marker_name_clean}_area_distribution_{safe_condition}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_filename.name}")
            plt.close(fig)
        
        # --- PLOT 4: Violin Plot - Supercomplex Area Distribution by Condition ---
        fig, ax = plt.subplots(dpi=300)
        
        # Filter out zero areas for visualization
        df_plot = df_marker[df_marker[area_column] > 0].copy()
        
        if not df_plot.empty:
            sns.violinplot(
                data=df_plot,
                x='Timepoint',
                y=area_column,
                hue='Condition',
                hue_order=condition_order,
                palette=palette,
                split=False,
                inner='quartile',
                ax=ax
            )
            
            ax.set_title(f"{marker_name}\nSupercomplex Area Distribution")
            ax.set_ylabel(f"Supercomplex Area ({area_unit})")
            ax.set_xlabel("Timepoint (minutes)")
            ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plot_filename = plotPath / f"violin_{marker_name_clean}_area_distribution.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_filename.name}")
            plt.close(fig)
        
        # --- PLOT 5: Box Plot - Number of Supercomplexes per Cell ---
        fig, ax = plt.subplots(dpi=300)
        
        sns.boxplot(
            data=cell_stats,
            x='Timepoint',
            y='Num_Supercomplexes',
            hue='Condition',
            hue_order=condition_order,
            palette=palette,
            ax=ax
        )
        
        ax.set_title(f"{marker_name}\nNumber of Supercomplexes per Cell (Box Plot)")
        ax.set_ylabel("Number of Supercomplexes\nper Cell")
        ax.set_xlabel("Timepoint (minutes)")
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plot_filename = plotPath / f"boxplot_{marker_name_clean}_num_supercomplexes.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_filename.name}")
        plt.close(fig)
    
    print("\n" + "="*80)
    print("All plots generated successfully!")
    print(f"Plots saved to: {plotPath}")
    print("="*80)

