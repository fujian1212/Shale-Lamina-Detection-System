#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch result merging."""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
import math
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 0

from .processing import process_image, process_folder

def merge_batch_results(base_output_dir, results_dirs):
    """Merge batch-processing results so the x-axis represents continuous depth.
    
    Args:
        base_output_dir: Output directory for the merged results.
        results_dirs: List of per-image result directories.
    """
    print(f"Merging results into: {base_output_dir}")
    print(f"Sub-directory count: {len(results_dirs)}")
    
    # Container for every detected layer
    all_layers = []
    all_image_names = []
    image_widths = {}  # per-image width
    
    # Make sure the base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # If there are no result directories at all
    if not results_dirs:
        print("No result directories to merge")
        # Create empty merged-result files
        create_empty_results(base_output_dir)
        return
    
    # Collect every image's detected layers
    for result_dir in results_dirs:
        image_name = os.path.basename(result_dir)
        all_image_names.append(image_name)
        
        # Read layer-info table
        csv_path = os.path.join(result_dir, "layer_info.csv")
        excel_path = os.path.join(result_dir, "layer_info.xlsx")
        excel_en_path = os.path.join(result_dir, "layer_info_en.xlsx")  # English copy
        
        # Prefer the English Excel, then the localized Excel, then CSV
        if os.path.exists(excel_en_path):
            try:
                df = pd.read_excel(excel_en_path)
                print(f"Loaded English Excel: {excel_en_path}")
            except Exception as e:
                print(f"Failed to read English Excel: {str(e)}; falling back to localized Excel")
                df = None
        elif os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path)
                print(f"Loaded localized Excel: {excel_path}")
            except Exception as e:
                print(f"Failed to read localized Excel: {str(e)}; falling back to CSV")
                df = None
        elif os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded CSV: {csv_path}")
            except Exception as e:
                print(f"Failed to read CSV: {str(e)}")
                df = None
        else:
            print(f"No layer-info file found: {excel_en_path}, {excel_path}, or {csv_path}")
            df = None
        
        # Try to read the image-width info (Excel first, then CSV)
        position_excel_path = os.path.join(result_dir, "position_info.xlsx")
        position_csv_path = os.path.join(result_dir, "position_info.csv")
        
        try:
            # Excel preferred
            if os.path.exists(position_excel_path):
                pos_df = pd.read_excel(position_excel_path)
                print(f"Loaded position Excel: {position_excel_path}")
            elif os.path.exists(position_csv_path):
                pos_df = pd.read_csv(position_csv_path)
                print(f"Loaded position CSV: {position_csv_path}")
            else:
                pos_df = None
                print(f"No position-info file found: {position_excel_path} or {position_csv_path}")
            
            if pos_df is not None and len(pos_df) > 0:
                # Find the position column (accept old & new names)
                position_col = None
                for col_name in ["position_px", "position", "position_x_px"]:
                    if col_name in pos_df.columns:
                        position_col = col_name
                        break
                
                if position_col:
                    # Max position serves as the image width
                    image_widths[image_name] = pos_df[position_col].max() + 1
                    print(f"Image {image_name} width: {image_widths[image_name]} px")
                else:
                    print(f"Position file lacks a recognized position column")
                    image_widths[image_name] = 800
        except Exception as e:
            print(f"Failed to read image-width info: {str(e)}")
            # Fall back to a default width
            image_widths[image_name] = 800
            
        # Process the loaded dataframe
        if df is not None and len(df) > 0:
            # Two-way column rename support for legacy / current formats
            legacy_to_english = {
                'position_x_px': 'position_x',
                'position_y_px': 'position_y',
                'spacing_to_next_px': 'spacing_to_next',
                # Legacy aliases retained for compatibility
                'layer_width': 'spacing_to_next',
                'pair_index': 'layer_index',
                'edge_strength': 'strength',
            }

            for src_name, dst_name in legacy_to_english.items():
                if src_name in df.columns and dst_name not in df.columns:
                    df[dst_name] = df[src_name]
                    print(f"Renamed column '{src_name}' -> '{dst_name}'")

            # Additional legacy fallbacks
            if 'layer_width' in df.columns and 'spacing_to_next' not in df.columns:
                df['spacing_to_next'] = df['layer_width']
            if 'pair_index' in df.columns and 'layer_index' not in df.columns:
                df['layer_index'] = df['pair_index']
            
            required_columns = ['position_x', 'position_y', 'strength', 'scan_line', 'layer_index']
            missing_columns = []
            
            for col in required_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            # If position_x is still missing, try other recovery strategies
            if 'position_x' in missing_columns:
                if 'position' in df.columns:
                    # Parse from the position string when possible
                    try:
                        import ast
                        df['position_x'] = df['position'].apply(
                            lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.strip() else 0
                        )
                        print(f"Created position_x from the position string column")
                        missing_columns.remove('position_x')
                    except Exception as e:
                        print(f"Could not create position_x from the position string column: {str(e)}")
                        # Fall back to the row index as the position
                        df['position_x'] = df.index
                        missing_columns.remove('position_x')
                else:
                    # No positional info at all -- use the row index
                    df['position_x'] = df.index
                    missing_columns.remove('position_x')
                    print(f"Using the row index as position_x")
            
            # Default values for any remaining missing columns
            default_values = {
                'position_y': 0,
                'strength': 1.0,
                'scan_line': 0,
                'layer_index': 0,
                'spacing_to_next': 0
            }
            
            for col in missing_columns:
                if col in default_values:
                    df[col] = default_values[col]
                    print(f"Defaulted missing column '{col}' to: {default_values[col]}")
                
            # Tag with the image name
            df['image_name'] = image_name
            all_layers.append(df)
            print(f"Loaded data file with {len(df)} rows")
        else:
            print(f"Data file is missing or empty")
    
    # No layer data at all
    if not all_layers:
        print("No layer data found to merge")
        # Create empty merged-result files
        create_empty_results(base_output_dir)
        return
    
    # Adjust the x-coordinates so they reflect continuous depth (cumulative width).
    # First sort the image names (by embedded numeric ID when possible).
    try:
        import re
        def extract_number(name):
            match = re.search(r'(\d+)', name)
            if match:
                return int(match.group(1))
            return 0  # no number -> 0
        
        sorted_image_names = sorted(all_image_names, key=extract_number)
    except Exception:
        # Numeric sort failed -> alphabetical order
        sorted_image_names = sorted(all_image_names)
    
    print(f"Sorted image order: {sorted_image_names}")
    
    # Per-image cumulative offset
    offset_dict = {}
    current_offset = 0
    
    for img_name in sorted_image_names:
        offset_dict[img_name] = current_offset
        # Use the actual width if available, otherwise the default
        width = image_widths.get(img_name, 800)
        current_offset += width
    
    # Adjust x-coordinates and concatenate
    adjusted_layers = []
    for df in all_layers:
        image_name = df['image_name'].iloc[0]
        offset = offset_dict[image_name]
        
        # Copy the dataframe
        adjusted_df = df.copy()
        
        # Shift the x-coordinate
        if 'position_x' in adjusted_df.columns:
            adjusted_df['position_x'] = adjusted_df['position_x'] + offset
        
        adjusted_layers.append(adjusted_df)
    
    # Concatenate adjusted layers
    merged_layers = pd.concat(adjusted_layers, ignore_index=True)
    
    # Depth column (combining image order and offset)
    merged_layers['depth_position'] = merged_layers.apply(
        lambda row: offset_dict.get(row['image_name'], 0) + (row['position_x'] - offset_dict.get(row['image_name'], 0)),
        axis=1
    )
    
    # Locate rows containing the image-index / position columns
    if merged_layers is not None and len(merged_layers) > 0:
        # Ensure image_index exists; build it from image_name otherwise
        if 'image_index' not in merged_layers.columns:
            image_index_map = {name: i for i, name in enumerate(sorted_image_names)}
            merged_layers['image_index'] = merged_layers['image_name'].map(image_index_map)
        
        # Ensure adjusted_position exists; derive from position_x / depth_position
        if 'adjusted_position' not in merged_layers.columns:
            if 'depth_position' in merged_layers.columns:
                merged_layers['adjusted_position'] = merged_layers['depth_position']
            elif 'position_x' in merged_layers.columns:
                merged_layers['adjusted_position'] = merged_layers['position_x']
            else:
                # Fall back to the row index
                merged_layers['adjusted_position'] = merged_layers.index.values
        
        # Sort before saving to keep the CSV deterministic
        try:
            merged_layers = merged_layers.sort_values(by=["image_index", "adjusted_position"])
        except Exception as e:
            print(f"Sort failed: {str(e)}; retrying with image_index only")
            try:
                merged_layers = merged_layers.sort_values(by=["image_index"])
            except Exception as e2:
                print(f"Sort retry failed: {str(e2)}; continuing without sorting")
    
    # English -> English column-name map (kept as a single source of truth)
    column_mapping = {
        "scan_line": "scan_line",
        "position_x": "position_x_px",
        "position_y": "position_y_px",
        "spacing_to_next": "spacing_to_next_px",
        "layer_index": "layer_index",
        "strength": "strength",
        "filename": "filename",
        "image_index": "image_index",
        "cumulative_offset": "cumulative_offset",
        "adjusted_position": "adjusted_position_px",
    }

    # Rename columns
    merged_layers_renamed = merged_layers.rename(columns=column_mapping)

    # Save the merged Excel
    merged_excel_path = os.path.join(base_output_dir, "all_layers.xlsx")
    merged_layers_renamed.to_excel(merged_excel_path, index=False)
    print(f"Merged Excel saved: {merged_excel_path}")
    
    # ==============================================
    # Merge position-info data
    # ==============================================
    try:
        print("Merging position-info data...")
        merge_position_info(base_output_dir, results_dirs, sorted_image_names, offset_dict, image_widths)
        print("Position-info merge done")
    except Exception as e:
        print(f"Error while merging position-info data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ==============================================
    # Batch summary statistics
    # ==============================================
    try:
        print("Generating batch summary statistics...")
        generate_batch_summary_statistics(
            base_output_dir,
            merged_layers,
            sorted_image_names,
            results_dirs,
            image_widths=image_widths,
        )
        print("Batch summary statistics done")
    except Exception as e:
        print(f"Error while generating batch summary statistics: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Visualization with error handling
    try:
        create_batch_visualizations(
            base_output_dir, 
            merged_layers, 
            sorted_image_names, 
            offset_dict, 
            image_widths
        )
    except Exception as e:
        print(f"Error while creating batch visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Batch merge complete")
    return merged_excel_path
def merge_position_info(base_output_dir, results_dirs, sorted_image_names, offset_dict, image_widths):
    """Merge per-image ``position_info`` data.
    
    Args:
        base_output_dir: Output directory for merged results.
        results_dirs: List of per-image result directories.
        sorted_image_names: Ordered list of image names.
        offset_dict: Per-image x-axis offset.
        image_widths: Per-image width.
    """
    print("Merging position-info data...")
    
    all_position_data = []
    
    # Collect position-info data for each image
    for result_dir in results_dirs:
        image_name = os.path.basename(result_dir)
        
        # Find the position-info file
        position_excel_path = os.path.join(result_dir, "position_info.xlsx")
        position_csv_path = os.path.join(result_dir, "position_info.csv")
        
        position_df = None
        
        # Prefer Excel
        if os.path.exists(position_excel_path):
            try:
                position_df = pd.read_excel(position_excel_path)
                print(f"Loaded position Excel: {position_excel_path}")
            except Exception as e:
                print(f"Failed to read position Excel: {str(e)}; falling back to CSV")
        
        # Fall back to CSV
        if position_df is None and os.path.exists(position_csv_path):
            try:
                position_df = pd.read_csv(position_csv_path)
                print(f"Loaded position CSV: {position_csv_path}")
            except Exception as e:
                print(f"Failed to read position CSV: {str(e)}")
        
        if position_df is not None and len(position_df) > 0:
            # Locate columns (accept several legacy / current names)
            position_col = None
            density_col = None
            intensity_col = None
            norm_intensity_col = None
            
            # Position column
            for col_name in ["position_px", "position", "position_x_px"]:
                if col_name in position_df.columns:
                    position_col = col_name
                    break
            
            # Density column
            for col_name in ["density_per_100px", "density"]:
                if col_name in position_df.columns:
                    density_col = col_name
                    break
            
            # Intensity column
            for col_name in ["strength", "intensity"]:
                if col_name in position_df.columns:
                    intensity_col = col_name
                    break
            
            # Normalized intensity column
            for col_name in ["strength_normalized", "normalized_intensity"]:
                if col_name in position_df.columns:
                    norm_intensity_col = col_name
                    break
            
            if position_col is not None:
                # This image's offset
                offset = offset_dict.get(image_name, 0)
                
                # Copy and shift positional coordinates
                adjusted_df = position_df.copy()
                adjusted_df[position_col] = adjusted_df[position_col] + offset
                
                # Tag with image identity
                adjusted_df['image_name'] = image_name
                adjusted_df['image_index'] = sorted_image_names.index(image_name) if image_name in sorted_image_names else 0

                # Normalize column names
                column_rename_map = {}
                if position_col != "position_px":
                    column_rename_map[position_col] = "position_px"
                if density_col and density_col != "density_per_100px":
                    column_rename_map[density_col] = "density_per_100px"
                if intensity_col and intensity_col != "strength":
                    column_rename_map[intensity_col] = "strength"
                if norm_intensity_col and norm_intensity_col != "strength_normalized":
                    column_rename_map[norm_intensity_col] = "strength_normalized"
                
                if column_rename_map:
                    adjusted_df.rename(columns=column_rename_map, inplace=True)
                
                all_position_data.append(adjusted_df)
                print(f"Processed position data for image {image_name}; offset shift: {offset}")
            else:
                print(f"Image {image_name} position file lacks a recognized position column")
        else:
            print(f"Image {image_name} has no position_info file")
    
    if not all_position_data:
        print("No position_info data to merge")
        # Empty file
        empty_position_df = pd.DataFrame(columns=[
            "position_px", "density_per_100px", "strength", "strength_normalized", "image_name", "image_index"
        ])
        merged_position_path = os.path.join(base_output_dir, "merged_position_info.xlsx")
        empty_position_df.to_excel(merged_position_path, index=False)
        print(f"Created empty merged-position file: {merged_position_path}")
        return
    
    # Concatenate
    merged_position_df = pd.concat(all_position_data, ignore_index=True)
    
    # Sort by position
    merged_position_df = merged_position_df.sort_values(['image_index', 'position_px'])
    
    # Save merged position info
    merged_position_path = os.path.join(base_output_dir, "merged_position_info.xlsx")
    merged_position_df.to_excel(merged_position_path, index=False)
    print(f"Merged position info saved: {merged_position_path}")
    print(f"Merged row count: {len(merged_position_df)}")
    
    # Generate continuous-depth position statistics
    try:
        generate_continuous_position_statistics(base_output_dir, merged_position_df, sorted_image_names, offset_dict, image_widths)
    except Exception as e:
        print(f"Error while generating continuous-depth position stats: {str(e)}")
        import traceback
        traceback.print_exc()
def generate_continuous_position_statistics(base_output_dir, merged_position_df, sorted_image_names, offset_dict, image_widths):
    """Generate continuous-depth-based position statistics.
    
    Args:
        base_output_dir: Output directory.
        merged_position_df: Merged position dataframe.
        sorted_image_names: Ordered list of image names.
        offset_dict: Per-image x-axis offset.
        image_widths: Per-image width.
    """
    print("Generating continuous-depth position statistics...")

    # Total width
    total_width = sum(image_widths.get(name, 800) for name in sorted_image_names)
    
    # Continuous-depth position stats
    continuous_stats = []
    
    # Resolution = pixels per bin
    resolution = max(1, total_width // 1000)  # at most 1000 bins
    
    for pos in range(0, total_width, resolution):
        # Window around this position
        start_pos = pos
        end_pos = min(pos + resolution, total_width)
        
        # Data inside the window
        region_data = merged_position_df[
            (merged_position_df['position_px'] >= start_pos) &
            (merged_position_df['position_px'] < end_pos)
        ]
        
        # Per-bin statistics
        if len(region_data) > 0:
            avg_density = region_data['density_per_100px'].mean() if 'density_per_100px' in region_data.columns else 0
            avg_intensity = region_data['strength'].mean() if 'strength' in region_data.columns else 0
            avg_norm_intensity = region_data['strength_normalized'].mean() if 'strength_normalized' in region_data.columns else 0
            max_density = region_data['density_per_100px'].max() if 'density_per_100px' in region_data.columns else 0
            data_count = len(region_data)
            
            # Identify the source image for this position
            source_image = "unknown"
            for i, image_name in enumerate(sorted_image_names):
                img_start = offset_dict.get(image_name, 0)
                img_end = img_start + image_widths.get(image_name, 800)
                if img_start <= pos < img_end:
                    source_image = image_name
                    break
        else:
            avg_density = 0
            avg_intensity = 0
            avg_norm_intensity = 0
            max_density = 0
            data_count = 0
            source_image = "unknown"
        
        continuous_stats.append({
            "continuous_depth_px": pos,
            "bin_range": f"{start_pos}-{end_pos}",
            "avg_density_per_100px": round(avg_density, 4),
            "avg_strength": round(avg_intensity, 4),
            "avg_strength_normalized": round(avg_norm_intensity, 4),
            "max_density_per_100px": round(max_density, 4),
            "data_point_count": data_count,
            "source_image": source_image
        })

    # Save continuous-depth statistics
    continuous_df = pd.DataFrame(continuous_stats)
    continuous_path = os.path.join(base_output_dir, "continuous_position_statistics.xlsx")
    continuous_df.to_excel(continuous_path, index=False)
    print(f"Continuous-depth position stats saved: {continuous_path}")
    print(f"Bin count: {len(continuous_df)}, resolution: {resolution} px/bin")
def _augment_unique_laminae_with_thickness(all_unique_df, sorted_image_names,
                                            image_widths, per_image_summaries):
    """Add cross-image ``thickness_to_next_*`` columns to the merged unique-lamina table.

    Rules (matching the user spec):

      1. **Within image** -- a lamina whose neighbour is in the same image:
         ``thickness_to_next = next_x - this_x`` (pixels), converted to mm
         using the per-image ``scale_px_per_mm``.

      2. **Cross image** -- the lamina is the last one in image ``N`` and
         image ``N+1`` exists with at least one valid lamina. Then the
         thickness has to span the boundary:
         ``thickness_to_next = (img_N_width - this_x) + next_image_first_x``.

      3. **Edge cases** (no neighbour available on one side; the user's
         "after this image there are no more pictures, or at the start"
         clause):
         - The last lamina of the LAST image (no next): fall back to
           ``thickness_to_next = global_x_px``  -- i.e. the distance "from
           the very top of the batch to this position", which is the closest
           well-defined boundary measure available when no next lamina
           exists.
         - The first lamina of the FIRST image: it normally HAS a next, so
           the standard within-image / cross-image rule applies. We do *not*
           overwrite that forward gap here; instead the same "from top to
           this position" quantity is also written into the
           ``thickness_from_top_*`` columns for every lamina, so the first
           lamina's start-boundary thickness is preserved alongside the
           forward gap.

    The function also adds:
      ``global_position_px`` -- cumulative pixel offset (image_offset + x)
      ``global_position_mm`` -- same, divided by the image's px/mm
      ``thickness_from_top_px`` / ``thickness_from_top_mm`` -- the distance
        from the start of the FIRST image to this lamina (for every lamina;
        for the first lamina this is the only sensible "thickness" because
        no previous neighbour exists)
      ``thickness_case`` -- one of ``within_image`` / ``cross_image`` /
        ``fallback_from_top``, recording which rule produced the value.
    """
    if all_unique_df is None or len(all_unique_df) == 0:
        return all_unique_df, {}
    if "image_name" not in all_unique_df.columns:
        return all_unique_df, {}
    if "x_pos_px_mean" not in all_unique_df.columns:
        return all_unique_df, {}

    image_widths = image_widths or {}
    per_image_summaries = per_image_summaries or {}

    df = all_unique_df.copy()
    image_index = {name: i for i, name in enumerate(sorted_image_names)}
    df["_img_idx"] = df["image_name"].map(image_index)
    # Drop rows whose image is not in sorted_image_names (defensive)
    df = df.dropna(subset=["_img_idx"]).copy()
    df["_img_idx"] = df["_img_idx"].astype(int)
    df["_x_in_image"] = pd.to_numeric(df["x_pos_px_mean"], errors="coerce")
    df = df.dropna(subset=["_x_in_image"]).copy()
    df = df.sort_values(["_img_idx", "_x_in_image"], kind="stable").reset_index(drop=True)

    # Cumulative offset per image (in batch processing order)
    cumulative_offset = {}
    running = 0
    for name in sorted_image_names:
        cumulative_offset[name] = running
        running += int(image_widths.get(name, 0) or 0)

    # Per-image px/mm (might differ between images, but typically the same)
    def _ppm(name):
        s = per_image_summaries.get(name, {}) or {}
        v = s.get("scale_px_per_mm")
        try:
            v = float(v)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    df["_offset_px"] = df["image_name"].map(cumulative_offset).fillna(0).astype(float)
    df["global_position_px"] = (df["_offset_px"] + df["_x_in_image"]).round(2)

    n = len(df)
    thickness_px = [None] * n
    thickness_case = [""] * n
    for i in range(n):
        this_global = float(df["global_position_px"].iloc[i])
        if i < n - 1:
            next_global = float(df["global_position_px"].iloc[i + 1])
            this_img = df["image_name"].iloc[i]
            next_img = df["image_name"].iloc[i + 1]
            thickness_px[i] = round(next_global - this_global, 2)
            thickness_case[i] = "within_image" if this_img == next_img else "cross_image"
        else:
            # Last lamina of the LAST image -> fallback "from top to this position"
            thickness_px[i] = round(this_global, 2)
            thickness_case[i] = "fallback_from_top"

    df["thickness_to_next_px"] = thickness_px
    df["thickness_case"] = thickness_case

    # Convert px values to mm using the lamina's source-image scale
    px_per_mm_per_image = {name: _ppm(name) for name in sorted_image_names}

    def _to_mm(px_val, image_name):
        if px_val is None or pd.isna(px_val):
            return ""
        ppmm = px_per_mm_per_image.get(image_name)
        if ppmm is None or ppmm == 0:
            return ""
        try:
            return round(float(px_val) / float(ppmm), 3)
        except (TypeError, ValueError, ZeroDivisionError):
            return ""

    df["thickness_to_next_mm"] = [
        _to_mm(px, name) for px, name in zip(df["thickness_to_next_px"], df["image_name"])
    ]

    # "From top of batch to this lamina" for every lamina -- useful as the
    # first-lamina boundary thickness and as a cumulative depth measure.
    df["thickness_from_top_px"] = df["global_position_px"].round(2)
    df["thickness_from_top_mm"] = [
        _to_mm(px, name) for px, name in zip(df["thickness_from_top_px"], df["image_name"])
    ]
    df["global_position_mm"] = df["thickness_from_top_mm"]

    # Drop the internal helper columns
    df = df.drop(columns=["_img_idx", "_x_in_image", "_offset_px"])

    # Per-image aggregate of the cross-image-aware thickness (needed for
    # images_statistics.xlsx and batch_summary.xlsx so the numbers reflect
    # the correct definition rather than only the within-image spacing).
    per_image_thickness = {}
    for name in sorted_image_names:
        sub = df[df["image_name"] == name]
        if len(sub) == 0:
            continue
        sub_px = pd.to_numeric(sub["thickness_to_next_px"], errors="coerce")
        # Exclude the absolute last lamina of the LAST image (its value is the
        # ``fallback_from_top`` measure, which mixes scales; including it would
        # inflate "thickness" statistics).
        valid_mask = (sub["thickness_case"] != "fallback_from_top")
        sub_px_v = sub_px[valid_mask].dropna()
        sub_px_v = sub_px_v[sub_px_v > 0]
        if len(sub_px_v) == 0:
            continue

        ppmm = px_per_mm_per_image.get(name)
        sub_mm_v = (sub_px_v / ppmm) if (ppmm and ppmm > 0) else None

        per_image_thickness[name] = {
            "avg_thickness_to_next_px": round(float(sub_px_v.mean()), 2),
            "max_thickness_to_next_px": round(float(sub_px_v.max()), 2),
            "min_thickness_to_next_px": round(float(sub_px_v.min()), 2),
            "std_thickness_to_next_px": round(float(sub_px_v.std()), 2) if len(sub_px_v) > 1 else 0.0,
            "n_thickness_samples": int(len(sub_px_v)),
            "avg_thickness_to_next_mm": (round(float(sub_mm_v.mean()), 3)
                                         if sub_mm_v is not None else ""),
            "max_thickness_to_next_mm": (round(float(sub_mm_v.max()), 3)
                                         if sub_mm_v is not None else ""),
            "min_thickness_to_next_mm": (round(float(sub_mm_v.min()), 3)
                                         if sub_mm_v is not None else ""),
            "std_thickness_to_next_mm": (round(float(sub_mm_v.std()), 3)
                                         if sub_mm_v is not None and len(sub_mm_v) > 1 else ""),
        }

    return df, per_image_thickness


def _load_per_image_summary(result_dir):
    """Load the per-image ``summary.xlsx`` (one row) as a dict.

    Returns an empty dict if the file is missing or unreadable.
    """
    summary_path = os.path.join(result_dir, "summary.xlsx")
    if not os.path.exists(summary_path):
        return {}
    try:
        df = pd.read_excel(summary_path)
        if len(df) == 0:
            return {}
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"Cannot read {summary_path}: {e}")
        return {}


def _load_per_image_unique_laminae(result_dir, image_name):
    """Load the per-image ``lamina_summary.xlsx`` and filter to valid laminae.

    The detector exports two variants:
      - ``lamina_summary.xlsx``       all clusters including rejected smudges
      - ``lamina_summary_valid.xlsx`` valid laminae only (preferred)
    """
    valid_path = os.path.join(result_dir, "lamina_summary_valid.xlsx")
    full_path = os.path.join(result_dir, "lamina_summary.xlsx")

    target = valid_path if os.path.exists(valid_path) else full_path
    if not os.path.exists(target):
        return None

    try:
        df = pd.read_excel(target)
    except Exception as e:
        print(f"Cannot read {target}: {e}")
        return None

    if target == full_path and "is_valid_lamina" in df.columns:
        df = df[df["is_valid_lamina"] == "yes"].copy()
    else:
        df = df.copy()

    df["image_name"] = image_name
    return df


def generate_batch_summary_statistics(base_output_dir, merged_layers, sorted_image_names, results_dirs,
                                      image_widths=None):
    """Generate batch-processing summary statistics.

    Two statistical layers are produced:

    * **Unique-lamina level** (recommended for papers): read each image's
      ``summary.xlsx`` and ``lamina_summary.xlsx`` and aggregate the
      cross-line clustered laminae across all images. This matches what the
      single-image flow reports.

    * **Candidate-point level** (diagnostic): aggregate the per-scan-line
      candidate change points from ``merged_layers``. Useful to inspect
      detection density on individual scan lines but it is *not* the lamina
      count.
    
    Args:
        base_output_dir: Output directory.
        merged_layers: Merged lamina dataframe (candidate-point level).
        sorted_image_names: Ordered list of image names.
        results_dirs: List of per-image result directories.
        image_widths: Optional ``{image_name: width_px}`` mapping. When
            provided, the merged ``thickness_to_next_*`` columns are computed
            across image boundaries (last lamina of an image -> first lamina
            of the next image).
    """
    print("Computing batch summary statistics...")

    # ===== Load per-image unique-lamina data =====
    per_image_summaries = {}    # image_name -> dict
    per_image_laminae_dfs = []  # one DataFrame per image, valid laminae only

    name_to_dir = {os.path.basename(d): d for d in results_dirs}
    for image_name in sorted_image_names:
        result_dir = name_to_dir.get(image_name)
        if not result_dir:
            continue
        per_image_summaries[image_name] = _load_per_image_summary(result_dir)
        lam_df = _load_per_image_unique_laminae(result_dir, image_name)
        if lam_df is not None and len(lam_df) > 0:
            per_image_laminae_dfs.append(lam_df)

    # ===== Aggregate unique-lamina level =====
    unique_summary_stats = {
        "total_image_count": len(sorted_image_names),
        "successful_image_count": 0,
        "failed_image_count": 0,
        "total_unique_laminae": 0,
        "avg_unique_laminae_per_image": 0.0,
        "avg_lamina_spacing_px": 0.0,
        "lamina_spacing_std_px": 0.0,
        "max_lamina_spacing_px": 0.0,
        "min_lamina_spacing_px": 0.0,
        "avg_lamina_spacing_mm": "",
        "lamina_spacing_std_mm": "",
        "avg_dip_angle_deg": 0.0,
        "avg_fit_residual_px": 0.0,
        "avg_support_lines_per_lamina": 0.0,
        "avg_cross_line_support_ratio": 0.0,
    }

    all_unique_df = None
    per_image_thickness = {}
    if per_image_laminae_dfs:
        all_unique_df = pd.concat(per_image_laminae_dfs, ignore_index=True)
        unique_summary_stats["total_unique_laminae"] = int(len(all_unique_df))
        unique_summary_stats["avg_unique_laminae_per_image"] = round(
            float(len(all_unique_df)) / max(1, len(sorted_image_names)), 2
        )

        # Compute cross-image ``thickness_to_next_*`` columns and per-image
        # thickness aggregates -- the user-requested behaviour. These columns
        # are added to ``merged_unique_laminae.xlsx`` below.
        all_unique_df, per_image_thickness = _augment_unique_laminae_with_thickness(
            all_unique_df, sorted_image_names, image_widths, per_image_summaries
        )

        # Pull per-lamina spacing in pixels (column produced by the detector)
        if "spacing_to_next_px" in all_unique_df.columns:
            sp_px = pd.to_numeric(all_unique_df["spacing_to_next_px"], errors="coerce").dropna()
            sp_px = sp_px[sp_px > 0]
            if len(sp_px) > 0:
                unique_summary_stats["avg_lamina_spacing_px"] = round(float(sp_px.mean()), 2)
                unique_summary_stats["lamina_spacing_std_px"] = round(float(sp_px.std()), 2)
                unique_summary_stats["max_lamina_spacing_px"] = round(float(sp_px.max()), 2)
                unique_summary_stats["min_lamina_spacing_px"] = round(float(sp_px.min()), 2)

        if "spacing_to_next_mm" in all_unique_df.columns:
            sp_mm = pd.to_numeric(all_unique_df["spacing_to_next_mm"], errors="coerce").dropna()
            sp_mm = sp_mm[sp_mm > 0]
            if len(sp_mm) > 0:
                unique_summary_stats["avg_lamina_spacing_mm"] = round(float(sp_mm.mean()), 3)
                unique_summary_stats["lamina_spacing_std_mm"] = round(float(sp_mm.std()), 3)
                unique_summary_stats["max_lamina_spacing_mm"] = round(float(sp_mm.max()), 3)
                unique_summary_stats["min_lamina_spacing_mm"] = round(float(sp_mm.min()), 3)

        # Cross-image-aware thickness aggregates (separate from the within-image
        # ``spacing_*`` numbers above so consumers can pick the definition
        # they need).
        unique_summary_stats["avg_thickness_to_next_px"] = 0.0
        unique_summary_stats["max_thickness_to_next_px"] = 0.0
        unique_summary_stats["min_thickness_to_next_px"] = 0.0
        unique_summary_stats["thickness_to_next_std_px"] = 0.0
        unique_summary_stats["avg_thickness_to_next_mm"] = ""
        unique_summary_stats["max_thickness_to_next_mm"] = ""
        unique_summary_stats["min_thickness_to_next_mm"] = ""
        unique_summary_stats["thickness_to_next_std_mm"] = ""
        if "thickness_to_next_px" in all_unique_df.columns:
            # Exclude the last lamina (fallback_from_top) from the aggregate,
            # because its value is "from top to this position" rather than a
            # gap to the next lamina, and would dominate the average otherwise.
            t_mask = (all_unique_df.get("thickness_case", "") != "fallback_from_top")
            t_px = pd.to_numeric(
                all_unique_df.loc[t_mask, "thickness_to_next_px"], errors="coerce"
            ).dropna()
            t_px = t_px[t_px > 0]
            if len(t_px) > 0:
                unique_summary_stats["avg_thickness_to_next_px"] = round(float(t_px.mean()), 2)
                unique_summary_stats["max_thickness_to_next_px"] = round(float(t_px.max()), 2)
                unique_summary_stats["min_thickness_to_next_px"] = round(float(t_px.min()), 2)
                unique_summary_stats["thickness_to_next_std_px"] = round(float(t_px.std()), 2)
        if "thickness_to_next_mm" in all_unique_df.columns:
            t_mask = (all_unique_df.get("thickness_case", "") != "fallback_from_top")
            t_mm = pd.to_numeric(
                all_unique_df.loc[t_mask, "thickness_to_next_mm"], errors="coerce"
            ).dropna()
            t_mm = t_mm[t_mm > 0]
            if len(t_mm) > 0:
                unique_summary_stats["avg_thickness_to_next_mm"] = round(float(t_mm.mean()), 3)
                unique_summary_stats["max_thickness_to_next_mm"] = round(float(t_mm.max()), 3)
                unique_summary_stats["min_thickness_to_next_mm"] = round(float(t_mm.min()), 3)
                unique_summary_stats["thickness_to_next_std_mm"] = round(float(t_mm.std()), 3)

        if "dip_angle_deg" in all_unique_df.columns:
            dip = pd.to_numeric(all_unique_df["dip_angle_deg"], errors="coerce").dropna()
            if len(dip) > 0:
                unique_summary_stats["avg_dip_angle_deg"] = round(float(dip.mean()), 2)

        for resid_col in ("fit_mean_residual_px", "fit_max_residual_px"):
            if resid_col in all_unique_df.columns:
                resid = pd.to_numeric(all_unique_df[resid_col], errors="coerce").dropna()
                if len(resid) > 0:
                    unique_summary_stats["avg_fit_residual_px"] = round(float(resid.mean()), 2)
                    break

        if "n_support_lines_total" in all_unique_df.columns:
            sup = pd.to_numeric(all_unique_df["n_support_lines_total"], errors="coerce").dropna()
            if len(sup) > 0:
                unique_summary_stats["avg_support_lines_per_lamina"] = round(float(sup.mean()), 2)

        if "support_ratio" in all_unique_df.columns:
            sup_r = pd.to_numeric(all_unique_df["support_ratio"], errors="coerce").dropna()
            if len(sup_r) > 0:
                unique_summary_stats["avg_cross_line_support_ratio"] = round(float(sup_r.mean()), 3)

    # ===== Aggregate candidate-point level (diagnostic only) =====
    candidate_summary_stats = {
        "candidate_change_points_total": 0,
        "candidate_scan_line_total": 0,
        "avg_candidate_spacing_px": 0.0,
        "candidate_spacing_std_px": 0.0,
        "max_candidate_spacing_px": 0.0,
        "min_candidate_spacing_px": 0.0,
        "avg_candidate_strength": 0.0,
        "candidate_strength_std": 0.0,
    }
    
    if merged_layers is not None and len(merged_layers) > 0:
        candidate_summary_stats["candidate_change_points_total"] = int(len(merged_layers))
        
        if "scan_line" in merged_layers.columns:
            candidate_summary_stats["candidate_scan_line_total"] = int(merged_layers["scan_line"].nunique())
        
        sp_col = "spacing_to_next"
        if sp_col in merged_layers.columns:
            spacings = pd.to_numeric(merged_layers[sp_col], errors="coerce").dropna()
            spacings = spacings[spacings > 0]
            if len(spacings) > 0:
                candidate_summary_stats["avg_candidate_spacing_px"] = round(float(spacings.mean()), 2)
                candidate_summary_stats["candidate_spacing_std_px"] = round(float(spacings.std()), 2)
                candidate_summary_stats["max_candidate_spacing_px"] = round(float(spacings.max()), 2)
                candidate_summary_stats["min_candidate_spacing_px"] = round(float(spacings.min()), 2)

        if "strength" in merged_layers.columns:
            strengths = pd.to_numeric(merged_layers["strength"], errors="coerce").dropna()
            if len(strengths) > 0:
                candidate_summary_stats["avg_candidate_strength"] = round(float(strengths.mean()), 4)
                candidate_summary_stats["candidate_strength_std"] = round(float(strengths.std()), 4)
    
    # ===== Success / failure counts =====
    success_count = 0
    fail_count = 0
    for result_dir in results_dirs:
        layer_files = [
            os.path.join(result_dir, "layer_info.xlsx"),
            os.path.join(result_dir, "layer_info_en.xlsx"),
            os.path.join(result_dir, "layer_info.csv"),
        ]
        if any(os.path.exists(f) for f in layer_files):
            success_count += 1
        else:
            fail_count += 1
    
    unique_summary_stats["successful_image_count"] = success_count
    unique_summary_stats["failed_image_count"] = fail_count
    
    # ===== Per-image stats: unique-lamina view =====
    image_stats_list = []
    for image_name in sorted_image_names:
        per_summary = per_image_summaries.get(image_name, {})

        # Candidate-level fallback from merged_layers
        image_data = (
            merged_layers[merged_layers["image_name"] == image_name]
            if "image_name" in (merged_layers.columns if merged_layers is not None else [])
            else pd.DataFrame()
        )
        sp_col = "spacing_to_next"
        valid_sp = image_data[sp_col].dropna() if sp_col in image_data.columns else pd.Series(dtype=float)
        valid_sp = valid_sp[valid_sp > 0]

        thickness_stats = per_image_thickness.get(image_name, {}) or {}

        image_stat = {
            "image_name": image_name,
            # -- Unique-lamina view (recommended for papers) --
            "n_unique_laminae": int(per_summary.get("unique_laminae_cluster",
                                                    per_summary.get("total_laminae", 0)) or 0),
            "avg_lamina_spacing_px": _safe_round(per_summary.get("avg_lamina_spacing_px", 0), 2),
            "max_lamina_spacing_px": _safe_round(per_summary.get("max_lamina_spacing_px", 0), 2),
            "min_lamina_spacing_px": _safe_round(per_summary.get("min_lamina_spacing_px", 0), 2),
            "spacing_cv_percent": _safe_round(per_summary.get("spacing_cv_percent", 0), 1),
            "avg_lamina_spacing_mm": _safe_round(per_summary.get("avg_lamina_spacing_mm", ""), 3),
            # Cross-image-aware thickness (preferred for batch-level reasoning)
            "avg_thickness_to_next_px": thickness_stats.get("avg_thickness_to_next_px", ""),
            "max_thickness_to_next_px": thickness_stats.get("max_thickness_to_next_px", ""),
            "min_thickness_to_next_px": thickness_stats.get("min_thickness_to_next_px", ""),
            "std_thickness_to_next_px": thickness_stats.get("std_thickness_to_next_px", ""),
            "avg_thickness_to_next_mm": thickness_stats.get("avg_thickness_to_next_mm", ""),
            "max_thickness_to_next_mm": thickness_stats.get("max_thickness_to_next_mm", ""),
            "min_thickness_to_next_mm": thickness_stats.get("min_thickness_to_next_mm", ""),
            "std_thickness_to_next_mm": thickness_stats.get("std_thickness_to_next_mm", ""),
            "n_thickness_samples": thickness_stats.get("n_thickness_samples", 0),
            "avg_dip_angle_deg": _safe_round(per_summary.get("avg_dip_angle_deg", 0), 2),
            "avg_fit_residual_px": _safe_round(per_summary.get("avg_fit_residual_px", 0), 2),
            "avg_support_lines_per_lamina": _safe_round(per_summary.get("avg_support_lines_per_lamina", 0), 2),
            "avg_cross_line_support_ratio": _safe_round(per_summary.get("avg_cross_line_support_ratio", 0), 3),
            "n_scan_lines_total": int(per_summary.get("n_scan_lines_total",
                                                     per_summary.get("n_scan_lines", 0)) or 0),
            # -- Candidate-point view (diagnostic) --
            "candidate_change_points": int(per_summary.get("candidate_change_points_total", len(image_data)) or len(image_data)),
            "avg_candidate_spacing_px": round(float(valid_sp.mean()), 2) if len(valid_sp) > 0 else 0.0,
            "avg_candidate_strength": round(float(image_data["strength"].mean()), 4)
                if "strength" in image_data.columns and len(image_data) > 0 else 0.0,
        }
        image_stats_list.append(image_stat)
    
    # ===== Combine summaries and persist =====
    combined_summary = {**unique_summary_stats, **candidate_summary_stats}
    summary_df = pd.DataFrame([combined_summary])
    summary_path = os.path.join(base_output_dir, "batch_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"Batch summary saved: {summary_path}")
    
    # Per-image table
    if image_stats_list:
        image_stats_df = pd.DataFrame(image_stats_list)
        image_stats_path = os.path.join(base_output_dir, "images_statistics.xlsx")
        image_stats_df.to_excel(image_stats_path, index=False)
        print(f"Per-image statistics saved: {image_stats_path}")

    # Save the concatenated unique-lamina table (one row per valid lamina across all images)
    if all_unique_df is not None and len(all_unique_df) > 0:
        merged_unique_path = os.path.join(base_output_dir, "merged_unique_laminae.xlsx")
        try:
            # Move image_name to the front for readability
            cols = ["image_name"] + [c for c in all_unique_df.columns if c != "image_name"]
            all_unique_df = all_unique_df[cols]
            all_unique_df.to_excel(merged_unique_path, index=False)
            print(f"Merged unique-lamina table saved: {merged_unique_path} ({len(all_unique_df)} rows)")
        except Exception as e:
            print(f"Failed to write merged unique-lamina table: {e}")

    # Generate the textual report (still uses the combined summary)
    generate_batch_processing_report(base_output_dir, combined_summary, image_stats_list, sorted_image_names)

    print("Batch summary statistics done")


def _safe_round(value, ndigits):
    """Round a value if numeric, otherwise return it unchanged.

    Used when source data may contain blank strings (e.g. ``avg_lamina_spacing_mm``
    is empty when no scale calibration was set).
    """
    if value is None or value == "":
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if v != v:  # NaN
        return ""
    return round(v, ndigits)
def generate_batch_processing_report(base_output_dir, summary_stats, image_stats_list, sorted_image_names):
    """Write a batch-processing text report.
    
    Args:
        base_output_dir: Output directory.
        summary_stats: Summary statistics dict.
        image_stats_list: Per-image stats list.
        sorted_image_names: Ordered list of image names.
    """
    report_path = os.path.join(base_output_dir, "batch_processing_report.txt")
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Rock Core Lamina Identification System - Batch Processing Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Timestamp
            from datetime import datetime
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary -- group fields so unique-lamina view appears first
            unique_keys = [
                "total_image_count", "successful_image_count", "failed_image_count",
                "total_unique_laminae", "avg_unique_laminae_per_image",
                "avg_lamina_spacing_px", "lamina_spacing_std_px",
                "max_lamina_spacing_px", "min_lamina_spacing_px",
                "avg_lamina_spacing_mm", "lamina_spacing_std_mm",
                "max_lamina_spacing_mm", "min_lamina_spacing_mm",
                # Cross-image thickness (added in batch merge)
                "avg_thickness_to_next_px", "thickness_to_next_std_px",
                "max_thickness_to_next_px", "min_thickness_to_next_px",
                "avg_thickness_to_next_mm", "thickness_to_next_std_mm",
                "max_thickness_to_next_mm", "min_thickness_to_next_mm",
                "avg_dip_angle_deg", "avg_fit_residual_px",
                "avg_support_lines_per_lamina", "avg_cross_line_support_ratio",
            ]
            candidate_keys = [
                "candidate_change_points_total", "candidate_scan_line_total",
                "avg_candidate_spacing_px", "candidate_spacing_std_px",
                "max_candidate_spacing_px", "min_candidate_spacing_px",
                "avg_candidate_strength", "candidate_strength_std",
            ]

            f.write("Unique-lamina statistics (recommended for papers):\n")
            f.write("-" * 40 + "\n")
            for key in unique_keys:
                if key in summary_stats:
                    f.write(f"{key}: {summary_stats[key]}\n")
            f.write("\n")
            
            f.write("Candidate-point statistics (diagnostic; per-scan-line totals):\n")
            f.write("-" * 40 + "\n")
            for key in candidate_keys:
                if key in summary_stats:
                    f.write(f"{key}: {summary_stats[key]}\n")
            f.write("\n")

            extras = [k for k in summary_stats
                      if k not in unique_keys and k not in candidate_keys]
            if extras:
                f.write("Other:\n")
                f.write("-" * 40 + "\n")
                for key in extras:
                    f.write(f"{key}: {summary_stats[key]}\n")
                f.write("\n")

            # Success rate
            total_images = summary_stats["total_image_count"]
            success_rate = (summary_stats["successful_image_count"] / total_images * 100) if total_images > 0 else 0
            f.write(f"Success rate: {success_rate:.1f}%\n\n")

            # Per-image details
            f.write("Per-image processing details:\n")
            f.write("-" * 40 + "\n")
            for i, image_stat in enumerate(image_stats_list, 1):
                f.write(f"{i}. {image_stat['image_name']}\n")
                # Unique-lamina view (primary)
                f.write(f"   - Unique laminae (cross-line clustered): {image_stat.get('n_unique_laminae', 0)}\n")
                f.write(f"   - Avg lamina spacing (within-image): "
                        f"{image_stat.get('avg_lamina_spacing_px', 0)} px")
                mm_val = image_stat.get('avg_lamina_spacing_mm', '')
                if mm_val not in ('', None):
                    f.write(f" / {mm_val} mm")
                f.write("\n")
                # Cross-image-aware thickness (last lamina of this image -> first
                # lamina of the next image when needed)
                t_px = image_stat.get('avg_thickness_to_next_px', '')
                t_mm = image_stat.get('avg_thickness_to_next_mm', '')
                if t_px not in ('', None):
                    f.write(f"   - Avg thickness_to_next (cross-image aware): "
                            f"{t_px} px")
                    if t_mm not in ('', None):
                        f.write(f" / {t_mm} mm")
                    n_samples = image_stat.get('n_thickness_samples', 0)
                    if n_samples:
                        f.write(f" (n={n_samples})")
                    f.write("\n")
                f.write(f"   - Spacing CV: {image_stat.get('spacing_cv_percent', 0)} %\n")
                f.write(f"   - Avg dip angle: {image_stat.get('avg_dip_angle_deg', 0)} deg\n")
                f.write(f"   - Avg supporting scan lines: {image_stat.get('avg_support_lines_per_lamina', 0)} / {image_stat.get('n_scan_lines_total', 0)}\n")
                # Candidate view (diagnostic)
                f.write(f"   - Candidate change-points: {image_stat.get('candidate_change_points', 0)} "
                        f"(diagnostic; sum over all scan lines)\n")
                f.write("\n")
            
            # Image processing order
            f.write("Image processing order:\n")
            f.write("-" * 40 + "\n")
            for i, name in enumerate(sorted_image_names, 1):
                f.write(f"{i}. {name}\n")
            
        print(f"Batch processing report saved: {report_path}")
        
    except Exception as e:
        print(f"Error while writing batch processing report: {str(e)}")
def create_empty_results(base_output_dir):
    """Create empty merged-result files.
    
    Args:
        base_output_dir: Output directory.
    """
    print(f"Creating empty merged-result files in: {base_output_dir}")
    
    column_mapping = {
        "scan_line": "scan_line",
        "position_x": "position_x_px",
        "position_y": "position_y_px",
        "spacing_to_next": "spacing_to_next_px",
        "layer_index": "layer_index",
        "strength": "strength",
        "filename": "filename",
        "image_index": "image_index",
        "cumulative_offset": "cumulative_offset",
        "adjusted_position": "adjusted_position_px",
    }
    
    empty_df = pd.DataFrame(columns=[
        "filename", "scan_line", "position_x", "position_y",
        "spacing_to_next", "layer_index", "strength",
        "image_index", "cumulative_offset", "adjusted_position"
    ])
    
    # Rename columns
    empty_df_renamed = empty_df.rename(columns=column_mapping)

    # Save to Excel
    empty_df_renamed.to_excel(os.path.join(base_output_dir, "all_layers.xlsx"), index=False)
    print(f"Created empty merged-result file: {os.path.join(base_output_dir, 'all_layers.xlsx')}")

    # Empty heatmap placeholder
    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, "No lamina data detected", ha='center', va='center', fontsize=16)
    plt.title("Lamina intensity heatmap (empty)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "layer_intensity_heatmap.png"), dpi=150)
    plt.close()
    
    # Empty curve placeholder
    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, "No lamina data detected", ha='center', va='center', fontsize=16)
    plt.title("Transverse lamina intensity (empty)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "layer_intensity_curve.png"), dpi=150)
    plt.close()
    
    # Empty combined curve
    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, "No lamina data detected", ha='center', va='center', fontsize=16)
    plt.title("Combined lamina intensity curve (empty)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "combined_layer_intensity.png"), dpi=150)
    plt.close()

    # Add reference to ``create_batch_visualizations`` so the linter sees it
    # (the import sits at module top; calling here would be redundant).


# Lazy import to avoid circular dependency
from .batch_viz import create_batch_visualizations
