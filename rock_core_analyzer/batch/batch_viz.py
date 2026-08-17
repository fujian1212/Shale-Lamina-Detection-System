#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch processing visualization."""

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
from .merge import merge_batch_results

def create_batch_visualizations(base_output_dir, merged_layers, sorted_image_names, offset_dict, image_widths):
    """Create batch-processing visualizations whose x-axis represents continuous depth.

    Args:
        base_output_dir: Output directory.
        merged_layers: Merged lamina data.
        sorted_image_names: Ordered list of image names.
        offset_dict: Per-image x-axis offset.
        image_widths: Per-image width.
    """
    # Total width
    total_width = sum(image_widths.get(name, 800) for name in sorted_image_names)

    # Make sure strength values are floats
    if 'strength' in merged_layers.columns:
        merged_layers['strength'] = merged_layers['strength'].astype(float)

    # ==============================================
    # 1. Composite lamina display (geological column style)
    # ==============================================
    plt.figure(figsize=(15, 10))

    # Two subplots: left = column, right = intensity curve
    gs = plt.GridSpec(1, 2, width_ratios=[4, 1])
    ax_column = plt.subplot(gs[0])
    ax_intensity = plt.subplot(gs[1], sharey=ax_column)

    # Height unit: total_width / 30
    height_unit = total_width / 30

    # Per-image boundary positions
    boundaries = [offset_dict[name] for name in sorted_image_names]
    boundaries.append(total_width)  # trailing boundary

    # Image boundary horizontal lines
    for y_pos in boundaries:
        ax_column.axhline(y=y_pos, color='black', linestyle='-', linewidth=0.5)

    # Add image-name labels
    for i, name in enumerate(sorted_image_names):
        y_pos = boundaries[i]
        y_next = boundaries[i+1]
        # Center position
        y_center = (y_pos + y_next) / 2
        ax_column.text(-50, y_center, name, fontsize=8,
                       verticalalignment='center', horizontalalignment='right')

    # Heatmap data
    if not merged_layers.empty:
        # Build a 1D heatmap of lamina intensity
        # Resolution = total_width / 500
        resolution = max(1, int(total_width / 500))
        heatmap_width = total_width // resolution
        heatmap_data = np.zeros(heatmap_width)

        # Group by layer to compute mean intensity
        layer_groups = {}

        # Group by scan line and pair index to identify laminae
        for img_name in sorted_image_names:
            img_df = merged_layers[merged_layers['image_name'] == img_name]

            if 'position_x' in img_df.columns:
                try:
                    for _, row in img_df.iterrows():
                        pt_x = row['position_x']
                        strength = row.get('strength', 1.0)
                        idx = max(0, min(heatmap_width - 1, int(pt_x / resolution)))
                        heatmap_data[idx] += strength
                except Exception as e:
                    print(f"Error while processing lamina points for image {img_name}: {str(e)}")
            else:
                print(f"Image {img_name} lacks the position_x column; skipping")

        # Smooth
        smoothed_data = gaussian_filter1d(heatmap_data, sigma=3)

        # Normalize
        if len(smoothed_data) > 0 and np.max(smoothed_data) > 0:
            normalized_data = smoothed_data / np.max(smoothed_data)
        else:
            normalized_data = smoothed_data

        # Find lamina positions via peak detection
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(normalized_data, height=0.2, distance=5)
        peak_positions = peaks * resolution
        peak_strengths = normalized_data[peaks]

        # Main panel -- lamina column
        # Alpha range
        min_alpha = 0.2
        max_alpha = 0.9

        # Draw each detected lamina
        for pos, strength in zip(peak_positions, peak_strengths):
            # Modulate alpha by strength
            alpha = min_alpha + strength * (max_alpha - min_alpha)
            color = (0.2, 0.2, 0.2, alpha)

            # Horizontal line for the lamina position
            line_width = 2 + strength * 3  # stronger -> thicker
            # Clamp alpha
            line_alpha = min(1.0, alpha*1.2)
            ax_column.axhline(y=pos, color=color, linewidth=line_width, alpha=line_alpha)

        # Collect all lamina points
        all_layers_x = []
        all_layers_y = []
        all_strengths = []

        # Render the actual lamina points
        for i, img_name in enumerate(sorted_image_names):
            # Per-image data
            img_layers = merged_layers[merged_layers['image_name'] == img_name]
            if not img_layers.empty:
                # Group by scan line and pair index
                if 'depth_position' in img_layers.columns:
                    for _, row in img_layers.iterrows():
                        all_layers_x.append(row['position_x'])
                        all_layers_y.append(row['depth_position'])
                        all_strengths.append(row.get('strength', 1.0))

        # Normalize strengths
        if all_strengths:
            max_strength = max(all_strengths)
            if max_strength > 0:
                all_strengths = [s/max_strength for s in all_strengths]

            # Scatter plot -- size & color modulated by strength
            scatter = ax_column.scatter(
                [0] * len(all_layers_y),  # plot all points on the center axis
                all_layers_y,              # y-axis position
                s=[10 + s*30 for s in all_strengths],  # marker size by strength
                c=all_strengths,          # color by strength
                cmap='YlOrRd',            # yellow-orange-red colormap (geological style)
                alpha=0.6,                # semi-transparent
                edgecolors='black',       # black outline
                linewidths=0.5            # outline thickness
            )

        # Intensity curve on the right
        ax_intensity.plot(normalized_data, np.arange(0, total_width, resolution)[:len(normalized_data)], 'b-', linewidth=1.5)
        ax_intensity.fill_betweenx(np.arange(0, total_width, resolution)[:len(normalized_data)], normalized_data, color='skyblue', alpha=0.5)
        ax_intensity.set_xlabel('Lamina intensity')
        ax_intensity.set_xlim(0, 1.1)
        ax_intensity.grid(True, axis='x', alpha=0.3)

        # Main-panel styling
        ax_column.set_xlim(-100, 100)  # show only the center region
        ax_column.set_ylim(0, total_width)
        ax_column.set_yticks([])  # hide y ticks
        ax_column.set_xticks([])  # hide x ticks
        ax_column.grid(False)     # no grid

        # Center vertical line
        ax_column.axvline(x=0, color='gray', linestyle='-', linewidth=1)

        # Titles
        plt.suptitle('Rock-core lamina column (continuous depth)', fontsize=14)
        ax_column.set_title('Lamina distribution', fontsize=12)
        ax_intensity.set_title('Intensity curve', fontsize=12)

        # Flip the y-axis so depth grows downward (geological convention)
        ax_column.invert_yaxis()

        # Save
        column_path = os.path.join(base_output_dir, "layer_column_chart.png")
        try:
            plt.tight_layout()
            plt.subplots_adjust(top=0.95, wspace=0.05)  # tighten subplot spacing
        except Exception:
            pass
        plt.savefig(column_path, dpi=150)
        plt.close()
        print(f"Rock-core lamina column saved to: {column_path}")

    # ==============================================
    # 2. Combined intensity curve
    # ==============================================
    plt.figure(figsize=(15, 6))

    # One color per image
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_image_names)))

    # Max strength used for the boundary annotation height
    max_strength = merged_layers['strength'].max() if not merged_layers.empty else 1

    # Image boundary markers
    for i, image_name in enumerate(sorted_image_names):
        offset = offset_dict[image_name]
        if i > 0:
            plt.axvline(x=offset, color='gray', linestyle='--', alpha=0.5)
            plt.text(offset + 10, max_strength * 0.9, image_name, rotation=90, verticalalignment='top')

    # Per-image lamina strength
    for i, image_name in enumerate(sorted_image_names):
        image_layers = merged_layers[merged_layers['image_name'] == image_name]
        if not image_layers.empty:
            # Scatter every point with its strength
            scatter = plt.scatter(image_layers['depth_position'], image_layers['strength'],
                     marker='o', s=30, label=image_name,
                     color=colors[i], alpha=0.7)

            # Connect neighbouring laminae on the same scan line
            if len(image_layers) > 0 and 'scan_line' in image_layers.columns and 'depth_position' in image_layers.columns:
                try:
                    for scan_line in image_layers['scan_line'].unique():
                        sl_pts = image_layers[image_layers['scan_line'] == scan_line].sort_values('position_x')
                        if len(sl_pts) >= 2:
                            plt.plot(sl_pts['depth_position'], sl_pts['strength'],
                                     '-', color=colors[i], alpha=0.3, linewidth=1)
                except Exception as e:
                    print(f"Error while connecting scan-line points for image {image_name}: {str(e)}")

    plt.title("Combined lamina intensity curve (continuous depth)")
    plt.xlabel("Continuous depth position (px)")
    plt.ylabel("Lamina intensity (grayscale gradient)")
    plt.grid(True, alpha=0.3)
    # Place legend below the plot
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=min(len(sorted_image_names), 5))

    # Save combined curve
    combined_curve_path = os.path.join(base_output_dir, "combined_layer_intensity.png")
    plt.tight_layout()
    # Leave room at the bottom for the legend
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(combined_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined lamina intensity curve saved to: {combined_curve_path}")

    # ==============================================
    # 3. Heatmap (all images in a single row)
    # ==============================================
    if not merged_layers.empty:
        # 1D array representing the heatmap (all images merged into one row)
        heatmap_width = total_width
        heatmap_data = np.zeros(heatmap_width)

        # Fill heatmap with data; merge every image into the same row
        for i, image_name in enumerate(sorted_image_names):
            offset = offset_dict[image_name]
            image_width = image_widths.get(image_name, 800)

            image_layers = merged_layers[merged_layers['image_name'] == image_name]

            # Compute intensity per pixel
            for x in range(int(offset), int(offset + image_width)):
                # Find lamina points near this position
                nearby_points = image_layers[(image_layers['position_x'] >= x-3) &
                                           (image_layers['position_x'] <= x+3)]

                if len(nearby_points) > 0:
                    # Distance-weighted average intensity
                    weights = 1.0 / (1.0 + np.abs(nearby_points['position_x'] - x))
                    weighted_strength = np.sum(nearby_points['strength'] * weights) / np.sum(weights)
                    heatmap_data[x] = max(heatmap_data[x], weighted_strength)  # take max to avoid overwrites

        # *** Apply smart edge filling to the heatmap data ***
        edge_filter_percent = 0.05

        # Fill the edge region of each image
        for i, image_name in enumerate(sorted_image_names):
            offset = offset_dict[image_name]
            image_width = image_widths.get(image_name, 800)

            # Edge-filter pixel count for this image
            edge_filter_pixels = max(10, int(image_width * edge_filter_percent))

            # Start / end indices in the global array
            img_start = int(offset)
            img_end = int(offset + image_width)

            # Clamp
            img_start = max(0, img_start)
            img_end = min(heatmap_width, img_end)

            if img_end - img_start > edge_filter_pixels * 2:  # make sure the image is large enough
                # Valid-region boundaries
                valid_start = img_start + edge_filter_pixels
                valid_end = img_end - edge_filter_pixels

                # Intensity inside the valid region
                valid_region = heatmap_data[valid_start:valid_end]

                if len(valid_region) > 0 and np.sum(valid_region) > 0:
                    # Boundary and average values
                    left_boundary = heatmap_data[valid_start] if valid_start < heatmap_width else 0
                    right_boundary = heatmap_data[valid_end-1] if valid_end-1 >= 0 else 0
                    avg_intensity = np.mean(valid_region[valid_region > 0])

                    # Fill the left edge
                    for x in range(img_start, valid_start):
                        if x < heatmap_width:
                            # Distance-based weight
                            distance = x - img_start
                            weight = distance / edge_filter_pixels if edge_filter_pixels > 0 else 0
                            fill_value = (1 - weight) * avg_intensity * 0.5 + weight * left_boundary
                            heatmap_data[x] = fill_value

                    # Fill the right edge
                    for x in range(valid_end, img_end):
                        if x < heatmap_width:
                            distance = x - valid_end
                            weight = distance / edge_filter_pixels if edge_filter_pixels > 0 else 0
                            fill_value = (1 - weight) * right_boundary + weight * avg_intensity * 0.5
                            heatmap_data[x] = fill_value

                print(f"Heatmap - image {image_name}: edge fill done (pixel range: {img_start}-{img_end}, filter zone: {edge_filter_pixels}px)")

        # Normalize the heatmap
        if len(heatmap_data) > 0 and np.max(heatmap_data) > 0:
            heatmap_norm = heatmap_data / np.max(heatmap_data)
        else:
            heatmap_norm = heatmap_data

        # Heatmap image (single row)
        plt.figure(figsize=(15, 4))

        # Reshape to 2D with a single row
        heatmap_2d = heatmap_norm.reshape(1, -1)

        # Render the heatmap
        plt.imshow(heatmap_2d, aspect='auto', cmap='inferno',
                 extent=[0, total_width, 0, 1])

        # Image boundary markers
        for image_name in sorted_image_names:
            offset = offset_dict[image_name]
            if offset > 0:
                plt.axvline(x=offset, color='white', linestyle='--', alpha=0.5)

        # Annotate image names along the bottom
        ax = plt.gca()
        ax.set_yticks([])  # remove y ticks

        # Image-name labels
        for i, image_name in enumerate(sorted_image_names):
            offset = offset_dict[image_name]
            next_offset = total_width
            if i < len(sorted_image_names) - 1:
                next_offset = offset_dict[sorted_image_names[i+1]]

            # Center position
            center_pos = (offset + next_offset) / 2

            # Label at the bottom
            plt.text(center_pos, -0.2, image_name,
                   horizontalalignment='center',
                   verticalalignment='top',
                   fontsize=10,
                   rotation=0)  # horizontal label (no rotation)

        plt.colorbar(label='Lamina intensity (normalized)', orientation='horizontal', pad=0.2)
        plt.title("Lamina intensity heatmap (continuous depth) - with edge fill")
        plt.xlabel("Continuous depth position (px)")
        plt.tight_layout()

        # Save the heatmap
        heatmap_path = os.path.join(base_output_dir, "layer_intensity_heatmap.png")
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"Lamina intensity heatmap saved to: {heatmap_path}")

        # ==============================================
        # 4. Transverse intensity curve + histogram of the heatmap data
        # ==============================================
        # Gaussian-smooth the 1D heatmap data
        smoothed_intensity = gaussian_filter1d(heatmap_data, sigma=5)

        edge_filter_percent = 0.05

        # Fill the edge regions of each image
        for i, image_name in enumerate(sorted_image_names):
            offset = offset_dict[image_name]
            image_width = image_widths.get(image_name, 800)

            edge_filter_pixels = max(10, int(image_width * edge_filter_percent))

            img_start = int(offset)
            img_end = int(offset + image_width)

            img_start = max(0, img_start)
            img_end = min(total_width, img_end)

            if img_end - img_start > edge_filter_pixels * 2:  # make sure the image is large enough
                valid_start = img_start + edge_filter_pixels
                valid_end = img_end - edge_filter_pixels

                valid_region = smoothed_intensity[valid_start:valid_end]

                if len(valid_region) > 0 and np.sum(valid_region) > 0:
                    left_boundary = smoothed_intensity[valid_start] if valid_start < total_width else 0
                    right_boundary = smoothed_intensity[valid_end-1] if valid_end-1 >= 0 else 0
                    avg_intensity = np.mean(valid_region[valid_region > 0])

                    for x in range(img_start, valid_start):
                        if x < total_width:
                            distance = x - img_start
                            weight = distance / edge_filter_pixels if edge_filter_pixels > 0 else 0
                            fill_value = (1 - weight) * avg_intensity * 0.5 + weight * left_boundary
                            smoothed_intensity[x] = fill_value

                    for x in range(valid_end, img_end):
                        if x < total_width:
                            distance = x - valid_end
                            weight = distance / edge_filter_pixels if edge_filter_pixels > 0 else 0
                            fill_value = (1 - weight) * right_boundary + weight * avg_intensity * 0.5
                            smoothed_intensity[x] = fill_value

                print(f"Image {image_name}: edge fill done (pixel range: {img_start}-{img_end}, filter zone: {edge_filter_pixels}px)")

        # Plot
        plt.figure(figsize=(15, 6))

        # Curve
        x_range = np.arange(total_width)
        plt.plot(x_range, smoothed_intensity, 'b-', linewidth=2, label="Lamina intensity")

        # Highlight edge-fill regions per image
        for i, image_name in enumerate(sorted_image_names):
            offset = offset_dict[image_name]
            image_width = image_widths.get(image_name, 800)
            edge_filter_pixels = max(10, int(image_width * edge_filter_percent))

            img_start = int(offset)
            img_end = int(offset + image_width)

            # Highlight the fill region
            if edge_filter_pixels > 0:
                # Left fill region
                left_fill_end = min(img_start + edge_filter_pixels, total_width)
                if img_start < left_fill_end:
                    x_fill = np.arange(img_start, left_fill_end)
                    if len(x_fill) > 0:
                        plt.fill_between(x_fill, smoothed_intensity[x_fill],
                                       color='orange', alpha=0.3,
                                       label='Edge-fill region' if i == 0 else '')

                # Right fill region
                right_fill_start = max(img_end - edge_filter_pixels, 0)
                if right_fill_start < img_end and img_end <= total_width:
                    x_fill = np.arange(right_fill_start, min(img_end, total_width))
                    if len(x_fill) > 0:
                        plt.fill_between(x_fill, smoothed_intensity[x_fill],
                                       color='orange', alpha=0.3)

        # Image boundary markers
        for image_name in sorted_image_names:
            offset = offset_dict[image_name]
            if offset > 0:
                plt.axvline(x=offset, color='gray', linestyle='--', alpha=0.5)
                plt.text(offset + 10, plt.ylim()[1] * 0.9 if plt.ylim()[1] > 0 else 1,
                         image_name, rotation=90, verticalalignment='top')

        plt.title("Transverse lamina intensity (with edge fill)")
        plt.xlabel("Continuous depth position (px)")
        plt.ylabel("Lamina intensity")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Save the curve
        curve_path = os.path.join(base_output_dir, "layer_intensity_curve.png")
        plt.tight_layout()
        plt.savefig(curve_path, dpi=150)
        plt.close()  # free memory

        print(f"Transverse lamina intensity curve saved to: {curve_path}")

    # Make sure every figure is closed
    plt.close('all')
    print("Batch visualization complete")
