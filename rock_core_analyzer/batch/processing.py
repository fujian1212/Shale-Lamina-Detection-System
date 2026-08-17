#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-image and folder batch processing."""

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

def process_image(image_path, output_dir="output"):
    """Process a single rock-core image.

    Args:
        image_path: Path to the rock-core image.
        output_dir: Output directory.
    """
    try:
        detector = RockCoreLayerDetector(image_path)
        detector.detect_layers()
        detector.calculate_statistics()
        detector.export_results(output_dir)
        print(f"Image processed successfully: {image_path}")
    except Exception as e:
        print(f"Error while processing image: {str(e)}")
def process_folder(folder_path, output_dir="output", image_ext=".jpg"):
    """Process all rock-core images in a folder.

    Args:
        folder_path: Folder containing rock-core images.
        output_dir: Output directory.
        image_ext: Image file extension.

    Returns:
        Number of images processed.
    """
    try:
        # Normalize paths via Path
        folder_path = Path(folder_path)
        output_path = Path(output_dir)

        # Create output directory
        output_path.mkdir(exist_ok=True, parents=True)

        # Validate the folder
        if not folder_path.exists():
            print(f"Error: folder does not exist: {folder_path}")
            return 0

        if not folder_path.is_dir():
            print(f"Error: the path is not a folder: {folder_path}")
            return 0

        # Gather all image files matching the extension
        image_files = sorted([f for f in folder_path.iterdir()
                             if f.is_file() and f.suffix.lower() == image_ext.lower()
                                or (not image_ext.startswith('.') and f.suffix.lower() == f".{image_ext.lower()}")])

        if not image_files:
            print(f"No {image_ext} images found in {folder_path}")
            return 0

        print(f"Found {len(image_files)} {image_ext} image(s); processing...")

        # Container for per-image results
        all_results = []
        processed_count = 0

        # Create a subdirectory for each image
        for i, image_path in enumerate(image_files):
            # Subdirectory named after the original filename
            image_name = image_path.stem
            image_output_dir = output_path / image_name

            try:
                print(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
                print(f"Full path: {image_path.absolute()}")

                # Initialize the detector and process the image
                detector = RockCoreLayerDetector(str(image_path.absolute()))
                detector.output_dir = str(image_output_dir)

                # Detect laminae
                detector.detect_layers()

                # Calculate statistics
                _, _, position_df = detector.calculate_statistics()

                # Export per-image results
                detector.export_results(str(image_output_dir))

                all_results.append({
                    "filename": image_path.name,
                    "detector": detector,
                    "position_df": position_df,
                    "index": i
                })

                processed_count += 1

            except Exception as e:
                print(f"Error while processing {image_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()

        if processed_count > 0:
            # Merge per-image results
            merge_batch_results(str(output_path), [r["detector"].output_dir for r in all_results])

        print(f"Done. Successfully processed {processed_count}/{len(image_files)} image(s).")
        return processed_count
    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0
