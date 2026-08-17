#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rock-core lamina analysis (compatibility entry point; prefer the rock_core_analyzer package)."""

from rock_core_analyzer.core import RockCoreLayerDetector
from rock_core_analyzer.batch import (
    process_image,
    process_folder,
    merge_batch_results,
    merge_position_info,
    generate_continuous_position_statistics,
    generate_batch_summary_statistics,
    generate_batch_processing_report,
    create_empty_results,
    create_batch_visualizations,
)

__all__ = [
    "RockCoreLayerDetector",
    "process_image",
    "process_folder",
    "merge_batch_results",
    "merge_position_info",
    "generate_continuous_position_statistics",
    "generate_batch_summary_statistics",
    "generate_batch_processing_report",
    "create_empty_results",
    "create_batch_visualizations",
]

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Rock-core lamina analysis tool")
    parser.add_argument("path", help="Path to a rock-core scan image or a folder containing images")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Batch-process all images in the folder")
    parser.add_argument("--ext", default=".jpg", help="Image file extension to scan for in batch mode")
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.path):
        process_folder(args.path, args.output, args.ext)
    else:
        process_image(args.path, args.output)
