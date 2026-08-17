#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch worker subprocesses."""

import os
import sys
import argparse
import platform
import psutil
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from ttkthemes import ThemedTk
import re
import time
import traceback
import gc
import cv2
from PIL import Image, ImageTk, ImageOps
import pandas as pd

from rock_core_analyzer.core import RockCoreLayerDetector
from rock_core_analyzer.batch import merge_batch_results

def _apply_depth_interpolation_to_detector(detector, start_depth, end_depth):
    """Mirror the single-image GUI's depth-interpolation step inside the worker.

    The single-image flow assigns a linear pixel->depth mapping to
    ``detailed`` / ``position`` / ``laminae`` DataFrames after
    ``calculate_statistics``. We replicate the same logic here so the batch
    output Excel files carry a ``depth_m`` column when a depth range is set.
    """
    stats = detector.layer_stats or {}
    if not stats:
        return

    image_width = getattr(detector, "width", 0) or 0
    if image_width <= 0 or end_depth == start_depth:
        return

    depth_per_pixel = (end_depth - start_depth) / float(image_width)

    # detailed DataFrame: pick the first available position column
    detailed_df = stats.get("detailed")
    if detailed_df is not None and not detailed_df.empty:
        for col in ("position_x", "position_x_px", "x_position", "x", "row_y"):
            if col in detailed_df.columns:
                detailed_df = detailed_df.copy()
                detailed_df["depth_m"] = (start_depth + detailed_df[col] * depth_per_pixel).round(3)
                stats["detailed"] = detailed_df
                break

    # position DataFrame uses ``position_px`` natively
    position_df = stats.get("position")
    if position_df is not None and not position_df.empty:
        for col in ("position_px", "position", "position_x", "x"):
            if col in position_df.columns:
                position_df = position_df.copy()
                position_df["depth_m"] = (start_depth + position_df[col] * depth_per_pixel).round(3)
                stats["position"] = position_df
                break

    # laminae DataFrame -- annotate every unique lamina's depth
    laminae_df = stats.get("laminae")
    if laminae_df is not None and not laminae_df.empty and "x_pos_px_mean" in laminae_df.columns:
        laminae_df = laminae_df.copy()
        laminae_df["depth_m"] = (start_depth + laminae_df["x_pos_px_mean"] * depth_per_pixel).round(3)
        stats["laminae"] = laminae_df

    # Persist depth range in the summary dict for downstream consumers
    summary = stats.get("summary")
    if isinstance(summary, dict):
        summary["depth_range_start_m"] = start_depth
        summary["depth_range_end_m"] = end_depth
        summary["depth_per_pixel_m"] = round(depth_per_pixel, 6)
        stats["summary"] = summary

    detector.layer_stats = stats


def _batch_worker(args):
    """Batch worker: process a single image.

    Must live at module top level so ProcessPoolExecutor can pickle it.
    ``args`` is a dict containing every parameter the worker needs.
    Returns ``(success: bool, image_file: str, output_dir: str, error_msg: str)``.
    """
    import matplotlib
    matplotlib.use('Agg')  # disable interactive backend in worker
    
    image_file = args.get("image_file", "")
    image_path = args["image_path"]
    output_dir = args["output_dir"]
    
    try:
        detector = RockCoreLayerDetector(image_path)
        detector.output_dir = output_dir
        # In batch mode, only emit final results and skip diagnostic intermediates
        # (binary / canny / validation_lines / validated_grid)
        detector.save_diagnostics = False

        # Batch lamina orientation policy (sub-folder calibration + vertical snap)
        if args.get("batch_lamina_mode"):
            detector.batch_lamina_mode = True
            detector.batch_group_slope_hint = args.get("batch_group_slope_hint", 0.0)
            detector.batch_max_dip_after_align_deg = float(
                args.get("batch_max_dip_after_align_deg", 7.0)
            )
            detector.batch_force_vertical_after_align = bool(
                args.get("batch_force_vertical_after_align", True)
            )
        
        if args.get("pixel_per_mm") is not None:
            detector.pixel_per_mm = args["pixel_per_mm"]
        
        detector.preprocess_image(
            blur_size=args["blur_size"],
            clahe_clip=args["clahe_clip"],
            clahe_grid=args["clahe_grid"],
            brightness=args["brightness"],
            contrast=args["contrast"],
            gamma=args["gamma"]
        )
        
        batch_scan_lines = args.get("batch_scan_lines")
        if batch_scan_lines is None:
            scan_lines = None
            scan_line_count = args["scan_line_count"]
        elif isinstance(batch_scan_lines, int):
            scan_lines = None
            scan_line_count = batch_scan_lines
        else:
            scan_lines = batch_scan_lines
            scan_line_count = len(batch_scan_lines)
        
        detect_success = detector.detect_layers(
            threshold_method=args["threshold_method"],
            min_layer_width=args["min_layer_width"],
            scan_lines=scan_lines,
            scan_line_count=scan_line_count,
            min_validation_lines=args["min_validation_lines"],
            align_core=args["align_core"],
            alignment_angle=args["alignment_angle"]
        )
        
        if not detect_success:
            return (False, image_file, output_dir, "No valid laminae detected")

        detector.calculate_statistics()

        # Apply depth interpolation when the GUI passes a depth range
        enable_depth = bool(args.get("enable_depth_range"))
        start_depth = args.get("start_depth")
        end_depth = args.get("end_depth")
        if enable_depth and start_depth is not None and end_depth is not None and start_depth != end_depth:
            try:
                _apply_depth_interpolation_to_detector(detector, float(start_depth), float(end_depth))
            except Exception as depth_err:
                # Depth interpolation should never block the main result export
                print(f"[Batch worker] depth interpolation failed: {depth_err}")

        detector.export_results(output_dir)
        return (True, image_file, output_dir, "")

    except Exception as e:
        import traceback as tb
        return (False, image_file, output_dir, f"{e}\n{tb.format_exc()}")
def _paper_export_worker(args):
    """Batch paper-export worker: re-run detection for a single image and export paper figures.

    Returns ``(success: bool, image_file: str, paper_dir: str, error_msg: str)``.
    """
    import matplotlib
    matplotlib.use('Agg')
    
    image_file = args.get("image_file", "")
    image_path = args["image_path"]
    output_dir = args["output_dir"]
    
    try:
        detector = RockCoreLayerDetector(image_path)
        detector.output_dir = output_dir
        
        if args.get("pixel_per_mm") is not None:
            detector.pixel_per_mm = args["pixel_per_mm"]
        
        detector.preprocess_image(
            blur_size=args["blur_size"],
            clahe_clip=args["clahe_clip"],
            clahe_grid=args["clahe_grid"],
            brightness=args["brightness"],
            contrast=args["contrast"],
            gamma=args["gamma"]
        )
        
        batch_scan_lines = args.get("batch_scan_lines")
        if batch_scan_lines is None:
            scan_lines = None
            scan_line_count = args["scan_line_count"]
        elif isinstance(batch_scan_lines, int):
            scan_lines = None
            scan_line_count = batch_scan_lines
        else:
            scan_lines = batch_scan_lines
            scan_line_count = len(batch_scan_lines)
        
        detect_success = detector.detect_layers(
            threshold_method=args["threshold_method"],
            min_layer_width=args["min_layer_width"],
            scan_lines=scan_lines,
            scan_line_count=scan_line_count,
            min_validation_lines=args["min_validation_lines"],
            align_core=args["align_core"],
            alignment_angle=args["alignment_angle"]
        )
        
        if not detect_success:
            return (False, image_file, output_dir, "No valid laminae detected")

        detector.calculate_statistics()

        # Trigger paper-figure export (written into ``paper_export/`` under this image's directory)
        paper_dir = detector.export_paper_figures(
            output_dir,
            start_depth=args.get("start_depth"),
            end_depth=args.get("end_depth")
        )
        return (True, image_file, paper_dir, "")
    
    except Exception as e:
        import traceback as tb
        return (False, image_file, output_dir, f"{e}\n{tb.format_exc()}")
