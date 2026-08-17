#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main application window."""

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
from rock_core_analyzer.gui.utils import get_default_settings, get_system_info
from .ui_setup import UiSetupMixin
from .single_image import SingleImageMixin
from .export_ui import ExportUiMixin
from .scan_lines import ScanLinesMixin
from .batch_ui import BatchUiMixin

class RockCoreAnalyzerApp(UiSetupMixin, SingleImageMixin, ExportUiMixin, ScanLinesMixin, BatchUiMixin):
    pass

def main(image=None, output=None):
    """Main program entry point."""
    print("Application starting...")

    # CLI arguments
    parser = argparse.ArgumentParser(description='Rock Core Lamina Identification System')

    # Detect default settings
    default_settings = get_default_settings()

    # Runtime parameters
    parser.add_argument('--max-image-size', type=int, default=default_settings['max_image_size'],
                        help=f'Maximum image size for processing (default: {default_settings["max_image_size"]} px)')
    parser.add_argument('--enable-multiprocessing', action='store_true',
                        help='Enable multiprocessing')
    parser.add_argument('--threads', type=int, default=default_settings['threads'],
                        help=f'Thread count (0 = auto, default: {default_settings["threads"]})')
    parser.add_argument('--memory-limit', type=int, default=default_settings['memory_limit'],
                        help=f'Memory limit (MB; 0 = unlimited, default: {default_settings["memory_limit"]})')
    parser.add_argument('--optimize-ui', action='store_true', default=True,
                        help='Enable UI responsiveness (default: on)')
    parser.add_argument('--no-optimize-ui', action='store_false', dest='optimize_ui',
                        help='Disable UI responsiveness optimization')
    parser.add_argument('--cache-size', type=int, default=default_settings['cache_size'],
                        help=f'Image cache size (default: {default_settings["cache_size"]})')
    parser.add_argument('--progressive-loading', action='store_true', default=True,
                        help='Enable progressive loading (default: on)')
    parser.add_argument('--no-progressive-loading', action='store_false', dest='progressive_loading',
                        help='Disable progressive loading')
    parser.add_argument('--lazy-load', action='store_true', default=True,
                        help='Enable lazy loading (default: on)')
    parser.add_argument('--no-lazy-load', action='store_false', dest='lazy_load',
                        help='Disable lazy loading')

    # Application arguments -- parsed from CLI if not passed in
    parser.add_argument('--batch', action='store_true', help='Enable batch mode')
    parser.add_argument('--list', type=str, help='Batch image list file')
    parser.add_argument('--base-dir', type=str, help='Image base directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--image', type=str, help='Path to a single image to process')
    parser.add_argument('--info', action='store_true', help='Show system info')

    args = parser.parse_args()

    # Function parameters override CLI arguments
    if image:
        args.image = image
    if output:
        args.output_dir = output

    # System info
    if args.info:
        system_info = get_system_info()
        print("\nSystem info:")
        print("-" * 40)
        for key, value in system_info.items():
            print(f"{key}: {value}")
        print("-" * 40 + "\n")

    # Environment variables
    os.environ['ROCK_MAX_IMAGE_SIZE'] = str(args.max_image_size)
    os.environ['ROCK_ENABLE_MULTIPROCESSING'] = '1' if args.enable_multiprocessing else '0'

    if args.threads > 0:
        os.environ['ROCK_THREADS'] = str(args.threads)
    else:
        # Default to physical CPU core count
        cores = psutil.cpu_count(logical=False) or 2
        os.environ['ROCK_THREADS'] = str(cores)

    if args.memory_limit > 0:
        os.environ['ROCK_MEMORY_LIMIT'] = str(args.memory_limit)

    os.environ['ROCK_OPTIMIZE_UI'] = '1' if args.optimize_ui else '0'
    os.environ['ROCK_PROGRESSIVE_LOADING'] = '1' if args.progressive_loading else '0'
    os.environ['ROCK_LAZY_LOAD'] = '1' if args.lazy_load else '0'
    os.environ['ROCK_CACHE_SIZE'] = str(args.cache_size)

    print("Starting Rock Core Lamina Identification System...")

    # Matplotlib font setup
    import matplotlib as mpl
    mpl.rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # Create the main window directly (no separate splash screen)
        root = ThemedTk(theme='arc')
        root.title("Rock Core Lamina Identification System")
        root.geometry("1280x800")

        # App instance
        app = RockCoreAnalyzerApp(root)

        # Load a single image immediately when requested
        if args.image and os.path.exists(args.image):
            app.image_path.set(args.image)
            root.after(1000, lambda: app.analyze_image())

        # Kick off batch mode if requested
        elif args.batch and args.list and args.base_dir:
            try:
                # Enable batch mode
                app.batch_mode.set(True)
                app.toggle_batch_mode()

                # Set the base directory
                app.image_path.set(args.base_dir)

                # Output directory
                if args.output_dir:
                    app.save_path.set(args.output_dir)

                # Read the image-list file
                with open(args.list, 'r', encoding='utf-8') as f:
                    file_lines = f.readlines()

                # Parse file paths
                image_files = []
                for line in file_lines:
                    line = line.strip()
                    # Skip blanks and comments
                    if not line or line.startswith('#'):
                        continue
                    image_files.append(line)

                if not image_files:
                    print("Error: image list is empty")
                    return

                # Push the list into the app
                app.batch_image_files = image_files
                app.file_count_var.set(f"Loaded {len(image_files)} image path(s) from file")

                # Kick off batch processing (wait for the UI to settle)
                root.after(1000, lambda: app.process_batch_from_list(
                    image_files,
                    args.base_dir,
                    args.output_dir
                ))
            except Exception as e:
                print(f"Batch initialization error: {str(e)}\n{traceback.format_exc()}")

        # Main loop
        root.mainloop()

    except Exception as e:
        print(f"Startup error: {str(e)}")
        traceback.print_exc()
        try:
            messagebox.showerror("Startup failed", f"Application failed to start:\n{str(e)}")
        except Exception:
            pass


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
