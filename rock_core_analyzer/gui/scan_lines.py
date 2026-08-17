#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scan-line picking."""

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
from rock_core_analyzer.gui.utils import natural_sort_key, get_default_settings, get_system_info, resource_path
from rock_core_analyzer.gui.workers import _batch_worker, _paper_export_worker

class ScanLinesMixin:
    def select_scan_lines(self):
        """Pick custom scan lines."""
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image file first")
            return

        # Open image
        try:
            # Validate the image path
            image_path = self.image_path.get()
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file does not exist: {image_path}")

            # Read the image
            image = cv2.imread(image_path)
            if image is None and any(ord(c) > 127 for c in image_path):
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                raise FileNotFoundError(f"Failed to load image: {image_path}")

            # Scan-line picker window
            self.scan_line_window = tk.Toplevel(self.root)
            self.scan_line_window.title("Pick scan lines")
            self.scan_line_window.geometry("800x600")

            # Display the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save the original image dimensions
            self.original_height, self.original_width = image_rgb.shape[:2]
            print(f"Original image size: {self.original_width} x {self.original_height}")

            # If the image is too wide, crop horizontally
            max_display_width = 1200  # max display width
            crop_start_x = 0

            if self.original_width > max_display_width:
                # Start cropping from the horizontal center
                crop_start_x = (self.original_width - max_display_width) // 2
                crop_end_x = crop_start_x + max_display_width

                # Crop the central strip
                image_rgb = image_rgb[:, crop_start_x:crop_end_x]
                print(f"Image too wide ({self.original_width}px); cropping centre: {crop_start_x} - {crop_end_x}")

                # Update the working size
                current_width = max_display_width
            else:
                current_width = self.original_width

            # Resize to fit the window
            max_width = 700
            max_height = 500
            scale_x = max_width / current_width
            scale_y = max_height / self.original_height
            scale = min(scale_x, scale_y)

            print(f"Scale: {scale:.4f} (X: {scale_x:.4f}, Y: {scale_y:.4f})")

            # Compute display size
            if scale < 1:
                display_width = int(current_width * scale)
                display_height = int(self.original_height * scale)
                image_rgb = cv2.resize(image_rgb, (display_width, display_height), interpolation=cv2.INTER_AREA)
                print(f"Resized display size: {display_width} x {display_height}")
            else:
                display_width = current_width
                display_height = self.original_height
                scale = 1.0
                print(f"No resize needed; display size: {display_width} x {display_height}")

            # Convert to a tk-displayable image
            img = Image.fromarray(image_rgb)
            img_tk = ImageTk.PhotoImage(img)

            # Canvas
            canvas = tk.Canvas(self.scan_line_window, width=display_width, height=display_height)
            canvas.pack(padx=10, pady=10)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            setattr(canvas, 'image', img_tk)  # retain a reference

            # Scan-line state
            self.canvas = canvas
            self.scan_lines_y = []
            self.scan_line_ids = []
            self.display_height = display_height
            self.display_width = display_width
            self.scale_factor = scale  # remember the scale factor
            self.crop_start_x = crop_start_x  # remember the crop offset

            print(f"Stored scale factor: {self.scale_factor:.4f}")
            print(f"Stored crop offset: {self.crop_start_x}px")

            # Click handler
            canvas.bind("<Button-1>", self.add_scan_line)

            # Instructions and buttons
            instruction_text = (
                f"Click the image to add a scan line "
                f"(original: {self.original_width}x{self.original_height}, "
                f"display: {display_width}x{display_height}, "
                f"scale: {scale:.3f}"
            )
            if crop_start_x > 0:
                instruction_text += f", crop offset: {crop_start_x}px"
            instruction_text += ")"

            instruction = ttk.Label(self.scan_line_window, text=instruction_text)
            instruction.pack(pady=5)

            btn_frame = ttk.Frame(self.scan_line_window)
            btn_frame.pack(pady=10)

            ttk.Button(btn_frame, text="OK", command=self.confirm_scan_lines).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Clear all", command=self.clear_current_scan_lines).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Cancel", command=lambda: self.scan_line_window.destroy()).pack(side=tk.LEFT, padx=5)

            # Live count of currently selected scan lines
            self.scan_line_count_var = tk.StringVar(self.root, value="Selected scan lines: 0")
            ttk.Label(self.scan_line_window, textvariable=self.scan_line_count_var).pack(pady=5)

            # Switch to manual mode
            self.scan_mode.set("manual")

        except Exception as e:
            messagebox.showerror("Error", f"Cannot open the image for scan-line selection: {str(e)}")
    def add_scan_line(self, event):
        """Add a scan line."""
        y = event.y
        x = event.x
        print(f"Click position: ({x}, {y})")

        # Draw the line on the canvas
        line_id = self.canvas.create_line(0, y, self.display_width, y, fill="red", width=2)

        # Track the line id and display-y coordinate
        self.scan_line_ids.append(line_id)
        self.scan_lines_y.append(y)

        # *** Print the converted original coordinates for debugging ***
        original_y = int(y / self.scale_factor) if hasattr(self, 'scale_factor') and self.scale_factor < 1 else y

        # Apply the crop offset to recover the original x as well
        crop_offset = getattr(self, 'crop_start_x', 0)
        original_x = int(x / self.scale_factor) + crop_offset if hasattr(self, 'scale_factor') else x + crop_offset

        print(f"Added scan line: display=({x}, {y}), original=({original_x}, {original_y}), "
              f"scale={getattr(self, 'scale_factor', 1.0):.4f}, crop offset={crop_offset}px")

        # Update the counter
        self.scan_line_count_var.set(f"Selected scan lines: {len(self.scan_lines_y)}")
    def clear_current_scan_lines(self):
        """Clear all scan lines in the current window."""
        # Remove the lines from the canvas
        for line_id in self.scan_line_ids:
            self.canvas.delete(line_id)

        # Reset state
        self.scan_line_ids.clear()
        self.scan_lines_y.clear()

        # Update the counter
        self.scan_line_count_var.set("Selected scan lines: 0")
    def confirm_scan_lines(self):
        """Confirm the picked scan lines."""
        if not self.scan_lines_y:
            messagebox.showinfo("Info", "No scan line was selected; auto mode will be used")
            self.scan_mode.set("auto")
            self.scan_line_window.destroy()
            return

        # Convert to original-image coordinates
        # Step 1: GUI display y -> original image y
        original_scan_lines = []
        print(f"\n=== Scan-line coordinate conversion ===")
        print(f"GUI display image size: {getattr(self, 'display_width', 'N/A')} x {getattr(self, 'display_height', 'N/A')}")
        print(f"Original image size:    {getattr(self, 'original_width', 'N/A')} x {getattr(self, 'original_height', 'N/A')}")
        print(f"GUI scale factor:       {getattr(self, 'scale_factor', 'N/A'):.4f}")
        print(f"Crop offset:            {getattr(self, 'crop_start_x', 0)}px")

        for i, display_y in enumerate(self.scan_lines_y):
            # Step 1: GUI display y -> original image y
            if hasattr(self, 'scale_factor') and self.scale_factor < 1:
                original_y = int(display_y / self.scale_factor)
            else:
                original_y = display_y

            original_scan_lines.append(original_y)
            print(f"Scan line {i+1}: GUI display Y={display_y} -> original image Y={original_y}")

        print(f"Scan lines mapped to the original image: {original_scan_lines}")

        # *** Step 2: detector resizes the image based on environment variables ***
        max_image_size = int(os.environ.get('ROCK_MAX_IMAGE_SIZE', '0'))
        memory_limit = int(os.environ.get('ROCK_MEMORY_LIMIT', '0'))

        print(f"\n=== Detector resize parameters ===")
        print(f"Max image-size limit: {max_image_size}")
        print(f"Memory limit:         {memory_limit} MB")

        # Mirror the detector's resize logic
        detector_scan_lines = []
        original_width = getattr(self, 'original_width', 0)
        original_height = getattr(self, 'original_height', 0)

        # Working size as the detector sees it
        detector_width = original_width
        detector_height = original_height
        detector_scale = 1.0

        # Apply the size-limit downscale
        if max_image_size > 0 and (original_width > max_image_size or original_height > max_image_size):
            size_scale = min(max_image_size / original_width, max_image_size / original_height)
            if size_scale < 1.0:
                detector_width = int(original_width * size_scale)
                detector_height = int(original_height * size_scale)
                detector_scale = size_scale
                print(f"Detector size-limit downscale: {size_scale:.4f}")

        # Apply the memory-limit downscale (rough estimate)
        if memory_limit > 0:
            # Estimated memory footprint (3 channels * 1 byte)
            estimated_memory = (detector_width * detector_height * 3) / (1024 * 1024)
            if estimated_memory > memory_limit:
                memory_scale = (memory_limit / estimated_memory) ** 0.5
                detector_width = int(detector_width * memory_scale)
                detector_height = int(detector_height * memory_scale)
                detector_scale *= memory_scale
                print(f"Detector memory-limit downscale: {memory_scale:.4f}")

        print(f"Final detector image size: {detector_width} x {detector_height}")
        print(f"Total detector scale:      {detector_scale:.4f}")

        # Map original-image coordinates to detector coordinates
        for i, original_y in enumerate(original_scan_lines):
            if detector_scale < 1.0:
                detector_y = int(original_y * detector_scale)
            else:
                detector_y = original_y

            detector_scan_lines.append(detector_y)
            print(f"Scan line {i+1}: original Y={original_y} -> detector Y={detector_y}")

        # Make sure the scan lines fall inside the detector image
        valid_detector_scan_lines = []
        for y in detector_scan_lines:
            if 0 <= y < detector_height:
                valid_detector_scan_lines.append(y)
            else:
                print(f"Warning: scan line Y={y} is outside the detector image range [0, {detector_height}); ignored")

        if not valid_detector_scan_lines:
            messagebox.showerror("Error", "All scan lines fall outside the detector image range!")
            return

        self.custom_scan_lines = valid_detector_scan_lines

        # Update the scan-line count
        self.scan_line_count.set(len(self.custom_scan_lines))

        print(f"Final detector scan-line positions: {self.custom_scan_lines}")
        print(f"=== Conversion complete ===\n")

        crop_info = f", crop offset: {getattr(self, 'crop_start_x', 0)}px" if getattr(self, 'crop_start_x', 0) > 0 else ""
        messagebox.showinfo(
            "Success",
            f"Selected {len(self.custom_scan_lines)} scan line(s)\n"
            f"GUI display image: {getattr(self, 'display_width', 'N/A')}x{getattr(self, 'display_height', 'N/A')}{crop_info}\n"
            f"Original image: {original_width}x{original_height}\n"
            f"Detector image: {detector_width}x{detector_height}\n"
            f"Final scan-line positions: {self.custom_scan_lines}",
        )
        self.scan_line_window.destroy()
    def clear_scan_lines(self):
        """Clear custom scan lines."""
        self.custom_scan_lines = []
        self.scan_mode.set("auto")
        # Restore the default scan-line count
        self.scan_line_count.set(5)  # default = 5
        messagebox.showinfo("Info", "Custom scan lines cleared; auto mode will be used")
