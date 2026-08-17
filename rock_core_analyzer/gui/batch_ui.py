#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch processing UI."""

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

class BatchUiMixin:
    def load_file_list(self):
        """Load the file list from the batch folder."""
        folder_path = self.image_path.get()
        
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder containing images first")
            return
        
        if not os.path.isdir(folder_path):
            messagebox.showerror("Error", f"The path is not a valid folder: {folder_path}")
            return
        
        # Normalize path
        folder_path = os.path.normpath(os.path.abspath(folder_path))
        print(f"Loading file list: {folder_path}")
        
        try:
            # Enumerate files in the folder
            image_ext = self.image_ext.get()
            include_sub = bool(getattr(self, 'include_subfolders', None) and self.include_subfolders.get())
            
            # Filter image files
            self.batch_image_files = []
            if include_sub:
                # Walk sub-folders recursively; store paths relative to folder_path
                for root_dir, _dirs, files in os.walk(folder_path):
                    for f in files:
                        if f.lower().endswith(image_ext.lower()):
                            full_path = os.path.join(root_dir, f)
                            if os.path.isfile(full_path):
                                rel_path = os.path.relpath(full_path, folder_path)
                                self.batch_image_files.append(rel_path)
            else:
                all_files = os.listdir(folder_path)
                for f in all_files:
                    if f.lower().endswith(image_ext.lower()):
                        full_path = os.path.join(folder_path, f)
                        if os.path.isfile(full_path):
                            self.batch_image_files.append(f)
            
            # Sort by name (natural sort handles embedded numbers correctly)
            self.batch_image_files.sort(key=natural_sort_key)
            
            # Print sorted file order (debug)
            print(f"Sorted image order: {self.batch_image_files[:10]}")  # show first 10
            
            # Update UI
            file_count = len(self.batch_image_files)
            if file_count > 0:
                # Count affected sub-folders
                subfolders = set()
                for rel in self.batch_image_files:
                    sub = os.path.dirname(rel)
                    subfolders.add(sub if sub else '.')
                folder_info = f", spread across {len(subfolders)} sub-directories" if include_sub else ""
                self.file_count_var.set(f"Found {file_count} image file(s){folder_info}")
                self.batch_scan_btn.config(state=tk.NORMAL)  # enable the scan-line picker
                
                # Show a few file names
                if file_count > 0:
                    preview_files = self.batch_image_files[:min(5, file_count)]
                    file_names = ", ".join(preview_files)
                    print(f"Found {file_count} image file(s); first few: {file_names}")
                    extra = f"\n\nSpanning {len(subfolders)} sub-directories" if include_sub else ""
                    messagebox.showinfo("File list", f"Found {file_count} image file(s){extra}\n\nFirst few: {file_names}")
            else:
                self.file_count_var.set("No valid image files found")
                self.batch_scan_btn.config(state=tk.DISABLED)
                hint = " (recursive sub-folder search enabled)" if include_sub else ""
                messagebox.showinfo("Info", f"No {image_ext} image files found in {folder_path}{hint}")
        
        except Exception as e:
            print(f"Error loading file list: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error while loading the file list: {str(e)}")
            self.file_count_var.set("Failed to load files")
            self.batch_scan_btn.config(state=tk.DISABLED)
    def load_list_from_file(self):
        """Load an image list from a text file."""
        # Make sure a base folder was selected first
        base_folder = self.image_path.get()
        if not base_folder:
            messagebox.showerror("Error", "Please select a base folder containing images first")
            return
            
        if not os.path.isdir(base_folder):
            messagebox.showerror("Error", f"The path is not a valid folder: {base_folder}")
            return
            
        # Select the text file
        file_path = filedialog.askopenfilename(
            title="Select the text file containing the image list",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # user cancelled
            
        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = f.readlines()
            
            # Process file paths (skip blanks/comments/whitespace)
            image_files = []
            for line in file_lines:
                line = line.strip()
                # Skip blanks and comments
                if not line or line.startswith('#'):
                    continue
                image_files.append(line)
                
            if not image_files:
                messagebox.showinfo("Info", "The file list is empty")
                return
                
            # Update UI
            file_count = len(image_files)
            self.batch_image_files = image_files
            self.file_count_var.set(f"Loaded {file_count} image path(s) from file")
            self.batch_scan_btn.config(state=tk.NORMAL)  # enable the scan-line picker
            
            # Show a few file names
            preview_files = image_files[:min(5, file_count)]
            file_names = ", ".join(preview_files)
            print(f"Loaded {file_count} image path(s) from file; first few: {file_names}")
            
            # Verify the files exist
            existing_count = 0
            for img_file in image_files:
                full_path = os.path.join(base_folder, img_file) if not os.path.isabs(img_file) else img_file
                if os.path.isfile(full_path):
                    existing_count += 1
                    
            # Show the verification result
            if existing_count < file_count:
                messagebox.showwarning("Warning",
                    f"Loaded {file_count} image path(s); only {existing_count} file(s) exist\n\n" +
                    f"First few paths: {file_names}"
                )
            else:
                messagebox.showinfo("File list",
                    f"Loaded {file_count} image path(s); all files exist\n\n" +
                    f"First few paths: {file_names}"
                )
                
        except Exception as e:
            print(f"Error loading list from file: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error while loading list from file: {str(e)}")
            self.file_count_var.set("Failed to load files")
            self.batch_scan_btn.config(state=tk.DISABLED)
    def select_batch_scan_lines_ui(self):
        """UI entry point for picking batch-mode scan lines."""
        folder_path = self.image_path.get()
        
        if not folder_path or not os.path.isdir(folder_path) or not self.batch_image_files:
            messagebox.showerror("Error", "Please load the file list first")
            return
        
        try:
            # Pick the first image as the reference
            if len(self.batch_image_files) > 0:
                first_image_name = self.batch_image_files[0]
                first_image_path = os.path.join(folder_path, first_image_name)
                first_image_path = os.path.normpath(os.path.abspath(first_image_path))
                
                print(f"Selecting batch scan lines: using the first image {first_image_name}")
                print(f"Full path: {first_image_path}")
                
                # Make sure the file exists
                if not os.path.isfile(first_image_path):
                    raise FileNotFoundError(f"First image not found: {first_image_path}")
                
                # Call the scan-line picker
                result = self._select_batch_scan_lines(first_image_path)
                
                if result is not None:
                    if isinstance(result, list):
                        # Manual mode: returns a list of scan-line y coordinates
                        self.batch_scan_lines = result
                        line_count = len(result)
                        messagebox.showinfo("Scan lines set", f"Set {line_count} batch-mode scan line(s)")
                    elif isinstance(result, int):
                        # Auto mode: returns the scan-line count
                        self.batch_scan_lines = result
                        messagebox.showinfo("Scan lines set", f"Configured to auto-generate {result} scan line(s)")
                    else:
                        # Unknown type
                        self.batch_scan_lines = None
                        messagebox.showinfo("Defaults", "Default scan-line settings will be used")
                else:
                    # User cancelled
                    self.batch_scan_lines = None
                    print("User cancelled scan-line selection")
            else:
                messagebox.showerror("Error", "No valid image files found")
        
        except Exception as e:
            print(f"Error during batch scan-line selection: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error while picking batch scan lines: {str(e)}")
            self.batch_scan_lines = None
    def _select_batch_scan_lines(self, image_path):
        """Let the user pick scan lines for the batch.
        
        Args:
            image_path: path to the first image used for selecting scan lines.
            
        Returns:
            list or int: list of scan-line y coordinates in manual mode, count in auto mode, None if cancelled.
        """
        # Build a transient window that shows the image and lets the user pick scan lines
        select_window = tk.Toplevel(self.root)
        select_window.title("Pick batch scan lines")
        select_window.geometry("1000x800")
        select_window.transient(self.root)  # parent to the main window
        select_window.grab_set()  # modal dialog
        
        # Image-display canvas
        canvas_frame = ttk.Frame(select_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        
        canvas = tk.Canvas(
            canvas_frame, 
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set
        )
        
        h_scrollbar.config(command=canvas.xview)
        v_scrollbar.config(command=canvas.yview)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Load image
        try:
            # Load the original image
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)  # auto-rotate per EXIF
            
            # Store the original image size
            original_width, original_height = img.width, img.height
            
            # If the image is too wide, crop it horizontally
            max_display_width = 1200  # max display width
            crop_start_x = 0
            
            if original_width > max_display_width:
                # Compute the crop start position (centered)
                crop_start_x = (original_width - max_display_width) // 2
                crop_end_x = crop_start_x + max_display_width
                
                # Crop the centre strip
                img = img.crop((crop_start_x, 0, crop_end_x, original_height))
                print(f"Image too wide ({original_width}px); cropping centre: {crop_start_x} - {crop_end_x}")
                
                # Update the working size
                current_width = max_display_width
            else:
                current_width = original_width
            
            # Compute the canvas size (allowing for window borders and scrollbars)
            canvas_width = 950  # window width minus padding
            canvas_height = 650  # window height minus padding and control panel
            
            # Scale to fit the canvas while preserving aspect ratio
            scale_factor = min(canvas_width / current_width, canvas_height / original_height)
            
            # Correct the scale factor
            if scale_factor < 1.0:
                # Image needs downscaling
                new_width = int(current_width * scale_factor)
                new_height = int(original_height * scale_factor)
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_resized)
                display_width, display_height = new_width, new_height
                actual_scale_factor = scale_factor  # use the real scale factor
            else:
                # No downscaling needed; show at native size
                img_tk = ImageTk.PhotoImage(img)
                display_width, display_height = current_width, original_height
                actual_scale_factor = 1.0  # native size
            
            # Place the image on the canvas
            canvas.config(scrollregion=(0, 0, display_width, display_height))
            canvas_img = canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.itemconfig(canvas_img, image=img_tk)
            
            # Stash references and scale info
            setattr(canvas, 'img_tk', img_tk)
            setattr(canvas, 'original_image', img)
            setattr(canvas, 'scale_factor', actual_scale_factor)  # store the corrected scale factor
            setattr(canvas, 'display_width', display_width)
            setattr(canvas, 'display_height', display_height)
            setattr(canvas, 'original_width', original_width)  # remember original size
            setattr(canvas, 'original_height', original_height)
            setattr(canvas, 'crop_start_x', crop_start_x)  # remember crop offset
            
            # Display the scale info
            info_text = f"Image size: {original_width}x{original_height}"
            if crop_start_x > 0:
                info_text += f" (cropped: {crop_start_x}-{crop_start_x + max_display_width})"
            if scale_factor < 1.0:
                info_text += f" (scale: {scale_factor:.2f})"
            else:
                info_text += " (native size)"
                
            scale_info = ttk.Label(select_window, text=info_text)
            scale_info.pack(pady=(0, 5))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            select_window.destroy()
            return None
            
        # Store scan-line positions in original-image coordinates
        scan_lines = []
        
        # State variables for scan-line picking
        auto_mode = tk.BooleanVar(value=False)
        scan_line_count = tk.IntVar(value=self.scan_line_count.get())
        
        # Click handler for adding scan lines
        def add_scan_line(event):
            if auto_mode.get():
                return  # auto mode disables manual adds
                
            # Get the mouse position in canvas coordinates
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            
            # Convert to original-image coordinates
            original_y = int(y / getattr(canvas, 'scale_factor', 1.0))
            
            # If the image was cropped, shift x back to original-image coordinates
            crop_start_x = getattr(canvas, 'crop_start_x', 0)
            original_x = int(x / getattr(canvas, 'scale_factor', 1.0)) + crop_start_x
            
            # Validate the conversion
            original_height = getattr(canvas, 'original_height', 0)
            if original_y > original_height:
                print(f"WARNING: converted y ({original_y}) exceeds original image height ({original_height})!")
            
            # Record the original y in the scan-line list
            display_height = getattr(canvas, 'display_height', 0)
            if y >= 0 and y < display_height:
                scan_lines.append(original_y)
                
                # Draw the scan line on the canvas
                display_width = getattr(canvas, 'display_width', 0)
                line_id = canvas.create_line(
                    0, y, display_width, y, 
                    fill='red', width=2
                )
                
                # Annotate with the line number and coordinates
                display_text = f"Line {len(scan_lines)} (y={original_y}"
                if crop_start_x > 0:
                    display_text += f", x_offset={crop_start_x}"
                display_text += ")"
                
                text_id = canvas.create_text(
                    10, y-15, 
                    text=display_text, 
                    fill='red', 
                    font=('Arial', 12, 'bold'),
                    anchor=tk.W
                )
                
                # Update the UI
                scan_line_status.config(text=f"Added {len(scan_lines)} scan line(s)")
        
        # Bind the click handler
        canvas.bind("<Button-1>", add_scan_line)
        
        # Build the control panel
        control_frame = ttk.Frame(select_window)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Auto / manual mode toggle
        mode_frame = ttk.LabelFrame(control_frame, text="Scan-line mode")
        mode_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Auto-scan-line drawing helper
        def update_auto_scan_lines(*args):
            # Clear existing scan lines
            clear_scan_lines()
            
            # Draw preview scan lines in auto mode
            if auto_mode.get():
                # Read the requested scan-line count
                count = scan_line_count.get()
                if count < 1:
                    count = 1
                
                # Compute spacing
                if count > 1:
                    step = original_height / (count + 1)
                else:
                    step = original_height / 2
                
                # Draw the auto scan-line preview
                auto_scan_lines = []
                for i in range(1, count+1):
                    # Compute original-image coordinates
                    original_y = int(i * step)
                    auto_scan_lines.append(original_y)
                    
                    # Convert to display coordinates
                    display_y = original_y * getattr(canvas, 'scale_factor', 1.0)
                    
                    # Draw the line
                    display_width = getattr(canvas, 'display_width', 0)
                    line_id = canvas.create_line(
                        0, display_y, display_width, display_y,
                        fill='blue', width=2, dash=(4, 4)  # dashed lines distinguish auto from manual
                    )
                    
                    # Add the label
                    text_id = canvas.create_text(
                        10, display_y-15, 
                        text=f"Auto {i}", 
                        fill='blue', 
                        font=('Arial', 12, 'bold'),
                        anchor=tk.W
                    )
                
                scan_line_status.config(text=f"Generated {count} auto scan line(s)")
        
        # Trace the variable so scan lines refresh when the count changes
        scan_line_count.trace("w", update_auto_scan_lines)
        
        ttk.Radiobutton(
            mode_frame, 
            text="Pick manually", 
            variable=auto_mode, 
            value=False,
            command=update_auto_scan_lines
        ).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Radiobutton(
            mode_frame, 
            text="Auto-generate", 
            variable=auto_mode, 
            value=True,
            command=update_auto_scan_lines
        ).pack(anchor=tk.W, padx=5, pady=2)
        
        # Auto-mode scan-line count
        auto_frame = ttk.Frame(control_frame)
        auto_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(auto_frame, text="Auto scan-line count:").pack(side=tk.LEFT)
        scan_line_spinbox = ttk.Spinbox(
            auto_frame, 
            from_=1, 
            to=50, 
            width=5, 
            textvariable=scan_line_count,
            command=update_auto_scan_lines  # refresh when arrow buttons change the value
        )
        scan_line_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Bind Enter so manual typing also refreshes
        scan_line_spinbox.bind('<Return>', lambda e: update_auto_scan_lines())
        
        # Status label
        scan_line_status = ttk.Label(control_frame, text="Pick scan lines or use auto mode")
        scan_line_status.pack(side=tk.LEFT, padx=10)
        
        # Button area
        button_frame = ttk.Frame(select_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Clear button
        def clear_scan_lines():
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=getattr(canvas, 'img_tk', None))
            scan_lines.clear()
            scan_line_status.config(text="All scan lines cleared")
        
        clear_button = ttk.Button(
            button_frame, 
            text="Clear scan lines", 
            command=clear_scan_lines
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Confirm / cancel buttons
        result = [None]  # wrap in a list so inner closures can mutate
        
        def on_confirm():
            if auto_mode.get():
                # In auto mode return the scan-line count instead of None
                count = scan_line_count.get()
                result[0] = count  # return scan-line count
                select_window.destroy()
            else:
                if not scan_lines:
                    messagebox.showwarning("Warning", "Please add at least one scan line")
                    return
                result[0] = scan_lines.copy()
                select_window.destroy()
        
        def on_cancel():
            result[0] = None
            select_window.destroy()
            
        confirm_button = ttk.Button(
            button_frame, 
            text="OK", 
            command=on_confirm
        )
        confirm_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = ttk.Button(
            button_frame, 
            text="Cancel", 
            command=on_cancel
        )
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Wait for the window to close
        select_window.wait_window()
        
        return result[0]
    def process_batch_from_list(self, file_list, base_folder, output_dir=None):
        """Process a batch of images from a given file list.
        
        Args:
            file_list (list): list of image paths (absolute or relative to ``base_folder``).
            base_folder (str): base directory containing the images.
            output_dir (str, optional): output directory; defaults to the configured save path.
        
        Returns:
            bool: whether processing succeeded.
        """
        print(f"=== Starting batch processing ===")
        print(f"File-list size: {len(file_list)}")
        print(f"Base folder: {base_folder}")
        print(f"Output directory: {output_dir}")
        
        if not file_list:
            print("Error: file list is empty")
            return
            
        # Configure the output directory
        if output_dir is None:
            # If no output directory was passed, check the UI setting
            user_save_path = self.save_path.get()
            if user_save_path:
                output_dir = os.path.join(user_save_path, "batch_results")
            else:
                output_dir = os.path.join(base_folder, "batch_results")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Resolved output directory: {output_dir}")
        print(f"Output directory exists: {os.path.exists(output_dir)}")
        
        total_files = len(file_list)
        self.progress_var.set(0)
        self.status_var.set("Starting batch processing...")
        
        # Read batch parameters
        threshold_method = self.threshold_method.get()
        min_layer_width = self.min_layer_width.get()
        scan_line_count = self.scan_line_count.get()
        min_validation_lines = self.min_validation_lines.get()
        align_core = self.align_core.get()
        alignment_angle = self.alignment_angle.get()
        
        print(f"Batch parameters:")
        print(f"  - threshold method: {threshold_method}")
        print(f"  - min lamina width: {min_layer_width}")
        print(f"  - scan-line count: {scan_line_count}")
        print(f"  - min validation lines: {min_validation_lines}")
        print(f"  - align core: {align_core}")
        print(f"  - alignment angle: {alignment_angle}")
        
        batch_scan_lines = getattr(self, 'batch_scan_lines', None)
        print(f"Batch scan-line setting: {batch_scan_lines}")

        # Sub-folder dip calibration (first 5 images per group, or all if fewer)
        detector_params = self._build_batch_detector_params(batch_scan_lines)
        file_groups = self._group_batch_files_by_subfolder(file_list)
        # Process the root group first (no sub-folder), then sub-folders in
        # natural order. ``ordered_file_list`` linearises ``file_groups`` so
        # that all files from one sub-folder appear consecutively -- this is
        # what lets us close out each sub-folder (merge + optional analysis)
        # before starting the next.
        group_order = sorted(
            file_groups.keys(),
            key=lambda s: (s != "", natural_sort_key(s) if s else [""]),
        )
        group_lamina_kwargs = {}
        ordered_file_list = []
        self.status_var.set("Calibrating lamina orientation per sub-folder...")
        self.root.update()
        for sub in group_order:
            items = file_groups[sub]
            group_out = os.path.join(output_dir, sub) if sub else output_dir
            os.makedirs(group_out, exist_ok=True)
            calib, lamina_kw = self._calibrate_batch_group(
                base_folder, sub, items, detector_params, group_output_dir=group_out
            )
            group_lamina_kwargs[sub] = lamina_kw
            ordered_file_list.extend(items)
            print(f"[Batch dip calib] group '{sub or 'root'}': "
                  f"reference_slope={calib.get('reference_slope', 0)}, "
                  f"probes={calib.get('n_probe_success', 0)}/{calib.get('n_calibration_images', 0)}")
        file_list = ordered_file_list

        from rock_core_layer_detection import merge_batch_results

        run_analysis = bool(getattr(self, 'batch_run_sensitivity', None)
                            and self.batch_run_sensitivity.get())
        ran_analysis = False
        merged_dirs = []
        per_group_summary = []
        merged_display_dir = None

        def _finalize_group(prev_sub, prev_group_results):
            """Merge + optional sensitivity for one finished sub-folder."""
            nonlocal ran_analysis, merged_display_dir
            if not prev_group_results:
                return
            group_label = prev_sub or '(root)'
            group_output_dir = (os.path.join(output_dir, prev_sub)
                                if prev_sub else output_dir)
            os.makedirs(group_output_dir, exist_ok=True)

            self.status_var.set(f"Merging results for {group_label}...")
            self.root.update()
            dirs = [r['output_dir'] for r in prev_group_results]
            print(f"  - merging group {group_label}: {len(dirs)} image(s) -> {group_output_dir}")
            try:
                merge_batch_results(group_output_dir, dirs)
                merged_dirs.append(group_output_dir)
                if merged_display_dir is None:
                    merged_display_dir = group_output_dir
            except Exception as merge_err:
                print(f"Error while merging group '{group_label}': {merge_err}")
                traceback.print_exc()

            if run_analysis:
                jobs = []
                for r in prev_group_results:
                    image_name_only = r.get('image_name') or os.path.basename(r['output_dir'])
                    image_path_full = (image_name_only if os.path.isabs(image_name_only)
                                       else os.path.join(base_folder, image_name_only))
                    if not os.path.isfile(image_path_full):
                        print(f"    warning: cannot find source image for analysis: {image_path_full}")
                        continue
                    label = os.path.splitext(os.path.basename(image_name_only))[0]
                    jobs.append((label, image_path_full))
                if jobs:
                    self.status_var.set(f"Running sensitivity + ablation for {group_label}...")
                    self.root.update()
                    self._run_batch_analysis(group_output_dir, jobs, detector_params)
                    ran_analysis = True

            per_group_summary.append({
                "subfolder": group_label,
                "succeeded": len(prev_group_results),
                "output_dir": group_output_dir,
            })

        # Accumulators for per-image results
        results = []
        successful_count = 0
        failed_count = 0
        current_group_sub = None
        current_group_results = []
        
        for i, file_item in enumerate(file_list):
            image_name, image_path, subfolder_rel, image_stem = self._resolve_batch_file_item(
                file_item, base_folder
            )
            image_basename_only = os.path.basename(image_name)

            # Detect sub-folder transitions: finish the previous group before continuing.
            if current_group_sub is None:
                current_group_sub = subfolder_rel
            elif subfolder_rel != current_group_sub:
                _finalize_group(current_group_sub, current_group_results)
                current_group_results = []
                current_group_sub = subfolder_rel

            print(f"\n--- Processing file {i+1}/{total_files}: {image_name} ---")
            print(f"Full path: {image_path}")
            if subfolder_rel:
                print(f"Sub-folder: {subfolder_rel}")
            
            # Check that the file exists
            if not os.path.exists(image_path):
                print(f"Error: file does not exist - {image_path}")
                failed_count += 1
                continue
                
            try:
                # Update progress
                progress = (i / total_files) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Processing {image_name} ({i+1}/{total_files})")

                self.root.update()
                print(f"Initializing detector...")
                detector = RockCoreLayerDetector(image_path)
                print(f"Detector initialized; image size: {detector.width}x{detector.height}")
                
                # Configure the output directory (group by sub-folder)
                if subfolder_rel:
                    detector.output_dir = os.path.join(output_dir, subfolder_rel, image_stem)
                else:
                    detector.output_dir = os.path.join(output_dir, image_stem)
                os.makedirs(detector.output_dir, exist_ok=True)
                print(f"Output directory set to: {detector.output_dir}")

                # Batch lamina orientation policy for this sub-folder
                self._apply_batch_lamina_to_detector(
                    detector, group_lamina_kwargs.get(subfolder_rel, {})
                )
                
                # Sync the scale
                if self.pixel_per_mm is not None:
                    detector.pixel_per_mm = self.pixel_per_mm
                
                # Preprocess (using the same UI parameters as single-image mode)
                blur_size = self.blur_size.get()
                clahe_clip = self.clahe_clip.get()
                clahe_grid = (self.clahe_grid_x.get(), self.clahe_grid_y.get())
                detector.preprocess_image(
                    blur_size=blur_size,
                    clahe_clip=clahe_clip,
                    clahe_grid=clahe_grid,
                    brightness=self.brightness.get(),
                    contrast=self.contrast.get(),
                    gamma=self.gamma.get()
                )
                
                print(f"Starting lamina detection...")
                # Process according to batch scan-line settings
                if batch_scan_lines is None:
                    print(f"Using default scan lines; count={scan_line_count}")
                    detect_success = detector.detect_layers(
                        threshold_method=threshold_method,
                        min_layer_width=min_layer_width,
                        scan_lines=None,
                        scan_line_count=scan_line_count,
                        min_validation_lines=min_validation_lines,
                        align_core=align_core,
                        alignment_angle=alignment_angle
                    )
                else:
                    if isinstance(batch_scan_lines, int):
                        print(f"Using auto scan lines; count={batch_scan_lines}")
                        detect_success = detector.detect_layers(
                            threshold_method=threshold_method,
                            min_layer_width=min_layer_width,
                            scan_lines=None,
                            scan_line_count=batch_scan_lines,
                            min_validation_lines=min_validation_lines,
                            align_core=align_core,
                            alignment_angle=alignment_angle
                        )
                    else:
                        print(f"Using manual scan lines; positions={batch_scan_lines}")
                        
                        # Inspect the detector's working image size and adjust scan-line coordinates
                        actual_height = detector.height
                        
                        # Use the first image to recover the original size for coordinate adjustment
                        try:
                            from PIL import Image, ImageOps
                            # Resolve the first image's path
                            first_file_item = file_list[0]
                            if isinstance(first_file_item, tuple) and len(first_file_item) == 2:
                                first_image_name, first_image_path = first_file_item
                            else:
                                if os.path.isabs(first_file_item):
                                    first_image_path = first_file_item
                                else:
                                    first_image_path = os.path.join(base_folder, first_file_item)
                            
                            temp_img = Image.open(first_image_path)
                            temp_img = ImageOps.exif_transpose(temp_img)
                            original_height = temp_img.height
                            temp_img = None  # release memory
                            
                            # Compute the scale ratio
                            scale_ratio = actual_height / original_height
                            
                            print(f"Original image height: {original_height}")
                            print(f"Detector image height: {actual_height}")
                            print(f"Scale ratio: {scale_ratio:.4f}")
                            print(f"Original scan lines: {batch_scan_lines}")
                            
                            # Adjust scan-line coordinates
                            adjusted_scan_lines = [int(y * scale_ratio) for y in batch_scan_lines]
                            print(f"Adjusted scan lines: {adjusted_scan_lines}")
                            
                            detect_success = detector.detect_layers(
                    threshold_method=threshold_method,
                    min_layer_width=min_layer_width,
                                scan_lines=adjusted_scan_lines,
                                scan_line_count=len(adjusted_scan_lines),
                                min_validation_lines=min_validation_lines,
                                align_core=align_core,
                                alignment_angle=alignment_angle
                            )
                        except Exception as coord_adjust_error:
                            print(f"Coordinate adjustment failed, using original coordinates: {str(coord_adjust_error)}")
                            detect_success = detector.detect_layers(
                                threshold_method=threshold_method,
                                min_layer_width=min_layer_width,
                                scan_lines=batch_scan_lines,
                                scan_line_count=len(batch_scan_lines),
                    min_validation_lines=min_validation_lines,
                    align_core=align_core,
                    alignment_angle=alignment_angle
                )
                
                print(f"Lamina detection complete; result: {detect_success}")
                    
                if detect_success:
                    print(f"Computing statistics...")
                # Compute statistics
                    stats, detailed_df, position_df = detector.calculate_statistics()
                    print(f"Statistics computed")
                    print(f"Unique laminae (cluster): {stats.get('unique_laminae_cluster', stats.get('total_lamina_count', 0))}")
                    print(f"Candidate points (across lines): {stats.get('candidate_points_total', '?')}")
                    print(f"Scan-line count: {stats.get('n_scan_lines', 0)}")

                    # Depth-range interpolation (mirrors single-image flow)
                    if (hasattr(self, 'enable_depth_range') and self.enable_depth_range.get()
                            and hasattr(self, 'start_depth') and hasattr(self, 'end_depth')
                            and self.start_depth.get() != self.end_depth.get()):
                        try:
                            from rock_core_analyzer.gui.workers import _apply_depth_interpolation_to_detector
                            _apply_depth_interpolation_to_detector(
                                detector,
                                float(self.start_depth.get()),
                                float(self.end_depth.get()),
                            )
                            print("Depth interpolation applied")
                        except Exception as depth_err:
                            print(f"Depth interpolation failed: {depth_err}")

                    print(f"Exporting results...")
                    # Export results
                    result_path = detector.export_results(detector.output_dir)
                    print(f"Results exported: {result_path}")
                    
                    record = {
                        'image_name': image_name,
                        'success': True,
                        'output_dir': detector.output_dir,
                        'subfolder': subfolder_rel,
                        'stats': stats
                    }
                    results.append(record)
                    current_group_results.append(record)
                    successful_count += 1
                    print(f"File processed successfully: {image_name}")
                else:
                    print(f"No valid laminae detected in {image_name}")
                    results.append({
                        'image_name': image_name,
                        'success': False,
                        'subfolder': subfolder_rel,
                        'error': 'no valid laminae detected'
                    })
                    failed_count += 1
                
            except Exception as e:
                print(f"Error while processing {image_name}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                import traceback
                print(f"Stack trace:")
                traceback.print_exc()
                
                results.append({
                    'image_name': image_name,
                    'success': False,
                    'subfolder': subfolder_rel,
                    'error': str(e)
                })
                failed_count += 1
        
        # Close out the last sub-folder (the per-image loop only triggers
        # finalisation on transitions; the final group has no transition).
        if current_group_sub is not None:
            _finalize_group(current_group_sub, current_group_results)
            current_group_results = []

        print(f"\n=== Batch processing complete ===")
        print(f"Total files: {total_files}")
        print(f"Succeeded: {successful_count}")
        print(f"Failed: {failed_count}")

        # Update progress
        self.progress_var.set(100)
        self.status_var.set(f"Batch complete! Processed {successful_count}/{total_files} file(s) successfully")

        if successful_count > 0:
            # Prefer the root group result; otherwise show the first finished group.
            display_dir = (output_dir if output_dir in merged_dirs
                           else (merged_display_dir or output_dir))
            try:
                self.load_batch_results(display_dir)
            except Exception as load_err:
                print(f"Error while loading merged batch results: {load_err}")
                traceback.print_exc()
            if len(merged_dirs) > 1:
                print(f"Generated {len(merged_dirs)} merged group(s); "
                      f"currently showing: {display_dir}")
                self.status_var.set(
                    f"Batch complete: {successful_count}/{total_files} succeeded, "
                    f"merged {len(merged_dirs)} sub-directories; "
                    f"currently showing: {os.path.basename(display_dir)}"
                )

            if len(per_group_summary) > 1:
                summary_lines = [
                    f"  - {g['subfolder']}: {g['succeeded']} image(s)"
                    for g in per_group_summary
                ]
                print("Per sub-folder summary:")
                for line in summary_lines:
                    print(line)
        else:
            print("No successful results; skipping merge")

        if ran_analysis:
            print("Per sub-folder sensitivity + ablation results saved under each group's 'batch_analysis/'.")

        print(f"=== Batch processing finished ===\n")
    def load_batch_results(self, output_dir):
        """Load batch-processing results."""
        # Clear existing widgets to free memory
        for widget in self.batch_frame_combined.winfo_children():
            widget.destroy()
        for widget in self.batch_frame_heatmap.winfo_children():
            widget.destroy()
        for widget in self.batch_frame_curve.winfo_children():
            widget.destroy()
            
        # Trigger garbage collection
        import gc
        gc.collect()
        
        try:
            self.status_var.set("Loading batch-processing results...")
            self.root.update()
            
            # Async image loader
            def load_image_async(image_path, target_frame, resize_to, add_explanation=None):
                if not os.path.exists(image_path):
                    print(f"Image file does not exist: {image_path}")
                    return
                
                try:
                    # Limit image size to save memory
                    img = Image.open(image_path)
                    img.thumbnail(resize_to, Image.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                
                    # Display the image
                    label = ttk.Label(target_frame, image=img_tk)
                    label.image = img_tk  # keep a reference
                    label.pack(fill=tk.BOTH, expand=True)
                
                    # Add caption (if provided)
                    if add_explanation:
                        explanation = ttk.Label(target_frame, text=add_explanation)
                        explanation.pack(pady=(0, 5))
                    
                    # Release the source image memory
                    img = None
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
                    error_label = ttk.Label(target_frame, text=f"Cannot load image: {os.path.basename(image_path)}")
                    error_label.pack(pady=20)
            
            # Stagger image loading
            self.root.after(100, lambda: load_image_async(
                os.path.join(output_dir, "combined_layer_intensity.png"),
                self.batch_frame_combined,
                (800, 400)
            ))
            
            self.root.after(300, lambda: load_image_async(
                os.path.join(output_dir, "layer_intensity_heatmap.png"),
                self.batch_frame_heatmap,
                (800, 300)
            ))
            
            self.root.after(500, lambda: load_image_async(
                os.path.join(output_dir, "layer_intensity_curve.png"),
                self.batch_frame_curve,
                (800, 300),
                "The lateral strength curve shows the average lamina strength across all cores at each lateral position; yellow dots mark the major laminae."
            ))
            
            # Helper to add result buttons
            def add_buttons():
                # Button to view detailed results
                btn_frame = ttk.Frame(self.batch_frame_curve)
                btn_frame.pack(pady=10)
            
                ttk.Button(btn_frame, text="Open result folder",
                      command=lambda: os.startfile(os.path.abspath(output_dir)) if os.name == 'nt' 
                      else os.system(f"xdg-open {os.path.abspath(output_dir)}")).pack(side=tk.LEFT, padx=5)
            
                ttk.Button(btn_frame, text="View detailed statistics",
                      command=lambda: self.show_merged_statistics(output_dir)).pack(side=tk.LEFT, padx=5)
                
                ttk.Button(btn_frame, text="View summary statistics",
                      command=lambda: self.show_batch_summary(output_dir)).pack(side=tk.LEFT, padx=5)
                
                ttk.Button(btn_frame, text="View position info",
                      command=lambda: self.show_merged_position_info(output_dir)).pack(side=tk.LEFT, padx=5)
                
                # Button to switch to detailed view
                ttk.Button(btn_frame, text="Switch to detailed view",
                          command=lambda: self._switch_to_detail_view(output_dir)).pack(side=tk.LEFT, padx=5)
                
            self.status_var.set("Batch results loaded")
            
            # Finally add the buttons
            self.root.after(700, add_buttons)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error while loading batch results:\n{str(e)}")
            print(f"Error while loading batch results: {str(e)}")
            import traceback
            traceback.print_exc()
    def _switch_to_detail_view(self, batch_output_dir):
        """Switch to detailed view mode and update the batch path."""
        try:
            # Check whether a batch_results sub-directory exists (where per-image outputs live)
            batch_results_dir = os.path.join(batch_output_dir, "batch_results")
            if os.path.exists(batch_results_dir):
                actual_output_dir = batch_results_dir
            else:
                actual_output_dir = batch_output_dir
            
            # Set the output path to the actual batch-results directory
            self.save_path.set(actual_output_dir)
            
            print(f"Switching to detailed view; path: {actual_output_dir}")
            
            # Tell the user they can now inspect individual images
            messagebox.showinfo(
                "Switch complete",
                "Switched to detailed view!\n\n" +
                "Next steps:\n" +
                "1. Use the \"Original image\" tab to pick a single image\n" +
                "2. Use the \"Detection results\" tab to see the detection figures\n" +
                "3. Use the \"Statistics\" tab to see detailed statistics\n\n" +
                "Each tab's drop-down lets you switch between images."
            )
            
            # Switch to the original-image tab
            if hasattr(self, 'tab_control'):
                self.tab_control.select(0)  # select the first tab
            
            self.status_var.set("Switched to detailed view; inspect individual images")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error while switching to detailed view:\n{str(e)}")
            print(f"Error while switching to detailed view: {str(e)}")
    def show_merged_statistics(self, output_dir):
        """Show the merged statistics table."""
        try:
            # Load the merged statistics data
            stats_path = os.path.join(output_dir, "all_layers.xlsx")
            csv_path = os.path.join(output_dir, "all_layers.csv")
            
            # Try Excel first, fall back to CSV
            if os.path.exists(stats_path):
                # Read in chunks for performance
                self.status_var.set("Loading data, please wait...")
                self.root.update()
                
                # Read only column names and row count first
                xl = pd.ExcelFile(stats_path)
                stats_df_sample = pd.read_excel(xl, nrows=5)
                columns = stats_df_sample.columns.tolist()
                
                # Open a new window to display the table
                stats_window = tk.Toplevel(self.root)
                stats_window.title("Merged statistics table")
                stats_window.geometry("900x700")
                
                # Build the title and info bar
                info_frame = ttk.Frame(stats_window)
                info_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
                
                ttk.Label(info_frame, text="Merged statistics", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
                self.stats_status_var = tk.StringVar(stats_window, value="Loading data...")
                ttk.Label(info_frame, textvariable=self.stats_status_var).pack(side=tk.RIGHT)
                
                # Build the control panel (filter, paging, etc.)
                control_frame = ttk.Frame(stats_window)
                control_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
                
                # Build the table
                table_frame = ttk.Frame(stats_window)
                table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Use a Treeview as the table widget
                tree = ttk.Treeview(table_frame, columns=columns, show='headings')
                
                # Configure column headings and widths
                for col in columns:
                    tree.heading(col, text=col)
                    # Pick column widths based on the column name
                    if "name" in col or "filename" in col:
                        tree.column(col, width=200)
                    elif "position" in col:
                        tree.column(col, width=120)
                    else:
                        tree.column(col, width=100)
                
                # Add scrollbars
                vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
                hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
                tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
                
                # Lay out the table and scrollbars
                tree.grid(column=0, row=0, sticky='nsew')
                vsb.grid(column=1, row=0, sticky='ns')
                hsb.grid(column=0, row=1, sticky='ew')
                
                table_frame.grid_columnconfigure(0, weight=1)
                table_frame.grid_rowconfigure(0, weight=1)
                
                # Paging state
                page_size = 500  # rows per page
                current_page = tk.IntVar(stats_window, value=1)
                total_pages = tk.IntVar(stats_window, value=1)
                
                # Page selector
                page_frame = ttk.Frame(control_frame)
                page_frame.pack(side=tk.RIGHT, padx=10)
                
                ttk.Label(page_frame, text="Page:").pack(side=tk.LEFT)
                page_spin = ttk.Spinbox(
                    page_frame, 
                    from_=1, 
                    to=1, 
                    width=5, 
                    textvariable=current_page, 
                    state="readonly"
                )
                page_spin.pack(side=tk.LEFT, padx=5)
                ttk.Label(page_frame, text="/").pack(side=tk.LEFT)
                total_pages_label = ttk.Label(page_frame, textvariable=total_pages)
                total_pages_label.pack(side=tk.LEFT, padx=5)
                
                # Filter controls
                filter_frame = ttk.Frame(control_frame)
                filter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 5))
                filter_var = tk.StringVar(stats_window)
                filter_entry = ttk.Entry(filter_frame, textvariable=filter_var, width=25)
                filter_entry.pack(side=tk.LEFT, padx=5)
                
                # Column-selection drop-down
                filter_column_var = tk.StringVar(stats_window)
                filter_column_combo = ttk.Combobox(
                    filter_frame, 
                    textvariable=filter_column_var,
                    values=["All columns"] + columns,
                    width=15,
                    state="readonly"
                )
                filter_column_combo.current(0)
                filter_column_combo.pack(side=tk.LEFT, padx=5)
                
                # Add export buttons
                export_frame = ttk.Frame(stats_window)
                export_frame.pack(fill=tk.X, padx=10, pady=10)
                
                # Currently displayed DataFrame
                filtered_df = pd.DataFrame()
                
                # Helper to load and display data
                def load_page_data():
                    nonlocal filtered_df
                    page = current_page.get()
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    
                    # Clear existing rows
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    # Fetch the current page
                    page_data = filtered_df.iloc[start_idx:end_idx]
                    
                    # Insert data rows
                    for _, row in page_data.iterrows():
                        tree.insert('', tk.END, values=list(row))
                    
                    self.stats_status_var.set(f"Showing {len(page_data)} row(s) (out of {len(filtered_df)})")
                    stats_window.update()
                
                # Helper to apply the filter
                def apply_filter(*args):
                    nonlocal filtered_df
                    filter_text = filter_var.get().lower()
                    filter_column = filter_column_var.get()
                    
                    # Load the raw data
                    stats_df = pd.read_excel(xl)
                    
                    # Apply the filter
                    if filter_text:
                        if filter_column == "All columns":
                            # Search across all columns
                            mask = False
                            for col in stats_df.columns:
                                # Convert each column to string before matching
                                mask = mask | stats_df[col].astype(str).str.lower().str.contains(filter_text, na=False)
                            filtered_df = stats_df[mask]
                        else:
                            # Search in a specific column
                            filtered_df = stats_df[stats_df[filter_column].astype(str).str.lower().str.contains(filter_text, na=False)]
                    else:
                        filtered_df = stats_df
                    
                    # Update the page total
                    max_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)
                    total_pages.set(max_pages)
                    page_spin.config(to=max_pages)
                    
                    # Reset to the first page
                    current_page.set(1)
                    
                    # Load the current page
                    load_page_data()
                
                # Reload when the page changes
                def on_page_change(*args):
                    load_page_data()
                
                # Bind events
                current_page.trace("w", on_page_change)
                filter_var.trace("w", apply_filter)
                filter_column_var.trace("w", apply_filter)
                
                # Add export buttons
                ttk.Button(
                    export_frame, 
                    text="Export to Excel", 
                    command=lambda: self.export_to_excel(filtered_df, output_dir)
                ).pack(side=tk.LEFT, padx=5)
                
                ttk.Button(
                    export_frame, 
                    text="Close", 
                    command=stats_window.destroy
                ).pack(side=tk.RIGHT, padx=5)
                
                # Initial load (defer to keep the UI responsive)
                stats_window.after(100, apply_filter)
                
            elif os.path.exists(csv_path):
                messagebox.showinfo("Info", "Found CSV statistics; converting to Excel")
                # Read CSV and convert to Excel
                try:
                    stats_df = pd.read_csv(csv_path)
                    # Map English column names to readable headers
                    column_mapping = {
                        "scan_line": "scan_line",
                        "position_x": "position_x_px",
                        "position_y": "position_y_px",
                        "spacing_to_next": "spacing_to_next_px",
                        "layer_index": "lamina_index",
                        "strength": "strength",
                        "filename": "filename",
                        "image_index": "image_index",
                        "cumulative_offset": "cumulative_offset",
                        "adjusted_position": "adjusted_position_px"
                    }
                    # Apply the column mapping
                    renamed_columns = {}
                    for col in stats_df.columns:
                        if col in column_mapping:
                            renamed_columns[col] = column_mapping[col]
                        else:
                            renamed_columns[col] = col
                    
                    stats_df.rename(columns=renamed_columns, inplace=True)
                    stats_df.to_excel(stats_path, index=False)
                    # Recurse with the newly created Excel file
                    self.show_merged_statistics(output_dir)
                except Exception as e:
                    messagebox.showerror("Error", f"Error while converting the CSV file:\n{str(e)}")
            else:
                messagebox.showinfo("Info", f"Merged statistics not found: {stats_path} or {csv_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error while showing merged statistics:\n{str(e)}")
            print(f"Error while showing merged statistics: {str(e)}")
            import traceback
            traceback.print_exc()
    def show_batch_summary(self, output_dir):
        """Show the batch-processing summary statistics."""
        try:
            # Locate the summary file
            summary_path = os.path.join(output_dir, "batch_summary.xlsx")
            image_stats_path = os.path.join(output_dir, "images_statistics.xlsx")
            report_path = os.path.join(output_dir, "batch_processing_report.txt")
            
            if not os.path.exists(summary_path):
                messagebox.showinfo("Info", f"Batch summary file not found: {summary_path}")
                return
            
            # Create the summary window
            summary_window = tk.Toplevel(self.root)
            summary_window.title("Batch summary statistics")
            summary_window.geometry("800x600")
            
            # Create the notebook
            notebook = ttk.Notebook(summary_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 1. Summary statistics tab
            summary_frame = ttk.Frame(notebook)
            notebook.add(summary_frame, text="Summary statistics")
            
            # Read the summary data
            summary_df = pd.read_excel(summary_path)
            
            # Build the summary display area
            summary_text = tk.Text(summary_frame, wrap=tk.WORD, font=("Segoe UI", 10))
            summary_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=summary_text.yview)
            summary_text.configure(yscrollcommand=summary_scroll.set)
            
            summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Show summary statistics
            summary_text.insert(tk.END, "Batch summary statistics\n")
            summary_text.insert(tk.END, "=" * 50 + "\n\n")
            
            for column in summary_df.columns:
                value = summary_df[column].iloc[0]
                summary_text.insert(tk.END, f"{column}: {value}\n")
            
            summary_text.config(state=tk.DISABLED)
            
            # 2. Per-image statistics tab
            if os.path.exists(image_stats_path):
                image_stats_frame = ttk.Frame(notebook)
                notebook.add(image_stats_frame, text="Per-image statistics")
                
                # Read per-image statistics
                image_stats_df = pd.read_excel(image_stats_path)
                
                # Build the table
                tree = ttk.Treeview(image_stats_frame, columns=list(image_stats_df.columns), show='headings')
                
                # Configure column headings
                for col in image_stats_df.columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=120)
                
                # Insert data
                for _, row in image_stats_df.iterrows():
                    tree.insert('', tk.END, values=list(row))
                
                # Add scrollbars
                tree_scroll = ttk.Scrollbar(image_stats_frame, orient="vertical", command=tree.yview)
                tree.configure(yscrollcommand=tree_scroll.set)
                
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 3. Processing-report tab
            if os.path.exists(report_path):
                report_frame = ttk.Frame(notebook)
                notebook.add(report_frame, text="Processing report")
                
                # Build the report display area
                report_text = tk.Text(report_frame, wrap=tk.WORD, font=("Courier New", 9))
                report_scroll = ttk.Scrollbar(report_frame, orient="vertical", command=report_text.yview)
                report_text.configure(yscrollcommand=report_scroll.set)
                
                report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                report_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Read and display the report
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    report_text.insert(tk.END, report_content)
                except Exception as e:
                    report_text.insert(tk.END, f"Error while reading the report file: {str(e)}")
                
                report_text.config(state=tk.DISABLED)
            
            # Add the bottom button bar
            button_frame = ttk.Frame(summary_window)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            ttk.Button(button_frame, text="Export summary", 
                      command=lambda: self.export_to_excel(summary_df, output_dir)).pack(side=tk.LEFT, padx=5)
            
            if os.path.exists(image_stats_path):
                ttk.Button(button_frame, text="Export per-image stats", 
                          command=lambda: self.export_to_excel(pd.read_excel(image_stats_path), output_dir)).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(button_frame, text="Open result folder",
                      command=lambda: os.startfile(os.path.abspath(output_dir)) if os.name == 'nt' 
                      else os.system(f"xdg-open {os.path.abspath(output_dir)}")).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(button_frame, text="Close", 
                      command=summary_window.destroy).pack(side=tk.RIGHT, padx=5)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error while showing batch summary:\n{str(e)}")
            print(f"Error while showing batch summary: {str(e)}")
            import traceback
            traceback.print_exc()
    def show_merged_position_info(self, output_dir):
        """Show the merged batch-processing position info."""
        try:
            # Locate the merged position-info files
            merged_position_path = os.path.join(output_dir, "merged_position_info.xlsx")
            continuous_position_path = os.path.join(output_dir, "continuous_position_statistics.xlsx")
            
            if not os.path.exists(merged_position_path):
                messagebox.showinfo("Info", f"Merged position-info file not found: {merged_position_path}")
                return
            
            print(f"Showing merged position info: {merged_position_path}")
            
            # Open a new window
            position_window = tk.Toplevel()
            position_window.title("Batch position info")
            position_window.geometry("1200x800")
            
            # Center the window
            self.center_window(position_window, 1200, 800)
            
            # Use a Notebook to host the different tables
            notebook = ttk.Notebook(position_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Read the merged position info
            merged_df = pd.read_excel(merged_position_path)
            print(f"Merged position info row count: {len(merged_df)}")
            
            # Tab 1: detailed position info
            detail_frame = ttk.Frame(notebook)
            notebook.add(detail_frame, text="Detailed positions")
            
            # Build the detail table
            self._create_optimized_table(
                notebook, 
                merged_df, 
                "Detailed positions", 
                merged_position_path
            )
            
            # Tab 2: continuous-depth statistics (if present)
            if os.path.exists(continuous_position_path):
                try:
                    continuous_df = pd.read_excel(continuous_position_path)
                    print(f"Continuous-depth statistics row count: {len(continuous_df)}")
                    
                    continuous_frame = ttk.Frame(notebook)
                    notebook.add(continuous_frame, text="Continuous-depth stats")
                    
                    # Build the continuous-depth stats table
                    self._create_optimized_table(
                        notebook,
                        continuous_df,
                        "Continuous-depth stats",
                        continuous_position_path
                    )
                except Exception as e:
                    print(f"Error while reading continuous-depth statistics: {str(e)}")
            
            # Tab 3: position info grouped by image
            if 'image_name' in merged_df.columns:
                grouped_frame = ttk.Frame(notebook)
                notebook.add(grouped_frame, text="Grouped by image")
                
                # Build the grouped view
                self._create_grouped_position_view(grouped_frame, merged_df)
            
            # Add the action-buttons frame
            button_frame = ttk.Frame(position_window)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(button_frame, text="Export to Excel", 
                      command=lambda: self.export_to_excel(merged_df, output_dir)).pack(side=tk.LEFT, padx=5)
            
            if os.path.exists(continuous_position_path):
                ttk.Button(button_frame, text="Export continuous stats", 
                          command=lambda: self._export_continuous_stats(continuous_position_path, output_dir)).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(button_frame, text="Close", 
                      command=position_window.destroy).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error while showing merged position info:\\n{str(e)}")
            print(f"Error while showing merged position info: {str(e)}")
            import traceback
            traceback.print_exc()
    def _create_grouped_position_view(self, parent_frame, merged_df):
        """Build the view that groups position info by image."""
        try:
            # Build the main frame
            main_frame = ttk.Frame(parent_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Build the image-selection frame
            select_frame = ttk.Frame(main_frame)
            select_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(select_frame, text="Select image:").pack(side=tk.LEFT, padx=(0, 5))
            
            # Collect all image names
            image_names = sorted(merged_df['image_name'].unique())
            image_var = tk.StringVar()
            image_combo = ttk.Combobox(select_frame, textvariable=image_var, 
                                      values=image_names, state="readonly", width=30)
            image_combo.pack(side=tk.LEFT, padx=(0, 5))
            
            if image_names:
                image_combo.set(image_names[0])
            
            # Build the table frame
            table_frame = ttk.Frame(main_frame)
            table_frame.pack(fill=tk.BOTH, expand=True)
            
            # Build the table
            columns = list(merged_df.columns)
            tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
            
            # Configure columns
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=120)
            
            # Add scrollbars
            v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            def update_table(*args):
                """Update the table to show data for the selected image."""
                selected_image = image_var.get()
                if selected_image:
                    # Clear existing rows
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    # Filter the data
                    filtered_df = merged_df[merged_df['image_name'] == selected_image]
                    
                    # Insert into the table
                    for index, row in filtered_df.iterrows():
                        values = [str(row[col]) if pd.notna(row[col]) else "" for col in columns]
                        tree.insert("", tk.END, values=values)
                    
                    print(f"Showing position data for image {selected_image}; {len(filtered_df)} row(s)")
            
            # Bind the selection event
            image_combo.bind('<<ComboboxSelected>>', update_table)
            
            # Initialise the display
            update_table()
            
        except Exception as e:
            print(f"Error while building the grouped position view: {str(e)}")
            import traceback
            traceback.print_exc()
    def export_batch_list(self):
        """Export the current batch file list to a text file."""
        if not self.batch_image_files:
            messagebox.showinfo("Info", "Nothing to export; load a file list first")
            return
            
        # Choose save path
        file_path = filedialog.asksaveasfilename(
            title="Save file list",
            filetypes=[("Text files", "*.txt")],
            defaultextension=".txt"
        )
        
        if not file_path:
            return  # user cancelled
            
        try:
            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write the header
                f.write("# Rock Core Lamina Identification System -- batch file list\n")
                f.write("# One file path per line; lines starting with # are comments\n")
                f.write("# Generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("# Base directory: " + self.image_path.get() + "\n\n")
                
                # Write the file list
                for img_file in self.batch_image_files:
                    f.write(img_file + "\n")
                    
            messagebox.showinfo("Success", f"Exported {len(self.batch_image_files)} file path(s) to:\n{file_path}")
            
        except Exception as e:
            print(f"Error while exporting the file list: {str(e)}")
            messagebox.showerror("Error", f"Error while exporting the file list: {str(e)}")
    def batch_process_folder(self, folder_path):
        """Batch-process every image in a folder."""
        if not os.path.isdir(folder_path):
            messagebox.showerror("Error", f"The path is not a valid folder: {folder_path}")
            return
            
        try:
            # Read all required parameters
            output_dir = self.save_path.get()
            if not output_dir:
                output_dir = "output"
                self.save_path.set(output_dir)
                
            image_ext = self.image_ext.get()
            include_sub = bool(getattr(self, 'include_subfolders', None) and self.include_subfolders.get())
            
            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Enumerate images that match the extension (optionally recursive)
            try:
                if include_sub:
                    image_files = []
                    for root_dir, _dirs, files in os.walk(folder_path):
                        for f in files:
                            if f.lower().endswith(image_ext.lower()):
                                rel_path = os.path.relpath(os.path.join(root_dir, f), folder_path)
                                image_files.append(rel_path)
                    image_files.sort(key=natural_sort_key)
                else:
                    image_files = sorted([f for f in os.listdir(folder_path)
                                         if f.lower().endswith(image_ext.lower())])
            except Exception as e:
                messagebox.showerror("Error", f"Cannot read folder contents: {str(e)}")
                return
            
            if not image_files:
                hint = " (sub-folder recursion enabled)" if include_sub else ""
                messagebox.showinfo("Info", f"No {image_ext} image files found in {folder_path}{hint}")
                return
                
            # Read scan-line settings
            batch_scan_lines = self.batch_scan_lines
            
            # If no scan lines were preset, let the user pick
            if batch_scan_lines is None and len(image_files) > 0:
                try:
                    first_image_path = os.path.join(folder_path, image_files[0])
                    print(f"Opening the first image for scan-line selection: {first_image_path}")
                    if os.path.isfile(first_image_path):
                        batch_scan_lines = self._select_batch_scan_lines(first_image_path)
                        self.batch_scan_lines = batch_scan_lines
                    else:
                        raise IOError(f"File does not exist: {first_image_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Cannot open the image for scan-line selection: {str(e)}")
                    traceback.print_exc()  # log detailed stack trace
                    return
                    
                if batch_scan_lines is None:  # user cancelled
                    return
            elif batch_scan_lines is not None:
                print(f"Using preset scan lines for batch processing")
            
            # Show a progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Batch progress")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)  # parent to the main window
            progress_window.grab_set()  # modal dialog
            
            # Progress label
            progress_label = ttk.Label(progress_window, text="Processing...")
            progress_label.pack(pady=10)
            
            # Progress bar
            progress_var = tk.DoubleVar(progress_window)
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            # Status label
            status_label = ttk.Label(progress_window, text="")
            status_label.pack(pady=5)
            
            # Cancel button
            cancel_button = ttk.Button(progress_window, text="Cancel", command=progress_window.destroy)
            cancel_button.pack(pady=10)
            
            # Refresh the UI
            self.root.update()
            
            # Read the processing parameters
            threshold_method = self.threshold_method.get()
            min_layer_width = self.min_layer_width.get()
            blur_size = self.blur_size.get()
            clahe_clip = self.clahe_clip.get()
            clahe_grid = (self.clahe_grid_x.get(), self.clahe_grid_y.get())
            scan_line_count = self.scan_line_count.get()
            min_validation_lines = self.min_validation_lines.get()
            align_core = self.align_core.get()
            alignment_angle = self.alignment_angle.get()
            
            # Update the label
            progress_label.config(text=f"Found {len(image_files)} image file(s); starting processing...")
            
            # Set up multi-process parallelism
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            # Pick worker count: default min(4, cpu//2) to avoid OOM on large images
            cpu_count = os.cpu_count() or 1
            max_workers = max(1, min(4, cpu_count // 2))
            print(f"Batch processing: CPU={cpu_count}, using {max_workers} parallel worker(s)")

            detector_params = self._build_batch_detector_params(batch_scan_lines)
            file_groups = self._group_batch_files_by_subfolder(image_files)
            # Process the root group (no sub-folder) first, then sub-folders in
            # natural order. This keeps logs predictable and matches the user
            # expectation of "finish one sub-folder before starting the next".
            group_order = sorted(
                file_groups.keys(),
                key=lambda s: (s != "", natural_sort_key(s) if s else [""]),
            )
            n_groups = len(group_order)

            from rock_core_layer_detection import merge_batch_results

            run_analysis = bool(getattr(self, 'batch_run_sensitivity', None)
                                and self.batch_run_sensitivity.get())
            ran_analysis = False
            cancelled = False
            all_results = []
            processed_count = 0
            total_files = len(image_files)
            merged_display_dir = None
            merged_dirs = []
            per_group_summary = []

            # ============================================================
            # Iterate sub-folders sequentially: each group is fully processed
            # (calibrate -> detect -> merge -> optional sensitivity/ablation)
            # before the next group starts.
            # ============================================================
            for group_idx, sub in enumerate(group_order, 1):
                if not progress_window.winfo_exists():
                    cancelled = True
                    break

                group_files = file_groups[sub]
                group_label = sub or '(root)'
                group_output_dir = os.path.join(output_dir, sub) if sub else output_dir
                os.makedirs(group_output_dir, exist_ok=True)

                # ----- Step 1: dip calibration for this sub-folder -----
                status_label.config(
                    text=f"[{group_idx}/{n_groups}] {group_label}: calibrating lamina orientation..."
                )
                progress_window.update()
                calib, lamina_kw = self._calibrate_batch_group(
                    folder_path, sub, group_files, detector_params,
                    group_output_dir=group_output_dir,
                )
                print(f"[Batch dip calib] group '{group_label}': "
                      f"reference_slope={calib.get('reference_slope', 0)}, "
                      f"probes={calib.get('n_probe_success', 0)}/{calib.get('n_calibration_images', 0)}")

                # ----- Step 2: build worker args for this group only -----
                task_args_list = []
                for image_file in group_files:
                    sub_rel = os.path.dirname(image_file).replace('\\', '/')
                    base_only = os.path.basename(image_file)
                    image_stem = os.path.splitext(base_only)[0]
                    if sub_rel:
                        image_output_dir = os.path.join(output_dir, sub_rel, image_stem)
                    else:
                        image_output_dir = os.path.join(output_dir, image_stem)
                    task_args = {
                        "image_file": image_file,
                        "subfolder": sub_rel,
                        "image_path": os.path.join(folder_path, image_file),
                        "output_dir": image_output_dir,
                        "pixel_per_mm": self.pixel_per_mm,
                        "blur_size": blur_size,
                        "clahe_clip": clahe_clip,
                        "clahe_grid": clahe_grid,
                        "brightness": self.brightness.get(),
                        "contrast": self.contrast.get(),
                        "gamma": self.gamma.get(),
                        "threshold_method": threshold_method,
                        "min_layer_width": min_layer_width,
                        "scan_line_count": scan_line_count,
                        "min_validation_lines": min_validation_lines,
                        "align_core": align_core,
                        "alignment_angle": alignment_angle,
                        "batch_scan_lines": batch_scan_lines,
                        "enable_depth_range": (self.enable_depth_range.get()
                                               if hasattr(self, 'enable_depth_range') else False),
                        "start_depth": (self.start_depth.get()
                                        if hasattr(self, 'start_depth') else None),
                        "end_depth": (self.end_depth.get()
                                      if hasattr(self, 'end_depth') else None),
                    }
                    task_args.update(lamina_kw)
                    task_args_list.append(task_args)

                # ----- Step 3: parallel detection (only this group's tasks) -----
                group_results = []
                group_processed = 0
                group_total = len(task_args_list)
                status_label.config(
                    text=f"[{group_idx}/{n_groups}] {group_label}: detecting {group_total} image(s)..."
                )
                progress_window.update()

                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        future_to_args = {
                            executor.submit(_batch_worker, args): args
                            for args in task_args_list
                        }
                        completed_in_group = 0
                        for future in as_completed(future_to_args):
                            if not progress_window.winfo_exists():
                                cancelled = True
                                executor.shutdown(wait=False, cancel_futures=True)
                                break

                            completed_in_group += 1
                            try:
                                success, image_file, image_output_dir, err_msg = future.result()
                            except Exception as e:
                                success = False
                                image_file = future_to_args[future].get("image_file", "?")
                                image_output_dir = future_to_args[future].get("output_dir", "")
                                err_msg = str(e)

                            global_done = processed_count + completed_in_group
                            progress_var.set((global_done / max(1, total_files)) * 100)
                            if success:
                                group_processed += 1
                                status_label.config(
                                    text=f"[{group_idx}/{n_groups}] {group_label}: "
                                         f"{completed_in_group}/{group_total} {image_file}"
                                )
                                group_results.append({
                                    "filename": image_file,
                                    "subfolder": future_to_args[future].get("subfolder", ""),
                                    "output_dir": image_output_dir,
                                    "index": len(all_results) + len(group_results),
                                })
                            else:
                                status_label.config(
                                    text=f"[{group_idx}/{n_groups}] {group_label}: "
                                         f"{image_file} failed: {err_msg[:80]}"
                                )
                                print(f"[Batch] {image_file} failed: {err_msg}")
                            progress_window.update()
                except Exception as pool_err:
                    print(f"Multi-process pool error in group '{group_label}'; "
                          f"falling back to single-thread: {pool_err}")
                    for args in task_args_list:
                        if not progress_window.winfo_exists():
                            cancelled = True
                            break
                        success, image_file, image_output_dir, err_msg = _batch_worker(args)
                        if success:
                            group_processed += 1
                            group_results.append({
                                "filename": image_file,
                                "subfolder": args.get("subfolder", ""),
                                "output_dir": image_output_dir,
                                "index": len(all_results) + len(group_results),
                            })
                        global_done = processed_count + len(group_results)
                        progress_var.set((global_done / max(1, total_files)) * 100)
                        progress_window.update()

                if cancelled:
                    break

                all_results.extend(group_results)
                processed_count += group_processed

                # ----- Step 4: merge this group's results -----
                if group_results:
                    status_label.config(
                        text=f"[{group_idx}/{n_groups}] {group_label}: merging results..."
                    )
                    progress_window.update()
                    dirs = [r["output_dir"] for r in group_results]
                    print(f"Merging group {group_label}: {len(dirs)} image(s) -> {group_output_dir}")
                    try:
                        merge_batch_results(group_output_dir, dirs)
                        merged_dirs.append(group_output_dir)
                        if merged_display_dir is None:
                            merged_display_dir = group_output_dir
                    except Exception as merge_err:
                        print(f"Error while merging group '{group_label}': {merge_err}")
                        traceback.print_exc()

                    # ----- Step 5: optional sensitivity / ablation for this group -----
                    if run_analysis:
                        jobs = []
                        for r in group_results:
                            rel_filename = r.get("filename")
                            if not rel_filename:
                                continue
                            image_path_full = (rel_filename if os.path.isabs(rel_filename)
                                               else os.path.join(folder_path, rel_filename))
                            if not os.path.isfile(image_path_full):
                                print(f"  warning: cannot find source image for analysis: {image_path_full}")
                                continue
                            jobs.append((os.path.splitext(os.path.basename(rel_filename))[0],
                                         image_path_full))
                        if jobs:
                            status_label.config(
                                text=f"[{group_idx}/{n_groups}] {group_label}: "
                                     f"running sensitivity + ablation..."
                            )
                            progress_window.update()
                            self._run_batch_analysis(
                                group_output_dir, jobs, detector_params,
                                progress_window=progress_window,
                                status_label=status_label,
                                progress_var=progress_var,
                            )
                            ran_analysis = True

                per_group_summary.append({
                    "subfolder": group_label,
                    "total": group_total,
                    "succeeded": group_processed,
                    "output_dir": group_output_dir,
                })
                print(f"Group '{group_label}' done: {group_processed}/{group_total} image(s) succeeded")

            if cancelled:
                self.status_var.set("Batch processing cancelled by user")
                progress_window.destroy()
                return

            # ============================================================
            # All sub-folders processed: load the first merged group into the UI
            # ============================================================
            progress_var.set(100)
            if processed_count > 0:
                self.last_batch_folder = folder_path
                self.last_batch_output_dir = output_dir
                self.last_batch_image_files = [r["filename"] for r in all_results]

                # Prefer the root-group result; otherwise show the first group
                display_dir = (output_dir if output_dir in merged_dirs
                               else (merged_display_dir or output_dir))
                self.load_batch_results(display_dir)
                if len(merged_dirs) > 1:
                    print(f"Generated {len(merged_dirs)} merged group(s); "
                          f"currently showing: {display_dir}")

            # Close the progress dialog
            progress_window.destroy()

            # Update status
            self.status_var.set(
                f"Batch processing complete: {processed_count}/{total_files} image(s) succeeded "
                f"across {n_groups} group(s)"
            )

            # Show the completion message
            msg = (f"Batch processing complete; {processed_count}/{total_files} "
                   f"image(s) succeeded across {n_groups} sub-folder(s).")
            if len(per_group_summary) > 1:
                detail_lines = [
                    f"  - {g['subfolder']}: {g['succeeded']}/{g['total']}"
                    for g in per_group_summary
                ]
                msg += "\n\n" + "\n".join(detail_lines)
            if ran_analysis:
                msg += "\n\nBatch sensitivity + ablation results saved under each group's 'batch_analysis/'."
            messagebox.showinfo("Done", msg)

            # Switch to the batch-results tab
            self.tab_control.select(self.tab_batch)

        except Exception as e:
            messagebox.showerror("Error", f"Error during batch processing:\n{str(e)}")
            self.status_var.set("Batch processing failed")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Batch lamina orientation (sub-folder dip calibration)
    # ------------------------------------------------------------------
    def _group_batch_files_by_subfolder(self, file_list):
        """Group relative image paths by their sub-folder key ('' = root)."""
        groups = {}
        for item in file_list:
            if isinstance(item, tuple) and len(item) == 2:
                rel = item[0]
            else:
                rel = item
            sub = os.path.dirname(str(rel)).replace("\\", "/")
            groups.setdefault(sub, []).append(item)
        for sub in groups:
            groups[sub].sort(key=lambda x: natural_sort_key(x[0] if isinstance(x, tuple) else x))
        return groups

    def _build_batch_detector_params(self, batch_scan_lines=None):
        """Shared detector-parameter dict for batch workers and dip calibration."""
        return self._collect_detector_params_for_analysis(batch_scan_lines=batch_scan_lines)

    def _resolve_batch_file_item(self, file_item, base_folder):
        """Return ``(relative_name, absolute_path, subfolder_rel, basename_stem)``."""
        if isinstance(file_item, tuple) and len(file_item) == 2:
            image_name, image_path = file_item
        elif os.path.isabs(str(file_item)):
            image_path = str(file_item)
            image_name = os.path.basename(image_path)
        else:
            image_name = str(file_item)
            image_path = os.path.join(base_folder, image_name)
        subfolder_rel = os.path.dirname(image_name).replace("\\", "/")
        image_basename_only = os.path.basename(image_name)
        image_stem = os.path.splitext(image_basename_only)[0]
        return image_name, image_path, subfolder_rel, image_stem

    def _calibrate_batch_group(self, base_folder, subfolder, file_items, detector_params,
                               group_output_dir=None):
        """Run sub-folder dip calibration; persist JSON when output dir is known."""
        from rock_core_analyzer.batch.dip_calibration import (
            calibrate_subfolder_dip,
            save_group_calibration,
            batch_lamina_kwargs_from_calibration,
        )

        paths = []
        for item in file_items:
            _name, path, _sub, _stem = self._resolve_batch_file_item(item, base_folder)
            if os.path.isfile(path):
                paths.append(path)

        probe_params = dict(detector_params)
        if group_output_dir:
            probe_params["_probe_dir"] = group_output_dir

        calib = calibrate_subfolder_dip(paths, probe_params)
        calib["subfolder"] = subfolder or "(root)"
        if group_output_dir:
            save_group_calibration(group_output_dir, calib)
        if calib.get("warnings"):
            for w in calib["warnings"]:
                print(f"[Batch dip calib {subfolder or 'root'}] {w}")
        return calib, batch_lamina_kwargs_from_calibration(calib)

    def _apply_batch_lamina_to_detector(self, detector, lamina_kwargs):
        """Attach batch lamina policy attributes to a live detector instance."""
        if not lamina_kwargs.get("batch_lamina_mode"):
            return
        detector.batch_lamina_mode = True
        detector.batch_group_slope_hint = lamina_kwargs.get("batch_group_slope_hint", 0.0)
        detector.batch_max_dip_after_align_deg = float(
            lamina_kwargs.get("batch_max_dip_after_align_deg", 7.0)
        )
        detector.batch_force_vertical_after_align = bool(
            lamina_kwargs.get("batch_force_vertical_after_align", True)
        )

    # ------------------------------------------------------------------
    # Batch analysis helpers (sensitivity + ablation after merge)
    # ------------------------------------------------------------------
    def _collect_detector_params_for_analysis(self, batch_scan_lines=None):
        """Snapshot the current UI parameters into the dict layout expected by
        ``rock_core_analyzer.batch.batch_sensitivity``.
        """
        if batch_scan_lines is None:
            batch_scan_lines = getattr(self, 'batch_scan_lines', None)
        return {
            "pixel_per_mm": self.pixel_per_mm,
            "blur_size": self.blur_size.get(),
            "clahe_clip": self.clahe_clip.get(),
            "clahe_grid": (self.clahe_grid_x.get(), self.clahe_grid_y.get()),
            "brightness": self.brightness.get(),
            "contrast": self.contrast.get(),
            "gamma": self.gamma.get(),
            "threshold_method": self.threshold_method.get(),
            "min_layer_width": self.min_layer_width.get(),
            "scan_line_count": self.scan_line_count.get(),
            "min_validation_lines": self.min_validation_lines.get(),
            "align_core": self.align_core.get(),
            "alignment_angle": self.alignment_angle.get(),
            "batch_scan_lines": batch_scan_lines,
        }

    def _run_batch_analysis(self, group_output_dir, jobs, detector_params,
                             progress_window=None, status_label=None,
                             progress_var=None):
        """Run sensitivity + ablation on a merged group, with optional UI feedback."""
        from rock_core_analyzer.batch import run_batch_sensitivity_and_ablation

        total_jobs = len(jobs)
        if total_jobs == 0:
            return None

        print(f"[Batch analysis] {os.path.basename(group_output_dir)}: {total_jobs} image(s)")

        def _on_progress(completed, total, image_name, status):
            label = f"Analysis {completed}/{total}: {image_name} [{status}]"
            if status_label is not None:
                try:
                    status_label.config(text=label)
                except Exception:
                    pass
            if progress_var is not None:
                try:
                    progress_var.set((completed / total) * 100 if total else 0)
                except Exception:
                    pass
            if progress_window is not None:
                try:
                    progress_window.update()
                except Exception:
                    pass

        try:
            return run_batch_sensitivity_and_ablation(
                output_dir=group_output_dir,
                image_jobs=jobs,
                detector_params=detector_params,
                tolerance_px=10,
                progress_callback=_on_progress,
            )
        except Exception as e:
            print(f"[Batch analysis] Failed for {group_output_dir}: {e}")
            traceback.print_exc()
            return None
