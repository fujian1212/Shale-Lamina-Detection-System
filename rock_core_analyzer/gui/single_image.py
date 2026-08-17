#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-image analysis."""

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

class SingleImageMixin:
    def analyze_image(self):
        """Run analysis on the selected image."""
        
        # Prevent duplicate clicks
        if self.is_processing:
            messagebox.showwarning("Info", "Processing is in progress, please wait...")
            return
            
        try:
            path = self.image_path.get()
            if not path:
                messagebox.showerror("Error", "Please select an image file or a folder containing images first")
                return
                
            # Verify the path exists
            if not os.path.exists(path):
                messagebox.showerror("Error", f"The path does not exist: {path}")
                return
                
            # Mark as processing and disable the button
            self.is_processing = True
            self.analyze_btn.configure(state='disabled', text="Processing...")
                
            # Update status
            self.status_var.set("Processing image, please wait...")
            self.root.update()
            
            # Check whether we are in batch mode
            if self.batch_mode.get() or os.path.isdir(path):
                # Verify the base directory
                base_folder = path if os.path.isdir(path) else os.path.dirname(path)
                if not os.path.isdir(base_folder):
                    messagebox.showerror("Error", f"Base folder not found: {base_folder}")
                    return
                    
                # If a folder was picked but no file list loaded, do it automatically
                if os.path.isdir(path) and not self.batch_image_files:
                    print(f"Auto-loading images from folder: {path}")
                    # Call the batch-process-folder routine directly
                    self.batch_process_folder(path)
                    return
                    
                # Check whether a file list was loaded (for file-list driven runs)
                if not self.batch_image_files:
                    response = messagebox.askyesno("Info", "No file list loaded; load one now?")
                    if response:
                        self.load_file_list()
                        if not self.batch_image_files:
                            return  # still empty after loading -- bail out
                    else:
                        return
                        
                # Check whether scan lines were selected
                if self.batch_scan_lines is None:
                    response = messagebox.askyesno("Info", "No batch scan lines selected; select now?")
                    if response:
                        self.select_batch_scan_lines_ui()
                        # User may cancel; scan lines are not required
                
                # Run batch processing driven by the file list
                print(f"Running batch processing via file list; base folder: {base_folder}")
                
                # Read the user-configured save path
                user_output_dir = self.save_path.get()
                if not user_output_dir:
                    # Fall back to a default path if not set
                    user_output_dir = os.path.join(base_folder, "batch_results")
                else:
                    # Use the configured path and create a batch_results sub-directory inside it
                    user_output_dir = os.path.join(user_output_dir, "batch_results")
                
                print(f"User-configured output path: {user_output_dir}")
                self.process_batch_from_list(self.batch_image_files, base_folder, user_output_dir)
            else:
                # Verify the path is a file
                if not os.path.isfile(path):
                    messagebox.showerror("Error", f"Single-image mode requires a file path: {path}")
                    return
                    
                # Run the single-image pipeline
                self.process_single_image(path)
                
                # Show the results
                self.show_original()
                self.show_results()
                self.show_statistics()
                
                # Update status
                self.status_var.set("Image processing complete")
                self.root.update()
            
        except Exception as e:
            # Show the error
            error_msg = f"Error while analyzing image: {str(e)}\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Processing failed")
            print(f"Error during analysis:\n{error_msg}")
        finally:
            # Restore the button state regardless of success/failure
            self.is_processing = False
            self.analyze_btn.configure(state='normal', text="Analyze image")
            self.root.update()
    def process_single_image(self, image_path):
        """Process a single image."""
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file does not exist: {image_path}")
            return False
        
        # Update status
        self.status_var.set("Starting image processing...")
        self.root.update()
        
        # Show the loading progress dialog
        if self._check_optimization_flag('ROCK_PROGRESSIVE_LOADING'):
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing")
            progress_window.geometry("300x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Center the progress window
            progress_window.geometry("+%d+%d" % (
                self.root.winfo_rootx() + self.root.winfo_width()//2 - 150,
                self.root.winfo_rooty() + self.root.winfo_height()//2 - 75
            ))
            
            # Progress label and bar
            load_label = ttk.Label(progress_window, text="Loading image...", font=("Segoe UI", 10))
            load_label.pack(pady=(20, 5))
            
            progress = ttk.Progressbar(progress_window, orient="horizontal", 
                                     length=250, mode="indeterminate")
            progress.pack(pady=10, padx=25)
            progress.start(10)
            
            # Refresh the UI
            self.root.update()
        
        try:
            # Create the output directory
            output_dir = self.save_path.get()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Pick the base name
            image_basename = os.path.basename(image_path).split('.')[0]
            output_subdir = os.path.join(output_dir, image_basename)
            
            threads = int(os.environ.get('ROCK_THREADS', '0'))
            
            # Update status
            if 'progress_window' in locals():
                load_label.config(text="Initializing detector...")
                self.root.update()
            else:
                self.status_var.set("Initializing detector...")
                self.root.update()
            
            # Initialize the detector
            start_time = time.time()
            self.detector = RockCoreLayerDetector(image_path)
            self.detector.output_dir = output_subdir
            
            # Sync the scale-calibration parameter
            if self.pixel_per_mm is not None:
                self.detector.pixel_per_mm = self.pixel_per_mm
            
            if 'progress_window' in locals():
                load_label.config(text="Preprocessing image...")
                progress.configure(mode="determinate", value=20)
                self.root.update()
            else:
                self.status_var.set("Preprocessing image...")
                self.root.update()
            
            # Read parameters
            blur_size = self.blur_size.get()
            clahe_clip = self.clahe_clip.get()
            clahe_grid = (self.clahe_grid_x.get(), self.clahe_grid_y.get())
            
            # Preprocess image (brightness / contrast / gamma enhancement)
            self.detector.preprocess_image(
                blur_size=blur_size, 
                clahe_clip=clahe_clip, 
                clahe_grid=clahe_grid,
                brightness=self.brightness.get(),
                contrast=self.contrast.get(),
                gamma=self.gamma.get()
            )
            
            # Update progress
            if 'progress_window' in locals():
                load_label.config(text="Detecting laminae...")
                progress.configure(value=40)
                self.root.update()
            else:
                self.status_var.set("Detecting laminae...")
                self.root.update()
            
            # Read parameters
            threshold_method = self.threshold_method.get()
            min_layer_width = self.min_layer_width.get()
            scan_line_count = self.scan_line_count.get()
            min_validation_lines = self.min_validation_lines.get()
            align_core = self.align_core.get()
            alignment_angle = self.alignment_angle.get()
            # Optional manual lamina direction (fractured-core override)
            user_dip_angle = None
            if getattr(self, 'use_manual_lamina_angle', None) is not None and self.use_manual_lamina_angle.get():
                try:
                    user_dip_angle = float(self.manual_lamina_angle.get())
                except (TypeError, ValueError, tk.TclError):
                    user_dip_angle = None
            
            # Minimum line-coverage fraction a lamina must span (e.g. 70%)
            try:
                self.detector.min_support_ratio = max(0.1, min(1.0,
                    float(self.min_support_pct.get()) / 100.0))
            except (TypeError, ValueError, tk.TclError):
                self.detector.min_support_ratio = 0.70
            
            # Detect laminae using preset or auto-generated scan lines
            scan_lines = self.custom_scan_lines if self.custom_scan_lines else None
            
            detect_success = self.detector.detect_layers(
                threshold_method=threshold_method,
                min_layer_width=min_layer_width,
                scan_lines=scan_lines,
                scan_line_count=scan_line_count if scan_lines is None else len(scan_lines),
                min_validation_lines=min_validation_lines,
                align_core=align_core,
                alignment_angle=alignment_angle,
                user_dip_angle_deg=user_dip_angle
            )
            
            if not detect_success:
                messagebox.showwarning("Info", "No lamina change-points detected; results may be empty.\nTry adjusting parameters (reduce min lamina width, increase scan-line count, etc.).")
            else:
                prep_meta = getattr(self.detector, '_preprocess_meta', {}) or {}
                det_diag = getattr(self.detector, '_detection_diagnostics', {}) or {}
                tips = []
                # Cross-line clustering result (primary display)
                lam_settings = getattr(self.detector, '_lamina_settings', {}) or {}
                if lam_settings:
                    n_unique = lam_settings.get('n_valid_laminae', 0)
                    n_total = lam_settings.get('n_clusters', 0)
                    min_sup = lam_settings.get('min_support', 0)
                    n_sl = lam_settings.get('n_scan_lines', 0)
                    n_cand = lam_settings.get('candidate_points', 0)
                    tips.append(
                        f"Cross-line clustering: {n_unique} unique laminae (candidate clusters={n_total}, candidate points={n_cand}, "
                        f"requires >= {min_sup}/{n_sl} supporting lines)"
                    )
                if prep_meta.get('dark_mode_applied'):
                    tips.append(
                        f"Dark-core auto enhancement applied (mean={prep_meta.get('image_mean', 0):.1f}, "
                        f"std={prep_meta.get('image_std', 0):.1f})"
                    )
                fbs = det_diag.get('fallbacks_triggered', [])
                if fbs:
                    tips.append(f"Detection triggered {len(fbs)} filter fallback(s) (results may be looser than configured):")
                    tips.extend(f"  · {x}" for x in fbs[:6])
                    if len(fbs) > 6:
                        tips.append(f"  ... (remaining {len(fbs) - 6} entries; see console / sample_metadata.json)")
                if tips:
                    self.status_var.set(tips[0])
                    print("[GUI diagnostics]")
                    for t in tips:
                        print(" ", t)
            
            # Update progress
            if 'progress_window' in locals():
                load_label.config(text="Computing statistics...")
                progress.configure(value=70)
                self.root.update()
            else:
                self.status_var.set("Computing statistics...")
                self.root.update()
            
            # Compute statistics
            stats_result = self.detector.calculate_statistics()
            
            # If a depth range is enabled, run depth interpolation
            enable_depth_range = self.enable_depth_range.get()
            start_depth = self.start_depth.get()
            end_depth = self.end_depth.get()
            
            if enable_depth_range and start_depth != end_depth and stats_result:
                stats, detailed_df, position_df = stats_result
                
                print(f"=== Depth-range processing ===")
                print(f"Start depth: {start_depth} m, end depth: {end_depth} m")
                print(f"Original detailed_df shape: {detailed_df.shape if detailed_df is not None else 'None'}")
                print(f"Original detailed_df columns: {list(detailed_df.columns) if detailed_df is not None else 'None'}")
                
                # Run depth interpolation
                depth_processed_stats = self._process_depth_interpolation(
                    stats, detailed_df, position_df, start_depth, end_depth
                )
                
                # Update detector statistics
                if depth_processed_stats:
                    self.detector.layer_stats = depth_processed_stats
            
            # Ensure layer_stats exists even if depth processing was disabled
            if not hasattr(self.detector, 'layer_stats') or not self.detector.layer_stats:
                # Make sure layer_stats is set correctly
                # calculate_statistics() should have set layer_stats, but double-check just in case
                if not hasattr(self.detector, 'layer_stats'):
                    print("Warning: calculate_statistics() did not set the layer_stats attribute")
                    # Fallback: manually trigger statistics calculation
                    self.detector.calculate_statistics()
            
            # Export results
            if 'progress_window' in locals():
                load_label.config(text="Exporting results...")
                progress.configure(value=90)
                self.root.update()
            else:
                self.status_var.set("Exporting results...")
                self.root.update()
                
            self.detector.export_results(output_subdir)
            
            # Stop the timer
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Close the progress window
            if 'progress_window' in locals():
                progress_window.destroy()
            
            # Show the results
            self.show_results()
            
            # Update status
            self.status_var.set(f"Processing complete in {processing_time:.2f}s; results in: {os.path.basename(output_subdir)}")
            
            result_msg = f"Single-image processing complete!\n\n"
            result_msg += f"Processing time: {processing_time:.2f} s\n"
            result_msg += f"Results saved to: {output_subdir}\n\n"
            result_msg += f"Core data exported:\n"
            result_msg += f"  - layer_detection.png  -- detection-result annotation\n"
            result_msg += f"  - layer_info.xlsx  -- detailed lamina table\n"
            result_msg += f"  - summary.xlsx  -- summary statistics\n"
            result_msg += f"  - lamina_variation_curve.xlsx  -- lamina variation curve\n\n"
            result_msg += f"For the full pipeline images and complete paper figures,\n"
            result_msg += f"please click the \"Export paper figures + data\" button."
            
            messagebox.showinfo("Processing complete", result_msg)
            
            # Select the tab
            self.tab_control.select(self.tab_results)
            
            return True
            
        except Exception as e:
            # Close the progress window
            if 'progress_window' in locals():
                progress_window.destroy()
                
            # Show the error
            error_msg = str(e)
            messagebox.showerror("Processing error", f"Error while processing image:\n{error_msg}")
            self.status_var.set("Processing failed")
            
            return False
    def show_original(self):
        """Display the original image."""
        image_path = self.image_path.get()
        if not image_path:
            messagebox.showerror("Error", "Please select an image file first")
            return
        
        # Clear the tab contents
        for widget in self.tab_original.winfo_children():
            widget.destroy()
        
        try:
            # Check whether we are in batch mode and look for images in sub-folders
            if self.batch_mode.get() and os.path.isdir(image_path):
                folder_path = image_path
                save_path = self.save_path.get()
                image_ext = self.image_ext.get()
                
                # Enumerate sub-folder names (folders named after each image)
                # Batch results live under the batch_results sub-directory
                batch_results_path = os.path.join(save_path, "batch_results")
                subdirs = []
                try:
                    # Check whether batch_results exists
                    if os.path.exists(batch_results_path):
                        actual_results_path = batch_results_path
                    else:
                        actual_results_path = save_path
                    
                    for d in os.listdir(actual_results_path):
                        d_path = os.path.join(actual_results_path, d)
                        if os.path.isdir(d_path) and not d.startswith('.'):
                            # Check whether processing results are present
                            has_results = (
                                os.path.exists(os.path.join(d_path, "layer_detection.png")) or
                                os.path.exists(os.path.join(d_path, "layer_info.xlsx")) or
                                os.path.exists(os.path.join(d_path, "layer_info.csv"))
                            )
                            if has_results:
                                subdirs.append(d)
                    
                    # Apply natural sort to the sub-directories
                    subdirs.sort(key=natural_sort_key)
                except Exception as e:
                    pass
                
                # Find the original image that matches each result folder
                image_files = []
                for subdir in subdirs:
                    # Try to find the matching original image
                    found_image = False
                    for ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                        potential_file = subdir + ext
                        if os.path.exists(os.path.join(folder_path, potential_file)):
                            image_files.append(potential_file)
                            found_image = True
                            break
                    
                    if not found_image:
                        # Skip if no matching image is found
                        pass
                
                if not image_files:
                    messagebox.showinfo("Info", f"No {image_ext} image files found in {folder_path}")
                    return
                
                # Frame for the image and controls
                main_frame = ttk.Frame(self.tab_original)
                main_frame.pack(fill=tk.BOTH, expand=True)
                
                # Top control bar
                control_frame = ttk.Frame(main_frame)
                control_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # File-switching drop-down
                ttk.Label(control_frame, text="Select image:").pack(side=tk.LEFT, padx=(0, 5))
                
                # Track the currently selected image
                self.current_image_var = tk.StringVar(self.root)
                
                # Build the drop-down
                image_combo = ttk.Combobox(control_frame, textvariable=self.current_image_var, 
                                         width=40, state="readonly")
                image_combo.pack(side=tk.LEFT, padx=5)
                
                # Populate the drop-down and pick a default
                image_combo['values'] = image_files
                if image_files:
                    image_combo.current(0)
                
                # Frame that hosts the image
                image_frame = ttk.Frame(main_frame)
                image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                
                # Helper to refresh the image display
                def update_image(*args):
                    selected_image = self.current_image_var.get()
                    if selected_image:
                        full_path = os.path.join(folder_path, selected_image)
                        # Clear the image frame
                        for widget in image_frame.winfo_children():
                            widget.destroy()
                        
                        try:
                            image = self._load_image_from_path(full_path)
                            if image is None:
                                raise FileNotFoundError(f"Failed to load image: {full_path}")
                                
                            # Convert color space
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            # Resize the image to fit the frame
                            height, width = image.shape[:2]
                            max_height = 600
                            max_width = 800
                            
                            scale = min(max_width/width, max_height/height)
                            if scale < 1:
                                width = int(width * scale)
                                height = int(height * scale)
                                image = cv2.resize(image, (width, height))
                            
                            # Convert to a Tkinter-displayable image
                            img = Image.fromarray(image)
                            img_tk = ImageTk.PhotoImage(img)
            
            # Display the image on the canvas
                            canvas = tk.Canvas(image_frame, width=img_tk.width(), height=img_tk.height())
                            canvas.pack(fill=tk.BOTH, expand=True)
                            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                            canvas.image = img_tk  # keep a reference
                            
                            self.status_var.set(f"Showing original image: {selected_image}")
                        except Exception as e:
                            ttk.Label(image_frame, text=f"Cannot display image: {str(e)}").pack(pady=20)
                    
                    # Update the scroll region
                    self.root.after_idle(self.update_display_scroll_region)
                
                # Bind the drop-down change event
                self.current_image_var.trace("w", update_image)
                
                # Show the first image initially
                if image_files:
                    update_image()
                
            else:
                # Single-image mode (legacy logic)
                # Check the image path
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file does not exist: {image_path}")
                
                image = self._load_image_from_path(image_path)
                if image is None:
                    raise FileNotFoundError(f"Failed to load image: {image_path}; check the path and image format")
                
                # Convert color space
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize the image to fit the tab
                height, width = image.shape[:2]
                max_height = 700
                max_width = 800
                
                scale = min(max_width/width, max_height/height)
                if scale < 1:
                    width = int(width * scale)
                    height = int(height * scale)
                    image = cv2.resize(image, (width, height))
                
                # Convert to a Tkinter-displayable image
                img = Image.fromarray(image)
                img_tk = ImageTk.PhotoImage(img)
                
                # Display the image on the canvas
                canvas = tk.Canvas(self.tab_original, width=img_tk.width(), height=img_tk.height())
                canvas.pack(fill=tk.BOTH, expand=True)
                canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                canvas.image = img_tk  # keep a reference
                
                # Update the scroll region
                self.root.after_idle(self.update_display_scroll_region)
                
                # Switch to the original-image tab
                self.tab_control.select(self.tab_original)
                self.status_var.set(f"Showing original image: {os.path.basename(image_path)}")
        
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Image file not found or unreadable")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display image:\n{str(e)}")
            self.status_var.set("Failed to display image")
    def show_results(self):
        """Show detection results (core processing, lamina density, lamina width)."""
        # Batch mode
        if self.batch_mode.get() and os.path.isdir(self.image_path.get()):
            folder_path = self.image_path.get()
            save_path = self.save_path.get()
            image_ext = self.image_ext.get()
            
            # Enumerate sub-folder names (folders named after each image)
            # Batch results live under the batch_results sub-directory
            batch_results_path = os.path.join(save_path, "batch_results")
            subdirs = []
            try:
                # Check whether batch_results exists
                if os.path.exists(batch_results_path):
                    actual_results_path = batch_results_path
                else:
                    actual_results_path = save_path
                
                for d in os.listdir(actual_results_path):
                    d_path = os.path.join(actual_results_path, d)
                    if os.path.isdir(d_path) and not d.startswith('.'):
                        # Check whether processing results are present
                        has_results = (
                            os.path.exists(os.path.join(d_path, "layer_detection.png")) or
                            os.path.exists(os.path.join(d_path, "layer_info.xlsx")) or
                            os.path.exists(os.path.join(d_path, "layer_info.csv"))
                        )
                        if has_results:
                            subdirs.append(d)
                
                # Apply natural sort to the sub-directories
                subdirs.sort(key=natural_sort_key)
            except Exception as e:
                pass
            
            if not subdirs:
                messagebox.showinfo("Info", f"No result folders found in {save_path}")
                return
            
            # Clear the tab contents
            for widget in self.tab_results.winfo_children():
                widget.destroy()
            
            # Build the main frame
            main_frame = ttk.Frame(self.tab_results)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Top control bar
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # File-switching drop-down
            ttk.Label(control_frame, text="Select image:").pack(side=tk.LEFT, padx=(0, 5))
            
            # Track the selected result folder
            self.current_result_var = tk.StringVar(self.root)
            
            # Build the drop-down
            result_combo = ttk.Combobox(control_frame, textvariable=self.current_result_var, 
                                     width=40, state="readonly")
            result_combo.pack(side=tk.LEFT, padx=5)
            
            # Populate the drop-down and pick a default
            result_combo['values'] = subdirs
            if subdirs:
                result_combo.current(0)
            
            # Build the scrollable content frame
            content_outer_frame = ttk.Frame(main_frame)
            content_outer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Build the vertical scrollbar
            content_vscroll = ttk.Scrollbar(content_outer_frame, orient="vertical")
            content_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Canvas to host the scrolling content
            content_canvas = tk.Canvas(content_outer_frame, yscrollcommand=content_vscroll.set)
            content_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Wire the scrollbar
            content_vscroll.config(command=content_canvas.yview)
            
            # Build the actual content frame inside the canvas
            content_frame = ttk.Frame(content_canvas)
            content_frame_window = content_canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)
            
            # Three sub-frames for the result
            detected_frame = ttk.LabelFrame(content_frame, text="Core processing")
            detected_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            
            density_frame = ttk.LabelFrame(content_frame, text="Lateral lamina distribution")
            density_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            
            width_frame = ttk.LabelFrame(content_frame, text="Lamina strength curve")
            width_frame.pack(fill=tk.BOTH, expand=True)
            
            # Configure the content scroll region
            def configure_content_scroll_region(event=None):
                content_canvas.configure(scrollregion=content_canvas.bbox("all"))
                # Resize the canvas window to match the container
                canvas_width = content_canvas.winfo_width()
                if canvas_width > 1:
                    content_canvas.itemconfig(content_frame_window, width=canvas_width-20)
            
            # Bind events
            content_frame.bind("<Configure>", configure_content_scroll_region)
            content_canvas.bind("<Configure>", configure_content_scroll_region)
            
            # Bind mouse-wheel events
            def _on_content_mousewheel(event):
                content_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
            content_canvas.bind("<MouseWheel>", _on_content_mousewheel)
            
            # Helper to refresh the result display
            def update_results(*args):
                selected_subdir = self.current_result_var.get()
                if selected_subdir:
                    # Clear all sub-frames
                    for widget in detected_frame.winfo_children():
                        widget.destroy()
                    for widget in density_frame.winfo_children():
                        widget.destroy()
                    for widget in width_frame.winfo_children():
                        widget.destroy()
                    
                    # Compute the actual result_path based on actual_results_path
                    if os.path.exists(batch_results_path):
                                                result_path = os.path.join(batch_results_path, selected_subdir)
                    else:
                        result_path = os.path.join(save_path, selected_subdir)
                    
                    # Show the detection-result image
                    detected_img_path = os.path.join(result_path, "layer_detection.png")
                    if os.path.exists(detected_img_path):
                        try:
                            img = Image.open(detected_img_path)
                            # Resize to fit the frame
                            img = img.resize((800, 200), Image.LANCZOS)
                            img_tk = ImageTk.PhotoImage(img)
                            
                            # Show the image
                            label = ttk.Label(detected_frame, image=img_tk)
                            label.image = img_tk  # keep a reference
                            label.pack(fill=tk.BOTH, expand=True)
                        except Exception as e:
                            ttk.Label(detected_frame, text=f"Cannot load image: {str(e)}").pack(pady=20)
                    else:
                        ttk.Label(detected_frame, text="Detection-result image not found").pack(pady=20)
                    
                    # Show the density figure
                    density_img_paths = [
                        os.path.join(result_path, "layer_density.png"),
                        os.path.join(result_path, "density_distribution.png"),
                        os.path.join(result_path, "lateral_lamina_distribution.png")
                    ]
                    density_img_path = None
                    for path in density_img_paths:
                        if os.path.exists(path):
                            density_img_path = path
                            break
                    
                    if density_img_path:
                        try:
                            img = Image.open(density_img_path)
                            img = img.resize((800, 200), Image.LANCZOS)
                            img_tk = ImageTk.PhotoImage(img)
                            
                            label = ttk.Label(density_frame, image=img_tk)
                            label.image = img_tk
                            label.pack(fill=tk.BOTH, expand=True)
                        except Exception as e:
                            ttk.Label(density_frame, text=f"Cannot load image: {str(e)}").pack(pady=20)
                    else:
                        # List the files actually present
                        available_files = os.listdir(result_path) if os.path.exists(result_path) else []
                        ttk.Label(density_frame, text=f"Density image not found\nAvailable files: {available_files}").pack(pady=20)
                    
                    # Show the strength curve
                    intensity_img_paths = [
                        os.path.join(result_path, "layer_intensity.png"),
                        os.path.join(result_path, "intensity_curve.png"),
                        os.path.join(result_path, "lamina_strength_curve.png"),
                        os.path.join(result_path, "width_distribution.png")
                    ]
                    width_img_path = None
                    for path in intensity_img_paths:
                        if os.path.exists(path):
                            width_img_path = path
                            break
                    
                    if width_img_path:
                        try:
                            img = Image.open(width_img_path)
                            img = img.resize((800, 200), Image.LANCZOS)
                            img_tk = ImageTk.PhotoImage(img)
                            
                            label = ttk.Label(width_frame, image=img_tk)
                            label.image = img_tk
                            label.pack(fill=tk.BOTH, expand=True)
                        except Exception as e:
                            ttk.Label(width_frame, text=f"Cannot load image: {str(e)}").pack(pady=20)
                    else:
                        # List the files actually present
                        available_files = os.listdir(result_path) if os.path.exists(result_path) else []
                        png_files = [f for f in available_files if f.endswith('.png')]
                        ttk.Label(width_frame, text=f"Strength-curve image not found\nAvailable PNGs: {png_files}").pack(pady=20)
                
                # Update the scroll region
                self.root.after_idle(configure_content_scroll_region)
            
            # Bind the drop-down change event
            self.current_result_var.trace("w", update_results)
            
            # Show the first result initially
            if subdirs:
                update_results()
            
            # Switch to the detection-results tab
            self.tab_control.select(self.tab_results)
            self.status_var.set("Showing batch-mode detection results")
            
        elif not self.detector or not self.detector.layers:
            messagebox.showerror("Error", "Please analyze an image first")
            return
        else:
            # Single-image mode (legacy logic)
            try:
                # Switch to the detection-results tab
                self.tab_control.select(self.tab_results)
                
                # Show core-processing result
                self._show_detected_in_frame(self.results_frame_detected)
                
                # Show lamina density
                self._show_density_in_frame(self.results_frame_density)
                
                # Show lamina strength variation
                self._show_width_hist_in_frame(self.results_frame_width)
                
                # Refresh the main display scroll region
                self.root.after_idle(lambda: (
                    self.display_canvas.configure(scrollregion=self.display_canvas.bbox("all")),
                    self.display_canvas.itemconfig(self.display_frame_window, 
                                                 width=max(1, self.display_canvas.winfo_width()-20))
                ))
                
                self.status_var.set("Showing detection results")
            
            except Exception as e:
                messagebox.showerror("Error", f"Cannot display detection results:\n{str(e)}")
                self.status_var.set("Failed to display detection results")
    def _show_detected_in_frame(self, frame):
        """Show detection results in the given frame: top = image + markers, bottom = gradient pseudo-color."""
        for widget in frame.winfo_children():
            widget.destroy()
        
        h, w = self.detector.image.shape[:2]
        
        # --- Top: image + vertical lamina-marker lines ---
        overlay = self.detector.image.copy()
        
        scan_ys = sorted(set(sr["y"] for sr in self.detector.layers))
        
        for scan_result in self.detector.layers:
            y = scan_result["y"]
            pts = scan_result.get("validated_points", scan_result["points"])
            
            idx = scan_ys.index(y)
            y_top = (scan_ys[idx - 1] + y) // 2 if idx > 0 else max(0, y - 15)
            y_bot = (y + scan_ys[idx + 1]) // 2 if idx < len(scan_ys) - 1 else min(h - 1, y + 15)
            
            for x in pts:
                cv2.line(overlay, (x, y_top), (x, y_bot), (0, 0, 255), 1)
                cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
        
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # --- Bottom: gradient pseudo-color image showing color variation ---
        gray = cv2.cvtColor(self.detector.image, cv2.COLOR_BGR2GRAY) if len(self.detector.image.shape) == 3 else self.detector.image.copy()
        
        # Compute the cross-lamina gradient (laminae are vertical, so the X-derivative is most sensitive)
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_abs = np.abs(grad)
        # Normalize to 0-255
        if grad_abs.max() > 0:
            grad_norm = (grad_abs / grad_abs.max() * 255).astype(np.uint8)
        else:
            grad_norm = np.zeros_like(gray)
        
        # Apply the JET pseudo-color map (blue=no change, red=strong change)
        grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)
        grad_color_rgb = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)
        
        # Stack: marker image on top, gradient image at the bottom
        combined = np.vstack([overlay_rgb, grad_color_rgb])
        
        # Resize for display
        ch, cw = combined.shape[:2]
        max_height = 400
        max_width = 800
        scale = min(max_width / cw, max_height / ch, 1.0)
        if scale < 1:
            combined = cv2.resize(combined, (int(cw * scale), int(ch * scale)))
        
        img = Image.fromarray(combined)
        img_tk = ImageTk.PhotoImage(img)
        
        # Caption
        ttk.Label(frame, text="Top: lamina overlay (orange = detected laminae)   Bottom: gradient pseudo-color image (red = strong color change)",
                  font=("Segoe UI", 8), foreground="gray").pack(anchor=tk.W, padx=5)
        
        canvas = tk.Canvas(frame, width=img_tk.width(), height=img_tk.height())
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk
    def _show_density_in_frame(self, frame):
        """Show the lamina-distribution figure in the given frame."""
        # Clear the frame
        for widget in frame.winfo_children():
            widget.destroy()
        
        # Check whether depth-range mode is enabled and depth data exists
        depth_enabled = self.enable_depth_range.get() if hasattr(self, 'enable_depth_range') else False
        has_depth_stats = (hasattr(self.detector, 'layer_stats') and 
                         self.detector.layer_stats and 
                         'detailed' in self.detector.layer_stats)
        
        # Additionally verify the presence of depth data (English/Chinese column names)
        has_depth_data = False
        if has_depth_stats:
            # Check the detailed DataFrame first
            detailed_df = self.detector.layer_stats.get('detailed')
            if detailed_df is not None and not detailed_df.empty:
                for col in ['depth_m', 'depth']:
                    if col in detailed_df.columns:
                        has_depth_data = True
                        break
            
            # If the detailed DataFrame has no depth, check the position DataFrame
            if not has_depth_data and hasattr(self.detector, 'layer_stats') and self.detector.layer_stats and 'position' in self.detector.layer_stats:
                position_df = self.detector.layer_stats['position']
                if position_df is not None and not position_df.empty:
                    for col in ['depth_m', 'depth']:
                        if col in position_df.columns:
                            has_depth_data = True
                            break
        
        if depth_enabled and has_depth_data:
            # Show the depth-based plot
            self._show_depth_density_in_frame(frame)
        else:
            # Show the pixel-position-based plot (legacy logic)
            self._show_pixel_density_in_frame(frame)
    def _show_pixel_density_in_frame(self, frame):
        """Show the pixel-position density figure (legacy logic)."""
        # Check whether position statistics are available
        if (hasattr(self.detector, 'layer_stats') and 
            self.detector.layer_stats and 
            'position' in self.detector.layer_stats):
            
            position_df = self.detector.layer_stats['position']
            if position_df is not None and not position_df.empty:
                # Use the position statistics
                from matplotlib.figure import Figure
                fig = Figure(figsize=(8, 3), dpi=100)
                ax = fig.add_subplot(111)
                
                # Find the position and density columns
                position_col = None
                density_col = None
                
                for col in ['position_px', 'position']:
                    if col in position_df.columns:
                        position_col = col
                        break
                
                for col in ['density_per_100px', 'density']:
                    if col in position_df.columns:
                        density_col = col
                        break
                
                if position_col and density_col:
                    positions = position_df[position_col].values
                    densities = position_df[density_col].values
                    
                    ax.bar(positions, densities, width=3)
                    ax.set_title("Lateral lamina distribution")
                    ax.set_xlabel("Lateral position (px)")
                    ax.set_ylabel("Lamina density")
                    ax.grid(True)
                    
                    # Embed the figure in the frame
                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    return
        
        # Fall back to the legacy logic when no position statistics exist
        # Build a position-vs-count histogram
        position_counts = {}
        if self.detector and self.detector.layers:
            for scan_result in self.detector.layers:
                for pos in scan_result["points"]:
                    if pos in position_counts:
                        position_counts[pos] += 1
                    else:
                        position_counts[pos] = 1
        
        if not position_counts:
            label = ttk.Label(frame, text="No valid lamina data detected", font=self.default_font)
            label.pack(pady=20)
            return
        
        # Build the matplotlib figure
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot the bar chart
        positions = sorted(position_counts.keys())
        counts = [position_counts[pos] for pos in positions]
        
        ax.bar(positions, counts, width=3)
        ax.set_title("Lateral lamina distribution")
        ax.set_xlabel("Lateral position (px)")
        ax.set_ylabel("Change-point count")
        ax.grid(True)
        
        # Embed the figure in the frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def _show_depth_density_in_frame(self, frame):
        """Show the depth-based density figure (depth on the x-axis)."""
        try:
            # First check whether position statistics contain depth info
            position_df = None
            if (hasattr(self.detector, 'layer_stats') and 
                self.detector.layer_stats and 
                'position' in self.detector.layer_stats):
                position_df = self.detector.layer_stats['position']
            
            # Use the depth column from position statistics if present
            has_depth_column = False
            depth_column = None
            if position_df is not None and not position_df.empty:
                for col in ['depth_m', 'depth']:
                    if col in position_df.columns:
                        has_depth_column = True
                        depth_column = col
                        break
            
            if has_depth_column:
                from matplotlib.figure import Figure
                import numpy as np
                
                fig = Figure(figsize=(8, 4), dpi=100)
                ax = fig.add_subplot(111)
                
                depths = position_df[depth_column].values
                start_depth = self.start_depth.get()
                end_depth = self.end_depth.get()
                
                # Find the density column
                density_col = None
                for col in ['density_per_100px', 'density']:
                    if col in position_df.columns:
                        density_col = col
                        break
                
                if density_col:
                    densities = position_df[density_col].values
                    
                    # Plot bar chart with depth on the X-axis
                    ax.bar(depths, densities, width=(end_depth-start_depth)/len(depths)*0.8, alpha=0.7, 
                           color='skyblue', edgecolor='black', linewidth=0.5)
                    
                    # Configure labels and title
                    ax.set_xlabel("Depth (m)", fontsize=12)
                    ax.set_ylabel("Lamina density", fontsize=12)
                    ax.set_title(f"Lateral lamina distribution by depth\nDepth range: {start_depth:.1f} m - {end_depth:.1f} m",
                                fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Set the depth range
                    ax.set_xlim(start_depth, end_depth)
                    
                    # Annotate statistics
                    total_positions = len(position_df)
                    avg_depth = depths.mean()
                    max_density = np.max(densities) if len(densities) > 0 else 0
                    stats_text = f'Positions counted: {total_positions}\nMean depth: {avg_depth:.2f} m\nMax density: {max_density:.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Embed the figure in the frame
                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    return
            
            # If position statistics lack depth info, try the detailed DataFrame
            detailed_df = None
            if (hasattr(self.detector, 'layer_stats') and 
                self.detector.layer_stats and 
                'detailed' in self.detector.layer_stats):
                detailed_df = self.detector.layer_stats.get('detailed')
            
            # Check whether the detailed DataFrame has a depth column
            has_detailed_depth = False
            detailed_depth_column = None
            if detailed_df is not None and not detailed_df.empty:
                for col in ['depth_m', 'depth']:
                    if col in detailed_df.columns:
                        has_detailed_depth = True
                        detailed_depth_column = col
                        break
            
            if not has_detailed_depth:
                label = ttk.Label(frame, text="No depth data", font=self.default_font)
                label.pack(pady=20)
                return
            
            from matplotlib.figure import Figure
            import numpy as np
            
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            depths = detailed_df[detailed_depth_column].values
            start_depth = self.start_depth.get()
            end_depth = self.end_depth.get()
            depth_range = end_depth - start_depth
            num_bins = min(50, max(10, int(depth_range * 5)))
            
            # Build depth bins and compute density per bin
            depth_bins = np.linspace(start_depth, end_depth, num_bins + 1)
            bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
            depth_counts, _ = np.histogram(depths, bins=depth_bins)
            
            # Plot bar chart with depth on the X-axis
            bin_width = depth_bins[1] - depth_bins[0]
            ax.bar(bin_centers, depth_counts, width=bin_width*0.8, alpha=0.7, 
                   color='skyblue', edgecolor='black', linewidth=0.5)
            
            # Configure labels and title
            ax.set_xlabel("Depth (m)", fontsize=12)
            ax.set_ylabel("Lamina count", fontsize=12)
            ax.set_title(f"Lateral lamina distribution by depth\nDepth range: {start_depth:.1f} m - {end_depth:.1f} m",
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set the depth range
            ax.set_xlim(start_depth, end_depth)
            
            # Annotate statistics
            total_layers = len(detailed_df)
            avg_depth = detailed_df[detailed_depth_column].mean()
            max_count = np.max(depth_counts) if len(depth_counts) > 0 else 0
            stats_text = f'Total laminae: {total_layers}\nMean depth: {avg_depth:.2f} m\nMax density: {max_count}/bin'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Embed the figure in the frame
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            label = ttk.Label(frame, text=f"Error while showing depth distribution: {str(e)}", font=self.default_font)
            label.pack(pady=20)
    def _show_width_hist_in_frame(self, frame):
        """Show the lamina-strength curve in the given frame."""
        # Clear the frame
        for widget in frame.winfo_children():
            widget.destroy()
        
        if not self.detector:
            label = ttk.Label(frame, text="No valid detector data", font=self.default_font)
            label.pack(pady=20)
            return
        
        try:
            # Check whether depth-range mode is enabled and depth data exists
            depth_enabled = self.enable_depth_range.get() if hasattr(self, 'enable_depth_range') else False
            has_depth_stats = (hasattr(self.detector, 'layer_stats') and 
                             self.detector.layer_stats and 
                             'detailed' in self.detector.layer_stats)
            
            # Additionally verify the presence of depth data (English/Chinese column names)
            has_depth_data = False
            if has_depth_stats:
                detailed_df = self.detector.layer_stats.get('detailed')
                if detailed_df is not None and not detailed_df.empty:
                    for col in ['depth_m', 'depth']:
                        if col in detailed_df.columns:
                            has_depth_data = True
                            break
                
                # If the detailed DataFrame has no depth, check the position DataFrame
                if not has_depth_data and 'position' in self.detector.layer_stats:
                    position_df = self.detector.layer_stats['position']
                    if position_df is not None and not position_df.empty:
                        for col in ['depth_m', 'depth']:
                            if col in position_df.columns:
                                has_depth_data = True
                                break
            
            if depth_enabled and has_depth_data:
                # Show the depth-based plot
                self._show_depth_based_chart_in_frame(frame)
            else:
                # Show the pixel-position-based plot (legacy logic)
                self._show_pixel_based_chart_in_frame(frame)
        
        except Exception as e:
            label = ttk.Label(frame, text=f"Error while displaying chart: {str(e)}", font=self.default_font)
            label.pack(pady=20)
    def _show_pixel_based_chart_in_frame(self, frame):
        """Show the pixel-position strength curve (legacy logic)."""
        try:
            # Check whether position statistics are available
            if (hasattr(self.detector, 'layer_stats') and 
                self.detector.layer_stats and 
                'position' in self.detector.layer_stats):
                
                position_df = self.detector.layer_stats['position']
                if position_df is not None and not position_df.empty:
                    # Use the position statistics
                    from matplotlib.figure import Figure
                    fig = Figure(figsize=(8, 3), dpi=100)
                    ax = fig.add_subplot(111)
                    
                    # Find the position and strength columns
                    position_col = None
                    intensity_col = None
                    
                    for col in ['position_px', 'position']:
                        if col in position_df.columns:
                            position_col = col
                            break
                    
                    for col in ['strength_normalized', 'intensity']:
                        if col in position_df.columns:
                            intensity_col = col
                            break
                    
                    if position_col and intensity_col:
                        positions = position_df[position_col].values
                        intensities = position_df[intensity_col].values
                        
                        # Plot the strength curve
                        ax.plot(positions, intensities, 'b-', linewidth=1.5)
                        ax.fill_between(positions, intensities, color='skyblue', alpha=0.4)
                        ax.set_title("Lamina strength curve")
                        ax.set_xlabel("Lateral position (px)")
                        ax.set_ylabel("Lamina strength (normalised)")
                        ax.set_xlim(positions.min(), positions.max())
                        ax.set_ylim(0, max(1.05, intensities.max() * 1.05))
                        ax.grid(True)
                        
                        # Embed the figure in the frame
                        canvas = FigureCanvasTkAgg(fig, master=frame)
                        canvas.draw()
                        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                        return
            
            # Fall back to the legacy logic when no position statistics exist
            if not self.detector or not hasattr(self.detector, 'image') or self.detector.image is None:
                label = ttk.Label(frame, text="No valid image data", font=self.default_font)
                label.pack(pady=20)
                return
            
            # Image width
            width = self.detector.image.shape[1]
            
            # Build the lateral-position strength array
            x_positions = np.arange(width)
            intensity = np.zeros(width)
            
            # Apply Gaussian smoothing
            sigma = 5  # Gaussian kernel sigma (smoothing strength)
            
            # Add strength at the location of each detected change-point
            if hasattr(self.detector, 'layers') and self.detector.layers:
                for scan_result in self.detector.layers:
                    for pos in scan_result["points"]:
                        if 0 <= pos < width:
                            # Increment strength at the change-point
                            intensity[pos] += 1
            
            # Apply Gaussian smoothing
            if len(intensity) > 0:
                from scipy.ndimage import gaussian_filter1d
                intensity = gaussian_filter1d(intensity.astype(float), sigma=sigma)
                
                # Normalize strength to 0-1
                if len(intensity) > 0 and np.max(intensity) > 0:
                    intensity = intensity / np.max(intensity)
                
                # Build the matplotlib figure
                from matplotlib.figure import Figure
                fig = Figure(figsize=(8, 3), dpi=100)
                ax = fig.add_subplot(111)
                
                # Plot the strength curve
                ax.plot(x_positions, intensity, 'b-', linewidth=1.5)
                ax.fill_between(x_positions, intensity, color='skyblue', alpha=0.4)
                ax.set_title("Lamina strength curve")
                ax.set_xlabel("Lateral position (px)")
                ax.set_ylabel("Lamina strength (normalised)")
                ax.set_xlim(0, width)
                ax.set_ylim(0, 1.05)
                ax.grid(True)
                
                # Embed the figure in the frame
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                label = ttk.Label(frame, text="No valid lamina data detected", font=self.default_font)
                label.pack(pady=20)
        
        except Exception as e:
            label = ttk.Label(frame, text=f"Error while drawing curve: {str(e)}", font=self.default_font)
            label.pack(pady=20)
    def _show_depth_based_chart_in_frame(self, frame):
        """Show the depth-based strength curve (depth on the x-axis)."""
        try:
            # First check whether position statistics contain depth info
            position_df = None
            if (hasattr(self.detector, 'layer_stats') and 
                self.detector.layer_stats and 
                'position' in self.detector.layer_stats):
                position_df = self.detector.layer_stats['position']
            
            # Use the depth column from position statistics if present
            has_depth_column = False
            depth_column = None
            if position_df is not None and not position_df.empty:
                for col in ['depth_m', 'depth']:
                    if col in position_df.columns:
                        has_depth_column = True
                        depth_column = col
                        break
            
            if has_depth_column:
                from matplotlib.figure import Figure
                import numpy as np
                
                fig = Figure(figsize=(8, 4), dpi=100)
                ax = fig.add_subplot(111)
                
                depths = position_df[depth_column].values
                start_depth = self.start_depth.get()
                end_depth = self.end_depth.get()
                
                # Find the strength column
                intensity_col = None
                for col in ['strength_normalized', 'intensity']:
                    if col in position_df.columns:
                        intensity_col = col
                        break
                
                if intensity_col:
                    intensities = position_df[intensity_col].values
                    
                    # Plot the curve horizontally with depth as the X-axis
                    ax.plot(depths, intensities, 'b-', linewidth=2, label='Lamina strength')
                    ax.fill_between(depths, intensities, color='skyblue', alpha=0.4)
                    
                    # Add data points
                    """ax.scatter(depths, intensities, c='red', s=20, alpha=0.6, zorder=5)"""
                    
                    # Configure labels and title
                    ax.set_xlabel('Depth (m)', fontsize=12)
                    ax.set_ylabel('Lamina strength', fontsize=12)
                    ax.set_title(f'Lamina strength vs depth\nDepth range: {start_depth:.1f} m - {end_depth:.1f} m',
                                fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Set the depth range
                    ax.set_xlim(start_depth, end_depth)
                    
                    # Annotate statistics
                    total_positions = len(position_df)
                    avg_depth = depths.mean()
                    avg_intensity = intensities.mean()
                    stats_text = f'Positions counted: {total_positions}\nMean depth: {avg_depth:.2f} m\nMean strength: {avg_intensity:.3f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Embed the figure in the frame
                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    return
            
            # If position statistics lack depth info, try the detailed DataFrame
            detailed_df = None
            if (hasattr(self.detector, 'layer_stats') and 
                self.detector.layer_stats and 
                'detailed' in self.detector.layer_stats):
                detailed_df = self.detector.layer_stats.get('detailed')
            
            # Check whether the detailed DataFrame has a depth column
            has_detailed_depth = False
            detailed_depth_column = None
            if detailed_df is not None and not detailed_df.empty:
                for col in ['depth_m', 'depth']:
                    if col in detailed_df.columns:
                        has_detailed_depth = True
                        detailed_depth_column = col
                        break
            
            if not has_detailed_depth:
                label = ttk.Label(frame, text="No depth data", font=self.default_font)
                label.pack(pady=20)
                return
            
            from matplotlib.figure import Figure
            import numpy as np
            
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            depths = detailed_df[detailed_depth_column].values
            start_depth = self.start_depth.get()
            end_depth = self.end_depth.get()
            
            # Build depth bins and compute strength per bin
            depth_range = end_depth - start_depth
            num_bins = min(100, max(20, int(depth_range * 10)))
            
            depth_bins = np.linspace(start_depth, end_depth, num_bins + 1)
            bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
            depth_counts, _ = np.histogram(depths, bins=depth_bins)
            
            # Smoothing
            from scipy.ndimage import gaussian_filter1d
            smoothed_counts = gaussian_filter1d(depth_counts.astype(float), sigma=1.0)
            
            # Plot the curve horizontally with depth as the X-axis
            if len(bin_centers) > 1:
                ax.plot(bin_centers, smoothed_counts, 'b-', linewidth=2, label='Lamina strength')
                ax.fill_between(bin_centers, smoothed_counts, color='skyblue', alpha=0.4)
                
                # Add data points
                ax.scatter(bin_centers, smoothed_counts, c='red', s=20, alpha=0.6, zorder=5)
            
            # Configure labels and title
            ax.set_xlabel('Depth (m)', fontsize=12)
            ax.set_ylabel('Lamina density', fontsize=12)
            ax.set_title(f'Lamina strength vs depth\nDepth range: {start_depth:.1f} m - {end_depth:.1f} m',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set the depth range
            ax.set_xlim(start_depth, end_depth)
            
            # Annotate statistics
            total_layers = len(detailed_df)
            avg_depth = detailed_df[detailed_depth_column].mean()
            stats_text = f'Total laminae: {total_layers}\nMean depth: {avg_depth:.2f} m'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Embed the figure in the frame
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            label = ttk.Label(frame, text=f"Error while showing depth chart: {str(e)}", font=self.default_font)
            label.pack(pady=20)
    def show_statistics(self):
        """Show the statistics tables."""
        # Batch mode
        if self.batch_mode.get() and os.path.isdir(self.image_path.get()):
            folder_path = self.image_path.get()
            save_path = self.save_path.get()
            
            # Enumerate sub-folder names (folders named after each image)
            # Batch results live under the batch_results sub-directory
            batch_results_path = os.path.join(save_path, "batch_results")
            subdirs = []
            try:
                # Check whether batch_results exists
                if os.path.exists(batch_results_path):
                    actual_results_path = batch_results_path
                else:
                    actual_results_path = save_path
                
                for d in os.listdir(actual_results_path):
                    d_path = os.path.join(actual_results_path, d)
                    if os.path.isdir(d_path) and not d.startswith('.'):
                        # Check whether processing results are present
                        has_results = (
                            os.path.exists(os.path.join(d_path, "layer_detection.png")) or
                            os.path.exists(os.path.join(d_path, "layer_info.xlsx")) or
                            os.path.exists(os.path.join(d_path, "layer_info.csv"))
                        )
                        if has_results:
                            subdirs.append(d)
                
                # Apply natural sort to the sub-directories
                subdirs.sort(key=natural_sort_key)
                print(f"Statistics tab found {len(subdirs)} result folder(s): {subdirs}")
            except Exception as e:
                print(f"Error while listing statistics sub-folders: {str(e)}")
            
            if not subdirs:
                messagebox.showinfo("Info", f"No result folders found in {save_path}")
                return
            
            # Clear the tab contents
            for widget in self.tab_stats.winfo_children():
                widget.destroy()
            
            # Build the main frame
            main_frame = ttk.Frame(self.tab_stats)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Top control bar
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # File-switching drop-down
            ttk.Label(control_frame, text="Select image:").pack(side=tk.LEFT, padx=(0, 5))
            
            # Track the selected result folder
            self.current_stats_var = tk.StringVar(self.root)
            
            # Build the drop-down
            stats_combo = ttk.Combobox(control_frame, textvariable=self.current_stats_var, 
                                     width=40, state="readonly")
            stats_combo.pack(side=tk.LEFT, padx=5)
            
            # Populate the drop-down and pick a default
            stats_combo['values'] = subdirs
            if subdirs:
                stats_combo.current(0)
            
            # Add the "View merged statistics" button
            ttk.Button(control_frame, text="View merged statistics", 
                      command=lambda: self.show_merged_statistics(batch_results_path if os.path.exists(batch_results_path) else save_path)).pack(side=tk.RIGHT, padx=5)
            
            # Build the content frame
            content_frame = ttk.Frame(main_frame)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Helper to refresh the statistics table
            def update_stats(*args):
                # Update status
                self.status_var.set("Loading statistics...")
                self.root.update()
                
                # Trigger garbage collection
                import gc
                gc.collect()
                
                selected_subdir = self.current_stats_var.get()
                if selected_subdir:
                    # Clear the content frame
                    for widget in content_frame.winfo_children():
                        widget.destroy()
                    
                    # Compute the actual result_path based on actual_results_path
                    if os.path.exists(batch_results_path):
                        result_path = os.path.join(batch_results_path, selected_subdir)
                    else:
                        result_path = os.path.join(save_path, selected_subdir)
                    
                    # Build a Notebook for the different statistics tables
                    stats_notebook = ttk.Notebook(content_frame)
                    stats_notebook.pack(fill=tk.BOTH, expand=True)
                    
                    # Read summary statistics -- try Excel first, then CSV
                    summary_xlsx = os.path.join(result_path, "summary.xlsx")
                    summary_csv = os.path.join(result_path, "summary.csv")
                    summary_path = summary_xlsx if os.path.exists(summary_xlsx) else summary_csv
                    
                    if os.path.exists(summary_path):
                        try:
                            if summary_path.endswith('.xlsx'):
                                summary_df = pd.read_excel(summary_path)
                            else:
                                summary_df = pd.read_csv(summary_path)
                            
                            # Build the summary tab
                            summary_frame = ttk.Frame(stats_notebook)
                            stats_notebook.add(summary_frame, text="Summary statistics")
                            
                            # Convert the DataFrame into readable text
                            summary_text = ""
                            for col in summary_df.columns:
                                val = summary_df[col].iloc[0]
                                summary_text += f"{col}: {val}\n"
                            
                            # Show the summary text
                            summary_label = ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT)
                            summary_label.pack(padx=20, pady=20, anchor=tk.W)
                        except Exception as e:
                            ttk.Label(content_frame, text=f"Cannot load summary statistics: {str(e)}").pack(pady=20)
                    
                    # Load the detailed-data table
                    self._load_optimized_table(
                        stats_notebook, 
                        result_path,
                        "layer_info", 
                        "Lamina info"
                    )
                    
                    # Load the position-data table
                    self._load_optimized_table(
                        stats_notebook, 
                        result_path,
                        "position_info", 
                        "Position info"
                    )
                    
                    # Fallback when no data files are found
                    if (not os.path.exists(summary_path) 
                        and not os.path.exists(os.path.join(result_path, "layer_info.xlsx"))
                        and not os.path.exists(os.path.join(result_path, "layer_info.csv"))
                        and not os.path.exists(os.path.join(result_path, "position_info.xlsx"))
                        and not os.path.exists(os.path.join(result_path, "position_info.csv"))):
                        ttk.Label(content_frame, text="Statistics files not found").pack(pady=20)
                    
                    self.status_var.set(f"Loaded statistics for {selected_subdir}")
            
            # Bind the drop-down change event
            self.current_stats_var.trace("w", update_stats)
            
            # Show the first result initially
            if subdirs:
                update_stats()
            
            # Switch to the statistics tab
            self.tab_control.select(self.tab_stats)
            
        elif not self.detector or not hasattr(self.detector, 'layer_stats') or not self.detector.layer_stats:
            messagebox.showerror("Error", "Please analyze an image first")
            return
        else:
            # Single-image mode
            try:
                # Clear the tab contents
                for widget in self.tab_stats.winfo_children():
                    widget.destroy()
                
                # Build a Notebook for the different statistics tables
                stats_notebook = ttk.Notebook(self.tab_stats)
                stats_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Summary statistics tab
                summary_frame = ttk.Frame(stats_notebook)
                stats_notebook.add(summary_frame, text="Summary statistics")
                
                # Show the summary
                summary_data = self.detector.layer_stats["summary"]
                summary_text = "\n".join([f"{key}: {value}" for key, value in summary_data.items()])
                
                summary_label = ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT)
                summary_label.pack(padx=20, pady=20, anchor=tk.W)
                
                # Show the tables
                self._create_optimized_table(
                    stats_notebook, 
                    self.detector.layer_stats["detailed"],
                    "Detailed statistics",
                    self.save_path.get()
                )
                
                self._create_optimized_table(
                    stats_notebook, 
                    self.detector.layer_stats["position"],
                    "Position statistics",
                    self.save_path.get()
                )
                
                # Switch to the statistics tab
                self.tab_control.select(self.tab_stats)
                self.status_var.set("Showing statistics tables")
            
            except Exception as e:
                messagebox.showerror("Error", f"Cannot display statistics tables:\n{str(e)}")
                self.status_var.set("Failed to display statistics")
                print(f"Failed to display statistics: {str(e)}")
                import traceback
                traceback.print_exc()
    def _load_optimized_table(self, parent_notebook, result_path, base_filename, tab_title):
        """Optimized table loader that supports both Excel and CSV.

        Args:
            parent_notebook: parent Notebook widget
            result_path: directory containing the data file
            base_filename: base file name without extension
            tab_title: tab title
        """
        xlsx_path = os.path.join(result_path, f"{base_filename}.xlsx")
        csv_path = os.path.join(result_path, f"{base_filename}.csv")
        
        # Determine which file exists
        if os.path.exists(xlsx_path):
            file_path = xlsx_path
            is_excel = True
        elif os.path.exists(csv_path):
            file_path = csv_path
            is_excel = False
        else:
            return  # file does not exist
        
        try:
            # Build the tab
            table_frame = ttk.Frame(parent_notebook)
            parent_notebook.add(table_frame, text=tab_title)
            
            # Status and control area
            control_frame = ttk.Frame(table_frame)
            control_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
            
            # Status label
            status_var = tk.StringVar(self.root, value="Loading data...")
            status_label = ttk.Label(control_frame, textvariable=status_var)
            status_label.pack(side=tk.RIGHT)
            
            # Paging controls
            page_frame = ttk.Frame(control_frame)
            page_frame.pack(side=tk.LEFT)
            
            ttk.Label(page_frame, text="Page:").pack(side=tk.LEFT)
            current_page = tk.IntVar(self.root, value=1)
            page_spin = ttk.Spinbox(
                page_frame, from_=1, to=1, width=5, 
                textvariable=current_page, state="readonly"
            )
            page_spin.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(page_frame, text="/").pack(side=tk.LEFT)
            total_pages = tk.IntVar(self.root, value=1)
            total_label = ttk.Label(page_frame, textvariable=total_pages)
            total_label.pack(side=tk.LEFT, padx=5)
            
            # Filter frame
            filter_frame = ttk.Frame(control_frame)
            filter_frame.pack(side=tk.LEFT, padx=20)
            
            ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
            filter_var = tk.StringVar(self.root)
            filter_entry = ttk.Entry(filter_frame, textvariable=filter_var, width=15)
            filter_entry.pack(side=tk.LEFT, padx=5)
            
            # Build the table area
            table_container = ttk.Frame(table_frame)
            table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Paging parameters
            page_size = 200  # rows per page
            
            # Helper to load data
            def load_data():
                nonlocal is_excel
                try:
                    # Read 5 sample rows to discover the column layout
                    if is_excel:
                        xl = pd.ExcelFile(file_path)
                        sample_df = pd.read_excel(xl, nrows=5)
                    else:
                        sample_df = pd.read_csv(file_path, nrows=5)
                    
                    # Build the table
                    columns = list(sample_df.columns)
                    tree = ttk.Treeview(table_container, columns=columns, show='headings')
                    
                    # Configure column headings and widths
                    for col in columns:
                        tree.heading(col, text=col)
                        # Pick widths based on the column name
                        if "name" in col.lower() or "filename" in col.lower():
                            tree.column(col, width=150)
                        elif "position" in col.lower():
                             tree.column(col, width=100)
                        else:
                            tree.column(col, width=80)
                    
                    # Add scrollbars
                    vsb = ttk.Scrollbar(table_container, orient="vertical", command=tree.yview)
                    hsb = ttk.Scrollbar(table_container, orient="horizontal", command=tree.xview)
                    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
                    
                    # Lay out the table and scrollbars
                    tree.grid(column=0, row=0, sticky='nsew')
                    vsb.grid(column=1, row=0, sticky='ns')
                    hsb.grid(column=0, row=1, sticky='ew')
                    
                    table_container.grid_columnconfigure(0, weight=1)
                    table_container.grid_rowconfigure(0, weight=1)
                    
                    # Add the export button
                    btn_frame = ttk.Frame(table_frame)
                    btn_frame.pack(fill=tk.X, padx=5, pady=5)
                    
                    # Data loading and display logic
                    data_df = None
                    filtered_df = None
                    
                    # Load a specific page
                    def load_page_data():
                        nonlocal filtered_df
                        if filtered_df is None:
                            return
                            
                        # Clear existing rows
                        for item in tree.get_children():
                            tree.delete(item)
                        
                        # Compute the current page range
                        page = current_page.get()
                        start_idx = (page - 1) * page_size
                        end_idx = min(start_idx + page_size, len(filtered_df))
                        
                        # Fetch the current page
                        page_data = filtered_df.iloc[start_idx:end_idx]
                        
                        # Insert rows
                        for _, row in page_data.iterrows():
                            tree.insert('', tk.END, values=list(row))
                        
                        # Update status
                        status_var.set(f"Showing rows {start_idx+1}-{end_idx} (of {len(filtered_df)})")
                    
                    # Apply the filter
                    def apply_filter(*args):
                        nonlocal data_df, filtered_df
                        
                        # Lazy-load the full data
                        if data_df is None:
                            status_var.set("Loading the full data...")
                            parent_notebook.update()
                            
                            if is_excel:
                                data_df = pd.read_excel(file_path)
                            else:
                                data_df = pd.read_csv(file_path)
                        
                        # Run the filter
                        filter_text = filter_var.get().lower()
                        if filter_text:
                            # Search across all columns
                            mask = False
                            for col in data_df.columns:
                                # Convert each column to string before matching
                                mask = mask | data_df[col].astype(str).str.lower().str.contains(filter_text, na=False)
                            filtered_df = data_df[mask]
                        else:
                            filtered_df = data_df
                        
                        # Update the paging controls
                        max_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)
                        total_pages.set(max_pages)
                        page_spin.config(to=max_pages)
                        
                        # Reset to the first page
                        current_page.set(1)
                        
                        # Load the first page
                        load_page_data()
                    
                    # Reload when the page changes
                    current_page.trace("w", lambda *args: load_page_data())
                    filter_var.trace("w", lambda *args: apply_filter())
                    
                    # Export button
                    ttk.Button(
                        btn_frame, 
                        text="Export to Excel", 
                        command=lambda: self.export_to_excel(filtered_df if filtered_df is not None else data_df, result_path)
                    ).pack(side=tk.LEFT, padx=5)
                    
                    # Initial load
                    parent_notebook.after(100, apply_filter)
                    
                except Exception as e:
                    print(f"Error while loading table data: {str(e)}")
                    err_label = ttk.Label(table_container, text=f"Failed to load data: {str(e)}")
                    err_label.pack(pady=20)
            
            # Async data load
            parent_notebook.after(10, load_data)
            
        except Exception as e:
            print(f"Error while creating the table tab: {str(e)}")
            import traceback
            traceback.print_exc()
    def _create_optimized_table(self, parent_notebook, data_df, tab_title, save_path):
        """Build a table view for the DataFrame.

        Args:
            parent_notebook: parent Notebook widget
            data_df: the DataFrame to display
            tab_title: tab title
            save_path: save path used by the export button
        """
        if data_df is None or data_df.empty:
            return
            
        try:
            # Build the tab
            table_frame = ttk.Frame(parent_notebook)
            parent_notebook.add(table_frame, text=tab_title)
            
            # Status and control area
            control_frame = ttk.Frame(table_frame)
            control_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
            
            # Status label
            status_var = tk.StringVar(self.root, value="Loading data...")
            status_label = ttk.Label(control_frame, textvariable=status_var)
            status_label.pack(side=tk.RIGHT)
            
            # Paging controls
            page_frame = ttk.Frame(control_frame)
            page_frame.pack(side=tk.LEFT)
            
            ttk.Label(page_frame, text="Page:").pack(side=tk.LEFT)
            current_page = tk.IntVar(self.root, value=1)
            page_spin = ttk.Spinbox(
                page_frame, from_=1, to=1, width=5, 
                textvariable=current_page, state="readonly"
            )
            page_spin.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(page_frame, text="/").pack(side=tk.LEFT)
            total_pages = tk.IntVar(self.root, value=1)
            total_label = ttk.Label(page_frame, textvariable=total_pages)
            total_label.pack(side=tk.LEFT, padx=5)
            
            # Filter frame
            filter_frame = ttk.Frame(control_frame)
            filter_frame.pack(side=tk.LEFT, padx=20)
            
            ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
            filter_var = tk.StringVar(self.root)
            filter_entry = ttk.Entry(filter_frame, textvariable=filter_var, width=15)
            filter_entry.pack(side=tk.LEFT, padx=5)
            
            # Build the table area
            table_container = ttk.Frame(table_frame)
            table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Paging parameters
            page_size = 200  # rows per page
            
            # Build the table
            columns = list(data_df.columns)
            tree = ttk.Treeview(table_container, columns=columns, show='headings')
            
            # Configure column headings and widths
            for col in columns:
                tree.heading(col, text=col)
                # Pick widths based on the column name
                if "name" in col.lower() or "filename" in col.lower():
                    tree.column(col, width=150)
                elif "position" in col.lower():
                    tree.column(col, width=100)
                else:
                    tree.column(col, width=80)
                    
                    # Add scrollbars
                    vsb = ttk.Scrollbar(table_container, orient="vertical", command=tree.yview)
                    hsb = ttk.Scrollbar(table_container, orient="horizontal", command=tree.xview)
                    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
                    
                    # Lay out the table and scrollbars
                    tree.grid(column=0, row=0, sticky='nsew')
                    vsb.grid(column=1, row=0, sticky='ns')
                    hsb.grid(column=0, row=1, sticky='ew')
                    
            table_container.grid_columnconfigure(0, weight=1)
            table_container.grid_rowconfigure(0, weight=1)
                    
                    # Add the export button
            btn_frame = ttk.Frame(table_frame)
            btn_frame.pack(fill=tk.X, padx=5, pady=5)
            
            filtered_df = data_df.copy()
            
            # Load a specific page
            def load_page_data():
                # Clear existing rows
                for item in tree.get_children():
                    tree.delete(item)
                
                # Compute the current page range
                page = current_page.get()
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(filtered_df))
                
                # Fetch the current page
                page_data = filtered_df.iloc[start_idx:end_idx]
                
                # Insert rows
                for _, row in page_data.iterrows():
                    tree.insert('', tk.END, values=list(row))
                
                # Update status
                status_var.set(f"Showing rows {start_idx+1}-{end_idx} (of {len(filtered_df)})")
            
            # Apply the filter
            def apply_filter(*args):
                nonlocal filtered_df
                
                # Run the filter
                filter_text = filter_var.get().lower()
                if filter_text:
                    # Search across all columns
                    mask = False
                    for col in data_df.columns:
                        # Convert each column to string before matching
                        mask = mask | data_df[col].astype(str).str.lower().str.contains(filter_text, na=False)
                    filtered_df = data_df[mask]
                else:
                    filtered_df = data_df.copy()
                
                # Update the paging controls
                max_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)
                total_pages.set(max_pages)
                page_spin.config(to=max_pages)
                
                # Reset to the first page
                current_page.set(1)
                
                # Load the first page
                load_page_data()
            
            # Reload when the page changes
            current_page.trace("w", lambda *args: load_page_data())
            filter_var.trace("w", lambda *args: apply_filter())
            
            # Export button
            ttk.Button(
                btn_frame, 
                text="Export to Excel", 
                command=lambda: self.export_to_excel(filtered_df, save_path)
            ).pack(side=tk.LEFT, padx=5)
            
            # Initial load
            parent_notebook.after(100, apply_filter)
            
        except Exception as e:
            print(f"Error while creating the table tab: {str(e)}")
            import traceback
            traceback.print_exc()
    def save_all_results(self):
        """Save all analysis results."""
        if not self.detector or not hasattr(self.detector, 'layer_stats') or not self.detector.layer_stats:
            messagebox.showerror("Error", "Please analyze an image first")
            return
        
        save_dir = self.save_path.get()
        if not save_dir:
            messagebox.showerror("Error", "Please pick an output directory first")
            return
        
        try:
            self.status_var.set("Saving results...")
            self.root.update()
            
            # Export results
            self.detector.export_results(save_dir)
            
            messagebox.showinfo("Success", f"All results saved to:\n{save_dir}")
            self.status_var.set(f"Results saved to: {save_dir}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
            self.status_var.set("Failed to save results")
    def _load_image_from_path(self, path):
        """Load an image from a path (supports Unicode paths)."""
        if not path or not os.path.isfile(path):
            return None
        try:
            if any(ord(c) > 127 for c in path):
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(path)
            return img
        except Exception:
            return None
    def _apply_enhancement(self, image):
        """Apply brightness/contrast/gamma enhancement to the image."""
        img = image.copy().astype(np.float32)
        
        alpha = self.contrast.get()
        img = img * alpha
        
        beta = self.brightness.get()
        img = img + beta
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        gamma = self.gamma.get()
        if abs(gamma - 1.0) > 0.01:
            lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
            img = cv2.LUT(img, lut)
        
        return img
    def _preview_enhancement(self):
        """Preview the enhancement effect in the original-image tab."""
        image_path = self.image_path.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showinfo("Info", "Please select an image file or folder first")
            return
        
        # In batch mode (folder), use the first image as a preview.
        # Behaviour: search the folder itself first; if nothing is found, walk
        # into sub-folders and pick the first image there. Only complain when
        # neither location has any recognised image file.
        if os.path.isdir(image_path):
            valid_exts = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif')
            found = None
            try:
                # 1) Direct children of the chosen folder
                for f in sorted(os.listdir(image_path)):
                    full = os.path.join(image_path, f)
                    if os.path.isfile(full) and f.lower().endswith(valid_exts):
                        found = full
                        break

                # 2) Fall back to sub-folders
                if found is None:
                    for root_dir, _dirs, files in os.walk(image_path):
                        if root_dir == image_path:
                            continue  # already scanned above
                        for f in sorted(files):
                            if f.lower().endswith(valid_exts):
                                candidate = os.path.join(root_dir, f)
                                if os.path.isfile(candidate):
                                    found = candidate
                                    break
                        if found:
                            break
            except Exception as e:
                messagebox.showerror("Error", f"Cannot read directory:\n{e}")
                return

            if found is None:
                messagebox.showinfo(
                    "Info",
                    "No recognised image files were found in the selected folder or "
                    "its sub-folders.\n\nPlease check that the path is correct and "
                    "that the file extension matches the expected formats "
                    "(.bmp / .jpg / .jpeg / .png / .tiff / .tif).",
                )
                return
            image_path = found
        
        if not os.path.isfile(image_path):
            messagebox.showinfo("Info", "Please select an image file first")
            return
        
        src = self._load_image_from_path(image_path)
        if src is None:
            messagebox.showerror("Error", f"Cannot load image: {image_path}")
            return
        
        enhanced = self._apply_enhancement(src)
        
        # Display the preview in the original-image tab
        self.tab_control.select(self.tab_original)
        
        for widget in self.tab_original.winfo_children():
            widget.destroy()
        
        ttk.Label(self.tab_original, text='Enhancement preview (click "Analyze image" when satisfied)',
                  font=("Segoe UI", 9, "bold"), foreground="#CC6600").pack(anchor=tk.W, padx=5, pady=2)
        
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB) if len(enhanced.shape) == 3 else enhanced
        
        h, w = img_rgb.shape[:2]
        max_h, max_w = 500, 800
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1:
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
        
        pil_img = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(pil_img)
        
        canvas = tk.Canvas(self.tab_original, width=img_tk.width(), height=img_tk.height())
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk
    def _reset_enhancement(self):
        """Reset image-enhancement parameters to defaults."""
        self.brightness.set(0)
        self.contrast.set(1.0)
        self.gamma.set(1.0)
