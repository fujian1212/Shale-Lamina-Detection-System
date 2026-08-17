#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export and scale calibration."""

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

class ExportUiMixin:
    def start_scale_calibration(self):
        """Open the scale-calibration window (works in single and batch modes)."""
        # Single-image mode already has an image loaded; batch mode picks a reference image
        ref_image = self.image
        ref_loaded_temporarily = False

        if ref_image is None:
            ref_path = None
            current_path = self.image_path.get().strip() if hasattr(self, 'image_path') else ""

            # 1) If the input path is a folder, use the first image inside as the reference
            if current_path and os.path.isdir(current_path):
                for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                            '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'):
                    candidates = [f for f in os.listdir(current_path) if f.endswith(ext)]
                    if candidates:
                        ref_path = os.path.join(current_path, sorted(candidates)[0])
                        break
            # 2) If the input is a single file
            elif current_path and os.path.isfile(current_path):
                ref_path = current_path
            # 3) Fall back to the most recent batch folder
            elif getattr(self, 'last_batch_folder', None) and os.path.isdir(self.last_batch_folder):
                for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                            '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'):
                    candidates = [f for f in os.listdir(self.last_batch_folder) if f.endswith(ext)]
                    if candidates:
                        ref_path = os.path.join(self.last_batch_folder, sorted(candidates)[0])
                        break
            # 4) Last resort: ask the user to pick an image manually
            if not ref_path:
                ref_path = filedialog.askopenfilename(
                    title="Pick an image as the scale-calibration reference",
                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                               ("All files", "*.*")]
                )
                if not ref_path:
                    return

            try:
                ref_image = self._load_image_from_path(ref_path)
                if ref_image is None:
                    messagebox.showerror("Error", f"Failed to load image: {ref_path}")
                    return
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                return

            # Temporarily store as self.image so the existing pipeline can be reused; restore on close
            self.image = ref_image
            ref_loaded_temporarily = True
            print(f"[Calibration] Using reference image: {ref_path}")

        calib_win = tk.Toplevel(self.root)
        calib_win.title("Scale calibration" + (" (batch mode reference image)" if ref_loaded_temporarily else ""))
        calib_win.geometry("900x700")
        calib_win.transient(self.root)
        calib_win.grab_set()

        # If a reference image was loaded temporarily, restore self.image=None on close (keep pixel_per_mm result)
        if ref_loaded_temporarily:
            def _on_close_calib():
                self.image = None
                calib_win.destroy()
            calib_win.protocol("WM_DELETE_WINDOW", _on_close_calib)

        # State
        points = []  # calibration points [(x,y), (x,y)]
        point_ids = []  # canvas marker ids
        line_id = [None]  # connecting line

        # Top instructions
        info_frame = ttk.Frame(calib_win, padding=10)
        info_frame.pack(fill=tk.X)

        step_var = tk.StringVar(value="Step 1/3: click the first calibration point on the image")
        ttk.Label(info_frame, textvariable=step_var,
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(info_frame,
                  text="Tip: pick two points whose true distance is known (e.g. ends of the scale bar or core edges)",
                  font=("Segoe UI", 9), foreground="gray").pack(anchor=tk.W, pady=(2, 0))

        # Image display area
        canvas_frame = ttk.Frame(calib_win)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(canvas_frame, bg='gray20', cursor='crosshair')
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Resize for display
        img_h, img_w = self.image.shape[:2]
        max_display = 800
        scale = min(max_display / img_w, max_display / img_h, 1.0)
        display_w = int(img_w * scale)
        display_h = int(img_h * scale)

        if len(self.image.shape) == 2:
            pil_img = Image.fromarray(self.image)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

        pil_img_resized = pil_img.resize((display_w, display_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img_resized)

        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.configure(scrollregion=(0, 0, display_w, display_h))
        canvas._img_ref = tk_img  # keep a reference to avoid GC

        # Bottom input area
        bottom_frame = ttk.Frame(calib_win, padding=10)
        bottom_frame.pack(fill=tk.X)

        dist_frame = ttk.Frame(bottom_frame)
        dist_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dist_frame, text="Actual distance between points:").pack(side=tk.LEFT)
        dist_var = tk.StringVar()
        dist_entry = ttk.Entry(dist_frame, textvariable=dist_var, width=12, state='disabled')
        dist_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(dist_frame, text="mm").pack(side=tk.LEFT)

        pixel_info_var = tk.StringVar(value="Pixel distance: --")
        ttk.Label(dist_frame, textvariable=pixel_info_var, foreground="blue").pack(side=tk.LEFT, padx=20)

        btn_frame_cal = ttk.Frame(bottom_frame)
        btn_frame_cal.pack(fill=tk.X, pady=5)

        def on_canvas_click(event):
            cx = canvas.canvasx(event.x)
            cy = canvas.canvasy(event.y)

            # Convert back to original-image coordinates
            orig_x = cx / scale
            orig_y = cy / scale

            if len(points) >= 2:
                return

            points.append((orig_x, orig_y))

            r = 5
            pid = canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill='red', outline='yellow', width=2)
            canvas.create_text(cx + 12, cy - 12,
                             text=f"P{len(points)}", fill='yellow',
                             font=("Segoe UI", 10, "bold"))
            point_ids.append(pid)

            if len(points) == 1:
                step_var.set("Step 2/3: click the second calibration point on the image")
            elif len(points) == 2:
                # Draw the connecting line
                p1_cx = points[0][0] * scale
                p1_cy = points[0][1] * scale
                p2_cx = points[1][0] * scale
                p2_cy = points[1][1] * scale
                line_id[0] = canvas.create_line(p1_cx, p1_cy, p2_cx, p2_cy,
                                                fill='lime', width=2, dash=(4, 4))

                # Compute the pixel distance
                dx = points[1][0] - points[0][0]
                dy = points[1][1] - points[0][1]
                pixel_dist = (dx**2 + dy**2) ** 0.5
                pixel_info_var.set(f"Pixel distance: {pixel_dist:.1f} px")

                step_var.set("Step 3/3: enter the actual distance in millimetres and click Confirm")
                dist_entry.configure(state='normal')
                dist_entry.focus_set()

        canvas.bind("<Button-1>", on_canvas_click)

        def reset_points():
            points.clear()
            for pid in point_ids:
                canvas.delete(pid)
            point_ids.clear()
            if line_id[0]:
                canvas.delete(line_id[0])
                line_id[0] = None
            # Clear all overlay items (point labels etc.)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

            dist_var.set("")
            dist_entry.configure(state='disabled')
            pixel_info_var.set("Pixel distance: --")
            step_var.set("Step 1/3: click the first calibration point on the image")

        def confirm_calibration():
            if len(points) < 2:
                messagebox.showwarning("Info", "Please pick two calibration points on the image first", parent=calib_win)
                return

            try:
                actual_dist = float(dist_var.get())
            except (ValueError, TypeError):
                messagebox.showwarning("Info", "Please enter a valid actual distance (number)", parent=calib_win)
                return

            if actual_dist <= 0:
                messagebox.showwarning("Info", "Actual distance must be greater than zero", parent=calib_win)
                return

            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            pixel_dist = (dx**2 + dy**2) ** 0.5

            if pixel_dist < 1:
                messagebox.showwarning("Info", "The two points are too close together; please pick again", parent=calib_win)
                return

            self.pixel_per_mm = pixel_dist / actual_dist

            self.scale_status_var.set(
                f"Scale: {self.pixel_per_mm:.2f} px/mm ({actual_dist:.1f} mm = {pixel_dist:.0f} px)"
            )

            # Push to the detector
            if self.detector:
                self.detector.pixel_per_mm = self.pixel_per_mm

            messagebox.showinfo("Calibration complete",
                f"Scale calibration succeeded!\n\n"
                f"Pixel distance: {pixel_dist:.1f} px\n"
                f"Actual distance: {actual_dist:.1f} mm\n"
                f"Scale ratio: {self.pixel_per_mm:.2f} px/mm\n\n"
                f"Subsequent thickness measurements will use this scale.",
                parent=calib_win)

            if ref_loaded_temporarily:
                self.image = None
            calib_win.destroy()

        def cancel_calibration():
            if ref_loaded_temporarily:
                self.image = None
            calib_win.destroy()

        ttk.Button(btn_frame_cal, text="Pick again", command=reset_points).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_cal, text="Confirm calibration", command=confirm_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_cal, text="Cancel", command=cancel_calibration).pack(side=tk.RIGHT, padx=5)

        # Show existing calibration if available
        if self.pixel_per_mm is not None:
            ttk.Label(btn_frame_cal, text=f"(current: {self.pixel_per_mm:.2f} px/mm)",
                     foreground="green").pack(side=tk.LEFT, padx=10)
    def export_paper_figures(self):
        """Export the full set of paper figures and data.

        - Single-image mode: export for the current detector
        - Batch mode (after batch processing completes): ask the user, then export per image
        """
        # Check whether we are in batch mode with prior results
        is_batch_ready = (
            self.last_batch_folder is not None
            and self.last_batch_output_dir is not None
            and self.last_batch_image_files
            and os.path.isdir(self.last_batch_folder)
        )

        if is_batch_ready:
            choice = messagebox.askyesnocancel(
                "Export paper figures",
                f"Batch mode detected ({len(self.last_batch_image_files)} image(s)).\n\n"
                "How do you want to export?\n"
                "  Yes - export the full paper figure set per image\n"
                "  No  - export only the currently selected image\n"
                "  Cancel - do not export"
            )
            if choice is None:
                return
            if choice:
                self._export_paper_figures_batch()
                return

        # Single-image mode
        if not self.detector or not self.detector.layers:
            messagebox.showerror("Error", "Please analyze an image before exporting paper figures")
            return

        export_dir = filedialog.askdirectory(title="Choose the paper-figure export directory")
        if not export_dir:
            return

        include_sensitivity, include_ablation = self._ask_paper_export_options()

        try:
            self.status_var.set("Exporting paper figures and data...")
            self.root.update()

            start_depth = None
            end_depth = None
            if self.enable_depth_range.get():
                start_depth = self.start_depth.get()
                end_depth = self.end_depth.get()

            paper_dir = self.detector.export_paper_figures(
                export_dir,
                start_depth=start_depth,
                end_depth=end_depth,
                include_sensitivity=include_sensitivity,
                include_ablation=include_ablation,
            )

            self.status_var.set("Paper figures exported")

            result_msg = "Full paper figure set exported successfully!\n\n"
            result_msg += f"Export directory: {paper_dir}\n\n"
            result_msg += "00_input/ - original/ROI/scaled image, sample metadata\n"
            result_msg += "01_preprocessing/ - gray/denoise/CLAHE/Canny/Hough/before-after correction\n"
            result_msg += "02_scanline_detection/ - main scan lines / validation lines / gradient curves / candidates\n"
            result_msg += "03_crossline_validation/ - before/after validation, rejected noise, consistency illustration\n"
            result_msg += "04_results/ - auto-detection figure, boundary-strength heatmap, lamina attribute table, classification stats\n"
            result_msg += "05_method_comparison/ - Sobel / Canny / rule-based / proposed\n"
            result_msg += "parameters/ - scale_calibration.json + preprocessing parameters JSON\n"
            if include_sensitivity:
                result_msg += "06_sensitivity/ - sensitivity curves for 4 parameters + CSV\n"
            if include_ablation:
                result_msg += "07_ablation/ - ablation bar charts + CSV\n"
            result_msg += "\nfigures/ - high-resolution analysis figures\n"
            result_msg += "data/ - all CSV + Excel data files\n\n"
            result_msg += "All figures are high-resolution PNG. Each sub-directory's _README_metrics.txt explains the metrics."

            messagebox.showinfo("Export complete", result_msg)

            try:
                os.startfile(paper_dir)
            except Exception:
                pass

        except Exception as e:
            error_msg = f"Error while exporting paper figures:\n{str(e)}\n{traceback.format_exc()}"
            messagebox.showerror("Export error", error_msg)
            self.status_var.set("Paper-figure export failed")
    def _ask_paper_export_options(self):
        """Ask whether to also run parameter sensitivity / ablation studies.

        Returns:
            (include_sensitivity, include_ablation)
        """
        dlg = tk.Toplevel(self.root)
        dlg.title("Paper export options")
        dlg.geometry("520x300")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(
            dlg,
            text="Pick the optional analyses to include in this export:",
            font=self.default_font,
        ).pack(padx=20, pady=(15, 5), anchor="w")

        sens_var = tk.BooleanVar(dlg, value=False)
        abl_var = tk.BooleanVar(dlg, value=False)

        ttk.Checkbutton(
            dlg,
            text="Parameter sensitivity analysis (re-runs 19 detections, takes several minutes)",
            variable=sens_var,
        ).pack(padx=30, pady=4, anchor="w")
        ttk.Checkbutton(
            dlg,
            text="Ablation study (re-runs 4 detections, takes 1-2 minutes)",
            variable=abl_var,
        ).pack(padx=30, pady=4, anchor="w")

        info = (
            "Note: optional analyses re-run detection on the same image using the same default parameters.\n"
            "match_ratio / n_matched in the results are measured against the current default-parameter\n"
            "detection, not human annotations -- mark this clearly when used in a paper."
        )
        ttk.Label(dlg, text=info, foreground="#555", justify="left",
                  wraplength=480, font=self.default_font).pack(padx=20, pady=10, anchor="w")

        result = {"ok": False}

        def _ok():
            result["ok"] = True
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="Start export", command=_ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Skip optional analyses", command=_cancel).pack(side=tk.LEFT, padx=10)

        dlg.wait_window()

        if not result["ok"]:
            return False, False
        return sens_var.get(), abl_var.get()
    def _export_paper_figures_batch(self):
        """Export the paper figure set for every image in a batch (parallel processes)."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        folder_path = self.last_batch_folder
        output_dir = self.last_batch_output_dir
        image_files = self.last_batch_image_files

        # Progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch paper-figure export")
        progress_window.geometry("500x200")
        progress_window.transient(self.root)
        progress_window.grab_set()

        progress_label = ttk.Label(progress_window, text=f"Preparing to export paper figures for {len(image_files)} image(s)...")
        progress_label.pack(pady=10)

        progress_var = tk.DoubleVar(progress_window)
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)

        status_label = ttk.Label(progress_window, text="")
        status_label.pack(pady=5)

        cancel_button = ttk.Button(progress_window, text="Cancel", command=progress_window.destroy)
        cancel_button.pack(pady=10)

        self.root.update()

        # Gather parameters (same as batch processing)
        threshold_method = self.threshold_method.get()
        min_layer_width = self.min_layer_width.get()
        blur_size = self.blur_size.get()
        clahe_clip = self.clahe_clip.get()
        clahe_grid = (self.clahe_grid_x.get(), self.clahe_grid_y.get())
        scan_line_count = self.scan_line_count.get()
        min_validation_lines = self.min_validation_lines.get()
        align_core = self.align_core.get()
        alignment_angle = self.alignment_angle.get()

        # Depth range (split evenly across images when enabled)
        enable_depth = self.enable_depth_range.get()
        global_start = self.start_depth.get() if enable_depth else None
        global_end = self.end_depth.get() if enable_depth else None

        n = len(image_files)
        task_args_list = []
        for i, image_file in enumerate(image_files):
            image_name = os.path.splitext(image_file)[0]
            image_output_dir = os.path.join(output_dir, image_name)

            # Split the depth range evenly across images
            if enable_depth and global_start is not None and global_end is not None and global_start != global_end:
                seg = (global_end - global_start) / n
                img_start = global_start + i * seg
                img_end = global_start + (i + 1) * seg
            else:
                img_start = None
                img_end = None

            task_args_list.append({
                "image_file": image_file,
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
                "batch_scan_lines": self.batch_scan_lines,
                "start_depth": img_start,
                "end_depth": img_end,
            })

        cpu_count = os.cpu_count() or 1
        max_workers = max(1, min(4, cpu_count // 2))
        print(f"Batch paper export: CPU={cpu_count}, using {max_workers} parallel workers")

        success_count = 0
        failed_files = []
        cancelled = False

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_args = {
                    executor.submit(_paper_export_worker, args): args
                    for args in task_args_list
                }

                completed = 0
                for future in as_completed(future_to_args):
                    if not progress_window.winfo_exists():
                        cancelled = True
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    completed += 1
                    try:
                        success, image_file, paper_dir, err_msg = future.result()
                    except Exception as e:
                        success = False
                        image_file = future_to_args[future].get("image_file", "?")
                        err_msg = str(e)

                    progress_var.set((completed / n) * 100)
                    if success:
                        success_count += 1
                        status_label.config(text=f"Completed {completed}/{n}: {image_file}")
                    else:
                        failed_files.append((image_file, err_msg))
                        status_label.config(text=f"Image {image_file} failed: {err_msg[:60]}")
                        print(f"[Paper export] {image_file} failed: {err_msg}")
                    progress_window.update()
        except Exception as pool_err:
            print(f"Process pool error, falling back to single-thread: {pool_err}")
            for args in task_args_list:
                if not progress_window.winfo_exists():
                    cancelled = True
                    break
                success, image_file, paper_dir, err_msg = _paper_export_worker(args)
                if success:
                    success_count += 1
                else:
                    failed_files.append((image_file, err_msg))
                progress_var.set((len(failed_files) + success_count) / n * 100)
                progress_window.update()

        if progress_window.winfo_exists():
            progress_window.destroy()

        if cancelled:
            self.status_var.set("Batch paper export cancelled by user")
            return

        self.status_var.set(f"Batch paper export complete: {success_count}/{n} succeeded")

        msg = f"Batch paper-figure export complete!\n\n"
        msg += f"Success: {success_count}/{n} image(s)\n"
        msg += f"Output: each image's sub-directory paper_export/ folder\n"
        msg += f"  e.g.: {output_dir}/<image name>/paper_export/\n\n"
        if failed_files:
            msg += f"Failures ({len(failed_files)}):\n"
            for fname, err in failed_files[:5]:
                msg += f"  - {fname}: {err[:60]}\n"
            if len(failed_files) > 5:
                msg += f"  ... and {len(failed_files) - 5} more\n"
        messagebox.showinfo("Export complete", msg)

        try:
            os.startfile(output_dir)
        except Exception:
            pass
    def export_to_excel(self, df, output_dir):
        """Export a DataFrame to an Excel file."""
        try:
            # Ask for a save path
            file_path = filedialog.asksaveasfilename(
                initialdir=output_dir,
                title="Save Excel file",
                filetypes=[("Excel files", "*.xlsx")],
                defaultextension=".xlsx"
            )

            if file_path:
                # Column-name mapping (English internal -> friendlier English headers)
                column_mapping = {
                    "scan_line": "scan_line",
                    "position_x": "position_x_px",
                    "position_y": "position_y_px",
                    "spacing_to_next": "spacing_to_next_px",
                    "layer_index": "lamina_index",
                    "strength": "strength",
                    "count": "count",
                    "avg_spacing": "avg_spacing_px",
                    "density": "density_per_100px",
                    "total_count": "total_count",
                    "avg_density": "avg_density_per_100px",
                    "filename": "filename",
                    "image_index": "image_index",
                    "cumulative_offset": "cumulative_offset",
                    "position": "position_px",
                    "adjusted_position": "adjusted_position_px",
                    "depth_m": "depth_m",
                }

                # Copy the DataFrame so we don't mutate the original
                df_copy = df.copy()

                # Apply the column mapping
                renamed_columns = {}
                for col in df_copy.columns:
                    if col in column_mapping:
                        renamed_columns[col] = column_mapping[col]
                    else:
                        renamed_columns[col] = col

                df_copy.rename(columns=renamed_columns, inplace=True)

                # Export
                df_copy.to_excel(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported successfully to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Error while exporting Excel:\n{str(e)}")
    def _process_depth_interpolation(self, stats, detailed_df, position_df, start_depth, end_depth):
        """Interpolate depths from pixel positions.

        Args:
            stats: original summary statistics
            detailed_df: detailed lamina DataFrame
            position_df: position-statistics DataFrame
            start_depth: start depth (m)
            end_depth: end depth (m)

        Returns:
            dict: statistics augmented with interpolated depths
        """
        try:
            import pandas as pd
            import numpy as np

            # Important: position data is based on the detector's working image, which is downscaled
            if not self.detector or not hasattr(self.detector, 'width'):
                print("Error: detector image width is unavailable")
                return None

            # Detector image width
            image_width = self.detector.width

            # Original image width (for comparison only)
            original_image_width = None
            if hasattr(self, 'image_path') and self.image_path.get():
                try:
                    from PIL import Image, ImageOps
                    temp_img = Image.open(self.image_path.get())
                    temp_img = ImageOps.exif_transpose(temp_img)
                    original_image_width = temp_img.width
                    temp_img = None  # release memory
                except Exception as e:
                    print(f"Cannot read the original image width: {str(e)}")
            depth_range = end_depth - start_depth

            # Show the scale ratio when original width is available
            if original_image_width:
                scale_ratio = image_width / original_image_width

            # Depth mapping must span the full image width so it covers the user-specified range
            # Left edge -> start depth, right edge -> end depth
            depth_per_pixel = depth_range / image_width



            # Process the detailed lamina DataFrame
            if detailed_df is not None and not detailed_df.empty:
                # Add a depth column
                x_column = None
                possible_x_columns = ['position_x', 'position_x_px', 'x_position', 'x']

                for col in possible_x_columns:
                    if col in detailed_df.columns:
                        x_column = col
                        break

                if x_column:
                    detailed_df = detailed_df.copy()
                    # Simple depth calc: linear pixel-to-depth mapping
                    detailed_df['depth_m'] = start_depth + (detailed_df[x_column] * depth_per_pixel)
                    detailed_df['depth_m'] = detailed_df['depth_m'].round(3)

            # Process the position-statistics DataFrame
            if position_df is not None and not position_df.empty:
                position_df = position_df.copy()

                # Find the position column
                position_column = None
                possible_position_columns = ['position', 'position_px', 'position_x', 'x_position', 'x']

                for col in possible_position_columns:
                    if col in position_df.columns:
                        position_column = col
                        break

                if position_column:
                    # Simple depth calc: linear pixel-to-depth mapping
                    position_df['depth_m'] = start_depth + (position_df[position_column] * depth_per_pixel)
                    position_df['depth_m'] = position_df['depth_m'].round(3)

                    # Make sure depth_m is the first column
                    columns = list(position_df.columns)
                    if 'depth_m' in columns:
                        columns.remove('depth_m')
                        columns.insert(0, 'depth_m')
                        position_df = position_df[columns]

                    print(f"Position-statistics columns: {list(position_df.columns)}")
                    if 'depth_m' in position_df.columns:
                        print(f"Depth range: {position_df['depth_m'].min():.3f} - {position_df['depth_m'].max():.3f} m")
                else:
                    print(f"Warning: no position column found; available columns: {list(position_df.columns)}")

            # Update the summary statistics
            updated_stats = stats.copy() if stats else {}
            updated_stats['start_depth_m'] = start_depth
            updated_stats['end_depth_m'] = end_depth
            updated_stats['depth_range_m'] = depth_range
            updated_stats['depth_resolution_m_per_px'] = round(depth_per_pixel, 6)
            updated_stats['image_width_px'] = image_width
            updated_stats['mapping_range'] = f"0 - {image_width} px -> {start_depth} - {end_depth} m"

            # Compute depth statistics if lamina data exists
            if detailed_df is not None and not detailed_df.empty and 'depth_m' in detailed_df.columns:
                actual_min_depth = detailed_df['depth_m'].min()
                actual_max_depth = detailed_df['depth_m'].max()

                depth_stats = {
                    'shallowest_lamina_depth_m': actual_min_depth,
                    'deepest_lamina_depth_m': actual_max_depth,
                    'mean_lamina_depth_m': detailed_df['depth_m'].mean().round(3),
                    'lamina_depth_std_m': detailed_df['depth_m'].std().round(3),
                }
                updated_stats.update(depth_stats)

                # Depth coverage (informational)
                depth_coverage = (actual_max_depth - actual_min_depth) / depth_range * 100



            # Return the augmented statistics
            return {
                'summary': updated_stats,
                'detailed': detailed_df,
                'position': position_df
            }

        except Exception as e:
            return None
    def _export_continuous_stats(self, file_path, output_dir):
        """Export the continuous-depth statistics file."""
        try:
            import shutil
            from tkinter import filedialog

            # Ask the user for a save path
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save continuous-depth statistics"
            )

            if save_path:
                shutil.copy2(file_path, save_path)
                messagebox.showinfo("Success", f"File saved to: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error while exporting file: {str(e)}")
