#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UI layout and file browsing."""

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

class UiSetupMixin:
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("Rock Core Lamina Identification System")
        self.root.geometry("1280x800")  # default window size

        # Settings (pass the root window as the master)
        self.image_path = tk.StringVar(root)
        self.save_path = tk.StringVar(root, value="output")
        self.threshold_method = tk.StringVar(root, value="otsu")
        self.min_layer_width = tk.IntVar(root, value=5)
        self.blur_size = tk.IntVar(root, value=5)
        self.clahe_clip = tk.DoubleVar(root, value=2.0)
        self.clahe_grid_x = tk.IntVar(root, value=8)
        self.clahe_grid_y = tk.IntVar(root, value=8)
        self.scan_line_count = tk.IntVar(root, value=5)
        self.min_validation_lines = tk.IntVar(root, value=3)
        self.align_core = tk.BooleanVar(root, value=True)
        self.alignment_angle = tk.DoubleVar(root, value=0.0)
        # Manual lamina-direction override (single-image, fractured cores)
        self.use_manual_lamina_angle = tk.BooleanVar(root, value=False)
        self.manual_lamina_angle = tk.DoubleVar(root, value=0.0)
        # Minimum fraction of scan lines a lamina must span to be accepted (%)
        self.min_support_pct = tk.IntVar(root, value=70)

        # Depth-range settings
        self.enable_depth_range = tk.BooleanVar(root, value=False)
        self.start_depth = tk.DoubleVar(root, value=0.0)
        self.end_depth = tk.DoubleVar(root, value=100.0)

        # Batch-processing settings
        self.batch_mode = tk.BooleanVar(root, value=False)
        self.image_ext = tk.StringVar(root, value=".jpg")
        self.file_count_var = tk.StringVar(root, value="No file loaded")
        self.include_subfolders = tk.BooleanVar(root, value=False)
        # When enabled, the batch flow runs per-image sensitivity + ablation
        # tests after the normal merge step finishes and aggregates the
        # results across the batch. See rock_core_analyzer/batch/batch_sensitivity.py.
        self.batch_run_sensitivity = tk.BooleanVar(root, value=False)

        # Scan-line mode
        self.scan_mode = tk.StringVar(root, value="auto")

        # Batch-mode state
        self.batch_image_files = []  # batch file list
        self.batch_scan_lines = None  # scan lines selected for batch
        self.last_batch_folder = None  # input folder of last batch
        self.last_batch_output_dir = None  # output directory of last batch
        self.last_batch_image_files = []  # image files of last batch

        # Image and scan-line state
        self.image = None
        self.detector = None
        self.scan_lines = []
        self.scan_line_ids = []
        self.custom_scan_lines = []  # custom scan-line positions

        # Pick the right UI font
        self.default_font = self.get_chinese_font()

        # Status variables
        self.status_var = tk.StringVar(root, value="Initializing UI...")
        self.progress_var = tk.DoubleVar(root, value=0.0)  # progress variable

        # Processing flag to avoid duplicate clicks
        self.is_processing = False

        # Scale-calibration parameters
        self.pixel_per_mm = None  # pixels per millimetre; None = not calibrated

        # Image-enhancement parameters
        self.brightness = tk.IntVar(root, value=0)       # -100 ~ +100
        self.contrast = tk.DoubleVar(root, value=1.0)    # 0.1 ~ 3.0
        self.gamma = tk.DoubleVar(root, value=1.0)       # 0.1 ~ 3.0

        # Build UI
        self.setup_ui()
        self.status_var.set("Ready")
    def _check_optimization_flag(self, flag_name):
        """Read an optimization flag from environment variables."""
        return os.environ.get(flag_name, '0') == '1'
    def get_chinese_font(self):
        """Pick a UI font appropriate for the current platform."""
        if platform.system() == 'Windows':
            return ('Segoe UI', 10)
        elif platform.system() == 'Darwin':  # macOS
            return ('Helvetica', 10)
        else:  # Linux / others
            return ('DejaVu Sans', 10)
    def apply_font_to_widget(self, widget):
        """Apply the UI font to a widget and its children."""
        if hasattr(widget, 'configure'):
            try:
                widget.configure(font=self.default_font)
            except:
                pass  # ignore widgets that do not accept font config

        # Recurse over children
        for child in widget.winfo_children():
            self.apply_font_to_widget(child)
    def setup_ui(self):
        """Build the UI."""
        # If lazy-loading already created main_frame, reuse it
        if not hasattr(self, 'main_frame'):
            # Window title and size
            self.root.title("Rock Core Lamina Identification System")
            self.root.geometry("1280x800")  # default window size

            # Theme styles
            style = ttk.Style()
            style.configure('TLabelframe', borderwidth=2)
            style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
            style.configure('TButton', font=self.default_font)

            # Main frame
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Title and control panel
            title_label = ttk.Label(self.main_frame, text="Rock Core Lamina Identification System",
                                    font=("Segoe UI", 16, "bold"))
            title_label.pack(pady=(5, 15))

        # Status variables
        if not hasattr(self, 'status_var'):
            self.status_var = tk.StringVar(self.root, value="Loading UI...")
            status_frame = ttk.Frame(self.root)
            status_frame.pack(side=tk.BOTTOM, fill=tk.X)
            ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

            # Progress bar in the status bar
            self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100, length=200)
            self.progress_bar.pack(side=tk.RIGHT, padx=10)

        # Left control panel
        control_outer_frame = ttk.Frame(self.main_frame, width=400)
        control_outer_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Vertical scrollbar for the control panel
        control_vscroll = ttk.Scrollbar(control_outer_frame, orient="vertical")
        control_vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas for the control panel (no focus border)
        control_canvas = tk.Canvas(control_outer_frame, yscrollcommand=control_vscroll.set,
                                 highlightthickness=0, bd=0)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Wire the scrollbar to the canvas
        control_vscroll.config(command=control_canvas.yview)

        # Build the control-panel frame inside the canvas
        control_frame = ttk.Frame(control_canvas)
        control_frame_window = control_canvas.create_window((0, 0), window=control_frame, anchor=tk.NW)

        # Parameter-settings frame -- improved padding and style
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding=(12, 8))
        param_frame.pack(fill=tk.X, pady=(0, 12), padx=5)

        # Mode tabs
        mode_notebook = ttk.Notebook(param_frame)
        mode_notebook.pack(fill=tk.X, pady=(4, 4), padx=2)

        # Single-image and batch-mode tabs
        single_mode_frame = ttk.Frame(mode_notebook)
        batch_mode_frame = ttk.Frame(mode_notebook)

        # Add tabs
        mode_notebook.add(single_mode_frame, text="Single image")
        mode_notebook.add(batch_mode_frame, text="Batch mode")

        # Default to single-image mode
        mode_notebook.select(0)
        self.batch_mode.set(False)

        # Handle tab switching
        def on_tab_change(event):
            selected_tab = event.widget.select()
            tab_text = event.widget.tab(selected_tab, "text")
            if tab_text == "Batch mode":
                self.batch_mode.set(True)
            else:
                self.batch_mode.set(False)
            self.toggle_batch_mode()

        mode_notebook.bind("<<NotebookTabChanged>>", on_tab_change)

        # ========== Single-image tab ==========
        # Image-path control
        path_frame = ttk.Frame(single_mode_frame)
        path_frame.pack(fill=tk.X, pady=4)
        ttk.Label(path_frame, text="Image path:").pack(side=tk.LEFT)
        ttk.Entry(path_frame, textvariable=self.image_path, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(path_frame, text="Browse...", command=self.browse_image).pack(side=tk.LEFT)

        # Buttons for manual scan-line picking
        scan_btn_frame = ttk.Frame(single_mode_frame)
        scan_btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(scan_btn_frame, text="Pick scan lines", command=self.select_scan_lines).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(scan_btn_frame, text="Clear scan lines", command=self.clear_scan_lines).pack(side=tk.LEFT)
        # Save a reference to the scan-button frame
        self.scan_btn_frame = scan_btn_frame

        # ========== Batch tab ==========
        # Folder selection
        folder_frame = ttk.Frame(batch_mode_frame)
        folder_frame.pack(fill=tk.X, pady=4)
        ttk.Label(folder_frame, text="Image folder:").pack(side=tk.LEFT)
        ttk.Entry(folder_frame, textvariable=self.image_path, width=25).pack(side=tk.LEFT, padx=5)
        self.btn_browse_folder = ttk.Button(folder_frame, text="Browse folder", command=self.browse_folder)
        self.btn_browse_folder.pack(side=tk.LEFT)

        # Batch-processing buttons
        batch_btns_frame = ttk.Frame(batch_mode_frame)
        batch_btns_frame.pack(fill=tk.X, pady=4)
        self.file_list_btn = ttk.Button(batch_btns_frame, text="Load file list", command=self.load_file_list)
        self.file_list_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.load_list_file_btn = ttk.Button(batch_btns_frame, text="Load list from file", command=self.load_list_from_file)
        self.load_list_file_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.batch_scan_btn = ttk.Button(batch_btns_frame, text="Pick batch scan lines",
                                          command=self.select_batch_scan_lines_ui, state=tk.DISABLED)
        self.batch_scan_btn.pack(side=tk.LEFT)

        # Export-list button
        batch_export_frame = ttk.Frame(batch_mode_frame)
        batch_export_frame.pack(fill=tk.X, pady=4)
        self.export_list_btn = ttk.Button(batch_export_frame, text="Export file list", command=self.export_batch_list)
        self.export_list_btn.pack(side=tk.LEFT)

        # File-format settings
        ext_frame = ttk.Frame(batch_mode_frame)
        ext_frame.pack(fill=tk.X, pady=4)
        ttk.Label(ext_frame, text="Image extension:").pack(side=tk.LEFT)
        ttk.Entry(ext_frame, textvariable=self.image_ext, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            ext_frame,
            text="Include subfolders",
            variable=self.include_subfolders,
        ).pack(side=tk.LEFT, padx=(15, 0))
        # Save a reference so toggle_batch_mode can show/hide it
        self.ext_frame = ext_frame

        # Optional post-batch analysis (parameter sensitivity + ablation
        # study aggregated across the batch). This re-runs detection many
        # times per image and is purposely separated from the normal batch
        # flow so the standard run stays fast.
        analysis_frame = ttk.Frame(batch_mode_frame)
        analysis_frame.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(
            analysis_frame,
            text="After batch: run sensitivity + ablation tests (slow, paper output)",
            variable=self.batch_run_sensitivity,
        ).pack(side=tk.LEFT)
        self.batch_analysis_frame = analysis_frame

        # File-count display
        file_count_frame = ttk.Frame(batch_mode_frame)
        file_count_frame.pack(fill=tk.X, pady=4)
        ttk.Label(file_count_frame, textvariable=self.file_count_var).pack(side=tk.LEFT)
        self.file_count_frame = file_count_frame

        # ========== Common settings ==========
        ttk.Separator(param_frame, orient='horizontal').pack(fill=tk.X, pady=4)

        # --- Basic settings (always visible) ---
        save_frame = ttk.Frame(param_frame)
        save_frame.pack(fill=tk.X, pady=2)
        ttk.Label(save_frame, text="Output dir:").pack(side=tk.LEFT)
        ttk.Entry(save_frame, textvariable=self.save_path, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(save_frame, text="...", width=3, command=self.browse_save_path).pack(side=tk.LEFT)

        scan_line_frame = ttk.Frame(param_frame)
        scan_line_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scan_line_frame, text="Scan lines:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(scan_line_frame, from_=1, to=20, width=5, textvariable=self.scan_line_count).pack(side=tk.LEFT)
        ttk.Label(scan_line_frame, text="  Min width:").pack(side=tk.LEFT)
        ttk.Spinbox(scan_line_frame, from_=1, to=100, width=5, textvariable=self.min_layer_width).pack(side=tk.LEFT)

        # --- Collapsible: image enhancement ---
        def make_collapsible(parent, title, expanded=False):
            """Build a collapsible panel."""
            container = ttk.Frame(parent)
            container.pack(fill=tk.X, pady=(4, 0))
            content = ttk.Frame(container)
            is_open = [expanded]

            def toggle():
                if is_open[0]:
                    content.pack_forget()
                    toggle_btn.config(text="> " + title)
                    is_open[0] = False
                else:
                    content.pack(fill=tk.X, padx=8, pady=(0, 4))
                    toggle_btn.config(text="v " + title)
                    is_open[0] = True
                # Refresh the left-panel scroll region
                parent.after_idle(lambda: control_canvas.configure(
                    scrollregion=control_canvas.bbox("all")))

            prefix = "v " if expanded else "> "
            toggle_btn = ttk.Button(container, text=prefix + title, command=toggle,
                                    style="Toolbutton")
            toggle_btn.pack(fill=tk.X)
            if expanded:
                content.pack(fill=tk.X, padx=8, pady=(0, 4))
            return content

        # == Image-enhancement panel (open by default) ==
        enhance_content = make_collapsible(param_frame, "Image enhancement (dark-image tuning)", expanded=True)

        for label_text, var, from_, to_ in [("Brightness:", self.brightness, -100, 100),
                                              ("Contrast:", self.contrast, 0.1, 3.0),
                                              ("Gamma:", self.gamma, 0.1, 3.0)]:
            row = ttk.Frame(enhance_content)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label_text, width=10).pack(side=tk.LEFT)
            ttk.Scale(row, from_=from_, to=to_, variable=var, orient=tk.HORIZONTAL).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            val_lbl = ttk.Label(row, text="", width=4)
            val_lbl.pack(side=tk.LEFT)
            if isinstance(var, tk.IntVar):
                var.trace_add("write", lambda *_, v=var, l=val_lbl: l.config(text=str(v.get())))
                val_lbl.config(text=str(var.get()))
            else:
                var.trace_add("write", lambda *_, v=var, l=val_lbl: l.config(text=f"{v.get():.1f}"))
                val_lbl.config(text=f"{var.get():.1f}")

        enh_btn_row = ttk.Frame(enhance_content)
        enh_btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(enh_btn_row, text="Preview", command=self._preview_enhancement).pack(side=tk.LEFT, padx=2)
        ttk.Button(enh_btn_row, text="Reset", command=self._reset_enhancement).pack(side=tk.LEFT, padx=2)

        # == Detection-parameter panel (collapsed by default) ==
        detect_content = make_collapsible(param_frame, "Detection parameters", expanded=False)

        t_frame = ttk.Frame(detect_content)
        t_frame.pack(fill=tk.X, pady=2)
        ttk.Label(t_frame, text="Threshold:").pack(side=tk.LEFT)
        ttk.Combobox(t_frame, textvariable=self.threshold_method,
                     values=["otsu", "adaptive", "manual"], width=10, state="readonly").pack(side=tk.LEFT, padx=3)

        b_frame = ttk.Frame(detect_content)
        b_frame.pack(fill=tk.X, pady=2)
        ttk.Label(b_frame, text="Blur kernel:").pack(side=tk.LEFT)
        ttk.Spinbox(b_frame, from_=1, to=21, increment=2, width=4, textvariable=self.blur_size).pack(side=tk.LEFT, padx=3)
        ttk.Label(b_frame, text="CLAHE:").pack(side=tk.LEFT)
        ttk.Spinbox(b_frame, from_=0.5, to=10.0, increment=0.5, width=4, textvariable=self.clahe_clip).pack(side=tk.LEFT, padx=3)

        g_frame = ttk.Frame(detect_content)
        g_frame.pack(fill=tk.X, pady=2)
        ttk.Label(g_frame, text="CLAHE grid:").pack(side=tk.LEFT)
        ttk.Spinbox(g_frame, from_=1, to=16, width=3, textvariable=self.clahe_grid_x).pack(side=tk.LEFT, padx=1)
        ttk.Label(g_frame, text="x").pack(side=tk.LEFT)
        ttk.Spinbox(g_frame, from_=1, to=16, width=3, textvariable=self.clahe_grid_y).pack(side=tk.LEFT, padx=1)
        ttk.Label(g_frame, text="  Validation:").pack(side=tk.LEFT)
        ttk.Spinbox(g_frame, from_=1, to=10, width=3, textvariable=self.min_validation_lines).pack(side=tk.LEFT, padx=1)

        a_frame = ttk.Frame(detect_content)
        a_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(a_frame, text="Align tilted core", variable=self.align_core).pack(side=tk.LEFT)
        ttk.Label(a_frame, text=" Angle:").pack(side=tk.LEFT)
        ttk.Spinbox(a_frame, from_=-45, to=45, increment=0.5, width=5, textvariable=self.alignment_angle).pack(side=tk.LEFT, padx=2)
        ttk.Label(a_frame, text="deg").pack(side=tk.LEFT)

        # Minimum line coverage: a lamina must span at least this fraction of the
        # scan lines (default 70%) to be accepted, rejecting short fragments.
        s_frame = ttk.Frame(detect_content)
        s_frame.pack(fill=tk.X, pady=2)
        ttk.Label(s_frame, text="Min line coverage:").pack(side=tk.LEFT)
        ttk.Spinbox(s_frame, from_=10, to=100, increment=5, width=4,
                    textvariable=self.min_support_pct).pack(side=tk.LEFT, padx=2)
        ttk.Label(s_frame, text="% of scan lines").pack(side=tk.LEFT)

        # Manual lamina-angle override for badly fractured cores (single-image).
        # When enabled, the auto direction estimate is bypassed and every lamina is
        # drawn / counted along the angle entered here (deviation from vertical).
        m_frame = ttk.Frame(detect_content)
        m_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(m_frame, text="Manual lamina angle (single image)",
                        variable=self.use_manual_lamina_angle).pack(side=tk.LEFT)
        ttk.Label(m_frame, text=" Dip:").pack(side=tk.LEFT)
        ttk.Spinbox(m_frame, from_=-80, to=80, increment=0.5, width=5,
                    textvariable=self.manual_lamina_angle).pack(side=tk.LEFT, padx=2)
        ttk.Label(m_frame, text="deg from vertical").pack(side=tk.LEFT)

        # == Depth-range panel (collapsed by default) ==
        depth_content = make_collapsible(param_frame, "Depth range", expanded=False)

        ttk.Checkbutton(depth_content, text="Enable depth range (single-image mode)",
                        variable=self.enable_depth_range).pack(fill=tk.X, pady=2)

        d_row = ttk.Frame(depth_content)
        d_row.pack(fill=tk.X, pady=2)
        ttk.Label(d_row, text="Start:").pack(side=tk.LEFT)
        ttk.Spinbox(d_row, from_=0, to=10000, increment=0.1, width=7, textvariable=self.start_depth).pack(side=tk.LEFT, padx=2)
        ttk.Label(d_row, text="m  End:").pack(side=tk.LEFT)
        ttk.Spinbox(d_row, from_=0, to=10000, increment=0.1, width=7, textvariable=self.end_depth).pack(side=tk.LEFT, padx=2)
        ttk.Label(d_row, text="m").pack(side=tk.LEFT)

        # Actions panel
        btn_frame = ttk.LabelFrame(control_frame, text="Actions", padding=(12, 8))
        btn_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        # Themed button factory
        def create_styled_button(parent, text, command, icon=None):
            btn = ttk.Button(parent, text=text, command=command)
            return btn

        # Analyze-image button
        self.analyze_btn = create_styled_button(btn_frame, "Analyze image", self.analyze_image)
        self.analyze_btn.pack(fill=tk.X, pady=(5, 5), padx=5)

        # Other buttons
        view_original_btn = create_styled_button(btn_frame, "Show original image", self.show_original)
        view_original_btn.pack(fill=tk.X, pady=(0, 5), padx=5)

        view_results_btn = create_styled_button(btn_frame, "Show detection results", self.show_results)
        view_results_btn.pack(fill=tk.X, pady=(0, 5), padx=5)

        view_stats_btn = create_styled_button(btn_frame, "Show statistics", self.show_statistics)
        view_stats_btn.pack(fill=tk.X, pady=(0, 5), padx=5)

        save_all_btn = create_styled_button(btn_frame, "Save all results", self.save_all_results)
        save_all_btn.pack(fill=tk.X, pady=(0, 5), padx=5)

        # Paper-figure export button
        export_paper_btn = create_styled_button(btn_frame, "Export paper figures + data", self.export_paper_figures)
        export_paper_btn.pack(fill=tk.X, pady=(0, 5), padx=5)

        # Scale-calibration button
        scale_btn = create_styled_button(btn_frame, "Set scale", self.start_scale_calibration)
        scale_btn.pack(fill=tk.X, pady=(0, 5), padx=5)

        # Scale-status label
        self.scale_status_var = tk.StringVar(self.root, value="Scale: not calibrated")
        scale_label = ttk.Label(btn_frame, textvariable=self.scale_status_var,
                               font=("Segoe UI", 8), foreground="gray")
        scale_label.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Central display area with scrollbars
        display_outer_frame = ttk.Frame(self.main_frame)
        display_outer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Vertical scrollbar for the right pane
        display_vscroll = ttk.Scrollbar(display_outer_frame, orient="vertical")
        display_vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar for the right pane
        display_hscroll = ttk.Scrollbar(display_outer_frame, orient="horizontal")
        display_hscroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas for the right pane (no focus border)
        display_canvas = tk.Canvas(display_outer_frame, yscrollcommand=display_vscroll.set, xscrollcommand=display_hscroll.set,
                                 highlightthickness=0, bd=0)
        display_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Wire scrollbars
        display_vscroll.config(command=display_canvas.yview)
        display_hscroll.config(command=display_canvas.xview)

        # Build the right-pane display frame inside the canvas
        display_frame = ttk.LabelFrame(display_canvas, text="Results", padding=(12, 8))
        display_frame_window = display_canvas.create_window((0, 0), window=display_frame, anchor=tk.NW)

        self.display_canvas = display_canvas
        self.display_frame = display_frame
        self.display_frame_window = display_frame_window

        # Result tabs
        self.tab_control = ttk.Notebook(display_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Original-image tab
        self.tab_original = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_original, text="Original image")

        # Detection-results tab -- core processing, lamina density, lamina spacing
        self.tab_results = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_results, text="Detection results")

        # Three sub-areas inside the detection-results tab
        self.results_frame_detected = ttk.LabelFrame(self.tab_results, text="Core processing", padding=(10, 5))
        self.results_frame_detected.pack(fill=tk.BOTH, expand=True, pady=(5, 5), padx=5)

        self.results_frame_density = ttk.LabelFrame(self.tab_results, text="Lateral lamina distribution", padding=(10, 5))
        self.results_frame_density.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)

        self.results_frame_width = ttk.LabelFrame(self.tab_results, text="Lamina strength curve", padding=(10, 5))
        self.results_frame_width.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)

        # Statistics tab
        self.tab_stats = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_stats, text="Statistics")

        # Batch-results tab
        self.tab_batch = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_batch, text="Batch results")

        # Sub-areas inside the batch tab
        self.batch_frame_combined = ttk.LabelFrame(self.tab_batch, text="Combined lamina strength curve", padding=(10, 5))
        self.batch_frame_combined.pack(fill=tk.BOTH, expand=True, pady=(5, 5), padx=5)

        self.batch_frame_heatmap = ttk.LabelFrame(self.tab_batch, text="Lamina-strength heatmap", padding=(10, 5))
        self.batch_frame_heatmap.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)

        # Lateral strength curve
        self.batch_frame_curve = ttk.LabelFrame(self.tab_batch, text="Lateral combined lamina-strength curve", padding=(10, 5))
        self.batch_frame_curve.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)

        # Help panel on the far right
        help_frame_outer = ttk.LabelFrame(self.main_frame, text="Usage & Help", padding=(8, 5))
        help_frame_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), pady=0)

        # Help-panel scrollbar
        help_vscroll = ttk.Scrollbar(help_frame_outer, orient=tk.VERTICAL)
        help_vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Help text widget (no focus border)
        help_text = tk.Text(help_frame_outer, width=40, height=40, wrap=tk.WORD,
                           yscrollcommand=help_vscroll.set, font=self.default_font,
                           highlightthickness=0, bd=1)
        help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        help_vscroll.config(command=help_text.yview)

        # Text tags
        default_font_name, default_font_size = self.default_font
        help_text.tag_configure("title", font=(default_font_name, 12, "bold"), foreground="navy")
        help_text.tag_configure("heading", font=(default_font_name, 11, "bold"), foreground="blue")
        help_text.tag_configure("subheading", font=(default_font_name, 10, "bold"), foreground="darkblue")
        help_text.tag_configure("bold", font=(default_font_name, 9, "bold"))
        help_text.tag_configure("normal", font=(default_font_name, 9))
        help_text.tag_configure("emphasis", font=(default_font_name, 9, "italic"))

        # Help content
        help_text.insert(tk.END, "Rock-core lamina identification - Usage\n", "title")
        help_text.insert(tk.END, "======================================\n\n", "title")

        # Parameter-settings section
        help_text.insert(tk.END, "Parameter overview\n", "heading")
        help_text.insert(tk.END, "------------------\n\n", "heading")

        help_text.insert(tk.END, "[Basic]\n", "subheading")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Batch mode: ", "bold")
        help_text.insert(tk.END, "process every image in a folder instead of a single image\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Image path: ", "bold")
        help_text.insert(tk.END, "pick a core image file or folder\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Output dir: ", "bold")
        help_text.insert(tk.END, "directory for the results\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Image extension: ", "bold")
        help_text.insert(tk.END, "extension to filter on in batch mode\n\n", "normal")

        help_text.insert(tk.END, "[Image preprocessing]\n", "subheading")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Blur kernel: ", "bold")
        help_text.insert(tk.END, "Gaussian-blur kernel size used for denoising; larger = more noise removal, fewer details\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "CLAHE clip: ", "bold")
        help_text.insert(tk.END, "contrast-limited adaptive histogram equalisation limit; boosts low-contrast images\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "CLAHE grid: ", "bold")
        help_text.insert(tk.END, "tile size of the local contrast enhancement\n\n", "normal")

        help_text.insert(tk.END, "[Lamina detection]\n", "subheading")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Threshold method:\n", "bold")
        help_text.insert(tk.END, "  - ", "normal")
        help_text.insert(tk.END, "Otsu: ", "bold")
        help_text.insert(tk.END, "auto-pick threshold; works well for high-contrast images\n", "normal")
        help_text.insert(tk.END, "  - ", "normal")
        help_text.insert(tk.END, "Adaptive: ", "bold")
        help_text.insert(tk.END, "threshold computed locally; handles uneven illumination\n", "normal")
        help_text.insert(tk.END, "  - ", "normal")
        help_text.insert(tk.END, "Manual: ", "bold")
        help_text.insert(tk.END, "fixed threshold; good for batches of similar images\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Min lamina width: ", "bold")
        help_text.insert(tk.END, "smallest lamina width to keep (pixels); too small lets noise through, too large drops thin laminae\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Scan lines: ", "bold")
        help_text.insert(tk.END, "number of horizontal scan lines; more lines means more coverage at the cost of compute\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Min validation lines: ", "bold")
        help_text.insert(tk.END, "min number of validation lines that must agree; higher = stricter, may drop real laminae\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Align tilted core: ", "bold")
        help_text.insert(tk.END, "shear-correct tilted cores so laminae are closer to vertical\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "Alignment angle: ", "bold")
        help_text.insert(tk.END, "shear angle (0 = auto-detect)\n\n", "normal")

        # Output-image section
        help_text.insert(tk.END, "Output images\n", "heading")
        help_text.insert(tk.END, "-------------\n\n", "heading")

        help_text.insert(tk.END, "[Single-image outputs]\n", "subheading")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "layer_detection.png: ", "bold")
        help_text.insert(tk.END, "main result figure showing detected laminae and scan lines\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "binary_image.png: ", "bold")
        help_text.insert(tk.END, "binary image (useful for verifying the threshold)\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "validation_lines.png: ", "bold")
        help_text.insert(tk.END, "image overlaid with validation lines\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "validated_grid.png: ", "bold")
        help_text.insert(tk.END, "validated laminae visualised on the image\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "layer_density.png: ", "bold")
        help_text.insert(tk.END, "lateral lamina-density curve\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "layer_intensity.png: ", "bold")
        help_text.insert(tk.END, "lateral lamina-strength curve\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "detected_layers.png: ", "bold")
        help_text.insert(tk.END, "side-by-side comparison of original image and detection result\n\n", "normal")

        help_text.insert(tk.END, "[Batch-mode outputs]\n", "subheading")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "all_layers.csv: ", "bold")
        help_text.insert(tk.END, "combined lamina records across all images\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "combined_layer_intensity.png: ", "bold")
        help_text.insert(tk.END, "combined lamina-strength curve across all images\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "layer_intensity_heatmap.png: ", "bold")
        help_text.insert(tk.END, "lamina-strength heatmap\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "layer_intensity_curve.png: ", "bold")
        help_text.insert(tk.END, "lateral combined lamina-strength curve\n", "normal")
        help_text.insert(tk.END, "- ", "bold")
        help_text.insert(tk.END, "layer_column_chart.png: ", "bold")
        help_text.insert(tk.END, "geological-column-like lamina distribution\n\n", "normal")

        # Tips section
        help_text.insert(tk.END, "Tips\n", "heading")
        help_text.insert(tk.END, "----\n\n", "heading")
        help_text.insert(tk.END, "- For sharp laminae, use Otsu and fewer scan lines\n", "normal")
        help_text.insert(tk.END, "- For low-contrast laminae:\n", "normal")
        help_text.insert(tk.END, "  - Use adaptive thresholding\n", "normal")
        help_text.insert(tk.END, "  - Increase the CLAHE clip and tune the grid\n", "normal")
        help_text.insert(tk.END, "  - Reduce the minimum lamina width\n", "normal")
        help_text.insert(tk.END, "  - Reduce the minimum validation-line count\n", "normal")
        help_text.insert(tk.END, "- When running batches, validate parameters on a single image first\n", "normal")
        help_text.insert(tk.END, "- Scan-line positions can be picked manually; cover several regions for better coverage\n\n", "normal")

        help_text.insert(tk.END, "\n(c) 2025 Rock Core Lamina Identification System v1.0", "emphasis")

        help_text.config(state=tk.DISABLED)  # read-only

        # Configure the canvas scroll region
        def configure_scroll_region(event):
            # Refresh the canvas scroll region
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        control_frame.bind("<Configure>", configure_scroll_region)

        # Configure the main display-area scroll region
        def configure_display_scroll_region(event=None):
            # Refresh the main display canvas
            self.display_canvas.configure(scrollregion=self.display_canvas.bbox("all"))
            # Resize the canvas window so it fits the content
            canvas_width = self.display_canvas.winfo_width()
            if canvas_width > 1:  # make sure the canvas has been drawn
                self.display_canvas.itemconfig(self.display_frame_window, width=canvas_width-20)

        # Bindings
        self.display_frame.bind("<Configure>", configure_display_scroll_region)
        display_canvas.bind("<Configure>", configure_display_scroll_region)

        # Mouse-wheel scrolling for the main display
        def _on_display_mousewheel(event):
            self.display_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        display_canvas.bind_all("<MouseWheel>", _on_display_mousewheel)

        # Helper for callers that need to update the scroll region
        self.update_display_scroll_region = configure_display_scroll_region

        # Update status
        self.status_var.set("UI ready")

        if self._check_optimization_flag('ROCK_OPTIMIZE_UI'):
            gc.collect()
    def toggle_batch_mode(self):
        """Toggle batch-processing mode."""
        if self.batch_mode.get():
            # Switching to batch mode
            # Update the file-count label
            if hasattr(self, 'batch_image_files') and self.batch_image_files:
                self.file_count_var.set(f"Loaded {len(self.batch_image_files)} image(s)")
            else:
                self.file_count_var.set("No file loaded")

            # Refresh the UI
            self.root.update()
        else:
            # Switching to single-image mode
            # Update the file-count label
            self.file_count_var.set("Single-image mode")

            # Refresh the UI
            self.root.update()
    def browse_folder(self):
        """Browse for and select a folder containing images."""
        folder_path = filedialog.askdirectory(title="Select a folder containing core images")
        if folder_path:
            self.image_path.set(folder_path)
            self.status_var.set(f"Folder selected: {folder_path}")
    def browse_save_path(self):
        """Browse for and select an output directory."""
        dir_path = filedialog.askdirectory(title="Select output directory")
        if dir_path:
            self.save_path.set(dir_path)
            self.status_var.set(f"Output dir: {dir_path}")
    def browse_image(self):
        """Browse for and select an image file."""
        # Single image only; batch mode uses browse_folder
        file_path = filedialog.askopenfilename(
                title="Select a core image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
            )
        if file_path:
                self.image_path.set(file_path)
                self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
                # Load the original image automatically
                self.show_original()
