#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-figure export."""

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


class PaperExportMixin:
    def export_paper_figures(self, output_dir, start_depth=None, end_depth=None,
                             include_sensitivity=False, include_ablation=False,
                             sensitivity_tolerance_px=10):
        """Export the full set of paper figures and data (process images + statistics + data files).

        Process images (process_images/) numbered by the pipeline:
          - 01_original_color.png       -- original color image
          - 02_grayscale.png            -- grayscale conversion
          - 03_gaussian_denoised.png    -- Gaussian-denoised image
          - 04_clahe_enhanced.png       -- CLAHE-enhanced image
          - 05_geometry_corrected.png   -- geometry-corrected image (or final preprocessing result)
          - 06_binary.png              -- binary image
          - 07_detection_overlay.png   -- lamina overlay
          + canny_edges.png            -- Canny edges (extra)
          + validation_lines.png       -- validation lines (extra)
          + validated_grid.png         -- validated grid (extra)
          + 04D1/04D2/04D3 step_window small/medium/large -- per-window step responses (2D)
          + 04D4_step_window_fused      -- multi-branch fused candidate-boundary map (2D)
          + 04D5_multi_window_panel     -- small/medium/large/fused panel with colour bars
          + 04E_sobel_x_response        -- Sobel-X cross-lamina gradient magnitude (2D)
          + 04F_blackhat_response       -- black-hat output (dark thin laminae)
          + 04H_detection_input_fused   -- final fused image actually scanned by the detector
          + 04G_response_maps_panel     -- combined response-map panel with colour bars

        Analysis figures (figures/):
          - preprocessing_comparison.png  -- preprocessing pipeline comparison
          - scanline_profiles.png         -- scan-line gray profile and detection points
          - layer_classification_pie.png  -- seven-tier classification pie chart
          - layer_thickness_histogram.png -- lamina-spacing histogram
          - layer_strength_boxplot.png    -- per-class lamina-strength boxplot
          - detection_overlay.png         -- original + lamina overlay
          - density_depth_curve.png       -- lamina density vs depth
          - intensity_heatmap_single.png  -- lamina-strength heatmap
          - layer_density.png             -- lamina-density curve
          - layer_intensity.png           -- lamina-strength curve
          - layer_spacing_histogram.png   -- lamina-spacing histogram

        Data files (data/):
          - paper_layer_data.csv       -- full lamina data
          - paper_summary_stats.csv    -- summary statistics
          - paper_classification.csv   -- seven-tier classification statistics
          - lamina_variation_curve.csv -- lamina-variation curve
          - position_info.csv          -- per-pixel position statistics
        """
        import shutil
        from scipy.ndimage import gaussian_filter1d
        
        paper_dir = os.path.join(output_dir, "paper_export")
        process_dir = os.path.join(paper_dir, "process_images")
        figures_dir = os.path.join(paper_dir, "figures")
        data_dir = os.path.join(paper_dir, "data")
        for d in [paper_dir, process_dir, figures_dir, data_dir]:
            os.makedirs(d, exist_ok=True)
        
        print(f"=== Exporting full paper figure set to: {paper_dir} ===")

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        if not self.layers:
            print("No detection results; cannot export paper figures")
            return paper_dir

        h, w = self.height, self.width

        # ================================================================
        # Part 1: process images -- saved step-by-step
        # _imwrite_safe is used to support Unicode paths
        # ================================================================
        print("--- Exporting process images ---")
        steps = getattr(self, '_preprocess_steps', {})
        # 01-04 use the *pristine* (pre-alignment) versions so they do not double up with 05_geometry_corrected.
        # Otherwise the "original color image" also appears sheared, stacking on top of the "geometry correction" step
        # and visually looking like "two shears".
        steps_pristine = getattr(self, '_preprocess_steps_pristine', steps) or steps
        img_for_01 = getattr(self, 'image_original', None)
        if img_for_01 is None:
            img_for_01 = self.image
        gray_for_02 = getattr(self, 'gray_pristine', None)
        if gray_for_02 is None:
            gray_for_02 = self.gray
        clahe_for_04 = steps_pristine.get("clahe")
        if clahe_for_04 is None:
            clahe_for_04 = getattr(self, 'enhanced_no_grad_pristine', None)
        if clahe_for_04 is None:
            clahe_for_04 = self.enhanced_no_grad
        
        save_list = [
            ("01_original_color.png",      "Original color image (pre-alignment)",       img_for_01),
            ("02_grayscale.png",           "Grayscale image (pre-alignment)",       gray_for_02),
            ("03_gaussian_denoised.png",   "Gaussian-denoised image (pre-alignment)",   steps_pristine.get("blurred")),
            ("04_clahe_enhanced.png",      "CLAHE-enhanced image (pre-alignment)", clahe_for_04),
            ("05_geometry_corrected.png" if self.aligned else "05_preprocessed_final.png",
             "Geometry-corrected image" if self.aligned else "Final preprocessed image",
             self.processed),
            ("06_binary.png",             "Binary image (from geometry-corrected source)",          self.binary),
        ]
        
        for fname, desc, img_data in save_list:
            try:
                if img_data is not None:
                    ok = self._imwrite_safe(os.path.join(process_dir, fname), img_data)
                    if ok:
                        print(f"  {desc} saved: {fname}")
                    else:
                        print(f"  {desc} write failed: {fname}")
                else:
                    print(f"  {desc} data unavailable, skipped")
            except Exception as e:
                print(f"  Error saving {desc}: {e}")
        
        # 7. Lamina overlay (needs to be drawn)
        try:
            overlay = self.image.copy()
            for layer_data in self.layers:
                y = layer_data["y"]
                pts = layer_data.get("validated_points", layer_data["points"])
                for pt in pts:
                    cv2.line(overlay, (pt, max(0, y - 10)), (pt, min(h - 1, y + 10)),
                             (0, 0, 255), 2)
                    cv2.circle(overlay, (pt, y), 3, (0, 255, 255), -1)
            ok = self._imwrite_safe(os.path.join(process_dir, "07_detection_overlay.png"), overlay)
            print(f"  Lamina overlay {'saved' if ok else 'write failed'}")
        except Exception as e:
            print(f"  Error saving lamina overlay: {e}")
        
        # Additional: copy diagnostic images from the detection output directory
        extra_diag = {
            "binary_image.png": "Binary image (raw)",
            "canny_edges.png": "Canny edge map",
            "validation_lines.png": "Validation-line detection",
            "validated_grid.png": "Validated grid annotation",
            "layer_detection.png": "Detection-result annotation",
        }
        for fname, desc in extra_diag.items():
            src_path = os.path.join(self.output_dir, fname)
            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, os.path.join(process_dir, fname))
                    print(f"  Additional: {desc} copied")
                except Exception as e:
                    print(f"  Error copying {desc}: {e}")

        # ================================================================
        # Part 2: gather lamina data
        # ================================================================
        all_layer_records = []
        layer_id = 0
        for scan_result in self.layers:
            y = scan_result["y"]
            pts = scan_result.get("validated_points", scan_result["points"])
            consistency_scores = scan_result.get("consistency_scores", {})
            for idx, pt_x in enumerate(pts):
                strength = 1.0
                left_m, right_m, delta_g = 0.0, 0.0, 0.0
                if self.processed is not None and 0 <= y < self.processed.shape[0]:
                    hw = max(5, 8)
                    x_lo = max(0, pt_x - hw)
                    x_hi = min(self.processed.shape[1], pt_x + hw + 1)
                    left_r = self.processed[y, x_lo:pt_x].astype(np.float64)
                    right_r = self.processed[y, pt_x:x_hi].astype(np.float64)
                    if len(left_r) > 0 and len(right_r) > 0:
                        left_m = float(np.mean(left_r))
                        right_m = float(np.mean(right_r))
                        delta_g = abs(right_m - left_m)
                        strength = float(np.log1p(delta_g) * 5.0)
                
                spacing_px = pts[idx + 1] - pt_x if idx < len(pts) - 1 else 0
                
                depth_m = None
                if start_depth is not None and end_depth is not None and start_depth != end_depth:
                    depth_m = start_depth + (end_depth - start_depth) * (pt_x / w)
                
                if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                    spacing_mm = spacing_px / self.pixel_per_mm if spacing_px > 0 else 0
                else:
                    spacing_mm = spacing_px * (100.0 / w) if spacing_px > 0 else 0
                
                classification = self._classify_layer(spacing_mm) if spacing_px > 0 else ""
                
                layer_id += 1
                all_layer_records.append({
                    "lamina_id": layer_id,
                    "scan_line_y": y,
                    "position_x": pt_x,
                    "spacing_to_next_px": spacing_px,
                    "spacing_to_next_mm": round(spacing_mm, 2) if spacing_px > 0 else "",
                    "depth_m": round(depth_m, 4) if depth_m is not None else "",
                    "strength": round(strength, 4),
                    "log_strength": round(float(np.log1p(strength)), 4),
                    "delta_gray": round(delta_g, 1),
                    "left_mean": round(left_m, 1),
                    "right_mean": round(right_m, 1),
                    "crossline_consistency": consistency_scores.get(pt_x, 0),
                    "spacing_class": classification
                })

        layer_df = pd.DataFrame(all_layer_records) if all_layer_records else pd.DataFrame()

        # ================================================================
        # Part 3: analysis figures -- paper-grade high resolution
        # ================================================================
        print("--- Exporting analysis figures ---")
        
        try:
            self._export_preprocessing_comparison(figures_dir)
        except Exception as e:
            print(f"  Preprocessing comparison error: {e}")
        
        try:
            self._export_scanline_profiles(figures_dir)
        except Exception as e:
            print(f"  Scan-line profile error: {e}")
        
        try:
            if not layer_df.empty:
                self._export_classification_pie(figures_dir, layer_df)
        except Exception as e:
            print(f"  Classification pie-chart error: {e}")
        
        try:
            if not layer_df.empty:
                self._export_thickness_histogram(figures_dir, layer_df)
        except Exception as e:
            print(f"  Thickness histogram error: {e}")
        
        try:
            if not layer_df.empty:
                self._export_strength_boxplot(figures_dir, layer_df)
        except Exception as e:
            print(f"  Strength boxplot error: {e}")
        
        try:
            self._export_detection_overlay(figures_dir)
        except Exception as e:
            print(f"  Overlay figure error: {e}")
        
        try:
            if not layer_df.empty:
                self._export_density_curve(figures_dir, layer_df, start_depth, end_depth)
        except Exception as e:
            print(f"  Density curve error: {e}")
        
        try:
            self._export_intensity_heatmap(figures_dir)
        except Exception as e:
            print(f"  Heatmap error: {e}")
        
        # Call visualize_layers to generate the full statistics figures (density, strength, spacing histogram)
        try:
            self.visualize_layers(figures_dir)
            print("  Density/strength/spacing statistics saved")
        except Exception as e:
            print(f"  Statistics figure error: {e}")

        # ================================================================
        # Part 4: data files
        # ================================================================
        print("--- Exporting data files ---")
        try:
            if not layer_df.empty:
                csv_path = os.path.join(data_dir, "paper_layer_data.csv")
                layer_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"  Lamina data exported: {csv_path}")
                
                # Also export Excel
                layer_df.to_excel(os.path.join(data_dir, "paper_layer_data.xlsx"), index=False)
                
                # Summary statistics
                calibrated = self.pixel_per_mm is not None and self.pixel_per_mm > 0
                spacing_col = "spacing_to_next_px"
                spacing_mm_col = "spacing_to_next_mm"
                valid_sp = pd.to_numeric(layer_df[spacing_col], errors='coerce').dropna()
                valid_sp = valid_sp[valid_sp > 0]
                valid_sp_mm = pd.to_numeric(layer_df[spacing_mm_col], errors='coerce').dropna()
                valid_sp_mm = valid_sp_mm[valid_sp_mm > 0]
                
                all_consistency = layer_df["crossline_consistency"].tolist()
                
                # ====== Unique laminae (cross-line clustering) statistics ======
                laminae = getattr(self, 'laminae', None) or []
                lam_settings = getattr(self, '_lamina_settings', None) or {}
                valid_laminae = [la for la in laminae if la.get("is_valid")]
                valid_sorted_la = sorted(valid_laminae, key=lambda la: la["x_mean"])
                unique_sp_px = [
                    valid_sorted_la[i + 1]["x_mean"] - valid_sorted_la[i]["x_mean"]
                    for i in range(len(valid_sorted_la) - 1)
                ]
                unique_sp_mm = (
                    [s / self.pixel_per_mm for s in unique_sp_px]
                    if calibrated and unique_sp_px else []
                )
                avg_support = (
                    float(np.mean([la["n_support_lines"] for la in valid_laminae]))
                    if valid_laminae else 0.0
                )
                avg_support_ratio = (
                    float(np.mean([la["support_ratio"] for la in valid_laminae]))
                    if valid_laminae else 0.0
                )
                
                summary = {
                    # -- unique laminae (paper should cite this set) --
                    "unique_laminae_cluster": len(valid_laminae),
                    "cluster_candidate_total": lam_settings.get("n_clusters", 0),
                    "cluster_tolerance_px": lam_settings.get("tolerance_px", 0),
                    "min_support_required": lam_settings.get("min_support", 0),
                    "n_scan_lines": len(self.layers),
                    "avg_support_per_lamina": round(avg_support, 2),
                    "avg_support_ratio": round(avg_support_ratio, 3),
                    "unique_mean_spacing_px": round(float(np.mean(unique_sp_px)), 2) if unique_sp_px else "N/A",
                    "unique_spacing_std_px": round(float(np.std(unique_sp_px)), 2) if unique_sp_px else "N/A",
                    "unique_mean_spacing_mm": round(float(np.mean(unique_sp_mm)), 3) if unique_sp_mm else "N/A",
                    "unique_spacing_cv_percent": (
                        round(float(np.std(unique_sp_px)) / float(np.mean(unique_sp_px)) * 100, 1)
                        if unique_sp_px and np.mean(unique_sp_px) > 0 else "N/A"
                    ),
                    "scale_calibrated": "yes" if calibrated else "no_estimated",
                    "scale_px_per_mm": round(self.pixel_per_mm, 4) if calibrated else "N/A",
                    # -- candidate-point level (diagnostic only; reflects raw detections) --
                    "candidate_points_total": len(layer_df),
                    "avg_candidates_per_line": round(len(layer_df) / max(1, len(self.layers)), 2),
                    "candidate_avg_spacing_px": round(valid_sp.mean(), 2) if len(valid_sp) > 0 else "N/A",
                    "candidate_avg_spacing_mm": round(valid_sp_mm.mean(), 2) if len(valid_sp_mm) > 0 else "N/A",
                    "mean_strength": round(layer_df["strength"].mean(), 4),
                    "strength_std": round(layer_df["strength"].std(), 4),
                    "avg_crossline_consistency": round(np.mean(all_consistency), 2) if all_consistency else "N/A",
                    "high_consistency_percent": round(sum(1 for c in all_consistency if c >= 2) / max(1, len(all_consistency)) * 100, 1),
                }
                summary_df = pd.DataFrame([summary])
                summary_df.to_csv(os.path.join(data_dir, "paper_summary_stats.csv"),
                                  index=False, encoding='utf-8-sig')
                summary_df.to_excel(os.path.join(data_dir, "paper_summary_stats.xlsx"), index=False)
                
                # Seven-tier classification statistics
                cls_col = "spacing_class"
                valid_cls = layer_df[layer_df[cls_col] != ""]
                if not valid_cls.empty:
                    cls_stats = valid_cls.groupby(cls_col).agg(
                        count=("lamina_id", "count"),
                        mean_spacing_px=("spacing_to_next_px", "mean"),
                        mean_spacing_mm=("spacing_to_next_mm", "mean"),
                        mean_strength=("strength", "mean"),
                        strength_std=("strength", "std"),
                    ).reset_index()
                    cls_stats["ratio_percent"] = (cls_stats["count"] / cls_stats["count"].sum() * 100).round(1)
                    cls_order = ["thin_lamina(<1mm)", "lamina(1-5mm)", "thick_lamina(5-10mm)",
                                 "thin_layer(1-5cm)", "layer(5-10cm)", "thick_layer(10-50cm)", "massive(>50cm)"]
                    cls_stats[cls_col] = pd.Categorical(cls_stats[cls_col], categories=cls_order, ordered=True)
                    cls_stats = cls_stats.sort_values(cls_col)
                    cls_stats.to_csv(os.path.join(data_dir, "paper_classification.csv"),
                                     index=False, encoding='utf-8-sig')
                    cls_stats.to_excel(os.path.join(data_dir, "paper_classification.xlsx"), index=False)
                    print("  Classification stats exported")
                
                # Position statistics
                if self.layer_stats and "position" in self.layer_stats and self.layer_stats["position"] is not None:
                    self.layer_stats["position"].to_csv(
                        os.path.join(data_dir, "position_info.csv"), index=False, encoding='utf-8-sig')
                    self.layer_stats["position"].to_excel(
                        os.path.join(data_dir, "position_info.xlsx"), index=False)
                    print("  Position stats exported")
                
                # Lamina variation curve
                self._export_variation_curve_data(data_dir)
                
        except Exception as e:
            print(f"  Error exporting data files: {e}")
            import traceback
            traceback.print_exc()

        # ================================================================
        # Part 5: modular paper directory structure (per the spec)
        # 00_input/ 01_preprocessing/ 02_scanline_detection/
        # 03_crossline_validation/ 04_results/ 05_method_comparison/
        # parameters/
        # ================================================================
        print("--- Writing modular paper directories ---")
        try:
            self._export_paper_modules(paper_dir, layer_df, start_depth, end_depth)
        except Exception as e:
            print(f"  Modular output error: {e}")
            import traceback
            traceback.print_exc()

        # ================================================================
        # Part 6: parameter sensitivity / ablation study (on demand)
        # ================================================================
        if include_sensitivity:
            try:
                print("--- Running parameter sensitivity analysis ---")
                self.run_parameter_sensitivity(
                    os.path.join(paper_dir, "06_sensitivity"),
                    tolerance_px=sensitivity_tolerance_px,
                )
            except Exception as e:
                print(f"  Parameter sensitivity error: {e}")
                import traceback; traceback.print_exc()

        if include_ablation:
            try:
                print("--- Running ablation study ---")
                self.run_ablation_study(
                    os.path.join(paper_dir, "07_ablation"),
                    tolerance_px=sensitivity_tolerance_px,
                )
            except Exception as e:
                print(f"  Ablation study error: {e}")
                import traceback; traceback.print_exc()

        print(f"=== Paper figures exported: {paper_dir} ===")
        return paper_dir
    def _export_variation_curve_data(self, data_dir):
        """Export lamina-variation-curve data (used by paper export)."""
        try:
            if self.image is None or not self.layers:
                return
            img_w = self.image.shape[1]
            bin_size = max(5, img_w // 200)
            n_bins = img_w // bin_size
            rows = []
            for b in range(n_bins):
                x_start = b * bin_size
                x_end = min((b + 1) * bin_size, img_w)
                x_center = (x_start + x_end) / 2
                count_in_bin = 0
                strength_sum = 0.0
                consistency_sum = 0.0
                for scan_result in self.layers:
                    pts = scan_result.get("validated_points", scan_result["points"])
                    scores = scan_result.get("consistency_scores", {})
                    for pt in pts:
                        if x_start <= pt < x_end:
                            count_in_bin += 1
                            consistency_sum += scores.get(pt, 0)
                            if self.processed is not None:
                                y = scan_result["y"]
                                hw = 5
                                lo = max(0, pt - hw)
                                hi = min(img_w, pt + hw + 1)
                                seg = self.processed[y, lo:hi].astype(np.float64)
                                if len(seg) > 1:
                                    strength_sum += float(np.log1p(np.mean(np.abs(np.diff(seg)))))
                n_lines = max(1, len(self.layers))
                row = {
                    "position_px": round(x_center, 1),
                    "bin_start": x_start,
                    "bin_end": x_end,
                    "lamina_count": count_in_bin,
                    "density_per_scanline": round(count_in_bin / n_lines, 3),
                    "mean_strength": round(strength_sum / count_in_bin, 3) if count_in_bin > 0 else 0,
                    "mean_consistency": round(consistency_sum / count_in_bin, 2) if count_in_bin > 0 else 0,
                }
                if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                    row["position_mm"] = round(x_center / self.pixel_per_mm, 2)
                rows.append(row)
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(data_dir, "lamina_variation_curve.csv"),
                          index=False, encoding='utf-8-sig')
                df.to_excel(os.path.join(data_dir, "lamina_variation_curve.xlsx"), index=False)
                print("  Lamina variation curve exported")
        except Exception as e:
            print(f"  Variation curve export error: {e}")
    def _export_paper_modules(self, paper_dir, layer_df, start_depth, end_depth):
        """Build the modular directory structure and full data/figures package described in the spec.

        Output directories:
          00_input/                <- raw input and sample metadata
          01_preprocessing/        <- grayscale/denoise/CLAHE/Canny/Hough/geometry correction
          02_scanline_detection/   <- scan lines and representative gray/gradient curves
          03_crossline_validation/ <- before/after cross-line validation and rejected points
          04_results/              <- detection overlays, strength heatmap, attribute table
          05_method_comparison/    <- Sobel/Canny/rule-based/proposed comparison
          parameters/              <- scale_calibration.json and other config
        """
        import json
        from pathlib import Path as _Path
        from scipy.ndimage import gaussian_filter1d
        
        # Create sub-directories
        dir_input  = os.path.join(paper_dir, "00_input")
        dir_prep   = os.path.join(paper_dir, "01_preprocessing")
        dir_scan   = os.path.join(paper_dir, "02_scanline_detection")
        dir_cross  = os.path.join(paper_dir, "03_crossline_validation")
        dir_result = os.path.join(paper_dir, "04_results")
        dir_method = os.path.join(paper_dir, "05_method_comparison")
        dir_param  = os.path.join(paper_dir, "parameters")
        for d in [dir_input, dir_prep, dir_scan, dir_cross, dir_result, dir_method, dir_param]:
            os.makedirs(d, exist_ok=True)
        
        sample_id = _Path(getattr(self, 'image_path', 'sample') or 'sample').stem
        h, w = int(self.height), int(self.width)
        
        # ---------- 00_input ----------
        try:
            if self.image is not None:
                self._imwrite_safe(os.path.join(dir_input, "00_original_core.png"), self.image)
                self._imwrite_safe(os.path.join(dir_input, "01_core_roi.png"), self.image)
            
            # Scale-bar image
            if self.image is not None and self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                scale_img = self.image.copy()
                bar_mm = 10
                bar_pixels = int(bar_mm * self.pixel_per_mm)
                x0, y0 = 20, h - 40
                if bar_pixels < w - 40:
                    cv2.rectangle(scale_img, (x0, y0), (x0 + bar_pixels, y0 + 10), (255, 255, 255), -1)
                    cv2.rectangle(scale_img, (x0, y0), (x0 + bar_pixels, y0 + 10), (0, 0, 0), 2)
                    cv2.putText(scale_img, f"{bar_mm} mm", (x0, y0 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                self._imwrite_safe(os.path.join(dir_input, "02_scale_calibrated_core.png"), scale_img)
            
            prep_meta = getattr(self, '_preprocess_meta', {}) or {}
            det_diag = getattr(self, '_detection_diagnostics', {}) or {}
            sample_meta = {
                "sample_id": sample_id,
                "image_path": getattr(self, 'image_path', ''),
                "image_height_pixel": h,
                "image_width_pixel": w,
                "image_mean_gray": float(np.mean(self.gray)) if self.gray is not None else None,
                "image_std_gray": float(np.std(self.gray)) if self.gray is not None else None,
                "is_dark_core": bool(prep_meta.get('is_dark_core', False)),
                "dark_mode_applied": bool(prep_meta.get('dark_mode_applied', False)),
                "blackhat_applied": bool(prep_meta.get('blackhat_applied', False)),
                "pixel_per_mm": float(self.pixel_per_mm) if self.pixel_per_mm else None,
                "image_width_mm": float(w / self.pixel_per_mm) if self.pixel_per_mm else None,
                "image_height_mm": float(h / self.pixel_per_mm) if self.pixel_per_mm else None,
                "start_depth_m": start_depth,
                "end_depth_m": end_depth,
                "core_aligned": bool(self.aligned),
                "n_main_scanlines": len(self.layers),
                "n_validation_lines": len(getattr(self, 'validation_lines', [])),
                "n_candidate_points_total": sum(len(sr.get("validated_points", sr.get("points", []))) for sr in self.layers),
                "lamina_clustering": getattr(self, '_lamina_settings', {}),
                "n_unique_laminae": len([la for la in (getattr(self, 'laminae', None) or []) if la.get("is_valid")]),
                "detection_fallbacks_triggered": det_diag.get("fallbacks_triggered", []),
            }
            with open(os.path.join(dir_input, "sample_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(sample_meta, f, ensure_ascii=False, indent=2, default=str)
            print("  00_input/ written")
        except Exception as e:
            print(f"  00_input/ error: {e}")
        
        # ---------- parameters ----------
        try:
            # scale_calibration.json
            if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                calib = {
                    "calibrated": True,
                    "pixel_per_mm": float(self.pixel_per_mm),
                    "mm_per_pixel": float(1.0 / self.pixel_per_mm),
                    "image_width_mm": float(w / self.pixel_per_mm),
                    "image_height_mm": float(h / self.pixel_per_mm),
                }
            else:
                calib = {
                    "calibrated": False,
                    "note": "Not calibrated; thickness statistics use an estimated value (assuming image width=100mm)",
                    "estimated_pixel_per_mm": float(w / 100.0),
                }
            with open(os.path.join(dir_param, "scale_calibration.json"), 'w', encoding='utf-8') as f:
                json.dump(calib, f, ensure_ascii=False, indent=2)
            
            # preprocessing_parameters.json
            prep_meta = dict(getattr(self, '_preprocess_meta', {}))
            prep_meta.update({
                "gaussian_kernel": [prep_meta.get('blur_size', 5)] * 2,
                "clahe_clip_limit": prep_meta.get('clahe_clip_effective', 2.0),
                "clahe_tile_grid": prep_meta.get('clahe_grid', [8, 8]),
                "canny_threshold_low": 50,
                "canny_threshold_high": 150,
                "hough_correction_enabled": True,
                "hough_correction_applied": bool(self.aligned),
                "rotation_angle_deg": float(getattr(self, 'alignment_angle', 0.0)),
                "use_shear": bool(getattr(self, 'use_shear', False)),
                "dark_core_auto_enhance": prep_meta.get('dark_mode_applied', False),
            })
            with open(os.path.join(dir_param, "preprocessing_parameters.json"), 'w', encoding='utf-8') as f:
                json.dump(prep_meta, f, ensure_ascii=False, indent=2, default=str)
            print("  parameters/ config files written")
        except Exception as e:
            print(f"  parameters/ error: {e}")
        
        # ---------- 01_preprocessing ----------
        try:
            steps = getattr(self, '_preprocess_steps', {})
            steps_pristine = getattr(self, '_preprocess_steps_pristine', None)
            if not steps_pristine:
                steps_pristine = steps
            img_pristine = getattr(self, 'image_original', None)
            if img_pristine is None:
                img_pristine = self.image
            gray_pristine = getattr(self, 'gray_pristine', None)
            if gray_pristine is None:
                gray_pristine = self.gray
            enhanced_pristine = getattr(self, 'enhanced_no_grad_pristine', None)
            if enhanced_pristine is None:
                enhanced_pristine = steps_pristine.get("clahe")
            if enhanced_pristine is None:
                enhanced_pristine = self.enhanced_no_grad
            preset = [
                # 03A-03D use the pre-alignment original; 03E shows the geometry-corrected result
                ("03A_original.png",            img_pristine),
                ("03B_grayscale.png",           gray_pristine),
                ("03C_gaussian_denoising.png",  steps_pristine.get("blurred", steps_pristine.get("denoised"))),
                ("03D_clahe_enhancement.png",   enhanced_pristine),
                ("03E_geometric_correction.png", self.processed),
                ("grayscale.png",               gray_pristine),
                ("gaussian_denoising.png",      steps_pristine.get("blurred", steps_pristine.get("denoised"))),
                ("clahe_enhancement.png",       enhanced_pristine),
                ("geometric_correction.png",    self.processed),
            ]
            # Dark-sample black-hat intermediate (only present when applied; uses the pristine version to avoid double shearing with 03E)
            if "blackhat" in steps_pristine and steps_pristine["blackhat"] is not None:
                bh = steps_pristine["blackhat"]
                # Normalise to 0-255 for display
                if bh.max() > 0:
                    bh_vis = np.uint8(np.clip(bh.astype(np.float32) * (255.0 / float(bh.max())), 0, 255))
                else:
                    bh_vis = bh
                preset.append(("03F_blackhat_enhancement.png", bh_vis))
            for fn, img in preset:
                if img is not None:
                    self._imwrite_safe(os.path.join(dir_prep, fn), img)
            
            # Canny edges + Hough lines
            if self.enhanced_no_grad is not None:
                canny = cv2.Canny(self.enhanced_no_grad, 50, 150)
                self._imwrite_safe(os.path.join(dir_prep, "04A_canny_edges.png"), canny)
                self._imwrite_safe(os.path.join(dir_prep, "canny_edges.png"), canny)
                
                hough_img = cv2.cvtColor(self.enhanced_no_grad, cv2.COLOR_GRAY2BGR)
                lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100,
                                        minLineLength=max(50, w // 4), maxLineGap=20)
                n_detected_lines = 0
                angles = []
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Keep only roughly horizontal lines (laminae)
                        dx, dy = x2 - x1, y2 - y1
                        if dx != 0:
                            angle = np.degrees(np.arctan2(dy, dx))
                            if abs(angle) <= 30:
                                cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                angles.append(angle)
                                n_detected_lines += 1
                self._imwrite_safe(os.path.join(dir_prep, "04B_hough_lines.png"), hough_img)
                self._imwrite_safe(os.path.join(dir_prep, "hough_lines.png"), hough_img)
                
                mean_angle = float(np.mean(angles)) if angles else 0.0
                geo_log = pd.DataFrame([{
                    "sample_id": sample_id,
                    "detected_line_number": n_detected_lines,
                    "mean_lamina_angle_deg": round(mean_angle, 3),
                    "rotation_angle_deg": round(float(getattr(self, 'alignment_angle', 0.0)), 3),
                    "correction_method": "hough+affine" if self.aligned else "none",
                    "correction_success": bool(self.aligned),
                }])
                geo_log.to_csv(os.path.join(dir_prep, "geometric_correction_log.csv"),
                              index=False, encoding='utf-8-sig')
            
            # Before/after correction comparison
            if self.gray is not None and self.processed is not None:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                axes[0].imshow(self.gray, cmap='gray')
                axes[0].set_title("Before correction")
                axes[0].axis('off')
                axes[1].imshow(self.processed, cmap='gray')
                axes[1].set_title("After correction" if self.aligned else "Preprocessed (no geom correction)")
                axes[1].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(dir_prep, "04C_before_after_correction.png"), dpi=150)
                plt.close()

            # 2D response maps: step-window response, Sobel-X gradient, black-hat
            try:
                self._export_response_maps(dir_prep)
            except Exception as e:
                print(f"  response-map export error: {e}")
                import traceback; traceback.print_exc()
            print("  01_preprocessing/ written")
        except Exception as e:
            print(f"  01_preprocessing/ error: {e}")
            import traceback; traceback.print_exc()
        
        # ---------- 02_scanline_detection ----------
        try:
            base_bgr = self.image.copy() if self.image is not None else cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
            
            # Main scan-line overlay
            main_overlay = base_bgr.copy()
            for sr in self.layers:
                y = int(sr["y"])
                if 0 <= y < h:
                    cv2.line(main_overlay, (0, y), (w - 1, y), (0, 0, 255), 1)
            self._imwrite_safe(os.path.join(dir_scan, "05A_main_scanlines_overlay.png"), main_overlay)
            self._imwrite_safe(os.path.join(dir_scan, "main_scanlines_overlay.png"), main_overlay)
            
            # Validation-line overlay
            val_lines = sorted(set(getattr(self, 'validation_lines', [])))
            val_overlay = base_bgr.copy()
            for vy in val_lines:
                if 0 <= vy < h:
                    cv2.line(val_overlay, (0, int(vy)), (w - 1, int(vy)), (255, 0, 0), 1)
            self._imwrite_safe(os.path.join(dir_scan, "05B_validation_lines_overlay.png"), val_overlay)
            self._imwrite_safe(os.path.join(dir_scan, "validation_lines_overlay.png"), val_overlay)
            
            # Combined figure
            combined = main_overlay.copy()
            for vy in val_lines:
                if 0 <= vy < h:
                    cv2.line(combined, (0, int(vy)), (w - 1, int(vy)), (255, 0, 0), 1)
            self._imwrite_safe(os.path.join(dir_scan, "05C_scanline_validation_overlay.png"), combined)
            
            # Scan-line coordinates CSV
            scan_rows = []
            sid = 0
            for sr in self.layers:
                sid += 1
                y = int(sr["y"])
                scan_rows.append({
                    "scanline_id": sid,
                    "line_type": "main",
                    "y_position_pixel": y,
                    "y_position_mm": round(y / self.pixel_per_mm, 3) if self.pixel_per_mm else "",
                    "line_length_pixel": w,
                })
            for vy in val_lines:
                sid += 1
                scan_rows.append({
                    "scanline_id": sid,
                    "line_type": "validation",
                    "y_position_pixel": int(vy),
                    "y_position_mm": round(vy / self.pixel_per_mm, 3) if self.pixel_per_mm else "",
                    "line_length_pixel": w,
                })
            pd.DataFrame(scan_rows).to_csv(
                os.path.join(dir_scan, "scanline_coordinates.csv"),
                index=False, encoding='utf-8-sig')
            
            # Representative scan-line gray/gradient curves (first 3) + data
            n_repr = min(3, len(self.layers))
            for i in range(n_repr):
                sr = self.layers[i]
                y = int(sr["y"])
                line_id = i + 1
                src_row = (self.gray[y, :].astype(np.float64)
                           if self.gray is not None else self.processed[y, :].astype(np.float64))
                grad_raw = np.abs(np.diff(src_row))
                grad_log = np.log1p(grad_raw)
                grad_smooth = gaussian_filter1d(grad_log, sigma=2)
                thresh = float(np.mean(grad_smooth) + 0.3 * np.std(grad_smooth))
                pts = list(sr.get("validated_points", sr["points"]))
                
                # Gray-profile figure
                plt.figure(figsize=(15, 4))
                plt.plot(src_row, color='black', linewidth=0.8)
                plt.title(f"Grayscale profile -- scanline {line_id} (y={y})")
                plt.xlabel("X (pixel)"); plt.ylabel("Gray value")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(dir_scan, f"06A_grayscale_profile_line_{line_id:03d}.png"), dpi=150)
                plt.close()
                
                # Gradient curve + dynamic threshold
                plt.figure(figsize=(15, 4))
                plt.plot(grad_smooth, color='steelblue', linewidth=0.9, label="log gradient")
                plt.axhline(thresh, color='r', linestyle='--', label=f"threshold={thresh:.2f}")
                for pt in pts:
                    if 0 <= pt < len(grad_smooth):
                        plt.axvline(pt, color='g', alpha=0.4, linewidth=0.6)
                plt.title(f"Gradient & Threshold - Scanline {line_id}")
                plt.xlabel("X (pixel)"); plt.ylabel("Gradient (log)")
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(dir_scan, f"06B_gradient_profile_line_{line_id:03d}.png"), dpi=150)
                plt.close()
                
                # Dynamic-threshold figure
                plt.figure(figsize=(15, 4))
                rolling_mean = gaussian_filter1d(grad_smooth, sigma=30)
                rolling_std = np.sqrt(gaussian_filter1d((grad_smooth - rolling_mean)**2, sigma=30))
                dyn_thresh = rolling_mean + 0.3 * rolling_std
                plt.plot(grad_smooth, color='steelblue', linewidth=0.8, label="log gradient")
                plt.plot(dyn_thresh, color='red', linestyle='--', linewidth=1, label="dynamic threshold")
                plt.title(f"Dynamic Threshold - Scanline {line_id}")
                plt.xlabel("X (pixel)"); plt.ylabel("Value")
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(dir_scan, f"06C_dynamic_threshold_line_{line_id:03d}.png"), dpi=150)
                plt.close()
                
                # Candidate-points figure
                plt.figure(figsize=(15, 4))
                plt.plot(src_row, color='black', linewidth=0.8, label="gray")
                for pt in pts:
                    if 0 <= pt < len(src_row):
                        plt.axvline(pt, color='red', alpha=0.6, linewidth=0.8)
                plt.title(f"Candidate Points - Scanline {line_id} ({len(pts)} points)")
                plt.xlabel("X (pixel)"); plt.ylabel("Gray value")
                plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(dir_scan, f"06D_candidate_points_line_{line_id:03d}.png"), dpi=150)
                plt.close()
                
                # Data CSV
                grad_full = np.concatenate([grad_smooth, [grad_smooth[-1] if len(grad_smooth) else 0]])
                dyn_full = np.concatenate([dyn_thresh, [dyn_thresh[-1] if len(dyn_thresh) else 0]])
                is_cand = np.zeros(len(src_row), dtype=int)
                for pt in pts:
                    if 0 <= pt < len(is_cand):
                        is_cand[pt] = 1
                df_line = pd.DataFrame({
                    "x_pixel": np.arange(len(src_row)),
                    "x_mm": (np.arange(len(src_row)) / self.pixel_per_mm).round(3) if self.pixel_per_mm else "",
                    "gray_value": src_row.round(2),
                    "gradient_value": grad_full.round(4),
                    "dynamic_threshold": dyn_full.round(4),
                    "is_candidate_point": is_cand,
                })
                df_line.to_csv(os.path.join(dir_scan, f"line_{line_id:03d}_profile.csv"),
                              index=False, encoding='utf-8-sig')
            
            # Candidate-boundary overlay + CSV
            cand_overlay = base_bgr.copy()
            cand_rows = []
            for i, sr in enumerate(self.layers):
                y = int(sr["y"])
                raw_pts = sr.get("points", [])
                val_pts = set(sr.get("validated_points", []))
                for pt in raw_pts:
                    color = (0, 255, 255)  # candidate (yellow)
                    cv2.circle(cand_overlay, (int(pt), y), 2, color, -1)
                    cand_rows.append({
                        "sample_id": sample_id,
                        "scanline_id": i + 1,
                        "x_pixel": int(pt),
                        "y_pixel": y,
                        "x_mm": round(pt / self.pixel_per_mm, 3) if self.pixel_per_mm else "",
                        "y_mm": round(y / self.pixel_per_mm, 3) if self.pixel_per_mm else "",
                        "candidate_type": "validated" if pt in val_pts else "raw",
                    })
            self._imwrite_safe(os.path.join(dir_scan, "07A_candidate_points_overlay.png"), cand_overlay)
            self._imwrite_safe(os.path.join(dir_scan, "candidate_points_overlay.png"), cand_overlay)
            if cand_rows:
                pd.DataFrame(cand_rows).to_csv(
                    os.path.join(dir_scan, "candidate_boundary_points.csv"),
                    index=False, encoding='utf-8-sig')
            print("  02_scanline_detection/ written")
        except Exception as e:
            print(f"  02_scanline_detection/ error: {e}")
            import traceback; traceback.print_exc()
        
        # ---------- 03_crossline_validation ----------
        try:
            base_bgr = self.image.copy() if self.image is not None else cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
            val_lines = sorted(set(getattr(self, 'validation_lines', [])))
            
            # Before validation
            before = base_bgr.copy()
            for sr in self.layers:
                y = int(sr["y"])
                for pt in sr.get("points", []):
                    cv2.circle(before, (int(pt), y), 2, (0, 255, 255), -1)
            self._imwrite_safe(os.path.join(dir_cross, "08A_before_crossline_validation.png"), before)
            self._imwrite_safe(os.path.join(dir_cross, "before_validation.png"), before)
            
            # After validation
            after = base_bgr.copy()
            for sr in self.layers:
                y = int(sr["y"])
                for pt in sr.get("validated_points", []):
                    cv2.circle(after, (int(pt), y), 3, (0, 0, 255), -1)
            self._imwrite_safe(os.path.join(dir_cross, "08B_after_crossline_validation.png"), after)
            self._imwrite_safe(os.path.join(dir_cross, "after_validation.png"), after)
            
            # Rejected points
            rejected = base_bgr.copy()
            for sr in self.layers:
                y = int(sr["y"])
                raw_set = set(sr.get("points", []))
                val_set = set(sr.get("validated_points", []))
                for pt in (raw_set - val_set):
                    cv2.circle(rejected, (int(pt), y), 2, (128, 128, 128), -1)
            self._imwrite_safe(os.path.join(dir_cross, "08C_rejected_noise_overlay.png"), rejected)
            self._imwrite_safe(os.path.join(dir_cross, "rejected_noise.png"), rejected)
            
            # Validated-boundary figure (same as after)
            self._imwrite_safe(os.path.join(dir_cross, "08D_validated_boundaries_overlay.png"), after)
            self._imwrite_safe(os.path.join(dir_cross, "validated_boundaries.png"), after)
            
            # Cross-line consistency illustration
            try:
                consistency_img = base_bgr.copy()
                for sr in self.layers:
                    y = int(sr["y"])
                    scores = sr.get("consistency_scores", {})
                    for pt in sr.get("validated_points", []):
                        s = scores.get(pt, 0)
                        color = (0, 255, 0) if s >= 2 else (0, 165, 255) if s >= 1 else (0, 0, 255)
                        cv2.circle(consistency_img, (int(pt), y), 3, color, -1)
                cv2.putText(consistency_img, "Green: high (>=2), Orange: mid, Red: low",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self._imwrite_safe(os.path.join(dir_cross, "08E_crossline_consistency_check.png"), consistency_img)
            except Exception:
                pass
            
            # crossline_validation_results.csv
            n_val = len(val_lines) if val_lines else 0
            cross_rows = []
            bid = 0
            for i, sr in enumerate(self.layers):
                y = int(sr["y"])
                raw_set = set(sr.get("points", []))
                val_set = set(sr.get("validated_points", []))
                scores = sr.get("consistency_scores", {})
                for pt in raw_set:
                    bid += 1
                    is_valid = pt in val_set
                    score = scores.get(pt, 0)
                    cross_rows.append({
                        "boundary_id": bid,
                        "main_scanline_id": i + 1,
                        "candidate_x_pixel": int(pt),
                        "candidate_y_pixel": y,
                        "consistency_score": score,
                        "matched_validation_lines": score,
                        "total_validation_lines": n_val,
                        "validation_ratio": round(score / n_val, 3) if n_val > 0 else "",
                        "is_valid_boundary": 1 if is_valid else 0,
                        "rejection_reason": "" if is_valid else ("low_consistency" if score < 1 else "fracture_zone"),
                    })
            if cross_rows:
                pd.DataFrame(cross_rows).to_csv(
                    os.path.join(dir_cross, "crossline_validation_results.csv"),
                    index=False, encoding='utf-8-sig')
            print("  03_crossline_validation/ written")
        except Exception as e:
            print(f"  03_crossline_validation/ error: {e}")
            import traceback; traceback.print_exc()
        
        # ---------- 04_results ----------
        try:
            # 10A original image (reused)
            if self.image is not None:
                self._imwrite_safe(os.path.join(dir_result, "10A_original_core.png"), self.image)
            
            # 10B automatic-detection overlay
            det_overlay = base_bgr.copy() if self.image is None else self.image.copy()
            for sr in self.layers:
                y = int(sr["y"])
                for pt in sr.get("validated_points", sr["points"]):
                    cv2.line(det_overlay, (int(pt), max(0, y - 8)), (int(pt), min(h - 1, y + 8)),
                            (0, 0, 255), 1)
            self._imwrite_safe(os.path.join(dir_result, "10B_auto_detection_overlay.png"), det_overlay)
            self._imwrite_safe(os.path.join(dir_result, "automatic_detection_overlay.png"), det_overlay)
            
            # 10C red thin-line version (paper-ready figure)
            red_only = self.image.copy() if self.image is not None else base_bgr.copy()
            for sr in self.layers:
                y = int(sr["y"])
                for pt in sr.get("validated_points", sr["points"]):
                    cv2.line(red_only, (int(pt), max(0, y - 5)), (int(pt), min(h - 1, y + 5)),
                            (0, 0, 255), 1)
            self._imwrite_safe(os.path.join(dir_result, "10C_detected_boundaries_red.png"), red_only)
            
            # 10D annotated version (id + weak-boundary mark)
            annot = det_overlay.copy()
            scores = sr.get("consistency_scores", {}) if self.layers else {}
            label_id = 0
            for sr in self.layers:
                y = int(sr["y"])
                local_scores = sr.get("consistency_scores", {})
                for pt in sr.get("validated_points", sr["points"]):
                    label_id += 1
                    s = local_scores.get(pt, 0)
                    color = (0, 255, 0) if s >= 2 else (0, 165, 255)
                    if label_id % 5 == 0:  # label every 5th point
                        cv2.putText(annot, str(label_id), (int(pt) + 2, y - 3),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            self._imwrite_safe(os.path.join(dir_result, "10D_detection_with_annotations.png"), annot)
            
            # Boundary-strength overlay + heatmap
            if not layer_df.empty:
                strength_overlay = self.image.copy() if self.image is not None else base_bgr.copy()
                max_s = max(1e-6, float(layer_df["strength"].max()))
                for _, row in layer_df.iterrows():
                    pt_x = int(row["position_x"])
                    y = int(row["scan_line_y"])
                    s = float(row["strength"])
                    norm = min(1.0, s / max_s)
                    color = (0, int(255 * (1 - norm)), int(255 * norm))
                    cv2.circle(strength_overlay, (pt_x, y), max(2, int(2 + norm * 3)), color, -1)
                self._imwrite_safe(os.path.join(dir_result, "09A_boundary_strength_overlay.png"), strength_overlay)
                
                # Strength heatmap
                try:
                    heat = np.zeros((h, w), dtype=np.float32)
                    for _, row in layer_df.iterrows():
                        pt_x = int(row["position_x"])
                        y = int(row["scan_line_y"])
                        s = float(row["strength"])
                        if 0 <= y < h and 0 <= pt_x < w:
                            cv2.circle(heat, (pt_x, y), 8, s, -1)
                    heat = cv2.GaussianBlur(heat, (31, 31), 0)
                    if heat.max() > 0:
                        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
                    else:
                        heat_norm = heat.astype(np.uint8)
                    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                    if self.image is not None:
                        heat_blend = cv2.addWeighted(self.image, 0.5, heat_color, 0.5, 0)
                    else:
                        heat_blend = heat_color
                    self._imwrite_safe(os.path.join(dir_result, "09B_boundary_strength_heatmap.png"), heat_blend)
                    self._imwrite_safe(os.path.join(dir_result, "boundary_strength_heatmap.png"), heat_blend)
                except Exception:
                    pass
                
                # ---- candidate_points_per_scanline.csv (candidate change-points per scan line; diagnostic) ----
                cand_rows = []
                for _, row in layer_df.iterrows():
                    cand_rows.append({
                        "sample_id": sample_id,
                        "candidate_id": row["lamina_id"],
                        "x_pixel": row["position_x"],
                        "y_pixel": row["scan_line_y"],
                        "spacing_to_next_in_line_mm": row.get("spacing_to_next_mm", ""),
                        "spacing_to_next_in_line_pixel": row.get("spacing_to_next_px", ""),
                        "boundary_strength": row.get("strength", 0),
                        "log_strength": row.get("log_strength", 0),
                        "delta_gray": row.get("delta_gray", 0),
                        "left_mean_gray": row.get("left_mean", ""),
                        "right_mean_gray": row.get("right_mean", ""),
                        "consistency_score": row.get("crossline_consistency", 0),
                        "depth_m": row.get("depth_m", ""),
                    })
                pd.DataFrame(cand_rows).to_csv(
                    os.path.join(dir_result, "candidate_points_per_scanline.csv"),
                    index=False, encoding='utf-8-sig')
                
                # ---- lamination_attributes.csv (one row per cross-line-clustered "unique lamina") ----
                # These are the "lamina attributes" that the paper should cite: each lamina has a unique id, a precise location, and support across multiple scan lines
                laminae = getattr(self, 'laminae', None) or []
                lam_settings = getattr(self, '_lamina_settings', None) or {}
                lam_rows = []
                
                # Aggregate candidate-point strength per unique lamina (mean delta_gray, mean log_strength)
                # Index layer_df by (x, y) so we can join back
                cand_by_xy = {}
                for _, r in layer_df.iterrows():
                    cand_by_xy[(int(r["position_x"]), int(r["scan_line_y"]))] = r
                
                # Sort the valid laminae by x_mean to build a global ordering for lamina_id
                valid_laminae = sorted([la for la in laminae if la.get("is_valid")],
                                       key=lambda la: la["x_mean"])
                
                for new_id, la in enumerate(valid_laminae, start=1):
                    # Look up layer_df by the cluster member points (x, y) (layer_df only contains the main scan lines,
                    # validation-line points are missing, so some lookups will not match)
                    member_strengths = []
                    member_log = []
                    member_dgray = []
                    member_cons = []
                    member_left = []
                    member_right = []
                    for (x_m, y_m) in la.get("member_points_xy", []):
                        key = (int(x_m), int(y_m))
                        if key in cand_by_xy:
                            r = cand_by_xy[key]
                            member_strengths.append(float(r.get("strength", 0) or 0))
                            member_log.append(float(r.get("log_strength", 0) or 0))
                            member_dgray.append(float(r.get("delta_gray", 0) or 0))
                            cs = r.get("crossline_consistency", 0)
                            try:
                                member_cons.append(float(cs))
                            except (TypeError, ValueError):
                                pass
                            lm = r.get("left_mean", None)
                            rm = r.get("right_mean", None)
                            try:
                                member_left.append(float(lm))
                                member_right.append(float(rm))
                            except (TypeError, ValueError):
                                pass
                    
                    # Spacing (thickness) to the next unique lamina
                    sp_px = la.get("spacing_to_next_px")
                    if sp_px is not None and self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                        sp_mm = sp_px / self.pixel_per_mm
                    elif sp_px is not None:
                        sp_mm = sp_px * (100.0 / w)
                    else:
                        sp_mm = None
                    cls_type = self._classify_layer(sp_mm) if sp_mm is not None and sp_mm > 0 else ""
                    
                    # Depth (linear interpolation of x_mean in the depth range)
                    if start_depth is not None and end_depth is not None and start_depth != end_depth:
                        depth_m = start_depth + (end_depth - start_depth) * (la["x_mean"] / w)
                    else:
                        depth_m = None
                    
                    lam_rows.append({
                        "sample_id": sample_id,
                        "lamina_id": new_id,
                        "x_mean_pixel": round(la["x_mean"], 1),
                        "x_min_pixel": la["x_min"],
                        "x_max_pixel": la["x_max"],
                        "x_std_pixel": round(la["x_std"], 2),
                        # 2D fit
                        "fit_slope_dx_per_dy": round(la.get("fit_slope", 0.0), 4),
                        "dip_angle_deg": round(la.get("dip_angle_deg", 0.0), 2),
                        "x_at_top_pixel": round(la.get("x_at_top", la["x_mean"]), 1),
                        "x_at_bottom_pixel": round(la.get("x_at_bottom", la["x_mean"]), 1),
                        "fit_mean_residual_px": round(la.get("mean_residual_px", 0.0), 2),
                        "fit_max_residual_px": round(la.get("max_residual_px", 0.0), 2),
                        # Support
                        "n_supporting_scanlines": la["n_support_lines"],
                        "n_supporting_main": la.get("n_support_main", 0),
                        "n_supporting_validation": la.get("n_support_validation", 0),
                        "n_total_scanlines": lam_settings.get("n_scan_lines", len(self.layers)),
                        "support_ratio": round(la["support_ratio"], 3),
                        # Thickness and class
                        "thickness_to_next_pixel": round(float(sp_px), 1) if sp_px is not None else "",
                        "thickness_to_next_mm": round(float(sp_mm), 3) if sp_mm is not None else "",
                        "lamination_type": cls_type,
                        # Boundary strength
                        "boundary_strength_mean": round(float(np.mean(member_strengths)), 4) if member_strengths else 0,
                        "boundary_strength_std": round(float(np.std(member_strengths)), 4) if len(member_strengths) > 1 else 0,
                        "log_strength_mean": round(float(np.mean(member_log)), 4) if member_log else 0,
                        "delta_gray_mean": round(float(np.mean(member_dgray)), 2) if member_dgray else 0,
                        "left_mean_gray": round(float(np.mean(member_left)), 1) if member_left else "",
                        "right_mean_gray": round(float(np.mean(member_right)), 1) if member_right else "",
                        "consistency_score_mean": round(float(np.mean(member_cons)), 2) if member_cons else 0,
                        "n_candidate_points": la["n_points_in_cluster"],
                        "depth_m": round(depth_m, 4) if depth_m is not None else "",
                    })
                
                lam_df_unique = pd.DataFrame(lam_rows)
                lam_df_unique.to_csv(
                    os.path.join(dir_result, "lamination_attributes.csv"),
                    index=False, encoding='utf-8-sig')
                
                # ---- boundary_strength_results.csv now per-unique-lamina (aligned with lamination_attributes) ----
                # The original per-candidate version is preserved as boundary_strength_per_candidate.csv
                bs_per_cand_rows = []
                max_g = max(1e-6, float(layer_df["delta_gray"].max())) if not layer_df.empty else 1.0
                for _, row in layer_df.iterrows():
                    spacing_mm = row["spacing_to_next_mm"] if row["spacing_to_next_mm"] != "" else 0
                    spacing_mm_val = float(spacing_mm) if spacing_mm != "" else 0
                    delta_g = float(row["delta_gray"])
                    norm_grad = delta_g / max_g
                    strength_formula = float(np.log1p(5 * norm_grad))
                    bs_per_cand_rows.append({
                        "candidate_id": row["lamina_id"],
                        "center_x_pixel": row["position_x"],
                        "center_y_pixel": row["scan_line_y"],
                        "in_line_thickness_mm": spacing_mm_val,
                        "delta_gray": delta_g,
                        "normalized_gradient": round(norm_grad, 4),
                        "boundary_strength": round(strength_formula, 4),
                        "consistency_score": row["crossline_consistency"],
                        "depth_m": row.get("depth_m", ""),
                    })
                pd.DataFrame(bs_per_cand_rows).to_csv(
                    os.path.join(dir_result, "boundary_strength_per_candidate.csv"),
                    index=False, encoding='utf-8-sig')
                
                # Overwrite boundary_strength_results.csv with the unique-lamina version
                if not lam_df_unique.empty:
                    bs_unique_rows = []
                    max_bs = max(1e-6, float(lam_df_unique["delta_gray_mean"].max()))
                    for _, row in lam_df_unique.iterrows():
                        dg = float(row["delta_gray_mean"])
                        norm = dg / max_bs
                        bs_unique_rows.append({
                            "lamina_id": row["lamina_id"],
                            "x_mean_pixel": row["x_mean_pixel"],
                            "thickness_to_next_mm": row["thickness_to_next_mm"],
                            "delta_gray_mean": dg,
                            "normalized_strength": round(norm, 4),
                            "boundary_strength": round(float(np.log1p(5 * norm)), 4),
                            "n_supporting_scanlines": row["n_supporting_scanlines"],
                            "support_ratio": row["support_ratio"],
                            "depth_m": row.get("depth_m", ""),
                        })
                    pd.DataFrame(bs_unique_rows).to_csv(
                        os.path.join(dir_result, "boundary_strength_results.csv"),
                    index=False, encoding='utf-8-sig')
                
                # ---- lamination_statistics.csv aggregated per "unique lamina" class ----
                stats_rows = []
                if not lam_df_unique.empty and "lamination_type" in lam_df_unique.columns:
                    valid_cls = lam_df_unique[lam_df_unique["lamination_type"] != ""]
                    if not valid_cls.empty:
                        for cls, sub in valid_cls.groupby("lamination_type"):
                            stats_rows.append({
                                "lamination_type": cls,
                                "count": len(sub),
                                "ratio_percent": round(len(sub) / len(valid_cls) * 100, 2),
                                "mean_thickness_mm": round(pd.to_numeric(
                                    sub["thickness_to_next_mm"], errors='coerce').mean(), 3),
                                "mean_boundary_strength": round(sub["boundary_strength_mean"].mean(), 4),
                                "mean_support_ratio": round(sub["support_ratio"].mean(), 3),
                            })
                pd.DataFrame(stats_rows).to_csv(
                    os.path.join(dir_result, "lamination_statistics.csv"),
                    index=False, encoding='utf-8-sig')
            print("  04_results/ written")
        except Exception as e:
            print(f"  04_results/ error: {e}")
            import traceback; traceback.print_exc()
        
        # ---------- 05_method_comparison ----------
        try:
            self._export_method_comparison(dir_method)
            print("  05_method_comparison/ written")
        except Exception as e:
            print(f"  05_method_comparison/ error: {e}")
        
        print(f"=== Modular paper export complete: {paper_dir} ===")
    def _export_response_maps(self, out_dir):
        """Export three 2D response maps used as paper process figures.

        Files (written into ``01_preprocessing/``):

          - 04D1_step_window_small / 04D2_..._medium / 04D3_..._large
              Local grayscale step-window response computed separately with the
              small / medium / large window of detector Method A. Small windows
              catch narrow laminae; large windows catch wide gentle changes.
          - 04D4_step_window_fused (== 04D_step_window_response)
              Multi-branch fusion (max of the three windows): a more stable
              candidate-boundary response map; strong boundaries are brighter.
          - 04D5_multi_window_panel.png
              Combined small / medium / large / fused panel with colour bars.
          - 04E_sobel_x_response.png / _color.png
              |Sobel_X| cross-lamina gradient magnitude map. Laminae are
              vertical, so their boundaries are vertical edges whose gradient
              lies along X; bright = strong vertical lamina edge.
          - 04F_blackhat_response.png / _color.png
              Black-hat output; emphasises dark thin bands and narrow dark
              lamina boundaries against a brighter matrix.
          - 04H_detection_input_fused.png / _color.png (alias detection_input_fused.png)
              The final fused image actually scanned by the detector
              (``self.processed``): the result after ALL enhancement branches
              are applied and fused (gamma + sigmoid -> CLAHE -> black-hat ->
              Sobel-X -> geometric correction). This is the most directly
              useful "fused" map for detection.
          - 04G_response_maps_panel.png
              Combined step / sobel-y / black-hat panel with colour bars.

        Each map is saved both as a raw grayscale PNG (bright = strong) and as
        a colour-mapped PNG (``*_color.png``, inferno: hotter = stronger).
        """
        from scipy.ndimage import gaussian_filter1d, convolve1d

        # Source image fed to detection (gradient-enhanced, aligned)
        proc = getattr(self, 'processed', None)
        if proc is None:
            print("  response maps skipped: no processed image")
            return
        proc_f = proc.astype(np.float64)
        h, w = proc_f.shape[:2]

        prep_meta = getattr(self, '_preprocess_meta', {}) or {}
        mlw = int(getattr(self, '_last_min_layer_width', 5) or 5)
        blur_size = int(prep_meta.get('blur_size', 5) or 5)

        def _norm_u8(arr):
            arr = np.asarray(arr, dtype=np.float64)
            mx = float(arr.max()) if arr.size else 0.0
            if mx > 0:
                arr = arr / mx * 255.0
            return np.clip(arr, 0, 255).astype(np.uint8)

        def _save_pair(basename, u8):
            self._imwrite_safe(os.path.join(out_dir, f"{basename}.png"), u8)
            color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
            self._imwrite_safe(os.path.join(out_dir, f"{basename}_color.png"), color)

        # ---- 1) Multi-scale step-window response (detection Method A, 2D) ----
        # Three branches: small / medium / large window. Small windows catch
        # narrow laminae; large windows catch wide gentle changes. Each branch
        # is saved separately, then fused (max across branches) into a single
        # more stable candidate-boundary response map.
        row_smooth = gaussian_filter1d(proc_f, sigma=1.5, axis=1)
        scales = [max(5, mlw // 2), max(8, mlw), max(15, mlw * 2)]
        branch_labels = ["small", "medium", "large"]
        branch_u8 = {}
        step_response = np.zeros_like(proc_f)
        for label, scale_win in zip(branch_labels, scales):
            lr_kernel = np.zeros(scale_win * 2, dtype=np.float64)
            lr_kernel[:scale_win] = -1.0 / scale_win
            lr_kernel[scale_win:] = 1.0 / scale_win
            sig = np.abs(convolve1d(row_smooth, lr_kernel, axis=1, mode='nearest'))
            step_response = np.maximum(step_response, sig)
            u8 = _norm_u8(sig)
            branch_u8[label] = (scale_win, u8)

        # Per-branch maps (window size annotated in the filename)
        _save_pair("04D1_step_window_small", branch_u8["small"][1])
        _save_pair("04D2_step_window_medium", branch_u8["medium"][1])
        _save_pair("04D3_step_window_large", branch_u8["large"][1])

        # Fused (max across the three windows): the stable candidate-boundary map
        step_u8 = _norm_u8(step_response)
        _save_pair("04D_step_window_response", step_u8)
        _save_pair("04D4_step_window_fused", step_u8)

        # Side-by-side panel: small / medium / large / fused
        try:
            fig, axes = plt.subplots(1, 4, figsize=(28, 6))
            multi_panels = [
                (branch_u8["small"][1], f"(a) Small window (w={branch_u8['small'][0]} px)\nnarrow laminae"),
                (branch_u8["medium"][1], f"(b) Medium window (w={branch_u8['medium'][0]} px)"),
                (branch_u8["large"][1], f"(c) Large window (w={branch_u8['large'][0]} px)\nwide gentle changes"),
                (step_u8, "(d) Fused response (max of a-c)\nstable candidate boundaries"),
            ]
            for ax, (img_u8, title) in zip(axes, multi_panels):
                im = ax.imshow(img_u8, cmap='inferno', aspect='auto', vmin=0, vmax=255)
                ax.set_title(title, fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.suptitle("Multi-window step response and multi-branch fusion",
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "04D5_multi_window_panel.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  multi-window panel error: {e}")

        # ---- 2) Cross-lamina gradient magnitude (Sobel-X) ----
        # Laminae are vertical (the core is laid horizontally), so their
        # boundaries are vertical edges whose gradient lies along X. Sobel-X
        # (dx=1, dy=0) lights up the laminae; Sobel-Y would respond to
        # horizontal edges and leave the laminae dark, so we use Sobel-X here.
        grad_src = getattr(self, 'enhanced_no_grad', None)
        if grad_src is None:
            grad_src = proc
        sobel_x = cv2.Sobel(grad_src, cv2.CV_64F, 1, 0, ksize=3)
        sobel_u8 = _norm_u8(np.abs(sobel_x))
        _save_pair("04E_sobel_x_response", sobel_u8)

        # ---- 3) Black-hat output (dark thin laminae / narrow dark bands) ----
        steps = getattr(self, '_preprocess_steps', {}) or {}
        blackhat = steps.get('blackhat')
        if blackhat is None:
            # Not stored (non-dark sample): compute on the CLAHE-enhanced image.
            # A vertical dark lamina is narrow along X, so the structuring element
            # must be HORIZONTAL (width N, height 1) to isolate it.
            bh_src = grad_src
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(5, blur_size * 3), 1))
            blackhat = cv2.morphologyEx(bh_src, cv2.MORPH_BLACKHAT, kernel)
        bh_u8 = _norm_u8(blackhat)
        _save_pair("04F_blackhat_response", bh_u8)

        # ---- 4) Final fused detection image (the image actually scanned) ----
        # This is ``self.processed`` -- the output after ALL preprocessing /
        # enhancement branches are applied and fused (gamma + sigmoid non-linear
        # stretch -> CLAHE -> black-hat blend -> Sobel-X gradient blend ->
        # geometric correction). It is the single image the detector actually
        # scans line by line, so it is the most directly useful "fused" map.
        if proc.dtype != np.uint8:
            detect_u8 = _norm_u8(proc_f)
        else:
            detect_u8 = proc
        _save_pair("04H_detection_input_fused", detect_u8)
        # Convenience alias so it is easy to find
        self._imwrite_safe(os.path.join(out_dir, "detection_input_fused.png"), detect_u8)

        # ---- Combined matplotlib panel with colour bars (paper-ready) ----
        try:
            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            panels = [
                (step_u8, "(a) Step-window response\n(local grayscale step; bright = strong boundary)"),
                (sobel_u8, "(b) Sobel-X gradient magnitude\n(cross-lamina; bright = vertical lamina edge)"),
                (bh_u8, "(c) Black-hat output\n(dark thin laminae / narrow dark bands)"),
            ]
            for ax, (img_u8, title) in zip(axes, panels):
                im = ax.imshow(img_u8, cmap='inferno', aspect='auto', vmin=0, vmax=255)
                ax.set_title(title, fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.suptitle("2D response maps used for lamina-boundary detection",
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "04G_response_maps_panel.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  response-map panel error: {e}")

        print("  response maps written: 04D step (small/medium/large/fused) / "
              "04E sobel-y / 04F black-hat / 04H detection-input-fused (+panels)")
    def _export_method_comparison(self, out_dir):
        """Build the Sobel/Canny/rule-based/proposed method comparison figure."""
        if self.gray is None or self.enhanced_no_grad is None:
            return
        src = self.enhanced_no_grad
        h, w = src.shape[:2]
        base_bgr = self.image.copy() if self.image is not None else cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        
        # Sobel baseline: use the cross-lamina X-derivative so the baseline
        # targets the same vertical laminae as our method (a fair comparison).
        sobel_x = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        sobel_abs = np.uint8(np.clip(np.abs(sobel_x), 0, 255))
        _, sobel_bin = cv2.threshold(sobel_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sobel_overlay = base_bgr.copy()
        sobel_overlay[sobel_bin > 0] = [0, 0, 255]
        self._imwrite_safe(os.path.join(out_dir, "12A_sobel_result.png"), sobel_overlay)
        
        # Canny
        canny = cv2.Canny(src, 50, 150)
        canny_overlay = base_bgr.copy()
        canny_overlay[canny > 0] = [0, 255, 255]
        self._imwrite_safe(os.path.join(out_dir, "12B_canny_result.png"), canny_overlay)
        
        # Rule-based: detect change-points on each scan line with a fixed threshold (mean+2*std)
        rule_overlay = base_bgr.copy()
        n_scan_lines = max(20, min(60, h // 20))
        spacing = max(1, h // (n_scan_lines + 1))
        rule_count = 0
        for ly in range(spacing, h - spacing, spacing):
            row = src[ly, :].astype(np.float64)
            grad = np.abs(np.diff(row))
            from scipy.signal import find_peaks as _fp
            if np.std(grad) > 0:
                th = np.mean(grad) + 2.0 * np.std(grad)
                pks, _ = _fp(grad, height=th, distance=5)
                for pk in pks:
                    cv2.circle(rule_overlay, (int(pk), ly), 2, (0, 255, 0), -1)
                    rule_count += 1
        self._imwrite_safe(os.path.join(out_dir, "12C_rule_based_result.png"), rule_overlay)
        
        # Proposed method (our result)
        proposed = base_bgr.copy()
        for sr in self.layers:
            y = int(sr["y"])
            for pt in sr.get("validated_points", sr["points"]):
                cv2.line(proposed, (int(pt), max(0, y - 6)), (int(pt), min(h - 1, y + 6)),
                        (0, 0, 255), 1)
        self._imwrite_safe(os.path.join(out_dir, "12E_proposed_method.png"), proposed)
        
        # Composite 2x2 figure
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes[0, 0].imshow(cv2.cvtColor(sobel_overlay, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Sobel"); axes[0, 0].axis('off')
            axes[0, 1].imshow(cv2.cvtColor(canny_overlay, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("Canny"); axes[0, 1].axis('off')
            axes[1, 0].imshow(cv2.cvtColor(rule_overlay, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f"Rule-based ({rule_count} pts)"); axes[1, 0].axis('off')
            n_proposed = sum(len(sr.get("validated_points", sr["points"])) for sr in self.layers)
            axes[1, 1].imshow(cv2.cvtColor(proposed, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title(f"Proposed ({n_proposed} pts)"); axes[1, 1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "method_comparison_summary.png"), dpi=150)
            plt.close()
        except Exception:
            pass
    def _classify_layer(self, thickness_mm):
        """Classify a lamina by the seven-tier scheme (mm).

        Rules (matching the paper):
          <1 mm        -> thin_lamina
          1-5 mm       -> lamina
          5-10 mm      -> thick_lamina
          10-50 mm     -> thin_layer (1-5 cm)
          50-100 mm    -> layer       (5-10 cm)
          100-500 mm   -> thick_layer (10-50 cm)
          >500 mm      -> massive     (>50 cm)
        """
        if thickness_mm < 1:
            return "thin_lamina(<1mm)"
        elif thickness_mm < 5:
            return "lamina(1-5mm)"
        elif thickness_mm < 10:
            return "thick_lamina(5-10mm)"
        elif thickness_mm < 50:
            return "thin_layer(1-5cm)"
        elif thickness_mm < 100:
            return "layer(5-10cm)"
        elif thickness_mm < 500:
            return "thick_layer(10-50cm)"
        else:
            return "massive(>50cm)"
    def _export_preprocessing_comparison(self, paper_dir):
        """Figure 1: preprocessing-step comparison."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original grayscale
        if hasattr(self, 'gray') and self.gray is not None:
            axes[0].imshow(self.gray, cmap='gray', aspect='auto')
            axes[0].set_title('(a) Grayscale', fontsize=12)
        else:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) > 2 else self.image
            axes[0].imshow(gray, cmap='gray', aspect='auto')
            axes[0].set_title('(a) Grayscale', fontsize=12)
        
        # After denoising
        gray_base = self.gray if hasattr(self, 'gray') and self.gray is not None else \
                    (cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) > 2 else self.image)
        denoised = cv2.bilateralFilter(gray_base, d=9, sigmaColor=75, sigmaSpace=75)
        axes[1].imshow(denoised, cmap='gray', aspect='auto')
        axes[1].set_title('(b) Bilateral denoised', fontsize=12)
        
        # CLAHE enhancement
        if hasattr(self, 'enhanced_no_grad') and self.enhanced_no_grad is not None:
            axes[2].imshow(self.enhanced_no_grad, cmap='gray', aspect='auto')
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(cv2.GaussianBlur(denoised, (5, 5), 0))
            axes[2].imshow(enhanced, cmap='gray', aspect='auto')
        axes[2].set_title('(c) CLAHE enhanced', fontsize=12)
        
        # Final processed result (with gradient fusion)
        if self.processed is not None:
            axes[3].imshow(self.processed, cmap='gray', aspect='auto')
        axes[3].set_title('(d) Gradient enhanced', fontsize=12)
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Image preprocessing pipeline', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "preprocessing_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Preprocessing comparison saved")
    def _export_scanline_profiles(self, paper_dir):
        """Figure 2: scan-line gray profile with detection-point markers."""
        n_lines = min(3, len(self.layers))
        if n_lines == 0:
            return
        
        fig, axes = plt.subplots(n_lines, 1, figsize=(14, 3 * n_lines))
        if n_lines == 1:
            axes = [axes]
        
        for idx in range(n_lines):
            layer_data = self.layers[idx]
            y = layer_data["y"]
            points = layer_data["points"]
            
            if self.processed is not None and y < self.processed.shape[0]:
                row = self.processed[y, :].astype(np.float64)
            else:
                continue
            
            ax = axes[idx]
            x_range = np.arange(len(row))
            ax.plot(x_range, row, 'b-', linewidth=0.8, alpha=0.7, label='Gray value')
            
            pts = layer_data.get("validated_points", points)
            for pt in pts:
                if pt < len(row):
                    ax.axvline(x=pt, color='red', linestyle='--', linewidth=0.6, alpha=0.6)
                    ax.plot(pt, row[pt], 'rv', markersize=5)
            
            ax.set_title(f'Gray profile of scan-line y={y} (detected {len(pts)} laminae)', fontsize=11)
            ax.set_xlabel('Lateral position (px)')
            ax.set_ylabel('Gray value')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Scan-line gray profile and lamina-boundary detection', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "scanline_profiles.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Scan-line profile saved")
    def _export_classification_pie(self, paper_dir, layer_df):
        """Figure 3: seven-tier lamina-classification pie chart."""
        cls_col = "spacing_class"
        if cls_col not in layer_df.columns:
            print("Missing spacing_class column; skipping pie chart")
            return
        valid_df = layer_df[layer_df[cls_col] != ""]
        if valid_df.empty:
            return
        cls_counts = valid_df[cls_col].value_counts()
        cls_order = ["thin_lamina(<1mm)", "lamina(1-5mm)", "thick_lamina(5-10mm)",
                     "thin_layer(1-5cm)", "layer(5-10cm)", "thick_layer(10-50cm)", "massive(>50cm)"]
        ordered_counts = pd.Series(dtype=int)
        for c in cls_order:
            if c in cls_counts.index:
                ordered_counts[c] = cls_counts[c]
        
        if ordered_counts.empty:
            return
        
        colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8', '#87CEEB', '#9370DB', '#C0C0C0']
        
        fig, ax = plt.subplots(figsize=(9, 7))
        wedges, texts, autotexts = ax.pie(
            ordered_counts.values,
            labels=ordered_counts.index,
            autopct='%1.1f%%',
            colors=colors[:len(ordered_counts)],
            startangle=90,
            pctdistance=0.75,
            textprops={'fontsize': 10}
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight('bold')
        ax.set_title('Lamina-type distribution', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "layer_classification_pie.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Classification pie chart saved")
    def _export_thickness_histogram(self, paper_dir, layer_df):
        """Figure 4: lamina-spacing histogram."""
        col = "spacing_to_next_mm"
        if col not in layer_df.columns:
            print("No spacing data; skipping histogram")
            return
        spacings = pd.to_numeric(layer_df[col], errors='coerce').dropna()
        spacings = spacings[spacings > 0].values
        if len(spacings) == 0:
            print("No valid spacing data; skipping histogram")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(spacings, bins=30, color='#4ECDC4', edgecolor='black',
                                    alpha=0.8, linewidth=0.5)
        
        mean_val = np.mean(spacings)
        median_val = np.median(spacings)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f} mm')
        ax.axvline(median_val, color='orange', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f} mm')
        
        calibrated = self.pixel_per_mm is not None and self.pixel_per_mm > 0
        unit_note = "calibrated" if calibrated else "estimated"
        ax.set_xlabel(f'Lamina spacing (mm, {unit_note})', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title = 'Lamina spacing histogram'
        if calibrated:
            title += f' (scale: {self.pixel_per_mm:.2f} px/mm)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        stats_text = (f'Count: {len(spacings)}\n'
                      f'Mean: {mean_val:.2f} mm\n'
                      f'Std: {np.std(spacings):.2f} mm\n'
                      f'Range: {np.min(spacings):.2f} - {np.max(spacings):.2f} mm')
        if calibrated:
            stats_text += f'\nScale: {self.pixel_per_mm:.2f} px/mm'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "layer_spacing_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Spacing histogram saved")
    def _export_strength_boxplot(self, paper_dir, layer_df):
        """Figure 5: per-class lamina-strength boxplot."""
        cls_col = "spacing_class"
        str_col = "strength"
        if cls_col not in layer_df.columns or str_col not in layer_df.columns:
            print("Missing class/strength columns; skipping boxplot")
            return
        valid_df = layer_df[layer_df[cls_col] != ""]
        cls_order = ["thin_lamina(<1mm)", "lamina(1-5mm)", "thick_lamina(5-10mm)",
                     "thin_layer(1-5cm)", "layer(5-10cm)", "thick_layer(10-50cm)", "massive(>50cm)"]
        present = [c for c in cls_order if c in valid_df[cls_col].values]
        if not present:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        data_groups = [valid_df[valid_df[cls_col] == c][str_col].values for c in present]
        
        bp = ax.boxplot(data_groups, labels=present, patch_artist=True, notch=True,
                        medianprops=dict(color='red', linewidth=2))
        
        colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8', '#87CEEB', '#9370DB', '#C0C0C0']
        for patch, color in zip(bp['boxes'], colors[:len(present)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Annotate the mean
        for i, grp in enumerate(data_groups):
            if len(grp) > 0:
                ax.plot(i + 1, np.mean(grp), 'D', color='green', markersize=6, zorder=5)
        
        ax.set_xlabel('Lamina class', fontsize=12)
        ax.set_ylabel('Edge strength', fontsize=12)
        ax.set_title('Edge-strength distribution by lamina class', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Median'),
                           Line2D([0], [0], marker='D', color='green', linestyle='None',
                                  markersize=6, label='Mean')]
        ax.legend(handles=legend_elements, fontsize=10)
        
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "layer_strength_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Strength boxplot saved")
    def _export_detection_overlay(self, paper_dir):
        """Figure 6: original + lamina overlay, with unique-lamina continuous figure."""
        overlay = self.image.copy()
        h, w = overlay.shape[:2]
        
        for layer_data in self.layers:
            y = layer_data["y"]
            cv2.line(overlay, (0, y), (w - 1, y), (0, 200, 0), 1)
            
            pts = layer_data.get("validated_points", layer_data["points"])
            for pt in pts:
                cv2.line(overlay, (pt, max(0, y - 8)), (pt, min(h - 1, y + 8)), (0, 0, 255), 2)
                cv2.circle(overlay, (pt, y), 3, (255, 0, 0), -1)
        
        cv2.imwrite(os.path.join(paper_dir, "detection_overlay.png"), overlay)
        
        # ====== Unique-lamina continuous figure: draw a single continuous line from top to bottom for each cross-line-clustered lamina ======
        # This is the figure that lets the user check whether change-points across scan lines actually connect into a line
        laminae = getattr(self, 'laminae', None) or []
        if laminae:
            unique_overlay = self.image.copy()
            # First draw scan-lines (light grey)
            for layer_data in self.layers:
                y = layer_data["y"]
                cv2.line(unique_overlay, (0, y), (w - 1, y), (180, 180, 180), 1)
            
            for la in laminae:
                if la.get("is_valid"):
                    # Use the 2D-fit line for the continuous lamina (tilt allowed)
                    x_top = int(round(la.get("x_at_top", la["x_mean"])))
                    x_bot = int(round(la.get("x_at_bottom", la["x_mean"])))
                    cv2.line(unique_overlay, (x_top, 0), (x_bot, h - 1), (0, 0, 220), 2)
                    # Mark blue dots on every supporting scan-line (member markers projected onto the fit)
                    slope = la.get("fit_slope", 0.0)
                    intercept = la.get("fit_intercept", la["x_mean"])
                    for y in la["support_lines_y"]:
                        x_on_line = int(round(slope * y + intercept))
                        if 0 <= x_on_line < w and 0 <= y < h:
                            cv2.circle(unique_overlay, (x_on_line, int(y)), 3, (220, 60, 0), -1)
                else:
                    # Invalid (fit failed) = grey "X" mark ("smudge")
                    x = int(round(la["x_mean"]))
                    if not (0 <= x < w):
                        continue
                    for y in la["support_lines_y"]:
                        if 0 <= y < h:
                            cy = int(y)
                            cv2.line(unique_overlay, (x - 4, cy - 4), (x + 4, cy + 4), (140, 140, 140), 1)
                            cv2.line(unique_overlay, (x - 4, cy + 4), (x + 4, cy - 4), (140, 140, 140), 1)
            self._imwrite_safe(
                os.path.join(paper_dir, "unique_laminae_overlay.png"),
                unique_overlay,
            )
            
            # matplotlib version: side-by-side panels with the original image and the continuous figure, plus a legend
            try:
                fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
                # Prefer the *pre-alignment* original for the left panel; fall back to self.image
                _left_img = getattr(self, 'image_original', None)
                if _left_img is None or (hasattr(_left_img, 'shape') and _left_img.shape != self.image.shape):
                    _left_img = self.image
                    _left_title = '(a) Original image'
                else:
                    _left_title = '(a) Original image (pre-alignment)'
                axes2[0].imshow(cv2.cvtColor(_left_img, cv2.COLOR_BGR2RGB), aspect='auto')
                axes2[0].set_title(_left_title, fontsize=12)
                axes2[0].set_xticks([]); axes2[0].set_yticks([])
                
                axes2[1].imshow(cv2.cvtColor(unique_overlay, cv2.COLOR_BGR2RGB), aspect='auto')
                n_valid = sum(1 for la in laminae if la.get("is_valid"))
                n_total = len(laminae)
                ls = getattr(self, '_lamina_settings', {}) or {}
                _right_suffix = ' (aligned)' if getattr(self, 'aligned', False) else ''
                axes2[1].set_title(
                    f'(b) Unique laminae from cross-line clustering{_right_suffix} (continuous red lines={n_valid}/{n_total}, '
                    f'requires >= {ls.get("min_support", "?")}/{ls.get("n_scan_lines", "?")} supporting lines)',
                    fontsize=11,
                )
                axes2[1].set_xticks([]); axes2[1].set_yticks([])
                plt.suptitle('Cross-scan-line consistency check for unique laminae', fontsize=13, fontweight='bold')
                plt.tight_layout()
                fig2.savefig(os.path.join(paper_dir, "unique_laminae_comparison.png"),
                             dpi=300, bbox_inches='tight')
                plt.close(fig2)
            except Exception as e:
                print(f"  Unique lamina matplotlib comparison error: {e}")
        
        # Also produce a matplotlib version for the paper
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img_rgb, aspect='auto')
        axes[0].set_title('(a) Original core scan', fontsize=12)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        axes[1].imshow(overlay_rgb, aspect='auto')
        axes[1].set_title('(b) Lamina detection overlay', fontsize=12)
        axes[1].set_xlabel('Lateral position (px)', fontsize=11)
        axes[1].set_yticks([])
        
        plt.suptitle('Core lamina detection comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "detection_overlay_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Detection overlay saved")
    def _export_density_curve(self, paper_dir, layer_df, start_depth=None, end_depth=None):
        """Figure 7: lamina density vs depth/position curve."""
        from scipy.ndimage import gaussian_filter1d
        
        w = self.width
        position_counts = np.zeros(w)
        for scan_result in self.layers:
            for pt in scan_result.get("validated_points", scan_result["points"]):
                if 0 <= pt < w:
                    position_counts[pt] += 1
        
        # Density via sliding window (lamina count per 100 px)
        window = max(20, w // 50)
        density = np.convolve(position_counts, np.ones(window) / window, mode='same')
        density_smooth = gaussian_filter1d(density, sigma=5)
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        if start_depth is not None and end_depth is not None and start_depth != end_depth:
            x_axis = np.linspace(start_depth, end_depth, w)
            x_label = 'Depth (m)'
            title = 'Lamina density vs depth'
        else:
            x_axis = np.arange(w)
            x_label = 'Lateral position (px)'
            title = 'Lamina density curve'
        
        ax.plot(x_axis, density_smooth, 'r-', linewidth=1.5)
        ax.fill_between(x_axis, density_smooth, color='red', alpha=0.2)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Lamina density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "density_depth_curve.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Density curve saved")
    def _export_intensity_heatmap(self, paper_dir):
        """Figure 8: lamina-strength heatmap."""
        from scipy.ndimage import gaussian_filter1d
        
        w = self.width
        intensity = np.zeros(w)
        
        for scan_result in self.layers:
            for pt in scan_result.get("validated_points", scan_result["points"]):
                if 0 <= pt < w:
                    intensity[pt] += 1
        
        smoothed = gaussian_filter1d(intensity.astype(float), sigma=5)
        if np.max(smoothed) > 0:
            smoothed = smoothed / np.max(smoothed)
        
        # 2D heatmap (lamina strength)
        heatmap = smoothed.reshape(1, -1)
        heatmap_multi = np.repeat(heatmap, max(1, w // 20), axis=0)
        
        fig, ax = plt.subplots(figsize=(14, 3))
        im = ax.imshow(heatmap_multi, aspect='auto', cmap='hot_r', extent=[0, w, 0, 1])
        ax.set_yticks([])
        ax.set_xlabel('Lateral position (px)', fontsize=11)
        ax.set_title('Lamina-strength heatmap', fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.25, shrink=0.8)
        cbar.set_label('Lamina strength (normalised)', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(os.path.join(paper_dir, "intensity_heatmap_single.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Intensity heatmap saved")
