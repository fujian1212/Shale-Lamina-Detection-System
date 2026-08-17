#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Result export."""

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


class ExportMixin:
    def export_results(self, output_dir="output"):
        """Export analysis results (compact mode: core data only).

        Outputs:
          - layer_info.xlsx           - Detailed lamina table.
          - summary.xlsx              - Summary statistics.
          - lamina_variation_curve.xlsx - Lamina variation curve (for cross-comparison
                                          with other measurements).
          - layer_detection.png       - Detection-result annotated image.

        Use ``export_paper_figures()`` for the full paper-figure pipeline.
        """
        print(f"=== Exporting results (compact mode) ===")
        print(f"Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Save the annotated detection-result image (main result only)
        try:
            self._export_detection_result_image(output_dir)
        except Exception as e:
            print(f"Error while saving detection-result image: {str(e)}")

        # Connection visualization (link cross-line candidate points via 2D-fit tilted lines)
        try:
            self._export_lamina_connections_image(output_dir)
        except Exception as e:
            print(f"Error while saving lamina-connections image: {str(e)}")

        # Save the non-linear narrow-band-enhanced grayscale image
        # (after gamma + sigmoid, before CLAHE). This figure clearly shows *why*
        # laminae that look blurred-together in raw dark mudstones/shales become
        # visible after our preprocessing -- referees / papers cite it directly.
        try:
            self._export_nonlinear_enhanced_image(output_dir)
        except Exception as e:
            print(f"Error while saving non-linear enhanced image: {str(e)}")

        # Make sure statistics have been computed
        if not self.layer_stats:
            try:
                self.calculate_statistics()
            except Exception as e:
                print(f"Error while computing statistics: {str(e)}")
                return None

        print(f"Detected laminae: {len(self.layers) if self.layers else 0}")

        # Column-name map for the user-facing Excel output
        column_mapping = {
            "scan_line": "scan_line",
            "position_x": "position_x_px",
            "position_y": "position_y_px",
            "spacing_to_next": "spacing_to_next_px",
            "spacing_mm": "spacing_mm",
            "layer_index": "layer_index",
            "strength": "strength",
            "log_strength": "log_strength",
            "consistency": "cross_line_consistency",
            "left_mean": "left_mean",
            "right_mean": "right_mean",
            "delta_gray": "delta_gray",
            "count": "count",
            "avg_spacing": "avg_spacing_px",
            "density": "density_per_100px",
            "total_count": "total_count",
            "avg_density": "avg_density_per_100px",
            "position": "position_px",
        }

        # Layer position info: each change point = one lamina row
        layer_info_rows = []
        print(f"Building layer position info; scan lines: {len(self.layers) if self.layers else 0}")

        for i, layer_data in enumerate(self.layers):
            y = layer_data["y"]
            pts = layer_data.get("validated_points", layer_data["points"])
            consistency_scores = layer_data.get("consistency_scores", {})
            print(f"Processing scan line {i+1}, y={y}, laminae={len(pts)}")

            for idx, pt_x in enumerate(pts):
                strength = 1.0
                left_mean_val = 0.0
                right_mean_val = 0.0
                delta_gray = 0.0

                if self.processed is not None and 0 <= y < self.processed.shape[0]:
                    half_w = max(5, 8)
                    x_lo = max(0, pt_x - half_w)
                    x_hi = min(self.processed.shape[1], pt_x + half_w + 1)

                    left_region = self.processed[y, x_lo:pt_x].astype(np.float64)
                    right_region = self.processed[y, pt_x:x_hi].astype(np.float64)

                    if len(left_region) > 0 and len(right_region) > 0:
                        left_mean_val = float(np.mean(left_region))
                        right_mean_val = float(np.mean(right_region))
                        delta_gray = float(abs(right_mean_val - left_mean_val))
                        strength = float(np.log1p(delta_gray) * 5.0)

                spacing = pts[idx + 1] - pt_x if idx < len(pts) - 1 else 0
                consistency = consistency_scores.get(pt_x, 0)

                row = {
                    "scan_line": i,
                    "position_x": pt_x,
                    "position_y": y,
                    "spacing_to_next": spacing,
                    "layer_index": idx,
                    "strength": round(strength, 3),
                    "log_strength": round(float(np.log1p(strength)), 3),
                    "consistency": consistency,
                    "left_mean": round(left_mean_val, 1),
                    "right_mean": round(right_mean_val, 1),
                    "delta_gray": round(delta_gray, 1),
                }

                if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                    row["spacing_mm"] = round(spacing / self.pixel_per_mm, 3) if spacing > 0 else 0

                layer_info_rows.append(row)

        print(f"Generated {len(layer_info_rows)} layer-info row(s)")

        # Write lamina-info Excel
        layer_info_path = os.path.join(output_dir, "layer_info.xlsx")
        print(f"Saving layer_info.xlsx to: {layer_info_path}")

        if not layer_info_rows:
            print(f"Warning: no laminae detected; creating an empty file")
            empty_df = pd.DataFrame(columns=list(column_mapping.keys()))
            empty_df.rename(columns=column_mapping).to_excel(layer_info_path, index=False)
        else:
            try:
                layer_info_df = pd.DataFrame(layer_info_rows)

                # Also save the raw-English-column copy (used internally by batch merging)
                layer_info_df.to_excel(
                    os.path.join(output_dir, "layer_info_en.xlsx"), index=False)

                # User-facing version (English columns by default)
                layer_info_df.rename(columns=column_mapping).to_excel(
                    layer_info_path, index=False)
                print(f"Lamina data saved: {layer_info_path} ({len(layer_info_df)} rows)")
            except Exception as e:
                print(f"Error while writing layer_info.xlsx: {str(e)}")
                import traceback
                traceback.print_exc()

        # Summary statistics
        summary_path = os.path.join(output_dir, "summary.xlsx")
        try:
            summary_dict = dict(self.layer_stats["summary"])
            # Stringify array-valued fields so Excel cells accept them
            list_keys = [k for k, v in summary_dict.items() if isinstance(v, list)]
            for k in list_keys:
                summary_dict[k] = str(summary_dict[k])
            summary_df = pd.DataFrame([summary_dict])
            summary_df.to_excel(summary_path, index=False)
            print(f"Summary statistics saved: {summary_path}")
        except Exception as e:
            print(f"Error while writing summary.xlsx: {str(e)}")

        # Unique-lamina table (each row = one valid lamina after cross-line clustering)
        try:
            lamina_df = self.layer_stats.get("laminae")
            if lamina_df is not None and len(lamina_df) > 0:
                lamina_path = os.path.join(output_dir, "lamina_summary.xlsx")
                lamina_df.to_excel(lamina_path, index=False)
                # Also save a "valid only" copy for paper use
                if "is_valid_lamina" in lamina_df.columns:
                    valid_only = lamina_df[lamina_df["is_valid_lamina"] == "yes"]
                    if len(valid_only) > 0:
                        valid_only.to_excel(
                            os.path.join(output_dir, "lamina_summary_valid.xlsx"),
                            index=False,
                        )
                print(f"Unique-lamina data saved: {lamina_path} ({len(lamina_df)} rows)")
        except Exception as e:
            print(f"Error while writing lamina_summary.xlsx: {str(e)}")

        # Position statistics (batch-merge stage needs this file for image-width info)
        position_path = os.path.join(output_dir, "position_info.xlsx")
        try:
            if "position" in self.layer_stats and self.layer_stats["position"] is not None \
                    and len(self.layer_stats["position"]) > 0:
                self.layer_stats["position"].to_excel(position_path, index=False)
                print(f"Position statistics saved: {position_path}")
            else:
                pd.DataFrame(columns=["position_px", "density_per_100px",
                                       "strength", "strength_normalized"]).to_excel(position_path, index=False)
        except Exception as e:
            print(f"Error while writing position_info.xlsx: {str(e)}")

        # Export the lamina variation curve (useful for cross-comparison with
        # geochemistry / well-log measurements)
        try:
            variation_path = os.path.join(output_dir, "lamina_variation_curve.xlsx")
            variation_rows = []
            if self.image is not None:
                img_w = self.image.shape[1]
                # Aggregate detections across scan lines along the x-axis
                bin_size = max(5, img_w // 200)
                n_bins = img_w // bin_size

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

                    n_lines = len(self.layers) if self.layers else 1
                    row = {
                        "position_px": round(x_center, 1),
                        "bin_start": x_start,
                        "bin_end": x_end,
                        "lamina_count": count_in_bin,
                        "lamina_density_per_line": round(count_in_bin / n_lines, 3),
                        "avg_strength": round(strength_sum / count_in_bin, 3) if count_in_bin > 0 else 0,
                        "avg_consistency": round(consistency_sum / count_in_bin, 2) if count_in_bin > 0 else 0,
                    }

                    if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                        row["position_mm"] = round(x_center / self.pixel_per_mm, 2)

                    variation_rows.append(row)

                if variation_rows:
                    pd.DataFrame(variation_rows).to_excel(variation_path, index=False)
                    print(f"Lamina-variation curve saved: {variation_path}")
        except Exception as e:
            print(f"Error while writing lamina-variation curve: {str(e)}")

        print(f"=== Result export complete ===")
        return layer_info_path
