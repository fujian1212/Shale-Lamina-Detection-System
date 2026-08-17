#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistical analysis."""

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


class StatisticsMixin:
    def calculate_statistics(self):
        """Compute lamina statistics.

        Returns:
            ``(summary_dict, detailed_df, position_df)``.
        """
        print(f"=== Computing statistics ===")

        if not self.layers:
            print(f"Warning: no lamina data found; running detection again...")
            self.detect_layers()

        print(f"Current lamina-data count: {len(self.layers) if self.layers else 0}")

        # Collect lamina counts and adjacent-lamina spacings on every scan line
        all_spacings = []  # adjacent-lamina spacings (px)
        layer_counts = []
        for scan_result in self.layers:
            pts = scan_result.get("validated_points", scan_result["points"])
            layer_counts.append(len(pts))
            if len(pts) >= 2:
                spacings = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
                all_spacings.extend(spacings)

        print(f"Laminae per scan line: {layer_counts}")
        print(f"Collected {len(all_spacings)} lamina-spacing sample(s)")

        if self.image is not None:
            width = self.image.shape[1]
            density = sum(layer_counts) / (len(self.layers) * width) if self.layers else 0
        else:
            width = 0
            density = 0

        print(f"Image width: {width}")
        print(f"Computed lamina density: {density}")

        # Cross-scan-line consistency statistics
        all_consistency = []
        for scan_result in self.layers:
            scores = scan_result.get("consistency_scores", {})
            pts = scan_result.get("validated_points", scan_result["points"])
            for pt in pts:
                all_consistency.append(scores.get(pt, 0))

        avg_consistency = np.mean(all_consistency) if all_consistency else 0
        high_consistency_ratio = (
            sum(1 for c in all_consistency if c >= 2) / len(all_consistency) * 100
            if all_consistency else 0
        )

        # Spacing coefficient of variation
        spacing_cv = (np.std(all_spacings) / np.mean(all_spacings) * 100) if all_spacings and np.mean(all_spacings) > 0 else 0

        # ====== Unique-lamina metrics from cross-line clustering (recommended) ======
        laminae = getattr(self, 'laminae', None) or []
        lamina_settings = getattr(self, '_lamina_settings', None) or {}
        valid_laminae = [la for la in laminae if la.get("is_valid")]
        n_unique_laminae = len(valid_laminae)

        valid_sorted = sorted(valid_laminae, key=lambda la: la["x_mean"])
        unique_spacings_px = [
            valid_sorted[i + 1]["x_mean"] - valid_sorted[i]["x_mean"]
            for i in range(len(valid_sorted) - 1)
        ]

        avg_support = (
            float(np.mean([la["n_support_lines"] for la in valid_laminae]))
            if valid_laminae else 0.0
        )
        avg_support_ratio = (
            float(np.mean([la["support_ratio"] for la in valid_laminae]))
            if valid_laminae else 0.0
        )

        # Average dip angle / residual (computed from the fits of valid laminae)
        avg_dip = (
            float(np.mean([la.get("dip_angle_deg", 0) for la in valid_laminae]))
            if valid_laminae else 0.0
        )
        avg_res = (
            float(np.mean([la.get("mean_residual_px", 0) for la in valid_laminae]))
            if valid_laminae else 0.0
        )

        stats = {
            # -- Primary indicators: unique laminae (post 2D-fit clustering) --
            "unique_laminae_cluster": n_unique_laminae,
            "total_candidate_clusters": lamina_settings.get("n_clusters", 0),
            "cluster_tolerance_px": lamina_settings.get("tolerance_px", 0),
            "min_support_lines": lamina_settings.get("min_support", 0),
            "n_scan_lines": len(self.layers),  # legacy field: number of main scan lines
            "n_scan_lines_total": lamina_settings.get("n_scan_lines", len(self.layers)),
            "n_main_scan_lines": lamina_settings.get("n_main_scan_lines", len(self.layers)),
            "n_validation_lines": lamina_settings.get("n_validation_lines", 0),
            "avg_support_lines_per_lamina": round(avg_support, 2),
            "avg_cross_line_support_ratio": round(avg_support_ratio, 3),
            "avg_dip_angle_deg": round(avg_dip, 2),
            "avg_fit_residual_px": round(avg_res, 2),
            "max_allowed_dip_deg": lamina_settings.get("max_dip_angle_deg", 0),
            "max_allowed_residual_px": lamina_settings.get("max_residual_px", 0),
            "rejected_by_support": lamina_settings.get("rejected_by_support", 0),
            "rejected_by_dip": lamina_settings.get("rejected_by_dip", 0),
            "rejected_by_residual": lamina_settings.get("rejected_by_residual", 0),
            # -- Candidate-point level (per-scan-line totals; diagnostic only) --
            "candidate_change_points_total": sum(layer_counts),
            "candidate_points_per_scan_line": layer_counts,
            "avg_candidate_points_per_line": round(float(np.mean(layer_counts)), 2) if layer_counts else 0,
            "lamina_density_per_pixel": density,
            # -- Spacing statistics (from unique laminae) --
            "avg_lamina_spacing_px": round(float(np.mean(unique_spacings_px)), 2) if unique_spacings_px else 0,
            "max_lamina_spacing_px": int(np.max(unique_spacings_px)) if unique_spacings_px else 0,
            "min_lamina_spacing_px": int(np.min(unique_spacings_px)) if unique_spacings_px else 0,
            "lamina_spacing_std_px": round(float(np.std(unique_spacings_px)), 2) if unique_spacings_px else 0,
            "spacing_cv_percent": (
                round(float(np.std(unique_spacings_px)) / float(np.mean(unique_spacings_px)) * 100, 1)
                if unique_spacings_px and np.mean(unique_spacings_px) > 0 else 0
            ),
            # -- Consistency (per-candidate; reflects raw detection quality) --
            "avg_cross_line_consistency": round(avg_consistency, 2),
            "high_consistency_ratio_percent": round(high_consistency_ratio, 1),
        }

        # Legacy compatibility: keep ``total_laminae`` aliased to the unique count
        stats["total_laminae"] = n_unique_laminae

        if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
            if unique_spacings_px:
                mm_spacings = [s / self.pixel_per_mm for s in unique_spacings_px]
                stats["avg_lamina_spacing_mm"] = round(float(np.mean(mm_spacings)), 3)
                stats["max_lamina_spacing_mm"] = round(float(np.max(mm_spacings)), 3)
                stats["min_lamina_spacing_mm"] = round(float(np.min(mm_spacings)), 3)
                stats["lamina_spacing_std_mm"] = round(float(np.std(mm_spacings)), 3)
            stats["scale_px_per_mm"] = self.pixel_per_mm

        print(f"Summary statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")

        # Detailed DataFrame: each change point = one lamina entry
        rows = []
        for i, scan_result in enumerate(self.layers):
            y = scan_result["y"]
            pts = scan_result.get("validated_points", scan_result["points"])

            row_data = {"scan_line": i + 1, "row_y": y, "lamina_count": len(pts)}

            if len(pts) > 0:
                row_data["lamina_positions"] = str(pts)
                if len(pts) >= 2:
                    spacings = [pts[j + 1] - pts[j] for j in range(len(pts) - 1)]
                    row_data["avg_spacing_px"] = np.mean(spacings)
                    row_data["max_spacing_px"] = np.max(spacings)
                    row_data["min_spacing_px"] = np.min(spacings)
                    if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                        row_data["avg_spacing_mm"] = np.mean(spacings) / self.pixel_per_mm
                        row_data["max_spacing_mm"] = np.max(spacings) / self.pixel_per_mm
                        row_data["min_spacing_mm"] = np.min(spacings) / self.pixel_per_mm
                else:
                    row_data["avg_spacing_px"] = 0
                    row_data["max_spacing_px"] = 0
                    row_data["min_spacing_px"] = 0
            else:
                row_data["lamina_positions"] = ""
                row_data["avg_spacing_px"] = 0
                row_data["max_spacing_px"] = 0
                row_data["min_spacing_px"] = 0

            rows.append(row_data)

        detailed_df = pd.DataFrame(rows)

        # New: per-x-position lamina statistics
        width = self.image.shape[1]
        position_stats = []

        # Position -> count map
        position_counts = {}
        for scan_result in self.layers:
            for pos in scan_result.get("validated_points", scan_result["points"]):
                if pos in position_counts:
                    position_counts[pos] += 1
                else:
                    position_counts[pos] = 1

        # Per-x-position intensity array
        x_positions = np.arange(width)
        raw_intensity = np.zeros(width)

        # Edge-filter parameters (mirroring ``detect_layers``)
        edge_filter_percent = 0.05
        edge_filter_pixels = max(10, int(width * edge_filter_percent))

        print(f"Computing per-x-position intensity, image width={width}")
        print(f"Edge-filter regions: 0-{edge_filter_pixels} and {width-edge_filter_pixels}-{width}")

        # Accumulate +1 of intensity at every detected change point
        total_points_count = 0
        valid_region_points = 0

        for scan_result in self.layers:
            for pos in scan_result.get("validated_points", scan_result["points"]):
                total_points_count += 1
                if 0 <= pos < width:
                    raw_intensity[pos] += 1
                    if edge_filter_pixels <= pos < (width - edge_filter_pixels):
                        valid_region_points += 1

        print(f"Total points processed: {total_points_count}, in valid region: {valid_region_points}")

        # Gaussian smoothing for the intensity curve
        sigma = 5  # Gaussian std-dev; controls smoothing strength
        smoothed_intensity = gaussian_filter1d(raw_intensity, sigma=sigma) if len(raw_intensity) > 0 else raw_intensity

        if edge_filter_pixels > 0:
            # Intensity stats inside the valid region
            valid_region_intensity = smoothed_intensity[edge_filter_pixels:width-edge_filter_pixels]

            if len(valid_region_intensity) > 0 and np.sum(valid_region_intensity) > 0:
                # Option 1: extend with boundary values
                left_boundary_value = smoothed_intensity[edge_filter_pixels]
                right_boundary_value = smoothed_intensity[width-edge_filter_pixels-1]

                # Option 2: average intensity inside the valid region
                avg_valid_intensity = np.mean(valid_region_intensity[valid_region_intensity > 0])

                # Fill the left edge: weighted blend of boundary value and average
                for i in range(edge_filter_pixels):
                    # Farther from the boundary -> use more of the average
                    weight = i / edge_filter_pixels  # in [0, 1]
                    fill_value = (1 - weight) * avg_valid_intensity * 0.5 + weight * left_boundary_value
                    smoothed_intensity[i] = fill_value

                # Fill the right edge with a symmetric blend
                for i in range(width-edge_filter_pixels, width):
                    distance_from_boundary = i - (width-edge_filter_pixels)
                    weight = distance_from_boundary / edge_filter_pixels  # in [0, 1]
                    fill_value = (1 - weight) * right_boundary_value + weight * avg_valid_intensity * 0.5
                    smoothed_intensity[i] = fill_value

                print(f"Edge fill done: left boundary={left_boundary_value:.3f}, right boundary={right_boundary_value:.3f}, average={avg_valid_intensity:.3f}")
            else:
                print(f"Valid region intensity is zero; falling back to a default fill value")
                # If the valid region has no intensity, fill with a tiny default
                default_value = 0.1
                smoothed_intensity[:edge_filter_pixels] = default_value
                smoothed_intensity[width-edge_filter_pixels:] = default_value

        # Normalize intensities
        if len(smoothed_intensity) > 0 and np.max(smoothed_intensity) > 0:
            normalized_intensity = smoothed_intensity / np.max(smoothed_intensity)
        else:
            normalized_intensity = smoothed_intensity

        # Per-position lamina density = fraction of scan lines that have a change point here
        num_scan_lines = len(self.layers)

        # Per-position DataFrame
        for pos in range(width):
            count = position_counts.get(pos, 0)
            density = count / num_scan_lines if num_scan_lines > 0 else 0

            position_stats.append({
                "position_px": pos,
                "density_per_100px": density * 100,  # per-100-pixel density
                "strength": smoothed_intensity[pos],
                "strength_normalized": normalized_intensity[pos]
            })

        position_df = pd.DataFrame(position_stats)

        # ====== Unique-lamina table (one row per clustered lamina) ======
        lamina_rows = []
        for la in laminae:
            row = {
                "lamina_id": la["lamina_id"],
                "x_pos_px_mean": round(la["x_mean"], 1),
                "x_pos_px_median": round(la["x_median"], 1),
                "x_pos_range_min": la["x_min"],
                "x_pos_range_max": la["x_max"],
                "x_pos_std_px": round(la["x_std"], 2),
                "dip_angle_deg": round(la.get("dip_angle_deg", 0.0), 2),
                "fit_slope_dx_dy": round(la.get("fit_slope", 0.0), 4),
                "fit_max_residual_px": round(la.get("max_residual_px", 0.0), 2),
                "fit_mean_residual_px": round(la.get("mean_residual_px", 0.0), 2),
                "x_at_top_px": round(la.get("x_at_top", la["x_mean"]), 1),
                "x_at_bottom_px": round(la.get("x_at_bottom", la["x_mean"]), 1),
                "n_support_lines_total": la["n_support_lines"],
                "n_support_main": la.get("n_support_main", 0),
                "n_support_validation": la.get("n_support_validation", 0),
                "support_ratio": round(la["support_ratio"], 3),
                "n_points_in_cluster": la["n_points_in_cluster"],
                "is_valid_lamina": "yes" if la["is_valid"] else "no",
                "rejection_reasons": "; ".join(la.get("rejection_reasons", [])),
                "support_lines_y": ";".join(str(y) for y in la["support_lines_y"]),
            }
            sp_px = la.get("spacing_to_next_px")
            if la["is_valid"] and sp_px is not None:
                row["spacing_to_next_px"] = round(float(sp_px), 2)
            else:
                row["spacing_to_next_px"] = ""
            if self.pixel_per_mm is not None and self.pixel_per_mm > 0:
                row["x_pos_mm"] = round(la["x_mean"] / self.pixel_per_mm, 3)
                if la["is_valid"] and sp_px is not None:
                    row["spacing_to_next_mm"] = round(float(sp_px) / self.pixel_per_mm, 3)
                else:
                    row["spacing_to_next_mm"] = ""
            lamina_rows.append(row)
        lamina_df = pd.DataFrame(lamina_rows)

        # Save all statistical results
        self.layer_stats = {
            "summary": stats,
            "detailed": detailed_df,
            "position": position_df,
            "laminae": lamina_df,
        }

        return stats, detailed_df, position_df
