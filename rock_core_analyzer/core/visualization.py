#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Result visualization."""

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


class VisualizationMixin:
    def _export_nonlinear_enhanced_image(self, output_dir):
        """Save the non-linear narrow-band-enhanced grayscale image.

        Source: the intermediate output of preprocessing after the gamma + sigmoid
        stretch and before CLAHE (``self.gray_nonlinear_enhanced``). If dark-core
        enhancement was not triggered for this sample, the figure falls back to
        the raw grayscale; the file is still written to keep the pipeline consistent.

        Papers can cite this figure directly to compare *raw grayscale* vs. *enhanced*
        contrast on dark samples, demonstrating that the sigmoid narrow-band non-linear
        amplification reveals previously invisible lamina boundaries.
        """
        nle = getattr(self, 'gray_nonlinear_enhanced', None)
        if nle is None:
            return
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        meta = getattr(self, '_preprocess_meta', {}) or {}
        sig = meta.get('sigmoid_stretch', {}) or {}
        sig2 = meta.get('sigmoid_stretch_secondary', {}) or {}
        tier = meta.get('enhancement_tier', '?')

        # Single image (uint8 grayscale): write directly through OpenCV
        ok1 = self._imwrite_safe(str(output_path / "nonlinear_enhanced.png"), nle)
        if ok1:
            extra = (f" [tier={tier}, gamma={meta.get('tier_auto_gamma', 0):.2f}, "
                     f"sigmoid k={sig.get('steepness', 0):.0f}, "
                     f"x{sig.get('center_amplification', 0):.1f}]")
            print(f"Non-linear enhanced image saved: {output_path / 'nonlinear_enhanced.png'}{extra}")

        # Three-panel comparison: raw / non-linear stretch / post-CLAHE (+unsharp).
        # This is the image actually fed to the lamina detection step, so the
        # visual gap between the panels mirrors what the detector sees.
        gray_src = getattr(self, 'gray', None)
        enhanced_final = getattr(self, 'enhanced_no_grad', None)
        if gray_src is None or nle is None:
            return
        try:
            n_panels = 3 if (enhanced_final is not None and
                             enhanced_final.shape == gray_src.shape) else 2
            fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
            if n_panels == 2:
                axes = list(axes)
            axes[0].imshow(gray_src, cmap='gray', aspect='auto', vmin=0, vmax=255)
            axes[0].set_title(
                f"(a) Raw grayscale\nmean={meta.get('image_mean', 0):.1f}, "
                f"std={meta.get('image_std', 0):.1f}",
                fontsize=11)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            # Panel (b): pure non-linear stretch (gamma + sigmoid pair).
            axes[1].imshow(nle, cmap='gray', aspect='auto', vmin=0, vmax=255)
            sig_band = f"[{sig.get('lo', 0):.0f}, {sig.get('hi', 0):.0f}]" if sig.get('applied') else "n/a"
            second_pass = ""
            if sig2.get('applied'):
                second_pass = (f"\n+ secondary sigmoid k={sig2.get('steepness', 0):.0f} "
                               f"x{sig2.get('center_amplification', 0):.1f}")
            b_title = (f"(b) Gamma {meta.get('tier_auto_gamma', 0):.2f} + sigmoid "
                       f"k={sig.get('steepness', 0):.0f} x{sig.get('center_amplification', 0):.1f}\n"
                       f"band={sig_band}, "
                       f"mean={meta.get('post_nonlinear_mean', 0):.1f}, "
                       f"std={meta.get('post_nonlinear_std', 0):.1f}"
                       f"{second_pass}")
            axes[1].set_title(b_title, fontsize=10)
            axes[1].set_xticks([]); axes[1].set_yticks([])

            # Panel (c): the image actually fed to detection
            if n_panels == 3:
                axes[2].imshow(enhanced_final, cmap='gray', aspect='auto', vmin=0, vmax=255)
                c_title = (f"(c) + CLAHE (clip={meta.get('clahe_clip_effective', 0):.1f}, "
                           f"grid={tuple(meta.get('clahe_grid_effective', [0, 0]))})"
                           + (" + unsharp" if meta.get('unsharp_applied') else "")
                           + (" + black-hat" if meta.get('blackhat_applied') else "")
                           + "\n(image actually used by the detector)")
                axes[2].set_title(c_title, fontsize=10)
                axes[2].set_xticks([]); axes[2].set_yticks([])

            plt.suptitle(
                f"Adaptive non-linear enhancement pipeline (strength tier={tier})",
                fontsize=13, fontweight='bold')
            plt.tight_layout()
            fig.savefig(str(output_path / "nonlinear_enhanced_comparison.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Non-linear enhancement comparison saved: {output_path / 'nonlinear_enhanced_comparison.png'}")
        except Exception as e:
            print(f"Failed to export non-linear enhancement comparison: {e}")
    def _export_detection_result_image(self, output_dir):
        """Export the annotated detection-result image (compact mode only)."""
        if not self.layers or self.image is None:
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        plt.figure(figsize=(12, 10))
        plt.imshow(self.image, aspect='auto')

        for layer_data in self.layers:
            y = layer_data["y"]
            plt.axhline(y=y, color='blue', linestyle='--', alpha=0.3, linewidth=0.5)
            pts = layer_data.get("validated_points", layer_data["points"])
            for pt in pts:
                plt.plot([pt, pt], [y - 6, y + 6], color='red', linewidth=2, alpha=0.8)
                plt.plot(pt, y, 'o', color='yellow', markersize=3)

        plt.title("Rock-core lamina detection result")
        plt.xlabel("Horizontal position (px)")
        plt.ylabel("Vertical position (px)")
        plt.tight_layout()
        plt.savefig(str(output_path / "layer_detection.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Detection-result image saved: {output_path / 'layer_detection.png'}")
    def _export_lamina_connections_image(self, output_dir):
        """Export the 2D-fit lamina connection image.

        - Green solid line = valid lamina (fit line through the clustered points,
          may be slightly tilted).
        - Yellow dots = detected points on every supporting scan line.
        - Gray crosses = isolated points whose fit failed (treated as smudges).
        """
        if self.image is None:
            return
        laminae = getattr(self, 'laminae', None) or []
        if not laminae:
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        canvas = self.image.copy()
        h, w = canvas.shape[:2]

        # Draw scan lines / validation lines first (faint gray)
        for layer_data in self.layers:
            y = layer_data["y"]
            cv2.line(canvas, (0, y), (w - 1, y), (200, 200, 200), 1)
        for vr in getattr(self, '_validation_results', []) or []:
            y = int(vr["y"])
            cv2.line(canvas, (0, y), (w - 1, y), (220, 220, 220), 1)

        n_valid = 0
        n_invalid = 0
        for la in laminae:
            x_top = int(round(la["x_at_top"]))
            x_bot = int(round(la["x_at_bottom"]))
            if la.get("is_valid"):
                # Fit line (green, drawn at the actual dip angle)
                cv2.line(canvas, (x_top, 0), (x_bot, h - 1), (0, 200, 0), 2)
                # Supporting points (yellow filled circles)
                slope = la["fit_slope"]
                intercept = la["fit_intercept"]
                for y in la["support_lines_y"]:
                    x_on_line = int(round(slope * y + intercept))
                    if 0 <= x_on_line < w and 0 <= y < h:
                        cv2.circle(canvas, (x_on_line, int(y)), 4, (0, 255, 255), -1)
                n_valid += 1
            else:
                # Fit failed -> smudge (gray cross)
                for y in la["support_lines_y"]:
                    if 0 <= int(la["x_mean"]) < w and 0 <= y < h:
                        cx = int(la["x_mean"]); cy = int(y)
                        cv2.line(canvas, (cx - 4, cy - 4), (cx + 4, cy + 4), (140, 140, 140), 1)
                        cv2.line(canvas, (cx - 4, cy + 4), (cx + 4, cy - 4), (140, 140, 140), 1)
                n_invalid += 1

        ok = self._imwrite_safe(str(output_path / "lamina_connections.png"), canvas)
        if ok:
            print(f"Lamina-connection visualization saved: {output_path / 'lamina_connections.png'} "
                  f"(valid laminae={n_valid}, failed fits/smudges={n_invalid})")

        # matplotlib version: left = raw image, right = connection image with legend
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            # Left panel prefers the *pre-alignment* original (if alignment ran);
            # otherwise falls back to the current ``self.image``. Without this both
            # panels would show the warped image and visually look "sheared twice".
            left_img = getattr(self, 'image_original', None)
            if left_img is None or (hasattr(left_img, 'shape') and left_img.shape != self.image.shape):
                left_img = self.image
                left_title = '(a) Original image'
            else:
                left_title = '(a) Original image (before alignment)'
            axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB), aspect='auto')
            axes[0].set_title(left_title, fontsize=12)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), aspect='auto')
            ls = getattr(self, '_lamina_settings', {}) or {}
            right_label_suffix = ' (aligned)' if getattr(self, 'aligned', False) else ''
            axes[1].set_title(
                f'(b) Lamina connections{right_label_suffix} '
                f'(green={n_valid} valid, gray X={n_invalid} failed fits)\n'
                f'requires >= {ls.get("min_support", "?")}/{ls.get("n_scan_lines", "?")} '
                f'line support, dip <= {ls.get("max_dip_angle_deg", 0):.0f} deg, '
                f'>=80% points residual <= {ls.get("max_residual_px", 0):.1f} px',
                fontsize=11,
            )
            axes[1].set_xticks([]); axes[1].set_yticks([])
            plt.suptitle('Cross-scan-line 2D fitting: only connected lines count as laminae',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            fig.savefig(str(output_path / "lamina_connections_comparison.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Lamina-connection comparison plot failed (matplotlib): {e}")
    def visualize_layers(self, output_dir="output"):
        """Visualize the detected laminae (full version: density / intensity / histograms)."""
        if not self.layers:
            print("No laminae detected; cannot generate visualizations.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Annotated detection image
        plt.figure(figsize=(12, 10))
        plt.imshow(self.image, aspect='auto')

        # Draw scan lines and detected change points
        for i, layer_data in enumerate(self.layers):
            y = layer_data["y"]
            plt.axhline(y=y, color='blue', linestyle='--', alpha=0.5, linewidth=1)

            pts = layer_data.get("validated_points", layer_data["points"])
            for pt in pts:
                plt.plot([pt, pt], [y - 6, y + 6], color='red', linewidth=2, alpha=0.8)
                plt.plot(pt, y, 'o', color='yellow', markersize=3)

        plt.title("Rock-core lamina detection result")
        plt.xlabel("Horizontal position (px)")
        plt.ylabel("Vertical position (px)")
        plt.axis('on')
        plt.tight_layout()
        plt.savefig(str(output_path / "layer_detection.png"), dpi=150, bbox_inches='tight')
        plt.close()  # free memory

        # Make sure statistics exist; compute them lazily if needed
        if not self.layer_stats:
            print("Statistics not present; computing now...")
            try:
                self.calculate_statistics()
            except Exception as e:
                print(f"Error while computing statistics: {str(e)}")
                return

        # Verify the position dataframe exists
        if not self.layer_stats or "position" not in self.layer_stats:
            print("Position statistics missing; cannot generate density / intensity plots.")
            return

        position_data = self.layer_stats["position"]

        # Check the required columns
        required_columns = ["position_px", "density_per_100px", "strength_normalized"]
        missing_columns = [col for col in required_columns if col not in position_data.columns]

        if missing_columns:
            print(f"Position statistics missing required columns: {missing_columns}")
            return

        # Bail out if the dataframe is empty
        if len(position_data) == 0:
            print("Position statistics are empty; cannot generate density / intensity plots.")
            return

        # Lamina-density plot
        try:
            plt.figure(figsize=(10, 6))

            # Use depth as x-axis when available
            depth_column = None
            for col in ["depth_m"]:
                if col in position_data.columns:
                    depth_column = col
                    break

            if depth_column:
                x_data = position_data[depth_column]
                x_label = "Depth (m)"
                title = "Lamina density vs. depth"
                print(f"Depth data detected (column={depth_column}); using depth as x-axis")
            else:
                x_data = position_data["position_px"]
                x_label = "Horizontal position (px)"
                title = "Lamina density curve"
                print("No depth data; using pixel position as x-axis")

            plt.plot(x_data, position_data["density_per_100px"], 'r-', linewidth=2)
            plt.fill_between(x_data, position_data["density_per_100px"], color='red', alpha=0.3)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel("Lamina density")
            plt.grid(True)
            plt.savefig(str(output_path / "layer_density.png"))
            plt.close()
            print(f"Density plot saved: {output_path / 'layer_density.png'}")
        except Exception as e:
            print(f"Error while generating density plot: {str(e)}")
            plt.close()

        # Lamina-intensity plot
        try:
            plt.figure(figsize=(10, 6))

            depth_column = None
            for col in ["depth_m"]:
                if col in position_data.columns:
                    depth_column = col
                    break

            if depth_column:
                x_positions = position_data[depth_column]
                x_label = "Depth (m)"
                title = "Lamina intensity vs. depth"
                print(f"Depth data detected (column={depth_column}); using depth as x-axis for intensity plot")

                plt.plot(x_positions, position_data["strength_normalized"], 'b-', linewidth=1.5, label='Lamina intensity')
                plt.fill_between(x_positions, position_data["strength_normalized"], color='skyblue', alpha=0.4)

            else:
                x_positions = position_data["position_px"]
                x_label = "Horizontal position (px)"
                title = "Lamina intensity curve (with edge fill)"
                print(f"No depth data; using pixel position as x-axis for intensity plot")

                plt.plot(x_positions, position_data["strength_normalized"], 'b-', linewidth=1.5, label='Lamina intensity')
                plt.fill_between(x_positions, position_data["strength_normalized"], color='skyblue', alpha=0.4)

                # Highlight the edge-fill regions (pixel mode only)
                intensity_values = position_data["strength_normalized"]
                width = len(x_positions)
                edge_filter_percent = 0.05
                edge_filter_pixels = max(10, int(width * edge_filter_percent))

                if edge_filter_pixels > 0 and width > edge_filter_pixels * 2:
                    left_indices = x_positions < edge_filter_pixels
                    if np.any(left_indices):
                        plt.fill_between(x_positions[left_indices], intensity_values[left_indices],
                                       color='orange', alpha=0.3, label='Edge-fill region')

                    right_indices = x_positions >= (width - edge_filter_pixels)
                    if np.any(right_indices):
                        plt.fill_between(x_positions[right_indices], intensity_values[right_indices],
                                       color='orange', alpha=0.3)

                    # Draw the boundary markers
                    plt.axvline(x=edge_filter_pixels, color='red', linestyle='--', alpha=0.7, linewidth=1)
                    plt.axvline(x=width-edge_filter_pixels, color='red', linestyle='--', alpha=0.7, linewidth=1)

            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel("Lamina intensity (normalized)")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(str(output_path / "layer_intensity.png"))
            plt.close()
            print(f"Intensity plot saved: {output_path / 'layer_intensity.png'}")
        except Exception as e:
            print(f"Error while generating intensity plot: {str(e)}")
            plt.close()

        # Lamina-spacing histogram
        try:
            all_spacings_vis = []
            for layer_data in self.layers:
                pts = layer_data.get("validated_points", layer_data["points"])
                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        sp = pts[i + 1] - pts[i]
                        if sp > 0:
                            all_spacings_vis.append(sp)

            if all_spacings_vis:
                plt.figure(figsize=(10, 6))
                plt.hist(all_spacings_vis, bins=20, color='green', alpha=0.7, edgecolor='black')
                plt.title("Lamina-spacing distribution")
                plt.xlabel("Lamina spacing (px)")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)

                mean_sp = np.mean(all_spacings_vis)
                plt.axvline(mean_sp, color='red', linestyle='--', linewidth=2, label=f'Mean spacing: {mean_sp:.1f}')
                plt.legend()

                plt.tight_layout()
                plt.savefig(str(output_path / "layer_spacing_histogram.png"))
                plt.close()
                print(f"Lamina-spacing histogram saved")
            else:
                print("No valid lamina-spacing data; cannot generate histogram.")
        except Exception as e:
            print(f"Error while generating lamina-spacing histogram: {str(e)}")
            plt.close()
