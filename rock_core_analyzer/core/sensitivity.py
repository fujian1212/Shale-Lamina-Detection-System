#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parameter sensitivity analysis and ablation study.

Design notes:
  This module reports reproducible, interpretable, unambiguous metrics in the
  absence of an expert-annotated ground truth. Every "comparison against
  baseline" uses the **user's current detection result** as the reference, so
  the paper must explicitly state
  ``reference = "default-parameter detection on the same image"`` to avoid
  readers misinterpreting the numbers as accuracy / recall against expert
  ground truth.

  For each variant (different parameter or module combination) the following
  metrics are produced:
    n_detected               Number of detected lamina edges (after validation).
    mean_consistency         Mean cross-line consistency score.
    mean_strength            Mean edge strength.
    processing_time_s        Processing time (s).
    n_matched_vs_reference   Points matched against the reference within tolerance.
    match_ratio_vs_reference n_matched / n_reference (fraction of reference points reproduced).
    n_extra_vs_reference     Points in the current result that did not match the reference.
    n_missed_vs_reference    Reference points not covered by the current result.
    mean_position_offset_px  Mean pixel offset between matched pairs.
    mean_position_offset_mm  Mean millimetre offset between matched pairs (requires calibration).

  Matching rule: for every main scan line ``y``, given the reference list
  ``R = {r_1, ..., r_m}`` and the current list ``C = {c_1, ..., c_n}``, greedily
  pair entries in ascending order; ``|r_i - c_j| <= tolerance_px`` counts as a
  match, and each point can be matched at most once.
"""

import os
import json
import time
import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


DEFAULT_PARAMS = {
    "blur_size": 5,
    "clahe_clip": 2.0,
    "clahe_grid": (8, 8),
    "threshold_method": "otsu",
    "min_layer_width": 5,
    "scan_line_count": 5,
    "min_validation_lines": 2,
    "align_core": True,
    "alignment_angle": 0.0,
    "disable_clahe": False,
}


def _collect_points(detector, use_validated=True):
    """Collect detection points as ``{y: sorted list of x}``."""
    result = {}
    for sr in getattr(detector, "layers", []) or []:
        y = int(sr["y"])
        if use_validated:
            pts = sr.get("validated_points", sr.get("points", []))
        else:
            pts = sr.get("points", [])
        result[y] = sorted(set(int(p) for p in pts))
    return result


def _collect_metrics(detector, use_validated=True):
    """Aggregate per-point strength and consistency."""
    strengths = []
    consistencies = []
    for sr in getattr(detector, "layers", []) or []:
        y = int(sr["y"])
        pts = sr.get("validated_points", sr["points"]) if use_validated else sr.get("points", [])
        scores = sr.get("consistency_scores", {})
        if detector.processed is None:
            continue
        h_img, w_img = detector.processed.shape[:2]
        if y < 0 or y >= h_img:
            continue
        for pt in pts:
            consistencies.append(scores.get(pt, 0))
            hw = 8
            x_lo = max(0, int(pt) - hw)
            x_hi = min(w_img, int(pt) + hw + 1)
            left = detector.processed[y, x_lo:int(pt)].astype(np.float64)
            right = detector.processed[y, int(pt):x_hi].astype(np.float64)
            if len(left) > 0 and len(right) > 0:
                strengths.append(float(np.log1p(abs(float(right.mean()) - float(left.mean()))) * 5.0))
    return strengths, consistencies


def _match_against_reference(current_pts, ref_pts, tolerance_px, y_tolerance_px=None):
    """2D greedy 1:1 matching. Returns ``(matched, extra, missing, x_offsets, y_offsets)``.

    Match reference points ``(yr, xr)`` and current points ``(yc, xc)`` in 2D:
      if ``|xr - xc| <= tolerance_px`` and ``|yr - yc| <= y_tolerance_px`` they
      may be paired.
    Greedy strategy: order candidate pairs by Euclidean distance ascending; each
    point can be paired at most once.

    ``y_tolerance_px`` defaults to ``tolerance_px``. Experiments that change the
    scan-line density may relax it to one scan-line interval to absorb y-axis drift.

    ``x_offsets`` / ``y_offsets`` record the per-axis distance for each pair.
    """
    if y_tolerance_px is None:
        y_tolerance_px = tolerance_px

    flat_ref = [(int(y), int(x), "ref", i)
                for y, xs in ref_pts.items() for i, x in enumerate(xs)]
    flat_cur = [(int(y), int(x), "cur", i)
                for y, xs in current_pts.items() for i, x in enumerate(xs)]

    n_ref = len(flat_ref)
    n_cur = len(flat_cur)
    if n_ref == 0 or n_cur == 0:
        return 0, n_cur, n_ref, [], []

    candidate_pairs = []
    for ri, (yr, xr, _, _) in enumerate(flat_ref):
        for ci, (yc, xc, _, _) in enumerate(flat_cur):
            dx = abs(xc - xr)
            dy = abs(yc - yr)
            if dx <= tolerance_px and dy <= y_tolerance_px:
                dist = (dx * dx + dy * dy) ** 0.5
                candidate_pairs.append((dist, ri, ci, dx, dy))

    candidate_pairs.sort(key=lambda p: p[0])

    used_ref = [False] * n_ref
    used_cur = [False] * n_cur
    matched = 0
    x_offs = []
    y_offs = []
    for dist, ri, ci, dx, dy in candidate_pairs:
        if used_ref[ri] or used_cur[ci]:
            continue
        used_ref[ri] = True
        used_cur[ci] = True
        matched += 1
        x_offs.append(dx)
        y_offs.append(dy)

    missing = sum(1 for u in used_ref if not u)
    extra = sum(1 for u in used_cur if not u)
    return matched, extra, missing, x_offs, y_offs


class SensitivityMixin:

    def _run_detection_variant(self, override, use_validated_for_metrics=True):
        """Run one variant detection on the same image without disturbing the current detector state.

        Args:
            override: Dict overriding ``DEFAULT_PARAMS``.
            use_validated_for_metrics: Whether to use validated points for metrics.
                Set to ``False`` for the "no cross-line validation" ablation.

        Returns:
            ``(variant_detector, elapsed_s, use_validated_for_metrics)``
        """
        from .detector import RockCoreLayerDetector

        params = dict(DEFAULT_PARAMS)
        params.update(override)
        if isinstance(params.get("clahe_grid"), list):
            params["clahe_grid"] = tuple(params["clahe_grid"])

        t0 = time.time()
        det = RockCoreLayerDetector(self.image_path)
        det.output_dir = os.path.join(
            getattr(self, "output_dir", "."),
            "_variant_tmp"
        )
        det.save_diagnostics = False
        os.makedirs(det.output_dir, exist_ok=True)

        det.preprocess_image(
            blur_size=params["blur_size"],
            clahe_clip=params["clahe_clip"],
            clahe_grid=params["clahe_grid"],
            disable_clahe=params["disable_clahe"],
        )
        det.detect_layers(
            threshold_method=params["threshold_method"],
            min_layer_width=params["min_layer_width"],
            scan_line_count=params["scan_line_count"],
            min_validation_lines=params["min_validation_lines"],
            align_core=params["align_core"],
            alignment_angle=params["alignment_angle"],
        )
        elapsed = time.time() - t0
        return det, elapsed, use_validated_for_metrics

    def _variant_metrics(self, det, elapsed, use_validated, ref_pts, tolerance_px,
                         y_tolerance_px=None):
        """Pull the unified set of metrics used in this study from one detection."""
        cur_pts = _collect_points(det, use_validated=use_validated)
        n_detected = sum(len(v) for v in cur_pts.values())
        strengths, consistencies = _collect_metrics(det, use_validated=use_validated)

        matched, extra, missing, x_offs, y_offs = _match_against_reference(
            cur_pts, ref_pts, tolerance_px, y_tolerance_px=y_tolerance_px,
        )
        n_ref = sum(len(v) for v in ref_pts.values())
        match_ratio = matched / n_ref if n_ref > 0 else float("nan")
        mean_x_off = float(np.mean(x_offs)) if x_offs else float("nan")
        mean_y_off = float(np.mean(y_offs)) if y_offs else float("nan")
        ppmm = getattr(det, "pixel_per_mm", None) or getattr(self, "pixel_per_mm", None)
        if ppmm and ppmm > 0 and x_offs:
            mean_off_mm = float(np.mean(x_offs) / ppmm)
        else:
            mean_off_mm = float("nan")

        return {
            "n_detected": n_detected,
            "mean_consistency": round(float(np.mean(consistencies)), 4) if consistencies else 0.0,
            "mean_strength": round(float(np.mean(strengths)), 4) if strengths else 0.0,
            "processing_time_s": round(float(elapsed), 3),
            "n_reference_points": n_ref,
            "n_matched_vs_reference": matched,
            "match_ratio_vs_reference": round(match_ratio, 4) if not np.isnan(match_ratio) else "",
            "n_extra_vs_reference": extra,
            "n_missed_vs_reference": missing,
            "mean_position_offset_px": round(mean_x_off, 3) if not np.isnan(mean_x_off) else "",
            "mean_position_offset_mm": round(mean_off_mm, 4) if not np.isnan(mean_off_mm) else "",
            "mean_y_offset_px": round(mean_y_off, 3) if not np.isnan(mean_y_off) else "",
        }

    def run_parameter_sensitivity(self, out_dir, tolerance_px=10):
        """Parameter sensitivity analysis.

        Tested parameters (matching paper Fig. 5):
          - Gaussian kernel size: 3, 5, 7, 9
          - CLAHE clip limit:     1.0, 1.5, 2.0, 2.5, 3.0
          - Scan-line interval:   H/20, H/30, H/40, H/50, H/60
          - Validation threshold: 1, 2, 3, 4, 5

        Reference = current ``self.layers`` produced under the default parameters.
        """
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== Parameter sensitivity analysis -> {out_dir} ===")

        if not self.layers:
            print("Current detector has no results; run detect_layers first")
            return out_dir

        ref_pts = _collect_points(self, use_validated=True)
        n_ref = sum(len(v) for v in ref_pts.values())
        h = int(self.height)

        test_configs = [
            ("gaussian_kernel", [3, 5, 7, 9], "blur_size"),
            ("clahe_clip_limit", [1.0, 1.5, 2.0, 2.5, 3.0], "clahe_clip"),
            ("scanline_interval", [20, 30, 40, 50, 60], "scan_line_count"),
            ("min_validation_lines", [1, 2, 3, 4, 5], "min_validation_lines"),
        ]

        all_rows = []
        for param_name, values, override_key in test_configs:
            print(f"  Testing {param_name}: {values}")
            for v in values:
                override = {}
                if override_key == "scan_line_count":
                    override["scan_line_count"] = max(2, int(h / v))
                    label_value = f"H/{v}"
                    # When the scan-line density changes, loosen the y-tolerance to
                    # 1/20 of the image height to absorb positional drift
                    y_tol = max(tolerance_px, h // 20)
                else:
                    override[override_key] = (int(v) if isinstance(v, int) else float(v))
                    label_value = v
                    y_tol = tolerance_px

                try:
                    det, elapsed, used_val = self._run_detection_variant(override)
                    metrics = self._variant_metrics(det, elapsed, used_val, ref_pts,
                                                   tolerance_px, y_tolerance_px=y_tol)
                except Exception as e:
                    print(f"    {param_name}={v} failed: {e}")
                    metrics = {
                        "n_detected": "",
                        "mean_consistency": "",
                        "mean_strength": "",
                        "processing_time_s": "",
                        "n_reference_points": n_ref,
                        "n_matched_vs_reference": "",
                        "match_ratio_vs_reference": "",
                        "n_extra_vs_reference": "",
                        "n_missed_vs_reference": "",
                        "mean_position_offset_px": "",
                        "mean_position_offset_mm": "",
                        "mean_y_offset_px": "",
                    }

                row = {
                    "parameter_name": param_name,
                    "parameter_value": label_value,
                    "parameter_value_numeric": v,
                    "y_tolerance_px": y_tol,
                    **metrics,
                }
                all_rows.append(row)
                print(f"    {param_name}={label_value} -> n_detected={metrics['n_detected']}, "
                      f"match_ratio={metrics['match_ratio_vs_reference']}")

        # Clean up the temp directory
        tmp = os.path.join(getattr(self, "output_dir", "."), "_variant_tmp")
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(out_dir, "parameter_sensitivity_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        try:
            df.to_excel(os.path.join(out_dir, "parameter_sensitivity_results.xlsx"), index=False)
        except Exception:
            pass

        # Plots
        try:
            self._plot_sensitivity(out_dir, df, n_ref, tolerance_px)
        except Exception as e:
            print(f"  Error while plotting sensitivity: {e}")

        # Description file
        self._write_sensitivity_readme(out_dir, tolerance_px, n_ref)

        print(f"=== Parameter sensitivity analysis complete: {csv_path} ===")
        return out_dir

    def _plot_sensitivity(self, out_dir, df, n_ref, tolerance_px):
        """Produce one line-plot per parameter plus a four-panel summary."""
        params_info = [
            ("gaussian_kernel",      "13A_gaussian_kernel_sensitivity.png",
             "Gaussian kernel size (px)"),
            ("clahe_clip_limit",     "13B_clahe_clip_sensitivity.png",
             "CLAHE clip limit"),
            ("scanline_interval",    "13C_scanline_interval_sensitivity.png",
             "Scan-line interval (H/N)"),
            ("min_validation_lines", "13D_validation_threshold_sensitivity.png",
             "Min validation lines"),
        ]

        for param_name, fname, xlabel in params_info:
            sub = df[df["parameter_name"] == param_name].copy()
            if sub.empty:
                continue
            x_labels = sub["parameter_value"].astype(str).tolist()
            x_pos = np.arange(len(x_labels))

            n_det = pd.to_numeric(sub["n_detected"], errors="coerce").fillna(0).values
            ratio = pd.to_numeric(sub["match_ratio_vs_reference"], errors="coerce").fillna(0).values * 100
            offset = pd.to_numeric(sub["mean_position_offset_px"], errors="coerce").fillna(0).values
            t_proc = pd.to_numeric(sub["processing_time_s"], errors="coerce").fillna(0).values

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Parameter sensitivity -- {param_name}\n"
                         f"reference = default-parameter detection "
                         f"(n={n_ref} points, tolerance={tolerance_px} px)",
                         fontsize=12, fontweight="bold")

            ax = axes[0, 0]
            ax.plot(x_pos, n_det, "o-", color="#2E86AB", linewidth=2, markersize=7)
            ax.set_xticks(x_pos); ax.set_xticklabels(x_labels)
            ax.set_xlabel(xlabel); ax.set_ylabel("n_detected")
            ax.set_title("Detected lamina count"); ax.grid(alpha=0.3)

            ax = axes[0, 1]
            ax.plot(x_pos, ratio, "s-", color="#A23B72", linewidth=2, markersize=7)
            ax.set_xticks(x_pos); ax.set_xticklabels(x_labels)
            ax.set_xlabel(xlabel); ax.set_ylabel("Match ratio vs reference (%)")
            ax.set_ylim(0, max(105, ratio.max() * 1.05 if ratio.size else 100))
            ax.set_title(f"Match ratio (tolerance {tolerance_px} px)"); ax.grid(alpha=0.3)

            ax = axes[1, 0]
            ax.plot(x_pos, offset, "^-", color="#F18F01", linewidth=2, markersize=7)
            ax.set_xticks(x_pos); ax.set_xticklabels(x_labels)
            ax.set_xlabel(xlabel); ax.set_ylabel("Mean position offset (px)")
            ax.set_title("Mean position offset of matched points"); ax.grid(alpha=0.3)

            ax = axes[1, 1]
            ax.plot(x_pos, t_proc, "d-", color="#3B7A57", linewidth=2, markersize=7)
            ax.set_xticks(x_pos); ax.set_xticklabels(x_labels)
            ax.set_xlabel(xlabel); ax.set_ylabel("Processing time (s)")
            ax.set_title("Per-run processing time"); ax.grid(alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close()

        # Summary plot
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("Parameter sensitivity summary (4 parameters x key metrics)",
                         fontsize=13, fontweight="bold")
            for ax, (param_name, _, xlabel) in zip(axes.flat, params_info):
                sub = df[df["parameter_name"] == param_name].copy()
                if sub.empty:
                    ax.axis("off")
                    continue
                x_labels = sub["parameter_value"].astype(str).tolist()
                x_pos = np.arange(len(x_labels))
                ratio = pd.to_numeric(sub["match_ratio_vs_reference"], errors="coerce").fillna(0).values * 100
                n_det = pd.to_numeric(sub["n_detected"], errors="coerce").fillna(0).values
                ax2 = ax.twinx()
                l1 = ax.plot(x_pos, ratio, "o-", color="#A23B72", linewidth=2, markersize=6, label="Match ratio (%)")
                l2 = ax2.plot(x_pos, n_det, "s--", color="#2E86AB", linewidth=2, markersize=6, label="n_detected")
                ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=9)
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel("Match ratio (%)", color="#A23B72")
                ax2.set_ylabel("Detected laminae", color="#2E86AB")
                ax.set_title(param_name)
                ax.grid(alpha=0.3)
                lines = l1 + l2
                ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=9)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(out_dir, "13E_parameter_sensitivity_summary.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"  Error while drawing summary plot: {e}")

    def _write_sensitivity_readme(self, out_dir, tolerance_px, n_ref):
        text = (
            "Parameter sensitivity analysis -- metric definitions\n"
            "===================================================\n\n"
            f"The reference for every metric in this directory is the current\n"
            f"detector result.\n"
            f"Reference point count = {n_ref}.\n"
            f"Match tolerance = +/- {tolerance_px} px (closest point on the same y).\n\n"
            "Column descriptions:\n"
            "  parameter_name           Name of the parameter under test.\n"
            "  parameter_value          Parameter value (scanline_interval shown as H/N).\n"
            "  parameter_value_numeric  Numeric value.\n"
            "  n_detected               Lamina edges detected by this variant (after validation).\n"
            "  mean_consistency         Mean cross-line consistency across all detected points.\n"
            "  mean_strength            Mean edge strength ln(1 + DeltaI) * 5.\n"
            "  processing_time_s        Time for one preprocess + detect run (s).\n"
            "  n_reference_points       Reference-result point count (see above).\n"
            "  n_matched_vs_reference   Points matched against the reference.\n"
            "  match_ratio_vs_reference n_matched / n_reference_points.\n"
            "  n_extra_vs_reference     Current-result points that did not match the reference.\n"
            "  n_missed_vs_reference    Reference points not covered by the current result.\n"
            "  mean_position_offset_px  Mean x-offset across matched pairs.\n"
            "  mean_position_offset_mm  Mean x-offset in millimetres (empty if not calibrated).\n"
            "  mean_y_offset_px         Mean y-offset across matched pairs.\n"
            "  y_tolerance_px           Maximum allowed y-offset for matching.\n\n"
            "Writing guidance:\n"
            "  Always state the reference definition in the paper, e.g.:\n"
            "  'Match ratio is computed against the detection result obtained with\n"
            "   the default parameter set on the same sample, since no expert-annotated\n"
            "   ground truth is available in this experiment.'\n"
            "  Do NOT equate match_ratio with accuracy/recall against ground truth.\n"
        )
        with open(os.path.join(out_dir, "_README_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    def run_ablation_study(self, out_dir, tolerance_px=10):
        """Ablation study.

        Variants:
          full_model                   Full method.
          without_clahe                Disable CLAHE.
          without_geometric_correction Disable geometric correction.
          without_crossline_validation Disable cross-line validation (use raw points).
        """
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== Ablation study -> {out_dir} ===")

        if not self.layers:
            print("Current detector has no results; run detect_layers first")
            return out_dir

        ref_pts = _collect_points(self, use_validated=True)
        n_ref = sum(len(v) for v in ref_pts.values())

        variants = [
            ("full_model", "Full model", {}, True),
            ("without_clahe", "No CLAHE", {"disable_clahe": True}, True),
            ("without_geometric_correction", "No geometric correction", {"align_core": False}, True),
            ("without_crossline_validation", "No cross-line validation", {}, False),
        ]

        rows = []
        for variant_id, variant_label, override, used_val in variants:
            print(f"  Running variant: {variant_id}")
            try:
                det, elapsed, _ = self._run_detection_variant(override)
                metrics = self._variant_metrics(det, elapsed, used_val, ref_pts, tolerance_px)
            except Exception as e:
                print(f"    {variant_id} failed: {e}")
                metrics = {
                    "n_detected": "",
                    "mean_consistency": "",
                    "mean_strength": "",
                    "processing_time_s": "",
                    "n_reference_points": n_ref,
                    "n_matched_vs_reference": "",
                    "match_ratio_vs_reference": "",
                    "n_extra_vs_reference": "",
                    "n_missed_vs_reference": "",
                    "mean_position_offset_px": "",
                    "mean_position_offset_mm": "",
                    "mean_y_offset_px": "",
                }
            row = {
                "model_variant": variant_id,
                "model_variant_label": variant_label,
                "metrics_use_validated_points": used_val,
                **metrics,
            }
            rows.append(row)
            print(f"    {variant_id} -> n_detected={metrics['n_detected']}, "
                  f"match_ratio={metrics['match_ratio_vs_reference']}")

        tmp = os.path.join(getattr(self, "output_dir", "."), "_variant_tmp")
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "ablation_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        try:
            df.to_excel(os.path.join(out_dir, "ablation_results.xlsx"), index=False)
        except Exception:
            pass

        try:
            self._plot_ablation(out_dir, df, n_ref, tolerance_px)
        except Exception as e:
            print(f"  Error while plotting ablation: {e}")

        self._write_ablation_readme(out_dir, tolerance_px, n_ref)

        print(f"=== Ablation study complete: {csv_path} ===")
        return out_dir

    def _plot_ablation(self, out_dir, df, n_ref, tolerance_px):
        labels = df["model_variant_label"].tolist()
        x = np.arange(len(labels))

        n_det = pd.to_numeric(df["n_detected"], errors="coerce").fillna(0).values
        ratio = pd.to_numeric(df["match_ratio_vs_reference"], errors="coerce").fillna(0).values * 100
        cons = pd.to_numeric(df["mean_consistency"], errors="coerce").fillna(0).values
        t_proc = pd.to_numeric(df["processing_time_s"], errors="coerce").fillna(0).values

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"Ablation study\nreference = full_model "
                     f"(n={n_ref} validated points, tolerance={tolerance_px} px)",
                     fontsize=13, fontweight="bold")

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#3B7A57"]

        ax = axes[0, 0]
        bars = ax.bar(x, n_det, color=colors[:len(labels)])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("Detected laminae")
        ax.set_title("Detected lamina count n_detected")
        for b, v in zip(bars, n_det):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{int(v)}", ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        ax = axes[0, 1]
        bars = ax.bar(x, ratio, color=colors[:len(labels)])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("Match ratio vs reference (%)")
        ax.set_title(f"Match ratio vs full_model (tolerance {tolerance_px} px)")
        for b, v in zip(bars, ratio):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(105, (ratio.max() if ratio.size else 0) * 1.1))
        ax.grid(alpha=0.3, axis="y")

        ax = axes[1, 0]
        bars = ax.bar(x, cons, color=colors[:len(labels)])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("Mean cross-line consistency")
        ax.set_title("Mean cross-line consistency mean_consistency")
        for b, v in zip(bars, cons):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        ax = axes[1, 1]
        bars = ax.bar(x, t_proc, color=colors[:len(labels)])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("Processing time (s)")
        ax.set_title("Processing time processing_time_s")
        for b, v in zip(bars, t_proc):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(os.path.join(out_dir, "14_ablation_summary.png"), dpi=200)
        plt.close()

    def _write_ablation_readme(self, out_dir, tolerance_px, n_ref):
        text = (
            "Ablation study -- metric definitions\n"
            "====================================\n\n"
            f"Reference = full_model detection result.\n"
            f"Reference point count = {n_ref}, match tolerance = +/- {tolerance_px} px.\n\n"
            "Variant definitions:\n"
            "  full_model                   Full method (user's current default parameters).\n"
            "  without_clahe                preprocess_image(disable_clahe=True);\n"
            "                               skips CLAHE only, other preprocessing kept.\n"
            "  without_geometric_correction detect_layers(align_core=False);\n"
            "                               skips Hough-based geometric correction.\n"
            "  without_crossline_validation Validation still runs, but metrics use raw\n"
            "                               candidate points (no cross-line filter) as output.\n\n"
            "Important: metrics_use_validated_points=False means the variant's metrics\n"
            "are based on unvalidated candidates, so n_detected is typically much higher\n"
            "than full_model -- this is expected behaviour.\n\n"
            "Writing guidance: clearly state each variant's disabled-component rule and\n"
            "the reference definition in the methods section or appendix. ``match_ratio``\n"
            "is NOT accuracy against ground truth; it is consistency with the default full\n"
            "model.\n"
        )
        with open(os.path.join(out_dir, "_README_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    def export_sensitivity_and_ablation(self, paper_dir, tolerance_px=10):
        """One-shot export of parameter sensitivity + ablation results.

        Output directories:
          ``{paper_dir}/06_sensitivity/``  Parameter sensitivity.
          ``{paper_dir}/07_ablation/``     Ablation study.
        """
        sens_dir = os.path.join(paper_dir, "06_sensitivity")
        abl_dir = os.path.join(paper_dir, "07_ablation")
        self.run_parameter_sensitivity(sens_dir, tolerance_px=tolerance_px)
        self.run_ablation_study(abl_dir, tolerance_px=tolerance_px)
        return sens_dir, abl_dir
