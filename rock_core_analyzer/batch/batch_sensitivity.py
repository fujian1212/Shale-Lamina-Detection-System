#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch parameter sensitivity and ablation analysis.

The single-image ``SensitivityMixin`` produces metrics for one sample at a
time. For paper-grade evidence we need to demonstrate that the sensitivity
findings (and the ablation conclusions) hold across multiple samples in a
batch. This module:

  1. Iterates every successful image in a batch run, re-instantiates the
     detector, runs the same sensitivity + ablation tests that the
     single-image flow uses, and saves the per-image CSV / Excel / PNG
     outputs under ``<batch_output>/batch_analysis/<image_name>/``.
  2. Aggregates the per-image results across the entire batch and writes
     ``batch_parameter_sensitivity.{csv, xlsx}`` and
     ``batch_ablation_results.{csv, xlsx}`` together with summary plots in
     ``<batch_output>/batch_analysis/``.

The aggregation reports the mean +/- std of every metric across images at
each parameter value (or model variant), which is the right shape for a
paper figure.
"""

import os
import json
import time
import traceback
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------------------------------------------------
# Per-image worker
# ----------------------------------------------------------------------
def run_single_image_sensitivity_and_ablation(image_path, detector_params,
                                              out_dir, tolerance_px=10):
    """Run sensitivity + ablation for one image and save the per-image artifacts.

    Args:
        image_path: Path to the source image.
        detector_params: Dict of detector parameters (matches the keys used by
            the batch worker, see ``rock_core_analyzer.gui.workers._batch_worker``).
        out_dir: Output directory; sub-directories ``06_sensitivity`` and
            ``07_ablation`` will be created inside.
        tolerance_px: Matching tolerance in pixels (default 10).

    Returns:
        Dict with keys ``ok`` (bool), ``image_path``, ``out_dir``,
        ``sensitivity_csv`` (path), ``ablation_csv`` (path), and ``error``
        (only present on failure).
    """
    from rock_core_analyzer.core import RockCoreLayerDetector

    os.makedirs(out_dir, exist_ok=True)
    result = {"ok": False, "image_path": image_path, "out_dir": out_dir,
              "sensitivity_csv": None, "ablation_csv": None}
    try:
        det = RockCoreLayerDetector(image_path)
        det.output_dir = os.path.join(out_dir, "_baseline")
        os.makedirs(det.output_dir, exist_ok=True)
        det.save_diagnostics = False

        if detector_params.get("pixel_per_mm"):
            det.pixel_per_mm = detector_params["pixel_per_mm"]

        det.preprocess_image(
            blur_size=detector_params.get("blur_size", 5),
            clahe_clip=detector_params.get("clahe_clip", 2.0),
            clahe_grid=detector_params.get("clahe_grid", (8, 8)),
            brightness=detector_params.get("brightness", 0),
            contrast=detector_params.get("contrast", 1.0),
            gamma=detector_params.get("gamma", 1.0),
        )

        batch_scan_lines = detector_params.get("batch_scan_lines")
        if batch_scan_lines is None:
            scan_lines = None
            scan_line_count = detector_params.get("scan_line_count", 5)
        elif isinstance(batch_scan_lines, int):
            scan_lines = None
            scan_line_count = batch_scan_lines
        else:
            scan_lines = batch_scan_lines
            scan_line_count = len(batch_scan_lines)

        ok = det.detect_layers(
            threshold_method=detector_params.get("threshold_method", "otsu"),
            min_layer_width=detector_params.get("min_layer_width", 5),
            scan_lines=scan_lines,
            scan_line_count=scan_line_count,
            min_validation_lines=detector_params.get("min_validation_lines", 2),
            align_core=detector_params.get("align_core", True),
            alignment_angle=detector_params.get("alignment_angle", 0.0),
        )
        if not ok:
            result["error"] = "No valid laminae detected on the baseline run"
            return result

        det.calculate_statistics()

        sens_dir = os.path.join(out_dir, "06_sensitivity")
        abl_dir = os.path.join(out_dir, "07_ablation")
        det.run_parameter_sensitivity(sens_dir, tolerance_px=tolerance_px)
        det.run_ablation_study(abl_dir, tolerance_px=tolerance_px)

        result["sensitivity_csv"] = os.path.join(sens_dir, "parameter_sensitivity_results.csv")
        result["ablation_csv"] = os.path.join(abl_dir, "ablation_results.csv")
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"
    return result


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------
_SENSITIVITY_NUMERIC_COLS = [
    "n_detected", "match_ratio_vs_reference",
    "n_extra_vs_reference", "n_missed_vs_reference",
    "mean_position_offset_px", "mean_position_offset_mm",
    "mean_y_offset_px", "mean_consistency", "mean_strength",
    "processing_time_s",
]


def _to_numeric_df(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _aggregate_sensitivity(per_image_csvs):
    """Aggregate per-image sensitivity CSVs.

    Returns:
        ``(raw_df, agg_df)``; ``raw_df`` has one row per (image, parameter,
        value), ``agg_df`` has one row per (parameter, value) with the mean
        / std / n_images of every numeric metric.
    """
    frames = []
    for image_name, csv_path in per_image_csvs:
        if not csv_path or not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Cannot read {csv_path}: {e}")
            continue
        df.insert(0, "image_name", image_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = _to_numeric_df(raw_df, _SENSITIVITY_NUMERIC_COLS)

    group_cols = ["parameter_name", "parameter_value"]
    agg_rows = []
    for (param, value), grp in raw_df.groupby(group_cols, sort=False):
        row = {"parameter_name": param, "parameter_value": value,
               "n_images": int(grp.shape[0])}
        for col in _SENSITIVITY_NUMERIC_COLS:
            if col in grp.columns:
                series = grp[col].dropna()
                row[f"{col}_mean"] = round(float(series.mean()), 4) if len(series) > 0 else ""
                row[f"{col}_std"] = round(float(series.std(ddof=0)), 4) if len(series) > 1 else 0.0
                row[f"{col}_min"] = round(float(series.min()), 4) if len(series) > 0 else ""
                row[f"{col}_max"] = round(float(series.max()), 4) if len(series) > 0 else ""
        # Preserve the numeric parameter value when available
        if "parameter_value_numeric" in grp.columns:
            try:
                row["parameter_value_numeric"] = float(grp["parameter_value_numeric"].iloc[0])
            except (TypeError, ValueError):
                pass
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    return raw_df, agg_df


_ABLATION_NUMERIC_COLS = [
    "n_detected", "match_ratio_vs_reference",
    "n_extra_vs_reference", "n_missed_vs_reference",
    "mean_position_offset_px", "mean_position_offset_mm",
    "mean_y_offset_px", "mean_consistency", "mean_strength",
    "processing_time_s",
]


def _aggregate_ablation(per_image_csvs):
    """Aggregate per-image ablation CSVs (one row per model variant)."""
    frames = []
    for image_name, csv_path in per_image_csvs:
        if not csv_path or not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Cannot read {csv_path}: {e}")
            continue
        df.insert(0, "image_name", image_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = _to_numeric_df(raw_df, _ABLATION_NUMERIC_COLS)

    agg_rows = []
    for variant, grp in raw_df.groupby("model_variant", sort=False):
        row = {"model_variant": variant,
               "model_variant_label": grp["model_variant_label"].iloc[0]
                   if "model_variant_label" in grp.columns else variant,
               "n_images": int(grp.shape[0])}
        for col in _ABLATION_NUMERIC_COLS:
            if col in grp.columns:
                series = grp[col].dropna()
                row[f"{col}_mean"] = round(float(series.mean()), 4) if len(series) > 0 else ""
                row[f"{col}_std"] = round(float(series.std(ddof=0)), 4) if len(series) > 1 else 0.0
                row[f"{col}_min"] = round(float(series.min()), 4) if len(series) > 0 else ""
                row[f"{col}_max"] = round(float(series.max()), 4) if len(series) > 0 else ""
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    return raw_df, agg_df


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def _plot_batch_sensitivity_summary(agg_df, out_path, n_images):
    """Render the 4-panel batch sensitivity summary (mean +/- std per parameter)."""
    params_info = [
        ("gaussian_kernel",      "Gaussian kernel size (px)"),
        ("clahe_clip_limit",     "CLAHE clip limit"),
        ("scanline_interval",    "Scan-line interval (H/N)"),
        ("min_validation_lines", "Min validation lines"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Batch parameter sensitivity (mean +/- std across {n_images} image(s))",
                 fontsize=13, fontweight="bold")

    for ax, (param_name, xlabel) in zip(axes.flat, params_info):
        sub = agg_df[agg_df["parameter_name"] == param_name].copy()
        if sub.empty:
            ax.axis("off")
            continue
        x_labels = sub["parameter_value"].astype(str).tolist()
        x_pos = np.arange(len(x_labels))

        ratio_mean = pd.to_numeric(sub.get("match_ratio_vs_reference_mean"), errors="coerce").fillna(0).values * 100
        ratio_std = pd.to_numeric(sub.get("match_ratio_vs_reference_std"), errors="coerce").fillna(0).values * 100
        n_det_mean = pd.to_numeric(sub.get("n_detected_mean"), errors="coerce").fillna(0).values
        n_det_std = pd.to_numeric(sub.get("n_detected_std"), errors="coerce").fillna(0).values

        ax2 = ax.twinx()
        ax.errorbar(x_pos, ratio_mean, yerr=ratio_std, fmt="o-",
                    color="#A23B72", linewidth=2, markersize=6, capsize=4,
                    label="Match ratio mean (%)")
        ax2.errorbar(x_pos, n_det_mean, yerr=n_det_std, fmt="s--",
                     color="#2E86AB", linewidth=2, markersize=6, capsize=4,
                     label="n_detected mean")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Match ratio mean (%)", color="#A23B72")
        ax2.set_ylabel("Detected laminae mean", color="#2E86AB")
        ax.set_title(param_name)
        ax.grid(alpha=0.3)
        # Manually combine the two legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_batch_ablation_summary(agg_df, out_path, n_images):
    """Render the 2x2 batch ablation summary with error bars."""
    if agg_df.empty:
        return
    labels = agg_df["model_variant_label"].tolist()
    x = np.arange(len(labels))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#3B7A57",
              "#9B59B6", "#16A085", "#E74C3C", "#34495E"][:len(labels)]

    def _vals(col):
        m = pd.to_numeric(agg_df.get(f"{col}_mean"), errors="coerce").fillna(0).values
        s = pd.to_numeric(agg_df.get(f"{col}_std"), errors="coerce").fillna(0).values
        return m, s

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Batch ablation study (mean +/- std across {n_images} image(s))",
                 fontsize=13, fontweight="bold")

    panels = [
        ("n_detected", "Detected laminae (mean)", "Detected lamina count", 1.0),
        ("match_ratio_vs_reference", "Match ratio (%)",
         "Match ratio vs full_model", 100.0),
        ("mean_consistency", "Mean consistency",
         "Mean cross-line consistency", 1.0),
        ("processing_time_s", "Processing time (s)",
         "Processing time", 1.0),
    ]
    for ax, (col, ylabel, title, scale) in zip(axes.flat, panels):
        means, stds = _vals(col)
        means = means * scale
        stds = stds * scale
        bars = ax.bar(x, means, color=colors, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        for b, m, s in zip(bars, means, stds):
            tag = f"{m:.2f}" if scale == 1.0 else f"{m:.1f}%"
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + (s if s else 0),
                    tag, ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------------------------------------------------
# README writer
# ----------------------------------------------------------------------
def _write_batch_readme(out_dir, n_images, tolerance_px,
                        image_names, per_image_status):
    """Write a single README explaining the batch sensitivity / ablation outputs."""
    lines = [
        "Batch parameter sensitivity and ablation analysis -- column definitions",
        "=" * 70,
        "",
        f"Images analysed: {n_images}",
        f"Match tolerance (px): {tolerance_px}",
        "",
        "Per-image structure (each image has its own subfolder):",
        "  <image_name>/06_sensitivity/ -- single-image sensitivity CSV + PNG.",
        "  <image_name>/07_ablation/    -- single-image ablation CSV + PNG.",
        "",
        "Aggregated tables (this directory):",
        "  batch_parameter_sensitivity_raw.{csv,xlsx} -- one row per",
        "    (image, parameter_name, parameter_value). Includes the same",
        "    metrics produced by the single-image sensitivity test.",
        "  batch_parameter_sensitivity.{csv,xlsx}     -- one row per",
        "    (parameter_name, parameter_value), with mean / std / min / max",
        "    of every numeric metric across the images that were analysed.",
        "  batch_parameter_sensitivity_summary.png    -- 4-panel summary plot.",
        "  batch_ablation_raw.{csv,xlsx}              -- one row per",
        "    (image, model_variant).",
        "  batch_ablation_results.{csv,xlsx}          -- one row per",
        "    model_variant, with mean / std / min / max across images.",
        "  batch_ablation_summary.png                 -- bar-chart summary.",
        "",
        "Important reminders for paper writing:",
        "  1. Every metric here is computed against the per-image *default-parameter*",
        "     detection result of that same image; it is NOT accuracy against",
        "     manual annotation. Always state this in the methods section.",
        "  2. The aggregated mean / std summarise behaviour across the batch and",
        "     should be reported with the image count (n_images column).",
        "  3. The number of supporting images per row may differ when an image",
        "     fails baseline detection.",
        "",
        "Per-image status:",
    ]
    for name, status in per_image_status:
        lines.append(f"  - {name}: {status}")

    with open(os.path.join(out_dir, "_README_batch_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def run_batch_sensitivity_and_ablation(output_dir, image_jobs,
                                       detector_params,
                                       tolerance_px=10,
                                       progress_callback=None):
    """Run sensitivity + ablation on every image in a batch and aggregate.

    Args:
        output_dir: Top-level batch output directory. The aggregated tables
            and the per-image subfolders are written under
            ``<output_dir>/batch_analysis/``.
        image_jobs: Iterable of ``(image_name, image_path)`` pairs.
        detector_params: Common detector parameters (see
            ``run_single_image_sensitivity_and_ablation``).
        tolerance_px: Matching tolerance in pixels.
        progress_callback: Optional callable invoked with
            ``(completed, total, image_name, status)``.

    Returns:
        Path to ``<output_dir>/batch_analysis/`` (string).
    """
    analysis_dir = os.path.join(output_dir, "batch_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    image_jobs = list(image_jobs)
    total = len(image_jobs)
    if total == 0:
        print("Batch analysis: no images to process")
        return analysis_dir

    print(f"=== Batch sensitivity + ablation on {total} image(s) -> {analysis_dir} ===")

    sensitivity_csvs = []
    ablation_csvs = []
    per_image_status = []
    succeeded = 0

    t0 = time.time()
    for i, (image_name, image_path) in enumerate(image_jobs, 1):
        safe_name = image_name.replace("\\", "_").replace("/", "_")
        per_dir = os.path.join(analysis_dir, safe_name)
        print(f"[{i}/{total}] {image_name}")

        if progress_callback:
            try:
                progress_callback(i - 1, total, image_name, "running")
            except Exception:
                pass

        result = run_single_image_sensitivity_and_ablation(
            image_path=image_path,
            detector_params=detector_params,
            out_dir=per_dir,
            tolerance_px=tolerance_px,
        )
        if result["ok"]:
            succeeded += 1
            sensitivity_csvs.append((image_name, result["sensitivity_csv"]))
            ablation_csvs.append((image_name, result["ablation_csv"]))
            per_image_status.append((image_name, "ok"))
            print(f"  -> ok ({time.time() - t0:.1f}s elapsed)")
        else:
            per_image_status.append((image_name, f"failed: {result.get('error', 'unknown error')}"))
            print(f"  -> failed: {result.get('error', 'unknown error')}")

        if progress_callback:
            try:
                progress_callback(i, total, image_name,
                                  "ok" if result["ok"] else "failed")
            except Exception:
                pass

    print(f"Per-image runs complete: {succeeded}/{total} succeeded")

    # ----- Aggregation -----
    if succeeded > 0:
        try:
            sens_raw, sens_agg = _aggregate_sensitivity(sensitivity_csvs)
            if not sens_raw.empty:
                sens_raw.to_csv(os.path.join(analysis_dir,
                                              "batch_parameter_sensitivity_raw.csv"),
                                index=False, encoding="utf-8-sig")
                try:
                    sens_raw.to_excel(os.path.join(analysis_dir,
                                                    "batch_parameter_sensitivity_raw.xlsx"),
                                      index=False)
                except Exception:
                    pass
            if not sens_agg.empty:
                sens_agg.to_csv(os.path.join(analysis_dir,
                                              "batch_parameter_sensitivity.csv"),
                                index=False, encoding="utf-8-sig")
                try:
                    sens_agg.to_excel(os.path.join(analysis_dir,
                                                    "batch_parameter_sensitivity.xlsx"),
                                      index=False)
                except Exception:
                    pass
                try:
                    _plot_batch_sensitivity_summary(
                        sens_agg,
                        os.path.join(analysis_dir, "batch_parameter_sensitivity_summary.png"),
                        n_images=succeeded,
                    )
                except Exception as e:
                    print(f"Cannot draw sensitivity summary plot: {e}")
        except Exception as e:
            print(f"Sensitivity aggregation error: {e}")
            traceback.print_exc()

        try:
            abl_raw, abl_agg = _aggregate_ablation(ablation_csvs)
            if not abl_raw.empty:
                abl_raw.to_csv(os.path.join(analysis_dir, "batch_ablation_raw.csv"),
                               index=False, encoding="utf-8-sig")
                try:
                    abl_raw.to_excel(os.path.join(analysis_dir, "batch_ablation_raw.xlsx"),
                                     index=False)
                except Exception:
                    pass
            if not abl_agg.empty:
                abl_agg.to_csv(os.path.join(analysis_dir, "batch_ablation_results.csv"),
                               index=False, encoding="utf-8-sig")
                try:
                    abl_agg.to_excel(os.path.join(analysis_dir, "batch_ablation_results.xlsx"),
                                     index=False)
                except Exception:
                    pass
                try:
                    _plot_batch_ablation_summary(
                        abl_agg,
                        os.path.join(analysis_dir, "batch_ablation_summary.png"),
                        n_images=succeeded,
                    )
                except Exception as e:
                    print(f"Cannot draw ablation summary plot: {e}")
        except Exception as e:
            print(f"Ablation aggregation error: {e}")
            traceback.print_exc()

    # README + status report
    try:
        _write_batch_readme(analysis_dir, succeeded, tolerance_px,
                            [job[0] for job in image_jobs], per_image_status)
    except Exception as e:
        print(f"Cannot write README: {e}")

    # Store run config so it can be cross-referenced from the paper
    try:
        config = {
            "image_count": total,
            "succeeded": succeeded,
            "tolerance_px": tolerance_px,
            "detector_params": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in detector_params.items()
                if not isinstance(v, (bytes, bytearray))
            },
        }
        # Drop unpicklable / overly large values defensively
        for key in list(config["detector_params"].keys()):
            try:
                json.dumps(config["detector_params"][key])
            except TypeError:
                config["detector_params"][key] = str(config["detector_params"][key])
        with open(os.path.join(analysis_dir, "_run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Cannot write run config: {e}")

    print(f"=== Batch sensitivity + ablation finished: {analysis_dir} ===")
    return analysis_dir
