#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sub-folder lamina dip calibration for batch processing.

Batch mode uses two complementary rules:

1. **Group calibration (approach 1)** – within each sub-folder (or the root
   folder when there are no sub-folders), run a lightweight detection pass on
   the first *N* images (default 5; all images when fewer than *N* exist).
   The median lamina slope from those images becomes the ``reference_slope``
   applied to every image in the same group for clustering / linking.

2. **Post-align vertical constraint (approach 2)** – when core flattening
   (``align_core=True``) is enabled, laminae should be nearly vertical after
   shear. Any valid cluster whose fitted dip exceeds ``max_dip_after_align_deg``
   is rejected (bad shear or mis-linked smudge). Surviving laminae are snapped
   to a vertical line at their mean *x* for drawing and spacing statistics.
"""

import json
import math
import os
import traceback

import numpy as np

# Default limits for batch mode (post-alignment)
DEFAULT_MAX_CALIBRATION_IMAGES = 5
DEFAULT_MAX_DIP_AFTER_ALIGN_DEG = 7.0
# |slope| <= tan(7 deg) ~ 0.123
MAX_GROUP_SLOPE_AFTER_ALIGN = math.tan(math.radians(DEFAULT_MAX_DIP_AFTER_ALIGN_DEG))


def _median_or_zero(values):
    vals = [float(v) for v in values if v is not None and v == v]
    if not vals:
        return 0.0
    return float(np.median(vals))


def probe_image_dip(image_path, detector_params):
    """Run preprocess + detect on one image; return slope / dip probes.

    Does **not** write result files – used only for sub-folder calibration.
    """
    from rock_core_analyzer.core import RockCoreLayerDetector

    result = {
        "ok": False,
        "image_path": image_path,
        "slope_hint": 0.0,
        "valid_slopes": [],
        "valid_dips": [],
        "aligned": False,
        "n_valid_laminae": 0,
        "error": "",
    }
    try:
        det = RockCoreLayerDetector(image_path)
        det.output_dir = os.path.join(detector_params.get("_probe_dir", "."), "_dip_probe")
        det.save_diagnostics = False

        if detector_params.get("pixel_per_mm") is not None:
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
            result["error"] = "no valid laminae on probe run"
            return result

        ls = getattr(det, "_lamina_settings", {}) or {}
        valid_laminae = [la for la in (getattr(det, "laminae", None) or []) if la.get("is_valid")]

        result["ok"] = True
        result["slope_hint"] = float(ls.get("slope_hint", 0.0) or 0.0)
        result["valid_slopes"] = [float(la.get("fit_slope", 0.0)) for la in valid_laminae]
        result["valid_dips"] = [float(la.get("dip_angle_deg", 0.0)) for la in valid_laminae]
        result["aligned"] = bool(getattr(det, "aligned", False))
        result["n_valid_laminae"] = len(valid_laminae)
    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"
    return result


def calibrate_subfolder_dip(image_paths, detector_params,
                            max_images=DEFAULT_MAX_CALIBRATION_IMAGES):
    """Estimate a group reference slope from the first *max_images* paths.

    Args:
        image_paths: Ordered list of absolute image paths in one sub-folder.
        detector_params: Shared detector settings (same dict the batch worker uses).
        max_images: How many leading images to probe (default 5).

    Returns:
        dict with ``reference_slope``, ``reference_dip_deg``, probe metadata.
    """
    if not image_paths:
        return {
            "reference_slope": 0.0,
            "reference_dip_deg": 0.0,
            "n_calibration_images": 0,
            "n_probe_success": 0,
            "align_core": bool(detector_params.get("align_core", True)),
            "max_dip_after_align_deg": DEFAULT_MAX_DIP_AFTER_ALIGN_DEG,
            "force_vertical_after_align": True,
            "calibration_image_names": [],
            "warnings": ["no images in group"],
        }

    cal_paths = image_paths[:max_images] if len(image_paths) > max_images else list(image_paths)
    align_core = bool(detector_params.get("align_core", True))

    slope_hints = []
    valid_slopes = []
    valid_dips = []
    probe_records = []
    warnings = []

    for path in cal_paths:
        probe = probe_image_dip(path, detector_params)
        record = {
            "image": os.path.basename(path),
            "ok": probe["ok"],
            "slope_hint": probe.get("slope_hint", 0.0),
            "n_valid_laminae": probe.get("n_valid_laminae", 0),
            "median_valid_dip_deg": _median_or_zero(probe.get("valid_dips", [])),
            "error": probe.get("error", ""),
        }
        probe_records.append(record)
        if not probe["ok"]:
            warnings.append(f"calibration probe failed: {os.path.basename(path)}")
            continue
        slope_hints.append(probe["slope_hint"])
        valid_slopes.extend(probe.get("valid_slopes", []))
        valid_dips.extend(probe.get("valid_dips", []))

    n_success = sum(1 for r in probe_records if r["ok"])

    if align_core:
        # Approach 2: after shear, laminae should be near vertical.
        # Use the median of probe slope hints but clamp to a small angle.
        raw_slope = _median_or_zero(slope_hints if slope_hints else valid_slopes)
        if abs(raw_slope) > MAX_GROUP_SLOPE_AFTER_ALIGN:
            warnings.append(
                f"median calibration slope {raw_slope:+.3f} exceeds "
                f"{MAX_GROUP_SLOPE_AFTER_ALIGN:.3f} (~{DEFAULT_MAX_DIP_AFTER_ALIGN_DEG} deg); "
                f"shear may have failed – forcing reference_slope=0"
            )
            reference_slope = 0.0
        else:
            reference_slope = raw_slope
    else:
        # Approach 1 without shear: trust the group median slope from probes.
        reference_slope = _median_or_zero(valid_slopes if valid_slopes else slope_hints)

    reference_dip = math.degrees(math.atan(abs(reference_slope))) if reference_slope else 0.0
    median_valid_dip = _median_or_zero(valid_dips)

    if align_core and median_valid_dip > DEFAULT_MAX_DIP_AFTER_ALIGN_DEG:
        warnings.append(
            f"median valid-lamina dip across calibration images is "
            f"{median_valid_dip:.1f} deg (> {DEFAULT_MAX_DIP_AFTER_ALIGN_DEG} deg); "
            f"check shear alignment"
        )

    return {
        "reference_slope": round(reference_slope, 5),
        "reference_dip_deg": round(reference_dip, 2),
        "median_valid_dip_deg": round(median_valid_dip, 2),
        "n_calibration_images": len(cal_paths),
        "n_probe_success": n_success,
        "align_core": align_core,
        "max_dip_after_align_deg": DEFAULT_MAX_DIP_AFTER_ALIGN_DEG,
        "force_vertical_after_align": bool(align_core),
        "calibration_image_names": [os.path.basename(p) for p in cal_paths],
        "probe_records": probe_records,
        "warnings": warnings,
    }


def save_group_calibration(group_output_dir, calibration):
    """Persist calibration JSON under the merged group output directory."""
    os.makedirs(group_output_dir, exist_ok=True)
    path = os.path.join(group_output_dir, "batch_dip_calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    return path


def batch_lamina_kwargs_from_calibration(calibration):
    """Build the worker keyword args consumed by ``_batch_worker``."""
    if not calibration:
        return {"batch_lamina_mode": False}
    return {
        "batch_lamina_mode": True,
        "batch_group_slope_hint": calibration.get("reference_slope", 0.0),
        "batch_max_dip_after_align_deg": calibration.get(
            "max_dip_after_align_deg", DEFAULT_MAX_DIP_AFTER_ALIGN_DEG
        ),
        "batch_force_vertical_after_align": calibration.get(
            "force_vertical_after_align", True
        ),
    }
