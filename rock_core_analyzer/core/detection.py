#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lamina detection."""

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


class DetectionMixin:
    def detect_layers(self, threshold_method='otsu', min_layer_width=5, scan_lines=None, scan_line_count=5,
                     min_validation_lines=2, align_core=False, alignment_angle=0.0,
                     min_delta_gray=None, max_dip_angle_deg=45.0, user_dip_angle_deg=None):
        """Run lamina detection.

        Core strategy:
        1. Multi-scale step + logarithmic-gradient detection; each scan line emits candidate points.
        2. Absolute magnitude filter (``min_delta_gray``): drop changes too small to be lamina.
        3. Fracture / dark-trough rejection + direction-consistency constraint.
        4. Cross-scan-line 2D line fit: cluster points whose x is close on main and validation
           lines, fit them to a (possibly tilted) line allowing dip <= ``max_dip_angle_deg``.
           Fits that fail are treated as smudges.

        Parameters
        ----------
        threshold_method : str
            Threshold method: ``'otsu'`` / ``'adaptive'`` / ``'binary'``.
        min_layer_width : int
            Minimum lamina width in pixels.
        scan_lines : list or None
            Custom scan-line y-coordinates; ``None`` -> evenly spaced by ``scan_line_count``.
        scan_line_count : int
            Number of auto-generated scan lines.
        min_validation_lines : int
            Minimum number of validation lines that must confirm a candidate.
        align_core : bool
            Whether to flatten the core (geometric correction).
        alignment_angle : float
            Manually specified correction angle; 0 = auto-detect.
        min_delta_gray : float or None
            Lower bound on the absolute grayscale change |left_mean - right_mean| for a
            candidate to be kept. Below this is treated as noise. ``None`` -> auto
            ``max(5, 0.20 * img_std)``.
        max_dip_angle_deg : float
            Maximum allowed dip angle of laminae versus the vertical axis (default 45 deg).
            With the core laid horizontally, laminae should be roughly *longitudinal*
            (near vertical); minor tilts are allowed but heavily tilted long "lines" are
            almost always cracks / scan-marks / scratches and should not be treated as laminae.
            For severely tilted cores, this value can be raised manually (up to ~60 deg).
        user_dip_angle_deg : float or None
            Manually specified lamina dip angle (deviation from vertical, signed, in
            degrees). Intended for single-image mode when the core is badly fractured
            and the automatic direction estimate is unreliable. When given, the auto
            core-alignment is skipped and every lamina is drawn / counted along this
            direction. ``None`` -> fully automatic.
        """
        from scipy.signal import find_peaks

        print(f"=== Lamina detection ===")
        print(f"  threshold method={threshold_method}, min width={min_layer_width}, "
              f"scan lines={scan_line_count}, validation lines={min_validation_lines}")

        if self.processed is None:
            self.processed = self.preprocess_image()

        original_processed = self.processed.copy()

        self.aligned = False
        self.alignment_angle = 0.0
        self.use_shear = False

        # Manual lamina-direction override (single-image, fractured cores).
        # When the user supplies a dip angle, it dictates the drawing/statistics
        # direction; the automatic alignment is bypassed so the lines are rendered
        # at the requested angle on the original image.
        self.user_dip_angle_deg = None
        self._user_slope = None
        if user_dip_angle_deg is not None:
            try:
                _ua = float(user_dip_angle_deg)
            except (TypeError, ValueError):
                _ua = None
            if _ua is not None:
                _ua = max(-80.0, min(80.0, _ua))
                self.user_dip_angle_deg = _ua
                self._user_slope = math.tan(math.radians(_ua))
                if align_core:
                    print(f"  Manual lamina angle set ({_ua:+.1f} deg off vertical); "
                          f"auto core-alignment is skipped in favour of the manual direction")

        # Before any alignment, run a HoughLinesP pass to estimate the dominant slope.
        # The result is used:
        #   1) As the initial guess for the alignment module, so even large tilts (28 deg+) hit.
        #   2) As a ``slope_hint`` for the clustering stage even when alignment is disabled.
        pre_slope, pre_conf = self._estimate_dominant_slope(
            max_dip_angle_deg=max(max_dip_angle_deg, 45.0),
            src=self.gray if getattr(self, 'gray', None) is not None else self.processed,
        )
        self._pre_align_slope = pre_slope
        self._pre_align_slope_confidence = pre_conf
        pre_deg = math.degrees(math.atan(abs(pre_slope))) if pre_slope else 0.0
        print(f"  Dominant lamina dip (pre-alignment estimate): {pre_deg:.1f} deg off vertical "
              f"(dx/dy={pre_slope:+.3f}, conf={pre_conf:.2f})")

        if align_core and self.user_dip_angle_deg is None:
            # Save the original (pre-alignment) images. The refinement stage will use the
            # *cumulative* shear to warp the originals exactly once, instead of stacking
            # warps on top of already-warped images -- otherwise BORDER_CONSTANT creates
            # gray triangles on both sides, visually looking like double shearing.
            self._orig_image_pre_align = self.image.copy() if self.image is not None else None
            self._orig_gray_pre_align = self.gray.copy() if self.gray is not None else None
            self._orig_enhanced_no_grad_pre_align = self.enhanced_no_grad.copy() if self.enhanced_no_grad is not None else None
            self._orig_processed_pre_align = self.processed.copy()
            self._orig_gray_nl_pre_align = (
                self.gray_nonlinear_enhanced.copy()
                if getattr(self, 'gray_nonlinear_enhanced', None) is not None else None
            )
            self._orig_preprocess_steps_pre_align = {
                k: (v.copy() if v is not None else None)
                for k, v in getattr(self, '_preprocess_steps', {}).items()
            }
            # Pass the pre-alignment slope estimate to ``_align_core`` as the initial
            # guess / fast path
            self.processed = self._align_core(
                self.processed,
                alignment_angle,
                slope_hint=pre_slope,
                slope_hint_confidence=pre_conf,
            )
            os.makedirs(self.output_dir, exist_ok=True)

            # Only flag as aligned if an actual transform was applied; then propagate to
            # the other images.
            if abs(self.alignment_angle) > 1e-6:
                self.aligned = True
                ah, aw = self.processed.shape[:2]
                if self.use_shear:
                    if getattr(self, 'shear_axis', 'x') == 'y':
                        # Vertical shear: y' = y + sy * x
                        M = np.float32([[1.0, 0.0, 0.0],
                                        [self.alignment_angle, 1.0, 0.0]])
                    else:
                        # Horizontal shear: x' = x + sx * y
                        M = np.float32([[1.0, self.alignment_angle, 0.0],
                                        [0.0, 1.0, 0.0]])
                else:
                    M = cv2.getRotationMatrix2D((aw // 2, ah // 2), self.alignment_angle, 1.0)

                # Use BORDER_CONSTANT (mean color) to match ``_align_core`` and avoid
                # edge-replicate bands
                def _wf(img):
                    if img is None:
                        return None
                    try:
                        bv = float(np.mean(img)) if img.ndim == 2 else float(np.mean(img))
                    except Exception:
                        bv = 128.0
                    if img.ndim == 3:
                        bvt = (bv, bv, bv)
                    else:
                        bvt = bv
                    return cv2.warpAffine(img, M, (aw, ah),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=bvt)
                if self.gray is not None:
                    self.gray = _wf(self.gray)
                if self.image is not None:
                    self.image = _wf(self.image)
                if self.enhanced_no_grad is not None:
                    self.enhanced_no_grad = _wf(self.enhanced_no_grad)
                if getattr(self, 'gray_nonlinear_enhanced', None) is not None:
                    self.gray_nonlinear_enhanced = _wf(self.gray_nonlinear_enhanced)

                steps = getattr(self, '_preprocess_steps', {})
                for key in list(steps.keys()):
                    if steps[key] is not None:
                        steps[key] = _wf(steps[key])

                self.height, self.width = (self.image.shape[:2] if self.image is not None else self.processed.shape[:2])
                if self.use_shear:
                    axis = getattr(self, 'shear_axis', 'x')
                    print(f"  Alignment propagated to all images "
                          f"(shear factor={self.alignment_angle:.4f}, shear axis={axis})")
                else:
                    print(f"  Alignment propagated to all images "
                          f"(rotation angle={self.alignment_angle:.4f} deg)")
                # Key point: ``self.processed`` = enhanced + Sobel_X was computed
                # *before* alignment. After the affine warp, the gradient response
                # rides along with the pixels and now sits at the *old* edge
                # positions, misaligned with the now-straightened laminae, which would
                # break step detection. Recompute the cross-lamina Sobel_X on the
                # aligned ``enhanced_no_grad`` so the gradient stays aligned with the
                # new vertical laminae.
                if self.enhanced_no_grad is not None:
                    sobel_x_new = cv2.Sobel(self.enhanced_no_grad, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_x_new_abs = np.uint8(np.clip(np.abs(sobel_x_new), 0, 255))
                    self.processed = cv2.addWeighted(self.enhanced_no_grad, 0.7, sobel_x_new_abs, 0.3, 0)
                    # Sync ``original_processed`` so the restore at the end of this
                    # function does not revert to the stale version.
                    original_processed = self.processed.copy()
                    print(f"  Gradient recomputed on the aligned image to keep it in step with the new laminae")
            else:
                self.aligned = False
                print(f"  Alignment produced no actual transform (angle too small or detection failed)")

        # --- Binarization (auxiliary) ---
        if threshold_method == 'otsu':
            _, binary = cv2.threshold(self.processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(self.processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        else:
            _, binary = cv2.threshold(self.processed, 100, 255, cv2.THRESH_BINARY)

        self.binary = binary.copy() if binary is not None else None
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

        # --- Canny edges (auxiliary) ---
        canny_src = self.enhanced_no_grad if hasattr(self, 'enhanced_no_grad') else self.processed
        median_val = np.median(canny_src)
        canny_low = int(max(0, 0.33 * median_val))
        canny_high = int(min(255, 1.2 * median_val))
        edges = cv2.Canny(canny_src, canny_low, canny_high, apertureSize=3)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)

        h, w = self.processed.shape

        edge_filter_percent = 0.05
        edge_filter_pixels = max(10, int(w * edge_filter_percent))

        if scan_lines is None or len(scan_lines) == 0:
            step = h // (scan_line_count + 1)
            scan_lines = [step * (i + 1) for i in range(scan_line_count)]

        self.layers = []
        self.scan_lines = scan_lines

        # Diagnostic info collected for paper export and GUI hints
        self._detection_diagnostics = {
            "fallbacks_triggered": [],
        }

        # ============================================================
        # Precompute the fracture zone (full image, one shot)
        # Fracture = "dark trough" with a sharp brightness drop and recovery.
        # Dark samples have inherently low brightness and small dynamic range, so
        # ``dip_std`` is small as well; we must use a stricter multiplier or real
        # laminae will be misclassified as fractures and dropped.
        # ============================================================
        is_dark_sample = bool(getattr(self, '_preprocess_meta', {}).get('dark_mode_applied', False))
        fracture_zone = set()
        if hasattr(self, 'gray') and self.gray is not None:
            gray_for_fracture = self.gray.astype(np.float64)
            col_profile = np.mean(gray_for_fracture, axis=0)
            # Key: ``trend_sigma`` must be wider than a *single lamina*, or the laminae
            # themselves get treated as "fractures". With the core laid horizontally,
            # laminae are longitudinal dark bands ~5-10 px wide; real fractures are
            # >= 30 px. ``sigma=80`` lets the smoother span several laminae so we can
            # then ask: which of those troughs are actually *wide* dark bands?
            trend_sigma = max(80, w // 12)
            col_trend = gaussian_filter1d(col_profile, sigma=trend_sigma)
            dip = col_trend - col_profile
            edge_skip = int(max(20, w * 0.05))
            dip_central = dip[edge_skip:w - edge_skip] if w > 2 * edge_skip else dip
            dip_std = float(np.std(dip_central)) if len(dip_central) > 0 else 0.0
            dip_mult = 4.0 if is_dark_sample else 3.5
            if dip_std > 0:
                dip_threshold = dip_std * dip_mult
                raw_frac_cols = sorted(c for c in np.where(dip > dip_threshold)[0]
                                       if edge_skip <= c < (w - edge_skip))
                # Stitch consecutive fracture columns into segments and keep only those
                # >= 15 px wide (true fracture signature).
                min_frac_width = max(15, min_layer_width * 3)
                seg_start = None
                prev = None
                for c in raw_frac_cols:
                    if seg_start is None:
                        seg_start = c
                        prev = c
                    elif c - prev <= 3:
                        prev = c
                    else:
                        if prev - seg_start + 1 >= min_frac_width:
                            for fp in range(seg_start, prev + 1):
                                fracture_zone.add(int(fp))
                        seg_start = c
                        prev = c
                if seg_start is not None and prev - seg_start + 1 >= min_frac_width:
                    for fp in range(seg_start, prev + 1):
                        fracture_zone.add(int(fp))
            print(f"  Fracture zone detected: {len(fracture_zone)} pixels "
                  f"({len(fracture_zone)/w*100:.1f}%, "
                  f"trend_sigma={trend_sigma}, threshold multiplier={dip_mult}, "
                  f"min width >= {max(15, min_layer_width * 3)} px)")

        # ============================================================
        # Estimate dominant lamina slope (used for slope-aware clustering).
        # Laminae should be roughly longitudinal (small dip), with small tilts allowed.
        # HoughLinesP detection must respect ``max_dip_angle_deg`` to reject candidate
        # lines with extreme tilt (strong cracks / scan marks) that would otherwise
        # dominate the slope estimate.
        # ============================================================
        post_slope, post_conf = self._estimate_dominant_slope(max_dip_angle_deg)
        self._dominant_slope = post_slope
        self._dominant_slope_confidence = post_conf
        slope_deg = math.degrees(math.atan(abs(post_slope))) if post_slope else 0.0
        print(f"  Dominant lamina dip (post-alignment): {slope_deg:.1f} deg off vertical "
              f"(dx/dy={post_slope:+.3f}, conf={post_conf:.2f}, limit <= {max_dip_angle_deg:.0f} deg)")

        # ============================================================
        # Round 1: per-scan-line independent candidate detection.
        # Threshold strategy: rather miss a few weak laminae than promote noise to
        # laminae. The downstream 2D fit still keeps weak points that genuinely line
        # up, so this stage can afford to be conservative.
        # ============================================================
        if self.gray is not None:
            img_contrast = np.std(self.gray.astype(np.float64))
            img_brightness = np.mean(self.gray.astype(np.float64))
        else:
            img_contrast = np.std(self.processed.astype(np.float64))
            img_brightness = np.mean(self.processed.astype(np.float64))

        is_dark = img_brightness < 90

        # Keep change-point detection permissive: try to retain every grayscale jump
        # visible to the human eye. Noise / pseudo-laminae are filtered later by
        # [slope voting + 2D line fitting].
        if is_dark and img_contrast < 25:
            alpha_step, alpha_grad = 0.4, 0.7
            print(f"  Dark low-contrast (mean={img_brightness:.1f}, std={img_contrast:.1f}); sensitive mode")
        elif is_dark:
            alpha_step, alpha_grad = 0.5, 0.8
            print(f"  Dark core (mean={img_brightness:.1f}, std={img_contrast:.1f}); standard mode")
        elif img_contrast < 25:
            alpha_step, alpha_grad = 0.5, 0.9
            print(f"  Low-contrast image (std={img_contrast:.1f}); standard mode")
        elif img_contrast < 40:
            alpha_step, alpha_grad = 0.7, 1.1
            print(f"  Medium-low-contrast image (std={img_contrast:.1f}); standard mode")
        else:
            alpha_step, alpha_grad = 0.9, 1.3
            print(f"  Normal-contrast image (std={img_contrast:.1f}); standard mode")

        # Absolute magnitude threshold filters only "clearly-noise" tiny fluctuations
        # without acting as a strong constraint.
        # Real laminae in dark mudstone/shale often have a Delta-gray of only 2-4
        # (e.g. organic-rich layer vs. matrix differing by 2-3 levels); a hard floor
        # of 4.0 would wipe all of them out. The sigmoid in preprocessing already
        # amplifies the narrow-band contrast by ~2x, but the magnitude filter still
        # runs on the *raw* grayscale (closer to physics), so the floor must be
        # lowered here.
        if min_delta_gray is None:
            if is_dark and img_contrast < 25:
                # Dark low-contrast: physical Delta is often <= 3; a floor of 2.5 avoids
                # carving out real laminae.
                floor = 2.5
            elif is_dark:
                floor = 3.0
            else:
                floor = 4.0
            min_delta_gray = float(max(floor, 0.10 * img_contrast))
            print(f"  Auto absolute magnitude threshold: |Delta gray| >= {min_delta_gray:.1f} "
                  f"(floor={floor:.1f}, 0.10*std={0.10 * img_contrast:.2f}, lower floor used for dark samples)")
        else:
            min_delta_gray = float(min_delta_gray)
            print(f"  User-specified magnitude threshold: |Delta gray| >= {min_delta_gray:.1f}")
        # Store on the instance so validation-line detection and the cluster fit can reuse it
        self._min_delta_gray = min_delta_gray
        self._max_dip_angle_deg = float(max_dip_angle_deg)

        for line_idx, y in enumerate(scan_lines):
            if y >= h:
                continue

            gray_row = self.processed[y, :].astype(np.float64)

            # ---- Method A: multi-scale step detection ----
            # Small windows catch narrow laminae; large windows catch wide gentle changes.
            # We union the results.
            points_step = set()
            row_smooth = gaussian_filter1d(gray_row, sigma=1.5)

            for scale_win in [max(5, min_layer_width // 2), max(8, min_layer_width), max(15, min_layer_width * 2)]:
                lr_kernel = np.zeros(scale_win * 2)
                lr_kernel[:scale_win] = -1.0 / scale_win
                lr_kernel[scale_win:] = 1.0 / scale_win
                step_signal = np.abs(np.convolve(row_smooth, lr_kernel, mode='same'))

                vr = step_signal[edge_filter_pixels:w - edge_filter_pixels]
                if len(vr) > 0 and np.std(vr) > 0:
                    step_thresh = np.mean(vr) + alpha_step * np.std(vr)
                else:
                    step_thresh = np.mean(step_signal) + alpha_step * np.std(step_signal)

                peaks, _ = find_peaks(step_signal,
                                      height=step_thresh,
                                      distance=max(3, min_layer_width // 2))
                points_step.update(peaks.tolist())

            # ---- Method B: log-gradient enhancement + adaptive threshold (paper, sec. 3.3) ----
            grad_raw = np.abs(np.diff(gray_row))
            grad_log = np.log1p(grad_raw)
            grad_smooth = gaussian_filter1d(grad_log, sigma=2)
            grad_std = np.std(grad_smooth)
            if grad_std > 0:
                peak_height = np.mean(grad_smooth) + alpha_grad * grad_std
                peaks, _ = find_peaks(grad_smooth,
                                      height=peak_height,
                                      distance=max(3, min_layer_width // 2),
                                      prominence=grad_std * 0.1)
                points_grad = set(peaks.tolist())
            else:
                points_grad = set()

            # ---- Method D (dark-sample only): relative-change detection ----
            # Dark samples have small absolute gradients but large relative change,
            # e.g. 30 -> 40 is a 33% jump.
            points_rel = set()
            if is_dark:
                # Compute relative change on the raw grayscale (without the gradient overlay)
                src_row = self.gray[y, :].astype(np.float64) if self.gray is not None else gray_row
                src_smooth = gaussian_filter1d(src_row, sigma=2)
                # Local mean as denominator
                local_mean = gaussian_filter1d(src_smooth, sigma=20)
                local_mean = np.maximum(local_mean, 5.0)  # avoid divide-by-zero
                rel_change = np.abs(np.diff(src_smooth)) / local_mean[:-1]
                rel_smooth = gaussian_filter1d(rel_change, sigma=1.5)
                rel_std = np.std(rel_smooth)
                if rel_std > 0:
                    rel_thresh = np.mean(rel_smooth) + alpha_grad * rel_std
                    rel_peaks, _ = find_peaks(rel_smooth,
                                              height=rel_thresh,
                                              distance=max(3, min_layer_width // 2),
                                              prominence=rel_std * 0.1)
                    points_rel = set(rel_peaks.tolist())

            # ---- Method C (auxiliary): binary-diff + Canny ----
            row_bin = binary[y, :]
            diff_bin = np.abs(np.diff(row_bin.astype(np.int32)))
            points_bin = set(np.where(diff_bin > 0)[0])

            band_half = 2
            y_lo = max(0, y - band_half)
            y_hi = min(h, y + band_half + 1)
            edge_band = edges[y_lo:y_hi, :]
            edge_proj = np.sum(edge_band, axis=0).astype(np.float32)
            if np.std(edge_proj) > 0:
                edge_thresh = np.mean(edge_proj) + 0.5 * np.std(edge_proj)
                points_canny = set(np.where(edge_proj > edge_thresh)[0])
            else:
                points_canny = set()

            # ---- Fusion ----
            tolerance = max(3, min_layer_width // 3)
            primary_points = points_step | points_grad | points_rel
            secondary_only = (points_bin | points_canny) - primary_points

            extra = []
            for pt in secondary_only:
                if pt < edge_filter_pixels or pt >= (w - edge_filter_pixels):
                    continue
                support = sum(1 for mps in [points_bin, points_canny]
                              if any(abs(mp - pt) <= tolerance for mp in mps))
                if support >= 2:
                    extra.append(pt)

            all_detected = set()
            for pt in primary_points:
                if edge_filter_pixels <= pt < (w - edge_filter_pixels):
                    all_detected.add(pt)
            all_detected.update(extra)

            # Remove fracture zone
            if len(fracture_zone) < w * 0.3:
                all_detected -= fracture_zone

            high_conf_points = sorted(all_detected)

            # Merge points that are too close
            filtered_points = []
            last_point = -min_layer_width
            for point in high_conf_points:
                if point - last_point >= min_layer_width * 0.8:
                    filtered_points.append(point)
                    last_point = point

            # ---- Magnitude filter: drop changes that are too small to be lamina ----
            # Use ``self.gray`` (raw) rather than ``self.processed`` (CLAHE-stretched);
            # it is closer to the physical change.
            gray_src = self.gray if self.gray is not None else self.processed
            magnitude_kept = []
            for pt in filtered_points:
                hw = max(5, min_layer_width)
                x_lo = max(0, pt - hw)
                x_hi = min(w, pt + hw + 1)
                if pt - x_lo < 2 or x_hi - pt < 2:
                    continue
                left_seg = gray_src[y, x_lo:pt].astype(np.float64)
                right_seg = gray_src[y, pt:x_hi].astype(np.float64)
                delta = abs(float(np.mean(right_seg)) - float(np.mean(left_seg)))
                if delta >= min_delta_gray:
                    magnitude_kept.append(pt)

            print(f"  Scan line y={y}: multi-scale detected {len(filtered_points)} -> "
                  f"after magnitude filter {len(magnitude_kept)} (|Delta| >= {min_delta_gray:.1f})")
            self.layers.append({"y": y, "points": magnitude_kept})

        # We no longer apply a hard "same-x" consistency filter -- it kills all tilted
        # laminae. The downstream [slope voting + 2D fit] is the proper filter.
        for layer_data in self.layers:
            layer_data.setdefault("consistency_scores", {})
        print(f"Scan-line detection complete: {len(self.layers)} scan line(s) "
              f"(total candidates {sum(len(ld['points']) for ld in self.layers)}; "
              f"clustering + fitting will adjudicate next)")

        # --- Validation ---
        sorted_scan_lines = sorted(self.scan_lines)
        validation_success = False
        if len(sorted_scan_lines) >= 2:
            validation_success = self._validate_regions_between_scan_lines(
                sorted_scan_lines, min_validation_lines, min_layer_width)

        # ============================================================
        # Vote-based alignment refinement.
        # The first alignment may leave a residual of a few degrees. Use every detected
        # point to vote on the residual slope, then apply the residual as an *additional*
        # shear simultaneously to all images and all detected points. Because points and
        # images move together, ``_cluster_to_laminae`` sees laminae with slope ~0 in the
        # new coordinate system, and the connection lines render as *vertical* in the
        # final figure.
        # ============================================================
        if align_core and self.aligned:
            n_total_lines_chk = len(self.scan_lines) + len(getattr(self, '_validation_results', []) or [])
            _min_supp_ref = max(3, n_total_lines_chk // 2) if n_total_lines_chk >= 4 else 2
            # Safety cap on the per-iteration refinement: additional shear <=
            # tan(7 deg) ~ 0.123.
            # Important: the first alignment uses the dominant slope from long Hough
            # edges (laminae, core boundaries) -- the visible lamina direction. After
            # that pass, the visually-evident laminae should already be near vertical.
            # The "change points" on each scan line, however, include many narrow
            # cracks / scratches / scan noise, which may still appear tilted (sometimes
            # heavily so). If we let the vote chase those without limit, the image gets
            # pulled toward those minor features, *over-shooting* and shearing real
            # laminae into the opposite direction (precisely the "sheared twice"
            # illusion the user reports). So we cap the per-iteration magnitude at
            # 7 deg; anything larger is skipped, preserving the first Hough alignment.
            MAX_REFINE_SLOPE = 0.123
            # At most two iterations: the second pass mainly confirms that no large
            # directional residual remains. A single vote has small jitter due to grid
            # discretization (~0.011 = 0.6 deg), so chasing endlessly is pointless.
            for _refine_it in range(2):
                raw_pts_chk = []
                for ld in self.layers:
                    for p in ld["points"]:
                        raw_pts_chk.append((int(p), int(ld["y"])))
                for vr in getattr(self, '_validation_results', []) or []:
                    for p in vr.get("points", []):
                        raw_pts_chk.append((int(p), int(vr["y"])))
                if len(raw_pts_chk) < 12:
                    break
                res_slope, res_score, _ = self._vote_slope_from_points(
                    raw_pts_chk, max_dip_angle_deg=max_dip_angle_deg,
                    tolerance_px=max(4, min_layer_width), min_support=_min_supp_ref,
                )
                rdeg = math.degrees(math.atan(abs(res_slope))) if res_slope else 0.0
                # Convergence: residual dip < grid resolution (0.025 ~ 1.4 deg)
                # is considered converged. Otherwise the discrete vote grid will keep
                # bouncing back and forth.
                if abs(res_slope) < 0.025 or res_score < 200:
                    if _refine_it == 0:
                        print(f"  [align-vote refinement] residual vote_slope={res_slope:+.4f} "
                              f"(~{rdeg:.2f} deg, score={res_score}) already small enough; "
                              f"skipping further refinement")
                    else:
                        print(f"  [align-vote refinement #{_refine_it+1}] residual vote_slope={res_slope:+.4f} "
                              f"(~{rdeg:.2f} deg, score={res_score}) converged")
                    break
                # Residual magnitude beyond the safety cap -> very likely minor features
                # (cracks / scratches) skewing the vote; refuse to chase, otherwise we
                # over-shear into the opposite direction.
                if abs(res_slope) > MAX_REFINE_SLOPE:
                    print(f"  [align-vote refinement #{_refine_it+1}] residual vote_slope={res_slope:+.4f} "
                          f"(~{rdeg:.2f} deg, score={res_score}) exceeds safety cap "
                          f"{MAX_REFINE_SLOPE:.3f} (~7 deg); likely contaminated by "
                          f"minor features. Skipping to preserve the first Hough alignment.")
                    break
                extra = -float(res_slope)
                # Key: accumulate the total shear and warp *the originals* exactly once,
                # rather than stacking warps on already-warped images. Otherwise the
                # BORDER_CONSTANT triangles appear on both sides, looking like a double shear.
                self.alignment_angle = float(self.alignment_angle) + extra
                total_shear = float(self.alignment_angle)
                shear_axis = getattr(self, 'shear_axis', 'x')
                if shear_axis == 'y':
                    M_total = np.float32([[1.0, 0.0, 0.0], [total_shear, 1.0, 0.0]])
                else:
                    M_total = np.float32([[1.0, total_shear, 0.0], [0.0, 1.0, 0.0]])
                def _wf_from_orig(img):
                    if img is None:
                        return None
                    try:
                        bv = float(np.mean(img))
                    except Exception:
                        bv = 128.0
                    bvt = (bv, bv, bv) if img.ndim == 3 else bv
                    return cv2.warpAffine(img, M_total, (img.shape[1], img.shape[0]),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=bvt)
                self.gray = _wf_from_orig(self._orig_gray_pre_align)
                self.image = _wf_from_orig(self._orig_image_pre_align)
                self.enhanced_no_grad = _wf_from_orig(self._orig_enhanced_no_grad_pre_align)
                if getattr(self, '_orig_gray_nl_pre_align', None) is not None:
                    self.gray_nonlinear_enhanced = _wf_from_orig(self._orig_gray_nl_pre_align)
                # ``processed`` = ``enhanced_no_grad`` overlaid with a fresh Sobel_X
                # (cross-lamina gradient), keeping the gradient in sync with the
                # straightened vertical laminae.
                if self.enhanced_no_grad is not None:
                    sx_ = cv2.Sobel(self.enhanced_no_grad, cv2.CV_64F, 1, 0, ksize=3)
                    sxa_ = np.uint8(np.clip(np.abs(sx_), 0, 255))
                    self.processed = cv2.addWeighted(self.enhanced_no_grad, 0.7, sxa_, 0.3, 0)
                else:
                    self.processed = _wf_from_orig(self._orig_processed_pre_align)
                # Re-warp the preprocessing step images from the originals as well so
                # the paper figures do not retain double gray-triangle bands.
                steps_ref = getattr(self, '_preprocess_steps', {})
                orig_steps = getattr(self, '_orig_preprocess_steps_pre_align', {}) or {}
                for key in list(steps_ref.keys()):
                    if orig_steps.get(key) is not None:
                        steps_ref[key] = _wf_from_orig(orig_steps[key])
                original_processed = self.processed.copy()
                # Recompute ``self.binary`` so it shares the same composite-shear
                # coordinate system as ``self.processed``; otherwise
                # ``binary_image.png`` (first-warp) and ``05_geometry_corrected.png``
                # (composite-warp) would have inconsistent shear, visually looking
                # like one of them was "sheared again".
                if self.processed is not None:
                    if threshold_method == 'otsu':
                        _, _new_bin = cv2.threshold(self.processed, 0, 255,
                                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    elif threshold_method == 'adaptive':
                        _new_bin = cv2.adaptiveThreshold(self.processed, 255,
                                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 11, 2)
                    else:
                        _, _new_bin = cv2.threshold(self.processed, 100, 255, cv2.THRESH_BINARY)
                    self.binary = _new_bin.copy()
                    binary = _new_bin  # update the local ``binary`` so binary_image.png picks it up
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
                # Move every detected point in lockstep with the shear: x_new = x_old + extra * y
                for ld in self.layers:
                    ld_y = int(ld["y"])
                    ld["points"] = [int(round(p + extra * ld_y))
                                    for p in ld["points"]
                                    if 0 <= int(round(p + extra * ld_y)) < w]
                    if "validated_points" in ld:
                        ld["validated_points"] = [int(round(p + extra * ld_y))
                                                  for p in ld["validated_points"]
                                                  if 0 <= int(round(p + extra * ld_y)) < w]
                for vr in getattr(self, '_validation_results', []) or []:
                    v_y = int(vr["y"])
                    vr["points"] = [int(round(p + extra * v_y))
                                    for p in vr.get("points", [])
                                    if 0 <= int(round(p + extra * v_y)) < w]
                print(f"  [align-vote refinement #{_refine_it+1}] residual vote_slope={res_slope:+.4f} "
                      f"(~{rdeg:.2f} deg, score={res_score}) -> additional shear {extra:+.4f}, "
                      f"cumulative={self.alignment_angle:+.4f} (images and points moved in sync)")
            # Release the pre-alignment cache after refinement to free memory
            for _attr in ('_orig_image_pre_align', '_orig_gray_pre_align',
                          '_orig_enhanced_no_grad_pre_align', '_orig_processed_pre_align',
                          '_orig_preprocess_steps_pre_align', '_orig_gray_nl_pre_align'):
                if hasattr(self, _attr):
                    setattr(self, _attr, None)

        # --- Render result: mark every candidate (main scan lines + validation lines) ---
        result_visual = self.image.copy()
        layer_line_width = 2 if self.width > 1000 else 1

        for y in self.scan_lines:
            cv2.line(result_visual, (0, y), (w - 1, y), (0, 255, 0), 1)
        # Validation lines (blue dashed)
        for vr in getattr(self, '_validation_results', []) or []:
            vy = int(vr["y"])
            if 0 <= vy < h:
                cv2.line(result_visual, (0, vy), (w - 1, vy), (255, 80, 0), 1, cv2.LINE_AA)

        validated_count = 0
        # Main-scan-line candidates: short red ticks + blue dots
        for layer_data in self.layers:
            y = layer_data["y"]
            for pt in layer_data.get("validated_points", layer_data.get("points", [])):
                cv2.line(result_visual, (pt, max(0, y - 8)), (pt, min(h - 1, y + 8)), (0, 0, 255), layer_line_width)
                cv2.circle(result_visual, (pt, y), 3, (255, 0, 0), -1)
                validated_count += 1
        # Validation-line candidates: small orange circles (distinguished from main lines
        # so users can see every "change point")
        for vr in getattr(self, '_validation_results', []) or []:
            vy = int(vr["y"])
            for pt in vr.get("points", []):
                if 0 <= vy < h and 0 <= pt < w:
                    cv2.circle(result_visual, (int(pt), vy), 2, (0, 165, 255), -1)

        self.processed = original_processed
        os.makedirs(self.output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(self.output_dir, "layer_detection.png"), result_visual)

        # Diagnostic intermediates (skip in batch mode for faster I/O and smaller output)
        if self.save_diagnostics:
            cv2.imwrite(os.path.join(self.output_dir, "binary_image.png"), binary)
            cv2.imwrite(os.path.join(self.output_dir, "canny_edges.png"), edges)

        # If validation found nothing, fall back to the raw candidates
        if validated_count == 0:
            total_raw = sum(len(ld["points"]) for ld in self.layers)
            if total_raw > 0:
                print(f"No valid points after validation; falling back to raw candidates "
                      f"(total {total_raw})")
                self._detection_diagnostics["fallbacks_triggered"].append(
                    f"Cross-line validation failed; using unvalidated candidates ({total_raw} total)"
                )
                for ld in self.layers:
                    ld["validated_points"] = ld["points"]
                validated_count = total_raw

        # ============================================================
        # Round 3: cross-scan-line clustering -> unique laminae.
        # A true lamina should be detected at roughly the same x on multiple scan
        # lines. This step merges them into one unique lamina and rejects isolated
        # detections that appear on only 1-2 scan lines (usually noise or local cracks)
        # via the "minimum support lines" criterion.
        # ============================================================
        self._last_min_layer_width = min_layer_width

        # Post-alignment policy (applies in BOTH single-image and batch modes):
        # once the core is sheared straight, real laminae must be near vertical.
        # Default cap is 7 deg, raised to ``batch_max_dip_after_align_deg`` when
        # batch mode supplied an explicit override.
        post_align_dip_cap = float(getattr(self, "batch_max_dip_after_align_deg", 7.0))
        # When the user explicitly disables vertical snap (single-image edge case
        # with unusual cores), set ``allow_tilted_after_align = True`` on the
        # detector before calling detect_layers().
        allow_tilted = bool(getattr(self, "allow_tilted_after_align", False))

        if self.user_dip_angle_deg is not None:
            # Manual direction mode: the user dictates the orientation, so widen the
            # dip cap enough that the requested angle is never rejected as "too tilted".
            max_dip_angle_deg = max(float(max_dip_angle_deg),
                                    abs(self.user_dip_angle_deg) + 10.0)
            self._max_dip_angle_deg = max_dip_angle_deg
            print(f"  [manual-angle] user dip = {self.user_dip_angle_deg:+.1f} deg off vertical; "
                  f"drawing/statistics follow this direction (dip cap raised to {max_dip_angle_deg:.1f} deg)")
        elif getattr(self, "aligned", False) and not allow_tilted:
            max_dip_angle_deg = min(float(max_dip_angle_deg), post_align_dip_cap)
            self._max_dip_angle_deg = max_dip_angle_deg
            print(f"  [post-align] dip cap: {max_dip_angle_deg:.1f} deg (laminae must be near vertical)")

        self._cluster_to_laminae(max_dip_angle_deg=max_dip_angle_deg)

        if self.user_dip_angle_deg is not None:
            # Force every lamina onto the manually specified direction.
            self._apply_user_lamina_angle()
        # Apply the post-align vertical snap whenever the image was aligned,
        # regardless of whether batch dip calibration was used.
        elif getattr(self, "aligned", False) and not allow_tilted:
            self._apply_batch_lamina_constraints()
        elif getattr(self, "batch_lamina_mode", False):
            # Batch mode without shear still benefits from the calibration cap.
            self._apply_batch_lamina_constraints()
        ls = getattr(self, '_lamina_settings', {})
        n_unique = ls.get('n_valid_laminae', 0)
        n_clusters = ls.get('n_clusters', 0)
        min_sup = ls.get('min_support', 0)
        n_main_sl = ls.get('n_main_scan_lines', 0)
        n_val_sl = ls.get('n_validation_lines', 0)
        print(f"  [2D line-fit clustering] {n_clusters} candidate(s) -> {n_unique} valid lamina(e) "
              f"(requires >= {min_sup}/{n_main_sl + n_val_sl} line support, dip <= {ls.get('max_dip_angle_deg', 0):.0f} deg, "
              f">= 80% points residual <= {ls.get('max_residual_px', 0):.1f} px)")
        if n_clusters > n_unique:
            print(f"    Rejection breakdown -- insufficient support: {ls.get('rejected_by_support', 0)}  "
                  f"excessive dip: {ls.get('rejected_by_dip', 0)}  "
                  f"residual too large: {ls.get('rejected_by_residual', 0)}")

        has_layers = validated_count > 0
        n_fb = len(self._detection_diagnostics["fallbacks_triggered"])
        if n_fb > 0:
            print(f"  [diag] {n_fb} filtering fallback(s) triggered during detection "
                  f"(see ``_detection_diagnostics``)")
        print(f"=== Detection complete: candidate change points={validated_count}, "
              f"unique laminae={n_unique}, success={has_layers} ===")

        return has_layers
    def _estimate_dominant_slope(self, max_dip_angle_deg=45.0, src=None):
        """Estimate the dominant lamina slope dx/dy via HoughLinesP (with a dip-angle cap).

        With the core laid horizontally and laminae near vertical, accept only candidate
        lines whose ``|dx/dy| <= tan(max_dip_angle_deg)`` so long cracks / scan marks
        (very tilted) cannot dominate the slope estimate.

        ``src`` may be passed explicitly:
          - Pre-alignment: pass ``self.gray`` to estimate the original tilt.
          - Post-alignment: defaults to ``enhanced_no_grad`` / ``processed``.

        Returns ``(slope, confidence)``. ``slope = dx/dy``; ``confidence`` in [0,1]
        reflects candidate concentration (low IQR + many samples -> high confidence).
        """
        if src is None:
            if hasattr(self, 'enhanced_no_grad') and self.enhanced_no_grad is not None:
                src = self.enhanced_no_grad
            elif self.processed is not None:
                src = self.processed
        if src is None:
            return 0.0, 0.0
        h, w = src.shape[:2]
        med = float(np.median(src)) if src.size else 0.0
        canny_low = int(max(10, 0.55 * med))
        canny_high = int(min(255, 1.40 * med))
        try:
            edges = cv2.Canny(src, canny_low, canny_high, apertureSize=3)
        except Exception:
            return 0.0, 0.0
        max_slope_abs = math.tan(math.radians(max_dip_angle_deg))
        # Two Hough rounds: long lines (high weight) + short lines (fallback). This
        # helps robust hits on low-contrast images.
        rounds = [
            dict(threshold=60, minLineLength=max(40, h // 4), maxLineGap=8),
            dict(threshold=35, minLineLength=max(20, h // 6), maxLineGap=14),
        ]
        slopes = []
        lengths = []
        for rp in rounds:
            try:
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, **rp)
            except Exception:
                lines = None
            if lines is None:
                continue
            weight_mul = 1.5 if rp is rounds[0] else 1.0
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                dy = float(y2 - y1)
                dx = float(x2 - x1)
                if abs(dy) < 5:
                    continue
                slope = dx / dy
                if abs(slope) > max_slope_abs:
                    continue
                slopes.append(slope)
                lengths.append(math.hypot(dx, dy) * weight_mul)
        if not slopes:
            return 0.0, 0.0
        slopes_arr = np.asarray(slopes, dtype=np.float64)
        lengths_arr = np.asarray(lengths, dtype=np.float64)
        order = np.argsort(slopes_arr)
        sorted_slopes = slopes_arr[order]
        sorted_weights = lengths_arr[order]
        cumw = np.cumsum(sorted_weights)
        total_w = float(cumw[-1])
        if total_w <= 0:
            return 0.0, 0.0
        idx = int(np.searchsorted(cumw, total_w / 2.0))
        idx = max(0, min(idx, len(sorted_slopes) - 1))
        dominant = float(sorted_slopes[idx])
        q25_idx = max(0, min(int(np.searchsorted(cumw, total_w * 0.25)), len(sorted_slopes) - 1))
        q75_idx = max(0, min(int(np.searchsorted(cumw, total_w * 0.75)), len(sorted_slopes) - 1))
        iqr_slope = float(sorted_slopes[q75_idx] - sorted_slopes[q25_idx])
        conf_iqr = max(0.0, min(1.0, (0.4 - iqr_slope) / 0.35))
        conf_n = min(1.0, len(slopes) / 30.0)
        confidence = float(0.6 * conf_iqr + 0.4 * conf_n)
        return dominant, confidence
    def _vote_slope_from_points(self, points_xy, max_dip_angle_deg=45.0,
                                tolerance_px=6, min_support=2, n_steps=361):
        """Vote on the dominant dx/dy slope from detected (x, y) change points.

        Principle: for each candidate slope ``s``, project every point onto
        ``x' = x - s * y``. Points on the same lamina cluster together in that
        coordinate.

        Score = (supported clusters * 1000) + total points in supported clusters.
        The first term dominates so that slopes producing "clear, many clusters with
        enough per-cluster support" win, preventing one or two spurious large clusters
        from dictating the answer.

        A parabolic interpolation lifts the peak to sub-grid precision, removing
        discrete-grid jitter.
        """
        if not points_xy or len(points_xy) < 4:
            return 0.0, 0, {}
        pts = np.asarray(points_xy, dtype=np.float64)
        xs = pts[:, 0]
        ys = pts[:, 1]
        max_slope = math.tan(math.radians(max_dip_angle_deg))
        slope_grid = np.linspace(-max_slope, max_slope, n_steps)

        scores = np.zeros(n_steps, dtype=np.int64)
        for k, s in enumerate(slope_grid):
            x_adj = xs - s * ys
            order = np.argsort(x_adj)
            sorted_xadj = x_adj[order]
            sorted_ys = ys[order]
            n_supp_clusters = 0
            supp_total_pts = 0
            cur_y_set = {float(sorted_ys[0])}
            cur_pts = 1
            for i in range(1, len(sorted_xadj)):
                if sorted_xadj[i] - sorted_xadj[i-1] <= tolerance_px:
                    cur_y_set.add(float(sorted_ys[i]))
                    cur_pts += 1
                else:
                    if len(cur_y_set) >= min_support:
                        n_supp_clusters += 1
                        supp_total_pts += cur_pts
                    cur_y_set = {float(sorted_ys[i])}
                    cur_pts = 1
            if len(cur_y_set) >= min_support:
                n_supp_clusters += 1
                supp_total_pts += cur_pts
            scores[k] = n_supp_clusters * 1000 + supp_total_pts

        best_k = int(np.argmax(scores))
        best_score = int(scores[best_k])
        best_slope = float(slope_grid[best_k])
        # Parabolic interpolation: fit y = a x^2 + b x + c to three points; max at
        # -b/(2a). Drops grid quantization error from ~step/2 to ~step/10.
        if 0 < best_k < n_steps - 1 and scores[best_k] > 0:
            y_m = float(scores[best_k - 1])
            y_0 = float(scores[best_k])
            y_p = float(scores[best_k + 1])
            denom = (y_m - 2 * y_0 + y_p)
            if abs(denom) > 1e-9:
                offset = 0.5 * (y_m - y_p) / denom
                # Only accept the interpolation near the peak (-1 < offset < 1)
                if -1.0 < offset < 1.0:
                    step = float(slope_grid[1] - slope_grid[0])
                    best_slope = float(slope_grid[best_k]) + offset * step
        score_curve = {float(s): int(sc) for s, sc in zip(slope_grid, scores)}
        return best_slope, int(best_score), score_curve
    def _cluster_to_laminae(self, tolerance_px=None, min_support=None,
                            max_dip_angle_deg=None, max_residual_px=None):
        """Cluster detected points across main + validation lines and run a 2D line fit.

        A real lamina is a relatively straight line (some tilt allowed). This method:
          1) Builds single-link pre-clusters from all detected (x, y) points based
             on x proximity.
          2) Runs a first-order least-squares fit ``x = a + b * y`` on each cluster
             (tilt allowed).
          3) Computes the dip angle ``atan(|b|)`` and the fit residuals.
          4) Accepts a cluster as a "valid lamina" only if all of the following hold:
             ``support_lines >= min_support``,
             ``dip_angle <= max_dip_angle_deg``,
             ``80th-percentile residual <= max_residual_px``.

        Clusters that fail any condition are marked ``is_valid = False`` with
        ``rejection_reasons`` populated, and rendered as smudges in the visualization.
        """
        main_lines = list(self.layers) if self.layers else []
        val_results = list(getattr(self, '_validation_results', []) or [])
        n_main = len(main_lines)
        n_val = len(val_results)
        total_lines = n_main + n_val

        if total_lines == 0:
            self.laminae = []
            self._lamina_settings = {
                "tolerance_px": 0, "min_support": 0,
                "n_main_scan_lines": 0, "n_validation_lines": 0, "n_scan_lines": 0,
                "n_clusters": 0, "n_valid_laminae": 0,
                "max_dip_angle_deg": 0.0, "max_residual_px": 0,
                "candidate_points": 0,
            }
            return self.laminae

        if tolerance_px is None:
            mlw = getattr(self, '_last_min_layer_width', 5)
            tolerance_px = max(12, mlw * 2)

        if min_support is None:
            # A real lamina is a continuous line, so it should be traceable across
            # MOST of the scan lines: we require a configurable fraction (default
            # 70%) of the total lines to carry a point that lands on the fitted
            # lamina line. This rejects short 2-3 point fragments (cracks / local
            # texture / scan marks) that are not continuous laminae. The fraction
            # is read from ``self.min_support_ratio`` so it can be tuned per run.
            ratio = float(getattr(self, "min_support_ratio", 0.70) or 0.70)
            ratio = min(max(ratio, 0.1), 1.0)
            if total_lines >= 4:
                # ceil so that "70%+" is actually guaranteed (e.g. 7/9 = 77.8%).
                min_support = max(3, int(math.ceil(ratio * total_lines)))
            else:
                # Too few lines to apply a fraction; require all of them.
                min_support = max(2, total_lines)
        min_support = max(1, min(int(min_support), total_lines))

        if max_dip_angle_deg is None:
            max_dip_angle_deg = float(getattr(self, '_max_dip_angle_deg', 45.0))
        max_dip_angle_deg = float(max_dip_angle_deg)

        if max_residual_px is None:
            # Residual tolerance = cluster tolerance so that points sitting at the
            # tolerance boundary are not labeled "residual too large" simply because
            # the fit line happens to pass through the cluster center.
            max_residual_px = float(tolerance_px)
        max_residual_px = float(max_residual_px)

        # 1) Collect every candidate point (x, y, src_idx, src_type) from main + validation lines.
        # Note: use the raw ``points`` rather than ``validated_points`` so that the
        # slope vote and 2D fit decide authenticity, avoiding upstream over-filtering
        # that would drop real laminae.
        raw_pts_xy = []  # (x, y) only, for voting
        all_points_tmp = []  # full tuples (x_adjusted filled in below)
        for i, sr in enumerate(main_lines):
            y = int(sr["y"])
            for pt in sr.get("points", []):
                raw_pts_xy.append((int(pt), y))
                all_points_tmp.append((int(pt), y, i, 0))
        for j, vr in enumerate(val_results):
            y = int(vr["y"])
            for pt in vr.get("points", []):
                raw_pts_xy.append((int(pt), y))
                all_points_tmp.append((int(pt), y, n_main + j, 1))

        # 2) Slope voting: find the best ``slope`` (dominant) from the candidate points
        # themselves. Much more robust than HoughLinesP-on-image because it looks at
        # the change points directly. The vote uses a tighter tolerance
        # (~min_layer_width) so it can resolve fine slope differences; the clustering
        # stage below uses ``tolerance_px`` to absorb minor offsets.
        vote_tol = max(4, int(getattr(self, '_last_min_layer_width', 5)))
        vote_slope, vote_score, _vote_curve = self._vote_slope_from_points(
            raw_pts_xy,
            max_dip_angle_deg=max_dip_angle_deg,
            tolerance_px=vote_tol,
            min_support=min_support,
            n_steps=181,  # 0.5 deg resolution
        )
        # Adopt the vote if its support exceeds Hough/pre-align; otherwise keep falling back.
        hough_slope = float(getattr(self, '_dominant_slope', 0.0) or 0.0)
        pre_slope = float(getattr(self, '_pre_align_slope', 0.0) or 0.0)
        # ``slope_hint`` priority:
        #   batch group calibration > vote > hough > pre_slope > 0
        slope_hint = 0.0
        slope_hint_source = "none"
        user_slope = getattr(self, "_user_slope", None)
        batch_group_hint = getattr(self, "batch_group_slope_hint", None)
        if user_slope is not None:
            slope_hint = float(user_slope)
            slope_hint_source = "user_input"
        elif batch_group_hint is not None and getattr(self, "batch_lamina_mode", False):
            slope_hint = float(batch_group_hint)
            slope_hint_source = "batch_group"
        elif vote_score >= max(4, min_support * 2):
            slope_hint = vote_slope
            slope_hint_source = "vote"
        elif abs(hough_slope) >= 0.02:
            slope_hint = hough_slope
            slope_hint_source = "hough_post_align"
        elif abs(pre_slope) >= 0.05:
            slope_hint = pre_slope
            slope_hint_source = "hough_pre_align"
        # Critical safety: once the image is aligned, true laminae should be near
        # vertical (slope ~ 0). If the vote on an aligned image still reports a large
        # |slope| (e.g. 0.5+), it is almost certainly due to minor features (narrow
        # cracks / scratches). Using that value for the cluster x_adj correction
        # would *break* near-vertical laminae across multiple clusters.
        # Once aligned -> slope_hint capped at +/- tan(7 deg) = 0.123.
        max_hint_slope = 0.123
        if getattr(self, "batch_lamina_mode", False):
            batch_cap = getattr(self, "batch_max_dip_after_align_deg", 7.0)
            max_hint_slope = math.tan(math.radians(float(batch_cap)))
        if (slope_hint_source != "user_input"
                and getattr(self, 'aligned', False) and abs(slope_hint) > max_hint_slope):
            print(f"  [cluster] slope_hint={slope_hint:+.3f} is too large on an aligned image "
                  f"(>{max_hint_slope:.3f} ~ {math.degrees(math.atan(max_hint_slope)):.1f} deg); "
                  f"resetting to 0 to avoid splitting near-vertical laminae")
            slope_hint = 0.0
            if slope_hint_source == "batch_group":
                slope_hint_source = "batch_group_clamped_0"
            else:
                slope_hint_source = "aligned_fallback_0"
        slope_deg_used = math.degrees(math.atan(abs(slope_hint))) if slope_hint else 0.0
        print(f"  [cluster] slope_hint = {slope_hint:+.3f} (~{slope_deg_used:.1f} deg, "
              f"source={slope_hint_source}, vote_score={vote_score})")

        # 3) Build the 5-tuple with the final ``slope_hint`` and the adjusted x.
        all_points = []
        for (x, y, src_idx, src_type) in all_points_tmp:
            x_adj = float(x) - slope_hint * float(y)
            all_points.append((int(x), int(y), int(src_idx), int(src_type), x_adj))

        if not all_points:
            self.laminae = []
            self._lamina_settings = {
                "tolerance_px": int(tolerance_px), "min_support": int(min_support),
                "n_main_scan_lines": n_main, "n_validation_lines": n_val,
                "n_scan_lines": total_lines, "n_clusters": 0, "n_valid_laminae": 0,
                "candidate_points": 0,
                "max_dip_angle_deg": max_dip_angle_deg, "max_residual_px": max_residual_px,
                "slope_hint": slope_hint,
                "slope_hint_source": slope_hint_source,
                "vote_slope": float(vote_slope),
                "vote_score": int(vote_score),
                "rejected_by_support": 0, "rejected_by_dip": 0, "rejected_by_residual": 0,
            }
            return self.laminae

        # ---- Single-link pre-clustering on ``x_adjusted`` (vs the cluster mean) ----
        all_points.sort(key=lambda p: p[4])
        clusters = [[all_points[0]]]
        cur_sum = float(all_points[0][4])
        cur_n = 1
        for pt in all_points[1:]:
            cur_mean = cur_sum / cur_n
            if abs(pt[4] - cur_mean) <= tolerance_px:
                clusters[-1].append(pt)
                cur_sum += pt[4]
                cur_n += 1
            else:
                clusters.append([pt])
                cur_sum = float(pt[4])
                cur_n = 1

        # ---- 2D line fit + geometric adjudication for each cluster ----
        h_img = int(self.height) if hasattr(self, "height") and self.height else 0
        laminae = []
        for cidx, cluster in enumerate(clusters):
            xs_arr = np.array([p[0] for p in cluster], dtype=np.float64)
            ys_arr = np.array([p[1] for p in cluster], dtype=np.float64)
            line_idxs = sorted({p[2] for p in cluster})
            main_line_idxs = sorted({p[2] for p in cluster if p[3] == 0})
            val_line_idxs = sorted({p[2] for p in cluster if p[3] == 1})
            n_support = len(line_idxs)

            # Least-squares fit x = b * y + a with a one-pass outlier removal:
            # do an initial fit, drop the few highest-residual outliers (if any),
            # then refit.
            n_outliers = 0
            if len(np.unique(ys_arr)) >= 2:
                try:
                    coeffs = np.polyfit(ys_arr, xs_arr, 1)
                    init_resid = np.abs(xs_arr - (coeffs[0] * ys_arr + coeffs[1]))
                    if len(xs_arr) >= 5 and init_resid.max() > max_residual_px:
                        # Keep points with residual < 1.5x tolerance; retain >= 60%
                        inlier_mask = init_resid <= max_residual_px * 1.5
                        if inlier_mask.sum() >= max(3, int(0.6 * len(xs_arr))):
                            xs_used = xs_arr[inlier_mask]
                            ys_used = ys_arr[inlier_mask]
                            n_outliers = int((~inlier_mask).sum())
                            coeffs = np.polyfit(ys_used, xs_used, 1)
                    slope = float(coeffs[0])
                    intercept = float(coeffs[1])
                    # Compute residuals for all points (including outliers) for
                    # adjudication and diagnostics
                    all_resid = np.abs(xs_arr - (slope * ys_arr + intercept))
                    max_res = float(np.max(all_resid))
                    mean_res = float(np.mean(all_resid))
                    # Critical: use the 80th-percentile residual instead of the max.
                    # Real laminae across many scan lines occasionally have 1-2 stray
                    # points (local texture contamination, short cracks brushing past)
                    # and ``max(resid)`` would discard the whole lamina.
                    # ``p80`` means "at least 80% of the points line up" -- if the
                    # bulk of the points are on the line, we keep it.
                    resid_p80 = float(np.quantile(all_resid, 0.80)) if len(all_resid) > 0 else 0.0
                    dip_deg = math.degrees(math.atan(abs(slope)))
                    residuals = all_resid
                except Exception:
                    slope, intercept = 0.0, float(np.mean(xs_arr))
                    residuals = np.abs(xs_arr - intercept)
                    max_res = float(np.max(residuals))
                    mean_res = float(np.mean(residuals))
                    resid_p80 = float(np.quantile(residuals, 0.80)) if len(residuals) > 0 else 0.0
                    dip_deg = 0.0
            else:
                slope, intercept = 0.0, float(np.mean(xs_arr))
                residuals = np.zeros_like(xs_arr)
                max_res = 0.0
                mean_res = 0.0
                resid_p80 = 0.0
                dip_deg = 0.0

            reasons = []
            if n_support < min_support:
                reasons.append(f"insufficient_support({n_support}<{min_support})")
            if dip_deg > max_dip_angle_deg:
                reasons.append(f"dip_too_large({dip_deg:.1f}deg>{max_dip_angle_deg:.1f}deg)")
            # Adjudicate with p80: 80% of points must land within +/- max_residual_px
            # of the fit line.
            if resid_p80 > max_residual_px:
                reasons.append(f">=80%_points_residual_too_large(p80={resid_p80:.1f}>{max_residual_px:.1f}px)")
            is_valid = len(reasons) == 0

            # Fit-line endpoints at image top and bottom
            x_top = float(slope * 0.0 + intercept)
            x_bot = float(slope * float(max(0, h_img - 1)) + intercept)

            laminae.append({
                "lamina_id": cidx + 1,
                "x_mean": float(np.mean(xs_arr)),
                "x_median": float(np.median(xs_arr)),
                "x_min": int(np.min(xs_arr)),
                "x_max": int(np.max(xs_arr)),
                "x_std": float(np.std(xs_arr)) if len(xs_arr) > 1 else 0.0,
                # Raw (x, y) list inside the cluster -- used by paper_export to look
                # up candidate intensities.
                "member_points_xy": [(int(p[0]), int(p[1])) for p in cluster],
                "member_source_types": [int(p[3]) for p in cluster],
                # 2D fit
                "fit_slope": slope,
                "fit_intercept": intercept,
                "dip_angle_deg": dip_deg,
                "max_residual_px": max_res,
                "mean_residual_px": mean_res,
                "p80_residual_px": resid_p80,
                "n_outliers_removed": n_outliers,
                "x_at_top": x_top,
                "x_at_bottom": x_bot,
                # Support
                "n_support_lines": n_support,
                "n_support_main": len(main_line_idxs),
                "n_support_validation": len(val_line_idxs),
                "support_line_indices": line_idxs,
                "support_lines_y": sorted({int(p[1]) for p in cluster}),
                "n_points_in_cluster": len(cluster),
                "support_ratio": n_support / total_lines,
                # Adjudication
                "is_valid": is_valid,
                "rejection_reasons": reasons,
            })

        # Fill "spacing to next" for valid laminae
        valid_sorted = sorted([la for la in laminae if la["is_valid"]],
                              key=lambda la: la["x_mean"])
        for i, la in enumerate(valid_sorted):
            if i + 1 < len(valid_sorted):
                la["spacing_to_next_px"] = float(
                    valid_sorted[i + 1]["x_mean"] - la["x_mean"]
                )
            else:
                la["spacing_to_next_px"] = None
        for la in laminae:
            la.setdefault("spacing_to_next_px", None)

        n_by_reason = {"support": 0, "dip": 0, "residual": 0}
        for la in laminae:
            if not la["is_valid"]:
                for r in la["rejection_reasons"]:
                    if r.startswith("insufficient_support"):
                        n_by_reason["support"] += 1
                    elif r.startswith("dip_too_large"):
                        n_by_reason["dip"] += 1
                    elif r.startswith(">=80%_points_residual_too_large"):
                        n_by_reason["residual"] += 1

        valid_count = len(valid_sorted)
        self.laminae = laminae
        self._lamina_settings = {
            "tolerance_px": int(tolerance_px),
            "min_support": int(min_support),
            "n_main_scan_lines": n_main,
            "n_validation_lines": n_val,
            "n_scan_lines": int(total_lines),
            "n_clusters": len(laminae),
            "n_valid_laminae": int(valid_count),
            "candidate_points": int(len(all_points)),
            "max_dip_angle_deg": float(max_dip_angle_deg),
            "max_residual_px": float(max_residual_px),
            "slope_hint": slope_hint,
            "slope_hint_dip_deg": math.degrees(math.atan(abs(slope_hint))) if slope_hint else 0.0,
            "slope_hint_source": slope_hint_source,
            "vote_slope": float(vote_slope),
            "vote_score": int(vote_score),
            "rejected_by_support": n_by_reason["support"],
            "rejected_by_dip": n_by_reason["dip"],
            "rejected_by_residual": n_by_reason["residual"],
        }

        if hasattr(self, "_detection_diagnostics") and isinstance(self._detection_diagnostics, dict):
            self._detection_diagnostics["lamina_clustering"] = dict(self._lamina_settings)

        return laminae

    def _apply_user_lamina_angle(self):
        """Force every lamina onto the user-specified direction.

        Used in single-image mode when the core is badly fractured and the
        automatic slope estimate is unreliable. The user supplies the dip angle
        (deviation from vertical, signed); each lamina's fit slope is set to that
        value and its drawing endpoints are recomputed so they pivot around the
        lamina's mean point. Validity (support / residual) from the clustering
        stage is preserved; only the orientation is overridden.
        """
        user_ang = getattr(self, "user_dip_angle_deg", None)
        if user_ang is None or not getattr(self, "laminae", None):
            return
        user_slope = math.tan(math.radians(float(user_ang)))
        dip = abs(float(user_ang))
        h_img = int(self.height) if getattr(self, "height", None) else 0
        y_bot = float(max(0, h_img - 1))
        n = 0
        for la in self.laminae:
            pts = la.get("member_points_xy") or []
            if pts:
                mean_y = float(np.mean([p[1] for p in pts]))
            else:
                mean_y = y_bot / 2.0
            intercept = float(la.get("x_mean", 0.0)) - user_slope * mean_y
            la["fit_slope"] = user_slope
            la["fit_intercept"] = intercept
            la["dip_angle_deg"] = dip
            la["x_at_top"] = float(intercept)
            la["x_at_bottom"] = float(user_slope * y_bot + intercept)
            la["user_angle_applied"] = True
            n += 1
        ls = getattr(self, "_lamina_settings", None)
        if isinstance(ls, dict):
            ls["user_dip_angle_deg"] = float(user_ang)
            ls["user_slope"] = float(user_slope)
            ls["slope_hint_source"] = "user_input"
        print(f"  [manual-angle] applied user dip {user_ang:+.1f} deg to {n} lamina(e)")

    def _apply_batch_lamina_constraints(self):
        """Enforce post-alignment lamina orientation rules after clustering.

        Used by BOTH single-image and batch flows:
          - Approach 1 (batch only): sub-folder ``batch_group_slope_hint`` was
            already applied inside ``_cluster_to_laminae`` for linking.
          - Approach 2 (always when ``aligned=True`` and the caller has not
            opted out via ``allow_tilted_after_align``): reject laminae whose
            fitted dip exceeds the post-align cap (default 7 deg), then snap
            surviving laminae to a vertical line at their mean *x* so the
            connection drawing and spacing statistics stay consistent.
        """
        laminae = getattr(self, "laminae", None) or []
        if not laminae:
            return

        allow_tilted = bool(getattr(self, "allow_tilted_after_align", False))
        force_vertical = (
            not allow_tilted
            and bool(getattr(self, "batch_force_vertical_after_align", True))
            and bool(getattr(self, "aligned", False))
        )
        batch_max_dip = float(getattr(self, "batch_max_dip_after_align_deg", 7.0))
        n_rejected_dip = 0
        n_snapped = 0

        for la in laminae:
            if not la.get("is_valid"):
                continue

            dip_deg = float(la.get("dip_angle_deg", 0.0) or 0.0)
            if force_vertical and dip_deg > batch_max_dip:
                la["is_valid"] = False
                reasons = list(la.get("rejection_reasons", []))
                reasons.append(
                    f"batch_dip_too_large({dip_deg:.1f}deg>{batch_max_dip:.1f}deg)"
                )
                la["rejection_reasons"] = reasons
                n_rejected_dip += 1
                continue

            if force_vertical:
                x_ref = float(la.get("x_mean", la.get("x_median", 0.0)))
                la["fit_slope"] = 0.0
                la["fit_intercept"] = x_ref
                la["dip_angle_deg"] = 0.0
                la["x_at_top"] = x_ref
                la["x_at_bottom"] = x_ref
                n_snapped += 1

        valid_sorted = sorted([la for la in laminae if la.get("is_valid")],
                              key=lambda la: la["x_mean"])
        for i, la in enumerate(valid_sorted):
            if i + 1 < len(valid_sorted):
                la["spacing_to_next_px"] = float(
                    valid_sorted[i + 1]["x_mean"] - la["x_mean"]
                )
            else:
                la["spacing_to_next_px"] = None
        for la in laminae:
            if not la.get("is_valid"):
                la["spacing_to_next_px"] = None

        ls = getattr(self, "_lamina_settings", None) or {}
        ls["n_valid_laminae"] = len(valid_sorted)
        ls["batch_force_vertical"] = force_vertical
        ls["batch_max_dip_after_align_deg"] = batch_max_dip
        ls["batch_rejected_by_dip"] = n_rejected_dip
        ls["batch_snapped_vertical"] = n_snapped
        ls["batch_group_slope_hint"] = getattr(self, "batch_group_slope_hint", None)
        self._lamina_settings = ls

        if n_rejected_dip or n_snapped:
            print(f"  [batch lamina policy] rejected_by_dip={n_rejected_dip}, "
                  f"snapped_vertical={n_snapped}, valid={len(valid_sorted)}")

    def _validate_regions_between_scan_lines(self, sorted_scan_lines, min_validation_lines, min_layer_width, max_offset=30):
        """Use neighbouring scan lines to cross-validate and emit extra validation lines."""
        from scipy.signal import find_peaks

        # Generate validation lines: midpoints between neighbours + edge bookends
        validation_lines = []
        if len(sorted_scan_lines) >= 2:
            for i in range(len(sorted_scan_lines) - 1):
                mid_y = (sorted_scan_lines[i] + sorted_scan_lines[i+1]) // 2
                validation_lines.append(mid_y)

        if sorted_scan_lines:
            first_y, last_y = sorted_scan_lines[0], sorted_scan_lines[-1]
            max_dist = 100
            if first_y > 50:
                validation_lines.append(first_y - min(first_y // 2, max_dist))
            remaining = self.height - last_y
            if remaining > 50:
                below_y = last_y + min(remaining // 2, max_dist)
                if below_y < self.height:
                    validation_lines.append(below_y)

        # Save validation-line positions for paper export
        self.validation_lines = sorted(set(validation_lines))

        h, w = self.processed.shape
        edge_filter_pixels = max(10, int(w * 0.05))

        # Adaptive sensitivity (mirrors the main detector: conservative)
        if self.gray is not None:
            val_contrast = np.std(self.gray.astype(np.float64))
            val_brightness = np.mean(self.gray.astype(np.float64))
        else:
            val_contrast = np.std(self.processed.astype(np.float64))
            val_brightness = np.mean(self.processed.astype(np.float64))
        is_dark_val = val_brightness < 90
        # Validation-line thresholds match the main detector (equally permissive)
        if is_dark_val and val_contrast < 25:
            val_alpha_step, val_alpha_grad = 0.4, 0.7
        elif is_dark_val:
            val_alpha_step, val_alpha_grad = 0.5, 0.8
        elif val_contrast < 25:
            val_alpha_step, val_alpha_grad = 0.5, 0.9
        elif val_contrast < 40:
            val_alpha_step, val_alpha_grad = 0.7, 1.1
        else:
            val_alpha_step, val_alpha_grad = 0.9, 1.3

        # Absolute magnitude threshold (shared with the main detector)
        min_delta_gray_val = getattr(self, '_min_delta_gray', max(4.0, 0.12 * val_contrast))

        # Validation lines also use multi-scale step + log-gradient detection
        validation_results = []
        for y in validation_lines:
            if y >= h:
                continue

            gray_row = self.processed[y, :].astype(np.float64)

            # Multi-scale step detection
            row_smooth = gaussian_filter1d(gray_row, sigma=1.5)
            step_peaks_all = set()
            for sw in [max(5, min_layer_width // 2), max(8, min_layer_width), max(15, min_layer_width * 2)]:
                lr_k = np.zeros(sw * 2)
                lr_k[:sw] = -1.0 / sw
                lr_k[sw:] = 1.0 / sw
                ss = np.abs(np.convolve(row_smooth, lr_k, mode='same'))
                vr = ss[edge_filter_pixels:w - edge_filter_pixels]
                st = (np.mean(vr) + val_alpha_step * np.std(vr)) if len(vr) > 0 and np.std(vr) > 0 else 0
                sp, _ = find_peaks(ss, height=st, distance=max(3, min_layer_width // 2))
                step_peaks_all.update(sp.tolist())
            step_peaks = np.array(sorted(step_peaks_all))

            # Log-gradient enhancement
            grad_raw = np.abs(np.diff(gray_row))
            grad_log = np.log1p(grad_raw)
            grad_smooth = gaussian_filter1d(grad_log, sigma=2)
            grad_std = np.std(grad_smooth)
            if grad_std > 0:
                peak_height = np.mean(grad_smooth) + val_alpha_grad * grad_std
                grad_peaks, _ = find_peaks(grad_smooth, height=peak_height,
                                           distance=max(3, min_layer_width // 2))
            else:
                grad_peaks = np.array([])

            combined = sorted(set(step_peaks.tolist()) | set(grad_peaks.tolist()))

            filtered = []
            last_pt = -min_layer_width
            for pt in combined:
                if pt < edge_filter_pixels or pt >= (w - edge_filter_pixels):
                    continue
                if pt - last_pt >= min_layer_width * 0.7:
                    filtered.append(pt)
                    last_pt = pt

            # Magnitude filter (same as main detector)
            gray_src_v = self.gray if self.gray is not None else self.processed
            mag_kept = []
            for pt in filtered:
                hw = max(5, min_layer_width)
                x_lo = max(0, pt - hw)
                x_hi = min(w, pt + hw + 1)
                if pt - x_lo < 2 or x_hi - pt < 2:
                    continue
                left_seg = gray_src_v[y, x_lo:pt].astype(np.float64)
                right_seg = gray_src_v[y, pt:x_hi].astype(np.float64)
                if abs(float(np.mean(right_seg)) - float(np.mean(left_seg))) >= min_delta_gray_val:
                    mag_kept.append(pt)

            validation_results.append({"y": y, "points": mag_kept})

        # Save validation-line results for the cross-line 2D fit
        self._validation_results = validation_results

        # Visualization
        os.makedirs(self.output_dir, exist_ok=True)
        if self.processed is not None and self.save_diagnostics:
            vis = cv2.cvtColor(self.processed.copy(), cv2.COLOR_GRAY2BGR)
            for y in sorted_scan_lines:
                cv2.line(vis, (0, y), (w - 1, y), (0, 255, 0), 1)
            for vr in validation_results:
                cv2.line(vis, (0, vr["y"]), (w - 1, vr["y"]), (255, 0, 0), 1)
                for pt in vr["points"]:
                    cv2.circle(vis, (pt, vr["y"]), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.output_dir, "validation_lines.png"), vis)

        # We no longer apply a hard "must appear at the same x on some validation
        # line" filter. The [slope vote + 2D fit] before clustering performs that
        # adjudication properly. Here we just label all main-detection points as
        # ``validated_points`` (= ``points``), keeping the GUI display without
        # dropping anything.
        total_validated_points = 0
        for layer_data in self.layers:
            layer_data["validated_points"] = list(layer_data["points"])
            layer_data["point_count"] = len(layer_data["points"])
            total_validated_points += len(layer_data["validated_points"])

        # Generate the validated-grid image (only when diagnostics are enabled, to save I/O)
        if self.save_diagnostics:
            grid_img = np.copy(self.image)
            for layer_data in self.layers:
                y = layer_data["y"]
                cv2.line(grid_img, (0, y), (w-1, y), (0, 255, 0), 1)

                if "validated_points" in layer_data:
                    for pt in layer_data["validated_points"]:
                        cv2.line(grid_img, (pt, max(0, y - 6)), (pt, min(h - 1, y + 6)), (0, 0, 255), 2)
                        cv2.circle(grid_img, (pt, y), 3, (255, 0, 0), -1)

            # Draw validation lines
            for val_y in validation_lines:
                if val_y < h:
                    cv2.line(grid_img, (0, val_y), (w-1, val_y), (255, 0, 0), 1)

            cv2.imwrite(os.path.join(self.output_dir, "validated_grid.png"), grid_img)

        if total_validated_points == 0:
            return False

        return True
    def _generate_demo_hough_lines_image(self, image):
        """Generate a demo image showing Hough-transform line detection."""
        try:
            # Apply edge detection first
            edges = cv2.Canny(image, 50, 150, apertureSize=3)

            # Detect lines on the edge map
            h, w = image.shape
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=w//4, maxLineGap=20)


        except Exception as e:
            print(f"Error while generating Hough-transform demo image: {str(e)}")
