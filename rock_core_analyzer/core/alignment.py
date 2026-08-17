#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core alignment."""

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


class AlignmentMixin:
    def _align_core(self, image, angle=0.0, slope_hint=0.0, slope_hint_confidence=0.0):
        """Shear-flatten the rock-core image so that near-vertical laminae become truly vertical.

        Strategy stack (highest priority first):
          1) User-specified ``angle`` -> apply the manual shear directly.
          2) Upstream high-confidence HoughLinesP slope hint
             (``slope_hint_confidence >= 0.45``) -> use the hint as the initial guess
             and refine via cross-correlation within a small window around it.
          3) Otherwise fall back to a wide-range cross-correlation search
             (``+/- tan(45 deg)``).

        Key improvement: the search range is widened to ``+/- tan(45 deg) ~= 1.0`` so we
        can handle real cores with ~28 deg or larger tilts; the ``slope_hint`` then
        narrows the window for both speed and robustness.
        """
        h, w = image.shape[:2]

        # Manual mode (``angle`` is the horizontal shear factor)
        if angle != 0.0:
            M = np.float32([[1.0, float(angle), 0.0], [0.0, 1.0, 0.0]])
            sheared = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            self.alignment_angle = float(angle)
            self.use_shear = True
            self.shear_axis = "x"
            print(f"  [align] manual horizontal shear factor sx={angle:.4f}")
            return sheared

        # Prepare candidate detection sources
        candidate_sources = []
        if self.gray is not None and self.gray.shape[:2] == (h, w):
            candidate_sources.append(("gray", self.gray.astype(np.float64)))
        if getattr(self, 'enhanced_no_grad', None) is not None and self.enhanced_no_grad.shape[:2] == (h, w):
            candidate_sources.append(("enhanced_no_grad", self.enhanced_no_grad.astype(np.float64)))
        if not candidate_sources:
            base = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            candidate_sources.append(("image (fallback)", base.astype(np.float64)))

        # Decide on the search range: default covers +/- 45 deg (tan(45 deg) = 1.0), making
        # sure large tilts such as 28 deg are still hit.
        max_shear_full = 1.0
        # If a high-confidence Hough hint exists, refine within a window around it
        use_hint = (
            abs(float(slope_hint)) >= 0.05 and float(slope_hint_confidence) >= 0.45
        )
        hint_axis = "x"  # Hough hint maps to horizontal shear by default (vertical laminae)
        if use_hint:
            hint_deg = math.degrees(math.atan(abs(float(slope_hint))))
            print(f"  [align] using Hough slope hint = {float(slope_hint):+.4f} "
                  f"(~{hint_deg:.1f} deg, conf={slope_hint_confidence:.2f}) as initial guess")

        best = None  # (src_name, axis, shear_factor, conf, n_pairs, score)
        for src_name, src_arr in candidate_sources:
            if use_hint:
                # Hint mode: refine the x-axis within hint +/- window; keep y-axis on the
                # full range so we do not miss the horizontal-lamina case.
                sx, conf_x, n_x = self._detect_shear_by_correlation(
                    src_arr, axis="x",
                    max_shear=max_shear_full, n_lines=25,
                    center_shear=float(slope_hint),
                    window=max(0.10, 0.5 * abs(float(slope_hint)) + 0.05),
                )
                sy, conf_y, n_y = self._detect_shear_by_correlation(
                    src_arr, axis="y", max_shear=max_shear_full, n_lines=25,
                )
            else:
                sx, conf_x, n_x = self._detect_shear_by_correlation(
                    src_arr, axis="x", max_shear=max_shear_full, n_lines=25,
                )
                sy, conf_y, n_y = self._detect_shear_by_correlation(
                    src_arr, axis="y", max_shear=max_shear_full, n_lines=25,
                )
            print(f"  [align] source={src_name}: horizontal sx={sx:.5f} (n={n_x}, conf={conf_x:.3f}) | "
                  f"vertical sy={sy:.5f} (n={n_y}, conf={conf_y:.3f})")

            score_x = conf_x * np.sqrt(max(0, n_x))
            score_y = conf_y * np.sqrt(max(0, n_y))

            cand = []
            if n_x >= 3 and conf_x >= 0.10:
                cand.append(("x", sx, conf_x, n_x, score_x))
            if n_y >= 3 and conf_y >= 0.10:
                cand.append(("y", sy, conf_y, n_y, score_y))

            for axis, sf, cf, n, sc in cand:
                if best is None or sc > best[5]:
                    best = (src_name, axis, sf, cf, n, sc)

        # If cross-correlation fails but the Hough hint is confident, apply the hint
        # directly (prevents large tilts like 28 deg from being silently skipped).
        if best is None and use_hint:
            print(f"  [align] all cross-correlation candidates failed; "
                  f"Hough hint is confident -> apply hint shear directly")
            best = ("hough_hint", hint_axis, float(slope_hint), float(slope_hint_confidence), 0, float(slope_hint_confidence))

        if best is None:
            print(f"  [align] every source/axis correlation was too low and no "
                  f"reliable Hough hint is available, skipping; try toggling "
                  f"contrast/brightness enhancement or specifying ``alignment_angle`` manually")
            return image

        chosen_src, chosen, shear_factor, conf, n_pairs, score = best
        direction_desc = ("longitudinal laminae (core laid horizontally)"
                          if chosen == "x" else
                          "transverse laminae (core stood vertically)")
        print(f"  [align] chosen: source={chosen_src}, axis={chosen}, shear={shear_factor:.5f}, "
              f"conf={conf:.3f}, valid line pairs={n_pairs}")

        # Outlier filter: a shear factor beyond tan(50 deg) ~= 1.19 is untrustworthy
        if abs(shear_factor) > 1.2:
            print(f"  [align] shear factor out of range ({shear_factor:.4f}); skipping")
            return image
        if abs(shear_factor) < 0.003:
            print(f"  [align] dominant={direction_desc}, shear factor={shear_factor:.5f} "
                  f"too small; no transformation needed")
            self.alignment_angle = 0.0
            self.use_shear = False
            return image

        # Build the shear matrix.
        # Key point: the detected ``shear_factor`` is the laminae's own slope dx/dy
        # (positive = tilted to the lower-right). To straighten laminae we must apply
        # the *opposite* shear: x' = x - s*y. Hence we use ``-shear_factor`` in the
        # matrix; otherwise the tilt only gets worse.
        apply_shear = -float(shear_factor)
        if chosen == "x":
            M = np.float32([[1.0, apply_shear, 0.0], [0.0, 1.0, 0.0]])
        else:
            M = np.float32([[1.0, 0.0, 0.0], [apply_shear, 1.0, 0.0]])

        # Use the image mean as the borderValue. ``BORDER_REPLICATE`` would copy edge
        # pixels into wide flat bands, which the multi-scale step detector tends to
        # mistake for laminae. A constant grey fill lets ``detect_layers`` automatically
        # skip the information-free borders without introducing spurious step edges.
        try:
            border_val = float(np.mean(image))
        except Exception:
            border_val = 128.0
        sheared = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=border_val)
        eq_angle = float(np.degrees(np.arctan(abs(shear_factor))))
        print(f"  [align] dominant={direction_desc}, detected lamina slope={shear_factor:+.4f} "
              f"(~{eq_angle:.2f} deg); applying *opposite* shear={apply_shear:+.4f} "
              f"to make laminae vertical, shear axis={chosen}")

        # Record the shear used by the first alignment pass; the downstream stages
        # apply the same warp to ``gray``/``image``/etc. Residual-tilt refinement
        # happens inside ``detect_layers`` via ``vote_slope`` (Hough only sees strong
        # long edges after alignment, not thin laminae, so it cannot do refinement).
        self.alignment_angle = float(apply_shear)
        self.use_shear = True
        self.shear_axis = chosen
        return sheared
    def _detect_shear_by_correlation(self, src, axis="x", max_shear=1.0, n_lines=21,
                                     center_shear=0.0, window=None):
        """Detect shear amount via multi-scan-line cross-correlation.

        ``axis="x"``: horizontal scan lines detect horizontal shear (straightens slanted
                  vertical lines). For different y values, take ``row = src[y, :]`` and
                  measure the horizontal offset ``dx`` between them. ``dx = sx * dy`` ->
                  fit ``sx``.
        ``axis="y"``: vertical scan columns detect vertical shear (straightens slanted
                  horizontal lines); applies the same procedure to ``src.T``.

        Args:
            src: Grayscale image (float).
            axis: ``"x"`` or ``"y"``.
            max_shear: Upper bound for full-range search (``|sx| <= max_shear``).
            n_lines: Number of sampled scan lines.
            center_shear: Search center (from a Hough hint); defaults to 0.
            window: Half-window around ``center_shear``; ``None`` falls back to full range.

        Returns:
            ``(shear_factor, confidence, n_pairs)``.
        """
        if axis == "y":
            src = src.T

        H, W = src.shape[:2]
        margin = max(5, H // 12)
        if H - 2 * margin < 40 or W < 80:
            return 0.0, 0.0, 0

        ys = np.linspace(margin, H - margin - 1, n_lines).astype(int)
        band = max(3, min(10, H // 60))

        hp_kernel = max(11, W // 20)
        if hp_kernel % 2 == 0:
            hp_kernel += 1

        rows = []
        for y in ys:
            y0 = max(0, y - band)
            y1 = min(H, y + band + 1)
            row = np.mean(src[y0:y1, :], axis=0).astype(np.float32)
            row_lp = cv2.GaussianBlur(row.reshape(1, -1), (1, hp_kernel), 0).flatten()
            row_hp = row - row_lp
            std = float(np.std(row_hp))
            if std > 1e-6:
                row_hp = row_hp / std
            rows.append(row_hp.astype(np.float64))

        ref_idx = len(rows) // 2
        ref_y = ys[ref_idx]
        ref_row = rows[ref_idx]
        ref_norm = float(np.linalg.norm(ref_row))
        if ref_norm < 1e-3:
            return 0.0, 0.0, 0

        # Edge cropping must accommodate the search range: a shear of 1.0 with
        # |dy| = H/2 can produce a horizontal offset of H/2 pixels, so ``edge_cut``
        # must cover the worst-case shift or the template window slides off-screen
        # and the cross-correlation finds no peak.
        max_dy = int(np.max(np.abs(ys - ref_y))) if len(ys) > 1 else 0
        worst_shift = int(max(max_dy * max_shear, max_dy * (abs(center_shear) + (window or 0))) + 4)
        edge_cut = max(20, W // 20, worst_shift + 5)
        if W - 2 * edge_cut < 40:
            # Too narrow: ``max_shear`` is too large relative to the image width;
            # back off and reserve W/4 from each side.
            edge_cut = max(20, W // 4)
        ref_template = ref_row[edge_cut:W - edge_cut]
        if len(ref_template) < 20:
            return 0.0, 0.0, 0
        ref_t_norm = float(np.linalg.norm(ref_template))
        if ref_t_norm < 1e-3:
            return 0.0, 0.0, 0
        ref_template = ref_template / ref_t_norm

        shifts = []
        all_corrs = []
        for i, row in enumerate(rows):
            if i == ref_idx:
                continue
            dy = int(ys[i] - ref_y)

            # Search range: if a ``window`` is provided, scan ``center_shear +/- window``
            if window is not None and window > 0:
                lo_shear = center_shear - window
                hi_shear = center_shear + window
                shift_lo = int(min(dy * lo_shear, dy * hi_shear)) - 3
                shift_hi = int(max(dy * lo_shear, dy * hi_shear)) + 3
                shift_lo = max(shift_lo, -int(abs(dy) * max_shear) - 3)
                shift_hi = min(shift_hi, int(abs(dy) * max_shear) + 3)
            else:
                shift_lo = -max(5, int(abs(dy) * max_shear) + 2)
                shift_hi = max(5, int(abs(dy) * max_shear) + 2)

            best_shift = 0
            best_corr = -2.0
            for shift in range(shift_lo, shift_hi + 1):
                src_start = edge_cut + shift
                src_end = W - edge_cut + shift
                if src_start < 0 or src_end > W:
                    continue
                seg = row[src_start:src_end]
                seg_norm = float(np.linalg.norm(seg))
                if seg_norm < 1e-3:
                    continue
                corr = float(np.dot(ref_template, seg) / seg_norm)
                if corr > best_corr:
                    best_corr = corr
                    best_shift = shift
            all_corrs.append(best_corr)

            if best_corr > 0.10:
                shifts.append((dy, best_shift, best_corr))

        if len(shifts) < 3:
            return 0.0, 0.0, len(shifts)

        # Weighted least-squares fit: dx = sx * dy (forced through the origin)
        dys = np.array([s[0] for s in shifts], dtype=np.float64)
        dxs = np.array([s[1] for s in shifts], dtype=np.float64)
        ws  = np.array([s[2] for s in shifts], dtype=np.float64)

        num = float(np.sum(ws * dys * dxs))
        den = float(np.sum(ws * dys * dys)) + 1e-9
        shear = num / den

        # Drop high-residual points and refit (a tiny RANSAC-style step)
        predicted = shear * dys
        residuals = np.abs(dxs - predicted)
        if len(residuals) > 0:
            mad = float(np.median(residuals))
            keep_mask = residuals <= max(2.0, mad * 3.0)
            if int(np.sum(keep_mask)) >= 5 and int(np.sum(keep_mask)) < len(shifts):
                dys2 = dys[keep_mask]
                dxs2 = dxs[keep_mask]
                ws2  = ws[keep_mask]
                num2 = float(np.sum(ws2 * dys2 * dxs2))
                den2 = float(np.sum(ws2 * dys2 * dys2)) + 1e-9
                shear = num2 / den2
                confidence = float(np.mean(ws2))
                return shear, confidence, int(np.sum(keep_mask))

        confidence = float(np.mean(ws))
        return shear, confidence, len(shifts)
