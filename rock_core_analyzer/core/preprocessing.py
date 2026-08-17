#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image preprocessing."""

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


class PreprocessingMixin:
    def preprocess_image(self, blur_size=5, clahe_clip=2.0, clahe_grid=(8, 8),
                         brightness=0, contrast=1.0, gamma=1.0,
                         disable_clahe=False):
        """Preprocess the image: brightness/contrast/gamma -> grayscale -> denoise -> CLAHE -> gradient enhancement.

        Args:
            blur_size: Gaussian-blur kernel size.
            clahe_clip: CLAHE clip limit.
            clahe_grid: CLAHE tile-grid size.
            brightness: Brightness offset (-100 to +100); use positive values for dark images.
            contrast: Contrast multiplier (0.1 to 3.0); increase for dark images.
            gamma: Gamma correction (0.1 to 3.0); <1 brightens, >1 darkens.
            disable_clahe: Skip the CLAHE step (used only for ablation studies).

        Returns:
            The preprocessed image.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Keep an "absolutely pristine" original (before alignment and enhancement)
        # for the left half of ``*_comparison.png`` plots. Without this, ``self.image``
        # is already sheared after alignment and both panels would look sheared,
        # giving the visual impression of "shearing applied twice".
        self.image_original = self.image.copy() if self.image is not None else None

        # Step 1: brightness/contrast/gamma applied to the raw color image (before grayscale)
        src = self.image.copy()
        need_enhance = (brightness != 0) or (abs(contrast - 1.0) > 0.01) or (abs(gamma - 1.0) > 0.01)

        if need_enhance:
            src = src.astype(np.float32)
            src = src * contrast + brightness
            src = np.clip(src, 0, 255).astype(np.uint8)

            if abs(gamma - 1.0) > 0.01:
                lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
                src = cv2.LUT(src, lut)

        if len(src.shape) > 2:
            gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = src.copy()

        # ============================================================
        # Adaptive non-linear enhancement (gamma + sigmoid + post-CLAHE unsharp).
        # Applies to *every* image with strength scaled to the image's
        # darkness/contrast. This is the most critical step for revealing lamina
        # boundaries in dark mudstones/shales, so even moderate-contrast images
        # receive a noticeable non-linear amplification rather than a no-op.
        # ============================================================
        img_mean = float(np.mean(gray_img))
        img_std = float(np.std(gray_img))
        img_p1 = float(np.percentile(gray_img, 1))
        img_p99 = float(np.percentile(gray_img, 99))
        dynamic_range = img_p99 - img_p1

        # Strength tier:
        #   3 = very dark (mean<55):       extremely aggressive amplification
        #   2 = dark (55<=mean<90):        strong amplification (typical dark shale)
        #   1 = mid / low contrast:        moderate amplification (still visible)
        #   0 = bright/high contrast:      gentle "polish" stretch
        if img_mean < 55:
            strength_tier = 3
        elif img_mean < 90:
            strength_tier = 2
        elif img_mean < 130 or img_std < 35 or dynamic_range < 130:
            strength_tier = 1
        else:
            strength_tier = 0

        # Tier-specific parameters
        tier_gamma = {3: 0.45, 2: 0.60, 1: 0.80, 0: 1.00}[strength_tier]
        tier_steepness = {3: 14.0, 2: 12.0, 1: 10.0, 0: 7.0}[strength_tier]
        tier_quantiles = {3: (0.03, 0.97),
                          2: (0.04, 0.96),
                          1: (0.05, 0.95),
                          0: (0.06, 0.94)}[strength_tier]
        # Double sigmoid: for very dark samples, apply a second gentler pass
        # after the first one to push the centre band further apart.
        tier_double_sigmoid = strength_tier >= 3
        # Post-CLAHE unsharp mask: sharpens lamina edges; only useful when
        # the input has been substantially amplified.
        tier_unsharp = strength_tier >= 2

        is_dark_core = strength_tier >= 2
        dark_mode_applied = is_dark_core
        sigmoid_meta = {}
        sigmoid_meta_2 = {}

        gray_for_process = gray_img
        # Step A: gamma lift on the colour-converted grayscale to raise shadows
        if abs(tier_gamma - 1.0) > 0.01:
            lut = np.array([(i / 255.0) ** tier_gamma * 255 for i in range(256)],
                           dtype=np.uint8)
            gray_for_process = cv2.LUT(gray_for_process, lut)
        # Step B: percentile-pinned linear pre-stretch so the dense band fully
        # spans [0, 255] before the sigmoid. This prevents the sigmoid from
        # ``wasting'' amplification on cold/hot tails for severely under-exposed
        # images and gives the central sigmoid slope a fully-stretched canvas to
        # work on -- the *visible* contrast jump is much larger this way.
        ps_lo = float(np.percentile(gray_for_process, tier_quantiles[0] * 100.0))
        ps_hi = float(np.percentile(gray_for_process, tier_quantiles[1] * 100.0))
        if ps_hi - ps_lo > 5:
            pre_lut = np.clip(
                (np.arange(256, dtype=np.float32) - ps_lo) / (ps_hi - ps_lo),
                0.0, 1.0,
            )
            pre_lut = (pre_lut * 255.0).astype(np.uint8)
            gray_for_process = cv2.LUT(gray_for_process, pre_lut)
            prestretch_applied = True
        else:
            prestretch_applied = False
        # Step C: sigmoid non-linear stretch (centre-band amplification)
        gray_for_process, sigmoid_meta = self._sigmoid_dense_band_stretch(
            gray_for_process,
            dense_quantiles=tier_quantiles,
            steepness=tier_steepness,
        )
        # Step D: optional second-pass sigmoid for very dark samples
        if tier_double_sigmoid and sigmoid_meta.get("applied"):
            gray_for_process, sigmoid_meta_2 = self._sigmoid_dense_band_stretch(
                gray_for_process,
                dense_quantiles=(0.10, 0.90),
                steepness=6.0,
            )
        # Keep a copy of the post-non-linear (gamma + sigmoid), pre-CLAHE grayscale
        # so ``export_results`` can write it as a standalone product for paper
        # references and manual verification of the enhancement step.
        self.gray_nonlinear_enhanced = gray_for_process.copy()
        self.gray_nonlinear_enhanced_pristine = gray_for_process.copy()

        post_mean = float(np.mean(gray_for_process))
        post_std = float(np.std(gray_for_process))
        sig1_msg = ""
        if sigmoid_meta.get("applied"):
            sig1_msg = (f"sigmoid1 band[{sigmoid_meta['lo']:.0f},{sigmoid_meta['hi']:.0f}] "
                        f"k={sigmoid_meta['steepness']:.0f} "
                        f"x{sigmoid_meta['center_amplification']:.1f}")
        else:
            sig1_msg = f"sigmoid1 skipped ({sigmoid_meta.get('reason','?')})"
        sig2_msg = ""
        if sigmoid_meta_2.get("applied"):
            sig2_msg = (f" sigmoid2 k={sigmoid_meta_2['steepness']:.0f} "
                        f"x{sigmoid_meta_2['center_amplification']:.1f}")
        print(f"  Non-linear enhance: tier={strength_tier} "
              f"mean {img_mean:.1f}->{post_mean:.1f}, "
              f"std {img_std:.1f}->{post_std:.1f}, "
              f"gamma={tier_gamma:.2f}, {sig1_msg}{sig2_msg}"
              f"{' (stacked with user enhancement)' if need_enhance else ''}")

        # Bilateral filter: weaker parameters for high-tier (dark) samples to
        # preserve thin laminae; thin laminae (<10 grey-level contrast) are
        # easily wiped out by strong bilateral filtering on dark samples.
        if strength_tier >= 2:
            bil_sigma_color = 25
            bil_sigma_space = 25
        elif strength_tier == 1:
            bil_sigma_color = 45
            bil_sigma_space = 45
        else:
            bil_sigma_color = 75
            bil_sigma_space = 75
        denoised = cv2.bilateralFilter(
            gray_for_process, d=9,
            sigmaColor=bil_sigma_color, sigmaSpace=bil_sigma_space,
        )
        blurred = cv2.GaussianBlur(denoised, (blur_size, blur_size), 0)

        if disable_clahe:
            effective_clip = 0.0
            effective_grid = clahe_grid
            enhanced = blurred
        else:
            # CLAHE strength scales with the enhancement tier so the local
            # contrast amplification stays consistent with the global sigmoid.
            if strength_tier == 3:
                effective_clip = max(clahe_clip, 5.0)
                effective_grid = (max(4, clahe_grid[0] // 2), max(4, clahe_grid[1] // 2))
            elif strength_tier == 2:
                effective_clip = max(clahe_clip, 4.0)
                effective_grid = (max(4, clahe_grid[0] // 2), max(4, clahe_grid[1] // 2))
            elif strength_tier == 1:
                effective_clip = max(clahe_clip, 3.0)
                effective_grid = clahe_grid
            else:
                effective_clip = clahe_clip
                effective_grid = clahe_grid
            clahe = cv2.createCLAHE(clipLimit=effective_clip, tileGridSize=effective_grid)
            enhanced = clahe.apply(blurred)

        # Step E (post-CLAHE): unsharp mask to crisp lamina edges. Done only on
        # already heavily amplified images to avoid amplifying sensor noise on
        # bright/clean cores.
        unsharp_applied = False
        if tier_unsharp and not disable_clahe:
            unsharp_amount = 0.8 if strength_tier >= 3 else 0.6
            unsharp_sigma = 2.5 if strength_tier >= 3 else 2.0
            blurred_for_unsharp = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=unsharp_sigma)
            unsharped = cv2.addWeighted(
                enhanced, 1.0 + unsharp_amount,
                blurred_for_unsharp, -unsharp_amount, 0,
            )
            enhanced = np.clip(unsharped, 0, 255).astype(np.uint8)
            unsharp_applied = True

        # Black-hat enhancement for high-tier dark samples: highlights dark thin
        # laminae within a bright matrix (organic-rich layers).
        # black-hat = closing(I) - I; responds strongly to small dark troughs and
        # complements the cross-lamina gradient.
        # Geometry: the core is laid horizontally, so laminae run VERTICALLY in the
        # image. A vertical dark lamina is narrow along X, therefore the structuring
        # element must be HORIZONTAL (width N, height 1) to close it and isolate it.
        blackhat = None
        if strength_tier >= 2 and not disable_clahe:
            bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, blur_size * 3), 1))
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, bh_kernel)
            if blackhat.max() > 0:
                bh_scaled = np.uint8(np.clip(blackhat.astype(np.float32) *
                                              (60.0 / max(1.0, float(blackhat.max()))), 0, 255))
            else:
                bh_scaled = blackhat
            enhanced = cv2.addWeighted(enhanced, 0.85, bh_scaled, 0.15, 0)

        # Cross-lamina gradient: laminae are vertical, so their boundaries are
        # vertical edges whose intensity gradient lies along X. Use Sobel-X
        # (dx=1, dy=0); Sobel-Y would respond to horizontal edges and suppress
        # the vertical laminae we want to keep.
        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_abs = np.uint8(np.clip(np.abs(sobel_x), 0, 255))

        alpha = 0.7
        enhanced_with_grad = cv2.addWeighted(enhanced, alpha, sobel_x_abs, 1 - alpha, 0)

        self.processed = enhanced_with_grad
        self.gray = gray_img
        self.enhanced_no_grad = enhanced

        # Preprocessing metadata (used by paper export)
        self._preprocess_meta = {
            "image_mean": img_mean,
            "image_std": img_std,
            "image_p1": img_p1,
            "image_p99": img_p99,
            "dynamic_range": dynamic_range,
            "is_dark_core": is_dark_core,
            "dark_mode_applied": dark_mode_applied,
            "enhancement_tier": strength_tier,
            "tier_auto_gamma": tier_gamma,
            "tier_steepness": tier_steepness,
            "tier_quantiles": list(tier_quantiles),
            "tier_double_sigmoid": bool(tier_double_sigmoid),
            "prestretch_applied": bool(prestretch_applied),
            "prestretch_lo": ps_lo,
            "prestretch_hi": ps_hi,
            "unsharp_applied": bool(unsharp_applied),
            "blur_size": blur_size,
            "bilateral_sigma": bil_sigma_color,
            "clahe_clip_input": clahe_clip,
            "clahe_clip_effective": effective_clip,
            "clahe_grid_input": list(clahe_grid),
            "clahe_grid_effective": list(effective_grid),
            "clahe_disabled": bool(disable_clahe),
            "blackhat_applied": blackhat is not None,
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma,
            "user_enhance_applied": bool(need_enhance),
            "post_nonlinear_mean": post_mean,
            "post_nonlinear_std": post_std,
            "sigmoid_stretch": sigmoid_meta,
            "sigmoid_stretch_secondary": sigmoid_meta_2,
        }

        # Save intermediate results for each step (used by paper export).
        # Note: ``detect_layers`` warps every image in ``self._preprocess_steps``
        # while aligning the core, so we also keep a pristine (un-warped) copy
        # in ``_preprocess_steps_pristine``. The 01-04 paper figures (raw / gray /
        # denoised / CLAHE) use the pristine versions; otherwise they would all
        # appear sheared and visually duplicate ``05_geometry_corrected``,
        # which would look like the shear was applied twice.
        self._preprocess_steps = {
            "gray": gray_img,
            "denoised": denoised,
            "blurred": blurred,
            "clahe": enhanced,
        }
        if blackhat is not None:
            self._preprocess_steps["blackhat"] = blackhat
        self._preprocess_steps_pristine = {
            k: (v.copy() if v is not None else None)
            for k, v in self._preprocess_steps.items()
        }
        # Also keep the pre-alignment grayscale and CLAHE-enhanced images for paper export
        self.gray_pristine = gray_img.copy() if gray_img is not None else None
        self.enhanced_no_grad_pristine = enhanced.copy() if enhanced is not None else None

        return enhanced_with_grad
    def _sigmoid_dense_band_stretch(self, gray_img, dense_quantiles=(0.02, 0.98),
                                    steepness=8.0):
        """Non-linear sigmoid stretch on the densely-populated grayscale band.

        Dark mudstone/shale grey values cluster in a narrow [p2, p98] band such as
        [20, 70]. A linear stretch maps that band to 0-255 with uniform amplification,
        while a sigmoid is steepest at the band center and shallow at both ends,
        amplifying *central* contrast by ~steepness/4 extra. This is essential for
        revealing sub-pixel lamina boundaries:

            t = clip((x - lo) / (hi - lo), 0, 1)
            s = sigmoid(steepness * (t - 0.5))            # max slope at t=0.5
            normalize to [0, 1] then multiply by 255 to obtain the LUT

        Extra center amplification = (sigmoid central slope) / (linear slope)
                                   = steepness * 0.25.
        steepness=8 -> roughly x2 at center (versus a plain linear stretch).

        Args:
            gray_img: uint8 grayscale image.
            dense_quantiles: Quantile pair that defines the "dense" band (default 2% / 98%).
            steepness: Sigmoid steepness; higher values produce more aggressive
                center stretching (recommended 6-10).

        Returns:
            ``(stretched_img, meta_dict)``. ``meta`` contains lo/hi/steepness/
            center_amplification/applied/reason.
        """
        if gray_img is None or gray_img.size == 0:
            return gray_img, {"applied": False, "reason": "empty"}
        lo = float(np.percentile(gray_img, dense_quantiles[0] * 100.0))
        hi = float(np.percentile(gray_img, dense_quantiles[1] * 100.0))
        if hi - lo < 5:
            return gray_img, {"applied": False, "reason": "range_too_narrow",
                              "lo": lo, "hi": hi, "steepness": steepness}
        xs = np.arange(256, dtype=np.float32)
        t = np.clip((xs - lo) / (hi - lo), 0.0, 1.0)
        s_raw = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))
        # Endpoint alignment: rescale so that t=0/1 maps to LUT 0 / 255 exactly
        s_min = 1.0 / (1.0 + np.exp(steepness * 0.5))
        s_max = 1.0 / (1.0 + np.exp(-steepness * 0.5))
        if s_max - s_min < 1e-9:
            return gray_img, {"applied": False, "reason": "sigmoid_flat",
                              "lo": lo, "hi": hi, "steepness": steepness}
        s_norm = (s_raw - s_min) / (s_max - s_min)
        lut = np.clip(s_norm * 255.0, 0, 255).astype(np.uint8)
        stretched = cv2.LUT(gray_img, lut)
        # Center amplification relative to a *plain* linear stretch to [0,255]
        center_amp = float(steepness * 0.25 * (s_max - s_min) /
                           max(1e-9, (s_max - s_min)))
        return stretched, {
            "applied": True,
            "lo": lo, "hi": hi,
            "steepness": float(steepness),
            "center_amplification": float(steepness * 0.25),
            "dense_quantiles": list(dense_quantiles),
        }
