# Rock Core Lamina Detection System v2.0

A computer-vision based system for automatic detection and analysis of laminae
in rock-core scans. Supports single-image analysis, batch processing,
statistical export, and academic-paper figure generation.

## Directory Layout

```
project-root/
├── main.py                          # GUI entry point
├── rock_core_layer_detection.py     # Algorithm-module compatibility entry
├── visual_app.py                    # GUI compatibility entry
├── requirements.txt                 # Python dependencies
├── license.key                      # License file (if enabled)
└── rock_core_analyzer/              # Main package
    ├── core/                        # Core algorithms
    │   ├── detector.py              # Main detector class
    │   ├── image_io.py              # Image I/O helpers
    │   ├── preprocessing.py         # Image pre-processing
    │   ├── detection.py             # Lamina detection
    │   ├── alignment.py             # Core alignment
    │   ├── statistics.py            # Statistical analysis
    │   ├── visualization.py         # Result visualization
    │   ├── paper_export.py          # Paper-figure export
    │   └── export.py                # Result export (Excel/CSV)
    ├── batch/                       # Batch processing
    │   ├── processing.py            # Single-image / folder processing
    │   ├── merge.py                 # Result merging
    │   └── batch_viz.py             # Batch visualization
    └── gui/                         # Graphical user interface
        ├── app.py                   # Application entry + main window
        ├── ui_setup.py              # UI layout
        ├── single_image.py          # Single-image analysis
        ├── scan_lines.py            # Scan-line selection
        ├── export_ui.py             # Export + calibration UI
        ├── batch_ui.py              # Batch UI
        ├── workers.py               # Worker functions (multiprocessing)
        └── utils.py                 # Utility helpers
```

## Requirements

- Python 3.7+
- Windows 10/11 (recommended)
- 8 GB RAM or more
- 2 GB free disk space

## Installation & Running

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the GUI

```bash
python main.py
```

Or use the compatibility entry point:

```bash
python visual_app.py
```

### Command-line batch processing (algorithm module)

```bash
# Single image
python rock_core_layer_detection.py /path/to/image.jpg --output output

# Folder batch processing
python rock_core_layer_detection.py /path/to/folder --batch --output output
```

### Command-line arguments (GUI)

| Argument | Description |
|----------|-------------|
| `--image` | Analyze the given image right after start-up |
| `--batch --list --base-dir` | Batch mode driven by a file list |
| `--output-dir` | Batch output directory |
| `--max-image-size` | Maximum image size in pixels |
| `--memory-limit` | Memory limit (MB) |
| `--threads` | Number of worker threads |
| `--info` | Print system information |

## Basic Workflow

1. Click "Browse" and load a rock-core image
2. Adjust detection parameters (threshold method, minimum lamina width,
   number of scan lines, etc.)
3. Optional: manually place scan lines or run scale calibration
4. Click "Start analysis" or press F5
5. View the results in the "Detection results" / "Statistics" tabs
6. Click "Save results" to export Excel and image outputs

## Batch Processing (with sub-folder support)

In the Batch tab, tick **"Include sub-folders"** on the right of the
"Image file format" row to recursively scan the selected folder and all of
its sub-directories:

- **Unticked** (default): only images directly inside the selected folder
  are processed; output layout is identical to earlier versions.
- **Ticked**:
  - Recursively scan the selected folder and every sub-directory.
  - Results are kept separate per sub-directory:
    `<save_path>/<subdir_name>/<image_name>/...`.
  - **Composite images and merged results are produced per sub-directory**:
    each sub-directory gets its own `combined_layer_intensity.png`,
    `layer_intensity_heatmap.png`, `layer_intensity_curve.png`, etc., under
    `<save_path>/<subdir_name>/`.
  - Images in the root directory (not inside a sub-directory) are still
    aggregated at the top level under `<save_path>/`.
  - The GUI shows the root-directory group by default (or the first
    sub-directory group if none); merged results for other sub-directories
    can be inspected directly inside each sub-directory.

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O   | Open image |
| Ctrl+S   | Save results |
| F5       | Start analysis |
| Ctrl+Q   | Quit |

## Building a Stand-alone Executable

The project ships with `build_exe.bat` in the root directory. Double-click
the script or run it on the command line to build a stand-alone executable:

```bash
build_exe.bat
```

The resulting executable lives under `dist/RockCoreAnalyzer/`. Copy the
whole folder to another Windows machine and it will run without any
additional installation.

## Programmatic API Example

```python
from rock_core_analyzer.core import RockCoreLayerDetector

detector = RockCoreLayerDetector("core_image.jpg")
detector.preprocess_image()
detector.detect_layers(threshold_method="otsu", min_layer_width=5, scan_line_count=5)
detector.calculate_statistics()
detector.export_results("output")

detector.export_paper_figures(
    "paper_out",
    include_sensitivity=True,
    include_ablation=True,
)
```


## Dip-angle Detection and Core Flattening (large-dip laminae)

The software processes tilted laminae in the following order so that the
**true dip angle is detected accurately** (on sample 23-13h, a measured
~30° dip can be detected directly as 23 tilted laminae, and after
flattening, as 30 nearly-vertical laminae). Only then does the system
decide whether candidate points truly lie on the same line:

1. **Hough dip estimation (before flattening)**: two passes of
   `HoughLinesP` on the raw grayscale image produce candidate long lines.
   The dominant slope `dx/dy` is returned together with a `confidence`
   value. Only candidates with `|dip| <= max_dip_angle_deg` (default 45°)
   are accepted, so cracks/scan artifacts cannot hijack the main
   orientation.
2. **Core flattening `_align_core`**:
   - If the detected lamina slope is `s` (with `s>0` meaning "leans right
     when descending"), the algorithm applies the **opposite-direction**
     shear `M=[[1, -s, 0], [0, 1, 0]]`, "straightening" tilted laminae to
     near-vertical. **Applying a same-sign shear would only increase the
     dip angle** -- that was a directional bug in earlier versions and has
     been fixed.
   - The cross-correlation search range is expanded to
     `+/-tan(45°) ≈ 1.0`, so it covers very large dips like ~30°. If the
     upstream Hough estimate carries `confidence >= 0.45`, a narrow window
     refinement is performed centred on that hint.
   - If all cross-correlation searches fail but the Hough hint has high
     confidence, the system **uses the hint directly**, so large dips are
     not silently skipped.
   - Edge holes are filled with `BORDER_CONSTANT(mean)` instead of
     `BORDER_REPLICATE`: the latter copies edge pixels into long flat
     strips that the multi-scale step detector mistakes for laminae.
3. **Recompute gradients after flattening**: `self.processed =
   enhanced + Sobel_Y` is calculated before flattening; after the affine
   transform, the strong Sobel_Y response moves with the pixels and lands
   on the *old* horizontal edges, misaligned with the new lamina
   orientation. After flattening completes, the program recomputes
   `Sobel_Y` on `self.enhanced_no_grad` and writes the result back to
   `self.processed`, aligning the gradient with the new lamina direction.
4. **Keep change-point detection permissive**: collect all visually
   distinguishable gray-level jumps first
   (`alpha_step ≈ 0.5-1.3`, `min_delta_gray >= max(4, 0.12·std)`);
   do not over-filter at this stage. Noise points are pruned by the
   [slope voting + 2D line fitting] stage described below.
5. **Vote-driven residual refinement**: the first-pass Hough estimate
   works on raw grayscale and is limited by scan stripes, water lines,
   fractures, and other strong edges. In practice we still see about
   2° of residual tilt on 23-13h after the first flatten, making the
   final connections look slightly tilted. After all main scan lines
   plus verification lines have been processed, the program uses
   `_vote_slope_from_points` (see next section) to vote for the residual
   slope on every candidate point and then:
   - Adds the residual `res_slope` to `self.alignment_angle`, producing a
     *composite total shear factor* `total`.
   - **Re-warps the saved pre-flatten image `_orig_image_pre_align` /
     `_orig_gray_pre_align` / `_orig_enhanced_no_grad_pre_align` exactly
     once with `total` as the shear matrix**, rather than stacking another
     warp on the already-warped image. Stacking causes the
     `BORDER_CONSTANT` gray triangle to appear on both sides, looking
     like the image was sheared twice.
   - Updates the `x` coordinates of all already-detected points with the
     same factor: `x_new = x_old - res_slope * y`.

   Up to 2 refinement iterations are run with a convergence threshold of
   about 1.4° (the 0.5° vote grid plus ~0.1° parabolic-interpolation
   jitter margin), avoiding back-and-forth oscillation around 0 caused by
   the discrete grid. Measured on 23-13h, the residual dip drops from 2°
   to <= 1.3°, the connection lines look almost perfectly vertical, and
   only a single-side gray triangle remains (no double-sided gray edges
   caused by "double shearing").

   **Per-iteration refinement cap of 7° (`MAX_REFINE_SLOPE = tan(7°)
   ≈ 0.123`)**: when scan lines are concentrated in a small region of the
   core (e.g. all placed in the upper half), narrow fractures / scratches
   / scan noise inside that region can outnumber the real laminae many
   times over, and the vote ends up biased by these secondary features,
   reporting an excessive residual (e.g. +/-0.6). Without a cap, the
   image would be sheared further by another ~30° in the noise
   direction, with the real laminae ending up tilted the *other* way --
   which is exactly the "sheared twice but looks un-sheared" symptom
   reported earlier. Residuals larger than 7° are treated as
   interference: the program keeps the first-pass Hough flattening and
   stops chasing them. Similarly, `_cluster_to_laminae` falls back to a
   `slope_hint` of 0 when it sees `|slope_hint| > 0.123` on an already
   flattened image, preventing an incorrect large slope from splitting
   the supporting points of nearly-vertical laminae into different
   clusters.

## Spacing-based Slope Voting

> **Key idea**: real laminae appear on multiple scan lines *with the same
> lateral spacing pattern* -- the spacing patterns on different scan
> lines look almost identical. Put all candidate change-points together
> and find a slope `s` such that the projection `x' = x - s*y` produces
> the most "clear, multi-supported" clusters. That `s` is the true dip.

1. **Collect all candidates**: place every candidate point from the 5
   main scan lines plus 6 verification lines into a 2D point set
   `[(x, y), ...]`.
2. **Slope voting `_vote_slope_from_points`**: scan 361 candidate slopes
   from `[-tan(45°), tan(45°)]` with a 0.25° resolution:
   - Project `x' = x - s*y`, single-link cluster with
     `tolerance = max(4, min_layer_width)`.
   - Count clusters whose support-line count is `>= min_support` and
     the total number of supporting points.
   - **Score = number of supporting clusters x 1000 + total supporting
     points** (cluster count dominates so a couple of spurious huge
     clusters cannot inflate the score).
   - A final *parabolic interpolation* lifts the peak to sub-grid
     precision, so the returned slope is no longer stuck to the 0.5°
     grid.
3. **Three-tier `slope_hint` priority**: vote (if its score is large
   enough) > post-Hough flatten > pre-Hough flatten. Measured on the
   original 28°/30° tilted images, the vote directly recovers the true
   slope at +/-0.5.
4. **Dip-aware clustering + 2D line fitting**: use `slope_hint` to
   project all candidates to `x'`, perform 1D clustering, and then fit
   `x = a + b*y` for each cluster via least squares with one round of
   outlier rejection.
5. **Triple criterion for "valid lamina"**:
   - Supporting lines >= `min_support` (default ≈ total_lines / 2).
   - Fitted dip <= `max_dip_angle_deg` (default 45°).
   - Maximum fitting residual <= `max_residual_px` (default =
     `tolerance_px`).

Clusters that fail are marked as "smudges" (`is_valid=False`) and shown
as gray crosses in `lamina_connections.png`, distinguishing them from
the green fitted lines of valid laminae.

## Non-linear Narrow-band Enhancement for Dark Mudstones / Shales

The gray levels of dark mudstones / shales are usually crammed into a
narrow band such as `[20, 70]` (96% of all pixels). Linear stretching
amplifies contrast uniformly inside the band, while the "fine color
difference in the *centre* of the band" -- the very thing that decides
whether laminae are detected -- gets no extra gain. The pre-processor
performs the following inside the `is_dark_core` branch
(`mean < 90` and `p99 < 180`):

1. **Gamma brightening**: `output = input^gamma`, with `gamma ∈ {0.45,
   0.55, 0.70}` depending on the image mean -- the darker the image, the
   smaller `gamma`, lifting the shadows more.
   > Earlier versions mistakenly wrote `(i/255)^(1/gamma)`, treating
   > `gamma=0.45` as the inverse of a display gamma, which actually
   > **darkened** mean=34 to mean=3 and quantized away the gray detail.
   > Fixed to `(i/255)^gamma`, matching the "`gamma<1` brightens"
   > comment in the code.
2. **Sigmoid narrow-band non-linear stretching**
   (`_sigmoid_dense_band_stretch`):
   - Take `[p2, p98]` as the "concentrated band" `[lo, hi]`.
   - Build a LUT: `t = clip((x-lo)/(hi-lo), 0, 1)` ->
     `s = sigmoid(8*(t-0.5))` -> renormalised to `[0, 255]` after
     endpoint alignment.
   - The sigmoid is steepest in the centre and flattest at the ends, so
     color differences in the *centre* of the narrow band get an extra
     `steepness/4 ≈ 2x` amplification (relative to a purely linear
     stretch of the same range), while the dark/bright tails are
     compressed, suppressing random noise.
   - Measured on 23-13h, the narrow band of `[61, 143]` is mapped by the
     sigmoid to a curve that covers nearly the full `[0, 255]` range,
     and the number of detected laminae rises from 26 to 47.
3. Downstream **CLAHE** (for dark samples, `clipLimit >= 4.0` and
   `tileGrid` shrunk to 4x4) performs local equalisation, compensating
   for the global nature of the sigmoid.
4. **Black-hat** (only for dark samples): highlights dark fine
   structures embedded in the matrix (organic-rich layers), complementing
   the Sobel-Y gradient.

`_min_delta_gray` is lowered in parallel: the physical Δgray of dark
low-contrast samples is often only 2-3 levels, so the hard floor of
"absolute amplitude threshold" is lowered to **2.5** for dark samples
(it remains 4.0 for normal samples); otherwise real laminae would be
cut off during filtering.

`_cluster_to_laminae`'s `min_support` also depends on the dark-sample
flag:

- Dark samples: `max(3, total_lines/3)` ≈ 3/11 (3 supporting lines
  suffice to form a lamina).
- Normal samples: `max(3, total_lines/2)` ≈ 5/11 (keeps the original
  strictness).

Weak laminae often "fade in and out" laterally across the core -- a few
scan lines may happen to land on their fade-out segments, so they only
show up on 3-4 lines. Requiring half-support would erase such real but
weak laminae. The combination of 1/3 support plus the downstream
[dip <= 45°, residual <= 12 px] geometric checks both rescues weak
laminae and blocks isolated noise.

## Fracture Detection (avoid killing real laminae)

The core is laid horizontally, so real laminae are 5-10 px vertical
dark bands while real fractures are >= 30 px wide dark bands. The
program flags a region as a fracture only when *all three* of the
following hold:

- Column-mean dip threshold = `mean + (3.5~4.0) * std` (stricter for
  dark samples).
- Smoothing radius `trend_sigma = max(80, w/12)` must exceed a single
  lamina width.
- Continuous fracture width >= `max(15, min_layer_width * 3)` px.

> User perspective: both the **Batch** and the **Single-image** modes of
> the GUI expose a "Flatten tilted core" checkbox. Even with it
> unchecked, slope voting automatically connects candidate points along
> the real dip direction, so large-dip laminae (28°/30°) are still
> recognised correctly.

## Unique Laminae (spacing voting + 2D line fitting + cross-line clustering)

A real lamina is a "relatively straight line" (**slight tilt allowed**).
See the previous [Spacing-based slope voting] section for the full
pipeline: slope voting finds the dominant dip on all candidate points
automatically, dip-aware clustering and 2D fitting follow, and the
[support-lines / dip / residual] triple criterion keeps only the real
laminae.

**If any criterion is not met, the cluster is marked as a "smudge"
(`is_valid=False`) and `rejection_reasons` is recorded.**


## FAQ

**Image fails to load**: check that the file format and path are
correct, and that the file is not corrupted.

**Detection results look poor**: try a different threshold method,
adjust the minimum lamina width, or place scan lines manually.

**Runs slowly**: keep the image width around 1000-2000 px, or limit
the processing size via `--max-image-size`.

## Changelog

### v2.0.0
- Refactored by function into independent modules (core / batch / gui)
- Full integration between the GUI and the core algorithms
- Batch processing, paper-figure export, and scale calibration
