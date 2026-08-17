#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rock core lamina detector."""

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

from .image_io import ImageIOMixin
from .preprocessing import PreprocessingMixin
from .detection import DetectionMixin
from .alignment import AlignmentMixin
from .statistics import StatisticsMixin
from .visualization import VisualizationMixin
from .paper_export import PaperExportMixin
from .export import ExportMixin
from .sensitivity import SensitivityMixin


class RockCoreLayerDetector(ImageIOMixin, PreprocessingMixin, DetectionMixin, AlignmentMixin, StatisticsMixin, VisualizationMixin, PaperExportMixin, ExportMixin, SensitivityMixin):
    """Rock core lamina detection and analysis."""
    def __init__(self, image_path):
        """Initialize the rock core lamina detector.

        Args:
            image_path: Path to the image file.
        """
        self.image_path = image_path
        self.image = None
        self.processed = None
        self.binary = None  # Binary image attribute
        self.aligned = False
        self.alignment_angle = 0.0
        self.use_shear = False
        self.width = 0
        self.height = 0
        self.layers = []
        self.scan_lines = []
        self.output_dir = os.path.join(os.path.dirname(image_path), "output_" + os.path.basename(image_path).split('.')[0])

        # Scale calibration (pixels per millimeter); None means not calibrated
        self.pixel_per_mm = None

        # Whether to save diagnostic intermediate images
        # (binary_image / canny_edges / validation_lines / validated_grid)
        self.save_diagnostics = True

        # Memory limit
        self.memory_limit = int(os.environ.get('ROCK_MEMORY_LIMIT', '0'))

        # Maximum image size limit
        max_image_size = int(os.environ.get('ROCK_MAX_IMAGE_SIZE', '0'))

        # Image loader flag, OpenCV by default
        self._use_opencv = True

        # Load image
        self._load_image(max_image_size)

