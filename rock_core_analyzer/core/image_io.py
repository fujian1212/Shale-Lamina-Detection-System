#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image I/O."""

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


class ImageIOMixin:
    def _load_image(self, max_image_size=0):
        """Load image."""
        try:
            # Try the fast path first
            self._load_image_fast()

            # Fall back to the alternate loaders if needed
            if self.image is None or self.image.size == 0:
                self._load_image_fallback()

            # Verify the image was loaded successfully
            if self.image is None or self.image.size == 0:
                error_msg = f"Cannot load image: {self.image_path}. Please check the file path and format."
                raise ValueError(error_msg)

            # Image dimensions
            if len(self.image.shape) > 2:
                self.height, self.width, _ = self.image.shape
            else:
                self.height, self.width = self.image.shape

            # Keep the original image size
            self.original_height = self.height
            self.original_width = self.width
            print(f"Original image size: {self.original_width}x{self.original_height}")

            # Resize image when a size cap is configured
            if max_image_size > 0 and (self.width > max_image_size or self.height > max_image_size):
                self._resize_image(max_image_size)
                print(f"Image size after resizing: {self.width}x{self.height}")

            # Respect the memory limit
            if self.memory_limit > 0:
                self._check_memory_usage()
                print(f"Image size after memory-limit resize: {self.width}x{self.height}")

        except Exception as e:
            error_msg = f"Error while loading image: {str(e)}"
            raise ValueError(error_msg)
    def _load_image_fast(self):
        """Fast image loading via OpenCV with non-ASCII path support."""
        has_non_ascii = any(ord(c) > 127 for c in self.image_path)

        if not has_non_ascii:
            try:
                self.image = cv2.imread(self.image_path)
                if self.image is not None and self.image.size > 0:
                    return True
            except Exception:
                pass

        try:
            # Read bytes then decode: handles non-ASCII paths and avoids cv2.imread warnings
            with open(self.image_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), np.uint8)
                self.image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            return self.image is not None and self.image.size > 0
        except Exception:
            return False
    @staticmethod
    def _imwrite_safe(filepath, img):
        """Write image safely; supports non-ASCII paths (cv2.imwrite silently fails on those)."""
        if img is None:
            return False
        has_non_ascii = any(ord(c) > 127 for c in str(filepath))
        if not has_non_ascii:
            return cv2.imwrite(str(filepath), img)
        # Non-ASCII path: encode in memory then write the raw bytes
        ext = os.path.splitext(filepath)[1] if '.' in str(filepath) else '.png'
        success, buf = cv2.imencode(ext, img)
        if success:
            buf.tofile(str(filepath))
            return True
        return False
    def _load_image_fallback(self):
        """Fallback image loading."""
        # Try PIL
        try:
            from PIL import Image
            pil_img = Image.open(self.image_path).convert('RGB')
            self.image = np.array(pil_img)
            # Convert PIL RGB to OpenCV BGR
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            return True
        except Exception:
            pass

        # Try matplotlib
        try:
            import matplotlib.pyplot as plt
            plt_img = plt.imread(self.image_path)
            if len(plt_img.shape) == 3:  # color image
                self.image = (plt_img * 255).astype(np.uint8)
                # Convert matplotlib RGB to OpenCV BGR
                if plt_img.shape[2] >= 3:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            else:
                self.image = (plt_img * 255).astype(np.uint8)
            return self.image is not None and self.image.size > 0
        except Exception:
            pass

        # Final fallback: OpenCV imdecode
        try:
            with open(self.image_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), np.uint8)
                self.image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            return self.image is not None and self.image.size > 0
        except Exception:
            return False
    def _resize_image(self, max_size):
        """Resize image to fit within the maximum size."""
        # Compute the scale factor
        scale = min(max_size / self.width, max_size / self.height)

        # Only shrink, never enlarge
        if scale < 1.0:
            new_width = int(self.width * scale)
            new_height = int(self.height * scale)

            # Use a fast area-based interpolation for shrinking
            self.image = cv2.resize(self.image, (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)

            # Update dimensions
            self.height, self.width = new_height, new_width
    def _check_memory_usage(self):
        """Check memory usage and shrink the image when over budget."""
        # Image memory footprint in MB
        image_memory = self.image.nbytes / (1024 * 1024)

        # Reduce quality if we exceed the memory limit
        if image_memory > self.memory_limit:
            # Required scale factor
            scale = np.sqrt(self.memory_limit / image_memory)

            # Resize image
            new_width = int(self.width * scale)
            new_height = int(self.height * scale)
            self.image = cv2.resize(self.image, (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)

            # Update dimensions
            self.height, self.width = new_height, new_width
