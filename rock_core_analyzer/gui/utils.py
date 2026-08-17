#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI utility helpers."""

import os
import sys
import argparse
import platform
import psutil
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from ttkthemes import ThemedTk
import re
import time
import traceback
import gc
import cv2
from PIL import Image, ImageTk, ImageOps
import pandas as pd

from rock_core_analyzer.core import RockCoreLayerDetector
from rock_core_analyzer.batch import merge_batch_results

def natural_sort_key(text):
    """Natural sort key function: correctly orders strings that contain numbers.

    Args:
        text: The string to be sorted.

    Returns:
        A sort key.

    Examples:
        ``['9-1h', '9-2h', '9-10h']`` will be ordered correctly rather than
        ``['9-10h', '9-1h', '9-2h']``.
    """
    def convert(text_part):
        # Convert digits to int; lowercase everything else
        return int(text_part) if text_part.isdigit() else text_part.lower()

    # Split the string while preserving digit runs
    return [convert(part) for part in re.split(r'(\d+)', text)]
def resource_path(relative_path):
    """Return the absolute path to a resource; works for both source runs and PyInstaller bundles."""
    try:
        # PyInstaller stores extracted files under sys._MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
def get_system_info():
    """Return a snapshot of the host system."""
    system_info = {}
    system_info['platform'] = platform.system()
    system_info['platform_version'] = platform.version()
    system_info['architecture'] = platform.machine()
    system_info['processor'] = platform.processor()
    system_info['python_version'] = platform.python_version()

    # CPU core counts
    system_info['cpu_count'] = psutil.cpu_count(logical=False)
    system_info['cpu_count_logical'] = psutil.cpu_count(logical=True)

    # Memory info
    memory = psutil.virtual_memory()
    system_info['total_memory'] = memory.total / (1024 * 1024 * 1024)  # GB
    system_info['available_memory'] = memory.available / (1024 * 1024 * 1024)  # GB

    return system_info
def get_default_settings():
    """Derive default runtime settings from the host configuration."""
    settings = {}

    # Probe the system
    system_info = get_system_info()

    # Memory limit based on total RAM
    total_memory_mb = system_info['total_memory'] * 1024
    settings['memory_limit'] = int(total_memory_mb * 0.3)  # use 30% of system memory

    # Thread count based on CPU cores
    cpu_cores = system_info['cpu_count'] or 2
    settings['threads'] = max(2, cpu_cores - 1)  # leave one core free

    # Image size cap based on available RAM
    available_memory_mb = system_info['available_memory'] * 1024
    if available_memory_mb < 2048:  # < 2 GB free
        settings['max_image_size'] = 1024  # small image cap
    elif available_memory_mb < 4096:  # < 4 GB free
        settings['max_image_size'] = 2048  # medium image cap
    else:
        settings['max_image_size'] = 4096  # large image cap

    # Cache size based on available RAM
    if available_memory_mb < 2048:
        settings['cache_size'] = 3
    elif available_memory_mb < 4096:
        settings['cache_size'] = 5
    else:
        settings['cache_size'] = 10

    return settings
