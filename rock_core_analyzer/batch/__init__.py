"""Batch processing module."""
from .processing import process_image, process_folder
from .merge import (
    merge_batch_results,
    merge_position_info,
    generate_continuous_position_statistics,
    generate_batch_summary_statistics,
    generate_batch_processing_report,
    create_empty_results,
)
from .batch_viz import create_batch_visualizations
from .batch_sensitivity import (
    run_batch_sensitivity_and_ablation,
    run_single_image_sensitivity_and_ablation,
)
from .dip_calibration import (
    calibrate_subfolder_dip,
    save_group_calibration,
    batch_lamina_kwargs_from_calibration,
    DEFAULT_MAX_CALIBRATION_IMAGES,
)

__all__ = [
    "process_image",
    "process_folder",
    "merge_batch_results",
    "merge_position_info",
    "generate_continuous_position_statistics",
    "generate_batch_summary_statistics",
    "generate_batch_processing_report",
    "create_empty_results",
    "create_batch_visualizations",
    "run_batch_sensitivity_and_ablation",
    "run_single_image_sensitivity_and_ablation",
    "calibrate_subfolder_dip",
    "save_group_calibration",
    "batch_lamina_kwargs_from_calibration",
    "DEFAULT_MAX_CALIBRATION_IMAGES",
]
