"""Loaders PyTorch pour les datasets endoscopiques."""

from src.datasets.base import (
    BaseEndoscopyDataset,
    SampleRecord,
    assign_sequence_splits,
    detect_specular_mask,
    extract_texture_group,
    summarize_records_by_sequence,
    validate_no_texture_leakage,
)
from src.datasets.c3vd import C3VDDataset, discover_c3vd_records, prepare_c3vd_from_source
from src.datasets.factory import build_dataset, create_synthetic_c3vd_demo, resolve_dataset_dir
from src.datasets.simcol3d import SimCol3DDataset
from src.datasets.smoke import make_synthetic_endoscopy_batch

__all__ = [
    "BaseEndoscopyDataset",
    "C3VDDataset",
    "SampleRecord",
    "SimCol3DDataset",
    "assign_sequence_splits",
    "build_dataset",
    "create_synthetic_c3vd_demo",
    "detect_specular_mask",
    "discover_c3vd_records",
    "extract_texture_group",
    "make_synthetic_endoscopy_batch",
    "prepare_c3vd_from_source",
    "resolve_dataset_dir",
    "summarize_records_by_sequence",
    "validate_no_texture_leakage",
]
