"""Loaders PyTorch pour les datasets endoscopiques."""

from src.datasets.base import BaseEndoscopyDataset, SampleRecord, detect_specular_mask
from src.datasets.c3vd import C3VDDataset, prepare_c3vd_from_source
from src.datasets.factory import build_dataset, create_synthetic_c3vd_demo, resolve_dataset_dir
from src.datasets.simcol3d import SimCol3DDataset
from src.datasets.smoke import make_synthetic_endoscopy_batch

__all__ = [
    "BaseEndoscopyDataset",
    "C3VDDataset",
    "SampleRecord",
    "SimCol3DDataset",
    "build_dataset",
    "create_synthetic_c3vd_demo",
    "detect_specular_mask",
    "make_synthetic_endoscopy_batch",
    "prepare_c3vd_from_source",
    "resolve_dataset_dir",
]
