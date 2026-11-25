"""Socle commun des datasets PyTorch pour l'endoscopie (C3VD, SimCol3D, etc.)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    """Métadonnées d'une image et de ses annotations associées."""

    rgb_path: Path
    depth_path: Path
    sequence_name: str
    frame_id: str
    normal_path: Path | None = None
    occlusion_path: Path | None = None


def read_image_unchanged(path: Path) -> np.ndarray:
    """Lit une image (8-bit ou 16-bit PNG/TIFF) en gérant les chemins Unicode."""
    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Impossible de décoder l'image : {path}")
    return image


def detect_specular_mask(rgb_uint8: np.ndarray, v_thresh: int = 235, s_thresh: int = 45) -> np.ndarray:
    """Détecte approximativement les reflets spéculaires (forte luminosité, faible saturation).

    Retourne un masque booléen (H, W) où True indique un pixel saturé par un reflet.
    """
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    return (value >= v_thresh) & (saturation <= s_thresh)


def split_records_by_sequence(
    records: list[SampleRecord],
    split: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[SampleRecord]:
    """Sépare les échantillons en train/val/test par séquence pour éviter les fuites temporelles.

    Si moins de 3 séquences sont présentes (ex. premier test avec 1 seule séquence),
    bascule sur un découpage temporel contigu par blocs afin de ne pas obtenir de split vide.
    """
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"Split inconnu : {split}. Attendu : train, val, test ou all.")
    if split == "all" or not records:
        return records

    sequences = sorted({rec.sequence_name for rec in records})
    rng = random.Random(seed)
    rng.shuffle(sequences)

    if len(sequences) >= 3:
        n_val = max(1, round(len(sequences) * val_ratio))
        n_test = max(1, round(len(sequences) * test_ratio))
        if n_val + n_test >= len(sequences):
            n_val = 1
            n_test = 1
        val_seqs = set(sequences[:n_val])
        test_seqs = set(sequences[n_val : n_val + n_test])
        train_seqs = set(sequences[n_val + n_test :])
        target_seqs = {"train": train_seqs, "val": val_seqs, "test": test_seqs}[split]
        return [rec for rec in records if rec.sequence_name in target_seqs]

    if len(sequences) == 2:
        # 1 séquence en train, 1 partagée contiguë entre val et test
        if split == "train":
            return [rec for rec in records if rec.sequence_name == sequences[0]]
        second = [rec for rec in records if rec.sequence_name == sequences[1]]
        mid = max(1, len(second) // 2)
        return second[:mid] if split == "val" else second[mid:]

    # Cas 1 seule séquence : découpage contigu (70% train / 15% val / 15% test)
    n = len(records)
    i_val = max(1, int(n * (1.0 - val_ratio - test_ratio)))
    i_test = max(i_val + 1, int(n * (1.0 - test_ratio))) if n >= 3 else n
    if split == "train":
        return records[:i_val]
    if split == "val":
        return records[i_val:i_test] or records[-1:]
    return records[i_test:] or records[-1:]


class BaseEndoscopyDataset(Dataset):
    """Classe de base gérant le décodage, redimensionnement et l'augmentation."""

    def __init__(
        self,
        records: list[SampleRecord],
        image_size: tuple[int, int] = (256, 256),
        min_depth_mm: float = 1.0,
        max_depth_mm: float = 200.0,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.height, self.width = int(image_size[0]), int(image_size[1])
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _decode_depth_mm(self, record: SampleRecord) -> np.ndarray:
        """À implémenter dans chaque dataset pour convertir les pixels bruts en millimètres."""
        raise NotImplementedError

    def _decode_normals(self, record: SampleRecord) -> np.ndarray | None:
        """Retourne les normales (H, W, 3) dans [-1, 1] ou None si absentes."""
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        bgr_or_rgb = read_image_unchanged(record.rgb_path)
        if bgr_or_rgb.ndim == 2:
            rgb = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(bgr_or_rgb[:, :, :3], cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb / 256.0, 0, 255).astype(np.uint8)

        depth_mm = self._decode_depth_mm(record).astype(np.float32)
        normals = self._decode_normals(record)

        # Redimensionnement : bilinéaire pour RGB, plus proche voisin pour profondeur/masques
        rgb_resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(
            depth_mm, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )

        if normals is not None:
            normals_resized = cv2.resize(
                normals.astype(np.float32),
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            normals_resized = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Augmentation légère cohérente géométriquement (flip horizontal)
        if self.augment and random.random() < 0.5:
            rgb_resized = np.ascontiguousarray(rgb_resized[:, ::-1])
            depth_resized = np.ascontiguousarray(depth_resized[:, ::-1])
            if normals is not None:
                normals_resized = np.ascontiguousarray(normals_resized[:, ::-1])
                normals_resized[:, :, 0] *= -1.0  # Miroir gauche-droite => inverse Nx

        valid_mask = (
            np.isfinite(depth_resized)
            & (depth_resized >= self.min_depth_mm)
            & (depth_resized <= self.max_depth_mm)
        )

        if normals is not None:
            norm_len = np.linalg.norm(normals_resized, axis=-1, keepdims=True)
            valid_normals = norm_len.squeeze(-1) > 1e-3
            valid_mask = valid_mask & valid_normals
            normals_resized = normals_resized / np.clip(norm_len, 1e-6, None)

        specular_mask = detect_specular_mask(rgb_resized)

        image_tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float() / 255.0
        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).bool()
        normals_tensor = torch.from_numpy(normals_resized.transpose(2, 0, 1)).float()
        specular_tensor = torch.from_numpy(specular_mask).unsqueeze(0).bool()

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "normals": normals_tensor,
            "specular_mask": specular_tensor,
            "has_normals": bool(normals is not None),
            "sequence": record.sequence_name,
            "frame_id": record.frame_id,
        }
