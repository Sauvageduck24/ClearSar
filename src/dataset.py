from __future__ import annotations

import csv
import json
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings(
    "ignore",
    message=r"Error fetching version info.*",
    category=UserWarning,
    module=r"albumentations\.check_version",
)

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None


def coco_xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    """Convert COCO box [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def xyxy_to_coco_xywh(box: Sequence[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h]."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def get_train_transforms(image_size: Optional[Tuple[int, int]] = None) -> Any:
    """
    SAR-focused data augmentation pipeline preserving HBB signatures.

    Notes:
    - Boxes are handled in pascal_voc format [x1, y1, x2, y2].
    - Avoids transforms that destroy RGB signatures (e.g. grayscale).
    """
    if A is None:
        raise ImportError("albumentations is required for augmentation transforms.")

    transforms: List[Any] = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomSizedBBoxSafeCrop(width=500, height=500, erosion_rate=0.0, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.25,
            p=0.5,
        ),
    ]

    if image_size is not None:
        h, w = image_size
        transforms.append(A.Resize(height=h, width=w, p=1.0))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1,
            min_visibility=0.1,
        ),
    )


def get_valid_transforms(image_size: Optional[Tuple[int, int]] = None) -> Any:
    """Validation/test transforms with deterministic resize only."""
    if image_size is None:
        return None
    if A is None:
        raise ImportError("albumentations is required when image_size resize is requested.")

    h, w = image_size
    return A.Compose(
        [A.Resize(height=h, width=w, p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def _pil_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 image to CHW float tensor in [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image HWC, got shape={image.shape}")
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


@dataclass(frozen=True)
class SampleMeta:
    image_id: int
    file_name: str
    width: int
    height: int


class ClearSARCocoDataset(Dataset):
    """
    COCO-based dataset for Faster R-CNN style detectors.

    Returns:
        image: torch.Tensor [3, H, W]
        target: dict with boxes in xyxy, labels, image_id, area, iscrowd
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotation_path: str | Path,
        image_ids: Optional[Sequence[int]] = None,
        transforms: Optional[Callable[..., Any]] = None,
        category_id: int = 1,
        preprocess_mode: str = "none",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotation_path = Path(annotation_path)
        self.transforms = transforms
        self.category_id = category_id
        self.preprocess_mode = preprocess_mode

        if self.preprocess_mode != "none":
            raise ValueError(
                "Only preprocess_mode='none' is supported to preserve original RGB quicklooks."
            )

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        with self.annotation_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self._images: Dict[int, Dict[str, Any]] = {img["id"]: img for img in coco["images"]}
        anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in coco["annotations"]:
            if int(ann.get("category_id", self.category_id)) != self.category_id:
                continue
            anns_by_image[int(ann["image_id"])].append(ann)
        self._anns_by_image = anns_by_image

        if image_ids is None:
            self.image_ids = sorted(self._images.keys())
        else:
            missing = [img_id for img_id in image_ids if img_id not in self._images]
            if missing:
                raise ValueError(
                    f"Found image_ids not present in COCO images section: {missing[:10]}"
                )
            self.image_ids = list(image_ids)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = int(self.image_ids[idx])
        info = self._images[image_id]
        file_name = info["file_name"]
        img_path = self.images_dir / file_name

        if not img_path.exists():
            raise FileNotFoundError(f"Image listed in COCO not found on disk: {img_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = image.shape[:2]

        anns = self._anns_by_image.get(image_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        area: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            x1, y1, x2, y2 = coco_xywh_to_xyxy(ann["bbox"])
            x1 = float(np.clip(x1, 0, img_w - 1))
            y1 = float(np.clip(y1, 0, img_h - 1))
            x2 = float(np.clip(x2, 0, img_w - 1))
            y2 = float(np.clip(y2, 0, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(1)
            area.append(float((x2 - x1) * (y2 - y1)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = [list(map(float, b)) for b in transformed["bboxes"]]
            labels = [int(l) for l in transformed["labels"]]
            area = [float((b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
            iscrowd = [0] * len(boxes)

        image_tensor = _pil_to_tensor(np.asarray(image))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = torch.tensor(area, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }
        return image_tensor, target


class ClearSARTestDataset(Dataset):
    """
    Test dataset that always returns REAL competition image_id from mapping.

    This class explicitly avoids index-based IDs (0..N-1), a common source of
    invalid submissions.

    Returns:
        image: torch.Tensor [3, H, W]
        meta: dict with image_id, file_name, width, height
    """

    def __init__(
        self,
        images_dir: str | Path,
        filename_to_image_id: Mapping[str, int],
        transforms: Optional[Callable[..., Any]] = None,
        suffixes: Sequence[str] = (".png", ".jpg", ".jpeg"),
        preprocess_mode: str = "none",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.filename_to_image_id = {str(k): int(v) for k, v in filename_to_image_id.items()}
        self.preprocess_mode = preprocess_mode

        if self.preprocess_mode != "none":
            raise ValueError(
                "Only preprocess_mode='none' is supported to preserve original RGB quicklooks."
            )

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        suffixes_lc = {s.lower() for s in suffixes}
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes_lc
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

        missing = [p.name for p in self.image_paths if p.name not in self.filename_to_image_id]
        if missing:
            raise ValueError(
                "Missing filename->image_id mapping for test files. "
                f"Examples: {missing[:10]}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        image_id = int(self.filename_to_image_id[img_path.name])

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        image_tensor = _pil_to_tensor(np.asarray(image))
        meta = {
            "image_id": image_id,
            "file_name": img_path.name,
            "width": int(w),
            "height": int(h),
        }
        return image_tensor, meta


def collate_detection_batch(batch: Sequence[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
    """Collate function for torch DataLoader with variable-size targets."""
    return tuple(zip(*batch))


def split_train_val_image_ids(
    annotation_path: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    stratify_by_presence: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Reproducible split based on image IDs listed in COCO JSON.

    Returns train_ids and val_ids.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    rng = random.Random(seed)
    image_ids = sorted({int(img["id"]) for img in coco["images"]})

    if stratify_by_presence:
        positive_ids = {int(ann["image_id"]) for ann in coco.get("annotations", [])}
        pos_group = [img_id for img_id in image_ids if img_id in positive_ids]
        neg_group = [img_id for img_id in image_ids if img_id not in positive_ids]

        rng.shuffle(pos_group)
        rng.shuffle(neg_group)

        n_val_pos = max(1, int(round(len(pos_group) * val_fraction))) if pos_group else 0
        n_val_neg = max(1, int(round(len(neg_group) * val_fraction))) if neg_group else 0

        val_ids = sorted(pos_group[:n_val_pos] + neg_group[:n_val_neg])
        train_ids = sorted(pos_group[n_val_pos:] + neg_group[n_val_neg:])
    else:
        rng.shuffle(image_ids)
        n_val = max(1, int(round(len(image_ids) * val_fraction)))
        val_ids = sorted(image_ids[:n_val])
        train_ids = sorted(image_ids[n_val:])

    return train_ids, val_ids


def load_test_id_mapping(
    test_images_dir: str | Path,
    mapping_path: Optional[str | Path] = None,
    strict: bool = True,
) -> Dict[str, int]:
    """
    Build filename -> real image_id mapping for test set.

    Priority:
    1) mapping_path provided (.json, .csv, .parquet) -> strict validation.
    2) fallback to numeric filename stem if mapping_path is missing.

    strict=True enforces full coverage and uniqueness, preventing accidental
    0..N-1 reindexing in submission code.
    """
    test_images_dir = Path(test_images_dir)
    files = sorted(p.name for p in test_images_dir.iterdir() if p.is_file())
    if not files:
        raise ValueError(f"No files found in test_images_dir={test_images_dir}")

    mapping: Dict[str, int]
    if mapping_path is not None:
        mapping = _read_mapping_file(Path(mapping_path))
    else:
        mapping = {}
        for name in files:
            stem = Path(name).stem
            if not stem.isdigit():
                if strict:
                    raise ValueError(
                        "Cannot infer image_id from filename stem. "
                        f"Provide mapping_path. Offending file: {name}"
                    )
                continue
            mapping[name] = int(stem)

    if strict:
        missing = [f for f in files if f not in mapping]
        if missing:
            raise ValueError(
                "Mapping does not cover all test files. "
                f"Examples missing: {missing[:10]}"
            )

        ids = list(mapping.values())
        if len(set(ids)) != len(ids):
            raise ValueError("Mapping has duplicate image_id values; image_id must be unique per file.")

    return mapping


def _read_mapping_file(path: Path) -> Dict[str, int]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_mapping_json(path)
    if suffix == ".csv":
        return _read_mapping_csv(path)
    if suffix == ".parquet":
        return _read_mapping_parquet(path)
    raise ValueError(f"Unsupported mapping extension: {suffix}")


def _normalize_filename(name: str) -> str:
    return Path(str(name)).name


def _read_mapping_json(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)

    mapping: Dict[str, int] = {}

    if isinstance(content, dict):
        if "images" in content and isinstance(content["images"], list):
            for img in content["images"]:
                file_name = _normalize_filename(img["file_name"])
                image_id = int(img.get("image_id", img.get("id")))
                mapping[file_name] = image_id
            return mapping

        for k, v in content.items():
            mapping[_normalize_filename(k)] = int(v)
        return mapping

    if isinstance(content, list):
        for row in content:
            file_name = _normalize_filename(row["file_name"])
            image_id = int(row.get("image_id", row.get("id")))
            mapping[file_name] = image_id
        return mapping

    raise ValueError(f"Unsupported JSON structure for mapping in {path}")


def _read_mapping_csv(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = _normalize_filename(row["file_name"])
            image_id_raw = row.get("image_id", row.get("id"))
            if image_id_raw is None:
                raise ValueError("CSV mapping requires image_id or id column.")
            mapping[file_name] = int(image_id_raw)
    return mapping


def _read_mapping_parquet(path: Path) -> Dict[str, int]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading parquet mapping requires pandas + pyarrow (or fastparquet)."
        ) from exc

    df = pd.read_parquet(path)
    mapping: Dict[str, int] = {}

    if {"file_name", "image_id"}.issubset(df.columns):
        for _, row in df[["file_name", "image_id"]].iterrows():
            mapping[_normalize_filename(str(row["file_name"]))] = int(row["image_id"])
        return mapping

    if {"file_name", "id"}.issubset(df.columns):
        for _, row in df[["file_name", "id"]].iterrows():
            mapping[_normalize_filename(str(row["file_name"]))] = int(row["id"])
        return mapping

    if {"id", "assets"}.issubset(df.columns):
        for _, row in df[["id", "assets"]].iterrows():
            row_id = str(row["id"])
            assets = row["assets"]

            file_name = _normalize_filename(row_id)
            if isinstance(assets, dict):
                asset_meta = assets.get("asset", assets)
                if isinstance(asset_meta, dict) and "href" in asset_meta:
                    href_name = _normalize_filename(str(asset_meta["href"]))
                    if href_name:
                        file_name = href_name

            stem = Path(file_name).stem
            if stem.isdigit():
                if not Path(file_name).suffix:
                    file_name = f"{stem}.png"
                mapping[file_name] = int(stem)

        if mapping:
            return mapping

    # Generic fallback for EOTDL-like catalogs where `id` can encode path.
    if "id" in df.columns:
        for _, row in df[["id"]].iterrows():
            value = str(row["id"])
            file_name = _normalize_filename(value)
            stem = Path(file_name).stem
            if stem.isdigit():
                if not Path(file_name).suffix:
                    file_name = f"{stem}.png"
                mapping[file_name] = int(stem)

    if not mapping:
        raise ValueError(
            "Could not infer filename->image_id mapping from parquet. "
            "Expected columns like [file_name, image_id] or COCO-like metadata."
        )

    return mapping
