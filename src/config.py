from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class PathsConfig:
    project_root: Path
    train_images_dir: Path
    test_images_dir: Path
    train_annotations_path: Path
    test_id_mapping_path: Optional[Path]
    models_dir: Path
    outputs_dir: Path


@dataclass
class ModelConfig:
    architecture: str = "fasterrcnn_resnet50_fpn_v2"
    # Supported values:
    # - fasterrcnn_resnet50_fpn_v2
    # - fasterrcnn_mobilenet_v3_large_fpn
    # - cascade_rcnn_swin_l
    # - cascade_rcnn_convnext_xl
    # - cascade_rcnn_resnet50
    # - cascade_rcnn_resnet101
    # - cascade_rcnn_dcnv2
    # - cascade_rcnn_hrnet
    pretrained_weights: str = "DEFAULT"
    num_classes: int = 2  # background + RFI
    score_thresh: float = 0.001
    nms_thresh: float = 0.5
    detections_per_img: int = 1000
    ssl_backbone_path: Optional[Path] = None


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 12
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    momentum: float = 0.9
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    num_workers: int = 0
    val_fraction: float = 0.2
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    image_size: Optional[Tuple[int, int]] = (640, 640)
    grad_clip_norm: Optional[float] = 5.0
    category_id: int = 1
    min_score_eval: float = 0.001
    use_amp: bool = True
    grad_accum_steps: int = 4
    preprocess_mode: str = "none"
    max_train_steps_per_epoch: Optional[int] = None
    max_val_steps: Optional[int] = None
    save_top_k: int = 5


@dataclass
class InferenceConfig:
    batch_size: int = 4
    num_workers: int = 2
    min_score: float = 0.001
    max_detections_per_image: int = 300
    use_tta: bool = True
    use_wbf: bool = True
    wbf_iou_thr: float = 0.55
    wbf_skip_box_thr: float = 0.001
    nms_iou_thr: float = 0.5
    preprocess_mode: str = "none"


@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig
    train: TrainConfig
    inference: InferenceConfig

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return _stringify_paths(result)


def default_config(project_root: Optional[str | Path] = None) -> Config:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]

    paths = PathsConfig(
        project_root=root,
        train_images_dir=root / "data" / "images" / "train",
        test_images_dir=root / "data" / "images" / "test",
        train_annotations_path=root / "data" / "annotations" / "instances_train.json",
        test_id_mapping_path=root / "catalog.v1.parquet",
        models_dir=root / "models",
        outputs_dir=root / "outputs",
    )

    return Config(
        paths=paths,
        model=ModelConfig(),
        train=TrainConfig(),
        inference=InferenceConfig(),
    )


def ensure_dirs(cfg: Config) -> None:
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.outputs_dir.mkdir(parents=True, exist_ok=True)


def _stringify_paths(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {k: _stringify_paths(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_stringify_paths(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_stringify_paths(v) for v in data)
    return data
