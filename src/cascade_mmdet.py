from __future__ import annotations

import json
import os
import shutil
import builtins
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.config import Config
from src.dataset import split_train_val_image_ids

CASCADE_ARCH_SWIN_L = "cascade_rcnn_swin_l"
CASCADE_ARCH_CONVNEXT_XL = "cascade_rcnn_convnext_xl"
CASCADE_ARCHITECTURES = {CASCADE_ARCH_SWIN_L, CASCADE_ARCH_CONVNEXT_XL}


def is_cascade_architecture(architecture: str) -> bool:
    return architecture in CASCADE_ARCHITECTURES


def _load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_coco(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _subset_coco(coco: Dict[str, Any], image_ids: Iterable[int]) -> Dict[str, Any]:
    wanted = {int(x) for x in image_ids}
    images = [img for img in coco.get("images", []) if int(img.get("id", -1)) in wanted]
    anns = [ann for ann in coco.get("annotations", []) if int(ann.get("image_id", -1)) in wanted]

    subset: Dict[str, Any] = {
        "images": images,
        "annotations": anns,
        "categories": coco.get("categories", []),
    }
    if "info" in coco:
        subset["info"] = coco["info"]
    if "licenses" in coco:
        subset["licenses"] = coco["licenses"]
    return subset


def _extract_class_names_from_coco(coco: Dict[str, Any]) -> Tuple[str, ...]:
    categories = coco.get("categories", [])
    if not isinstance(categories, list):
        return ("rfi",)

    names: List[str] = []
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        raw_name = cat.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            names.append(raw_name.strip())

    if not names:
        return ("rfi",)

    # Keep insertion order while removing duplicates.
    uniq = list(dict.fromkeys(names))
    return tuple(uniq)


def _build_backbone_and_neck(arch: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if arch == CASCADE_ARCH_SWIN_L:
        backbone = {
            "type": "SwinTransformer",
            "embed_dims": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "window_size": 12,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "drop_path_rate": 0.10,
            "patch_norm": True,
            "out_indices": (0, 1, 2, 3),
            "with_cp": False,
            "convert_weights": True,
            "init_cfg": None,
        }
        neck = {
            "type": "FPN",
            "in_channels": [192, 384, 768, 1536],
            "out_channels": 256,
            "num_outs": 5,
        }
        return backbone, neck

    if arch == CASCADE_ARCH_CONVNEXT_XL:
        backbone = {
            # ConvNeXt is provided by MMPretrain in many MMDet installs.
            "type": "mmpretrain.ConvNeXt",
            "arch": "xlarge",
            "out_indices": [0, 1, 2, 3],
            "drop_path_rate": 0.10,
            "layer_scale_init_value": 1.0,
            "gap_before_final_norm": False,
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_3rdparty_in21k_20220124-f909bad7.pth",
                "prefix": "backbone",
            },
        }
        neck = {
            "type": "FPN",
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": 256,
            "num_outs": 5,
        }
        return backbone, neck

    raise ValueError(f"Unsupported Cascade architecture: {arch}")


def _build_mmdet_cfg(
    cfg: Config,
    train_ann_path: Path,
    val_ann_path: Path,
    work_dir: Path,
    class_names: Tuple[str, ...],
) -> Dict[str, Any]:
    if cfg.train.image_size is None:
        img_h, img_w = (1024, 1024)
    else:
        img_h, img_w = cfg.train.image_size

    backbone, neck = _build_backbone_and_neck(cfg.model.architecture)

    # Dataset statistics show tiny boxes and extreme aspect ratios; use aggressive
    # anchor ratios and small scales to increase recall on thin artifacts.
    num_fg_classes = len(class_names)

    warmup_end = min(5, max(2, int(cfg.train.epochs // 10)))

    model = {
        "type": "CascadeRCNN",
        "data_preprocessor": {
            "type": "DetDataPreprocessor",
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "bgr_to_rgb": True,
            "pad_size_divisor": 32,
        },
        "backbone": backbone,
        "neck": neck,
        "rpn_head": {
            "type": "RPNHead",
            "in_channels": 256,
            "feat_channels": 256,
            "anchor_generator": {
                "type": "AnchorGenerator",
                "scales": [1, 2, 4, 8],
                "ratios": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                "strides": [4, 8, 16, 32, 64],
            },
            "bbox_coder": {
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [1.0, 1.0, 1.0, 1.0],
            },
            "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": True, "loss_weight": 1.0},
            "loss_bbox": {"type": "L1Loss", "loss_weight": 1.0},
        },
        "roi_head": {
            "type": "CascadeRoIHead",
            "num_stages": 3,
            "stage_loss_weights": [1.0, 0.5, 0.25],
            "bbox_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {
                    "type": "RoIAlign",
                    "output_size": (4, 14),  # alto x ancho — más largo para capturar líneas
                    "sampling_ratio": 2,     # sampling_ratio > 0 es más preciso para objetos pequeños
                },
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32],
            },
            "bbox_head": [
                {
                    "type": "Shared2FCBBoxHead",
                    "in_channels": 256,
                    "fc_out_channels": 1024,
                    "roi_feat_size": (4, 14),
                    "num_classes": num_fg_classes,
                    "reg_decoded_bbox": True,
                    "bbox_coder": {
                        "type": "DeltaXYWHBBoxCoder",
                        "target_means": [0.0, 0.0, 0.0, 0.0],
                        "target_stds": [0.1, 0.1, 0.2, 0.2],
                    },
                    "reg_class_agnostic": True,
                    "loss_cls": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                    "loss_bbox": {"type": "GIoULoss", "loss_weight": 2.0},
                },
                {
                    "type": "Shared2FCBBoxHead",
                    "in_channels": 256,
                    "fc_out_channels": 1024,
                    "roi_feat_size": (4, 14),
                    "num_classes": num_fg_classes,
                    "reg_decoded_bbox": True,
                    "bbox_coder": {
                        "type": "DeltaXYWHBBoxCoder",
                        "target_means": [0.0, 0.0, 0.0, 0.0],
                        "target_stds": [0.05, 0.05, 0.1, 0.1],
                    },
                    "reg_class_agnostic": True,
                    "loss_cls": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                    "loss_bbox": {"type": "GIoULoss", "loss_weight": 2.0},
                },
                {
                    "type": "Shared2FCBBoxHead",
                    "in_channels": 256,
                    "fc_out_channels": 1024,
                    "roi_feat_size": (4, 14),
                    "num_classes": num_fg_classes,
                    "reg_decoded_bbox": True,
                    "bbox_coder": {
                        "type": "DeltaXYWHBBoxCoder",
                        "target_means": [0.0, 0.0, 0.0, 0.0],
                        "target_stds": [0.033, 0.033, 0.067, 0.067],
                    },
                    "reg_class_agnostic": True,
                    "loss_cls": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                    "loss_bbox": {"type": "GIoULoss", "loss_weight": 2.0},
                },
            ],
        },
        "train_cfg": {
            "rpn": {
                "assigner": {
                    "type": "MaxIoUAssigner",
                    "pos_iou_thr": 0.7,
                    "neg_iou_thr": 0.3,
                    "min_pos_iou": 0.3,
                    "match_low_quality": True,
                    "ignore_iof_thr": -1,
                },
                "sampler": {
                    "type": "RandomSampler",
                    "num": 256,
                    "pos_fraction": 0.5,
                    "neg_pos_ub": -1,
                    "add_gt_as_proposals": False,
                },
                "allowed_border": -1,
                "pos_weight": -1,
                "debug": False,
            },
            "rpn_proposal": {
                "nms_pre": 6000,
                "max_per_img": 3000,
                "nms": {"type": "nms", "iou_threshold": 0.7},
                "min_bbox_size": 0,
            },
            "rcnn": [
                {
                    "assigner": {
                        "type": "MaxIoUAssigner",
                        "pos_iou_thr": 0.5,
                        "neg_iou_thr": 0.5,
                        "min_pos_iou": 0.5,
                        "match_low_quality": False,
                        "ignore_iof_thr": -1,
                    },
                    "sampler": {
                        "type": "RandomSampler",
                        "num": 512,
                        "pos_fraction": 0.25,
                        "neg_pos_ub": -1,
                        "add_gt_as_proposals": True,
                    },
                    "pos_weight": -1,
                    "debug": False,
                },
                {
                    "assigner": {
                        "type": "MaxIoUAssigner",
                        "pos_iou_thr": 0.6,
                        "neg_iou_thr": 0.6,
                        "min_pos_iou": 0.6,
                        "match_low_quality": False,
                        "ignore_iof_thr": -1,
                    },
                    "sampler": {
                        "type": "RandomSampler",
                        "num": 512,
                        "pos_fraction": 0.25,
                        "neg_pos_ub": -1,
                        "add_gt_as_proposals": True,
                    },
                    "pos_weight": -1,
                    "debug": False,
                },
                {
                    "assigner": {
                        "type": "MaxIoUAssigner",
                        "pos_iou_thr": 0.6,
                        "neg_iou_thr": 0.6,
                        "min_pos_iou": 0.6,
                        "match_low_quality": False,
                        "ignore_iof_thr": -1,
                    },
                    "sampler": {
                        "type": "RandomSampler",
                        "num": 512,
                        "pos_fraction": 0.25,
                        "neg_pos_ub": -1,
                        "add_gt_as_proposals": True,
                    },
                    "pos_weight": -1,
                    "debug": False,
                },
            ],
        },
        "test_cfg": {
            "rpn": {
                "nms_pre": 4000,
                "max_per_img": 2000,
                "nms": {"type": "nms", "iou_threshold": 0.7},
                "min_bbox_size": 0,
            },
            "rcnn": {
                "score_thr": 0.0,
                "nms": {
                    "type": "soft_nms",        # en lugar de "nms"
                    "iou_threshold": 0.5,
                    "min_score": 0.001,
                },
                "max_per_img": int(cfg.model.detections_per_img),
            },
        },
    }

    train_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "with_bbox": True},
        {
            "type": "RandomChoiceResize",
            "scales": [(512, 512), (640, 640), (768, 768)],
            "keep_ratio": True,
        },
        {"type": "RandomFlip", "prob": 0.5, "direction": ["horizontal", "vertical"]},
        {
            "type": "PhotoMetricDistortion",
            "brightness_delta": 20,
            "contrast_range": (0.8, 1.2),
            "saturation_range": (1.0, 1.0),  # sin cambio de saturación
            "hue_delta": 0,                   # sin cambio de hue
        },
        {"type": "PackDetInputs"},
    ]
    test_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "Resize", "scale": (int(img_w), int(img_h)), "keep_ratio": True},
        {"type": "LoadAnnotations", "with_bbox": True},
        {
            "type": "PackDetInputs",
            "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        },
    ]

    dataroot = str(cfg.paths.project_root) + "/"
    train_dataset: Dict[str, Any] = {
        "type": "CocoDataset",
        "metainfo": {"classes": class_names},
        "data_root": dataroot,
        "ann_file": str(train_ann_path),
        "data_prefix": {"img": "data/images/train/"},
        "filter_cfg": {"filter_empty_gt": False, "min_size": 1},
        "pipeline": train_pipeline,
    }

    # NOTE:
    # In some mmcv/mmdet/torch combinations, RPN NMS can fail under autocast with:
    # "Index put requires the source and destination dtypes match".
    # We keep Cascade MMDet in fp32 for stability.
    use_mmdet_amp = False
    if cfg.train.use_amp:
        print(
            "[cascade] AMP requested but disabled for MMDet due to a known "
            "mmcv batched_nms dtype mismatch issue. Falling back to fp32."
        )

    optim_wrapper_type = "AmpOptimWrapper" if use_mmdet_amp else "OptimWrapper"
    optim_wrapper: Dict[str, Any] = {
        "type": optim_wrapper_type,
        "optimizer": {
            "type": "AdamW",
            "lr": float(cfg.train.learning_rate),
            "weight_decay": float(cfg.train.weight_decay),
        },
        "paramwise_cfg": {
            "decay_rate": 0.8,
            "decay_type": "layer_wise",
            "num_layers": 12,  # ConvNeXt-XL tiene ~12 bloques
        },
    }
    if use_mmdet_amp:
        optim_wrapper["dtype"] = "float16"
    if cfg.train.grad_clip_norm is not None:
        optim_wrapper["clip_grad"] = {
            "max_norm": float(cfg.train.grad_clip_norm),
            "norm_type": 2,
        }

    dist_backend = "gloo" if os.name == "nt" else "nccl"
    mp_start_method = "spawn" if os.name == "nt" else "fork"

    runtime_cfg: Dict[str, Any] = {
        "default_scope": "mmdet",
        "custom_imports": {
            "imports": ["mmpretrain.models"],
            "allow_failed_imports": False,
        },
        "work_dir": str(work_dir),
        "model": model,
        "train_dataloader": {
            "batch_size": int(cfg.train.batch_size),
            "num_workers": int(cfg.train.num_workers),
            "persistent_workers": bool(cfg.train.num_workers > 0),
            "sampler": {"type": "DefaultSampler", "shuffle": True},
            "batch_sampler": {"type": "AspectRatioBatchSampler"},
            "dataset": train_dataset,
        },
        "val_dataloader": {
            "batch_size": max(1, int(cfg.train.batch_size)),
            "num_workers": int(cfg.train.num_workers),
            "persistent_workers": bool(cfg.train.num_workers > 0),
            "drop_last": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": {
                "type": "CocoDataset",
                "metainfo": {"classes": class_names},
                "data_root": dataroot,
                "ann_file": str(val_ann_path),
                "data_prefix": {"img": "data/images/train/"},
                "test_mode": True,
                "pipeline": test_pipeline,
            },
        },
        "test_dataloader": {
            "batch_size": 1,
            "num_workers": int(cfg.train.num_workers),
            "persistent_workers": bool(cfg.train.num_workers > 0),
            "drop_last": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": {
                "type": "CocoDataset",
                "metainfo": {"classes": class_names},
                "data_root": dataroot,
                "ann_file": str(val_ann_path),
                "data_prefix": {"img": "data/images/train/"},
                "test_mode": True,
                "pipeline": test_pipeline,
            },
        },
        "val_evaluator": {
            "type": "CocoMetric",
            "ann_file": str(val_ann_path),
            "metric": "bbox",
            "format_only": False,
        },
        "test_evaluator": {
            "type": "CocoMetric",
            "ann_file": str(val_ann_path),
            "metric": "bbox",
            "format_only": False,
        },
        "train_cfg": {"type": "EpochBasedTrainLoop", "max_epochs": int(cfg.train.epochs), "val_interval": 1},
        "val_cfg": {"type": "ValLoop"},
        "test_cfg": {"type": "TestLoop"},
        "optim_wrapper": optim_wrapper,
        "param_scheduler": [
            {
                "type": "LinearLR",
                "start_factor": 0.1,
                "by_epoch": True,
                "begin": 0,
                "end": warmup_end,
            },
            {
                "type": "CosineAnnealingLR",
                "eta_min": 1e-6,
                "by_epoch": True,
                "begin": warmup_end,
                "end": int(cfg.train.epochs),
                "T_max": int(cfg.train.epochs) - warmup_end,
            },

        ],
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": 1,
                "max_keep_ckpts": int(cfg.train.save_top_k),
                "save_best": "coco/bbox_mAP",
                "rule": "greater",
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
        "env_cfg": {
            "cudnn_benchmark": True,
            "mp_cfg": {"mp_start_method": mp_start_method, "opencv_num_threads": 0},
            "dist_cfg": {"backend": dist_backend},
        },
        "log_processor": {"type": "LogProcessor", "window_size": 50, "by_epoch": True},
        "log_level": "INFO",
        "load_from": None,
        "resume": False,
        "auto_scale_lr": {
            "enable": False,
            "base_batch_size": 16  # Asumimos que 1e-4 es tu LR ideal para un batch total de 16
        },
    }

    return runtime_cfg


def _patch_transformers_nn_nameerror() -> None:
    """
    Work around a known transformers import bug that can raise:
    NameError: name 'nn' is not defined

    This happens when transformers disables torch integration (e.g. strict
    version gate) but still evaluates type hints referencing nn.Module.
    """
    try:
        import torch
    except Exception:
        return

    if not hasattr(builtins, "nn"):
        builtins.nn = torch.nn


def train_cascade_rcnn(cfg: Config) -> None:
    try:
        from mmengine.config import Config as MMConfig
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise ImportError(
            "Cascade R-CNN requires mmdet/mmengine/mmcv. Install with: "
            "pip install -U openmim && mim install mmengine mmcv mmdet"
        ) from exc

    _patch_transformers_nn_nameerror()

    if cfg.model.architecture == CASCADE_ARCH_CONVNEXT_XL:
        try:
            import mmpretrain  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ConvNeXt-XL requires mmpretrain. Install with: pip install mmpretrain"
            ) from exc

    try:
        register_all_modules(init_default_scope=True)
    except NameError as exc:
        if "nn" in str(exc):
            raise RuntimeError(
                "MMDetection failed due to an incompatible transformers build. "
                "Try pinning transformers to a stable version (for example 4.44.x) "
                "or upgrading torch to a version supported by your transformers package."
            ) from exc
        raise

    train_ids, val_ids = split_train_val_image_ids(
        annotation_path=cfg.paths.train_annotations_path,
        val_fraction=cfg.train.val_fraction,
        seed=cfg.train.seed,
    )

    coco = _load_coco(cfg.paths.train_annotations_path)
    class_names = _extract_class_names_from_coco(coco)
    print(f"[cascade] classes={class_names} (num_classes={len(class_names)})")

    train_ann_path = cfg.paths.outputs_dir / "cascade_train_split.json"
    val_ann_path = cfg.paths.outputs_dir / "cascade_val_split.json"
    _write_coco(train_ann_path, _subset_coco(coco, train_ids))
    _write_coco(val_ann_path, _subset_coco(coco, val_ids))

    work_dir = cfg.paths.outputs_dir / "cascade_runs" / cfg.model.architecture
    work_dir.mkdir(parents=True, exist_ok=True)

    mmdet_cfg = _build_mmdet_cfg(
        cfg=cfg,
        train_ann_path=train_ann_path,
        val_ann_path=val_ann_path,
        work_dir=work_dir,
        class_names=class_names,
    )
    config_path = work_dir / "cascade_config.py"
    MMConfig(mmdet_cfg).dump(str(config_path))
    print(f"[cascade] Config saved to {config_path}")

    runner = Runner.from_cfg(MMConfig(mmdet_cfg))
    runner.train()

    best_candidates = sorted(work_dir.glob("best*.pth"))
    best_ckpt = best_candidates[0] if best_candidates else None
    last_ckpt = work_dir / "latest.pth"
    target_best = cfg.paths.models_dir / f"best_{cfg.model.architecture}.pth"
    target_last = cfg.paths.models_dir / f"last_{cfg.model.architecture}.pth"

    if best_ckpt is not None and best_ckpt.exists():
        shutil.copy2(best_ckpt, target_best)
        print(f"[cascade] Best checkpoint copied to {target_best}")
    else:
        print("[cascade] Best checkpoint not found (hook name may differ).")

    if last_ckpt.exists():
        shutil.copy2(last_ckpt, target_last)
        print(f"[cascade] Last checkpoint copied to {target_last}")
