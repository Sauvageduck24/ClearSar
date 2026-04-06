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
CASCADE_ARCH_RESNET50 = "cascade_rcnn_resnet50"
CASCADE_ARCH_RESNET101 = "cascade_rcnn_resnet101"
CASCADE_ARCH_DCNV2 = "cascade_rcnn_dcnv2"
CASCADE_ARCH_HRNET = "cascade_rcnn_hrnet"
CASCADE_ARCHITECTURES = {
    CASCADE_ARCH_SWIN_L,
    CASCADE_ARCH_CONVNEXT_XL,
    CASCADE_ARCH_RESNET50,
    CASCADE_ARCH_RESNET101,
    CASCADE_ARCH_DCNV2,
    CASCADE_ARCH_HRNET,
}


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

    uniq = list(dict.fromkeys(names))
    return tuple(uniq)


def _coco_has_instance_masks(coco: Dict[str, Any]) -> bool:
    annotations = coco.get("annotations", [])
    if not isinstance(annotations, list):
        return False

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        seg = ann.get("segmentation")
        if isinstance(seg, list) and len(seg) > 0:
            return True
        if isinstance(seg, dict):
            counts = seg.get("counts")
            size = seg.get("size")
            if counts is not None and isinstance(size, list) and len(size) == 2:
                return True
    return False


def _analyze_bbox_distribution(coco: Dict[str, Any]) -> Dict[str, float]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        return {
            "num_boxes": 0.0,
            "small_ratio": 0.0,
            "elongated_ratio": 0.0,
            "large_ratio": 0.0,
        }

    image_area_by_id: Dict[int, float] = {}
    for img in images:
        if not isinstance(img, dict):
            continue
        img_id = img.get("id")
        w = img.get("width")
        h = img.get("height")
        if img_id is None:
            continue
        try:
            img_id_i = int(img_id)
            w_f = float(w)
            h_f = float(h)
        except (TypeError, ValueError):
            continue
        if w_f <= 0 or h_f <= 0:
            continue
        image_area_by_id[img_id_i] = w_f * h_f

    num_boxes = 0
    small = 0
    elongated = 0
    large = 0

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            bw = float(bbox[2])
            bh = float(bbox[3])
            img_id = int(ann.get("image_id"))
        except (TypeError, ValueError):
            continue
        if bw <= 0 or bh <= 0:
            continue

        img_area = image_area_by_id.get(img_id)
        if img_area is None or img_area <= 0:
            continue

        num_boxes += 1
        rel_area = (bw * bh) / img_area
        long_short = max(bw / bh, bh / bw)

        if rel_area < 0.0025:
            small += 1
        if long_short >= 8.0:
            elongated += 1
        if rel_area >= 0.15:
            large += 1

    if num_boxes == 0:
        return {
            "num_boxes": 0.0,
            "small_ratio": 0.0,
            "elongated_ratio": 0.0,
            "large_ratio": 0.0,
        }

    denom = float(num_boxes)
    return {
        "num_boxes": float(num_boxes),
        "small_ratio": small / denom,
        "elongated_ratio": elongated / denom,
        "large_ratio": large / denom,
    }


def _build_adaptive_bbox_losses(coco: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]], str]:
    stats = _analyze_bbox_distribution(coco)
    small_ratio = float(stats.get("small_ratio", 0.0))
    elongated_ratio = float(stats.get("elongated_ratio", 0.0))
    large_ratio = float(stats.get("large_ratio", 0.0))

    # Ajuste automático suave: más peso cuando predominan cajas pequeñas/elongadas/grandes.
    rpn_weight = min(20.0, 8.0 + 8.0 * small_ratio + 6.0 * elongated_ratio + 4.0 * large_ratio)
    stage1_weight = min(20.0, 8.0 + 10.0 * small_ratio + 6.0 * elongated_ratio)
    stage2_weight = min(20.0, 8.0 + 6.0 * small_ratio + 8.0 * elongated_ratio + 4.0 * large_ratio)
    stage3_weight = min(8.0, 1.5 + 2.0 * elongated_ratio + 3.0 * large_ratio)

    rpn_loss_bbox: Dict[str, Any] = {
        "type": "GIoULoss",
        "loss_weight": round(rpn_weight, 4),
    }

    stage3_loss_type = "CIoULoss" if elongated_ratio >= 0.2 else "GIoULoss"
    roi_stage_losses = (
        {"type": "CIoULoss", "loss_weight": round(stage1_weight, 4)},
        {"type": "CIoULoss", "loss_weight": round(stage2_weight, 4)},
        {"type": stage3_loss_type, "loss_weight": round(stage3_weight, 4)},
    )

    summary = (
        f"small={small_ratio:.3f}, elongated={elongated_ratio:.3f}, large={large_ratio:.3f}, "
        f"rpn={rpn_loss_bbox['type']}@{rpn_loss_bbox['loss_weight']:.3f}, "
        f"roi=[{roi_stage_losses[0]['type']}@{roi_stage_losses[0]['loss_weight']:.3f}, "
        f"{roi_stage_losses[1]['type']}@{roi_stage_losses[1]['loss_weight']:.3f}, "
        f"{roi_stage_losses[2]['type']}@{roi_stage_losses[2]['loss_weight']:.3f}]"
    )
    return rpn_loss_bbox, roi_stage_losses, summary


def _resolve_pretrained_checkpoint(arch: str, pretrained_weights: str) -> str:
    token = pretrained_weights.strip() if isinstance(pretrained_weights, str) else ""
    if token.upper() in {"NONE", "NULL", "FALSE", "NO"}:
        raise ValueError(
            "All Cascade backbones require pretrained weights. "
            f"Invalid pretrained_weights='{token}'."
        )
    if token and token.upper() != "DEFAULT":
        return token

    if arch == CASCADE_ARCH_SWIN_L:
        return "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"

    if arch == CASCADE_ARCH_CONVNEXT_XL:
        return "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_3rdparty_in21k_20220124-f909bad7.pth"

    if arch == CASCADE_ARCH_RESNET50:
        return "torchvision://resnet50"

    if arch == CASCADE_ARCH_RESNET101:
        return "torchvision://resnet101"

    if arch == CASCADE_ARCH_DCNV2:
        return "torchvision://resnet50"

    if arch == CASCADE_ARCH_HRNET:
        return "open-mmlab://msra/hrnetv2_w40"

    raise ValueError(f"Unsupported Cascade architecture: {arch}")


def _ensure_pretrained_backbone(arch: str, backbone: Dict[str, Any]) -> None:
    init_cfg = backbone.get("init_cfg") if isinstance(backbone, dict) else None
    if not isinstance(init_cfg, dict):
        raise ValueError(f"{arch} backbone is missing init_cfg for pretrained weights.")

    checkpoint = init_cfg.get("checkpoint")
    if init_cfg.get("type") != "Pretrained" or not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError(f"{arch} backbone must define a non-empty pretrained checkpoint.")


def _build_backbone_and_neck(arch: str, pretrained_weights: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pretrained_ckpt = _resolve_pretrained_checkpoint(arch, pretrained_weights)

    if arch == CASCADE_ARCH_SWIN_L:
        # FIX: window_size=12 was pretrained at 384px.
        # At 512px input the stage-1 feature map is 512/4=128px.
        # 128 % 12 != 0 → broken positional encoding, silent degradation.
        # window_size=16 → 128/16=8 ✓  (or 8 → 128/8=16 ✓)
        backbone = {
            "type": "SwinTransformer",
            "embed_dims": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "window_size": 16,           # FIX: was 12, must divide 128 evenly at 512px input
            "mlp_ratio": 4,
            "qkv_bias": True,
            "drop_path_rate": 0.10,
            "patch_norm": True,
            "out_indices": (0, 1, 2, 3),
            "with_cp": False,
            "convert_weights": True,
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": pretrained_ckpt,
            },
        }
        # FIX: Swin-L outputs at strides [4, 8, 16, 32].
        # num_outs=5 → FPN adds one extra coarse level → output strides [4, 8, 16, 32, 64].
        # AnchorGenerator and RoI extractor are set accordingly below.
        neck = {
            "type": "PAFPN",
            "in_channels": [192, 384, 768, 1536],
            "out_channels": 256,
            "num_outs": 5,               # FIX: was 6; 5 levels → strides [4,8,16,32,64]
        }
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    if arch == CASCADE_ARCH_CONVNEXT_XL:
        backbone = {
            "type": "mmpretrain.ConvNeXt",
            "arch": "xlarge",
            "out_indices": [0, 1, 2, 3],
            "drop_path_rate": 0.10,
            "layer_scale_init_value": 1.0,
            "gap_before_final_norm": False,
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": pretrained_ckpt,
                "prefix": "backbone",
            },
        }
        neck = {
            "type": "PAFPN",
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": 256,
            "num_outs": 5,               # FIX: was 6
        }
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    if arch in {CASCADE_ARCH_RESNET50, CASCADE_ARCH_RESNET101}:
        depth = 50 if arch == CASCADE_ARCH_RESNET50 else 101
        backbone = {
            "type": "ResNet",
            "depth": depth,
            "num_stages": 4,
            "out_indices": (0, 1, 2, 3),
            "frozen_stages": 1,
            "norm_cfg": {"type": "BN", "requires_grad": True},
            "norm_eval": True,
            "style": "pytorch",
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": pretrained_ckpt,
            },
        }
        neck = {
            "type": "PAFPN",
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": 256,
            "num_outs": 5,               # FIX: was 6
        }
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    if arch == CASCADE_ARCH_DCNV2:
        backbone = {
            "type": "ResNet",
            "depth": 50,
            "num_stages": 4,
            "out_indices": (0, 1, 2, 3),
            "frozen_stages": 1,
            "norm_cfg": {"type": "BN", "requires_grad": True},
            "norm_eval": True,
            "style": "pytorch",
            "dcn": {"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
            "stage_with_dcn": (False, True, True, True),
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": pretrained_ckpt,
            },
        }
        neck = {
            "type": "PAFPN",
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": 256,
            "num_outs": 5,               # FIX: was 6
        }
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    if arch == CASCADE_ARCH_HRNET:
        backbone = {
            "type": "HRNet",
            "extra": {
                "stage1": {
                    "num_modules": 1,
                    "num_branches": 1,
                    "block": "BOTTLENECK",
                    "num_blocks": (4,),
                    "num_channels": (64,),
                },
                "stage2": {
                    "num_modules": 1,
                    "num_branches": 2,
                    "block": "BASIC",
                    "num_blocks": (4, 4),
                    "num_channels": (40, 80),
                },
                "stage3": {
                    "num_modules": 4,
                    "num_branches": 3,
                    "block": "BASIC",
                    "num_blocks": (4, 4, 4),
                    "num_channels": (40, 80, 160),
                },
                "stage4": {
                    "num_modules": 3,
                    "num_branches": 4,
                    "block": "BASIC",
                    "num_blocks": (4, 4, 4, 4),
                    "num_channels": (40, 80, 160, 320),
                },
            },
            "init_cfg": {
                "type": "Pretrained",
                "checkpoint": pretrained_ckpt,
            },
        }
        neck = {
            "type": "PAFPN",
            "in_channels": [40, 80, 160, 320],
            "out_channels": 256,
            "num_outs": 5,               # FIX: was 6
        }
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    raise ValueError(f"Unsupported Cascade architecture: {arch}")


def _build_mmdet_cfg(
    cfg: Config,
    coco: Dict[str, Any],
    train_ann_path: Path,
    val_ann_path: Path,
    work_dir: Path,
    class_names: Tuple[str, ...],
    use_copy_paste: bool,
) -> Dict[str, Any]:
    if cfg.train.image_size is None:
        img_h, img_w = (1024, 1024)
    else:
        img_h, img_w = cfg.train.image_size

    backbone, neck = _build_backbone_and_neck(cfg.model.architecture, cfg.model.pretrained_weights)
    rpn_loss_bbox, roi_stage_losses, adaptive_loss_summary = _build_adaptive_bbox_losses(coco)

    num_fg_classes = len(class_names)
    warmup_end = min(5, max(3, int(cfg.train.epochs // 8)))

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
                "scales": [2, 4, 8],
                "ratios": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 33.0, 50.0],
                # FIX: was [2, 4, 8, 16, 32, 64] — stride 2 doesn't exist in FPN output
                # (ResNet/ConvNeXt/Swin all output finest at stride 4).
                # 5 strides must match num_outs=5 in FPN above.
                "strides": [4, 8, 16, 32, 64],
            },
            "bbox_coder": {
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [1.0, 1.0, 1.0, 1.0],
            },
            "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": True, "loss_weight": 1.0},
            "loss_bbox": rpn_loss_bbox,
        },
        "roi_head": {
            "type": "CascadeRoIHead",
            "num_stages": 3,
            "stage_loss_weights": [1.0, 1.0, 1.0],
            "bbox_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {
                    "type": "RoIAlign",
                    "output_size": (14, 14),
                    "sampling_ratio": 2,
                },
                "out_channels": 256,
                # FIX: was [2, 4, 8, 16, 32] — must match actual FPN output strides.
                # With num_outs=5 and ResNet backbone: FPN outputs at [4, 8, 16, 32, 64].
                # finest_scale=56 → boxes with sqrt(area) < 56 go to P2 (stride 4),
                # which is fine for our thin bands (e.g. 119×9=1071px² → sqrt=32 < 56 → P2).
                "featmap_strides": [4, 8, 16, 32, 64],
                "finest_scale": 56,
            },
            "bbox_head": [
                {
                    "type": "Shared4Conv1FCBBoxHead",
                    "in_channels": 256,
                    "fc_out_channels": 1024,
                    "roi_feat_size": (14, 14),
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
                    "loss_bbox": roi_stage_losses[0],
                },
                {
                    "type": "Shared4Conv1FCBBoxHead",
                    "in_channels": 256,
                    "fc_out_channels": 1024,
                    "roi_feat_size": (14, 14),
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
                    "loss_bbox": roi_stage_losses[1],
                },
                {
                    "type": "Shared4Conv1FCBBoxHead",
                    "in_channels": 256,
                    "fc_out_channels": 1024,
                    "roi_feat_size": (14, 14),
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
                    "loss_bbox": roi_stage_losses[2],
                },
            ],
        },
        "train_cfg": {
            "rpn": {
                "assigner": {
                    "type": "ATSSAssigner",
                    "topk": 9,
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
                "max_per_img": int(cfg.model.detections_per_img),
                "nms": {"type": "nms", "iou_threshold": 0.65},
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
                        "pos_iou_thr": 0.7,
                        "neg_iou_thr": 0.7,
                        "min_pos_iou": 0.7,
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
                "nms_pre": 6000,
                "max_per_img": int(cfg.model.detections_per_img),
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "min_bbox_size": 0,
            },
            "rcnn": {
                # FIX: was 0.0 — keeping every box down to near-zero confidence floods
                # the evaluator with false positives and degrades precision.
                # soft_nms min_score=0.001 already filters below that, but score_thr
                # is the post-NMS threshold. 0.05 is a safe competition default.
                "score_thr": 0.05,
                "nms": {
                    "type": "soft_nms",
                    "iou_threshold": 0.5,
                    "min_score": 0.001,
                },
                "max_per_img": int(cfg.model.detections_per_img),
            },
        },
    }

    multiscale_train_scales = [
        (int(img_w), int(img_h)),
        (int(img_w * 1.15), int(img_h * 1.15)),
        (int(img_w * 1.3), int(img_h * 1.3)),
    ]
    tta_scales = [
        (int(img_w * 0.9), int(img_h * 0.9)),
        (int(img_w), int(img_h)),
        (int(img_w * 1.1), int(img_h * 1.1)),
    ]

    # FIX: CopyPaste is a "mix transform" — it needs two images at once.
    # In a plain CocoDataset pipeline it silently does nothing because
    # results['mix_results'] is never populated.
    # The fix is to split the pipeline: CocoDataset handles only loading,
    # then MultiImageMixDataset wraps it and supplies the mixed second image,
    # and the full augmentation pipeline (including CopyPaste) runs on top.
    #
    # Pipeline for the inner CocoDataset — only loading, no augmentation:
    inner_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "with_bbox": True, "with_mask": bool(use_copy_paste)},
    ]

    # FIX: RandomCrop with crop_size=(float, float) is not supported in MMDet v3.
    # RandomCrop.crop_size must be (int_h, int_w). Use explicit pixel sizes here.
    # We compute a representative crop: 80% of the training image dimensions,
    # which gives a good balance between seeing full-width RFI bands and cropping.
    crop_h = max(64, int(img_h * 0.8))
    crop_w = max(64, int(img_w * 0.8))

    # Shared augmentations used in both training setups.
    # NOTE: allow_negative_crop=True avoids repeatedly returning None on sparse datasets.
    common_aug_pipeline = [
        {"type": "RandomChoiceResize", "scales": multiscale_train_scales, "keep_ratio": True},
        {"type": "RandomFlip", "prob": 0.5, "direction": ["horizontal"]},
        # Vertical flip: for horizontal RFI bands this just changes y-position,
        # which helps the model generalise to bands at different vertical positions.
        {"type": "RandomFlip", "prob": 0.5, "direction": ["vertical"]},
        {"type": "RandomShift", "max_shift_px": 32},
        # RandomCrop with explicit int sizes (not float ratios):
        {"type": "RandomCrop", "crop_size": (crop_h, crop_w), "allow_negative_crop": True},
        {
            "type": "PhotoMetricDistortion",
            "brightness_delta": 20,
            "contrast_range": (0.8, 1.2),
            "saturation_range": (1.0, 1.0),
            "hue_delta": 0,
        },
    ]

    # Full augmentation pipeline (runs inside MultiImageMixDataset when CopyPaste is enabled).
    train_pipeline = [*common_aug_pipeline, {"type": "PackDetInputs"}]
    if use_copy_paste:
        # CopyPaste needs gt masks; only enable it for segmentation-capable datasets.
        train_pipeline.insert(0, {"type": "CopyPaste", "max_num_pasted": 4})

    test_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "Resize", "scale": (int(img_w), int(img_h)), "keep_ratio": True},
        {
            "type": "PackDetInputs",
            "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        },
    ]
    tta_pipeline = [
        {"type": "LoadImageFromFile"},
        {
            "type": "TestTimeAug",
            "transforms": [
                [
                    {"type": "Resize", "scale": tta_scales[0], "keep_ratio": True},
                    {"type": "Resize", "scale": tta_scales[1], "keep_ratio": True},
                    {"type": "Resize", "scale": tta_scales[2], "keep_ratio": True},
                ],
                [
                    {"type": "RandomFlip", "prob": 0.0, "direction": "horizontal"},
                    {"type": "RandomFlip", "prob": 1.0, "direction": "horizontal"},
                ],
                [
                    {
                        "type": "PackDetInputs",
                        "meta_keys": (
                            "img_id",
                            "img_path",
                            "ori_shape",
                            "img_shape",
                            "scale_factor",
                            "flip",
                            "flip_direction",
                        ),
                    }
                ],
            ],
        },
    ]

    dataroot = str(cfg.paths.project_root) + "/"

    if use_copy_paste:
        # CopyPaste is a mix-transform and requires MultiImageMixDataset.
        train_dataset: Dict[str, Any] = {
            "type": "MultiImageMixDataset",
            "dataset": {
                "type": "CocoDataset",
                "metainfo": {"classes": class_names},
                "data_root": dataroot,
                "ann_file": str(train_ann_path),
                "data_prefix": {"img": "data/images/train/"},
                "filter_cfg": {"filter_empty_gt": False, "min_size": 1},
                "pipeline": inner_pipeline,
            },
            "pipeline": train_pipeline,
        }
    else:
        # Without CopyPaste, train directly with CocoDataset to avoid unnecessary wrapper retries.
        train_dataset = {
            "type": "CocoDataset",
            "metainfo": {"classes": class_names},
            "data_root": dataroot,
            "ann_file": str(train_ann_path),
            "data_prefix": {"img": "data/images/train/"},
            "filter_cfg": {"filter_empty_gt": False, "min_size": 1},
            "pipeline": [
                {"type": "LoadImageFromFile"},
                {"type": "LoadAnnotations", "with_bbox": True, "with_mask": False},
                *common_aug_pipeline,
                {"type": "PackDetInputs"},
            ],
        }

    optim_wrapper: Dict[str, Any] = {
        "type": "OptimWrapper",
        "optimizer": {
            "type": "AdamW",
            "lr": float(cfg.train.learning_rate),
            "weight_decay": float(cfg.train.weight_decay),
        },
    }
    if cfg.model.architecture in {CASCADE_ARCH_SWIN_L, CASCADE_ARCH_CONVNEXT_XL}:
        optim_wrapper["paramwise_cfg"] = {
            "decay_rate": 0.9,
            "decay_type": "layer_wise",
            "num_layers": 12,
        }
    if cfg.train.use_amp:
        print("[cascade] AMP requested, but disabled for MMDet Cascade stability.")
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
        "tta_model": {
            "type": "DetTTAModel",
            "tta_cfg": {
                "nms": {"type": "nms", "iou_threshold": 0.5},
                "max_per_img": int(cfg.model.detections_per_img),
            },
        },
        "tta_pipeline": tta_pipeline,
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
            "base_batch_size": 4,
        },
    }

    print(f"[cascade] adaptive bbox losses -> {adaptive_loss_summary}")

    return runtime_cfg


def _patch_transformers_nn_nameerror() -> None:
    """
    Work around a known transformers import bug that can raise:
    NameError: name 'nn' is not defined
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
    use_copy_paste = _coco_has_instance_masks(coco)
    print(f"[cascade] classes={class_names} (num_classes={len(class_names)})")
    if not use_copy_paste:
        print("[cascade] segmentation masks not found in COCO annotations; CopyPaste disabled.")

    train_ann_path = cfg.paths.outputs_dir / "cascade_train_split.json"
    val_ann_path = cfg.paths.outputs_dir / "cascade_val_split.json"
    _write_coco(train_ann_path, _subset_coco(coco, train_ids))
    _write_coco(val_ann_path, _subset_coco(coco, val_ids))

    work_dir = cfg.paths.outputs_dir / "cascade_runs" / cfg.model.architecture
    work_dir.mkdir(parents=True, exist_ok=True)

    mmdet_cfg = _build_mmdet_cfg(
        cfg=cfg,
        coco=coco,
        train_ann_path=train_ann_path,
        val_ann_path=val_ann_path,
        work_dir=work_dir,
        class_names=class_names,
        use_copy_paste=use_copy_paste,
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