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
    return tuple(dict.fromkeys(names))


def _analyze_bbox_distribution(coco: Dict[str, Any]) -> Dict[str, float]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        return {"num_boxes": 0.0, "small_ratio": 0.0, "elongated_ratio": 0.0, "large_ratio": 0.0}

    image_area_by_id: Dict[int, float] = {}
    for img in images:
        if not isinstance(img, dict):
            continue
        img_id = img.get("id")
        w, h = img.get("width"), img.get("height")
        if img_id is None:
            continue
        try:
            image_area_by_id[int(img_id)] = float(w) * float(h)
        except (TypeError, ValueError):
            continue

    rel_areas: List[float] = []
    elongations: List[float] = []

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            bw, bh = float(bbox[2]), float(bbox[3])
            img_id = int(ann.get("image_id"))
        except (TypeError, ValueError):
            continue
        if bw <= 0 or bh <= 0:
            continue
        img_area = image_area_by_id.get(img_id)
        if not img_area:
            continue
        rel_areas.append((bw * bh) / img_area)
        elongations.append(max(bw / bh, bh / bw))

    num_boxes = len(rel_areas)
    if num_boxes == 0:
        return {"num_boxes": 0.0, "small_ratio": 0.0, "elongated_ratio": 0.0, "large_ratio": 0.0,
                "small_thr": 0.0, "large_thr": 0.0, "elongated_thr": 0.0}

    def _q(vals: List[float], q: float) -> float:
        s = sorted(vals)
        return float(s[int((len(s) - 1) * max(0.0, min(1.0, q)))])

    small_thr = _q(rel_areas, 0.35)
    large_thr = _q(rel_areas, 0.95)
    elongated_thr = min(12.0, max(6.0, _q(elongations, 0.50)))

    return {
        "num_boxes": float(num_boxes),
        "small_ratio": sum(1 for x in rel_areas if x <= small_thr) / num_boxes,
        "elongated_ratio": sum(1 for x in elongations if x >= elongated_thr) / num_boxes,
        "large_ratio": sum(1 for x in rel_areas if x >= large_thr) / num_boxes,
        "small_thr": small_thr,
        "large_thr": large_thr,
        "elongated_thr": elongated_thr,
    }


def _build_adaptive_bbox_losses(
    coco: Dict[str, Any],
) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]], str]:
    stats = _analyze_bbox_distribution(coco)
    small_ratio = float(stats.get("small_ratio", 0.0))
    elongated_ratio = float(stats.get("elongated_ratio", 0.0))
    large_ratio = float(stats.get("large_ratio", 0.0))

    # FIX: RPN loss must be SmoothL1 — RPNHead predicts deltas (dx,dy,dw,dh), not decoded boxes.
    # GIoULoss/CIoULoss require absolute coordinates (reg_decoded_bbox=True) which RPNHead
    # does NOT support. Using them here causes NaN gradients or a runtime error.
    rpn_loss_bbox: Dict[str, Any] = {
        "type": "SmoothL1Loss",
        "loss_weight": 1.0,
        "beta": 1.0 / 9.0,
    }

    # FIX: loss_weight capped at 5.0 for RoI stages.
    # The previous formula produced weights of ~16 with elongated_ratio~0.80 (real value
    # for this dataset). That's 16x the cls_loss weight=1.0, which makes the model ignore
    # classification entirely — exactly the stagnation pattern observed in training.
    # Literature standard: 1.0–2.0, practical competition max: ~5.0.
    stage_loss_type = "CIoULoss" if elongated_ratio >= 0.20 else "GIoULoss"
    stage1_weight = round(min(5.0, 1.5 + 1.5 * small_ratio + 1.0 * elongated_ratio), 4)
    stage2_weight = round(min(5.0, 1.5 + 1.0 * small_ratio + 1.5 * elongated_ratio), 4)
    stage3_weight = round(min(4.0, 1.0 + 0.5 * elongated_ratio + 1.0 * large_ratio), 4)

    roi_stage_losses = (
        {"type": stage_loss_type, "loss_weight": stage1_weight},
        {"type": stage_loss_type, "loss_weight": stage2_weight},
        {"type": stage_loss_type, "loss_weight": stage3_weight},
    )

    summary = (
        f"small={small_ratio:.3f}, elongated={elongated_ratio:.3f}, large={large_ratio:.3f} | "
        f"rpn=SmoothL1@1.0 | "
        f"roi=[{stage_loss_type}@{stage1_weight}, {stage_loss_type}@{stage2_weight}, "
        f"{stage_loss_type}@{stage3_weight}]"
    )
    return rpn_loss_bbox, roi_stage_losses, summary


def _build_adaptive_rcnn_iou_thresholds(coco: Dict[str, Any]) -> Tuple[Tuple[float, float, float], str]:
    stats = _analyze_bbox_distribution(coco)
    small_ratio = float(stats.get("small_ratio", 0.0))
    elongated_ratio = float(stats.get("elongated_ratio", 0.0))
    large_ratio = float(stats.get("large_ratio", 0.0))

    relax = min(0.10, 0.06 * small_ratio + 0.05 * elongated_ratio)
    tighten = min(0.03, 0.05 * large_ratio)

    stage1 = max(0.40, min(0.55, 0.50 - relax + tighten))
    stage2 = max(stage1 + 0.08, min(0.65, 0.60 - 0.85 * relax + 0.8 * tighten))
    stage3 = max(stage2 + 0.08, min(0.75, 0.70 - 0.70 * relax + 0.9 * tighten))

    thresholds = (round(stage1, 3), round(stage2, 3), round(stage3, 3))
    summary = (
        f"small={small_ratio:.3f}, elongated={elongated_ratio:.3f}, large={large_ratio:.3f} | "
        f"rcnn_iou={thresholds}"
    )
    return thresholds, summary


def _build_adaptive_rpn_anchor_generator(coco: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    stats = _analyze_bbox_distribution(coco)
    small_ratio = float(stats.get("small_ratio", 0.0))
    elongated_ratio = float(stats.get("elongated_ratio", 0.0))

    # Keep priors for elongated SAR targets, but cap anchors-per-location to keep RPN fast.
    ratios: List[float] = [0.2, 0.5, 1.0, 2.0, 5.0]
    if elongated_ratio >= 0.30:
        ratios.append(10.0)

    # Prefer one scale for throughput; enable an extra small scale only when needed.
    scales = [4, 8] if small_ratio >= 0.45 else [8]

    anchor_generator = {
        "type": "AnchorGenerator",
        "scales": scales,
        "ratios": ratios,
        "strides": [4, 8, 16, 32, 64],
    }
    anchors_per_loc = len(scales) * len(ratios)
    summary = (
        f"small={small_ratio:.3f}, elongated={elongated_ratio:.3f} | "
        f"rpn_anchors=scales{tuple(scales)} xratios{tuple(ratios)} "
        f"(anchors_per_loc={anchors_per_loc})"
    )
    return anchor_generator, summary


def _register_cascade_progress_hook() -> bool:
    try:
        from mmengine.hooks import Hook
        from mmengine.registry import HOOKS
        from tqdm.auto import tqdm
    except Exception:
        return False

    @HOOKS.register_module(force=True)
    class CascadeProgressBarHook(Hook):
        priority = "LOW"

        def __init__(self, update_interval: int = 1) -> None:
            self.update_interval = max(1, int(update_interval))
            self._pbar = None
            self._last_step = 0

        def before_train_epoch(self, runner) -> None:
            total = len(runner.train_dataloader) if runner.train_dataloader is not None else None
            self._pbar = tqdm(
                total=total,
                desc=f"Epoch {runner.epoch + 1}/{runner.max_epochs}",
                dynamic_ncols=True,
                leave=True,
                unit="it",
            )
            self._last_step = 0

        def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
            if self._pbar is None:
                return
            current_step = int(batch_idx) + 1
            delta = current_step - self._last_step
            if delta >= self.update_interval:
                self._pbar.update(delta)
                self._last_step = current_step

            if isinstance(outputs, dict):
                loss_value = outputs.get("loss")
                if loss_value is not None:
                    try:
                        self._pbar.set_postfix(loss=f"{float(loss_value):.4f}", refresh=False)
                    except Exception:
                        pass

        def after_train_epoch(self, runner) -> None:
            if self._pbar is None:
                return
            if self._pbar.total is not None and self._last_step < self._pbar.total:
                self._pbar.update(self._pbar.total - self._last_step)
            self._pbar.close()
            self._pbar = None

        def after_run(self, runner) -> None:
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None

    return True


def _resolve_pretrained_checkpoint(arch: str, pretrained_weights: str) -> str:
    token = pretrained_weights.strip() if isinstance(pretrained_weights, str) else ""
    if token.upper() in {"NONE", "NULL", "FALSE", "NO"}:
        raise ValueError(f"All Cascade backbones require pretrained weights. Invalid: '{token}'.")
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
        backbone = {
            "type": "SwinTransformer",
            "embed_dims": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "window_size": 16,  # 512px → stage1 feature=128px; 128/16=8 ✓ (was 12, 128%12≠0)
            "mlp_ratio": 4,
            "qkv_bias": True,
            "drop_path_rate": 0.10,
            "patch_norm": True,
            "out_indices": (0, 1, 2, 3),
            "with_cp": False,
            "convert_weights": True,
            "init_cfg": {"type": "Pretrained", "checkpoint": pretrained_ckpt},
        }
        # FPN (not PAFPN): PAFPN adds a second bottom-up pass, ~30-40% slower per epoch
        # with marginal benefit for ~514px images. FPN is the right speed/quality trade-off.
        neck = {"type": "FPN", "in_channels": [192, 384, 768, 1536], "out_channels": 256, "num_outs": 5}
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
            "init_cfg": {"type": "Pretrained", "checkpoint": pretrained_ckpt, "prefix": "backbone"},
        }
        neck = {"type": "FPN", "in_channels": [256, 512, 1024, 2048], "out_channels": 256, "num_outs": 5}
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    if arch in {CASCADE_ARCH_RESNET50, CASCADE_ARCH_RESNET101}:
        depth = 50 if arch == CASCADE_ARCH_RESNET50 else 101
        backbone = {
            "type": "ResNet", "depth": depth, "num_stages": 4,
            "out_indices": (0, 1, 2, 3), "frozen_stages": 1,
            "norm_cfg": {"type": "BN", "requires_grad": True}, "norm_eval": True,
            "style": "pytorch",
            "init_cfg": {"type": "Pretrained", "checkpoint": pretrained_ckpt},
        }
        neck = {"type": "FPN", "in_channels": [256, 512, 1024, 2048], "out_channels": 256, "num_outs": 5}
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    if arch == CASCADE_ARCH_DCNV2:
        backbone = {
            "type": "ResNet", "depth": 50, "num_stages": 4,
            "out_indices": (0, 1, 2, 3), "frozen_stages": 1,
            "norm_cfg": {"type": "BN", "requires_grad": True}, "norm_eval": True,
            "style": "pytorch",
            "dcn": {"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
            "stage_with_dcn": (False, True, True, True),
            "init_cfg": {"type": "Pretrained", "checkpoint": pretrained_ckpt},
        }
        neck = {"type": "FPN", "in_channels": [256, 512, 1024, 2048], "out_channels": 256, "num_outs": 5}
        _ensure_pretrained_backbone(arch, backbone)
        return backbone, neck

    if arch == CASCADE_ARCH_HRNET:
        backbone = {
            "type": "HRNet",
            "extra": {
                "stage1": {"num_modules": 1, "num_branches": 1, "block": "BOTTLENECK",
                           "num_blocks": (4,), "num_channels": (64,)},
                "stage2": {"num_modules": 1, "num_branches": 2, "block": "BASIC",
                           "num_blocks": (4, 4), "num_channels": (40, 80)},
                "stage3": {"num_modules": 4, "num_branches": 3, "block": "BASIC",
                           "num_blocks": (4, 4, 4), "num_channels": (40, 80, 160)},
                "stage4": {"num_modules": 3, "num_branches": 4, "block": "BASIC",
                           "num_blocks": (4, 4, 4, 4), "num_channels": (40, 80, 160, 320)},
            },
            "init_cfg": {"type": "Pretrained", "checkpoint": pretrained_ckpt},
        }
        neck = {"type": "HRFPN", "in_channels": [40, 80, 160, 320], "out_channels": 256, "num_outs": 5}
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
    use_progress_bar: bool = False,
) -> Dict[str, Any]:
    if cfg.train.image_size is None:
        img_h, img_w = (1024, 1024)
    else:
        img_h, img_w = cfg.train.image_size

    backbone, neck = _build_backbone_and_neck(cfg.model.architecture, cfg.model.pretrained_weights)
    bbox_stats = _analyze_bbox_distribution(coco)
    rpn_loss_bbox, roi_stage_losses, adaptive_loss_summary = _build_adaptive_bbox_losses(coco)
    rcnn_iou_thresholds, adaptive_rcnn_summary = _build_adaptive_rcnn_iou_thresholds(coco)
    adaptive_anchor_generator, adaptive_anchor_summary = _build_adaptive_rpn_anchor_generator(coco)
    stage1_iou, stage2_iou, stage3_iou = rcnn_iou_thresholds
    elongated_ratio = float(bbox_stats.get("elongated_ratio", 0.0))

    num_fg_classes = len(class_names)
    warmup_end = min(5, max(3, int(cfg.train.epochs // 8)))
    speed_mode = "fast"

    if speed_mode == "fast":
        # Keep proposal counts bounded to avoid expensive RPN/ROI stages with little AP gain.
        train_rpn_nms_pre = 1200
        test_rpn_nms_pre = 1500
        max_train_proposals = min(int(cfg.model.detections_per_img), 300)
        max_test_dets = min(int(cfg.model.detections_per_img), 300)
    else:
        train_rpn_nms_pre = 2500
        test_rpn_nms_pre = 2500
        max_train_proposals = int(cfg.model.detections_per_img)
        max_test_dets = int(cfg.model.detections_per_img)

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
            "anchor_generator": adaptive_anchor_generator,
            "bbox_coder": {
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [1.0, 1.0, 1.0, 1.0],
            },
            "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": True, "loss_weight": 1.0},
            "loss_bbox": rpn_loss_bbox,  # SmoothL1 — see _build_adaptive_bbox_losses
        },
        "roi_head": {
            "type": "CascadeRoIHead",
            "num_stages": 3,
            "stage_loss_weights": [1.0, 1.0, 1.0],
            "bbox_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {"type": "RoIAlign", "output_size": (14, 14), "sampling_ratio": 2},
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32, 64],
                # finest_scale=56: boxes with sqrt(area)<56 → P2 (stride 4).
                # Thin bands 119×9=1071px² → sqrt=32 < 56 → P2 ✓
                "finest_scale": 56,
            },
            "bbox_head": [
                {
                    "type": "Shared4Conv1FCBBoxHead",
                    "in_channels": 256, "fc_out_channels": 1024,
                    "roi_feat_size": (14, 14), "num_classes": num_fg_classes,
                    "reg_decoded_bbox": True,
                    "bbox_coder": {"type": "DeltaXYWHBBoxCoder",
                                   "target_means": [0.0, 0.0, 0.0, 0.0],
                                   "target_stds": [0.1, 0.1, 0.2, 0.2]},
                    "reg_class_agnostic": True,
                    "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 1.0},
                    "loss_bbox": roi_stage_losses[0],
                },
                {
                    "type": "Shared4Conv1FCBBoxHead",
                    "in_channels": 256, "fc_out_channels": 1024,
                    "roi_feat_size": (14, 14), "num_classes": num_fg_classes,
                    "reg_decoded_bbox": True,
                    "bbox_coder": {"type": "DeltaXYWHBBoxCoder",
                                   "target_means": [0.0, 0.0, 0.0, 0.0],
                                   "target_stds": [0.05, 0.05, 0.1, 0.1]},
                    "reg_class_agnostic": True,
                    "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 1.0},
                    "loss_bbox": roi_stage_losses[1],
                },
                {
                    "type": "Shared4Conv1FCBBoxHead",
                    "in_channels": 256, "fc_out_channels": 1024,
                    "roi_feat_size": (14, 14), "num_classes": num_fg_classes,
                    "reg_decoded_bbox": True,
                    "bbox_coder": {"type": "DeltaXYWHBBoxCoder",
                                   "target_means": [0.0, 0.0, 0.0, 0.0],
                                   "target_stds": [0.033, 0.033, 0.067, 0.067]},
                    "reg_class_agnostic": True,
                    "loss_cls": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 1.0},
                    "loss_bbox": roi_stage_losses[2],
                },
            ],
        },
        "train_cfg": {
            "rpn": {
                "assigner": {
                    # FIX: Reverted from ATSSAssigner to MaxIoUAssigner.
                    # ATSSAssigner is designed for anchor-free one-stage detectors (FCOS/ATSS)
                    # and assumes GT centers fall within anchor regions per FPN level.
                    # For Cascade R-CNN's RPN with fixed anchors, ATSS can mis-assign
                    # horizontal thin bands (ar>20) that span multiple FPN levels.
                    # MaxIoUAssigner + match_low_quality=True guarantees every GT gets
                    # at least one positive anchor even below pos_iou_thr.
                    "type": "MaxIoUAssigner",
                    "pos_iou_thr": 0.5,
                    "neg_iou_thr": 0.3,
                    "min_pos_iou": 0.3,
                    "match_low_quality": True,
                    "ignore_iof_thr": -1,
                },
                "sampler": {
                    "type": "RandomSampler",
                    "num": 256, "pos_fraction": 0.5,
                    "neg_pos_ub": -1, "add_gt_as_proposals": False,
                },
                "allowed_border": -1,
                "pos_weight": -1,
                "debug": False,
            },
            "rpn_proposal": {
                "nms_pre": train_rpn_nms_pre,
                "max_per_img": max_train_proposals,
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "min_bbox_size": 0,
            },
            "rcnn": [
                {
                    "assigner": {"type": "MaxIoUAssigner",
                                 "pos_iou_thr": stage1_iou, "neg_iou_thr": stage1_iou,
                                 "min_pos_iou": stage1_iou, "match_low_quality": False,
                                 "ignore_iof_thr": -1},
                    "sampler": {"type": "RandomSampler", "num": 512, "pos_fraction": 0.25,
                                "neg_pos_ub": -1, "add_gt_as_proposals": True},
                    "pos_weight": -1, "debug": False,
                },
                {
                    "assigner": {"type": "MaxIoUAssigner",
                                 "pos_iou_thr": stage2_iou, "neg_iou_thr": stage2_iou,
                                 "min_pos_iou": stage2_iou, "match_low_quality": False,
                                 "ignore_iof_thr": -1},
                    "sampler": {"type": "RandomSampler", "num": 512, "pos_fraction": 0.25,
                                "neg_pos_ub": -1, "add_gt_as_proposals": True},
                    "pos_weight": -1, "debug": False,
                },
                {
                    "assigner": {"type": "MaxIoUAssigner",
                                 "pos_iou_thr": stage3_iou, "neg_iou_thr": stage3_iou,
                                 "min_pos_iou": stage3_iou, "match_low_quality": False,
                                 "ignore_iof_thr": -1},
                    "sampler": {"type": "RandomSampler", "num": 512, "pos_fraction": 0.25,
                                "neg_pos_ub": -1, "add_gt_as_proposals": True},
                    "pos_weight": -1, "debug": False,
                },
            ],
        },
        "test_cfg": {
            "rpn": {
                "nms_pre": test_rpn_nms_pre,
                "max_per_img": max_test_dets,
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "min_bbox_size": 0,
            },
            "rcnn": {
                "score_thr": 0.05,
                "nms": {"type": "soft_nms", "iou_threshold": 0.5, "min_score": 0.001},
                "max_per_img": max_test_dets,
            },
        },
    }

    if speed_mode == "fast":
        multiscale_train_scales = [
            (int(img_w), int(img_h)),
            (int(img_w * 1.1), int(img_h * 1.1)),
        ]
    else:
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

    common_aug = [
        {"type": "RandomChoiceResize", "scales": multiscale_train_scales, "keep_ratio": True},
        {"type": "RandomFlip", "prob": 0.5, "direction": ["horizontal"]},
        {"type": "RandomFlip", "prob": 0.5, "direction": ["vertical"]},
    ]
    if speed_mode == "quality":
        common_aug.extend([
            {"type": "RandomShift", "max_shift_px": 32},
            {"type": "PhotoMetricDistortion",
             "brightness_delta": 20, "contrast_range": (0.8, 1.2),
             "saturation_range": (1.0, 1.0), "hue_delta": 0},
        ])

    # Keep a single-image pipeline for stability with bbox-only annotations.
    # CopyPaste on MultiImageMixDataset can be brittle depending on sample shapes/types.
    train_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "with_bbox": True},
        *common_aug,
        {"type": "PackDetInputs"},
    ]

    train_dataset = {
        "type": "CocoDataset",
        "metainfo": {"classes": class_names},
        "data_root": str(cfg.paths.project_root) + "/",
        "ann_file": str(train_ann_path),
        "data_prefix": {"img": "data/images/train/"},
        "filter_cfg": {"filter_empty_gt": True, "min_size": 1},
        "pipeline": train_pipeline,
    }

    test_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "Resize", "scale": (int(img_w), int(img_h)), "keep_ratio": True},
        {"type": "PackDetInputs",
         "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")},
    ]
    tta_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "TestTimeAug", "transforms": [
            [{"type": "Resize", "scale": s, "keep_ratio": True} for s in tta_scales],
            [{"type": "RandomFlip", "prob": 0.0, "direction": "horizontal"},
             {"type": "RandomFlip", "prob": 1.0, "direction": "horizontal"}],
            [{"type": "PackDetInputs",
              "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape",
                            "scale_factor", "flip", "flip_direction")}],
        ]},
    ]

    # AMP can fail in Cascade RPN proposal generation due to a known dtype mismatch
    # in batched_nms (Half destination vs Float source) on some mmcv/mmdet builds.
    requested_amp = bool(getattr(cfg.train, "use_amp", False))
    use_amp = False
    if requested_amp:
        print("[cascade] AMP requested but disabled for stability (fp16 NMS dtype mismatch).")

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
    if cfg.train.grad_clip_norm is not None:
        optim_wrapper["clip_grad"] = {
            "max_norm": float(cfg.train.grad_clip_norm),
            "norm_type": 2,
        }

    dist_backend = "gloo" if os.name == "nt" else "nccl"
    mp_start_method = "spawn" if os.name == "nt" else "fork"
    dataroot = str(cfg.paths.project_root) + "/"

    runtime_cfg: Dict[str, Any] = {
        "default_scope": "mmdet",
        "custom_imports": {"imports": ["mmpretrain.models"], "allow_failed_imports": False},
        "work_dir": str(work_dir),
        "model": model,
        "tta_model": {
            "type": "DetTTAModel",
            "tta_cfg": {"nms": {"type": "nms", "iou_threshold": 0.5},
                        "max_per_img": max_test_dets},
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
        "val_evaluator": {"type": "CocoMetric", "ann_file": str(val_ann_path),
                          "metric": "bbox", "format_only": False},
        "test_evaluator": {"type": "CocoMetric", "ann_file": str(val_ann_path),
                           "metric": "bbox", "format_only": False},
        "train_cfg": {"type": "EpochBasedTrainLoop",
                      "max_epochs": int(cfg.train.epochs), "val_interval": 1},
        "val_cfg": {"type": "ValLoop"},
        "test_cfg": {"type": "TestLoop"},
        "optim_wrapper": optim_wrapper,
        "param_scheduler": [
            {"type": "LinearLR", "start_factor": 0.1, "by_epoch": True,
             "begin": 0, "end": warmup_end},
            {"type": "CosineAnnealingLR", "eta_min": 1e-6, "by_epoch": True,
             "begin": warmup_end, "end": int(cfg.train.epochs),
             "T_max": int(cfg.train.epochs) - warmup_end},
        ],
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook", "interval": 1,
                "max_keep_ckpts": int(cfg.train.save_top_k),
                "save_best": "coco/bbox_mAP", "rule": "greater",
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
        "auto_scale_lr": {"enable": False, "base_batch_size": 4},
    }

    if use_progress_bar:
        runtime_cfg["custom_hooks"] = [{"type": "CascadeProgressBarHook", "update_interval": 1}]

    print(f"[cascade] adaptive bbox losses   -> {adaptive_loss_summary}")
    print(f"[cascade] adaptive rcnn assigners -> {adaptive_rcnn_summary}")
    print(f"[cascade] adaptive rpn anchors    -> {adaptive_anchor_summary}")
    print(f"[cascade] speed mode              -> {speed_mode}")
    print(
        f"[cascade] proposals train/test    -> "
        f"nms_pre={train_rpn_nms_pre}/{test_rpn_nms_pre}, max={max_train_proposals}/{max_test_dets}"
    )
    print(f"[cascade] progress bar            -> {'enabled' if use_progress_bar else 'disabled'}")
    print(f"[cascade] AMP={'enabled (fp16)' if use_amp else 'disabled'}")

    return runtime_cfg


def _patch_transformers_nn_nameerror() -> None:
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
                "Try pinning transformers to a stable version (e.g. 4.44.x)."
            ) from exc
        raise

    progress_hook_enabled = _register_cascade_progress_hook()
    if not progress_hook_enabled:
        print("[cascade] tqdm no disponible: se mantiene logging clasico.")

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
        coco=coco,
        train_ann_path=train_ann_path,
        val_ann_path=val_ann_path,
        work_dir=work_dir,
        class_names=class_names,
        use_progress_bar=progress_hook_enabled,
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