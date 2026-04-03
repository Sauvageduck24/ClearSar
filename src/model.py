from __future__ import annotations

from typing import Optional

import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

from src.config import ModelConfig


def apply_checkpoint_model_hints(cfg: ModelConfig, checkpoint_path: str) -> None:
    """
    Update model config from checkpoint metadata when available.

    This prevents architecture mismatches when loading checkpoints trained
    with a different backbone/profile.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return

    cfg_blob = checkpoint.get("config")
    if not isinstance(cfg_blob, dict):
        return

    model_blob = cfg_blob.get("model") if isinstance(cfg_blob.get("model"), dict) else None
    if not model_blob:
        return

    arch = model_blob.get("architecture")
    if isinstance(arch, str) and arch:
        cfg.architecture = arch

    weights = model_blob.get("pretrained_weights")
    if isinstance(weights, str) and weights:
        cfg.pretrained_weights = weights

    score_thresh = model_blob.get("score_thresh")
    if isinstance(score_thresh, (int, float)):
        cfg.score_thresh = float(score_thresh)

    nms_thresh = model_blob.get("nms_thresh")
    if isinstance(nms_thresh, (int, float)):
        cfg.nms_thresh = float(nms_thresh)

    detections_per_img = model_blob.get("detections_per_img")
    if isinstance(detections_per_img, int):
        cfg.detections_per_img = detections_per_img


def _extract_ssl_encoder_state(checkpoint: dict) -> dict:
    if "encoder_state_dict" in checkpoint and isinstance(checkpoint["encoder_state_dict"], dict):
        return checkpoint["encoder_state_dict"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    return checkpoint


def _normalize_ssl_key(key: str) -> str:
    prefixes = ("encoder.", "backbone.", "module.encoder.", "module.")
    for pref in prefixes:
        if key.startswith(pref):
            return key[len(pref):]
    return key


def _load_ssl_backbone_if_available(model: nn.Module, cfg: ModelConfig) -> None:
    if cfg.ssl_backbone_path is None:
        return

    ssl_path = str(cfg.ssl_backbone_path)
    checkpoint = torch.load(ssl_path, map_location="cpu")
    state = _extract_ssl_encoder_state(checkpoint)

    if cfg.architecture != "fasterrcnn_resnet50_fpn_v2":
        print(
            "[ssl] ssl_backbone_path was provided but architecture is not "
            "fasterrcnn_resnet50_fpn_v2. Skipping SSL backbone loading."
        )
        return

    target_state = model.backbone.body.state_dict()
    mapped = {}
    for k, v in state.items():
        nk = _normalize_ssl_key(str(k))
        if nk in target_state and target_state[nk].shape == v.shape:
            mapped[nk] = v

    if not mapped:
        print(f"[ssl] No compatible backbone keys found in {ssl_path}")
        return

    model.backbone.body.load_state_dict(mapped, strict=False)
    print(f"[ssl] Loaded {len(mapped)} backbone tensors from {ssl_path}")


def _configure_rpn_for_small_objects(model: nn.Module) -> None:
    # Use smaller anchors and match the RPN head to the new anchor count.
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    num_anchors = anchor_generator.num_anchors_per_location()[0]
    in_channels = model.backbone.out_channels

    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(in_channels, num_anchors)


def build_model(cfg: ModelConfig, device: Optional[torch.device] = None) -> nn.Module:
    """
    Build detection model.

    Current production path uses Faster R-CNN from torchvision because it is
    stable and integrates directly with COCO-style evaluation.
    """
    if cfg.architecture == "fasterrcnn_resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=cfg.pretrained_weights,
        )
    elif cfg.architecture == "fasterrcnn_mobilenet_v3_large_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=cfg.pretrained_weights,
        )
    else:
        raise ValueError(
            "Unsupported architecture. Use 'fasterrcnn_resnet50_fpn_v2' or "
            "'fasterrcnn_mobilenet_v3_large_fpn'."
        )

    _configure_rpn_for_small_objects(model)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)

    model.roi_heads.score_thresh = float(cfg.score_thresh)
    model.roi_heads.nms_thresh = float(cfg.nms_thresh)
    model.roi_heads.detections_per_img = int(cfg.detections_per_img)

    _load_ssl_backbone_if_available(model, cfg)

    if device is not None:
        model.to(device)
    return model


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
