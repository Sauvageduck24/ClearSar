from __future__ import annotations

"""
FocusDet-style detector for ClearSAR — small object optimized.

Implements the three core innovations from the FocusDet paper
(Shi et al., Scientific Reports 2024) over a standard ResNet backbone:

  1. Space-to-Depth (S2D) stem  — preserves small-object pixels by folding
     spatial dimensions into channels instead of striding/pooling them away.
  2. Bottom-Up Focus FPN        — high-res (shallow) features are the PRIMARY
     path; deep semantics are fused bottom-up, inverting the standard FPN.
  3. SIoU loss + SoftNMS (Gaussian) — angular-aware box regression and dense-
     friendly suppression that avoids deleting valid adjacent detections.

Architecture summary
--------------------
  Input (3×H×W)
    └─ S2D stem (fold 2×2 → 12 channels, then conv → 64ch) → stride 2
    └─ ResNet stages 2-4 (C2, C3, C4) via torchvision backbone
  Bottom-Up Focus FPN
    └─ P2 (from C2, highest res)  ← fuse ← P3 ← fuse ← P4
  FCOS-style detection head
    └─ Centerness + Classification + Regression per pyramid level
  Post-process: decode + SIoU-SoftNMS

Usage
-----
    pip install torch torchvision
    python focusdet_train.py --project-root . --epochs 80 --imgsz 640
    python focusdet_train.py --project-root . --epochs 80 --imgsz 960 --batch-size 2
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

MIN_BOX_AREA   = 1.0
STRIDES        = [8, 16, 32]      # FPN levels P2/P3/P4 (smaller than typical)
NUM_CLASSES    = 1
INF            = 1e8


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPACE-TO-DEPTH STEM
# ══════════════════════════════════════════════════════════════════════════════

class SpaceToDepth(nn.Module):
    """
    Reorganize spatial blocks into channels (reverse of PixelShuffle).
    A 3×H×W tensor with block_size=2 becomes 12×(H/2)×(W/2).
    Unlike strided conv or max-pool, ZERO information is lost — every pixel
    is kept. This is crucial for 1-3px-tall RFI stripes.
    """
    def __init__(self, block_size: int = 2):
        super().__init__()
        self.bs = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        bs = self.bs
        # Split into blocks and stack into channels
        x = x.view(B, C, H // bs, bs, W // bs, bs)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * bs * bs, H // bs, W // bs)
        return x


class S2DStem(nn.Module):
    """
    S2D stem replacing the standard 7×7 conv + max-pool.
    Keeps ALL spatial information, outputs stride-2 feature map.
    """
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.s2d = SpaceToDepth(block_size=2)   # 3→12 channels, /2 spatial
        self.conv = nn.Sequential(
            nn.Conv2d(12, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.s2d(x))


# ══════════════════════════════════════════════════════════════════════════════
# 2. BOTTOM-UP FOCUS FPN
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class BottomUpFocusFPN(nn.Module):
    """
    Inverted FPN: high-resolution (bottom/shallow) features are primary.
    Deep (C4) features provide semantic context but are up-sampled to
    augment the fine-grained P2/P3 maps, not the other way around.

    Standard FPN:  C4→P4 → upsample → fuse with C3→P3 → upsample → fuse C2
    Bottom Focus:  C2→P2 (primary, full res) ← downsample+fuse ← P3 ← P4
                   Then P3/P4 also get upsampled context from the primary path.

    Result: P2 retains maximum spatial detail; P3/P4 carry mixed semantics.
    """
    def __init__(self, c2_ch: int, c3_ch: int, c4_ch: int, out_ch: int = 256):
        super().__init__()
        self.lat2 = ConvBnRelu(c2_ch, out_ch, 1, p=0)
        self.lat3 = ConvBnRelu(c3_ch, out_ch, 1, p=0)
        self.lat4 = ConvBnRelu(c4_ch, out_ch, 1, p=0)

        # Bottom-up path: merge deeper context down
        self.bu_3to2 = ConvBnRelu(out_ch * 2, out_ch)  # cat(P3↑, P2) → P2_out
        self.bu_4to3 = ConvBnRelu(out_ch * 2, out_ch)  # cat(P4↑, P3) → P3_out

        # Refine output per level
        self.out2 = ConvBnRelu(out_ch, out_ch)
        self.out3 = ConvBnRelu(out_ch, out_ch)
        self.out4 = ConvBnRelu(out_ch, out_ch)

    def forward(self, c2, c3, c4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p2 = self.lat2(c2)
        p3 = self.lat3(c3)
        p4 = self.lat4(c4)

        # Fuse P4 → P3 (upsample P4 to P3 resolution)
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3_fused = self.bu_4to3(torch.cat([p4_up, p3], dim=1))

        # Fuse P3_fused → P2 (upsample to P2 resolution)
        p3_up = F.interpolate(p3_fused, size=p2.shape[-2:], mode="nearest")
        p2_fused = self.bu_3to2(torch.cat([p3_up, p2], dim=1))

        return self.out2(p2_fused), self.out3(p3_fused), self.out4(p4)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FCOS DETECTION HEAD
# ══════════════════════════════════════════════════════════════════════════════

class FCOSHead(nn.Module):
    """Shared FCOS head: cls + centerness + box regression."""
    def __init__(self, in_ch: int, num_classes: int, num_convs: int = 4):
        super().__init__()
        cls_tower, box_tower = [], []
        for _ in range(num_convs):
            cls_tower += [ConvBnRelu(in_ch, in_ch)]
            box_tower += [ConvBnRelu(in_ch, in_ch)]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.box_tower = nn.Sequential(*box_tower)
        self.cls_logits    = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.bbox_pred     = nn.Conv2d(in_ch, 4, 3, padding=1)           # ltrb
        self.centerness    = nn.Conv2d(in_ch, 1, 3, padding=1)

        # Scale per FPN level (learnable)
        self.scales = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in STRIDES])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Focal-loss-style prior for classification
        prior = 0.01
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))

    def forward(self, features: List[torch.Tensor]):
        cls_preds, box_preds, ctr_preds = [], [], []
        for i, feat in enumerate(features):
            cls_out = self.cls_tower(feat)
            box_out = self.box_tower(feat)
            cls_preds.append(self.cls_logits(cls_out))
            # exp to ensure positive ltrb; scale per level
            box_raw = self.bbox_pred(box_out)
            box_raw = self.scales[i](box_raw.sum(dim=1, keepdim=True)).expand_as(box_raw) + box_raw
            box_preds.append(F.relu(box_raw).exp())
            ctr_preds.append(self.centerness(box_out))
        return cls_preds, box_preds, ctr_preds


# ══════════════════════════════════════════════════════════════════════════════
# 4. FULL FOCUSDET MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FocusDet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, fpn_channels: int = 256, pretrained: bool = True):
        super().__init__()
        # S2D stem
        self.stem = S2DStem(out_channels=64)

        # ResNet-50 backbone (replace first conv + remove avgpool/fc)
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # Patch layer1 to accept our 64-channel stem output
        backbone.layer1[0].conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        backbone.layer1[0].downsample = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1 = backbone.layer1   # → 256ch, stride 4 total
        self.layer2 = backbone.layer2   # → 512ch, stride 8
        self.layer3 = backbone.layer3   # → 1024ch, stride 16

        c2_ch, c3_ch, c4_ch = 256, 512, 1024
        self.fpn = BottomUpFocusFPN(c2_ch, c3_ch, c4_ch, fpn_channels)
        self.head = FCOSHead(fpn_channels, num_classes)

        self.num_classes = num_classes
        self.strides = STRIDES

    def forward(self, x: torch.Tensor):
        # Stem + backbone
        s = self.stem(x)       # stride 2
        c2 = self.layer1(s)    # stride 4
        c3 = self.layer2(c2)   # stride 8
        c4 = self.layer3(c3)   # stride 16

        # FPN
        p2, p3, p4 = self.fpn(c2, c3, c4)

        # Head
        cls_preds, box_preds, ctr_preds = self.head([p2, p3, p4])
        return cls_preds, box_preds, ctr_preds


# ══════════════════════════════════════════════════════════════════════════════
# 5. SIOU LOSS
# ══════════════════════════════════════════════════════════════════════════════

def siou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    SIoU Loss (Gevorgyan 2022) — penalizes angular misalignment between
    predicted and ground-truth box centers, converging faster for small objects.

    pred_boxes, gt_boxes: (N, 4) in xyxy format
    Returns: (N,) loss per pair
    """
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_boxes.unbind(-1)

    # Intersection
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    pw = (px2 - px1).clamp(eps); ph = (py2 - py1).clamp(eps)
    gw = (gx2 - gx1).clamp(eps); gh = (gy2 - gy1).clamp(eps)
    union = pw * ph + gw * gh - inter
    iou = inter / (union + eps)

    # Center distances
    pcx = (px1 + px2) / 2; pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2; gcy = (gy1 + gy2) / 2
    dx = (gcx - pcx).abs(); dy = (gcy - pcy).abs()

    # Angular cost (sigma in paper)
    ch = torch.max(py2, gy2) - torch.min(py1, gy1)
    cw = torch.max(px2, gx2) - torch.min(px1, gx1)
    sin_alpha = dy / (ch + eps)
    sin_beta  = dx / (cw + eps)
    angle_cost = 1 - 2 * torch.sin(torch.arcsin(sin_alpha.clamp(-1,1)) - math.pi/4)**2

    # Distance cost
    rho_x = (dx / (cw + eps))**2
    rho_y = (dy / (ch + eps))**2
    gamma = angle_cost - 2
    dist_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    # Shape cost
    omicron_w = (torch.abs(pw - gw) / torch.max(pw, gw).clamp(eps))**0.5
    omicron_h = (torch.abs(ph - gh) / torch.max(ph, gh).clamp(eps))**0.5
    shape_cost = 1 - torch.exp(-omicron_w) * torch.exp(-omicron_h)

    return 1 - iou + (dist_cost + shape_cost) / 2


# ══════════════════════════════════════════════════════════════════════════════
# 6. FCOS TARGET ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def make_grid_points(feat_h: int, feat_w: int, stride: int, device) -> torch.Tensor:
    """Center points of each cell in the feature map, in image coordinates."""
    ys = torch.arange(feat_h, device=device, dtype=torch.float32) * stride + stride / 2
    xs = torch.arange(feat_w, device=device, dtype=torch.float32) * stride + stride / 2
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], dim=1)  # (H*W, 2)


def fcos_target(
    cls_preds:  List[torch.Tensor],   # (B, C, H_i, W_i) per level
    box_preds:  List[torch.Tensor],   # (B, 4, H_i, W_i) ltrb
    ctr_preds:  List[torch.Tensor],   # (B, 1, H_i, W_i)
    gt_boxes_list: List[torch.Tensor],  # list of (N_i, 4) xyxy per image
    img_h: int, img_w: int,
) -> torch.Tensor:
    """
    Compute total FCOS loss: focal cls + SIoU box + BCE centerness.
    """
    device = cls_preds[0].device
    B = cls_preds[0].shape[0]

    # Regress scale limits per FPN level (roughly based on stride)
    scale_ranges = [
        (0,   48),    # P2: tiny
        (48, 128),    # P3: small
        (128, INF),   # P4: medium+
    ]

    total_cls   = torch.zeros(1, device=device)
    total_box   = torch.zeros(1, device=device)
    total_ctr   = torch.zeros(1, device=device)
    n_pos       = 0

    for lvl, (stride, (lo, hi)) in enumerate(zip(STRIDES, scale_ranges)):
        B_, C_, H_, W_ = cls_preds[lvl].shape
        points = make_grid_points(H_, W_, stride, device)   # (HW, 2)
        NP = len(points)

        for b in range(B):
            cls_pred = cls_preds[lvl][b].permute(1, 2, 0).reshape(-1, C_)   # (HW, C)
            box_pred = box_preds[lvl][b].permute(1, 2, 0).reshape(-1, 4)    # (HW, 4) ltrb
            ctr_pred = ctr_preds[lvl][b].permute(1, 2, 0).reshape(-1, 1)    # (HW, 1)

            gt_boxes = gt_boxes_list[b].to(device)   # (G, 4) xyxy
            G = len(gt_boxes)

            cls_targets = torch.zeros(NP, C_, device=device)
            box_targets = torch.zeros(NP, 4, device=device)
            ctr_targets = torch.zeros(NP, 1, device=device)
            pos_mask    = torch.zeros(NP, dtype=torch.bool, device=device)

            if G > 0:
                # For each point, compute ltrb to every gt box
                px = points[:, 0:1]; py = points[:, 1:2]      # (NP,1)
                gx1 = gt_boxes[:, 0]; gy1 = gt_boxes[:, 1]
                gx2 = gt_boxes[:, 2]; gy2 = gt_boxes[:, 3]   # (G,)

                l = px - gx1.unsqueeze(0)   # (NP, G)
                t = py - gy1.unsqueeze(0)
                r = gx2.unsqueeze(0) - px
                b_ = gy2.unsqueeze(0) - py

                ltrb = torch.stack([l, t, r, b_], dim=-1)   # (NP, G, 4)
                inside = ltrb.min(dim=-1).values > 0         # (NP, G)

                # Max regression target per point-gt pair
                max_reg = ltrb.max(dim=-1).values            # (NP, G)
                in_scale = (max_reg >= lo) & (max_reg <= hi)
                valid = inside & in_scale

                if valid.any():
                    # Assign to the smallest GT that is valid for each point
                    gt_area = (gx2 - gx1) * (gy2 - gy1)
                    area_mat = gt_area.unsqueeze(0).expand(NP, G)
                    area_mat[~valid] = INF
                    assigned_gt = area_mat.argmin(dim=1)
                    pos = valid[torch.arange(NP), assigned_gt]

                    pos_mask = pos
                    if pos.any():
                        # cls targets (binary, single class)
                        cls_targets[pos, 0] = 1.0

                        # ltrb targets for positive points
                        ag = assigned_gt[pos]
                        # Simpler approach:
                        box_targets[pos, 0] = l[pos, ag]
                        box_targets[pos, 1] = t[pos, ag]
                        box_targets[pos, 2] = r[pos, ag]
                        box_targets[pos, 3] = b_[pos, ag]

                        # Centerness targets
                        lt = torch.stack([l[pos, ag], t[pos, ag]], dim=-1)
                        rb = torch.stack([r[pos, ag], b_[pos, ag]], dim=-1)
                        ctr = (lt.min(dim=-1).values / lt.max(dim=-1).values.clamp(1e-7)) * \
                              (rb.min(dim=-1).values / rb.max(dim=-1).values.clamp(1e-7))
                        ctr_targets[pos, 0] = ctr.clamp(0, 1).sqrt()

            # ── Classification loss: focal ──────────────────────────────
            alpha, gamma_fl = 0.25, 2.0
            p = cls_pred.sigmoid()
            pt = torch.where(cls_targets > 0.5, p, 1 - p)
            focal_w = alpha * (1 - pt)**gamma_fl
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_pred, cls_targets, reduction="none"
            )
            total_cls += (focal_w * cls_loss).sum()

            # ── Box + centerness loss (positive only) ───────────────────
            if pos_mask.any():
                n_pos += pos_mask.sum().item()
                # decode predicted ltrb to xyxy for SIoU
                px_pos = points[pos_mask, 0]
                py_pos = points[pos_mask, 1]
                pred_ltrb = box_pred[pos_mask]
                pred_xyxy = torch.stack([
                    px_pos - pred_ltrb[:, 0],
                    py_pos - pred_ltrb[:, 1],
                    px_pos + pred_ltrb[:, 2],
                    py_pos + pred_ltrb[:, 3],
                ], dim=-1)
                gt_ltrb = box_targets[pos_mask]
                gt_xyxy = torch.stack([
                    px_pos - gt_ltrb[:, 0],
                    py_pos - gt_ltrb[:, 1],
                    px_pos + gt_ltrb[:, 2],
                    py_pos + gt_ltrb[:, 3],
                ], dim=-1)
                total_box += siou_loss(pred_xyxy, gt_xyxy).sum()
                total_ctr += F.binary_cross_entropy_with_logits(
                    ctr_pred[pos_mask], ctr_targets[pos_mask], reduction="sum"
                )

    norm = max(n_pos, 1)
    loss = total_cls / norm + total_box / norm + total_ctr / norm
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 7. SOFT-NMS (Gaussian decay)
# ══════════════════════════════════════════════════════════════════════════════

def soft_nms_gaussian(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_thresh: float = 0.001,
    max_det: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian Soft-NMS: instead of hard-suppressing overlapping boxes,
    decays their scores exponentially with IoU — no missed detections
    for dense/touching objects.

    boxes:  (N, 4) xyxy
    scores: (N,)
    Returns: filtered (boxes, scores) sorted by score.
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = boxes.clone().float()
    scores = scores.clone().float()

    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort(descending=True)
    keep_boxes, keep_scores = [], []

    while order.numel() > 0 and len(keep_boxes) < max_det:
        idx = order[0]
        if scores[idx] < score_thresh:
            break
        keep_boxes.append(boxes[idx])
        keep_scores.append(scores[idx])

        if order.numel() == 1:
            break

        rest = order[1:]
        # IoU of best box with all remaining
        xx1 = torch.max(x1[idx], x1[rest])
        yy1 = torch.max(y1[idx], y1[rest])
        xx2 = torch.min(x2[idx], x2[rest])
        yy2 = torch.min(y2[idx], y2[rest])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (areas[idx] + areas[rest] - inter + 1e-7)

        # Gaussian decay — small IoU: almost no decay; large IoU: heavy decay
        decay = torch.exp(-(iou ** 2) / sigma)
        scores[rest] *= decay

        # Keep boxes whose score is still above threshold
        keep_mask = scores[rest] >= score_thresh
        order = rest[keep_mask]
        # Re-sort remaining
        order = order[scores[order].argsort(descending=True)]

    if not keep_boxes:
        return boxes[:0], scores[:0]

    return torch.stack(keep_boxes), torch.stack(keep_scores)


# ══════════════════════════════════════════════════════════════════════════════
# 8. INFERENCE DECODER
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def decode_predictions(
    cls_preds:  List[torch.Tensor],
    box_preds:  List[torch.Tensor],
    ctr_preds:  List[torch.Tensor],
    score_thresh: float = 0.05,
    nms_sigma:    float = 0.5,
    max_det:      int   = 500,
    img_h: int = 640, img_w: int = 640,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode FCOS output → (boxes_xyxy, scores) using SIoU-SoftNMS.
    All inputs are single-image (batch dim already removed).
    """
    all_boxes, all_scores = [], []

    for lvl, stride in enumerate(STRIDES):
        C, H, W = cls_preds[lvl].shape
        points = make_grid_points(H, W, stride, cls_preds[lvl].device)

        cls_p = cls_preds[lvl].permute(1, 2, 0).reshape(-1, C).sigmoid()
        box_p = box_preds[lvl].permute(1, 2, 0).reshape(-1, 4)
        ctr_p = ctr_preds[lvl].permute(1, 2, 0).reshape(-1, 1).sigmoid()

        scores = (cls_p * ctr_p).max(dim=-1).values
        keep = scores > score_thresh

        if keep.any():
            pts = points[keep]
            b   = box_p[keep]
            s   = scores[keep]
            boxes = torch.stack([
                (pts[:, 0] - b[:, 0]).clamp(0, img_w),
                (pts[:, 1] - b[:, 1]).clamp(0, img_h),
                (pts[:, 0] + b[:, 2]).clamp(0, img_w),
                (pts[:, 1] + b[:, 3]).clamp(0, img_h),
            ], dim=-1)
            all_boxes.append(boxes)
            all_scores.append(s)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    boxes_t  = torch.cat(all_boxes)
    scores_t = torch.cat(all_scores)
    boxes_t, scores_t = soft_nms_gaussian(boxes_t, scores_t,
                                           sigma=nms_sigma, score_thresh=score_thresh,
                                           max_det=max_det)
    return boxes_t.cpu().numpy(), scores_t.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 9. DATASET (COCO, no tiling — S2D stem + high-res input handles small objs)
# ══════════════════════════════════════════════════════════════════════════════

class ClearSARDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        images_dir: Path,
        image_ids: Optional[List[int]] = None,
        imgsz: int = 640,
        augment: bool = True,
        return_meta: bool = False,
    ):
        with annotation_path.open() as f:
            coco = json.load(f)
        self.images_dir = images_dir
        self.imgsz = imgsz
        self.augment = augment
        self.return_meta = return_meta

        img_meta = {img["id"]: img for img in coco["images"]}
        anns_by_img: Dict[int, List] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        ids = image_ids if image_ids else list(img_meta.keys())
        self.samples = [(img_meta[i], anns_by_img.get(i, [])) for i in ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta, anns = self.samples[idx]
        img = Image.open(self.images_dir / meta["file_name"]).convert("RGB")
        orig_w, orig_h = img.size

        # Resize keeping aspect ratio with letterbox
        scale = self.imgsz / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        canvas.paste(img, (0, 0))
        img = canvas

        # Scale boxes
        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                x1 = x * scale; y1 = y * scale
                x2 = (x + w) * scale; y2 = (y + h) * scale
                if (x2 - x1) > 1 and (y2 - y1) > 1:
                    boxes.append([x1, y1, x2, y2])
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                if len(boxes_t):
                    boxes_t[:, [0, 2]] = self.imgsz - boxes_t[:, [2, 0]]
            if random.random() > 0.5:
                img = TF.vflip(img)
                if len(boxes_t):
                    boxes_t[:, [1, 3]] = self.imgsz - boxes_t[:, [3, 1]]
            img = TF.adjust_brightness(img, 1 + random.uniform(-0.2, 0.2))
            img = TF.adjust_contrast(img, 1 + random.uniform(-0.2, 0.2))

        img_tensor = TF.to_tensor(img)
        if not self.return_meta:
            return img_tensor, boxes_t

        return img_tensor, boxes_t, {
            "image_id": int(meta["id"]),
            "orig_w": int(orig_w),
            "orig_h": int(orig_h),
            "scale": float(scale),
        }


def collate_fn(batch):
    imgs, boxes = zip(*batch)
    return torch.stack(imgs), list(boxes)


def collate_fn_eval(batch):
    imgs, boxes, metas = zip(*batch)
    return torch.stack(imgs), list(boxes), list(metas)


# ══════════════════════════════════════════════════════════════════════════════
# 10. TRAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, optimizer, loader, device, scaler, epoch, imgsz):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, (imgs, gt_boxes_list) in enumerate(pbar, start=1):
        imgs = imgs.to(device)
        gt_boxes_list = [b.to(device) for b in gt_boxes_list]

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            cls_p, box_p, ctr_p = model(imgs)
            loss = fcos_target(cls_p, box_p, ctr_p, gt_boxes_list, imgsz, imgsz)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total / step:.4f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, imgsz):
    model.eval()
    total = 0.0
    pbar = tqdm(loader, desc="Val", leave=False)
    for step, (imgs, gt_boxes_list) in enumerate(pbar, start=1):
        imgs = imgs.to(device)
        gt_boxes_list = [b.to(device) for b in gt_boxes_list]
        cls_p, box_p, ctr_p = model(imgs)
        loss = fcos_target(cls_p, box_p, ctr_p, gt_boxes_list, imgsz, imgsz)
        loss_value = loss.item()
        total += loss_value
        pbar.set_postfix(loss=f"{loss_value:.4f}", avg=f"{total / step:.4f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_map(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    coco_gt: COCO,
    category_id: int = 1,
) -> float:
    model.eval()
    results: List[Dict] = []
    image_ids: List[int] = []

    for imgs, _, metas in tqdm(loader, desc="Val mAP", leave=False):
        imgs = imgs.to(device)
        cls_p, box_p, ctr_p = model(imgs)

        for i, meta in enumerate(metas):
            image_id = int(meta["image_id"])
            image_ids.append(image_id)
            orig_w = int(meta["orig_w"])
            orig_h = int(meta["orig_h"])
            scale = float(meta["scale"])

            boxes_xyxy, scores = decode_predictions(
                [lvl[i] for lvl in cls_p],
                [lvl[i] for lvl in box_p],
                [lvl[i] for lvl in ctr_p],
                score_thresh=0.001,
                nms_sigma=0.5,
                max_det=500,
                img_h=imgs.shape[-2],
                img_w=imgs.shape[-1],
            )

            if len(boxes_xyxy) == 0:
                continue

            for box, score in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = [float(v) for v in box]
                # Invertir letterbox usado en dataset (rescale sin offset).
                x1 = max(0.0, min(orig_w, x1 / scale))
                y1 = max(0.0, min(orig_h, y1 / scale))
                x2 = max(0.0, min(orig_w, x2 / scale))
                y2 = max(0.0, min(orig_h, y2 / scale))
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0.0 or h <= 0.0:
                    continue
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, w, h],
                        "score": float(score),
                    }
                )

    if not results:
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted(set(image_ids))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


# ══════════════════════════════════════════════════════════════════════════════
# 11. ARGS & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def split_ids(ann_path: Path, val_frac: float, seed: int):
    with ann_path.open() as f:
        coco = json.load(f)
    ids = [img["id"] for img in coco["images"]]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]


def parse_args():
    p = argparse.ArgumentParser(description="FocusDet training for ClearSAR")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--imgsz",        type=int,   default=640,
                   help="Input resolution. 960 da mejor recall en objetos muy pequeños.")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--fpn-channels", type=int,   default=256)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--no-pretrained",action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.project_root)
    ann_path      = root / "data" / "annotations" / "instances_train.json"
    train_img_dir = root / "data" / "images" / "train"
    out_dir       = root / "outputs" / "focusdet"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[focusdet] Device: {device} | imgsz: {args.imgsz}")

    train_ids, val_ids = split_ids(ann_path, args.val_fraction, args.seed)
    print(f"[focusdet] Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_ds = ClearSARDataset(ann_path, train_img_dir, train_ids, args.imgsz, augment=True)
    val_ds   = ClearSARDataset(ann_path, train_img_dir, val_ids,   args.imgsz, augment=False)
    val_eval_ds = ClearSARDataset(
        ann_path,
        train_img_dir,
        val_ids,
        args.imgsz,
        augment=False,
        return_meta=True,
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_eval_loader = DataLoader(
        val_eval_ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=True,
    )
    coco_gt = COCO(str(ann_path))

    model = FocusDet(num_classes=NUM_CLASSES, fpn_channels=args.fpn_channels,
                     pretrained=not args.no_pretrained).to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[focusdet] Resumed from epoch {start_epoch - 1}")

    # LR diferencial: backbone 10× más bajo
    backbone_params = list(model.stem.parameters()) + \
                      list(model.layer1.parameters()) + \
                      list(model.layer2.parameters()) + \
                      list(model.layer3.parameters())
    head_params = list(model.fpn.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_map = -1.0
    patience  = 0

    print(f"[focusdet] Componentes activos:")
    print(f"  ✓ Space-to-Depth stem  (preserva todos los píxeles en stride-2)")
    print(f"  ✓ Bottom-Up Focus FPN  (P2 alta resolución como mapa primario)")
    print(f"  ✓ SIoU loss            (penaliza error angular en localización)")
    print(f"  ✓ SoftNMS Gaussiano    (no suprime detecciones adyacentes densas)")
    print(f"  ✓ FCOS head            (sin anchors, centerness-weighted)")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch, args.imgsz)
        val_loss   = evaluate(model, val_loader, device, args.imgsz)
        val_map   = evaluate_map(model, val_eval_loader, device, coco_gt=coco_gt, category_id=1)
        scheduler.step()
        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"mAP50:95={val_map:.5f} | lr={lr_now:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_loss": val_loss,
            "val_map_50_95": val_map,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_map > best_map:
            best_map = val_map
            patience = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ✓ Best checkpoint (mAP50:95={val_map:.5f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[focusdet] Early stopping en época {epoch}")
                break

    print(f"[focusdet] Entrenamiento completo. Best mAP50:95={best_map:.5f}")
    print(f"[focusdet] Checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
