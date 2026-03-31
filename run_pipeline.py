from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional

from src.config import default_config
# Intentar importar get_yolo_predictions desde el módulo de yolo; si no existe,
# proporcionar una implementación local mínima que use SAHI para slicing.
try:
    from src.yolo_inference import get_yolo_predictions
except Exception:
    def get_yolo_predictions(checkpoint_path, images_dir, filename_to_image_id,
                             conf: float = 0.001, iou: float = 0.5, imgsz: int = 640,
                             use_tta=False):
        import torch
        from pathlib import Path
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError:
            raise ImportError("Por favor instala sahi: pip install sahi")

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(checkpoint_path),
            confidence_threshold=conf,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        images_dir = Path(images_dir)
        if isinstance(use_tta, str):
            use_tta = use_tta.lower() in ("true", "1", "yes")

        submission_rows = []
        for fname, img_id in filename_to_image_id.items():
            img_path = str(images_dir / fname)
            result = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=int(imgsz),
                slice_width=int(imgsz),
                overlap_height_ratio=0.3,
                overlap_width_ratio=0.3,
                perform_standard_pred=bool(use_tta),
                postprocess_type="NMM",
                postprocess_match_threshold=0.5,
            )

            for obj in result.object_prediction_list:
                b = obj.bbox
                coco_box = [b.minx, b.miny, b.maxx - b.minx, b.maxy - b.miny]
                if coco_box[2] <= 0.001 or coco_box[3] <= 0.001:
                    continue
                submission_rows.append({
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [float(v) for v in coco_box],
                    "score": float(obj.score.value),
                })

        return submission_rows

def _run_step(step_name: str, command: List[str], cwd: Path) -> None:
    print(f"\n[{step_name}] Ejecutando: {' '.join(shlex.quote(c) for c in command)}")
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Fallo en '{step_name}' con exit code {result.returncode}")


def _wait_for_gpu_free(timeout: int = 60, poll_interval: int = 2, max_used_mb: int = 200) -> None:
    """Wait until GPU memory usage drops below `max_used_mb` (MB).

    Uses `nvidia-smi` when available; otherwise falls back to a short sleep.
    """
    if shutil.which("nvidia-smi") is None:
        print("[pipeline] nvidia-smi no disponible; durmiendo 10s para liberar VRAM.")
        time.sleep(min(timeout, 10))
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ], encoding="utf-8")
            lines = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
            max_used = max(lines) if lines else 0
            print(f"[pipeline] GPU memory used: {max_used} MB (threshold {max_used_mb} MB)")
            if max_used <= max_used_mb:
                print("[pipeline] GPU memory below threshold; continuing.")
                return
        except Exception as e:
            print(f"[pipeline] nvidia-smi check failed: {e}; sleeping {poll_interval}s and retrying.")
        time.sleep(poll_interval)

    print(f"[pipeline] Timeout esperando VRAM < {max_used_mb} MB; continuando de todos modos.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquesta el pipeline ClearSAR: entrenamiento + inferencia + submission"
    )
    parser.add_argument("--project-root", type=str,
                        default=str(Path(__file__).resolve().parent))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint Faster R-CNN para inferencia")
    parser.add_argument("--yolo-checkpoint", type=str, default=None,
                        help="Checkpoint YOLO para inferencia/ensemble")
    parser.add_argument("--mapping-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--train-extra-args", type=str, default="",
                        help="Args adicionales para src.train")
    parser.add_argument("--yolo-extra-args", type=str, default="",
                        help="Args adicionales para src.yolo_train")
    parser.add_argument("--infer-extra-args", type=str, default="")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--train-profile", type=str,
                        choices=["fast", "balanced", "quality"], default=None)
    parser.add_argument("--window1", action="store_true",
                        help="Faster R-CNN supervisado + ensemble checkpoints")
    parser.add_argument("--window2", action="store_true",
                        help="Pseudo-labeling + fine-tuning Faster R-CNN")
    parser.add_argument("--window3", action="store_true",
                        help="SSL pretrain + fine-tuning Faster R-CNN")
    parser.add_argument("--window-yolo", action="store_true",
                        help="Entrenar YOLO + submission independiente")
    parser.add_argument("--window-ensemble", action="store_true",
                        help="Ensemble cross-arquitectura YOLO + Faster R-CNN con WBF")
    parser.add_argument("--all-windows", action="store_true",
                        help="Ejecuta window1 + window-yolo + window-ensemble")
    parser.add_argument("--ensemble-top-k", type=int, default=2)
    parser.add_argument("--yolo-model", type=str, default="yolo8n",
                        help="Variante YOLO: yolo11n/s/m/l/x o yolov8m etc.")
    parser.add_argument("--local-val-only", action="store_true")
    parser.add_argument("--use-tta", type=str, default="True", help="Habilitar TTA (True/False)")
    return parser.parse_args()


def _sanitize_train_extra_args(raw_args: str) -> list[str]:
    """Force preprocess mode to 'none' to preserve original RGB quicklooks."""
    if not raw_args.strip():
        return ["--preprocess", "none"]

    tokens = shlex.split(raw_args.strip())
    sanitized: list[str] = []
    forced_preprocess_none = False

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--preprocess":
            forced_preprocess_none = True
            i += 2
            continue
        sanitized.append(tok)
        i += 1

    if forced_preprocess_none:
        print("[pipeline] Aviso: '--preprocess' sobrescrito a 'none'.")

    sanitized.extend(["--preprocess", "none"])
    return sanitized


def _extract_arg_value(args: list[str], key: str) -> Optional[str]:
    for i, tok in enumerate(args):
        if tok == key and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith(f"{key}="):
            return tok.split("=", 1)[1]
    return None


def _resolve_mapping_path(project_root: Path, cli_mapping_path: Optional[str]) -> Optional[Path]:
    cfg = default_config(project_root=project_root)
    if cli_mapping_path:
        return Path(cli_mapping_path)
    return cfg.paths.test_id_mapping_path


def _cross_arch_ensemble(
    project_root: Path,
    mapping_path: Path,
    rcnn_checkpoint: Optional[Path],
    yolo_checkpoint: Optional[Path],
    output_path: Path,
    python_exe: str,
) -> None:
    """
    Ensemble Faster R-CNN + YOLO using WBF.
    Collects predictions from both models, fuses per image, saves submission.
    """
    try:
        import torch
        from src.config import default_config, ensure_dirs
        from src.dataset import load_test_id_mapping
        from src.inference import _fuse_predictions_single_image, tta_predict_batch
        from src.submission import predictions_to_submission_rows, save_submission_auto, validate_submission_schema
        from src.utils.repro import resolve_device
    except ImportError as e:
        raise ImportError(f"Missing dependency for ensemble: {e}")

    cfg = default_config(project_root=project_root)
    ensure_dirs(cfg)
    device = resolve_device()

    filename_to_image_id = load_test_id_mapping(
        test_images_dir=cfg.paths.test_images_dir,
        mapping_path=mapping_path,
        strict=True,
    )

    # Collect predictions from each model: dict image_id -> list of pred dicts
    all_preds: dict = {}  # image_id -> {"boxes_list": [], "scores_list": [], "width": int, "height": int}

    def _ensure_image(image_id: int, width: int, height: int) -> None:
        if image_id not in all_preds:
            all_preds[image_id] = {"boxes_list": [], "scores_list": [], "width": width, "height": height}

    # --- Faster R-CNN predictions ---
    if rcnn_checkpoint and rcnn_checkpoint.exists():
        print(f"[ensemble] Running Faster R-CNN: {rcnn_checkpoint}")
        from src.dataset import ClearSARTestDataset, collate_detection_batch
        from src.model import apply_checkpoint_model_hints, build_model, load_model_checkpoint
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        test_ds = ClearSARTestDataset(
            images_dir=cfg.paths.test_images_dir,
            filename_to_image_id=filename_to_image_id,
            transforms=None,
            preprocess_mode="none",
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.inference.batch_size, shuffle=False,
            num_workers=cfg.inference.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_detection_batch,
        )

        apply_checkpoint_model_hints(cfg.model, str(rcnn_checkpoint))
        rcnn_model = build_model(cfg.model, device=device)
        rcnn_model = load_model_checkpoint(rcnn_model, str(rcnn_checkpoint), device)
        rcnn_model.eval()

        with torch.inference_mode():
            for images, metas in tqdm(test_loader, desc="rcnn"):
                images_gpu = [img.to(device, non_blocking=True) for img in images]
                outs = tta_predict_batch(rcnn_model, images_gpu, use_tta=cfg.inference.use_tta)
                for out, meta in zip(outs, metas):
                    img_id = int(meta["image_id"])
                    w, h = int(meta["width"]), int(meta["height"])
                    _ensure_image(img_id, w, h)
                    if out["boxes"].numel() > 0:
                        all_preds[img_id]["boxes_list"].append(out["boxes"])
                        all_preds[img_id]["scores_list"].append(out["scores"])

        del rcnn_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- YOLO predictions ---
    if yolo_checkpoint and yolo_checkpoint.exists():
        print(f"[ensemble] Running YOLO: {yolo_checkpoint}")
        yolo_extra_args = shlex.split(args.yolo_extra_args)
        imgsz = _extract_arg_value(yolo_extra_args, "--image-size") or 640
        use_tta = args.use_tta

        yolo_rows = _get_yolo_predictions(
            args, yolo_checkpoint, cfg, filename_to_image_id,
            use_tta=use_tta, imgsz=int(imgsz)  # Dynamically pass imgsz
        )

        # Re-ingest YOLO rows into the all_preds structure for WBF
        import torch as _torch
        from pathlib import Path as _Path
        from PIL import Image as _Image
        import numpy as _np

        # We need width/height for YOLO preds — read from the row scores which have image_id
        # Width/height come from the original images
        img_sizes: dict = {}
        for img_path in cfg.paths.test_images_dir.iterdir():
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                fname = img_path.name
                if fname in filename_to_image_id:
                    iid = filename_to_image_id[fname]
                    if iid not in img_sizes:
                        img = _Image.open(img_path)
                        img_sizes[iid] = (img.width, img.height)

        # Group yolo rows by image_id and convert xywh -> xyxy
        yolo_by_img: dict = {}
        for row in yolo_rows:
            iid = int(row["image_id"])
            x, y, w, h = row["bbox"]
            x2, y2 = x + w, y + h
            yolo_by_img.setdefault(iid, {"boxes": [], "scores": []})
            yolo_by_img[iid]["boxes"].append([x, y, x2, y2])
            yolo_by_img[iid]["scores"].append(float(row["score"]))

        for iid, data in yolo_by_img.items():
            w_img, h_img = img_sizes.get(iid, (640, 640))
            _ensure_image(iid, w_img, h_img)
            boxes_t = _torch.tensor(data["boxes"], dtype=_torch.float32)
            scores_t = _torch.tensor(data["scores"], dtype=_torch.float32)
            all_preds[iid]["boxes_list"].append(boxes_t)
            all_preds[iid]["scores_list"].append(scores_t)

    # --- WBF fusion ---
    print(f"[ensemble] Fusing {len(all_preds)} images with WBF")
    rows = []
    for img_id, data in all_preds.items():
        w, h = data["width"], data["height"]
        if not data["boxes_list"]:
            continue

        import torch as _torch
        pred_list = [
            {"boxes": b, "scores": s}
            for b, s in zip(data["boxes_list"], data["scores_list"])
        ]
        fused = _fuse_predictions_single_image(
            pred_list=pred_list,
            width=w,
            height=h,
            use_wbf=True,
            wbf_iou_thr=cfg.inference.wbf_iou_thr,
            wbf_skip_box_thr=cfg.inference.wbf_skip_box_thr,
            nms_iou_thr=cfg.inference.nms_iou_thr,
        )
        meta = {"image_id": img_id, "width": w, "height": h}
        batch_rows = predictions_to_submission_rows(
            outputs=[fused], metas=[meta],
            score_threshold=cfg.inference.min_score,
            category_id=1,
            max_detections_per_image=cfg.inference.max_detections_per_image,
            box_format="xyxy",
        )
        rows.extend(batch_rows)

    validate_submission_schema(rows)
    save_submission_auto(rows, output_path)
    print(f"[ensemble] Cross-arch submission saved: {output_path} ({len(rows)} rows)")


def _get_yolo_predictions(args, yolo_checkpoint, cfg, filename_to_image_id, use_tta, imgsz):
    return get_yolo_predictions(
        checkpoint_path=yolo_checkpoint,
        images_dir=cfg.paths.test_images_dir,
        filename_to_image_id=filename_to_image_id,
        conf=0.001,
        iou=0.5,
        imgsz=imgsz,
        use_tta=use_tta,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    if not project_root.exists():
        raise FileNotFoundError(f"No existe project-root: {project_root}")

    python_exe = sys.executable

    run_w1 = args.window1 or args.all_windows
    run_w2 = args.window2
    run_w3 = args.window3
    run_wyolo = args.window_yolo or args.all_windows
    run_wensemble = args.window_ensemble or args.all_windows

    sanitized_train_extra_args = _sanitize_train_extra_args(args.train_extra_args)
    cfg = default_config(project_root=project_root)
    mapping_path = _resolve_mapping_path(project_root, args.mapping_path)

    needs_mapping = not args.local_val_only
    if needs_mapping:
        if mapping_path is None:
            raise ValueError(
                "No se encontro mapping-path. Usa --mapping-path o coloca catalog.v1.parquet en la raiz."
            )
        if not mapping_path.exists():
            raise FileNotFoundError(f"No existe mapping-path: {mapping_path}")

    base_model_name = str(cfg.model.architecture)
    extra_arch_override = _extract_arg_value(sanitized_train_extra_args, "--arch")

    def build_train_cmd(extra: list[str] | None = None) -> list[str]:
        cmd = [python_exe, "-m", "src.train", "--project-root", str(project_root)]
        if args.train_profile:
            cmd.extend(["--profile", args.train_profile])
        if args.fast:
            cmd.append("--fast")
        if sanitized_train_extra_args:
            cmd.extend(sanitized_train_extra_args)
        if extra:
            cmd.extend(extra)
        return cmd

    def build_infer_cmd(extra: list[str] | None = None) -> list[str]:
        cmd = [python_exe, "-m", "src.inference", "--project-root", str(project_root)]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if mapping_path is not None:
            cmd.extend(["--mapping-path", str(mapping_path)])
        if args.output:
            cmd.extend(["--output", args.output])
        if args.infer_extra_args.strip():
            cmd.extend(shlex.split(args.infer_extra_args.strip()))
        if extra:
            cmd.extend(extra)
        return cmd

    def build_yolo_train_cmd(extra: list[str] | None = None) -> list[str]:
        cmd = [python_exe, "-m", "src.yolo_train",
               "--project-root", str(project_root),
               "--model", args.yolo_model]
        if args.yolo_extra_args.strip():
            cmd.extend(shlex.split(args.yolo_extra_args.strip()))
        if extra:
            cmd.extend(extra)
        return cmd

    # Default behavior (no windows specified)
    if not (run_w1 or run_w2 or run_w3 or run_wyolo or run_wensemble):
        if not args.skip_train:
            _run_step("train", build_train_cmd(), cwd=project_root)
        if not args.local_val_only:
            _run_step("inference", build_infer_cmd(), cwd=project_root)
        print("\nPipeline finalizado correctamente.")
        return

    # ── Window 1: Faster R-CNN supervisado ───────────────────────────────────
    if run_w1:
        print("\n[window1] Faster R-CNN supervisado + ensemble de checkpoints")
        if not args.skip_train:
            model_name_w1 = extra_arch_override if extra_arch_override else base_model_name
            print(f"[window1] Modelo: {model_name_w1}")
            _run_step("window1-train",
                      build_train_cmd(["--save-top-k", str(args.ensemble_top_k)]),
                      cwd=project_root)

        if not args.local_val_only:
            w1_out = str(project_root / "outputs" / "submission_w1.zip")
            ensemble_cmd = [
                python_exe, "-m", "src.ensemble",
                "--project-root", str(project_root),
                "--max-models", str(args.ensemble_top_k),
                "--output", w1_out,
                "--tta", "--wbf",
            ]
            if mapping_path is not None:
                ensemble_cmd.extend(["--mapping-path", str(mapping_path)])
            _run_step("window1-ensemble", ensemble_cmd, cwd=project_root)

    # ── Window 2: Pseudo-labeling + fine-tuning ───────────────────────────────
    pseudo_ann = str(project_root / "outputs" / "pseudo" / "instances_pseudo_test.json")
    if run_w2:
        print("\n[window2] Pseudo-labeling + fine-tuning + ensemble")
        ckpt = args.checkpoint if args.checkpoint else str(project_root / "models" / "best_model.pt")
        pseudo_cmd = [
            python_exe, "-m", "src.pseudo_label",
            "--project-root", str(project_root),
            "--checkpoint", ckpt,
            "--output-annotations", pseudo_ann,
            "--score-threshold", "0.50",
            "--tta",
        ]
        if mapping_path is not None:
            pseudo_cmd.extend(["--mapping-path", str(mapping_path)])
        _run_step("window2-pseudo-label", pseudo_cmd, cwd=project_root)

        if not args.skip_train:
            train_w2 = build_train_cmd([
                "--save-top-k", str(args.ensemble_top_k),
                "--extra-images-dir", str(project_root / "data" / "images" / "test"),
                "--extra-annotations-path", pseudo_ann,
            ])
            _run_step("window2-train", train_w2, cwd=project_root)

        if not args.local_val_only:
            w2_out = str(project_root / "outputs" / "submission_w2.zip")
            ensemble_w2 = [
                python_exe, "-m", "src.ensemble",
                "--project-root", str(project_root),
                "--max-models", str(args.ensemble_top_k),
                "--output", w2_out, "--tta", "--wbf",
            ]
            if mapping_path is not None:
                ensemble_w2.extend(["--mapping-path", str(mapping_path)])
            _run_step("window2-ensemble", ensemble_w2, cwd=project_root)

    # ── Window 3: SSL pretrain + fine-tuning ─────────────────────────────────
    ssl_backbone = str(project_root / "models" / "ssl_backbone.pt")
    if run_w3:
        print("\n[window3] SSL pretrain + fine-tuning + ensemble")
        ssl_cmd = [
            python_exe, "-m", "src.ssl_pretrain",
            "--project-root", str(project_root),
            "--epochs", "10",
            "--batch-size", "32",
            "--output", ssl_backbone,
        ]
        _run_step("window3-ssl-pretrain", ssl_cmd, cwd=project_root)

        if not args.skip_train:
            train_w3 = build_train_cmd([
                "--save-top-k", str(args.ensemble_top_k),
                "--ssl-backbone-path", ssl_backbone,
                "--arch", "fasterrcnn_resnet50_fpn_v2",
            ])
            _run_step("window3-train", train_w3, cwd=project_root)

        if not args.local_val_only:
            w3_out = str(project_root / "outputs" / "submission_w3.zip")
            ensemble_w3 = [
                python_exe, "-m", "src.ensemble",
                "--project-root", str(project_root),
                "--max-models", str(args.ensemble_top_k),
                "--output", w3_out, "--tta", "--wbf",
            ]
            if mapping_path is not None:
                ensemble_w3.extend(["--mapping-path", str(mapping_path)])
            _run_step("window3-ensemble", ensemble_w3, cwd=project_root)

    # ── Window YOLO: entrenar YOLO y generar submission ───────────────────────
    if run_wyolo:
        print(f"\n[window-yolo] Entrenando YOLO ({args.yolo_model})")
        if not args.skip_train:
            _run_step("yolo-train", build_yolo_train_cmd(), cwd=project_root)
            # Tras entrenamiento, esperar a que la VRAM se libere (Windows puede tardar)
            _wait_for_gpu_free(timeout=60, poll_interval=2, max_used_mb=200)

        if not args.local_val_only:
            yolo_ckpt = args.yolo_checkpoint
            if yolo_ckpt is None:
                # Auto-detect best checkpoint
                yolo_ckpt_path = project_root / "models" / f"yolo_best_{args.yolo_model}.pt"
                if yolo_ckpt_path.exists():
                    yolo_ckpt = str(yolo_ckpt_path)
                else:
                    # Fallback: search runs dir
                    runs_dir = project_root / "outputs" / "yolo_runs"
                    candidates = list(runs_dir.glob(f"**/{args.yolo_model}*/weights/best.pt"))
                    if candidates:
                        yolo_ckpt = str(sorted(candidates)[-1])

            if yolo_ckpt:
                yolo_out = str(project_root / "outputs" / "submission_yolo.zip")
                # --- INICIO DE LA CORRECCIÓN ---
                # 1. Extraer image-size de yolo_extra_args si existe
                yolo_extra_list = shlex.split(args.yolo_extra_args)
                imgsz_val = _extract_arg_value(yolo_extra_list, "--image-size")

                yolo_infer_cmd = [
                    python_exe, "-m", "src.yolo_inference",
                    "--project-root", str(project_root),
                    "--checkpoint", yolo_ckpt,
                    "--output", yolo_out,
                ]

                # 2. Pasar el image-size si se encontró
                if imgsz_val:
                    yolo_infer_cmd.extend(["--image-size", imgsz_val])

                # 3. Manejar TTA correctamente (comparando como string)
                if str(args.use_tta).lower() == "true":
                    yolo_infer_cmd.append("--tta")
                else:
                    yolo_infer_cmd.append("--no-tta")
                # --- FIN DE LA CORRECCIÓN ---

                if mapping_path is not None:
                    yolo_infer_cmd.extend(["--mapping-path", str(mapping_path)])
                _run_step("yolo-inference", yolo_infer_cmd, cwd=project_root)
            else:
                print("[window-yolo] No se encontro checkpoint YOLO. Omitiendo inferencia.")

    # ── Window Ensemble: cross-arquitectura YOLO + Faster R-CNN ──────────────
    if run_wensemble and not args.local_val_only:
        print("\n[window-ensemble] Ensemble cross-arquitectura YOLO + Faster R-CNN")

        rcnn_ckpt = Path(args.checkpoint) if args.checkpoint else project_root / "models" / "best_model.pt"
        if not rcnn_ckpt.exists():
            rcnn_ckpt = None
            print("[window-ensemble] No se encontro checkpoint Faster R-CNN, se usara solo YOLO")

        yolo_ckpt_str = args.yolo_checkpoint
        if yolo_ckpt_str is None:
            yolo_ckpt_path = project_root / "models" / f"yolo_best_{args.yolo_model}.pt"
            if yolo_ckpt_path.exists():
                yolo_ckpt_str = str(yolo_ckpt_path)
        yolo_ckpt = Path(yolo_ckpt_str) if yolo_ckpt_str else None

        if rcnn_ckpt is None and yolo_ckpt is None:
            print("[window-ensemble] No hay checkpoints disponibles. Omitiendo ensemble.")
        else:
            ensemble_out = Path(args.output) if args.output else project_root / "outputs" / "submission_ensemble_final.zip"
            _cross_arch_ensemble(
                project_root=project_root,
                mapping_path=mapping_path,
                rcnn_checkpoint=rcnn_ckpt,
                yolo_checkpoint=yolo_ckpt,
                output_path=ensemble_out,
                python_exe=python_exe,
            )

    print("\nPipeline finalizado correctamente.")


if __name__ == "__main__":
    args = parse_args()
    main()