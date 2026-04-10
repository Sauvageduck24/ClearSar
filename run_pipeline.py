from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Optional

def _run_step(step_name: str, command: list[str], cwd: Path) -> None:
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
        description="Pipeline YOLO ClearSAR: slicing SAHI + train + inference"
    )
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))

    parser.add_argument("--skip-slicing", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")

    parser.add_argument("--mapping-path", type=str, default="catalog.v1.parquet")
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--yolo-model", type=str, default="yolo11s")
    parser.add_argument("--yolo-checkpoint", type=str, default=None)
    parser.add_argument("--yolo-extra-args", type=str, default="")
    parser.add_argument("--yolo-inference", type=str, default="")

    parser.add_argument("--cache", type=str, choices=["disk", "ram", "none"], default="disk")
    parser.add_argument(
        "--evolve",
        type=int,
        nargs="?",
        const=100,
        default=0,
        help="Habilitar evolve para YOLO (generaciones opcionales).",
    )

    parser.add_argument("--slice-size", type=int, default=256)
    parser.add_argument("--overlap-ratio", type=float, default=0.25)
    parser.add_argument("--train-annotations", type=str, default="data/annotations/instances_train.json")
    parser.add_argument("--train-images", type=str, default="data/images/train")
    parser.add_argument("--sliced-train-dir", type=str, default="data/sliced_dataset")

    parser.add_argument("--slice-test", action="store_true")
    parser.add_argument("--test-annotations", type=str, default=None)
    parser.add_argument("--test-images", type=str, default="data/images/test")
    parser.add_argument("--sliced-test-dir", type=str, default="data/sliced_test")

    parser.add_argument("--wait-gpu", action="store_true")
    return parser.parse_args()


def _extract_arg_value(tokens: list[str], key: str) -> Optional[str]:
    for i, tok in enumerate(tokens):
        if tok == key and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(f"{key}="):
            return tok.split("=", 1)[1]
    return None


def _has_arg(tokens: list[str], key: str) -> bool:
    return any(tok == key or tok.startswith(f"{key}=") for tok in tokens)


def _strip_arg(tokens: list[str], key: str) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == key:
            i += 2
            continue
        if tok.startswith(f"{key}="):
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    return cleaned


def _resolve_project_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def _find_latest_best_ckpt(project_root: Path, model_name: str) -> Optional[Path]:
    direct = project_root / "models" / f"yolo_best_{model_name.replace('.pt', '')}.pt"
    if direct.exists():
        return direct

    runs_dir = project_root / "outputs" / "yolo_runs"
    candidates = sorted(runs_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    if not project_root.exists():
        raise FileNotFoundError(f"No existe project-root: {project_root}")

    python_exe = sys.executable

    mapping_path = _resolve_project_path(project_root, args.mapping_path)
    if mapping_path is None or not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping-path: {mapping_path}")

    yolo_extra_tokens = shlex.split(args.yolo_extra_args.strip()) if args.yolo_extra_args.strip() else []
    yolo_infer_tokens = shlex.split(args.yolo_inference.strip()) if args.yolo_inference.strip() else []
    yolo_train_image_size = _extract_arg_value(yolo_extra_tokens, "--image-size")
    yolo_extra_tokens = _strip_arg(yolo_extra_tokens, "--image-size")

    if not args.skip_slicing:
        slicing_cmd = [
            python_exe,
            "-m",
            "src.slicing",
            "--project-root",
            str(project_root),
            "--train-annotations",
            args.train_annotations,
            "--train-images",
            args.train_images,
            "--train-output-dir",
            args.sliced_train_dir,
            "--slice-size",
            str(args.slice_size),
            "--overlap-ratio",
            str(args.overlap_ratio),
        ]
        if args.slice_test:
            slicing_cmd.append("--slice-test")
            if args.test_annotations:
                slicing_cmd.extend(["--test-annotations", args.test_annotations])
            slicing_cmd.extend([
                "--test-images",
                args.test_images,
                "--test-output-dir",
                args.sliced_test_dir,
            ])

        _run_step("slicing", slicing_cmd, cwd=project_root)

    if not args.skip_train:
        yolo_train_cmd = [
            python_exe,
            "-m",
            "src.yolo_train",
            "--project-root",
            str(project_root),
            "--model",
            args.yolo_model,
            "--image-size",
            str(yolo_train_image_size if yolo_train_image_size is not None else args.slice_size),
        ]

        dataset_source_override = _extract_arg_value(yolo_extra_tokens, "--dataset-source")
        if dataset_source_override is None:
            if args.skip_slicing:
                yolo_train_cmd.extend(["--dataset-source", "normal"])
            else:
                yolo_train_cmd.extend(["--dataset-source", "sahi", "--dataset-root", args.sliced_train_dir])
        elif dataset_source_override == "sahi" and _extract_arg_value(yolo_extra_tokens, "--dataset-root") is None:
            yolo_train_cmd.extend(["--dataset-root", args.sliced_train_dir])

        yolo_train_cmd.extend(yolo_extra_tokens)

        if not _has_arg(yolo_extra_tokens, "--cache"):
            yolo_train_cmd.extend(["--cache", args.cache])
        if args.evolve > 0 and not _has_arg(yolo_extra_tokens, "--evolve"):
            yolo_train_cmd.extend(["--evolve", str(args.evolve)])

        _run_step("yolo-train", yolo_train_cmd, cwd=project_root)

        if args.wait_gpu:
            _wait_for_gpu_free(timeout=60, poll_interval=2, max_used_mb=200)

    if not args.skip_inference:
        yolo_ckpt = _resolve_project_path(project_root, args.yolo_checkpoint) if args.yolo_checkpoint else None
        if yolo_ckpt is None:
            yolo_ckpt = _find_latest_best_ckpt(project_root, args.yolo_model)
        if yolo_ckpt is None or not yolo_ckpt.exists():
            raise FileNotFoundError(
                "No se encontro checkpoint YOLO. Usa --yolo-checkpoint o ejecuta sin --skip-train."
            )

        output_path = _resolve_project_path(project_root, args.output)
        if output_path is None:
            output_path = project_root / "outputs" / "submission_yolo.zip"

        yolo_infer_cmd = [
            python_exe,
            "-m",
            "src.yolo_inference",
            "--project-root",
            str(project_root),
            "--checkpoint",
            str(yolo_ckpt),
            "--mapping-path",
            str(mapping_path),
            "--output",
            str(output_path),
            "--slice-size",
            str(args.slice_size),
            "--overlap-ratio",
            str(args.overlap_ratio),
        ]

        if not _has_arg(yolo_infer_tokens, "--image-size"):
            yolo_infer_cmd.extend(["--image-size", str(args.slice_size)])

        yolo_infer_cmd.extend(yolo_infer_tokens)
        _run_step("yolo-inference", yolo_infer_cmd, cwd=project_root)

    print("\nPipeline finalizado correctamente.")


if __name__ == "__main__":
    main()