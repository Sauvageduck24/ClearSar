"""
Preprocessing module for ClearSAR filter stacking experiments.

Generates stacked 3-channel images for each combo defined in PLANNING.md.
Each channel is a single-channel filter applied to the original SAR image.
Images are saved to disk so training is fast and results are visually verifiable.

Combos (see PLANNING.md):
    run1: [Gray, CLAHE, Bilateral+TopHat]
    run2: [Gray, CLAHE, Gabor]
    run3: [Gray, CLAHE, Log]
    run4: [Gray, Gabor, Bilateral+TopHat]
    run5: [CLAHE, Gabor, Bilateral+TopHat]

Usage:
    python -m src.preprocess                      # all combos, train + test
    python -m src.preprocess --combo run1         # single combo
    python -m src.preprocess --split train        # only train images
"""

from __future__ import annotations

import argparse
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRAIN_DIR     = PROJECT_ROOT / "data" / "images" / "train"
TEST_DIR      = PROJECT_ROOT / "data" / "images" / "test"
HOLDOUT_DIR   = PROJECT_ROOT / "data" / "yolo" / "holdout" / "images" / "val"

# ---------------------------------------------------------------------------
# Filter parameters (from filter_optimization.py run on 500 images)
# ---------------------------------------------------------------------------

_CLAHE_PARAMS      = dict(clip_limit=1.5, tile_size=6)
_LOG_PARAMS        = dict(gain=10.0)
_GABOR_PARAMS      = dict(ksize=13, sigma=3.5, lam=8.0, gamma=0.5)
_BILAT_TH_PARAMS   = dict(d=7, sigma_color=50, sigma_space=50, kernel_w=17)

# ---------------------------------------------------------------------------
# Single-channel extractors — each returns a (H, W) uint8 array
# ---------------------------------------------------------------------------

def extract_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def extract_clahe(img_rgb: np.ndarray,
                  clip_limit: float = _CLAHE_PARAMS["clip_limit"],
                  tile_size: int    = _CLAHE_PARAMS["tile_size"]) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(l)


def extract_log(img_rgb: np.ndarray,
                gain: float = _LOG_PARAMS["gain"]) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    res = np.log1p(gray_f * gain) / np.log1p(gain)
    return (res * 255).astype(np.uint8)


def extract_gabor(img_rgb: np.ndarray,
                  ksize: int   = _GABOR_PARAMS["ksize"],
                  sigma: float = _GABOR_PARAMS["sigma"],
                  lam:   float = _GABOR_PARAMS["lam"],
                  gamma: float = _GABOR_PARAMS["gamma"]) -> np.ndarray:
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, np.pi / 2, lam, gamma, 0,
                                ktype=cv2.CV_32F)
    filtered   = cv2.filter2D(gray, -1, kernel)
    filtered_8u = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, mask = cv2.threshold(filtered_8u, 10, 255, cv2.THRESH_BINARY)
    return mask


def extract_bilateral_tophat(img_rgb: np.ndarray,
                              d:            int = _BILAT_TH_PARAMS["d"],
                              sigma_color:  int = _BILAT_TH_PARAMS["sigma_color"],
                              sigma_space:  int = _BILAT_TH_PARAMS["sigma_space"],
                              kernel_w:     int = _BILAT_TH_PARAMS["kernel_w"]) -> np.ndarray:
    smooth = cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)
    gray   = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    res    = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(res, 5, 255, cv2.THRESH_BINARY)
    return mask


_EXTRACTORS = {
    "gray":            extract_gray,
    "clahe":           extract_clahe,
    "log":             extract_log,
    "gabor":           extract_gabor,
    "bilateral_tophat": extract_bilateral_tophat,
}

# ---------------------------------------------------------------------------
# Combo definitions
# ---------------------------------------------------------------------------

COMBOS: dict[str, list[str]] = {
    "run1": ["gray", "clahe", "bilateral_tophat"],
    "run2": ["gray", "clahe", "gabor"],
    "run3": ["gray", "clahe", "log"],
    "run4": ["gray", "gabor", "bilateral_tophat"],
    "run5": ["clahe", "gabor", "bilateral_tophat"],
}

COMBO_LABELS: dict[str, str] = {
    "run1": "Gray | CLAHE | Bilateral+TopHat",
    "run2": "Gray | CLAHE | Gabor",
    "run3": "Gray | CLAHE | Log",
    "run4": "Gray | Gabor | Bilateral+TopHat",
    "run5": "CLAHE | Gabor | Bilateral+TopHat",
}

# ---------------------------------------------------------------------------
# Core stacking function (importable by inference.py)
# ---------------------------------------------------------------------------

def build_stacked(img_rgb: np.ndarray, combo: str) -> np.ndarray:
    """Return a (H, W, 3) uint8 stacked image for the given combo name."""
    channels = [_EXTRACTORS[name](img_rgb) for name in COMBOS[combo]]
    return np.stack(channels, axis=2)


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _process_image(args: tuple) -> str | None:
    src_path, dst_path, combo = args
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        return str(src_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    stacked  = build_stacked(img_rgb, combo)
    stacked_bgr = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), stacked_bgr)
    return None


def _copy_holdout_from_train(combo: str, ref_dir: Path) -> None:
    """
    Populate holdout_<combo>/ by copying the already-stacked images from train_<combo>/.
    ref_dir (data/yolo/holdout/images/val) is used only to know which filenames belong
    to the holdout split — the actual pixel data comes from train_<combo>.
    """
    src_dir = PROJECT_ROOT / "data" / "images" / f"train_{combo}"
    dst_dir = PROJECT_ROOT / "data" / "images" / f"holdout_{combo}"

    image_exts = {".png", ".jpg", ".jpeg"}
    ref_names  = [f.name for f in sorted(ref_dir.iterdir())
                  if f.is_file() and f.suffix.lower() in image_exts]
    if not ref_names:
        print(f"  [!] No reference images in {ref_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    for name in ref_names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied += 1
        else:
            missing += 1

    print(f"  Copied {copied}/{len(ref_names)} holdout images → {dst_dir}")
    if missing:
        print(f"  [!] {missing} files not found in {src_dir}")


def _preprocess_split(src_dir: Path, dst_dir: Path, combo: str, n_workers: int) -> None:
    image_paths = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg")) + \
                  sorted(src_dir.glob("*.jpeg"))
    if not image_paths:
        print(f"  [!] No images found in {src_dir}")
        return

    args = [(p, dst_dir / p.name, combo) for p in image_paths]

    failed = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(pool.imap_unordered(_process_image, args),
                           total=len(args), unit="img", leave=False):
            if result is not None:
                failed.append(result)

    print(f"  Saved {len(args) - len(failed)}/{len(args)} images → {dst_dir}")
    if failed:
        print(f"  [!] Failed to read {len(failed)} images")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filter-stacked images for training")
    parser.add_argument("--combo", type=str, default=None,
                        choices=list(COMBOS.keys()),
                        help="Generate only this combo. Default: all combos.")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "test", "holdout"],
                        help="Process only this split. Default: train + test.")
    args = parser.parse_args()

    combos   = [args.combo] if args.combo else list(COMBOS.keys())
    splits   = [args.split] if args.split else ["train", "test"]
    n_workers = min(8, cpu_count())

    src_map = {
        "train":   TRAIN_DIR,
        "test":    TEST_DIR,
        "holdout": HOLDOUT_DIR,
    }

    for combo in combos:
        print(f"\n[{combo}] {COMBO_LABELS[combo]}")
        for split in splits:
            src_dir = src_map[split]
            if split == "holdout":
                print(f"  holdout: copying from train_{combo} (ref={src_dir.name})")
                _copy_holdout_from_train(combo, src_dir)
            else:
                dst_dir = PROJECT_ROOT / "data" / "images" / f"{split}_{combo}"
                print(f"  {split}: {src_dir.name} → {dst_dir.name}")
                _preprocess_split(src_dir, dst_dir, combo, n_workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
