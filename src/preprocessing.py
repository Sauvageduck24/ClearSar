from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pywt


def _load_image_raw(img_path: Path) -> Optional[np.ndarray]:
    try:
        if img_path.suffix.lower() == ".npy":
            arr = np.load(img_path)
        else:
            try:
                from PIL import Image

                arr = np.array(Image.open(img_path))
            except Exception:
                arr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if arr is None:
                    return None
        return np.asarray(arr)
    except Exception:
        return None


def _save_image_raw(img_path: Path, arr: np.ndarray) -> bool:
    try:
        if img_path.suffix.lower() == ".npy":
            np.save(img_path, arr)
            return True

        try:
            from PIL import Image

            Image.fromarray(arr).save(img_path)
            return True
        except Exception:
            return bool(cv2.imwrite(str(img_path), arr))
    except Exception:
        return False


def _to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.mean(axis=-1)
    return np.asarray(arr)


def _load_image_gray(img_path: Path) -> Optional[np.ndarray]:
    """Load grayscale image from .npy or common image formats."""
    try:
        arr = _load_image_raw(img_path)
        if arr is None:
            return None
        return _to_gray(arr)
    except Exception:
        return None


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if max_v - min_v < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - min_v) / (max_v - min_v)


def _local_std(channel: np.ndarray, k: int = 7) -> np.ndarray:
    mean = cv2.GaussianBlur(channel, (k, k), 0)
    mean_sq = cv2.GaussianBlur(channel**2, (k, k), 0)
    return np.sqrt(np.clip(mean_sq - mean**2, 0, None))


def _apply_std_multi_norm(img: np.ndarray) -> np.ndarray:
    """
    Replica el preprocesado de visualize_image_og_and_filter:
      - R: VV log+norm
      - G: VH log+norm
      - B: STD_MULTI normalizado (std local multi-escala sobre VV log)
    """
    if img is None or img.size == 0:
        return img

    arr = np.asarray(img)
    if arr.ndim == 2:
        vv_raw = arr.astype(np.float32)
        vh_raw = vv_raw
    elif arr.ndim == 3:
        if arr.shape[2] >= 2:
            vv_raw = arr[:, :, 0].astype(np.float32)
            vh_raw = arr[:, :, 1].astype(np.float32)
        else:
            vv_raw = arr[:, :, 0].astype(np.float32)
            vh_raw = vv_raw
    else:
        return img

    vv_log = np.log1p(np.clip(vv_raw, a_min=0, a_max=None))
    vh_log = np.log1p(np.clip(vh_raw, a_min=0, a_max=None))

    vv_log_norm = _normalize01(vv_log)
    vh_log_norm = _normalize01(vh_log)

    std7 = _local_std(vv_log, 7)
    std15 = _local_std(vv_log, 15)
    std31 = _local_std(vv_log, 31)
    std_multi = ((std7 + std15 + std31) / 3.0) * 2.0
    std_multi_norm = _normalize01(std_multi)

    out = np.dstack([vv_log_norm, vh_log_norm, std_multi_norm])
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _apply_vv_vh_max(img: np.ndarray) -> np.ndarray:
    """
    Construye la representacion [VV, VH, max(VV, VH)].

    Usa los dos primeros canales de la imagen como VV y VH, y genera un tercer
    canal con el maximo pixel a pixel entre ambos.
    """
    if img is None or img.size == 0:
        return img

    arr = np.asarray(img)
    if arr.ndim == 2:
        vv = arr
        vh = arr
    elif arr.ndim == 3:
        vv = arr[:, :, 0]
        if arr.shape[2] >= 2:
            vh = arr[:, :, 1]
        else:
            vh = arr[:, :, 0]
    else:
        return img

    max_vv_vh = np.maximum(vv, vh)
    out = np.dstack([vv, vh, max_vv_vh])
    return np.asarray(out, dtype=arr.dtype)


def _reorder_channels_by_rfi_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Reordena los 3 canales BGR de mayor a menor contraste horizontal de RFI.
    """
    scores = []
    for i in range(3):
        ch = img_bgr[:, :, i].astype(np.float32)
        row_diff = np.abs(np.diff(ch, axis=0))
        scores.append(float(np.percentile(row_diff, 95)))

    order = sorted(range(3), key=lambda i: scores[i], reverse=True)
    return cv2.merge(
        [
            img_bgr[:, :, order[0]],
            img_bgr[:, :, order[1]],
            img_bgr[:, :, order[2]],
        ]
    )


def _inject_horizontal_wavelet(img: np.ndarray) -> np.ndarray:
    """
    Inyecta los detalles horizontales de la transformada wavelet Haar en el canal azul (indice 2).
    """
    if img is None or img.size == 0:
        return img

    try:
        canal_vv = img[:, :, 0].astype(np.float32)

        coeffs2 = pywt.dwt2(canal_vv, "haar")
        _ll, (lh, _hl, _hh) = coeffs2

        lh_resized = cv2.resize(lh, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        lh_normalized = cv2.normalize(np.abs(lh_resized), None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        img[:, :, 2] = lh_normalized
        return img
    except Exception as e:
        print(f"[wavelet] Error inyectando wavelet: {e}")
        return img


def _resize_image_to_square(img: np.ndarray, image_size: int) -> np.ndarray:
    """Resize directo a (image_size, image_size) sin conservar aspecto."""
    if image_size <= 0:
        return img
    if img is None or img.size == 0:
        return img

    target = (int(image_size), int(image_size))
    if img.ndim == 2:
        return cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)

    if img.ndim == 3:
        channels = [
            cv2.resize(img[:, :, c], target, interpolation=cv2.INTER_LINEAR) for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=-1).astype(img.dtype, copy=False)

    return img


def _resize_image_to_shape(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize a (out_w, out_h) conservando numero de canales/dtype."""
    if out_w <= 0 or out_h <= 0:
        return img
    if img is None or img.size == 0:
        return img

    target = (int(out_w), int(out_h))
    if img.ndim == 2:
        return cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)

    if img.ndim == 3:
        channels = [
            cv2.resize(img[:, :, c], target, interpolation=cv2.INTER_LINEAR) for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=-1).astype(img.dtype, copy=False)

    return img
