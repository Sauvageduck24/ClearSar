from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from src.utils.repro import resolve_device, set_seed


def sar_percentile_stretch_rgb(
    image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """
    Channel-wise percentile stretch for RGB quicklooks.

    Keeps the same interface previously imported from src.dataset.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image in HWC RGB format, got shape={image.shape}")

    x = image.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)

    for c in range(3):
        channel = x[..., c]
        lo = float(np.percentile(channel, low_percentile))
        hi = float(np.percentile(channel, high_percentile))
        if hi <= lo:
            out[..., c] = np.clip(channel, 0.0, 255.0)
        else:
            out[..., c] = np.clip((channel - lo) * (255.0 / (hi - lo)), 0.0, 255.0)

    return out.astype(np.uint8)


class RotationPretextDataset(Dataset):
    def __init__(
        self,
        image_dirs: Sequence[str | Path],
        image_size: int = 224,
        preprocess_mode: str = "percentile",
    ) -> None:
        self.image_paths: List[Path] = []
        for d in image_dirs:
            p = Path(d)
            if not p.exists():
                continue
            self.image_paths.extend([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".png", ".jpg", ".jpeg"}])

        if not self.image_paths:
            raise ValueError("No images found for SSL pretraining")

        self.preprocess_mode = preprocess_mode
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(size=image_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.02),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.preprocess_mode == "percentile":
            img = sar_percentile_stretch_rgb(img)

        label = random.randint(0, 3)
        img = np.rot90(img, k=label).copy()
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised pretraining (rotation prediction)")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--train-images-dir", type=str, default=None)
    parser.add_argument("--test-images-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--preprocess", type=str, choices=["none", "percentile"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]
    train_dir = Path(args.train_images_dir) if args.train_images_dir else (root / "data" / "images" / "train")
    test_dir = Path(args.test_images_dir) if args.test_images_dir else (root / "data" / "images" / "test")
    output_path = Path(args.output) if args.output else (root / "models" / "ssl_backbone.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device()

    ds = RotationPretextDataset(
        image_dirs=[train_dir, test_dir],
        image_size=args.image_size,
        preprocess_mode=args.preprocess,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = torchvision.models.resnet50(weights="DEFAULT")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for x, y in tqdm(loader, total=len(loader), desc=f"ssl epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(x.size(0))
            total_n += int(x.size(0))

        avg_loss = total_loss / max(1, total_n)
        print(f"SSL epoch {epoch:03d} | loss={avg_loss:.4f}")

    # Save only encoder/body weights for transfer to detector backbone.
    enc_state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("fc.")}
    torch.save(
        {
            "encoder_state_dict": enc_state,
            "epochs": args.epochs,
            "seed": args.seed,
            "method": "rotation_pretext",
        },
        output_path,
    )
    print(f"SSL backbone saved: {output_path}")


if __name__ == "__main__":
    main()
